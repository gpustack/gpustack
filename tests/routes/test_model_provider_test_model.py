"""The provider-model ping: when a retry is worth a second request.

A thinking-only Qwen model rejects the parameterless ping with a 400 naming
``enable_thinking``; the test retries once with it set to True. Every other
failure means the same thing on a retry, so it is reported as-is after one
request.
"""

import json
from typing import Callable, Dict, List

import httpx
import pytest

from gpustack.routes import model_provider as route_module
from gpustack.schemas.model_provider import (
    ModelProviderTypeEnum,
    OpenAIConfig,
    QwenConfig,
    TestProviderModelInput,
)


def _stub_upstream(monkeypatch, respond: Callable[[httpx.Request], httpx.Response]):
    """Answer test-model from a stub, returning the JSON bodies it was sent.

    The route builds its own client, so the client is what gets replaced; the
    recorded bodies are the assertion that matters -- how many requests went
    out and which parameters each carried.
    """
    asked: List[Dict] = []
    real_client = httpx.AsyncClient

    def factory(*, base_url, **_kwargs):
        def record(request: httpx.Request) -> httpx.Response:
            asked.append(
                {"path": request.url.path, "json": json.loads(request.content)}
            )
            return respond(request)

        return real_client(transport=httpx.MockTransport(record), base_url=base_url)

    monkeypatch.setattr(route_module.httpx, "AsyncClient", factory)
    return asked


def _test_model(config, model_name="qwen3.7-max"):
    return route_module.try_model_with_provider(
        TestProviderModelInput(
            api_token="sk-test", config=config, model_name=model_name
        )
    )


def _qwen() -> QwenConfig:
    return QwenConfig(type=ModelProviderTypeEnum.QWEN, qwenEnableCompatible=True)


def _thinking_restricted() -> httpx.Response:
    return httpx.Response(
        400,
        json={
            "error": {
                "message": (
                    "<400> InternalError.Algo.InvalidParameter: The value of the "
                    "enable_thinking parameter is restricted to True."
                ),
                "type": "invalid_request_error",
            }
        },
    )


def _ok() -> httpx.Response:
    return httpx.Response(200, json={"choices": []})


class TestThinkingRestrictedRetry:
    @pytest.mark.asyncio
    async def test_a_thinking_restricted_rejection_is_retried_with_it_true(
        self, monkeypatch
    ):
        def respond(request: httpx.Request) -> httpx.Response:
            if "enable_thinking" in json.loads(request.content):
                return _ok()
            return _thinking_restricted()

        asked = _stub_upstream(monkeypatch, respond)

        result = await _test_model(_qwen())

        assert result.accessible is True
        assert len(asked) == 2
        assert "enable_thinking" not in asked[0]["json"]
        assert asked[1]["json"]["enable_thinking"] is True

    @pytest.mark.asyncio
    async def test_the_retry_failing_reports_the_first_error(self, monkeypatch):
        asked = _stub_upstream(monkeypatch, lambda _request: _thinking_restricted())

        result = await _test_model(_qwen())

        assert result.accessible is False
        assert len(asked) == 2
        assert "enable_thinking parameter is restricted to True" in (
            result.error_message or ""
        )

    @pytest.mark.asyncio
    async def test_a_first_attempt_success_costs_one_request(self, monkeypatch):
        asked = _stub_upstream(monkeypatch, lambda _request: _ok())

        result = await _test_model(_qwen())

        assert result.accessible is True
        assert len(asked) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status,text",
        [
            (401, '{"error":{"message":"invalid api key"}}'),
            (400, '{"error":{"message":"insufficient balance"}}'),
        ],
    )
    async def test_other_failures_are_not_retried(self, monkeypatch, status, text):
        asked = _stub_upstream(
            monkeypatch, lambda _request: httpx.Response(status, text=text)
        )

        result = await _test_model(_qwen())

        assert result.accessible is False
        assert len(asked) == 1

    @pytest.mark.asyncio
    async def test_non_qwen_providers_never_send_enable_thinking(self, monkeypatch):
        def respond(request: httpx.Request) -> httpx.Response:
            if "enable_thinking" in json.loads(request.content):
                return httpx.Response(400, text="unknown parameter")
            return _ok()

        asked = _stub_upstream(monkeypatch, respond)

        result = await _test_model(OpenAIConfig(type=ModelProviderTypeEnum.OPENAI))

        assert result.accessible is True
        assert asked[0]["json"].get("enable_thinking") is None
