from typing import Callable, List

import httpx
import pytest

from gpustack.api.exceptions import InvalidException
from gpustack.routes import model_provider as route_module
from gpustack.schemas.model_provider import (
    ClaudeConfig,
    CloudflareConfig,
    ModelProviderTypeEnum,
    OpenAIConfig,
    ProviderModelsInput,
    V1_MODELS_URI,
)


def _stub_upstream(monkeypatch, respond: Callable[[str], httpx.Response]) -> List[str]:
    """Answer get-models from a stub, returning the paths it asked for.

    The route builds its own client, so the client is what gets replaced. The
    base URL is kept as the route computed it, since the paths under test are
    relative to it -- and the recorded list is the assertion that matters: how
    many requests went out, in which order.
    """
    asked: List[str] = []
    real_client = httpx.AsyncClient  # bound before the patch, or factory recurses

    def factory(*, base_url, **_kwargs):
        def record(request: httpx.Request) -> httpx.Response:
            asked.append(request.url.path)
            return respond(request.url.path)

        return real_client(transport=httpx.MockTransport(record), base_url=base_url)

    monkeypatch.setattr(route_module.httpx, "AsyncClient", factory)
    return asked


def _serving(*paths: str) -> Callable[[str], httpx.Response]:
    return lambda path: (
        httpx.Response(200, json={"data": [{"id": "a-model"}]})
        if path in paths
        else httpx.Response(404, text="no such route")
    )


def _get_models(config, **kwargs):
    return route_module.get_models_from_provider(
        ProviderModelsInput(api_token="sk-test", config=config, **kwargs)
    )


class TestModelListPathCandidates:
    """Which path get-models asks an Anthropic-compatible endpoint for.

    A custom endpoint behind a base path may mount its whole API under that
    prefix or only /v1/messages, and the config cannot tell the two apart -- so
    the derived path is tried first and the bare one as a fallback. The counts
    below are the point: a fallback that fires when the first path answered
    doubles every provider's model listing, and one that never fires leaves the
    root-mounted endpoint unusable.
    """

    def _claude(self, url=None) -> ClaudeConfig:
        return ClaudeConfig(type=ModelProviderTypeEnum.CLAUDE, claudeCustomUrl=url)

    @pytest.mark.asyncio
    async def test_the_root_path_is_tried_when_the_prefixed_one_404s(self, monkeypatch):
        asked = _stub_upstream(monkeypatch, _serving(V1_MODELS_URI))

        result = await _get_models(self._claude("http://192.168.50.14:8080/anthropic"))

        assert [model.id for model in result.data] == ["a-model"]
        assert asked == ["/anthropic/v1/models", V1_MODELS_URI]

    @pytest.mark.asyncio
    async def test_the_prefixed_path_answering_costs_one_request(self, monkeypatch):
        asked = _stub_upstream(monkeypatch, _serving("/anthropic/v1/models"))

        result = await _get_models(self._claude("https://gw.example.com/anthropic/"))

        assert [model.id for model in result.data] == ["a-model"]
        assert asked == ["/anthropic/v1/models"]

    @pytest.mark.asyncio
    async def test_no_base_path_leaves_one_candidate(self, monkeypatch):
        # The derived path is already the bare one here, so appending it would
        # ask api.anthropic.com for the same URL twice.
        asked = _stub_upstream(monkeypatch, _serving(V1_MODELS_URI))

        await _get_models(self._claude())

        assert asked == [V1_MODELS_URI]

    @pytest.mark.asyncio
    async def test_other_providers_keep_their_single_path(self, monkeypatch):
        # openaiCustomUrl derives a path the same way, but an OpenAI-compatible
        # server that serves the base path serves /models under it -- there is
        # no second shape to probe for.
        asked = _stub_upstream(monkeypatch, _serving("/openai/v1/models"))

        await _get_models(
            OpenAIConfig(
                type=ModelProviderTypeEnum.OPENAI,
                openaiCustomUrl="http://192.168.50.14:8080/openai/v1",
            )
        )

        assert asked == ["/openai/v1/models"]

    @pytest.mark.asyncio
    async def test_a_provider_with_no_model_path_asks_for_nothing(self, monkeypatch):
        # ai-proxy's cloudflare provider declares no model-listing capability,
        # so the config yields no path. The empty page is returned before a
        # client is built -- a candidate list of [None] would be requested as a
        # URL and fail with a TypeError no except clause here catches.
        asked = _stub_upstream(monkeypatch, _serving())

        result = await _get_models(
            CloudflareConfig(
                type=ModelProviderTypeEnum.CLOUDFLARE, cloudflareAccountId="acct-1"
            )
        )

        assert result.data == []
        assert asked == []

    @pytest.mark.asyncio
    async def test_every_candidate_failing_reports_the_derived_path(self, monkeypatch):
        # Both paths 404, so the error is the operator's own: it names the
        # status of the path their config points at, not the fallback's.
        asked = _stub_upstream(
            monkeypatch, lambda path: httpx.Response(404, text=f"no {path}")
        )

        with pytest.raises(InvalidException) as raised:
            await _get_models(self._claude("http://192.168.50.14:8080/anthropic"))

        assert asked == ["/anthropic/v1/models", V1_MODELS_URI]
        assert "no /anthropic/v1/models" in raised.value.message

    @pytest.mark.asyncio
    async def test_a_200_that_is_not_a_model_list_counts_as_a_failure(
        self, monkeypatch
    ):
        # A wrong base path in front of a gateway answers 200 with its own UI or
        # error object rather than 404, so a body that cannot be read as a model
        # list has to fall through to the next candidate instead of 500ing.
        def respond(path: str) -> httpx.Response:
            if path == V1_MODELS_URI:
                return httpx.Response(200, json={"data": [{"id": "a-model"}]})
            return httpx.Response(200, text="<html>not an API</html>")

        asked = _stub_upstream(monkeypatch, respond)

        result = await _get_models(self._claude("http://192.168.50.14:8080/anthropic"))

        assert [model.id for model in result.data] == ["a-model"]
        assert asked == ["/anthropic/v1/models", V1_MODELS_URI]

    @pytest.mark.asyncio
    async def test_a_json_body_of_the_wrong_shape_counts_as_a_failure(
        self, monkeypatch
    ):
        # Decodes fine, carries no model list: same treatment, since .get on a
        # list would be an AttributeError no candidate loop can recover from.
        asked = _stub_upstream(
            monkeypatch, lambda _path: httpx.Response(200, json=["a-model"])
        )

        with pytest.raises(InvalidException) as raised:
            await _get_models(self._claude("http://192.168.50.14:8080/anthropic"))

        assert asked == ["/anthropic/v1/models", V1_MODELS_URI]
        assert "expected a JSON object" in raised.value.message

    @pytest.mark.asyncio
    async def test_a_transport_failure_stops_at_the_first_path(self, monkeypatch):
        # Nothing about the host resolved, so another path on it cannot do
        # better -- and reporting a connect error as "provider said 404" would
        # send the operator looking at their base path instead of the network.
        def refuse(_path: str) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        asked = _stub_upstream(monkeypatch, refuse)

        with pytest.raises(Exception) as raised:
            await _get_models(self._claude("http://192.168.50.14:8080/anthropic"))

        assert asked == ["/anthropic/v1/models"]
        assert "Network error" in raised.value.message
