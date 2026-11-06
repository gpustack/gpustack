"""The non-streaming inference proxy must return the engine's answer as-is.

A request the engine rejects -- an unsupported sampling parameter, a malformed
body -- is a client error, and its status is what tells the caller whether
retrying can ever help. The streaming branch forwards those statuses; the
non-streaming one raised on them instead and let the catch-all report 503, so a
4xx lost both its status and the only useful part of its message.

Only the aiohttp session is faked here, so `_request_to_worker` -- the place
that decides whether a non-2xx raises -- runs for real.
"""

import asyncio
from types import SimpleNamespace
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from gpustack.api.exceptions import (
    GatewayTimeoutException,
    ServiceUnavailableException,
)
from gpustack.routes import openai as openai_route
from gpustack.server import worker_request
from gpustack.server.services import RouteTargetResolution
from tests.utils.mock import mock_async_session


class _FakeWorkerResponse:
    """Stand-in for the worker's aiohttp response."""

    def __init__(self, status_code: int, body: bytes, headers: Optional[dict] = None):
        self.status = status_code
        self.headers = headers or {"content-type": "application/json"}
        self._body = body

    async def read(self) -> bytes:
        return self._body

    async def text(self) -> str:
        return self._body.decode()

    def close(self):
        pass


def _request():
    request = MagicMock()
    request.url.path = "/v1/chat/completions"
    request.method = "POST"
    request.headers = {"content-type": "application/json"}
    return request


def _stub_pipeline(monkeypatch):
    """Stub every lookup that precedes the worker call, so only it remains."""
    instance = SimpleNamespace(
        id=11,
        model_id=1,
        worker_id=1,
        port=9000,
        worker_ip="10.0.0.1",
        distributed_servers=None,
    )
    worker = SimpleNamespace(
        id=1,
        name="worker-a",
        ip="10.0.0.1",
        port=8080,
        advertise_address=None,
        token="worker-token",
        get_proxy_address=lambda: None,
    )
    monkeypatch.setattr(openai_route, "async_session", lambda: mock_async_session())
    monkeypatch.setattr(
        openai_route,
        "parse_request_body",
        AsyncMock(
            return_value=("qwen3", False, {"model": "qwen3", "stream": False}, None)
        ),
    )
    monkeypatch.setattr(openai_route, "mutate_request", MagicMock())
    monkeypatch.setattr(
        openai_route, "get_running_instance", AsyncMock(return_value=instance)
    )
    monkeypatch.setattr(
        openai_route.ModelRouteService,
        "resolve_route_targets",
        AsyncMock(
            return_value=[
                RouteTargetResolution(model_id=1, overridden_model_name=None, weight=1)
            ]
        ),
    )
    monkeypatch.setattr(
        openai_route.ModelRouteService,
        "get_by_name",
        AsyncMock(return_value=SimpleNamespace(id=3)),
    )
    monkeypatch.setattr(
        openai_route.ModelService,
        "get_by_id",
        AsyncMock(return_value=SimpleNamespace(id=1)),
    )
    monkeypatch.setattr(
        openai_route.WorkerService, "get_by_id", AsyncMock(return_value=worker)
    )


def _patch_http_session(request, monkeypatch, *, response=None, error=None):
    """Make the worker answer with ``response``, or fail with ``error``."""
    monkeypatch.setattr(worker_request, "use_proxy_env_for_url", lambda _url: False)
    session = MagicMock()
    session.request = (
        AsyncMock(side_effect=error)
        if error is not None
        else AsyncMock(return_value=response)
    )
    request.app.state.http_client = session
    request.app.state.http_client_no_proxy = session


async def _proxy(monkeypatch, *, status_code=None, body=b"", error=None, headers=None):
    request = _request()
    _stub_pipeline(monkeypatch)
    _patch_http_session(
        request,
        monkeypatch,
        response=(
            None
            if error is not None
            else _FakeWorkerResponse(status_code, body, headers=headers)
        ),
        error=error,
    )
    return await openai_route.proxy_request_by_model(
        request=request, user=SimpleNamespace(id=1)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [400, 401, 404, 422])
async def test_rejected_request_reaches_the_client_unchanged(monkeypatch, status_code):
    """Every 4xx the engine sends is an answer about the request, not an outage."""
    body = b'{"error":{"message":"temperature must be a number"}}'

    response = await _proxy(monkeypatch, status_code=status_code, body=body)

    assert response.status_code == status_code
    assert response.body == body


@pytest.mark.asyncio
async def test_successful_answer_is_unchanged(monkeypatch):
    body = b'{"id":"chatcmpl-1","object":"chat.completion","choices":[]}'

    response = await _proxy(monkeypatch, status_code=200, body=body)

    assert response.status_code == 200
    assert response.body == body


@pytest.mark.asyncio
async def test_body_descriptor_headers_are_not_forwarded_verbatim(monkeypatch):
    """aiohttp hands over an already-decoded body, so the headers that
    described the worker-to-server transfer describe a body the caller never
    receives. Content-Length is recomputed for the bytes actually sent; the
    rest of the answer's headers pass through."""
    body = b'{"error":{"message":"bad"}}'

    response = await _proxy(
        monkeypatch,
        status_code=400,
        body=body,
        headers={
            "content-type": "application/json",
            "content-length": "9999",
            "content-encoding": "gzip",
            "transfer-encoding": "chunked",
            "connection": "keep-alive",
            "x-request-id": "req-1",
        },
    )

    assert "content-encoding" not in response.headers
    assert "transfer-encoding" not in response.headers
    assert "connection" not in response.headers
    assert response.headers["content-length"] == str(len(body))
    assert response.headers["content-type"] == "application/json"
    assert response.headers["x-request-id"] == "req-1"


@pytest.mark.asyncio
async def test_unreachable_worker_is_still_service_unavailable(monkeypatch):
    """An unreachable worker never answered, so it stays a 503."""
    with pytest.raises(ServiceUnavailableException) as exc:
        await _proxy(
            monkeypatch, error=aiohttp.ClientConnectionError("connection refused")
        )

    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_worker_timeout_is_gateway_timeout(monkeypatch):
    with pytest.raises(GatewayTimeoutException) as exc:
        await _proxy(monkeypatch, error=asyncio.TimeoutError())

    assert exc.value.status_code == 504
