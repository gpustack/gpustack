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
import logging
from types import SimpleNamespace
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gpustack.api.auth import get_current_user
from gpustack.api.exceptions import (
    GatewayTimeoutException,
    ServiceUnavailableException,
)
from gpustack.api.middlewares import (
    ModelUsageMiddleware,
    RefreshTokenMiddleware,
    RequestTimeMiddleware,
)
from gpustack.routes import openai as openai_route
from gpustack.schemas.config import GatewayModeEnum
from gpustack.server import worker_request
from gpustack.server.app import create_app
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
    request.json = AsyncMock(return_value={"model": "qwen3", "stream": False})
    request.receive = AsyncMock(side_effect=asyncio.Event().wait)
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
        openai_route.UserService,
        "model_allowed_for_user",
        AsyncMock(return_value=True),
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


@pytest.mark.parametrize("status_code", [200, 400, None])
@pytest.mark.parametrize("stream", [False, True])
def test_native_request_id_matches_worker_and_access_log(
    config, monkeypatch, caplog, status_code, stream
):
    app = create_app(
        config.model_copy(update={"gateway_mode": GatewayModeEnum.disabled})
    )
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=1)
    _stub_pipeline(monkeypatch)
    returned_headers = {"x-gpustack-request-id": "upstream-id"}
    body = b'data: [DONE]\n\n' if stream and status_code == 200 else b'{"id":"engine"}'
    outbound = {}
    error = (
        aiohttp.ClientConnectionError("connection refused")
        if status_code is None
        else None
    )
    _patch_http_session(
        SimpleNamespace(app=app),
        monkeypatch,
        response=_FakeWorkerResponse(status_code, body, headers=returned_headers),
        error=error,
    )

    if stream:

        async def stream_to_worker(**kwargs):
            outbound.update(kwargs["headers"])
            if error is not None:
                raise error
            yield body, returned_headers, status_code

        monkeypatch.setattr(openai_route, "stream_to_worker", stream_to_worker)

    with caplog.at_level(logging.INFO, logger="gpustack.inference.access"):
        response = TestClient(app).post(
            "/v1/chat/completions",
            json={"model": "qwen3", "stream": stream},
            headers={
                "X-Request-Id": "client-id",
                "X-GPUStack-Request-Id": "client-platform-id",
            },
        )
    if not stream:
        outbound.update(app.state.http_client.request.await_args.kwargs["headers"])

    expected_status = status_code if status_code is not None else 503
    assert response.status_code == expected_status
    if error is None:
        assert response.content == body
    else:
        assert response.json()["error"]["code"] == expected_status
    (request_id,) = response.headers.get_list("X-GPUStack-Request-Id")
    assert request_id not in {"client-id", "client-platform-id", "upstream-id"}
    assert outbound["x-request-id"] == request_id
    assert "x-gpustack-request-id" not in outbound
    (record,) = (
        record
        for record in caplog.records
        if record.name == "gpustack.inference.access"
    )
    assert record.request_id == request_id
    assert record.model_route_id == 3
    assert record.model_instance_id == 11
    assert record.worker_id == 1
    assert record.response_code == expected_status


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


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["headers", "body"])
async def test_disconnect_cancels_non_streaming_worker_request(monkeypatch, phase):
    _stub_pipeline(monkeypatch)
    app = FastAPI()
    app.include_router(openai_route.get_api_router(), prefix="/v1")
    app.add_middleware(RequestTimeMiddleware)
    app.add_middleware(ModelUsageMiddleware)
    app.add_middleware(RefreshTokenMiddleware)
    app.state.jwt_manager = None
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=1)
    started = asyncio.Event()
    cancelled = asyncio.Event()
    disconnect = asyncio.Event()
    closed = asyncio.Event()

    async def wait_for_worker():
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    async def request_worker(**kwargs):
        if phase == "headers":
            await wait_for_worker()
        return SimpleNamespace(status=200, read=wait_for_worker, close=closed.set)

    session = SimpleNamespace(request=request_worker)
    app.state.http_client = app.state.http_client_no_proxy = session
    sent = []
    body_pending = True

    async def receive():
        nonlocal body_pending
        if body_pending:
            body_pending = False
            return {"type": "http.request", "body": b'{"model":"qwen3"}'}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
    }
    task = asyncio.create_task(app(scope, receive, send))
    try:
        await asyncio.wait_for(started.wait(), 1)
        disconnect.set()
        await asyncio.wait_for(asyncio.shield(task), 1)
        assert cancelled.is_set()
        assert closed.is_set() == (phase == "body")
        assert (
            next(m["status"] for m in sent if m["type"] == "http.response.start") == 499
        )
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
