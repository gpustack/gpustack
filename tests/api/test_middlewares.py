"""Unit tests for usage-attribution helpers in ``api/middlewares.py``.

Focus: the direct (cookie-authed) path's ``X-Organization-Id`` validation,
which guards ``consumer_principal_id`` against a spoofed / non-existent
principal id that would otherwise violate the FK and roll back the whole
usage flush batch.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.sampling import ALWAYS_ON
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from gpustack.api import middlewares
from gpustack.api.auth import get_current_user
from gpustack.api.exceptions import ForbiddenException, NotFoundException
from gpustack.logging import _ApplicationLogFormatter
from gpustack.schemas.config import GatewayModeEnum
from gpustack.server.app import create_app


class _FakeSessionCtx:
    """Minimal ``async with async_session() as session`` stand-in — the real
    session is never touched because ``resolve_tenant_context`` is stubbed."""

    async def __aenter__(self):
        return SimpleNamespace()

    async def __aexit__(self, *exc):
        return False


@pytest.fixture(autouse=True)
def _stub_async_session(monkeypatch):
    monkeypatch.setattr("gpustack.server.db.async_session", lambda: _FakeSessionCtx())


def _stub_tenant_context(monkeypatch, *, result=None, exc=None):
    async def _fake(request, user, x_organization_id=None, session=None):
        if exc is not None:
            raise exc
        return result

    monkeypatch.setattr("gpustack.api.tenant.resolve_tenant_context", _fake)


_REQUEST = SimpleNamespace(state=SimpleNamespace())
_USER = SimpleNamespace(id=5, is_admin=False)


@pytest.mark.parametrize("stream", [False, True])
def test_model_usage_errors_do_not_log_response_values(caplog, stream):
    app = FastAPI()
    app.add_middleware(middlewares.ModelUsageMiddleware)
    payload = {"id": "completion", "choices": "private-response-value", "usage": {}}
    sse = f"data: {json.dumps(payload)}\n\n"

    @app.post("/v1/chat/completions")
    async def infer(request: Request):
        request.state.stream = stream
        if stream:
            return StreamingResponse(iter([sse]), media_type="text/event-stream")
        return JSONResponse(payload)

    response = TestClient(app).post("/v1/chat/completions")

    assert response.status_code == 200
    if stream:
        assert response.text == sse
    else:
        assert response.json() == payload
    (record,) = (r for r in caplog.records if r.name == middlewares.__name__)
    exported = _ApplicationLogFormatter("server").format(record)
    assert "ValidationError" in exported
    assert "private-response-value" not in exported


@pytest.mark.parametrize("stream", [False, True])
def test_native_inference_usage_links_to_the_request_id(monkeypatch, stream):
    collect = AsyncMock()
    monkeypatch.setattr(middlewares, "accumulate_gateway_metrics", collect)
    app = FastAPI()
    path = "/v1/chat/completions"
    app.add_middleware(middlewares.RequestTimeMiddleware)
    app.add_middleware(middlewares.ModelUsageMiddleware)
    app.add_middleware(middlewares.InferenceAccessMiddleware, paths={path})
    payload = {
        "id": "completion",
        "created": 0,
        "model": "test",
        "object": "chat.completion.chunk" if stream else "chat.completion",
        "choices": [],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    @app.post(path)
    async def infer(request: Request):
        request.state.stream = stream
        request.state.user = _USER
        request.state.model = SimpleNamespace(id=42, name="test")
        if stream:
            sse = f"data: {json.dumps(payload)}\n\ndata: [DONE]\n\n"
            return StreamingResponse(iter([sse]), media_type="text/event-stream")
        return JSONResponse(payload)

    response = TestClient(app).post(path, headers={"x-request-id": "client-value"})

    assert response.status_code == 200
    collect.assert_awaited_once()
    (metric,) = collect.await_args.args[0]
    assert metric.request_id == response.headers["X-GPUStack-Request-Id"]
    assert metric.request_id != "client-value"
    assert (metric.input_token, metric.output_token) == (10, 5)
    if stream:
        assert metric.ttft_ms >= 0
    else:
        assert metric.ttft_ms is None


@pytest.mark.parametrize(
    ("method", "path", "has_request_id"),
    [
        ("POST", "/v1/chat/completions", True),
        ("POST", "/v1-openai/chat/completions", True),
        ("POST", "/v1/rerank", True),
        ("GET", "/v1/models", False),
        ("GET", "/v2/models", False),
        ("POST", "/v1/not-an-inference-route", False),
    ],
)
def test_native_inference_request_id_on_entry(config, method, path, has_request_id):
    app = create_app(
        config.model_copy(update={"gateway_mode": GatewayModeEnum.disabled})
    )

    async def reject():
        raise HTTPException(status_code=401, detail="Authentication required")

    app.dependency_overrides[get_current_user] = reject
    response = TestClient(app).request(method, path, headers={"X-Request-Id": "client"})

    request_ids = response.headers.get_list("X-GPUStack-Request-Id")
    if has_request_id:
        assert response.status_code == 401
        assert len(request_ids) == 1
        assert UUID(request_ids[0]).version == 4
    else:
        assert request_ids == []


@pytest.mark.parametrize(
    "traceparent", [None, "invalid", "00-" + "a" * 32 + "-" + "b" * 16 + "-00"]
)
def test_native_rejection_has_trace_and_access_log(config, caplog, traceparent):
    app = create_app(
        config.model_copy(update={"gateway_mode": GatewayModeEnum.disabled})
    )
    exporter = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False, sampler=ALWAYS_ON)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    app.state.inference_tracer = provider.get_tracer(__name__)

    async def reject():
        raise HTTPException(status_code=401, detail="sensitive error message")

    app.dependency_overrides[get_current_user] = reject
    headers = {"Authorization": "sensitive credential", "tracestate": "vendor=secret"}
    if traceparent is not None:
        headers["traceparent"] = traceparent
    with caplog.at_level("INFO", logger="gpustack.inference.access"):
        response = TestClient(app).post(
            "/v1/chat/completions?secret=query",
            headers=headers,
            json={"secret": "body"},
        )
    assert response.status_code == 401
    (span,) = exporter.get_finished_spans()
    assert span.attributes["http.response.status_code"] == 401
    assert (
        span.attributes["gpustack.request_id"]
        == response.headers["X-GPUStack-Request-Id"]
    )
    if traceparent is not None and traceparent.startswith("00-"):
        assert span.context.trace_id == int("a" * 32, 16)
        assert span.parent.span_id == int("b" * 16, 16)
    else:
        assert span.parent is None
    assert span.context.trace_flags.sampled
    assert not span.context.trace_state
    assert not span.events
    assert "secret" not in str(dict(span.attributes))
    (record,) = [r for r in caplog.records if r.name == "gpustack.inference.access"]
    assert record.trace_id == f"{span.context.trace_id:032x}"
    assert record.span_id == f"{span.context.span_id:016x}"
    assert record.trace_flags & 1
    provider.shutdown()


def test_unhandled_inference_error_correlates_response_log_and_trace(
    config, caplog, monkeypatch
):
    app = create_app(
        config.model_copy(update={"gateway_mode": GatewayModeEnum.disabled})
    )
    exporter = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    app.state.inference_tracer = provider.get_tracer(__name__)

    async def fail(self, request, call_next):
        raise RuntimeError("private details")

    monkeypatch.setattr(middlewares.ModelUsageMiddleware, "dispatch", fail)
    with caplog.at_level("INFO", logger="gpustack.inference.access"):
        response = TestClient(app, raise_server_exceptions=False).post(
            "/v1/chat/completions"
        )

    assert response.status_code == 500
    assert response.text == "Internal Server Error"
    request_id = response.headers["X-GPUStack-Request-Id"]
    (record,) = [r for r in caplog.records if r.name == "gpustack.inference.access"]
    assert record.request_id == request_id
    assert record.response_code == 500
    (span,) = exporter.get_finished_spans()
    assert span.attributes["gpustack.request_id"] == request_id
    assert span.attributes["http.response.status_code"] == 500
    assert span.attributes["error.type"] == "RuntimeError"
    provider.shutdown()


@pytest.mark.parametrize(
    "gateway_mode", [GatewayModeEnum.disabled, GatewayModeEnum.embedded]
)
def test_native_inference_request_id_cors(config, gateway_mode):
    origin = "https://client.example.com"
    app = create_app(
        config.model_copy(
            update={
                "gateway_mode": gateway_mode,
                "enable_cors": True,
                "allow_origins": [origin],
            }
        )
    )

    async def reject():
        raise HTTPException(status_code=401, detail="Authentication required")

    app.dependency_overrides[get_current_user] = reject
    response = TestClient(app).post("/v1/chat/completions", headers={"Origin": origin})

    assert response.headers["access-control-allow-origin"] == origin
    if gateway_mode == GatewayModeEnum.disabled:
        assert (
            "x-gpustack-request-id"
            in response.headers["access-control-expose-headers"].lower()
        )
        assert "x-gpustack-request-id" in response.headers
    else:
        assert "x-gpustack-request-id" not in response.headers


@pytest.mark.asyncio
async def test_resolve_direct_consumer_org_returns_validated_id(monkeypatch):
    # A member acting in org 7 → validated current_principal_id → carried.
    _stub_tenant_context(monkeypatch, result=SimpleNamespace(current_principal_id=7))
    assert await middlewares._resolve_direct_consumer_org(_REQUEST, _USER, "7") == "7"


@pytest.mark.asyncio
async def test_resolve_direct_consumer_org_none_when_no_context(monkeypatch):
    # Admin with no effective org context → None → collector fallback applies.
    _stub_tenant_context(monkeypatch, result=SimpleNamespace(current_principal_id=None))
    assert await middlewares._resolve_direct_consumer_org(_REQUEST, _USER, "7") is None


@pytest.mark.asyncio
async def test_resolve_direct_consumer_org_nonexistent_id_dropped(monkeypatch):
    # A stale / spoofed id that doesn't resolve to a principal must NOT be
    # trusted — resolve_tenant_context raises NotFound, we swallow it and return
    # None so the FK can't be violated.
    _stub_tenant_context(
        monkeypatch, exc=NotFoundException(message="Organization 999999999 not found")
    )
    assert (
        await middlewares._resolve_direct_consumer_org(_REQUEST, _USER, "999999999")
        is None
    )


@pytest.mark.asyncio
async def test_resolve_direct_consumer_org_non_member_dropped(monkeypatch):
    # A real org the caller isn't a member of → Forbidden → dropped (no
    # mis-donation), falls back to the caller.
    _stub_tenant_context(
        monkeypatch, exc=ForbiddenException(message="Not a member of organization 7")
    )
    assert await middlewares._resolve_direct_consumer_org(_REQUEST, _USER, "7") is None
