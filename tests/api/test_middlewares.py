"""Unit tests for usage-attribution helpers in ``api/middlewares.py``.

Focus: the direct (cookie-authed) path's ``X-Organization-Id`` validation,
which guards ``consumer_principal_id`` against a spoofed / non-existent
principal id that would otherwise violate the FK and roll back the whole
usage flush batch.
"""

import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient

from gpustack.api import middlewares
from gpustack.api.exceptions import ForbiddenException, NotFoundException
from gpustack.logging import _ApplicationLogFormatter


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
