"""Unit tests for usage-attribution helpers in ``api/middlewares.py``.

Focus: the direct (cookie-authed) path's ``X-Organization-Id`` validation,
which guards ``consumer_principal_id`` against a spoofed / non-existent
principal id that would otherwise violate the FK and roll back the whole
usage flush batch.
"""

from types import SimpleNamespace

import pytest

from gpustack.api import middlewares
from gpustack.api.exceptions import ForbiddenException, NotFoundException


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


@pytest.mark.asyncio
async def test_record_model_usage_supports_provider_route_without_local_model(
    monkeypatch,
):
    """External ModelProvider routes set request.state.model = None and populate
    provider metadata; record_model_usage must record provider metrics without
    dereferencing a missing Model instance."""
    from datetime import datetime, timezone
    from openai.types import CompletionUsage
    from gpustack.schemas.model_usage import OperationEnum

    captured = []

    async def _capture_metrics(metrics):
        captured.extend(metrics)

    monkeypatch.setattr(middlewares, "accumulate_gateway_metrics", _capture_metrics)

    req = SimpleNamespace(
        headers={},
        state=SimpleNamespace(
            user=SimpleNamespace(id=5),
            model=None,
            model_name="gemini-2.5-flash",
            model_route_id=42,
            provider_id=11,
            provider_name="gemini-gateway",
            provider_type="gemini",
            api_key=SimpleNamespace(access_key="ak_test"),
            start_time=datetime(2026, 11, 26, 0, 0, 0, tzinfo=timezone.utc),
        ),
    )
    usage = CompletionUsage(prompt_tokens=12, completion_tokens=8, total_tokens=20)

    await middlewares.record_model_usage(req, usage, OperationEnum.CHAT_COMPLETION)

    assert len(captured) == 1
    metric = captured[0]
    assert metric.model == "gemini-2.5-flash"
    assert metric.model_id is None
    assert metric.cluster_id is None
    assert metric.model_route_id == 42
    assert metric.provider_id == 11
    assert metric.provider_name == "gemini-gateway"
    assert metric.provider_type == "gemini"
    assert metric.input_token == 12
    assert metric.output_token == 8
    assert metric.total_token == 20


def test_extract_response_usage_falls_back_when_envelope_omits_id():
    """Third-party OpenAI-compatible providers that omit strict ChatCompletion
    fields (e.g. id) still have their usage object extracted."""
    from openai.types.chat import ChatCompletion

    usage = middlewares._extract_response_usage(
        ChatCompletion,
        {
            "choices": [],
            "usage": {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10},
        },
    )
    assert usage is not None
    assert usage.prompt_tokens == 4
    assert usage.completion_tokens == 6
    assert usage.total_tokens == 10


def test_extract_response_usage_returns_none_on_invalid_usage_dict():
    """Malformed or partial usage dicts never raise out of _extract_response_usage."""
    from openai.types.chat import ChatCompletion

    usage = middlewares._extract_response_usage(
        ChatCompletion,
        {"choices": [], "usage": {"prompt_tokens": 4}},
    )
    assert usage is None
