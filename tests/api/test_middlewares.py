"""Unit tests for usage-attribution helpers in ``api/middlewares.py``.

Focus: the direct (cookie-authed) path's ``X-Organization-Id`` validation,
which guards ``consumer_principal_id`` against a spoofed / non-existent
principal id that would otherwise violate the FK and roll back the whole
usage flush batch.
"""

import time
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


# --- Session-cookie renewal attributes -------------------------------------
#
# The renewal in ``RefreshTokenMiddleware`` reissues the session cookie
# mid-session, so an attribute it omits is one the session silently loses
# ~105 minutes in while continuing to work.
#
# Most omissions are masked by Starlette's defaults happening to match what the
# login path passes (``samesite="lax"``, ``path="/"``). ``secure`` is not: it
# defaults to ``False``. So the assertions below check the whole attribute set
# rather than the one that was wrong, because which attribute is exposed by an
# omission changes as soon as a default stops matching — as ``path`` will once
# the server scopes cookies to its mount prefix.


def _renewal_set_cookie(scheme: str) -> str:
    """Drive the middleware to the renewal branch; return its Set-Cookie."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from gpustack.api.auth import SESSION_COOKIE_NAME
    from gpustack.api.middlewares import RefreshTokenMiddleware

    app = FastAPI()
    app.add_middleware(RefreshTokenMiddleware)

    @app.get("/")
    def _root():
        return {}

    # ``exp`` inside the 15-minute window is what selects the renewal branch.
    app.state.jwt_manager = SimpleNamespace(
        decode_jwt_token=lambda token: {"sub": "admin", "exp": time.time() + 60},
        create_jwt_token=lambda username: "renewed-token",
    )

    client = TestClient(app, base_url=f"{scheme}://testserver")
    client.cookies.set(SESSION_COOKIE_NAME, "about-to-expire")
    response = client.get("/")
    header = response.headers.get("set-cookie")
    assert header, "the renewal branch did not fire, so this asserts nothing"
    return header


def _attr_names(set_cookie: str) -> set:
    """Attribute names in a Set-Cookie header, lowercased, minus the pair."""
    return {part.strip().split("=", 1)[0].lower() for part in set_cookie.split(";")[1:]}


def test_renewed_session_cookie_keeps_samesite_and_secure():
    header = _renewal_set_cookie("https")

    assert "renewed-token" in header
    assert "samesite=lax" in header.lower()
    assert "secure" in _attr_names(header)


def test_renewed_session_cookie_omits_secure_over_plain_http():
    # Marking a cookie Secure on an http origin would stop the browser sending
    # it back at all, so the flag has to track the scheme rather than be
    # unconditional.
    header = _renewal_set_cookie("http")

    assert "samesite=lax" in header.lower()
    assert "secure" not in _attr_names(header)


def test_renewal_sets_every_attribute_the_shared_helper_defines():
    """The drift guard.

    Naming individual attributes only catches the one that happened to be
    exposed — ``secure``, because its default is ``False``. Comparing against
    ``auth_cookie_attrs`` instead covers the ones Starlette's defaults currently
    mask, so they are still covered when a default stops matching.
    """
    from gpustack.api.auth import auth_cookie_attrs

    expected = {
        # kwarg name -> the attribute name it lands as in the header
        "httponly": "httponly",
        "max_age": "max-age",
        "expires": "expires",
        "samesite": "samesite",
        "secure": "secure",
        # The one the guard was written for. Renewal that omits it writes a
        # *second* cookie of the same name at Path=/; both get sent, and
        # Starlette's parser keeps the last — the stale one. Symptom is a session
        # that drops at random.
        "path": "path",
    }
    helper_kwargs = auth_cookie_attrs(
        SimpleNamespace(url=SimpleNamespace(scheme="https")), 60
    )
    assert set(helper_kwargs) == set(expected), (
        "auth_cookie_attrs grew or lost a kwarg; map it in `expected` so this "
        "test keeps covering every attribute the login path sets"
    )

    present = _attr_names(_renewal_set_cookie("https"))
    assert {expected[k] for k in helper_kwargs} <= present
