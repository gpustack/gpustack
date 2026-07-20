import asyncio
import ssl
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient
from gpustack.api.exceptions import register_handlers
from gpustack.api.auth import (
    GATEWAY_AUTH_TOKEN_HEADER,
    client_ip_getter,
    get_current_user,
    worker_auth,
)
from gpustack.api.exceptions import BadRequestException, UnauthorizedException
from gpustack.schemas.config import GatewayModeEnum
from gpustack.schemas.users import AuthProviderEnum
from gpustack.routes import auth as auth_route
from gpustack.routes.auth import oidc_callback


class DummyWorkerConfig:
    token = "registration-token"

    def get_server_url(self):
        return "http://example.com"


@pytest.mark.asyncio
async def test_get_current_user_accepts_x_api_key(monkeypatch):
    session = object()
    request = type("Request", (), {})()
    request.state = type("State", (), {})()
    request.headers = {}
    request.client = type("Client", (), {"host": "10.0.0.1"})()
    request.app = type("App", (), {})()
    request.app.state = type("State", (), {})()
    request.app.state.server_config = type("Config", (), {"gateway_mode": None})()

    expected_user = type("User", (), {"is_active": True})()
    expected_key = object()

    auth_mock = AsyncMock(return_value=(expected_user, expected_key))
    monkeypatch.setattr("gpustack.api.auth.get_user_from_api_token", auth_mock)

    @asynccontextmanager
    async def fake_async_session():
        yield session

    monkeypatch.setattr("gpustack.api.auth.async_session", fake_async_session)

    user = await get_current_user(
        request=request,
        x_api_key="sk_test_value",
    )

    auth_mock.assert_awaited_once_with(session, "sk_test_value")
    assert user is expected_user
    assert request.state.user is expected_user
    assert request.state.api_key is expected_key


@pytest.mark.asyncio
async def test_authenticate_request_reuses_provided_session(monkeypatch):
    """``authenticate_request(..., session=<provided>)`` must reuse the
    caller's session (not open a fresh one) and still populate
    ``request.state.user`` / ``request.state.api_key``. Covers the
    session-reuse path used by ``/token-auth``."""
    from gpustack.api.auth import authenticate_request

    provided_session = object()
    request = _make_request()

    expected_user = type("User", (), {"is_active": True})()
    expected_key = object()

    captured = {}

    async def fake_get_user_from_api_token(session, token):
        captured["session"] = session
        captured["token"] = token
        return expected_user, expected_key

    monkeypatch.setattr(
        "gpustack.api.auth.get_user_from_api_token", fake_get_user_from_api_token
    )

    # If a fresh session were opened, this fires — proving the provided one
    # is reused instead. (A plain object() as the session also means any
    # attempt to close it would AttributeError, so this doubles as a
    # "caller's session is not closed" assertion.)
    def _no_fresh_session(*args, **kwargs):
        raise AssertionError(
            "async_session() must not be called when a session is provided"
        )

    monkeypatch.setattr("gpustack.api.auth.async_session", _no_fresh_session)

    user = await authenticate_request(
        request, x_api_key="sk_test_value", session=provided_session
    )

    assert user is expected_user
    assert captured["session"] is provided_session
    assert captured["token"] == "sk_test_value"
    assert request.state.user is expected_user
    assert request.state.api_key is expected_key


@pytest.mark.asyncio
async def test_captcha_disables_ordinary_user_basic_auth(monkeypatch):
    from fastapi.security import HTTPBasicCredentials

    from gpustack.api.auth import authenticate_request

    request = _make_request()
    request.app.state.server_config.enable_login_captcha = True
    basic_auth_mock = AsyncMock()
    monkeypatch.setattr("gpustack.api.auth.authenticate_basic_user", basic_auth_mock)

    def no_database_session():
        raise AssertionError("Rejected Basic auth must not open a database session")

    monkeypatch.setattr("gpustack.api.auth.async_session", no_database_session)

    with pytest.raises(UnauthorizedException) as exc_info:
        await authenticate_request(
            request,
            basic_credentials=HTTPBasicCredentials(
                username="admin", password="guessed-password"
            ),
        )

    assert "HTTP Basic is disabled" in exc_info.value.message
    basic_auth_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_captcha_keeps_system_principal_basic_auth():
    from fastapi.security import HTTPBasicCredentials

    from gpustack.api.auth import authenticate_request
    from gpustack.schemas.principals import PrincipalType

    request = _make_request()
    request.app.state.server_config.enable_login_captcha = True
    request.app.state.server_config.token = "server-token"

    principal = await authenticate_request(
        request,
        basic_credentials=HTTPBasicCredentials(
            username="system/worker/abc", password="server-token"
        ),
    )

    assert principal.kind == PrincipalType.SYSTEM
    assert principal.name == "system/worker/abc"


def _api_token_double(**overrides):
    fields = {
        "id": 11,
        "hashed_secret_key": "stored-hash",
        "secret_key_digest": None,
        "is_custom": False,
        "expires_at": None,
        "user_id": 7,
        "access_key": "abcd1234",
    }
    fields.update(overrides)
    return type("ApiKey", (), fields)()


class _RollbackRecordingSession:
    def __init__(self, events):
        self._events = events

    async def rollback(self):
        self._events.append("rollback")


@pytest.mark.asyncio
async def test_get_user_from_api_token_skips_verify_for_expired_key(monkeypatch):
    """An expired key must be rejected *before* the argon2 verify runs."""
    from datetime import datetime, timedelta, timezone

    from gpustack.api.auth import get_user_from_api_token

    expired = _api_token_double(
        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1)
    )

    async def fake_get_by_access_key(self, candidate):
        return expired

    verify_calls = {"n": 0}

    def fake_verify(hashed, secret):
        verify_calls["n"] += 1
        return True

    monkeypatch.setattr(
        "gpustack.api.auth.APIKeyService.get_by_access_key", fake_get_by_access_key
    )
    monkeypatch.setattr("gpustack.api.auth.verify_hashed_secret", fake_verify)

    session = _RollbackRecordingSession([])
    user, api_key = await get_user_from_api_token(
        session, "gpustack_abcd1234_c11c75ed6334ea9505da4ad9"
    )

    assert user is None and api_key is None
    assert verify_calls["n"] == 0  # argon2 skipped for expired key


@pytest.mark.asyncio
async def test_get_user_from_api_token_rollback_before_verify_via_thread(monkeypatch):
    """Valid key: the pooled connection is released (``rollback``) *before*
    the argon2 verify, and the verify runs off the event loop via
    ``asyncio.to_thread``."""
    import gpustack.api.auth as auth_module
    from gpustack.api.auth import get_user_from_api_token

    events = []
    valid = _api_token_double()
    expected_user = type("User", (), {"is_active": True, "id": 7})()

    async def fake_get_by_access_key(self, candidate):
        return valid

    async def fake_get_by_id(self, user_id):
        assert user_id == 7
        return expected_user

    def fake_verify(hashed, secret):
        events.append("verify")
        assert hashed == "stored-hash"
        return True

    monkeypatch.setattr(
        "gpustack.api.auth.APIKeyService.get_by_access_key", fake_get_by_access_key
    )
    monkeypatch.setattr("gpustack.api.auth.UserService.get_by_id", fake_get_by_id)
    monkeypatch.setattr("gpustack.api.auth.verify_hashed_secret", fake_verify)

    real_to_thread = auth_module.asyncio.to_thread
    used_thread = {"n": 0}

    async def spy_to_thread(fn, *args, **kwargs):
        used_thread["n"] += 1
        return await real_to_thread(fn, *args, **kwargs)

    monkeypatch.setattr(auth_module.asyncio, "to_thread", spy_to_thread)

    session = _RollbackRecordingSession(events)
    user, api_key = await get_user_from_api_token(
        session, "gpustack_abcd1234_c11c75ed6334ea9505da4ad9"
    )

    assert user is expected_user
    assert api_key is valid
    assert events == ["rollback", "verify"]  # connection released before argon2
    assert used_thread["n"] == 1  # argon2 ran via asyncio.to_thread


@pytest.mark.asyncio
async def test_get_user_from_api_token_prefers_the_digest(monkeypatch):
    """A key that has a digest must not touch argon2 at all — that is the whole
    point of the column — and must not need a worker thread either.
    """
    import gpustack.api.auth as auth_module
    from gpustack.api.auth import get_user_from_api_token
    from gpustack.security import generate_access_key, new_secret_key_digest

    access_key = generate_access_key()
    secret_key = "c11c75ed6334ea9505da4ad9c11c75ed"
    key = _api_token_double(
        access_key=access_key,
        secret_key_digest=new_secret_key_digest(
            secret_key=secret_key, is_custom=False, access_key=access_key
        ),
    )
    expected_user = type("User", (), {"is_active": True, "id": 7})()

    async def fake_get_by_access_key(self, candidate):
        return key

    async def fake_get_by_id(self, user_id):
        return expected_user

    def fail_verify(hashed, secret):  # pragma: no cover - must never run
        raise AssertionError("argon2 must not run for a key that has a digest")

    monkeypatch.setattr(
        "gpustack.api.auth.APIKeyService.get_by_access_key", fake_get_by_access_key
    )
    monkeypatch.setattr("gpustack.api.auth.UserService.get_by_id", fake_get_by_id)
    monkeypatch.setattr("gpustack.api.auth.verify_hashed_secret", fail_verify)

    thread_calls = {"n": 0}

    async def spy_to_thread(fn, *args, **kwargs):  # pragma: no cover
        thread_calls["n"] += 1
        raise AssertionError("no thread offload needed for a fast hash")

    monkeypatch.setattr(auth_module.asyncio, "to_thread", spy_to_thread)

    user, api_key = await get_user_from_api_token(
        _RollbackRecordingSession([]), f"gpustack_{access_key}_{secret_key}"
    )

    assert user is expected_user
    assert api_key is key
    assert thread_calls["n"] == 0


@pytest.mark.asyncio
async def test_digest_mismatch_is_rejected_without_falling_back_to_argon2(monkeypatch):
    """Both columns derive from the same plaintext, so a digest mismatch means the
    secret is simply wrong — falling back to argon2 would only pay 30 ms to reach
    the same answer.
    """
    from gpustack.api.auth import get_user_from_api_token
    from gpustack.security import generate_access_key, new_secret_key_digest

    access_key = generate_access_key()
    key = _api_token_double(
        access_key=access_key,
        secret_key_digest=new_secret_key_digest(
            secret_key="c11c75ed6334ea9505da4ad9c11c75ed",
            is_custom=False,
            access_key=access_key,
        ),
    )

    async def fake_get_by_access_key(self, candidate):
        return key

    monkeypatch.setattr(
        "gpustack.api.auth.APIKeyService.get_by_access_key", fake_get_by_access_key
    )
    monkeypatch.setattr(
        "gpustack.api.auth.verify_hashed_secret",
        lambda hashed, secret: (_ for _ in ()).throw(
            AssertionError("no argon2 fallback on digest mismatch")
        ),
    )

    user, api_key = await get_user_from_api_token(
        _RollbackRecordingSession([]),
        f"gpustack_{access_key}_ffffffffffffffffffffffffffffffff",
    )

    assert user is None and api_key is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stored_digest",
    [
        "garbage",
        "sha256$truncated",
        "sha256$deadbeef$",
        "blake3$deadbeef$abc",  # e.g. written by a newer server
    ],
)
async def test_unusable_digest_falls_back_to_argon2(monkeypatch, stored_digest):
    """A digest this build cannot check says nothing about the secret. argon2 is
    the permanent fallback verifier, so bad column data must cost the slow path
    rather than lock a valid key out.
    """
    from gpustack.api.auth import get_user_from_api_token
    from gpustack.security import generate_access_key

    access_key = generate_access_key()
    key = _api_token_double(access_key=access_key, secret_key_digest=stored_digest)
    expected_user = type("User", (), {"is_active": True, "id": 7})()

    async def fake_get_by_access_key(self, candidate):
        return key

    async def fake_get_by_id(self, user_id):
        return expected_user

    argon2_calls = {"n": 0}

    def fake_verify(hashed, secret):
        argon2_calls["n"] += 1
        return True

    monkeypatch.setattr(
        "gpustack.api.auth.APIKeyService.get_by_access_key", fake_get_by_access_key
    )
    monkeypatch.setattr("gpustack.api.auth.UserService.get_by_id", fake_get_by_id)
    monkeypatch.setattr("gpustack.api.auth.verify_hashed_secret", fake_verify)
    # An unusable digest is non-NULL, so the ``WHERE secret_key_digest IS NULL``
    # backfill would not repair it -- assert we do not silently rewrite it either.
    scheduled = {"n": 0}
    monkeypatch.setattr(
        "gpustack.api.auth._schedule_secret_key_digest_backfill",
        lambda api_key, secret: scheduled.__setitem__("n", scheduled["n"] + 1),
    )

    user, api_key = await get_user_from_api_token(
        _RollbackRecordingSession([]),
        f"gpustack_{access_key}_c11c75ed6334ea9505da4ad9c11c75ed",
    )

    assert user is expected_user
    assert api_key is key
    assert argon2_calls["n"] == 1


@pytest.mark.asyncio
async def test_successful_argon2_verify_backfills_the_digest(monkeypatch):
    """A key created before the column converges on first use, and the write stays
    off the request path.
    """
    import gpustack.api.auth as auth_module
    from gpustack.api.auth import get_user_from_api_token
    from gpustack.security import generate_access_key

    access_key = generate_access_key()
    secret_key = "c11c75ed6334ea9505da4ad9c11c75ed"
    key = _api_token_double(access_key=access_key, secret_key_digest=None)

    async def fake_get_by_access_key(self, candidate):
        return key

    async def fake_get_by_id(self, user_id):
        return type("User", (), {"is_active": True, "id": 7})()

    monkeypatch.setattr(
        "gpustack.api.auth.APIKeyService.get_by_access_key", fake_get_by_access_key
    )
    monkeypatch.setattr("gpustack.api.auth.UserService.get_by_id", fake_get_by_id)
    monkeypatch.setattr(
        "gpustack.api.auth.verify_hashed_secret", lambda hashed, secret: True
    )

    written = []

    async def fake_backfill(api_key_id, key_access_key, digest, expected_current=None):
        written.append((api_key_id, key_access_key, digest, expected_current))
        auth_module._backfill_in_flight.discard(api_key_id)

    monkeypatch.setattr(auth_module, "_backfill_secret_key_digest", fake_backfill)
    auth_module._backfill_in_flight.clear()

    user, _ = await get_user_from_api_token(
        _RollbackRecordingSession([]), f"gpustack_{access_key}_{secret_key}"
    )
    await asyncio.sleep(0)  # let the background task run

    assert user is not None
    assert len(written) == 1
    written_id, written_access_key, written_digest, expected_current = written[0]
    assert (written_id, written_access_key) == (11, access_key)
    # NULL guard: this row predates the column, so the write must not clobber a
    # value another process may have set in the meantime.
    assert expected_current is None
    from gpustack.security import verify_secret_key_digest

    assert verify_secret_key_digest(written_digest, secret_key)


@pytest.mark.asyncio
async def test_custom_key_is_never_backfilled(monkeypatch):
    """Custom keys stay on argon2 permanently — a fast hash over a user-chosen
    secret is the one outcome this design must not produce.
    """
    import gpustack.api.auth as auth_module
    from gpustack.api.auth import get_user_from_api_token

    key = _api_token_double(
        is_custom=True,
        secret_key_digest=None,
        access_key="0123456789abcdef0123456789abcdef",
    )

    async def fake_get_by_access_key(self, candidate):
        return key

    async def fake_get_by_id(self, user_id):
        return type("User", (), {"is_active": True, "id": 7})()

    monkeypatch.setattr(
        "gpustack.api.auth.APIKeyService.get_by_access_key", fake_get_by_access_key
    )
    monkeypatch.setattr("gpustack.api.auth.UserService.get_by_id", fake_get_by_id)
    monkeypatch.setattr(
        "gpustack.api.auth.verify_hashed_secret", lambda hashed, secret: True
    )

    scheduled = []
    monkeypatch.setattr(
        auth_module,
        "_backfill_secret_key_digest",
        lambda *args: scheduled.append(args),
    )
    auth_module._backfill_in_flight.clear()

    user, _ = await get_user_from_api_token(
        _RollbackRecordingSession([]), "a-user-chosen-custom-key"
    )
    await asyncio.sleep(0)

    assert user is not None
    assert scheduled == []


def test_concurrent_verifications_schedule_one_backfill(monkeypatch):
    """The first wave after an upgrade hits the same keys repeatedly; only one
    write should be in flight per key.
    """
    import gpustack.api.auth as auth_module
    from gpustack.security import generate_access_key

    access_key = generate_access_key()
    key = _api_token_double(access_key=access_key, secret_key_digest=None)

    created = []

    class _FakeTask:
        def add_done_callback(self, _cb):
            pass

    def fake_create_task(coro):
        coro.close()  # nothing awaits it here; this test only counts schedulings
        created.append(coro)
        return _FakeTask()

    monkeypatch.setattr(auth_module.asyncio, "create_task", fake_create_task)
    auth_module._backfill_in_flight.clear()

    auth_module._schedule_secret_key_digest_backfill(
        key, "c11c75ed6334ea9505da4ad9c11c75ed"
    )
    auth_module._schedule_secret_key_digest_backfill(
        key, "c11c75ed6334ea9505da4ad9c11c75ed"
    )

    assert len(created) == 1
    auth_module._backfill_in_flight.clear()
    auth_module._backfill_tasks.clear()


def _mock_backfill_session(execute=None):
    """An ``async_session()`` stand-in whose ``execute`` is awaitable."""
    session = MagicMock()
    session.execute = execute or AsyncMock()
    session.commit = AsyncMock()
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=session)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm, session


def _compiled(stmt) -> str:
    return str(stmt.compile(compile_kwargs={"literal_binds": True}))


@pytest.mark.asyncio
async def test_backfill_writes_only_the_digest_column(monkeypatch):
    """The argon2 hash must survive untouched — it is what an older version falls
    back to after a downgrade, and the two columns must keep describing the same
    plaintext. ``updated_at`` must not move either: it carries ``onupdate``, which
    fires on bulk UPDATEs, and the API sorts on it.
    """
    import gpustack.api.auth as auth_module

    cm, session = _mock_backfill_session()
    monkeypatch.setattr(auth_module, "async_session", lambda: cm)
    monkeypatch.setattr(auth_module, "delete_cache_by_key", AsyncMock())
    auth_module._backfill_in_flight.add(11)

    await auth_module._backfill_secret_key_digest(11, "abcd1234", "sha256$aa$bb")

    sql = _compiled(session.execute.await_args.args[0])
    assert "secret_key_digest='sha256$aa$bb'" in sql
    assert "hashed_secret_key" not in sql
    assert "api_keys.id = 11" in sql
    # Self-assignment, which is what keeps ``onupdate`` from restamping the row.
    assert "updated_at=api_keys.updated_at" in sql
    session.commit.assert_awaited_once()
    assert 11 not in auth_module._backfill_in_flight


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "expected_current, guard",
    [
        (None, "api_keys.secret_key_digest IS NULL"),
        ("garbage", "api_keys.secret_key_digest = 'garbage'"),
    ],
)
async def test_backfill_is_a_compare_and_set(monkeypatch, expected_current, guard):
    """Concurrent authentications of one key collapse into a single effective
    write, and a value another process wrote meanwhile is never clobbered.
    """
    import gpustack.api.auth as auth_module

    cm, session = _mock_backfill_session()
    monkeypatch.setattr(auth_module, "async_session", lambda: cm)
    monkeypatch.setattr(auth_module, "delete_cache_by_key", AsyncMock())

    await auth_module._backfill_secret_key_digest(
        11, "abcd1234", "sha256$aa$bb", expected_current=expected_current
    )

    assert guard in _compiled(session.execute.await_args.args[0])


@pytest.mark.asyncio
async def test_backfill_invalidates_the_key_the_read_path_reads(monkeypatch):
    """The one assertion that keeps the backfill from becoming a silent no-op.

    ``locked_cached`` stores under ``build_cache_key(unbound_f, *args[1:])`` while
    the invalidation passes a *bound* method. The two only agree because
    ``build_cache_key`` skips its self-stripping branch for bound methods — if
    that ever changes, every request keeps paying argon2 and rescheduling this
    backfill, with correct behaviour and no warning anywhere.
    """
    import gpustack.api.auth as auth_module
    from gpustack.server.cache import build_cache_key
    from gpustack.server.services import APIKeyService

    cm, _ = _mock_backfill_session()
    monkeypatch.setattr(auth_module, "async_session", lambda: cm)

    invalidated = []

    async def fake_delete(func, *args, **kwargs):
        invalidated.append(build_cache_key(func, *args, **kwargs))

    monkeypatch.setattr(auth_module, "delete_cache_by_key", fake_delete)

    await auth_module._backfill_secret_key_digest(11, "abcd1234", "sha256$aa$bb")

    # What ``locked_cached.decorator`` would have stored the row under.
    read_path_key = build_cache_key(APIKeyService.get_by_access_key, "abcd1234")
    assert invalidated == [read_path_key]


@pytest.mark.asyncio
async def test_backfill_failure_is_swallowed_and_releases_the_marker(monkeypatch):
    """A failed optimization must not surface anywhere, and must not cost the key
    its next attempt.
    """
    import gpustack.api.auth as auth_module

    cm, _ = _mock_backfill_session(execute=AsyncMock(side_effect=RuntimeError("no db")))
    monkeypatch.setattr(auth_module, "async_session", lambda: cm)
    monkeypatch.setattr(auth_module, "delete_cache_by_key", AsyncMock())
    auth_module._backfill_in_flight.add(11)

    await auth_module._backfill_secret_key_digest(11, "abcd1234", "sha256$aa$bb")

    assert 11 not in auth_module._backfill_in_flight


@pytest.mark.asyncio
async def test_scheduling_failure_never_reaches_the_caller(monkeypatch):
    """``get_user_from_api_token`` wraps its body in a try that raises 500, so a
    throw from scheduling would turn an already-authenticated request into an
    error over a missed optimization.
    """
    import gpustack.api.auth as auth_module
    from gpustack.security import generate_access_key

    key = _api_token_double(access_key=generate_access_key(), secret_key_digest=None)

    def boom(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(auth_module, "new_secret_key_digest", boom)
    auth_module._backfill_in_flight.clear()

    auth_module._schedule_secret_key_digest_backfill(
        key, "c11c75ed6334ea9505da4ad9c11c75ed"
    )

    assert auth_module._backfill_in_flight == set()


@pytest.mark.asyncio
async def test_failed_create_task_does_not_strand_the_marker(monkeypatch):
    """A stranded id would cost that key its backfill for the process's lifetime,
    since nothing else ever clears the set.
    """
    import gpustack.api.auth as auth_module
    from gpustack.security import generate_access_key

    key = _api_token_double(access_key=generate_access_key(), secret_key_digest=None)

    def failing_create_task(coro):
        coro.close()
        raise RuntimeError("no loop")

    monkeypatch.setattr(auth_module.asyncio, "create_task", failing_create_task)
    auth_module._backfill_in_flight.clear()

    auth_module._schedule_secret_key_digest_backfill(
        key, "c11c75ed6334ea9505da4ad9c11c75ed"
    )

    assert auth_module._backfill_in_flight == set()


@pytest.mark.asyncio
async def test_unusable_digest_warns_once_per_key(monkeypatch, caplog):
    """A polling worker key would otherwise repeat the same warning every request
    until the value is repaired.
    """
    import logging

    import gpustack.api.auth as auth_module
    from gpustack.api.auth import get_user_from_api_token
    from gpustack.security import generate_access_key

    access_key = generate_access_key()
    key = _api_token_double(access_key=access_key, secret_key_digest="garbage")

    async def fake_get_by_access_key(self, candidate):
        return key

    async def fake_get_by_id(self, user_id):
        return type("User", (), {"is_active": True, "id": 7})()

    monkeypatch.setattr(
        "gpustack.api.auth.APIKeyService.get_by_access_key", fake_get_by_access_key
    )
    monkeypatch.setattr("gpustack.api.auth.UserService.get_by_id", fake_get_by_id)
    monkeypatch.setattr(
        "gpustack.api.auth.verify_hashed_secret", lambda hashed, secret: True
    )
    monkeypatch.setattr(
        auth_module,
        "_schedule_secret_key_digest_backfill",
        lambda api_key, secret: None,
    )
    auth_module._warned_unusable_digest.clear()

    with caplog.at_level(logging.WARNING, logger="gpustack.api.auth"):
        for _ in range(3):
            await get_user_from_api_token(
                _RollbackRecordingSession([]),
                f"gpustack_{access_key}_c11c75ed6334ea9505da4ad9c11c75ed",
            )

    warnings = [r for r in caplog.records if "unusable secret_key_digest" in r.message]
    assert len(warnings) == 1
    auth_module._warned_unusable_digest.clear()


@pytest.mark.asyncio
async def test_get_user_from_api_token_rejects_wrong_secret(monkeypatch):
    """A key whose secret fails argon2 verification yields no user."""
    from gpustack.api.auth import get_user_from_api_token

    valid = _api_token_double()

    async def fake_get_by_access_key(self, candidate):
        return valid

    get_by_id_calls = {"n": 0}

    async def fake_get_by_id(self, user_id):
        get_by_id_calls["n"] += 1
        return type("User", (), {"is_active": True, "id": 7})()

    monkeypatch.setattr(
        "gpustack.api.auth.APIKeyService.get_by_access_key", fake_get_by_access_key
    )
    monkeypatch.setattr("gpustack.api.auth.UserService.get_by_id", fake_get_by_id)
    monkeypatch.setattr(
        "gpustack.api.auth.verify_hashed_secret", lambda hashed, secret: False
    )

    session = _RollbackRecordingSession([])
    user, api_key = await get_user_from_api_token(
        session, "gpustack_abcd1234_c11c75ed6334ea9505da4ad9"
    )

    assert user is None and api_key is None
    assert get_by_id_calls["n"] == 0  # never resolve a user for a bad secret


@pytest.mark.asyncio
async def test_worker_auth_accepts_x_api_key():
    request = type("Request", (), {})()
    request.headers = {"X-Higress-Llm-Model": "claude-sonnet"}
    request.app = type("App", (), {})()
    request.app.state = type("State", (), {})()
    request.app.state.token = "worker-token"
    request.app.state.config = DummyWorkerConfig()
    request.app.state.http_client_no_proxy = object()

    assert await worker_auth(request=request, x_api_key="worker-token") is None


@pytest.mark.asyncio
async def test_worker_auth_rejects_missing_credentials():
    request = type("Request", (), {})()
    request.headers = {"X-Higress-Llm-Model": "claude-sonnet"}
    request.app = type("App", (), {})()
    request.app.state = type("State", (), {})()
    request.app.state.token = "worker-token"
    request.app.state.config = DummyWorkerConfig()
    request.app.state.http_client_no_proxy = object()

    with pytest.raises(UnauthorizedException):
        await worker_auth(request=request)


@pytest.mark.asyncio
async def test_get_current_user_falls_back_to_x_api_key_when_bearer_empty(
    monkeypatch,
):
    session = object()
    request = type("Request", (), {})()
    request.state = type("State", (), {})()
    request.headers = {}
    request.client = type("Client", (), {"host": "10.0.0.1"})()
    request.app = type("App", (), {})()
    request.app.state = type("State", (), {})()
    request.app.state.server_config = type("Config", (), {"gateway_mode": None})()

    expected_user = type("User", (), {"is_active": True})()
    expected_key = object()

    auth_mock = AsyncMock(return_value=(expected_user, expected_key))
    monkeypatch.setattr("gpustack.api.auth.get_user_from_api_token", auth_mock)

    @asynccontextmanager
    async def fake_async_session():
        yield session

    monkeypatch.setattr("gpustack.api.auth.async_session", fake_async_session)

    user = await get_current_user(
        request=request,
        bearer_token=HTTPAuthorizationCredentials(scheme="Bearer", credentials=""),
        x_api_key="sk_test_value",
    )

    auth_mock.assert_awaited_once_with(session, "sk_test_value")
    assert user is expected_user


@pytest.mark.asyncio
async def test_worker_auth_falls_back_to_x_api_key_when_bearer_empty():
    request = type("Request", (), {})()
    request.headers = {"X-Higress-Llm-Model": "claude-sonnet"}
    request.app = type("App", (), {})()
    request.app.state = type("State", (), {})()
    request.app.state.token = "worker-token"
    request.app.state.config = DummyWorkerConfig()
    request.app.state.http_client_no_proxy = object()

    assert (
        await worker_auth(
            request=request,
            bearer_token=HTTPAuthorizationCredentials(scheme="Bearer", credentials=""),
            x_api_key="worker-token",
        )
        is None
    )


def _make_request(headers=None, client_host="127.0.0.1"):
    request = type("Request", (), {})()
    request.state = type("State", (), {})()
    request.headers = headers or {}
    request.client = type("Client", (), {"host": client_host})()
    request.app = type("App", (), {})()
    request.app.state = type("State", (), {})()
    request.app.state.server_config = type("Config", (), {"gateway_mode": None})()
    return request


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client_host,headers",
    [
        # Genuine local request — no longer auto-trusted.
        ("127.0.0.1", {"host": "127.0.0.1:30080"}),
        # Reverse-proxy-fronted remote attacker arriving with TCP peer 127.0.0.1.
        (
            "127.0.0.1",
            {"host": "gpustack.example.com", "x-forwarded-for": "8.8.8.8"},
        ),
        # IPv6 loopback.
        ("::1", {"host": "[::1]:30080"}),
        # External IP.
        ("10.0.0.1", {"host": "gpustack.example.com"}),
    ],
)
async def test_get_current_user_requires_credentials(monkeypatch, client_host, headers):
    # The auto-admin localhost shortcut has been removed entirely.
    # Every unauthenticated request — local, proxied, or remote — must be
    # rejected.
    request = _make_request(headers=headers, client_host=client_host)

    first_by_field = AsyncMock()
    get_by_username = AsyncMock()
    monkeypatch.setattr("gpustack.api.auth.User.first_by_field", first_by_field)
    monkeypatch.setattr(
        "gpustack.api.auth.UserService.get_by_username", get_by_username
    )

    with pytest.raises(UnauthorizedException):
        await get_current_user(request=request)
    # No DB lookup path may fire when there are no credentials.
    first_by_field.assert_not_awaited()
    get_by_username.assert_not_awaited()


@pytest.mark.asyncio
async def test_oidc_callback_uses_system_trust_store(monkeypatch):
    captured = {}

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def request(self, method, url, data=None):
            return type(
                "Resp",
                (),
                {
                    "status_code": 200,
                    "text": '{"access_token":"token","id_token":"id"}',
                },
            )()

    request = type("Request", (), {})()
    request.app = type("App", (), {})()
    request.app.state = type("State", (), {})()
    request.app.state.server_config = type(
        "Config",
        (),
        {
            "oidc_client_id": "client-id",
            "oidc_client_secret": "client-secret",
            "oidc_redirect_uri": "https://gpustack.example.com/auth/oidc/callback",
            "openid_configuration": {
                "token_endpoint": "https://issuer.example.com/token"
            },
            "external_auth_name": None,
            "external_auth_full_name": None,
            "external_auth_avatar_url": None,
            "external_auth_default_inactive": False,
            "external_auth_insecure_skip_tls_verify": False,
            # Group sync defaults to False; this test exercises the
            # trust-store path, not group sync.
            "external_auth_group_sync": False,
            "external_auth_groups": None,
        },
    )()
    request.app.state.jwt_manager = type(
        "JWTManager", (), {"create_jwt_token": lambda self, username: "jwt-token"}
    )()
    request.query_params = {"code": "auth-code", "state": "test-state"}
    request.cookies = {"gpustack_oidc_state": "test-state"}
    # Read by the session-cookie hardening (``secure=request.url.scheme
    # == "https"``). Pretend the inbound request was HTTPS so the
    # ``secure`` flag would have flipped on in production.
    request.url = type("URL", (), {"scheme": "https"})()

    monkeypatch.setattr("gpustack.routes.auth.httpx.AsyncClient", FakeAsyncClient)
    monkeypatch.setattr("gpustack.routes.auth.use_proxy_env_for_url", lambda url: False)
    monkeypatch.setattr(
        "gpustack.routes.auth.get_oidc_user_data",
        AsyncMock(return_value={"email": "user@example.com", "name": "Test User"}),
    )
    # Return an existing user already tagged as the OIDC source so the
    # cross-provider-takeover guard treats this as a legitimate repeat
    # login rather than a username collision from a different IdP.
    from gpustack.schemas.users import AuthProviderEnum

    existing_oidc_user = type(
        "User", (), {"is_active": True, "source": AuthProviderEnum.OIDC}
    )()
    monkeypatch.setattr(
        "gpustack.routes.auth.User.first_by_fields",
        AsyncMock(return_value=existing_oidc_user),
    )

    response = await oidc_callback(request=request, session=object())

    assert response.status_code in (302, 307)
    assert captured["trust_env"] is False
    assert captured["timeout"] is not None
    assert isinstance(captured["verify"], ssl.SSLContext)


@pytest.mark.asyncio
async def test_legacy_server_token_principal_authenticates():
    """Pre-2.0 workers authenticate every request with Basic
    ``system/worker/<uuid>:<server-token>``. The minted in-memory
    principal must construct without tripping the schema's
    is_admin-on-non-USER guard and pass the worker / cluster gates."""
    from fastapi.security import HTTPBasicCredentials

    from gpustack.api.auth import (
        authenticate_system_principal,
        get_cluster_principal,
        get_worker_principal,
        is_server_token_principal,
    )
    from gpustack.schemas.principals import PrincipalType

    config = type("Config", (), {"token": "server-token"})()
    principal = await authenticate_system_principal(
        config,
        HTTPBasicCredentials(username="system/worker/abc", password="server-token"),
    )

    assert principal is not None
    assert principal.kind == PrincipalType.SYSTEM
    assert principal.is_admin is False
    assert principal.id is None
    assert is_server_token_principal(principal)
    assert (await get_cluster_principal(principal)) is principal
    assert (await get_worker_principal(principal)) is principal

    # Wrong password mints nothing.
    rejected = await authenticate_system_principal(
        config,
        HTTPBasicCredentials(username="system/worker/abc", password="wrong"),
    )
    assert rejected is None


@pytest.mark.asyncio
async def test_persisted_system_principal_without_links_hits_admin_gate():
    """A persisted SYSTEM principal (id set) with neither worker nor
    cluster back-reference is NOT the server-token principal and must
    fall through to the admin gate."""
    from gpustack.api.auth import get_worker_principal, is_server_token_principal
    from gpustack.api.exceptions import ForbiddenException
    from gpustack.schemas.principals import Principal, PrincipalType

    principal = Principal(name="system/worker-orphan", kind=PrincipalType.SYSTEM)
    principal.id = 123

    assert not is_server_token_principal(principal)
    with pytest.raises(ForbiddenException):
        await get_worker_principal(principal)


def test_saml_settings_signature_floor():
    """Signature floor: at least one of ``wantAssertionsSigned`` /
    ``wantMessagesSigned`` must be True after the helper resolves the
    operator's ``--saml-security``. Both defaulting to False (the
    OneLogin ship state) would let ``process_response`` admit forged
    (unsigned) assertions — the vulnerability this fix exists to
    close. Operators who already opted in to either — some IdPs sign
    only the Response, others only the Assertion — keep their choice.
    """
    config = MagicMock()

    # Operator passed no security config → toolkit ships with both
    # off; helper defaults ``wantAssertionsSigned`` on to enforce the
    # floor.
    config.saml_security = "{}"
    security = auth_route._saml_settings(config)["security"]
    assert security.get("wantAssertionsSigned") is True

    # Operator signs only the outer ``<Response>`` (some IdPs do this,
    # not the Assertion). Respect that — don't force both.
    config.saml_security = '{"wantAssertionsSigned": false, "wantMessagesSigned": true}'
    security = auth_route._saml_settings(config)["security"]
    assert security.get("wantAssertionsSigned") is False
    assert security.get("wantMessagesSigned") is True

    # Operator signs only the ``<Assertion>`` (typical Keycloak
    # setup). Respect that too.
    config.saml_security = '{"wantAssertionsSigned": true, "wantMessagesSigned": false}'
    security = auth_route._saml_settings(config)["security"]
    assert security.get("wantAssertionsSigned") is True
    assert security.get("wantMessagesSigned") is False


def test_provider_gate_allows_only_matching_active_provider():
    """The SSO routes parse attacker-supplied IdP payloads, so each must
    stay closed unless its provider is the configured active one.
    ``external_auth_type`` resolves to at most one provider, so the gate
    passes only for its own provider and rejects every other value
    (including ``None``)."""
    all_providers = (
        AuthProviderEnum.OIDC,
        AuthProviderEnum.SAML,
        AuthProviderEnum.CAS,
    )
    for guarded in all_providers:
        gate = auth_route._provider_gate(guarded)
        request = MagicMock()
        # The matching provider passes.
        request.app.state.server_config.external_auth_type = guarded
        assert gate(request) is None
        # Every other provider (and the unconfigured None) is rejected.
        for other in (None, *all_providers):
            if other == guarded:
                continue
            request.app.state.server_config.external_auth_type = other
            with pytest.raises(BadRequestException):
                gate(request)


def _sso_test_client(external_auth_type) -> TestClient:
    """Mount the real auth router (prefix ``/auth``, matching production)
    on a bare app so the sub-router ``dependencies`` and ``include_router``
    wiring is exercised end-to-end over HTTP."""
    app = FastAPI()
    register_handlers(app)
    app.include_router(auth_route.router, prefix="/auth")
    app.state.server_config = SimpleNamespace(external_auth_type=external_auth_type)
    return TestClient(app)


@pytest.mark.parametrize(
    "login_path, provider",
    [
        ("/auth/oidc/login", AuthProviderEnum.OIDC),
        ("/auth/saml/login", AuthProviderEnum.SAML),
        ("/auth/cas/login", AuthProviderEnum.CAS),
    ],
)
def test_sso_login_route_gated_by_active_provider(login_path, provider):
    """Structural gating over real HTTP: each SSO route must return 400
    when ``external_auth_type`` names a different provider (or nothing).
    Asserting 400 rather than merely non-200 also catches a dropped
    ``include_router`` — an unmounted route would surface as 404."""
    for active in (
        None,
        AuthProviderEnum.OIDC,
        AuthProviderEnum.SAML,
        AuthProviderEnum.CAS,
    ):
        if active == provider:
            continue
        client = _sso_test_client(active)
        resp = client.get(login_path, follow_redirects=False)
        assert resp.status_code == 400, (login_path, active, resp.status_code)
        assert "is not configured" in resp.json()["message"]


def test_saml_unsigned_escape_hatch_detects_both_false():
    """The unsigned escape hatch flags only when *both*
    ``wantAssertionsSigned`` and ``wantMessagesSigned`` are the
    literal ``False`` from operator input — a missing key, or a
    half-opt-out, must not turn the hatch on. The callback branches
    on this to decide whether to skip signature verification, so
    correctness of the detection is security-load-bearing."""
    config = MagicMock()

    # Both explicitly False → hatch on
    config.saml_security = (
        '{"wantAssertionsSigned": false, "wantMessagesSigned": false}'
    )
    assert auth_route._saml_unsigned_escape_hatch(config) is True

    # Only one explicit False → hatch off (still hits the floor)
    config.saml_security = '{"wantAssertionsSigned": false}'
    assert auth_route._saml_unsigned_escape_hatch(config) is False

    # Missing keys → hatch off
    config.saml_security = "{}"
    assert auth_route._saml_unsigned_escape_hatch(config) is False

    # Explicit True somewhere → hatch off
    config.saml_security = '{"wantAssertionsSigned": true, "wantMessagesSigned": false}'
    assert auth_route._saml_unsigned_escape_hatch(config) is False


def test_extract_saml_attributes_unsigned_returns_nameid_and_attributes():
    """The unsigned parser must extract the same downstream shape the
    toolkit path produces (single-valued as bare string, multi-valued
    as list, plus a ``name_id`` key) so the rest of the callback runs
    unchanged."""
    xml = (
        '<samlp:Response xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol"'
        ' xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion">'
        "<saml:Assertion><saml:Subject><saml:NameID>alice@example.com"
        "</saml:NameID></saml:Subject><saml:AttributeStatement>"
        '<saml:Attribute Name="email"><saml:AttributeValue>alice@example.com'
        "</saml:AttributeValue></saml:Attribute>"
        '<saml:Attribute Name="Role"><saml:AttributeValue>engineer'
        "</saml:AttributeValue></saml:Attribute>"
        '<saml:Attribute Name="Role"><saml:AttributeValue>admin'
        "</saml:AttributeValue></saml:Attribute>"
        "</saml:AttributeStatement></saml:Assertion></samlp:Response>"
    ).encode()
    attrs = auth_route._extract_saml_attributes_unsigned(xml)
    assert attrs["name_id"] == "alice@example.com"
    assert attrs["email"] == "alice@example.com"
    # Repeated ``<Attribute Name="Role">`` should be merged into a
    # list, matching ``allowRepeatAttributeName=True`` in the toolkit
    # path.
    assert attrs["Role"] == ["engineer", "admin"]


def test_extract_saml_attributes_unsigned_rejects_xxe():
    """Even in the unsigned escape-hatch path, the parser must not
    resolve external entities — otherwise turning signature
    verification off for local IdP tests would also open an XXE
    file-read vector."""
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<!DOCTYPE samlp:Response ["
        '  <!ENTITY xxe SYSTEM "file:///etc/passwd">'
        "]>"
        '<samlp:Response xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol"'
        ' xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion">'
        "<saml:Assertion><saml:Subject><saml:NameID>&xxe;</saml:NameID>"
        "</saml:Subject></saml:Assertion></samlp:Response>"
    ).encode()
    attrs = auth_route._extract_saml_attributes_unsigned(xml)
    # No expansion happened — the parser didn't fetch /etc/passwd.
    assert "root:" not in (attrs.get("name_id") or "")
    assert "/etc/passwd" not in (attrs.get("name_id") or "")


_SAML_ENV_WRAPPER = (
    '<samlp:Response xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol" '
    'xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion" '
    'xmlns:ds="http://www.w3.org/2000/09/xmldsig#">'
    "{signatures}"
    "<saml:Assertion>{assertion_signature}<saml:Subject>"
    "<saml:NameID>alice@example.com</saml:NameID>"
    "</saml:Subject></saml:Assertion></samlp:Response>"
)
_FAKE_SIG = "<ds:Signature><ds:SignatureValue>dummy</ds:SignatureValue></ds:Signature>"


def _saml_response_xml(*, response_signed: bool, assertion_signed: bool) -> bytes:
    return _SAML_ENV_WRAPPER.format(
        signatures=_FAKE_SIG if response_signed else "",
        assertion_signature=_FAKE_SIG if assertion_signed else "",
    ).encode()


def test_saml_settings_adapts_to_response_only_signature():
    """When Keycloak / ADFS is configured to sign the outer
    ``<Response>`` but not the inner ``<Assertion>`` (Keycloak's own
    docs describe this as valid: ``Sign Documents=On, Sign
    Assertions=Off``), the callback must ask the toolkit to require
    the message signature rather than reject with "Assertion not
    signed". Adaptive detection sets ``wantMessagesSigned`` based on
    the DOM shape of what actually arrived — no ``--saml-security``
    tweaking needed for this common config."""

    config = MagicMock()
    config.saml_security = "{}"
    xml = _saml_response_xml(response_signed=True, assertion_signed=False)
    security = auth_route._saml_settings(config, xml_bytes=xml)["security"]
    assert security.get("wantMessagesSigned") is True
    # Assertion-signed floor NOT forced in — the response *is* signed,
    # just at the outer level.
    assert security.get("wantAssertionsSigned") is not True


def test_saml_settings_adapts_to_assertion_only_signature():
    """Symmetric case: only the inner ``<Assertion>`` is signed
    (Keycloak default when ``Sign Assertions=On, Sign Documents=Off``)."""

    config = MagicMock()
    config.saml_security = "{}"
    xml = _saml_response_xml(response_signed=False, assertion_signed=True)
    security = auth_route._saml_settings(config, xml_bytes=xml)["security"]
    assert security.get("wantAssertionsSigned") is True
    assert security.get("wantMessagesSigned") is not True


def test_saml_settings_operator_explicit_wins_over_adaptive():
    """Operator strictness must not be dialled *down* by adaptive
    detection: a deployment that mandates Assertion signing (via
    ``--saml-security``) keeps that requirement even if the incoming
    response only signs the outer ``<Response>``. Better to reject a
    non-compliant IdP than silently lower the bar."""

    config = MagicMock()
    config.saml_security = '{"wantAssertionsSigned": true}'
    xml = _saml_response_xml(response_signed=True, assertion_signed=False)
    security = auth_route._saml_settings(config, xml_bytes=xml)["security"]
    assert security.get("wantAssertionsSigned") is True
    # Adaptive still added the ``wantMessagesSigned`` since the
    # Response *is* signed and the operator didn't explicitly refuse.
    assert security.get("wantMessagesSigned") is True


def test_saml_settings_unsigned_response_still_hits_floor():
    """If the DOM has neither signature and the operator didn't
    opt in either way, the floor kicks in with
    ``wantAssertionsSigned=True`` so the toolkit refuses the
    (attacker-forgeable) unsigned response."""

    config = MagicMock()
    config.saml_security = "{}"
    xml = _saml_response_xml(response_signed=False, assertion_signed=False)
    security = auth_route._saml_settings(config, xml_bytes=xml)["security"]
    assert security.get("wantAssertionsSigned") is True


def test_saml_settings_defaults_allow_repeat_attribute_name():
    """SAML allows multi-valued attributes as either repeated
    ``<AttributeValue>`` inside one ``<Attribute>`` or as multiple
    ``<Attribute Name="X">`` elements. Keycloak's default mappers
    emit the latter (role_list + role_name both write ``Role``);
    the toolkit's out-of-box strict mode rejects that. Default the
    knob on so real IdPs work without extra config; operators who
    want strict mode can opt in via ``--saml-security``."""

    config = MagicMock()

    config.saml_security = "{}"
    assert (
        auth_route._saml_settings(config)["security"]["allowRepeatAttributeName"]
        is True
    )

    # Operator explicit opt-out is respected — strict-mode
    # deployments can still catch mis-configured IdPs.
    config.saml_security = '{"allowRepeatAttributeName": false}'
    assert (
        auth_route._saml_settings(config)["security"]["allowRepeatAttributeName"]
        is False
    )


def _saml_callback_request(saml_response_b64: str, **config_overrides) -> object:
    """Build a request double for the SAML callback tests. The
    callback derives OneLogin's ``current_url`` from ``saml_sp_acs_url``,
    not from ``request.url``, so the URL fields on the request are
    only used for the ``get_data`` payload."""

    request = MagicMock()
    request.method = "POST"

    async def _form():
        return {"SAMLResponse": saml_response_b64}

    request.form = _form
    request.query_params = {}

    cfg = request.app.state.server_config
    cfg.external_auth_type = AuthProviderEnum.SAML
    cfg.saml_security = "{}"
    cfg.saml_sp_acs_url = "http://localhost:9000/auth/saml/callback"
    cfg.external_auth_name = None
    cfg.external_auth_full_name = None
    cfg.external_auth_avatar_url = None
    cfg.external_auth_default_inactive = False
    for k, v in config_overrides.items():
        setattr(cfg, k, v)
    return request


def _patch_saml_auth(monkeypatch, **auth_overrides):
    """Replace ``OneLogin_Saml2_Auth`` with a scripted fake so tests
    can exercise the callback's flow around signature validation
    without producing real signed XML. The fake returns the caller's
    scripted ``get_errors`` / ``is_authenticated`` / ``get_nameid`` /
    ``get_attributes`` values from ``process_response`` onwards."""

    fake = MagicMock()
    fake.process_response = MagicMock()
    fake.get_errors = MagicMock(return_value=auth_overrides.get("errors", []))
    fake.get_last_error_reason = MagicMock(
        return_value=auth_overrides.get("error_reason", "")
    )
    fake.is_authenticated = MagicMock(
        return_value=auth_overrides.get("authenticated", True)
    )
    fake.get_nameid = MagicMock(return_value=auth_overrides.get("nameid", ""))
    fake.get_attributes = MagicMock(return_value=auth_overrides.get("attributes", {}))
    monkeypatch.setattr(auth_route, "OneLogin_Saml2_Auth", MagicMock(return_value=fake))
    return fake


@pytest.mark.asyncio
async def test_saml_callback_rejects_when_toolkit_reports_errors(monkeypatch):
    """Signature failure surfaces via ``auth.get_errors()``. The
    callback must raise so the decorator produces an ``auth_failed``
    redirect — never mint a JWT for an unverified assertion."""

    _patch_saml_auth(
        monkeypatch,
        errors=["invalid_response"],
        error_reason="Signature validation failed",
    )
    request = _saml_callback_request("dummy-b64")

    with pytest.raises(UnauthorizedException) as exc:
        await auth_route.saml_callback.__wrapped__(request=request, session=MagicMock())
    assert "Signature validation failed" in str(exc.value.message)


@pytest.mark.asyncio
async def test_saml_callback_rejects_when_not_authenticated(monkeypatch):
    """The toolkit can complete ``process_response`` with an empty
    error list but ``is_authenticated`` still False (e.g. status
    element carries a non-Success code). Treat that the same as a
    signature failure — refuse to trust NameID / attributes."""

    _patch_saml_auth(monkeypatch, errors=[], authenticated=False)
    request = _saml_callback_request("dummy-b64")

    with pytest.raises(UnauthorizedException) as exc:
        await auth_route.saml_callback.__wrapped__(request=request, session=MagicMock())
    assert "not authenticated" in str(exc.value.message).lower()


@pytest.mark.asyncio
async def test_saml_callback_missing_configured_username_attribute_rejects(
    monkeypatch,
):
    """When the operator pins ``external_auth_name`` to a specific
    attribute and the (verified) assertion doesn't carry it, fail
    loudly at the source rather than letting ``None`` flow into the
    user resolve/create path."""

    _patch_saml_auth(
        monkeypatch,
        authenticated=True,
        nameid="alice@example.com",
        attributes={"email": ["alice@example.com"]},
    )
    # Operator pointed at ``employeeId`` — the (mocked-verified)
    # assertion above only carries ``email`` and ``name_id``.
    request = _saml_callback_request("dummy-b64", external_auth_name="employeeId")

    with pytest.raises(UnauthorizedException) as exc:
        await auth_route.saml_callback.__wrapped__(request=request, session=MagicMock())
    assert "employeeId" in str(exc.value.message)


@pytest.mark.asyncio
async def test_saml_callback_derives_current_url_from_configured_acs(monkeypatch):
    """Reverse-proxy / UI-dev-server setups routinely land the callback
    request on an internal host:port that doesn't match what Keycloak
    signed the assertion for. The toolkit's Destination check would
    then reject valid assertions — the fix is to anchor
    ``current_url`` on the operator's configured ACS URL rather than
    on ``request.url``. This test pins that behaviour: even though
    the request's ``url`` claims a different host / port / scheme,
    ``OneLogin_Saml2_Auth`` is constructed with the ACS-derived
    request_data."""

    _patch_saml_auth(
        monkeypatch,
        authenticated=True,
        nameid="alice@example.com",
        attributes={"email": ["alice@example.com"]},
    )
    captured = {}

    def _capture(req, settings):
        captured["req"] = req
        captured["settings"] = settings
        fake = MagicMock()
        fake.process_response = MagicMock()
        fake.get_errors = MagicMock(return_value=[])
        fake.is_authenticated = MagicMock(return_value=True)
        fake.get_nameid = MagicMock(return_value="alice@example.com")
        fake.get_attributes = MagicMock(return_value={"email": ["alice@example.com"]})
        return fake

    monkeypatch.setattr(auth_route, "OneLogin_Saml2_Auth", _capture)
    monkeypatch.setattr(
        auth_route,
        "_resolve_or_provision_external_user",
        AsyncMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(
        auth_route, "_sync_saml_groups_if_enabled", AsyncMock(return_value=None)
    )

    request = _saml_callback_request(
        "dummy-b64",
        saml_sp_acs_url="https://gpustack.example.com:8443/auth/saml/callback",
    )
    # Whatever the *request* URL looks like (a UI dev server on 9000
    # proxying to a backend on some other port, TLS termination
    # elsewhere, ...) — the toolkit still gets the config-anchored
    # values.
    request.url.scheme = "http"
    request.url.hostname = "localhost"
    request.url.port = None
    request.url.path = "/proxied/path"

    request.app.state.jwt_manager.create_jwt_token = MagicMock(return_value="fake-jwt")

    await auth_route.saml_callback.__wrapped__(request=request, session=MagicMock())

    assert captured["req"]["http_host"] == "gpustack.example.com"
    assert captured["req"]["server_port"] == "8443"
    assert captured["req"]["https"] == "on"
    assert captured["req"]["script_name"] == "/auth/saml/callback"
    assert captured["req"]["post_data"] == {"SAMLResponse": "dummy-b64"}


@pytest.mark.asyncio
async def test_saml_callback_unsigned_escape_hatch_skips_toolkit(monkeypatch, caplog):
    """With the operator's explicit both-False opt-out, the callback
    must **not** call ``OneLogin_Saml2_Auth`` at all (its hard-coded
    "No Signature found" check would reject the unsigned response
    regardless of the ``wantX`` flags). It must instead extract
    NameID / attributes via the manual parser, log the loud warning
    on that path, and continue to JWT minting."""
    import base64
    import logging

    unsigned_xml = (
        '<samlp:Response xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol"'
        ' xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion">'
        "<saml:Assertion><saml:Subject><saml:NameID>alice@example.com"
        "</saml:NameID></saml:Subject><saml:AttributeStatement>"
        '<saml:Attribute Name="email"><saml:AttributeValue>alice@example.com'
        "</saml:AttributeValue></saml:Attribute></saml:AttributeStatement>"
        "</saml:Assertion></samlp:Response>"
    ).encode()
    encoded = base64.b64encode(unsigned_xml).decode("ascii")

    # Make the toolkit blow up loudly if it *is* invoked — proves the
    # escape hatch path really bypassed it.
    def _fail_if_called(*args, **kwargs):
        raise AssertionError(
            "OneLogin_Saml2_Auth must not be invoked on escape-hatch path"
        )

    monkeypatch.setattr(auth_route, "OneLogin_Saml2_Auth", _fail_if_called)
    monkeypatch.setattr(
        auth_route,
        "_resolve_or_provision_external_user",
        AsyncMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(
        auth_route, "_sync_saml_groups_if_enabled", AsyncMock(return_value=None)
    )

    request = _saml_callback_request(
        encoded,
        saml_security='{"wantAssertionsSigned": false, "wantMessagesSigned": false}',
    )
    request.app.state.jwt_manager.create_jwt_token = MagicMock(return_value="fake-jwt")

    with caplog.at_level(logging.WARNING, logger="gpustack.routes.auth"):
        await auth_route.saml_callback.__wrapped__(request=request, session=MagicMock())

    # Loud, unmissable warning fired on the request that took this path.
    assert any(
        "signature verification is disabled" in rec.message.lower()
        and "production" in rec.message.lower()
        for rec in caplog.records
    )


def _build_client_ip_request(headers, *, gateway_mode, gateway_token="tok"):
    request = type("Request", (), {})()
    request.headers = headers
    request.client = type("Client", (), {"host": "10.0.0.1"})()
    request.app = type("App", (), {})()
    request.app.state = type("State", (), {})()
    request.app.state.server_config = type(
        "Config",
        (),
        {
            "gateway_mode": gateway_mode,
            "get_derived_gateway_token": lambda self: gateway_token,
        },
    )()
    return request


def test_client_ip_getter_prefers_x_real_ip():
    request = _build_client_ip_request(
        {
            GATEWAY_AUTH_TOKEN_HEADER: "tok",
            "X-Real-IP": "1.2.3.4",
            "X-Forwarded-For": "9.9.9.9, 5.6.7.8",
        },
        gateway_mode=GatewayModeEnum.embedded,
    )

    assert client_ip_getter(request) == "1.2.3.4"


def test_client_ip_getter_falls_back_to_rightmost_xff():
    # No X-Real-IP: use the rightmost XFF entry (what the trusted edge proxy
    # observed), not the spoofable leftmost one.
    request = _build_client_ip_request(
        {
            GATEWAY_AUTH_TOKEN_HEADER: "tok",
            "X-Forwarded-For": "1.1.1.1, 5.6.7.8",
        },
        gateway_mode=GatewayModeEnum.embedded,
    )

    assert client_ip_getter(request) == "5.6.7.8"


def test_client_ip_getter_ignores_headers_without_valid_gateway_token():
    # Invalid gateway token: the forwarded IP headers must not be trusted;
    # fall back to the peer connection address.
    request = _build_client_ip_request(
        {
            GATEWAY_AUTH_TOKEN_HEADER: "wrong",
            "X-Real-IP": "1.2.3.4",
            "X-Forwarded-For": "1.1.1.1, 5.6.7.8",
        },
        gateway_mode=GatewayModeEnum.embedded,
    )

    assert client_ip_getter(request) == "10.0.0.1"


def test_client_ip_getter_ignores_headers_when_gateway_disabled():
    request = _build_client_ip_request(
        {
            GATEWAY_AUTH_TOKEN_HEADER: "tok",
            "X-Real-IP": "1.2.3.4",
        },
        gateway_mode=GatewayModeEnum.disabled,
    )

    assert client_ip_getter(request) == "10.0.0.1"
