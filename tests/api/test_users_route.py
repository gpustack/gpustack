"""Unit tests for :mod:`gpustack.routes.users` covering the
``source`` × ``password`` validation matrix on create and update
paths. Mock-based per :mod:`tests.api.test_p2_routes`.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from gpustack.api.exceptions import InternalServerErrorException, InvalidException
from gpustack.routes import users as users_route
from gpustack.schemas.users import AuthProviderEnum, UserCreate


def _session_no_existing():
    """Session whose dup-username lookup returns ``None``."""
    session = MagicMock()
    result = MagicMock()
    result.first = MagicMock(return_value=None)
    session.exec = AsyncMock(return_value=result)
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    session.add = MagicMock()
    return session


def _patch_creation_path(monkeypatch, *, is_admin=False):
    """Stub the persistence side of ``create_user``. Returns the
    ``set_password`` mock for invocation assertions."""
    created = MagicMock()
    created.id = 42
    created.is_admin = is_admin
    monkeypatch.setattr(users_route.User, "create", AsyncMock(return_value=created))
    set_password_mock = AsyncMock()
    monkeypatch.setattr(users_route, "set_password", set_password_mock)
    monkeypatch.setattr(
        users_route, "_to_user_public", AsyncMock(return_value=MagicMock())
    )
    return set_password_mock


@pytest.mark.asyncio
async def test_create_user_local_with_password_succeeds(monkeypatch):
    set_password_mock = _patch_creation_path(monkeypatch)
    session = _session_no_existing()
    body = UserCreate(
        username="alice",
        password="StrongPassw0rd!",
        source=AuthProviderEnum.Local,
    )
    await users_route.create_user(session=session, user_in=body)
    set_password_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_user_local_without_password_rejected():
    session = _session_no_existing()
    body = UserCreate(username="alice", source=AuthProviderEnum.Local)
    with pytest.raises(InvalidException):
        await users_route.create_user(session=session, user_in=body)


@pytest.mark.asyncio
async def test_create_user_oidc_with_password_rejected():
    session = _session_no_existing()
    body = UserCreate(
        username="alice",
        password="StrongPassw0rd!",
        source=AuthProviderEnum.OIDC,
    )
    with pytest.raises(InvalidException):
        await users_route.create_user(session=session, user_in=body)


@pytest.mark.asyncio
async def test_create_user_oidc_without_password_succeeds(monkeypatch):
    set_password_mock = _patch_creation_path(monkeypatch)
    session = _session_no_existing()
    body = UserCreate(username="alice", source=AuthProviderEnum.OIDC)
    await users_route.create_user(session=session, user_in=body)
    set_password_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_user_oidc_require_password_change_rejected():
    session = _session_no_existing()
    body = UserCreate(
        username="alice",
        source=AuthProviderEnum.OIDC,
        require_password_change=True,
    )
    with pytest.raises(InvalidException):
        await users_route.create_user(session=session, user_in=body)


def test_create_user_accepts_arbitrary_source_string():
    # ``source`` is a free-form string column so new SSO kinds
    # (LDAP, vendor-specific, etc.) can be recorded without a schema
    # change. The values in :class:`AuthProviderEnum` are references,
    # not a closed set — pick a name that is *not* one of them to
    # actually exercise the open-string contract.
    body = UserCreate(
        username="alice",
        source="Google",
    )
    assert body.source == "Google"


# ---- update_user / update_user_me password-bypass guards -------------------


def _existing_user(source: AuthProviderEnum, *, user_id: int = 42):
    u = MagicMock()
    u.id = user_id
    u.is_active = True
    u.is_admin = False
    u.source = source
    return u


def _session_mock():
    """Session double for :func:`update_user` tests. The route now
    composes multiple writes under a single transaction and commits
    once at the end, so ``commit`` needs to be awaitable — a bare
    ``MagicMock()`` would fail on ``await session.commit()``."""
    session = MagicMock()
    session.commit = AsyncMock()
    return session


@pytest.mark.asyncio
async def test_update_user_rejects_password_for_oidc_user(monkeypatch):
    session = MagicMock()
    user = _existing_user(AuthProviderEnum.OIDC)
    monkeypatch.setattr(users_route.User, "one_by_id", AsyncMock(return_value=user))
    body = users_route.UserUpdate(username="alice", password="StrongPassw0rd!")
    with pytest.raises(InvalidException):
        await users_route.update_user(session=session, id=42, user_in=body)


@pytest.mark.asyncio
async def test_update_user_rejects_require_password_change_for_oidc_user(monkeypatch):
    session = MagicMock()
    user = _existing_user(AuthProviderEnum.OIDC)
    monkeypatch.setattr(users_route.User, "one_by_id", AsyncMock(return_value=user))
    body = users_route.UserUpdate(username="alice", require_password_change=True)
    with pytest.raises(InvalidException):
        await users_route.update_user(session=session, id=42, user_in=body)


@pytest.mark.asyncio
async def test_update_user_allows_password_for_local_user(monkeypatch):
    session = _session_mock()
    user = _existing_user(AuthProviderEnum.Local)
    monkeypatch.setattr(users_route.User, "one_by_id", AsyncMock(return_value=user))

    service_instance = MagicMock()
    service_instance.update = AsyncMock()
    service_instance.invalidate_cache = AsyncMock()
    monkeypatch.setattr(
        users_route, "UserService", MagicMock(return_value=service_instance)
    )
    set_password_mock = AsyncMock()
    monkeypatch.setattr(users_route, "set_password", set_password_mock)
    monkeypatch.setattr(
        users_route, "_to_user_public", AsyncMock(return_value=MagicMock())
    )

    body = users_route.UserUpdate(username="alice", password="StrongPassw0rd!")
    await users_route.update_user(session=session, id=42, user_in=body)
    set_password_mock.assert_awaited_once()


# ---- update_user source-switch (Local ↔ SSO) -------------------------------


def _patch_update_path(monkeypatch):
    """Stub the persistence side of ``update_user``. Returns the
    ``set_password`` and ``clear_password`` mocks so individual tests
    can assert whether the credential side moved in lockstep with the
    source switch."""
    service_instance = MagicMock()
    service_instance.update = AsyncMock()
    service_instance.invalidate_cache = AsyncMock()
    monkeypatch.setattr(
        users_route, "UserService", MagicMock(return_value=service_instance)
    )
    set_password_mock = AsyncMock()
    clear_password_mock = AsyncMock()
    monkeypatch.setattr(users_route, "set_password", set_password_mock)
    monkeypatch.setattr(users_route, "clear_password", clear_password_mock)
    monkeypatch.setattr(
        users_route, "_to_user_public", AsyncMock(return_value=MagicMock())
    )
    return set_password_mock, clear_password_mock


@pytest.mark.asyncio
async def test_update_user_switch_local_to_cas_clears_password(monkeypatch):
    """Local → CAS must soft-delete the local password row so /login
    can't be used as a parallel ingress that bypasses the IdP."""
    session = _session_mock()
    user = _existing_user(AuthProviderEnum.Local)
    monkeypatch.setattr(users_route.User, "one_by_id", AsyncMock(return_value=user))
    set_password_mock, clear_password_mock = _patch_update_path(monkeypatch)

    body = users_route.UserUpdate(username="alice", source=AuthProviderEnum.CAS)
    await users_route.update_user(session=session, id=42, user_in=body)
    clear_password_mock.assert_awaited_once()
    set_password_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_user_switch_local_to_cas_rejects_password(monkeypatch):
    """A password in the *same* PUT that switches Local → CAS would be
    immediately discarded — reject loudly so the admin doesn't think
    they've configured a fallback credential."""
    session = MagicMock()
    user = _existing_user(AuthProviderEnum.Local)
    monkeypatch.setattr(users_route.User, "one_by_id", AsyncMock(return_value=user))
    body = users_route.UserUpdate(
        username="alice",
        source=AuthProviderEnum.CAS,
        password="StrongPassw0rd!",
    )
    with pytest.raises(InvalidException):
        await users_route.update_user(session=session, id=42, user_in=body)


@pytest.mark.asyncio
async def test_update_user_switch_cas_to_local_requires_password(monkeypatch):
    """SSO → Local without a new password would leave the account
    password-less and lock the user out of /login."""
    session = MagicMock()
    user = _existing_user(AuthProviderEnum.CAS)
    monkeypatch.setattr(users_route.User, "one_by_id", AsyncMock(return_value=user))
    body = users_route.UserUpdate(username="alice", source=AuthProviderEnum.Local)
    with pytest.raises(InvalidException):
        await users_route.update_user(session=session, id=42, user_in=body)


@pytest.mark.asyncio
async def test_update_user_switch_cas_to_local_with_password_succeeds(monkeypatch):
    """SSO → Local with a password sets the local credential in the
    same request — the user can log in via /login from then on."""
    session = _session_mock()
    user = _existing_user(AuthProviderEnum.CAS)
    monkeypatch.setattr(users_route.User, "one_by_id", AsyncMock(return_value=user))
    set_password_mock, clear_password_mock = _patch_update_path(monkeypatch)

    body = users_route.UserUpdate(
        username="alice",
        source=AuthProviderEnum.Local,
        password="StrongPassw0rd!",
    )
    await users_route.update_user(session=session, id=42, user_in=body)
    set_password_mock.assert_awaited_once()
    clear_password_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_user_omitted_source_leaves_existing_source(monkeypatch):
    """An admin saving an unrelated field (e.g. ``full_name``) on an
    SSO user must not silently flip the row back to Local. With
    ``source`` omitted from the PUT body the credential side stays
    untouched — neither ``set_password`` nor ``clear_password`` runs."""
    session = _session_mock()
    user = _existing_user(AuthProviderEnum.OIDC)
    monkeypatch.setattr(users_route.User, "one_by_id", AsyncMock(return_value=user))
    set_password_mock, clear_password_mock = _patch_update_path(monkeypatch)

    body = users_route.UserUpdate(username="alice", full_name="Alice")
    await users_route.update_user(session=session, id=42, user_in=body)
    set_password_mock.assert_not_awaited()
    clear_password_mock.assert_not_awaited()


def test_update_user_rejects_unknown_source_string():
    """Typos and freeform strings in ``source`` on the update path
    would silently corrupt the auth column and lock the user out —
    ``_resolve_external_user`` needs an exact enum match on the next
    login. Pydantic rejects at the schema layer, so a garbage value
    never reaches the DB write."""
    with pytest.raises(Exception):  # pydantic ValidationError
        users_route.UserUpdate(username="alice", source="banana")


def test_update_user_rejects_empty_source_string():
    """Empty string is the insidious case: truthiness-check treats
    it as "not switching", but the raw wire value would still land
    in the DB as ``""`` if it weren't rejected up front. Enum
    validation refuses it at the schema layer."""
    with pytest.raises(Exception):  # pydantic ValidationError
        users_route.UserUpdate(username="alice", source="")


@pytest.mark.asyncio
async def test_update_user_partial_put_does_not_reset_omitted_fields(monkeypatch):
    """A PUT that only touches ``full_name`` must NOT write the
    schema defaults for ``is_admin`` (False) and ``is_active`` (True)
    — that would silently demote admins and reactivate disabled
    accounts. ``exclude_unset=True`` on ``model_dump`` keeps the
    update to only the keys the client actually sent."""
    session = _session_mock()
    user = _existing_user(AuthProviderEnum.Local)
    monkeypatch.setattr(users_route.User, "one_by_id", AsyncMock(return_value=user))

    service_instance = MagicMock()
    service_instance.update = AsyncMock()
    service_instance.invalidate_cache = AsyncMock()
    monkeypatch.setattr(
        users_route, "UserService", MagicMock(return_value=service_instance)
    )
    monkeypatch.setattr(users_route, "set_password", AsyncMock())
    monkeypatch.setattr(users_route, "clear_password", AsyncMock())
    monkeypatch.setattr(
        users_route, "_to_user_public", AsyncMock(return_value=MagicMock())
    )

    body = users_route.UserUpdate(username="alice", full_name="Alice Renamed")
    await users_route.update_user(session=session, id=42, user_in=body)

    _args, kwargs = service_instance.update.await_args
    update_data = _args[1] if len(_args) > 1 else kwargs.get("source") or {}
    # Only the fields the client sent land in the write payload —
    # ``is_admin`` / ``is_active`` are absent, so the storage layer
    # leaves them untouched.
    assert "is_admin" not in update_data
    assert "is_active" not in update_data
    assert update_data.get("display_name") == "Alice Renamed"


@pytest.mark.asyncio
async def test_update_user_invalidates_cache_after_commit(monkeypatch):
    """Cache invalidation must run *after* the caller's commit —
    otherwise a concurrent read during the transaction window can
    refill the cache from the pre-commit row and outlive our
    invalidation, leaving stale entries around until TTL."""
    session = _session_mock()
    user = _existing_user(AuthProviderEnum.Local)
    monkeypatch.setattr(users_route.User, "one_by_id", AsyncMock(return_value=user))

    events = []
    session.commit = AsyncMock(side_effect=lambda: events.append("commit"))

    service_instance = MagicMock()
    service_instance.update = AsyncMock()
    service_instance.invalidate_cache = AsyncMock(
        side_effect=lambda *a, **kw: events.append("invalidate")
    )
    monkeypatch.setattr(
        users_route, "UserService", MagicMock(return_value=service_instance)
    )
    monkeypatch.setattr(users_route, "set_password", AsyncMock())
    monkeypatch.setattr(users_route, "clear_password", AsyncMock())
    monkeypatch.setattr(
        users_route, "_to_user_public", AsyncMock(return_value=MagicMock())
    )

    body = users_route.UserUpdate(username="alice", full_name="Alice")
    await users_route.update_user(session=session, id=42, user_in=body)

    # Strict ordering — invalidate must be observed *after* commit,
    # not before it or interleaved.
    assert events == ["commit", "invalidate"]


@pytest.mark.asyncio
async def test_update_user_source_switch_rolls_back_on_clear_password_failure(
    monkeypatch,
):
    """The three writes (source column, password create, password
    clear) must ride the same transaction. A ``clear_password``
    failure on a Local → SSO switch would otherwise leave the row
    committed with ``source=SSO`` while the local password hash is
    still verifiable — exactly the /login bypass state the switch is
    meant to eliminate."""
    session = _session_mock()
    user = _existing_user(AuthProviderEnum.Local)
    monkeypatch.setattr(users_route.User, "one_by_id", AsyncMock(return_value=user))

    service_instance = MagicMock()
    service_instance.update = AsyncMock()
    service_instance.invalidate_cache = AsyncMock()
    monkeypatch.setattr(
        users_route, "UserService", MagicMock(return_value=service_instance)
    )
    monkeypatch.setattr(users_route, "set_password", AsyncMock())
    monkeypatch.setattr(
        users_route,
        "clear_password",
        AsyncMock(side_effect=RuntimeError("simulated clear_password failure")),
    )
    monkeypatch.setattr(
        users_route, "_to_user_public", AsyncMock(return_value=MagicMock())
    )

    body = users_route.UserUpdate(username="alice", source=AuthProviderEnum.CAS)
    with pytest.raises(InternalServerErrorException):
        await users_route.update_user(session=session, id=42, user_in=body)

    # No commit — the source-column write must roll back with the
    # failed password clear, not persist on its own. And the middle
    # write must have used ``auto_commit=False`` so nothing had a
    # chance to leak through before the failure.
    session.commit.assert_not_awaited()
    service_instance.update.assert_awaited_once()
    _args, kwargs = service_instance.update.await_args
    assert kwargs.get("auto_commit") is False


@pytest.mark.asyncio
async def test_update_user_source_switch_commits_once_at_end(monkeypatch):
    """Sanity check the happy path: on a successful Local → SSO
    switch, the source-column write, the password-row clear, and the
    final commit are all separate calls, and ``commit`` fires exactly
    once at the tail — not once per write."""
    session = _session_mock()
    user = _existing_user(AuthProviderEnum.Local)
    monkeypatch.setattr(users_route.User, "one_by_id", AsyncMock(return_value=user))

    service_instance = MagicMock()
    service_instance.update = AsyncMock()
    service_instance.invalidate_cache = AsyncMock()
    monkeypatch.setattr(
        users_route, "UserService", MagicMock(return_value=service_instance)
    )
    monkeypatch.setattr(users_route, "set_password", AsyncMock())
    clear_password_mock = AsyncMock()
    monkeypatch.setattr(users_route, "clear_password", clear_password_mock)
    monkeypatch.setattr(
        users_route, "_to_user_public", AsyncMock(return_value=MagicMock())
    )

    body = users_route.UserUpdate(username="alice", source=AuthProviderEnum.CAS)
    await users_route.update_user(session=session, id=42, user_in=body)

    session.commit.assert_awaited_once()
    _args, update_kwargs = service_instance.update.await_args
    _args, clear_kwargs = clear_password_mock.await_args
    assert update_kwargs.get("auto_commit") is False
    assert clear_kwargs.get("auto_commit") is False


@pytest.mark.asyncio
async def test_update_user_me_rejects_password_for_oidc_user():
    session = MagicMock()
    user = _existing_user(AuthProviderEnum.OIDC)
    body = users_route.UserSelfUpdate(password="StrongPassw0rd!")
    with pytest.raises(InvalidException):
        await users_route.update_user_me(session=session, user=user, user_in=body)


@pytest.mark.asyncio
async def test_update_user_me_allows_password_for_local_user(monkeypatch):
    session = MagicMock()
    user = _existing_user(AuthProviderEnum.Local)

    service_instance = MagicMock()
    service_instance.update = AsyncMock()
    monkeypatch.setattr(
        users_route, "UserService", MagicMock(return_value=service_instance)
    )
    set_password_mock = AsyncMock()
    monkeypatch.setattr(users_route, "set_password", set_password_mock)
    monkeypatch.setattr(
        users_route, "_to_user_public", AsyncMock(return_value=MagicMock())
    )

    body = users_route.UserSelfUpdate(password="StrongPassw0rd!")
    await users_route.update_user_me(session=session, user=user, user_in=body)
    set_password_mock.assert_awaited_once()


# ── UserListParams sort field mapping ──────────────────────────────


def test_user_list_params_sort_by_username_maps_to_name():
    """Frontend sends sort_by=username; backend maps to DB column name."""
    params = users_route.UserListParams(sort_by="username")
    assert params.order_by == [("name", "asc")]


def test_user_list_params_sort_by_username_descending():
    params = users_route.UserListParams(sort_by="-username")
    assert params.order_by == [("name", "desc")]


def test_user_list_params_sort_by_full_name_maps_to_display_name():
    """Form full_name (UI), field full_name (API) → DB column display_name."""
    params = users_route.UserListParams(sort_by="full_name")
    assert params.order_by == [("display_name", "asc")]


def test_user_list_params_rejects_storage_sort_field_name():
    with pytest.raises(InvalidException, match="not sortable"):
        users_route.UserListParams(sort_by="name")


def test_user_list_params_rejects_storage_sort_field_display_name():
    with pytest.raises(InvalidException, match="not sortable"):
        users_route.UserListParams(sort_by="display_name")


def test_user_list_params_rejects_unknown_sort_field():
    with pytest.raises(InvalidException, match="not sortable"):
        users_route.UserListParams(sort_by="unknown_column")


def test_user_list_params_no_sort_by_returns_none():
    params = users_route.UserListParams()
    assert params.order_by is None
