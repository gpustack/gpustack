"""Unit tests for :mod:`gpustack.routes.users` covering the
``source`` × ``password`` validation matrix on create and update
paths. Mock-based per :mod:`tests.api.test_p2_routes`.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from gpustack.api.exceptions import InvalidException
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


def test_create_user_invalid_source_rejected_at_schema():
    # Pydantic enum validation catches this before the route runs,
    # so FastAPI emits 422 instead of 500 from the catch-all.
    with pytest.raises(ValidationError):
        UserCreate(
            username="alice",
            password="StrongPassw0rd!",
            source="Google",
        )


# ---- update_user / update_user_me password-bypass guards -------------------


def _existing_user(source: AuthProviderEnum, *, user_id: int = 42):
    u = MagicMock()
    u.id = user_id
    u.is_active = True
    u.is_admin = False
    u.source = source
    return u


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
    session = MagicMock()
    user = _existing_user(AuthProviderEnum.Local)
    monkeypatch.setattr(users_route.User, "one_by_id", AsyncMock(return_value=user))

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

    body = users_route.UserUpdate(username="alice", password="StrongPassw0rd!")
    await users_route.update_user(session=session, id=42, user_in=body)
    set_password_mock.assert_awaited_once()


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
