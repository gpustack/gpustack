from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gpustack.routes import api_keys
from gpustack.schemas.api_keys import (
    ApiKey,
    ApiKeyCreate,
    ApiKeyListParams,
    ApiKeysPublic,
)
from gpustack.schemas.common import Pagination
from gpustack.schemas.principals import OrgRole
from gpustack.schemas.users import User


def _ctx(*, user_id=1, is_admin=False, current_principal_id=10, org_role=None):
    return SimpleNamespace(
        user=User(id=user_id, name=f"user-{user_id}", is_admin=is_admin),
        current_principal_id=current_principal_id,
        org_role=org_role,
    )


class _NoopSession:
    """Stand-in for the session get_api_keys opens via async_session(); the DB
    query itself is monkeypatched, so it only needs the async context manager
    protocol."""

    async def __aenter__(self):
        return object()

    async def __aexit__(self, *exc):
        return False


@pytest.mark.asyncio
async def test_org_owner_lists_all_api_keys_in_current_org(monkeypatch):
    captured = {}

    async def fake_paginated_by_query(**kwargs):
        captured["fields"] = kwargs["fields"]
        return ApiKeysPublic(
            items=[],
            pagination=Pagination(page=1, perPage=100, total=0, totalPage=0),
        )

    monkeypatch.setattr(ApiKey, "paginated_by_query", fake_paginated_by_query)
    monkeypatch.setattr(api_keys, "async_session", lambda: _NoopSession())

    await api_keys.get_api_keys(
        ctx=_ctx(org_role=OrgRole.OWNER),
        params=ApiKeyListParams(),
        user_id=None,
    )

    assert captured["fields"] == {"owner_principal_id": 10}


@pytest.mark.asyncio
async def test_org_member_lists_only_own_api_keys_in_current_org(monkeypatch):
    captured = {}

    async def fake_paginated_by_query(**kwargs):
        captured["fields"] = kwargs["fields"]
        return ApiKeysPublic(
            items=[],
            pagination=Pagination(page=1, perPage=100, total=0, totalPage=0),
        )

    monkeypatch.setattr(ApiKey, "paginated_by_query", fake_paginated_by_query)
    monkeypatch.setattr(api_keys, "async_session", lambda: _NoopSession())

    await api_keys.get_api_keys(
        ctx=_ctx(user_id=2, org_role=OrgRole.MEMBER),
        params=ApiKeyListParams(),
        user_id=None,
    )

    assert captured["fields"] == {"owner_principal_id": 10, "user_id": 2}


@pytest.mark.asyncio
async def test_org_owner_can_delete_other_users_api_key_in_current_org(monkeypatch):
    deleted = {}
    api_key = ApiKey(
        id=5,
        name="team-key",
        user_id=2,
        owner_principal_id=10,
        access_key="access",
        hashed_secret_key="secret",
        created_at=datetime(2026, 1, 1),
        updated_at=datetime(2026, 1, 1),
    )

    monkeypatch.setattr(ApiKey, "one_by_id", AsyncMock(return_value=api_key))

    class FakeAPIKeyService:
        def __init__(self, session):
            self.session = session

        async def delete(self, api_key):
            deleted["api_key_id"] = api_key.id

    monkeypatch.setattr(api_keys, "APIKeyService", FakeAPIKeyService)

    await api_keys.delete_api_key(
        session=object(),
        ctx=_ctx(user_id=1, org_role=OrgRole.OWNER),
        id=5,
    )

    assert deleted["api_key_id"] == 5


@pytest.mark.asyncio
async def test_admin_all_mode_creates_key_with_null_owner(monkeypatch):
    """Admin without an Org context (no X-Organization-Id) creates an
    untenant-pinned key — ``owner_principal_id`` stays NULL so the
    request resolver falls through to user-based resolution on each
    call (admin → None → ``bypass_tenant_filter``), giving the key
    the same cross-principal reach as the admin's cookie session.
    """
    monkeypatch.setattr(ApiKey, "one_by_fields", AsyncMock(return_value=None))

    captured = {}

    async def fake_create(session, api_key):
        captured["owner_principal_id"] = api_key.owner_principal_id
        captured["user_id"] = api_key.user_id
        api_key.id = 1
        api_key.created_at = datetime(2026, 1, 1)
        api_key.updated_at = datetime(2026, 1, 1)
        return api_key

    monkeypatch.setattr(ApiKey, "create", fake_create)

    await api_keys.create_api_key(
        session=object(),
        ctx=_ctx(user_id=1, is_admin=True, current_principal_id=None),
        key_in=ApiKeyCreate(name="admin-mgmt"),
    )

    assert captured["owner_principal_id"] is None
    assert captured["user_id"] == 1


@pytest.mark.asyncio
async def test_admin_in_org_act_as_lands_key_in_that_org(monkeypatch):
    """Admin with an Org context (X-Organization-Id set, or any non-admin
    caller) still pins the key to the chosen principal — this is the
    explicit ``act-as Org`` flow, not the cross-principal admin reach."""
    monkeypatch.setattr(ApiKey, "one_by_fields", AsyncMock(return_value=None))

    captured = {}

    async def fake_create(session, api_key):
        captured["owner_principal_id"] = api_key.owner_principal_id
        api_key.id = 2
        api_key.created_at = datetime(2026, 1, 1)
        api_key.updated_at = datetime(2026, 1, 1)
        return api_key

    monkeypatch.setattr(ApiKey, "create", fake_create)

    await api_keys.create_api_key(
        session=object(),
        ctx=_ctx(user_id=1, is_admin=True, current_principal_id=99),
        key_in=ApiKeyCreate(name="acme-mgmt"),
    )

    assert captured["owner_principal_id"] == 99
