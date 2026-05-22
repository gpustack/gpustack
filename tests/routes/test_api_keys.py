from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gpustack.routes import api_keys
from gpustack.schemas.api_keys import ApiKey, ApiKeyListParams, ApiKeysPublic
from gpustack.schemas.common import Pagination
from gpustack.schemas.principals import OrgRole
from gpustack.schemas.users import User


def _ctx(*, user_id=1, is_admin=False, current_principal_id=10, org_role=None):
    return SimpleNamespace(
        user=User(id=user_id, name=f"user-{user_id}", is_admin=is_admin),
        current_principal_id=current_principal_id,
        org_role=org_role,
    )


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

    await api_keys.get_api_keys(
        session=object(),
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

    await api_keys.get_api_keys(
        session=object(),
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
