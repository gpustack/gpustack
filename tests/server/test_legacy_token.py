"""Deployment tokens must survive database storage and worker authentication."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gpustack.api.auth import get_user_from_api_token
from gpustack.schemas.api_keys import ApiKey
from gpustack.schemas.principals import Principal, PrincipalType
from gpustack.security import LEGACY_TOKEN_ACCESS_KEY
from gpustack.server import server as server_module
from gpustack.server.services import APIKeyService, UserService


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "token",
    [
        "123456",
        "e10adc3949ba59abbe56e057f20f883e",
        "gpustack_3192253c-c11c-75ed-6334-ea9505da4ad9_123456",
        "gpustack__123456",
    ],
)
async def test_bootstrap_token_authenticates_when_empty_strings_become_null(
    monkeypatch, token
):
    principal = Principal(id=7, kind=PrincipalType.SYSTEM, name="system/cluster/1")
    principal.api_keys = []
    cluster = SimpleNamespace(
        registration_token=None,
        system_principal_id=principal.id,
        update=AsyncMock(),
    )
    monkeypatch.setattr(
        server_module,
        "get_default_cluster_principal",
        AsyncMock(return_value=SimpleNamespace(cluster=cluster)),
    )
    monkeypatch.setattr(Principal, "one_by_id", AsyncMock(return_value=principal))
    stored = {}

    async def create(session, key, auto_commit):
        # openGauss A mode converts empty strings to NULL before applying
        # NOT NULL and uniqueness constraints to the access-key column.
        persisted_access = key.access_key or None
        assert persisted_access is not None
        assert persisted_access not in stored
        assert not auto_commit
        key.id = 11
        stored[persisted_access] = key
        return key

    create_mock = AsyncMock(side_effect=create)
    monkeypatch.setattr(ApiKey, "create", create_mock)
    session = AsyncMock()
    server = server_module.Server(SimpleNamespace(token=token), worker_process=None)

    await server._migrate_legacy_token(session)
    await server._migrate_legacy_token(session)

    create_mock.assert_awaited_once()
    session.commit.assert_awaited_once()
    assert cluster.registration_token == token
    key = stored[LEGACY_TOKEN_ACCESS_KEY]
    assert key.user_id == principal.id
    assert key.secret_key_digest is None

    async def lookup(self, access_key):
        return stored.get(access_key)

    monkeypatch.setattr(APIKeyService, "get_by_access_key", lookup)
    monkeypatch.setattr(UserService, "get_by_id", AsyncMock(return_value=principal))
    presentations = [token]
    if not token.startswith("gpustack_"):
        presentations.append(f"gpustack_3192253c-c11c-75ed-6334-ea9505da4ad9_{token}")
    for presented in presentations:
        assert await get_user_from_api_token(session, presented) == (principal, key)
