import hashlib
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gpustack.api.exceptions import NotFoundException
from gpustack.api.tenant import TenantContext
from gpustack.routes import model_provider as route_module
from gpustack.schemas.model_provider import ModelProviderCreate
from gpustack.schemas.principals import OrgRole, PrincipalType


SOURCE_TOKEN = "test-source-token"
SOURCE_HASH = hashlib.sha256(SOURCE_TOKEN.encode()).hexdigest()


@pytest.fixture
def provider_store(monkeypatch):
    now = datetime.now(timezone.utc)
    source = SimpleNamespace(
        id=7,
        owner_principal_id=10,
        deleted_at=None,
        api_tokens=[SOURCE_TOKEN, "test-unselected-token"],
    )
    lookup = AsyncMock(return_value=source)

    async def create(session, source):
        return dict(source, id=8, created_at=now, updated_at=now)

    create_mock = AsyncMock(side_effect=create)
    monkeypatch.setattr(
        route_module.ModelProvider, "one_by_fields", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(route_module.ModelProvider, "one_by_id", lookup)
    monkeypatch.setattr(route_module.ModelProvider, "create", create_mock)
    monkeypatch.setattr(route_module, "platform_principal_id", lambda: 1)
    return SimpleNamespace(source=source, lookup=lookup, create=create_mock)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "org_id,is_admin,source_state,allowed",
    [
        pytest.param(20, False, "active", False, id="other-org"),
        pytest.param(10, False, "active", True, id="same-org"),
        pytest.param(20, True, "active", False, id="admin-scoped-other-org"),
        pytest.param(10, True, "active", True, id="admin-scoped-same-org"),
        pytest.param(None, True, "active", True, id="admin-all-orgs"),
        pytest.param(10, False, "deleted", False, id="deleted-same-org"),
        pytest.param(20, False, "deleted", False, id="deleted-other-org"),
        pytest.param(None, True, "deleted", False, id="deleted-admin-all-orgs"),
        pytest.param(10, False, "missing", False, id="missing"),
    ],
)
async def test_clone_source_access(
    provider_store, org_id, is_admin, source_state, allowed
):
    ctx = TenantContext(
        user=SimpleNamespace(kind=PrincipalType.USER),
        is_platform_admin=is_admin,
        current_principal_id=org_id,
        org_role=OrgRole.OWNER,
    )
    if source_state == "deleted":
        provider_store.source.deleted_at = datetime.now(timezone.utc)
    elif source_state == "missing":
        provider_store.lookup.return_value = None
    input = ModelProviderCreate.model_validate(
        {
            "name": "cloned",
            "clone_from_id": 7,
            "api_tokens": [{"hash": SOURCE_HASH}],
            "config": {
                "type": "openai",
                "openaiCustomUrl": "https://destination.example/v1",
            },
            "models": [{"name": "x", "category": "llm"}],
        }
    )

    if not allowed:
        with pytest.raises(NotFoundException) as raised:
            await route_module.create_model_provider(None, ctx, input)
        assert raised.value.status_code == 404
        assert raised.value.message == "provider 7 to clone from not found"
        provider_store.create.assert_not_awaited()
        return

    result = await route_module.create_model_provider(None, ctx, input)

    provider_store.create.assert_awaited_once()
    saved = provider_store.create.call_args.kwargs["source"]
    assert saved["owner_principal_id"] == (org_id or 1)
    assert saved["api_tokens"] == [SOURCE_TOKEN]
    assert saved["config"]["openaiCustomUrl"] == "https://destination.example/v1"
    assert result.api_tokens[0].hash == SOURCE_HASH
    assert result.api_tokens[0].input is None


@pytest.mark.asyncio
async def test_create_without_clone(provider_store):
    ctx = TenantContext(
        user=SimpleNamespace(kind=PrincipalType.USER),
        is_platform_admin=False,
        current_principal_id=20,
        org_role=OrgRole.OWNER,
    )
    input = ModelProviderCreate.model_validate(
        {
            "name": "new-provider",
            "api_tokens": [{"input": "test-new-token"}],
            "config": {"type": "openai"},
            "models": [{"name": "x", "category": "llm"}],
        }
    )

    result = await route_module.create_model_provider(None, ctx, input)

    provider_store.lookup.assert_not_awaited()
    provider_store.create.assert_awaited_once()
    saved = provider_store.create.call_args.kwargs["source"]
    assert saved["owner_principal_id"] == 20
    assert saved["api_tokens"] == ["test-new-token"]
    assert result.owner_principal_id == 20
