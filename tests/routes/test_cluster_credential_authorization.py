from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gpustack.api.exceptions import InvalidException, NotFoundException
from gpustack.api.tenant import TenantContext
from gpustack.routes import cloud_credentials, clusters
from gpustack.schemas.clusters import (
    CloudCredential,
    CloudCredentialUpdate,
    Cluster,
    ClusterCreate,
    ClusterProvider,
    ClusterUpdate,
)
from gpustack.schemas.principals import OrgRole, PrincipalType


def context(admin=False, principal_id=101):
    return TenantContext(
        user=SimpleNamespace(id=11, kind=PrincipalType.USER),
        is_platform_admin=admin,
        current_principal_id=principal_id,
        org_role=OrgRole.OWNER if principal_id is not None else None,
    )


@pytest.fixture
def binding(monkeypatch):
    cluster = Cluster(
        id=10,
        name="cloud",
        owner_principal_id=101,
        provider=ClusterProvider.DigitalOcean,
        credential_id=20,
        server_url="https://example.invalid",
    )
    credential = CloudCredential(
        id=20,
        name="cloud-key",
        owner_principal_id=101,
        provider=ClusterProvider.DigitalOcean,
        secret="private-test-secret",
    )
    mocks = SimpleNamespace(
        cluster=cluster,
        credential=credential,
        session=AsyncMock(),
        lookup=AsyncMock(return_value=credential),
        create=AsyncMock(return_value=cluster),
        update=AsyncMock(),
        save=AsyncMock(),
    )
    monkeypatch.setattr(CloudCredential, "one_by_id", mocks.lookup)
    monkeypatch.setattr(Cluster, "one_by_id", AsyncMock(return_value=cluster))
    monkeypatch.setattr(Cluster, "one_by_fields", AsyncMock(return_value=None))
    monkeypatch.setattr(Cluster, "first_by_field", AsyncMock(return_value=cluster))
    monkeypatch.setattr(Cluster, "create", mocks.create)
    monkeypatch.setattr(Cluster, "update", mocks.update)
    monkeypatch.setattr(Cluster, "save", mocks.save)
    monkeypatch.setattr(
        clusters.Principal, "create", AsyncMock(return_value=SimpleNamespace(id=30))
    )
    monkeypatch.setattr(clusters.ApiKey, "create", AsyncMock())
    monkeypatch.setattr(clusters, "platform_principal_id", lambda: 1)
    monkeypatch.setattr(clusters, "get_secret_hash", lambda secret: "test-hash")
    monkeypatch.setattr(
        clusters,
        "get_global_config",
        lambda: SimpleNamespace(server_external_url="https://example.invalid"),
    )
    return mocks


async def call_cluster_route(binding, operation, ctx):
    if operation == "create":
        return await clusters.create_cluster(
            binding.session,
            ctx,
            ClusterCreate(
                name="cloud",
                owner_principal_id=101,
                provider=ClusterProvider.DigitalOcean,
                credential_id=20,
            ),
        )
    return await clusters.update_cluster(
        binding.session, ctx, 10, ClusterUpdate(name="cloud")
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["create", "update"])
@pytest.mark.parametrize(
    "invalid,exception",
    [
        ("missing", NotFoundException),
        ("deleted", NotFoundException),
        ("foreign", NotFoundException),
        ("ownerless", InvalidException),
        ("provider", InvalidException),
    ],
)
async def test_cluster_rejects_invalid_credential_before_writing(
    binding, operation, invalid, exception
):
    if invalid == "missing":
        binding.lookup.return_value = None
    elif invalid == "deleted":
        binding.credential.deleted_at = datetime.now(timezone.utc)
    elif invalid == "foreign":
        binding.credential.owner_principal_id = 202
    elif invalid == "ownerless":
        binding.credential.owner_principal_id = None
    else:
        binding.credential.provider = ClusterProvider.Shuihua

    with pytest.raises(exception) as exc:
        await call_cluster_route(binding, operation, context())

    assert "private-test-secret" not in exc.value.message
    binding.create.assert_not_awaited()
    binding.update.assert_not_awaited()
    binding.save.assert_not_awaited()
    binding.session.commit.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["create", "update"])
@pytest.mark.parametrize(
    "admin,principal_id", [(False, 101), (True, 101), (True, None)]
)
async def test_cluster_accepts_credential_of_its_owner(
    binding, operation, admin, principal_id
):
    result = await call_cluster_route(binding, operation, context(admin, principal_id))

    assert result is binding.cluster
    binding.lookup.assert_awaited_once_with(binding.session, 20)
    if operation == "create":
        binding.create.assert_awaited_once()
        assert binding.create.call_args.args[1].credential_id == 20
        binding.session.commit.assert_awaited_once()
    else:
        binding.update.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["create", "update"])
async def test_admin_cannot_bind_another_organizations_credential(binding, operation):
    binding.credential.owner_principal_id = 202

    with pytest.raises(InvalidException):
        await call_cluster_route(binding, operation, context(True, None))

    binding.create.assert_not_awaited()
    binding.update.assert_not_awaited()
    binding.session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_admin_acting_as_org_cannot_bind_invisible_credential(binding):
    with pytest.raises(NotFoundException):
        await call_cluster_route(binding, "create", context(True, 202))

    binding.create.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider", [ClusterProvider.Docker, ClusterProvider.Kubernetes]
)
async def test_non_cloud_cluster_needs_no_credential(binding, provider):
    cluster = ClusterCreate(name="local", provider=provider, owner_principal_id=101)

    await clusters.check_cluster_credential(binding.session, context(), cluster)

    binding.lookup.assert_not_awaited()


def test_cluster_update_cannot_replace_credential():
    update = ClusterUpdate.model_validate({"name": "cloud", "credential_id": 99})

    assert "credential_id" not in update.model_fields_set
    assert "credential_id" not in update.model_dump()


@pytest.mark.asyncio
@pytest.mark.parametrize("admin", [False, True])
@pytest.mark.parametrize(
    "changed",
    [
        {"owner_principal_id": 202},
        {"owner_principal_id": None},
        {"provider": ClusterProvider.Shuihua},
    ],
)
async def test_credential_update_preserves_binding_identity(
    binding, monkeypatch, admin, changed
):
    persist = AsyncMock()
    monkeypatch.setattr(CloudCredential, "update", persist)

    with pytest.raises(InvalidException):
        await cloud_credentials.update(
            binding.session,
            context(admin, None if admin else 101),
            20,
            CloudCredentialUpdate(name="cloud-key", **changed),
        )

    persist.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_identity", [False, True])
async def test_credential_secret_can_be_rotated(
    binding, monkeypatch, explicit_identity
):
    persist = AsyncMock()
    monkeypatch.setattr(CloudCredential, "update", persist)
    identity = (
        {"owner_principal_id": 101, "provider": ClusterProvider.DigitalOcean}
        if explicit_identity
        else {}
    )
    request = CloudCredentialUpdate(
        name="cloud-key", secret="rotated-test-secret", **identity
    )

    await cloud_credentials.update(binding.session, context(), 20, request)

    persist.assert_awaited_once_with(
        binding.credential, session=binding.session, source=request
    )
