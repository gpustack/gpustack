"""Cache-service route handler checks.

Handlers are driven directly with mocked tenant contexts and patched
ActiveRecord class methods, covering the create validation matrix, the
delete-protection scan over shared extended-KV-cache references, the
the user-facing immutability of identity
fields on update, and the instance-delete / log-proxy endpoints.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gpustack.api.exceptions import (
    AlreadyExistsException,
    BadRequestException,
    InternalServerErrorException,
    NotFoundException,
)
from gpustack.api.tenant import TenantContext
from gpustack.routes import cache_services as cache_services_route
from gpustack.schemas.cache_providers import (
    CacheProvider,
    CacheProviderField,
    CacheProviderIntegration,
    CacheProviderL2Backend,
    CacheProviderL2Field,
    CacheProviderVersionConfig,
)
from gpustack.schemas.cache_services import (
    CacheServiceConfig,
    CacheServiceCreate,
    CacheServiceL2Storage,
    CacheServiceStateEnum,
    CacheServiceUpdate,
)
from gpustack.worker.logs import LogOptions

from gpustack.schemas.models import ExtendedKVCacheConfig, KVCacheModeEnum
from gpustack.schemas.principals import PrincipalType

ORG_PRINCIPAL = 42


def _user_ctx(principal_id: int = ORG_PRINCIPAL) -> TenantContext:
    user = MagicMock()
    user.kind = PrincipalType.USER
    return TenantContext(
        user=user,
        is_platform_admin=False,
        current_principal_id=principal_id,
        org_role=None,
    )


def _system_ctx() -> TenantContext:
    """A worker/cluster service account (kind=SYSTEM, full bypass)."""
    user = MagicMock()
    user.kind = PrincipalType.SYSTEM
    return TenantContext(
        user=user,
        is_platform_admin=False,
        current_principal_id=None,
        org_role=None,
    )


def _provider(
    topology="replicas",
    custom_version=False,
    management_url=False,
) -> CacheProvider:
    return CacheProvider(
        name="LMCache",
        topology=topology,
        default_version="v1",
        versions={"v1": CacheProviderVersionConfig(image="lmcache:v1")},
        custom_version=custom_version,
        management_url=management_url,
        inference_backend_integrations=[CacheProviderIntegration(backend="vLLM")],
        fields=[
            CacheProviderField(name="ram_size", type="number", required=True),
            CacheProviderField(name="chunk_size", type="number"),
        ],
    )


def _provider_with_l2() -> CacheProvider:
    provider = _provider()
    provider.l2_adapter_flag = "--l2-adapter"
    provider.l2_backends = {
        "fs": CacheProviderL2Backend(
            fields=[
                CacheProviderL2Field(name="base_path", required=True),
                CacheProviderL2Field(name="use_odirect", type="boolean"),
            ]
        ),
        "resp": CacheProviderL2Backend(
            fields=[
                CacheProviderL2Field(name="host", required=True),
                CacheProviderL2Field(name="port", type="number", required=True),
                CacheProviderL2Field(name="username", env_name="LMCACHE_RESP_USERNAME"),
                CacheProviderL2Field(
                    name="password", type="password", env_name="LMCACHE_RESP_PASSWORD"
                ),
                CacheProviderL2Field(name="max_capacity_gb", type="number"),
            ]
        ),
    }
    return provider


def _provider_with_optional_l2_adapter() -> CacheProvider:
    provider = _provider_with_l2()
    provider.l2_backends["xdfs"] = CacheProviderL2Backend(
        adapter_flag_optional=True,
        adapter_flag_default=False,
        adapter_flag_label="Enable L2 Adapter Flag",
        fields=[CacheProviderL2Field(name="tenant_id", required=True)],
    )
    return provider


def _l2_config(backend: str, **params) -> CacheServiceConfig:
    return CacheServiceConfig(
        fields={"ram_size": 20},
        l2_storages=[CacheServiceL2Storage(backend=backend, params=params)],
    )


def _l2_cascade_config(*storages: CacheServiceL2Storage) -> CacheServiceConfig:
    return CacheServiceConfig(fields={"ram_size": 20}, l2_storages=list(storages))


def _patch_provider(monkeypatch, provider: CacheProvider):
    monkeypatch.setattr(
        cache_services_route,
        "get_cache_provider",
        lambda name: provider if name.lower() == provider.name.lower() else None,
    )


def _patch_create_prereqs(monkeypatch, existing=None, worker=None):
    """Make the pre-provider create checks pass: an Org-owned cluster and
    no name collision. ``worker`` (when given) backs the managed-mode
    worker lookup."""
    cluster = SimpleNamespace(id=1, deleted_at=None, owner_principal_id=ORG_PRINCIPAL)
    monkeypatch.setattr(
        cache_services_route.Cluster, "one_by_id", AsyncMock(return_value=cluster)
    )
    monkeypatch.setattr(
        cache_services_route.CacheService,
        "one_by_fields",
        AsyncMock(return_value=existing),
    )
    monkeypatch.setattr(
        cache_services_route.Worker, "one_by_id", AsyncMock(return_value=worker)
    )


def _managed_create(**overrides) -> CacheServiceCreate:
    fields = dict(
        name="svc",
        provider_name="LMCache",
        cluster_id=1,
        worker_id=5,
        config=CacheServiceConfig(fields={"ram_size": 20}),
    )
    fields.update(overrides)
    return CacheServiceCreate(**fields)


# ---- create validation matrix ----


@pytest.mark.asyncio
async def test_create_requires_cluster_id():
    create_in = CacheServiceCreate.model_construct(
        name="svc",
        provider_name="LMCache",
        cluster_id=None,
    )
    with pytest.raises(BadRequestException) as exc_info:
        await cache_services_route.create_cache_service(
            session=MagicMock(), ctx=_user_ctx(), cache_service_in=create_in
        )

    assert "cluster_id" in exc_info.value.message


@pytest.mark.asyncio
async def test_create_rejects_unknown_provider(monkeypatch):
    _patch_create_prereqs(monkeypatch)
    _patch_provider(monkeypatch, _provider())

    with pytest.raises(BadRequestException) as exc_info:
        await cache_services_route.create_cache_service(
            session=MagicMock(),
            ctx=_user_ctx(),
            cache_service_in=_managed_create(provider_name="nope"),
        )

    assert "Unknown cache provider" in exc_info.value.message


@pytest.mark.asyncio
async def test_create_rejects_a_provider_this_installation_cannot_run(monkeypatch):
    """The catalog lists a provider nothing here can launch so the choice
    stays visible; a service naming it would only fail later, with less to
    go on."""
    _patch_create_prereqs(monkeypatch)
    _patch_provider(
        monkeypatch,
        CacheProvider(
            name="LMCache",
            unavailable_reason={"default": "Available in GPUStack Enterprise."},
        ),
    )

    with pytest.raises(BadRequestException) as exc_info:
        await cache_services_route.create_cache_service(
            session=MagicMock(),
            ctx=_user_ctx(),
            cache_service_in=_managed_create(),
        )

    assert "not available" in exc_info.value.message
    assert "GPUStack Enterprise" in exc_info.value.message


@pytest.mark.asyncio
async def test_create_rejects_unknown_provider_version(monkeypatch):
    _patch_create_prereqs(monkeypatch)
    _patch_provider(monkeypatch, _provider())

    with pytest.raises(BadRequestException) as exc_info:
        await cache_services_route.create_cache_service(
            session=MagicMock(),
            ctx=_user_ctx(),
            cache_service_in=_managed_create(provider_version="v999"),
        )

    assert "version" in exc_info.value.message


# ---- custom version (user-supplied image) ----


@pytest.mark.asyncio
async def test_create_accepts_custom_version_with_image(monkeypatch):
    worker = SimpleNamespace(id=5, deleted_at=None, cluster_id=1)
    _patch_create_prereqs(monkeypatch, worker=worker)
    _patch_provider(monkeypatch, _provider(custom_version=True))
    monkeypatch.setattr(
        cache_services_route.CacheService,
        "create",
        AsyncMock(side_effect=lambda session, source: SimpleNamespace(**source)),
    )

    created = await cache_services_route.create_cache_service(
        session=MagicMock(),
        ctx=_user_ctx(),
        cache_service_in=_managed_create(
            provider_version="custom",
            config=CacheServiceConfig(
                fields={"ram_size": 20}, image="myteam/lmcache:dev"
            ),
        ),
    )

    assert created.provider_version == "custom"
    assert created.config["image"] == "myteam/lmcache:dev"


@pytest.mark.asyncio
@pytest.mark.parametrize("config", [None, CacheServiceConfig(fields={"ram_size": 20})])
async def test_create_rejects_custom_version_without_image(monkeypatch, config):
    worker = SimpleNamespace(id=5, deleted_at=None, cluster_id=1)
    _patch_create_prereqs(monkeypatch, worker=worker)
    _patch_provider(monkeypatch, _provider(custom_version=True))

    with pytest.raises(BadRequestException) as exc_info:
        await cache_services_route.create_cache_service(
            session=MagicMock(),
            ctx=_user_ctx(),
            cache_service_in=_managed_create(provider_version="custom", config=config),
        )

    assert "config.image is required" in exc_info.value.message


@pytest.mark.asyncio
async def test_create_defaults_to_custom_version_without_declared_versions(monkeypatch):
    """A provider publishing no image declares no version to name, so an
    omitted provider_version reads as the custom one."""
    worker = SimpleNamespace(id=5, deleted_at=None, cluster_id=1)
    _patch_create_prereqs(monkeypatch, worker=worker)
    provider = _provider(custom_version=True)
    provider.versions = {}
    provider.default_version = None
    _patch_provider(monkeypatch, provider)
    monkeypatch.setattr(
        cache_services_route.CacheService,
        "create",
        AsyncMock(side_effect=lambda session, source: SimpleNamespace(**source)),
    )

    created = await cache_services_route.create_cache_service(
        session=MagicMock(),
        ctx=_user_ctx(),
        cache_service_in=_managed_create(
            config=CacheServiceConfig(
                fields={"ram_size": 20}, image="myteam/meshfusion:dev"
            ),
        ),
    )

    assert created.provider_version == "custom"
    assert created.config["image"] == "myteam/meshfusion:dev"


@pytest.mark.asyncio
async def test_create_rejects_missing_image_without_declared_versions(monkeypatch):
    """The image is the only way such a provider reaches one, so leaving
    it out fails as a missing custom-version image."""
    worker = SimpleNamespace(id=5, deleted_at=None, cluster_id=1)
    _patch_create_prereqs(monkeypatch, worker=worker)
    provider = _provider(custom_version=True)
    provider.versions = {}
    provider.default_version = None
    _patch_provider(monkeypatch, provider)

    with pytest.raises(BadRequestException) as exc_info:
        await cache_services_route.create_cache_service(
            session=MagicMock(),
            ctx=_user_ctx(),
            cache_service_in=_managed_create(
                config=CacheServiceConfig(fields={"ram_size": 20})
            ),
        )

    assert "config.image is required" in exc_info.value.message


@pytest.mark.asyncio
async def test_create_rejects_custom_version_without_provider_opt_in(monkeypatch):
    _patch_create_prereqs(monkeypatch)
    _patch_provider(monkeypatch, _provider())

    with pytest.raises(BadRequestException) as exc_info:
        await cache_services_route.create_cache_service(
            session=MagicMock(),
            ctx=_user_ctx(),
            cache_service_in=_managed_create(
                provider_version="custom",
                config=CacheServiceConfig(
                    fields={"ram_size": 20}, image="myteam/lmcache:dev"
                ),
            ),
        )

    assert "does not allow the custom version" in exc_info.value.message


@pytest.mark.asyncio
async def test_create_rejects_image_with_declared_version(monkeypatch):
    """With a declared version the provider's image is rendered, so a
    supplied config.image would be silently ignored."""
    _patch_create_prereqs(monkeypatch)
    _patch_provider(monkeypatch, _provider(custom_version=True))

    with pytest.raises(BadRequestException) as exc_info:
        await cache_services_route.create_cache_service(
            session=MagicMock(),
            ctx=_user_ctx(),
            cache_service_in=_managed_create(
                provider_version="v1",
                config=CacheServiceConfig(
                    fields={"ram_size": 20}, image="myteam/lmcache:dev"
                ),
            ),
        )

    assert "config.image is only applicable" in exc_info.value.message


@pytest.mark.asyncio
async def test_create_rejects_management_url_without_provider_support(monkeypatch):
    _patch_create_prereqs(monkeypatch)
    _patch_provider(monkeypatch, _provider())

    with pytest.raises(BadRequestException) as exc_info:
        await cache_services_route.create_cache_service(
            session=MagicMock(),
            ctx=_user_ctx(),
            cache_service_in=_managed_create(
                config=CacheServiceConfig(
                    fields={"ram_size": 20},
                    management_url="https://console.example.com",
                ),
            ),
        )

    assert "not supported by cache provider" in exc_info.value.message


@pytest.mark.asyncio
async def test_create_managed_without_worker_id_auto_places(monkeypatch):
    """A replicas service without a pinned worker leaves placement to
    the controller; creation must not demand one."""
    _patch_create_prereqs(monkeypatch)
    _patch_provider(monkeypatch, _provider())
    monkeypatch.setattr(
        cache_services_route.CacheService,
        "create",
        AsyncMock(side_effect=lambda session, source: SimpleNamespace(**source)),
    )

    created = await cache_services_route.create_cache_service(
        session=MagicMock(),
        ctx=_user_ctx(),
        cache_service_in=_managed_create(worker_id=None),
    )

    assert created.worker_id is None


@pytest.mark.asyncio
async def test_create_managed_rejects_worker_in_other_cluster(monkeypatch):
    worker = SimpleNamespace(id=5, deleted_at=None, cluster_id=2)
    _patch_create_prereqs(monkeypatch, worker=worker)
    _patch_provider(monkeypatch, _provider())

    with pytest.raises(BadRequestException) as exc_info:
        await cache_services_route.create_cache_service(
            session=MagicMock(),
            ctx=_user_ctx(),
            cache_service_in=_managed_create(),
        )

    assert "cluster" in exc_info.value.message


@pytest.mark.asyncio
async def test_create_managed_rejects_missing_worker(monkeypatch):
    _patch_create_prereqs(monkeypatch, worker=None)
    _patch_provider(monkeypatch, _provider())

    with pytest.raises(BadRequestException) as exc_info:
        await cache_services_route.create_cache_service(
            session=MagicMock(),
            ctx=_user_ctx(),
            cache_service_in=_managed_create(),
        )

    assert "not found" in exc_info.value.message


@pytest.mark.asyncio
async def test_create_per_node_rejects_worker_id(monkeypatch):
    """Per-node providers derive placement from the cluster's workers, so
    a worker pick at creation is a contradiction."""
    _patch_create_prereqs(monkeypatch)
    _patch_provider(monkeypatch, _provider(topology="per_node"))

    with pytest.raises(BadRequestException) as exc_info:
        await cache_services_route.create_cache_service(
            session=MagicMock(),
            ctx=_user_ctx(),
            cache_service_in=_managed_create(worker_id=5),
        )

    assert "worker_id is not applicable" in exc_info.value.message
    assert "per worker node" in exc_info.value.message


@pytest.mark.asyncio
async def test_create_per_node_accepts_without_worker_id(monkeypatch):
    _patch_create_prereqs(monkeypatch)
    _patch_provider(monkeypatch, _provider(topology="per_node"))
    monkeypatch.setattr(
        cache_services_route.CacheService,
        "create",
        AsyncMock(side_effect=lambda session, source: SimpleNamespace(**source)),
    )

    created = await cache_services_route.create_cache_service(
        session=MagicMock(),
        ctx=_user_ctx(),
        cache_service_in=_managed_create(worker_id=None),
    )

    assert created.worker_id is None
    assert created.state == CacheServiceStateEnum.PENDING


@pytest.mark.asyncio
async def test_create_replicas_accepts_worker_selector(monkeypatch):
    """Replicas components pick their workers from the selector's
    matching set, so a selector applies to every managed topology."""
    worker = SimpleNamespace(id=5, deleted_at=None, cluster_id=1)
    _patch_create_prereqs(monkeypatch, worker=worker)
    _patch_provider(monkeypatch, _provider(topology="replicas"))
    monkeypatch.setattr(
        cache_services_route.CacheService,
        "create",
        AsyncMock(side_effect=lambda session, source: SimpleNamespace(**source)),
    )

    created = await cache_services_route.create_cache_service(
        session=MagicMock(),
        ctx=_user_ctx(),
        cache_service_in=_managed_create(worker_selector={"gpu": "a100"}),
    )

    assert created.worker_selector == {"gpu": "a100"}


@pytest.mark.asyncio
async def test_create_per_node_accepts_worker_selector(monkeypatch):
    _patch_create_prereqs(monkeypatch)
    _patch_provider(monkeypatch, _provider(topology="per_node"))
    monkeypatch.setattr(
        cache_services_route.CacheService,
        "create",
        AsyncMock(side_effect=lambda session, source: SimpleNamespace(**source)),
    )

    created = await cache_services_route.create_cache_service(
        session=MagicMock(),
        ctx=_user_ctx(),
        cache_service_in=_managed_create(
            worker_id=None, worker_selector={"gpu": "a100", "zone": "z1"}
        ),
    )

    assert created.worker_selector == {"gpu": "a100", "zone": "z1"}


@pytest.mark.asyncio
async def test_create_normalizes_empty_worker_selector(monkeypatch):
    """An empty selector means "every worker" and is stored as None, on
    any service shape."""
    worker = SimpleNamespace(id=5, deleted_at=None, cluster_id=1)
    _patch_create_prereqs(monkeypatch, worker=worker)
    _patch_provider(monkeypatch, _provider(topology="replicas"))
    monkeypatch.setattr(
        cache_services_route.CacheService,
        "create",
        AsyncMock(side_effect=lambda session, source: SimpleNamespace(**source)),
    )

    created = await cache_services_route.create_cache_service(
        session=MagicMock(),
        ctx=_user_ctx(),
        cache_service_in=_managed_create(worker_selector={}),
    )

    assert created.worker_selector is None


@pytest.mark.asyncio
async def test_create_rejects_duplicate_name_per_owner(monkeypatch):
    _patch_create_prereqs(monkeypatch, existing=SimpleNamespace(id=1))
    _patch_provider(monkeypatch, _provider())

    with pytest.raises(AlreadyExistsException):
        await cache_services_route.create_cache_service(
            session=MagicMock(),
            ctx=_user_ctx(),
            cache_service_in=_managed_create(),
        )


@pytest.mark.asyncio
async def test_create_stamps_owner_and_pending_state(monkeypatch):
    worker = SimpleNamespace(id=5, deleted_at=None, cluster_id=1)
    _patch_create_prereqs(monkeypatch, worker=worker)
    _patch_provider(monkeypatch, _provider())

    created = {}

    async def _create(session, source):
        created.update(source)
        return SimpleNamespace(**source)

    monkeypatch.setattr(cache_services_route.CacheService, "create", _create)

    await cache_services_route.create_cache_service(
        session=MagicMock(),
        ctx=_user_ctx(),
        cache_service_in=_managed_create(),
    )

    assert created["owner_principal_id"] == ORG_PRINCIPAL
    assert created["state"] == CacheServiceStateEnum.PENDING


# ---- update immutability ----


def _existing_service(**overrides):
    fields = dict(
        id=9,
        name="svc",
        provider_name="LMCache",
        cluster_id=1,
        owner_principal_id=ORG_PRINCIPAL,
        deleted_at=None,
        update=AsyncMock(),
        delete=AsyncMock(),
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _update_in(**overrides) -> CacheServiceUpdate:
    # Update runs the create validation set, so the default payload is a
    # valid managed replicas shape (worker + capacity).
    fields = dict(
        name="svc",
        provider_name="LMCache",
        cluster_id=1,
        worker_id=5,
        config=CacheServiceConfig(fields={"ram_size": 20}),
    )
    fields.update(overrides)
    return CacheServiceUpdate(**fields)


def _patch_worker_lookup(monkeypatch, cluster_id: int = 1):
    monkeypatch.setattr(
        cache_services_route.Worker,
        "one_by_id",
        AsyncMock(
            return_value=SimpleNamespace(id=5, cluster_id=cluster_id, deleted_at=None)
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        {"provider_name": "Other"},
        {"cluster_id": 2},
    ],
)
async def test_update_rejects_identity_changes_for_users(monkeypatch, change):
    service = _existing_service()
    monkeypatch.setattr(
        cache_services_route.CacheService, "one_by_id", AsyncMock(return_value=service)
    )

    with pytest.raises(BadRequestException) as exc_info:
        await cache_services_route.update_cache_service(
            session=MagicMock(),
            ctx=_user_ctx(),
            id=9,
            cache_service_in=_update_in(**change),
        )
    assert "cannot be changed" in exc_info.value.message
    service.update.assert_not_called()


@pytest.mark.asyncio
async def test_update_allows_config_edit_for_users(monkeypatch):
    service = _existing_service()
    monkeypatch.setattr(
        cache_services_route.CacheService, "one_by_id", AsyncMock(return_value=service)
    )
    _patch_provider(monkeypatch, _provider())
    _patch_worker_lookup(monkeypatch)

    update_in = _update_in()
    await cache_services_route.update_cache_service(
        session=MagicMock(), ctx=_user_ctx(), id=9, cache_service_in=update_in
    )
    service.update.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_allows_system_writeback_of_any_field(monkeypatch):
    """Workers report state/port/health through this endpoint; the
    identity-field gate must not apply to SYSTEM principals."""
    service = _existing_service()
    monkeypatch.setattr(
        cache_services_route.CacheService, "one_by_id", AsyncMock(return_value=service)
    )
    _patch_provider(monkeypatch, _provider())
    _patch_worker_lookup(monkeypatch, cluster_id=2)

    update_in = _update_in(
        cluster_id=2, state=CacheServiceStateEnum.RUNNING, healthy=True
    )
    await cache_services_route.update_cache_service(
        session=MagicMock(), ctx=_system_ctx(), id=9, cache_service_in=update_in
    )
    service.update.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_accepts_worker_selector_for_replicas(monkeypatch):
    service = _existing_service()
    monkeypatch.setattr(
        cache_services_route.CacheService, "one_by_id", AsyncMock(return_value=service)
    )
    _patch_provider(monkeypatch, _provider(topology="replicas"))
    _patch_worker_lookup(monkeypatch)

    await cache_services_route.update_cache_service(
        session=MagicMock(),
        ctx=_user_ctx(),
        id=9,
        cache_service_in=_update_in(worker_selector={"gpu": "a100"}),
    )
    service.update.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_accepts_worker_selector_for_per_node(monkeypatch):
    service = _existing_service()
    monkeypatch.setattr(
        cache_services_route.CacheService, "one_by_id", AsyncMock(return_value=service)
    )
    _patch_provider(monkeypatch, _provider(topology="per_node"))

    await cache_services_route.update_cache_service(
        session=MagicMock(),
        ctx=_user_ctx(),
        id=9,
        cache_service_in=_update_in(worker_id=None, worker_selector={"gpu": "a100"}),
    )

    service.update.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_accepts_custom_version_with_image(monkeypatch):
    service = _existing_service()
    monkeypatch.setattr(
        cache_services_route.CacheService, "one_by_id", AsyncMock(return_value=service)
    )
    _patch_provider(monkeypatch, _provider(custom_version=True))
    _patch_worker_lookup(monkeypatch)

    await cache_services_route.update_cache_service(
        session=MagicMock(),
        ctx=_user_ctx(),
        id=9,
        cache_service_in=_update_in(
            provider_version="custom",
            config=CacheServiceConfig(
                fields={"ram_size": 20}, image="myteam/lmcache:dev"
            ),
        ),
    )

    service.update.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_rejects_custom_version_without_image(monkeypatch):
    service = _existing_service()
    monkeypatch.setattr(
        cache_services_route.CacheService, "one_by_id", AsyncMock(return_value=service)
    )
    _patch_provider(monkeypatch, _provider(custom_version=True))

    with pytest.raises(BadRequestException) as exc_info:
        await cache_services_route.update_cache_service(
            session=MagicMock(),
            ctx=_user_ctx(),
            id=9,
            cache_service_in=_update_in(
                provider_version="custom",
                config=CacheServiceConfig(fields={"ram_size": 20}),
            ),
        )

    assert "config.image is required" in exc_info.value.message
    service.update.assert_not_called()


@pytest.mark.asyncio
async def test_update_rejects_image_with_declared_version(monkeypatch):
    service = _existing_service()
    monkeypatch.setattr(
        cache_services_route.CacheService, "one_by_id", AsyncMock(return_value=service)
    )
    _patch_provider(monkeypatch, _provider(custom_version=True))

    with pytest.raises(BadRequestException) as exc_info:
        await cache_services_route.update_cache_service(
            session=MagicMock(),
            ctx=_user_ctx(),
            id=9,
            cache_service_in=_update_in(
                provider_version="v1",
                config=CacheServiceConfig(
                    fields={"ram_size": 20}, image="myteam/lmcache:dev"
                ),
            ),
        )

    assert "config.image is only applicable" in exc_info.value.message
    service.update.assert_not_called()


# ---- config parameters validation ----


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_parameters", [{"": [""]}, {"": ["   "]}, {"": ["--ok", ""]}]
)
async def test_create_rejects_blank_config_parameters(monkeypatch, bad_parameters):
    worker = SimpleNamespace(id=5, deleted_at=None, cluster_id=1)
    _patch_create_prereqs(monkeypatch, worker=worker)
    _patch_provider(monkeypatch, _provider())

    with pytest.raises(BadRequestException) as exc_info:
        await cache_services_route.create_cache_service(
            session=MagicMock(),
            ctx=_user_ctx(),
            cache_service_in=_managed_create(
                config=CacheServiceConfig(
                    fields={"ram_size": 20}, parameters=bad_parameters
                )
            ),
        )

    assert "parameters" in exc_info.value.message


@pytest.mark.asyncio
async def test_update_rejects_blank_config_parameters(monkeypatch):
    service = _existing_service()
    monkeypatch.setattr(
        cache_services_route.CacheService, "one_by_id", AsyncMock(return_value=service)
    )

    with pytest.raises(BadRequestException) as exc_info:
        await cache_services_route.update_cache_service(
            session=MagicMock(),
            ctx=_user_ctx(),
            id=9,
            cache_service_in=_update_in(
                config=CacheServiceConfig(parameters={"": [" "]})
            ),
        )

    assert "parameters" in exc_info.value.message
    service.update.assert_not_called()


@pytest.mark.asyncio
async def test_create_managed_requires_ram_size(monkeypatch):
    worker = SimpleNamespace(id=5, deleted_at=None, cluster_id=1)
    _patch_create_prereqs(monkeypatch, worker=worker)
    _patch_provider(monkeypatch, _provider())

    with pytest.raises(BadRequestException) as exc_info:
        await cache_services_route.create_cache_service(
            session=AsyncMock(),
            ctx=_user_ctx(),
            cache_service_in=_managed_create(config=None),
        )
    assert "ram_size" in exc_info.value.message


@pytest.mark.asyncio
async def test_create_accepts_free_form_config_parameters(monkeypatch):
    worker = SimpleNamespace(id=5, deleted_at=None, cluster_id=1)
    _patch_create_prereqs(monkeypatch, worker=worker)
    _patch_provider(monkeypatch, _provider())
    monkeypatch.setattr(
        cache_services_route.CacheService,
        "create",
        AsyncMock(side_effect=lambda session, source: SimpleNamespace(**source)),
    )

    created = await cache_services_route.create_cache_service(
        session=MagicMock(),
        ctx=_user_ctx(),
        cache_service_in=_managed_create(
            config=CacheServiceConfig(
                fields={"ram_size": 20},
                parameters={"": ["--eviction-policy=LRU"]},
            )
        ),
    )

    assert created.config["parameters"] == {"": ["--eviction-policy=LRU"]}


@pytest.mark.asyncio
async def test_create_rejects_parameters_for_an_undeclared_component(monkeypatch):
    """Flags are filed under the component whose binary takes them, so a
    component the provider does not declare would send them nowhere."""
    worker = SimpleNamespace(id=5, deleted_at=None, cluster_id=1)
    _patch_create_prereqs(monkeypatch, worker=worker)
    _patch_provider(monkeypatch, _provider())

    with pytest.raises(BadRequestException) as exc_info:
        await cache_services_route.create_cache_service(
            session=MagicMock(),
            ctx=_user_ctx(),
            cache_service_in=_managed_create(
                config=CacheServiceConfig(
                    fields={"ram_size": 20},
                    parameters={"coordinator": ["--log-level=debug"]},
                )
            ),
        )

    assert "coordinator" in exc_info.value.message


# ---- l2 storage validation ----


@pytest.mark.asyncio
async def test_create_rejects_l2_storage_without_provider_support(monkeypatch):
    worker = SimpleNamespace(id=5, deleted_at=None, cluster_id=1)
    _patch_create_prereqs(monkeypatch, worker=worker)
    _patch_provider(monkeypatch, _provider())

    with pytest.raises(BadRequestException) as exc_info:
        await cache_services_route.create_cache_service(
            session=MagicMock(),
            ctx=_user_ctx(),
            cache_service_in=_managed_create(
                config=_l2_config("fs", base_path="/data/l2")
            ),
        )

    assert "does not support L2 storage" in exc_info.value.message


@pytest.mark.asyncio
async def test_create_rejects_unknown_l2_backend(monkeypatch):
    worker = SimpleNamespace(id=5, deleted_at=None, cluster_id=1)
    _patch_create_prereqs(monkeypatch, worker=worker)
    _patch_provider(monkeypatch, _provider_with_l2())

    with pytest.raises(BadRequestException) as exc_info:
        await cache_services_route.create_cache_service(
            session=MagicMock(),
            ctx=_user_ctx(),
            cache_service_in=_managed_create(config=_l2_config("s3")),
        )

    assert "no L2 storage backend 's3'" in exc_info.value.message


@pytest.mark.asyncio
@pytest.mark.parametrize("params", [{}, {"base_path": ""}])
async def test_create_rejects_missing_required_l2_param(monkeypatch, params):
    worker = SimpleNamespace(id=5, deleted_at=None, cluster_id=1)
    _patch_create_prereqs(monkeypatch, worker=worker)
    _patch_provider(monkeypatch, _provider_with_l2())

    with pytest.raises(BadRequestException) as exc_info:
        await cache_services_route.create_cache_service(
            session=MagicMock(),
            ctx=_user_ctx(),
            cache_service_in=_managed_create(config=_l2_config("fs", **params)),
        )

    assert "Missing required" in exc_info.value.message
    assert "base_path" in exc_info.value.message


@pytest.mark.asyncio
async def test_create_rejects_undeclared_l2_param(monkeypatch):
    worker = SimpleNamespace(id=5, deleted_at=None, cluster_id=1)
    _patch_create_prereqs(monkeypatch, worker=worker)
    _patch_provider(monkeypatch, _provider_with_l2())

    with pytest.raises(BadRequestException) as exc_info:
        await cache_services_route.create_cache_service(
            session=MagicMock(),
            ctx=_user_ctx(),
            cache_service_in=_managed_create(
                config=_l2_config("fs", base_path="/data/l2", odirect=True)
            ),
        )

    assert "Unknown L2 storage parameter" in exc_info.value.message
    assert "odirect" in exc_info.value.message


@pytest.mark.asyncio
async def test_create_rejects_non_numeric_l2_number_param(monkeypatch):
    worker = SimpleNamespace(id=5, deleted_at=None, cluster_id=1)
    _patch_create_prereqs(monkeypatch, worker=worker)
    _patch_provider(monkeypatch, _provider_with_l2())

    with pytest.raises(BadRequestException) as exc_info:
        await cache_services_route.create_cache_service(
            session=MagicMock(),
            ctx=_user_ctx(),
            cache_service_in=_managed_create(
                config=_l2_config("resp", host="10.0.0.8", port="6379")
            ),
        )

    assert "'port' must be a number" in exc_info.value.message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config",
    [
        _l2_config("fs", base_path="/data/l2"),
        _l2_config(
            "resp",
            host="10.0.0.8",
            port=6379,
            password="s3cret",
            max_capacity_gb=100,
        ),
    ],
)
async def test_create_accepts_valid_l2_storage(monkeypatch, config):
    worker = SimpleNamespace(id=5, deleted_at=None, cluster_id=1)
    _patch_create_prereqs(monkeypatch, worker=worker)
    _patch_provider(monkeypatch, _provider_with_l2())
    monkeypatch.setattr(
        cache_services_route.CacheService,
        "create",
        AsyncMock(side_effect=lambda session, source: SimpleNamespace(**source)),
    )

    created = await cache_services_route.create_cache_service(
        session=MagicMock(),
        ctx=_user_ctx(),
        cache_service_in=_managed_create(config=config),
    )

    assert created.config["l2_storages"] == [
        entry.model_dump() for entry in config.l2_storages
    ]


@pytest.mark.asyncio
async def test_create_optional_l2_adapter_can_be_disabled_without_fields(
    monkeypatch,
):
    worker = SimpleNamespace(id=5, deleted_at=None, cluster_id=1)
    _patch_create_prereqs(monkeypatch, worker=worker)
    _patch_provider(monkeypatch, _provider_with_optional_l2_adapter())
    monkeypatch.setattr(
        cache_services_route.CacheService,
        "create",
        AsyncMock(side_effect=lambda session, source: SimpleNamespace(**source)),
    )
    storage = CacheServiceL2Storage(
        backend="xdfs", adapter_flag_enabled=False, params={}
    )

    created = await cache_services_route.create_cache_service(
        session=MagicMock(),
        ctx=_user_ctx(),
        cache_service_in=_managed_create(
            config=CacheServiceConfig(fields={"ram_size": 20}, l2_storages=[storage])
        ),
    )

    assert created.config["l2_storages"][0]["adapter_flag_enabled"] is False
    assert created.config["l2_storages"][0]["params"] == {}


@pytest.mark.asyncio
async def test_create_optional_l2_adapter_requires_fields_when_enabled(monkeypatch):
    worker = SimpleNamespace(id=5, deleted_at=None, cluster_id=1)
    _patch_create_prereqs(monkeypatch, worker=worker)
    _patch_provider(monkeypatch, _provider_with_optional_l2_adapter())

    with pytest.raises(BadRequestException) as exc_info:
        await cache_services_route.create_cache_service(
            session=MagicMock(),
            ctx=_user_ctx(),
            cache_service_in=_managed_create(
                config=CacheServiceConfig(
                    fields={"ram_size": 20},
                    l2_storages=[
                        CacheServiceL2Storage(
                            backend="xdfs", adapter_flag_enabled=True, params={}
                        )
                    ],
                )
            ),
        )

    assert "tenant_id" in exc_info.value.message


@pytest.mark.asyncio
async def test_create_accepts_l2_cascade_preserving_order(monkeypatch):
    """A multi-entry cascade is stored in declared order; the order carries
    read-preference semantics, so it must survive the round trip."""
    worker = SimpleNamespace(id=5, deleted_at=None, cluster_id=1)
    _patch_create_prereqs(monkeypatch, worker=worker)
    _patch_provider(monkeypatch, _provider_with_l2())
    monkeypatch.setattr(
        cache_services_route.CacheService,
        "create",
        AsyncMock(side_effect=lambda session, source: SimpleNamespace(**source)),
    )
    config = _l2_cascade_config(
        CacheServiceL2Storage(backend="fs", params={"base_path": "/data/ssd"}),
        CacheServiceL2Storage(
            backend="resp",
            params={"host": "10.0.0.8", "port": 6379, "password": "s3cret"},
        ),
    )

    created = await cache_services_route.create_cache_service(
        session=MagicMock(),
        ctx=_user_ctx(),
        cache_service_in=_managed_create(config=config),
    )

    assert [entry["backend"] for entry in created.config["l2_storages"]] == [
        "fs",
        "resp",
    ]


@pytest.mark.asyncio
async def test_create_accepts_repeated_l2_backend_without_env_fields(monkeypatch):
    """A backend whose fields all ride in the adapter JSON may appear in
    several cascade tiers."""
    worker = SimpleNamespace(id=5, deleted_at=None, cluster_id=1)
    _patch_create_prereqs(monkeypatch, worker=worker)
    _patch_provider(monkeypatch, _provider_with_l2())
    monkeypatch.setattr(
        cache_services_route.CacheService,
        "create",
        AsyncMock(side_effect=lambda session, source: SimpleNamespace(**source)),
    )
    config = _l2_cascade_config(
        CacheServiceL2Storage(backend="fs", params={"base_path": "/data/ssd"}),
        CacheServiceL2Storage(backend="fs", params={"base_path": "/data/hdd"}),
    )

    created = await cache_services_route.create_cache_service(
        session=MagicMock(),
        ctx=_user_ctx(),
        cache_service_in=_managed_create(config=config),
    )

    assert len(created.config["l2_storages"]) == 2


@pytest.mark.asyncio
async def test_create_rejects_l2_env_collision(monkeypatch):
    """Env vars are process-global: two entries delivering a value through
    the same env var cannot coexist in one cascade."""
    worker = SimpleNamespace(id=5, deleted_at=None, cluster_id=1)
    _patch_create_prereqs(monkeypatch, worker=worker)
    _patch_provider(monkeypatch, _provider_with_l2())
    config = _l2_cascade_config(
        CacheServiceL2Storage(
            backend="resp",
            params={"host": "10.0.0.8", "port": 6379, "password": "one"},
        ),
        CacheServiceL2Storage(
            backend="resp",
            params={"host": "10.0.0.9", "port": 6380, "password": "two"},
        ),
    )

    with pytest.raises(BadRequestException) as exc_info:
        await cache_services_route.create_cache_service(
            session=MagicMock(),
            ctx=_user_ctx(),
            cache_service_in=_managed_create(config=config),
        )

    assert "LMCACHE_RESP_PASSWORD" in exc_info.value.message
    assert "resp" in exc_info.value.message


@pytest.mark.asyncio
async def test_create_normalizes_empty_l2_storages_to_none(monkeypatch):
    """An empty list means "no L2 storage"; the stored config keeps the
    canonical None form."""
    worker = SimpleNamespace(id=5, deleted_at=None, cluster_id=1)
    _patch_create_prereqs(monkeypatch, worker=worker)
    _patch_provider(monkeypatch, _provider_with_l2())
    monkeypatch.setattr(
        cache_services_route.CacheService,
        "create",
        AsyncMock(side_effect=lambda session, source: SimpleNamespace(**source)),
    )

    created = await cache_services_route.create_cache_service(
        session=MagicMock(),
        ctx=_user_ctx(),
        cache_service_in=_managed_create(
            config=CacheServiceConfig(fields={"ram_size": 20}, l2_storages=[])
        ),
    )

    assert created.config["l2_storages"] is None


@pytest.mark.asyncio
async def test_update_rejects_invalid_l2_storage(monkeypatch):
    service = _existing_service()
    monkeypatch.setattr(
        cache_services_route.CacheService, "one_by_id", AsyncMock(return_value=service)
    )
    _patch_provider(monkeypatch, _provider_with_l2())

    with pytest.raises(BadRequestException) as exc_info:
        await cache_services_route.update_cache_service(
            session=MagicMock(),
            ctx=_user_ctx(),
            id=9,
            cache_service_in=_update_in(config=_l2_config("fs")),
        )

    assert "base_path" in exc_info.value.message
    service.update.assert_not_called()


@pytest.mark.asyncio
async def test_update_accepts_valid_l2_storage(monkeypatch):
    service = _existing_service()
    monkeypatch.setattr(
        cache_services_route.CacheService, "one_by_id", AsyncMock(return_value=service)
    )
    _patch_provider(monkeypatch, _provider_with_l2())
    _patch_worker_lookup(monkeypatch)

    await cache_services_route.update_cache_service(
        session=MagicMock(),
        ctx=_user_ctx(),
        id=9,
        cache_service_in=_update_in(config=_l2_config("fs", base_path="/data/l2")),
    )

    service.update.assert_awaited_once()


# ---- instance delete ----


def _service_instance(**overrides):
    fields = dict(
        id=21,
        cache_service_id=9,
        worker_id=5,
        cluster_id=1,
        state=CacheServiceStateEnum.RUNNING,
        update=AsyncMock(),
        delete=AsyncMock(),
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _patch_service_instances(monkeypatch, instances):
    monkeypatch.setattr(
        cache_services_route.CacheServiceInstance,
        "all_by_fields",
        AsyncMock(return_value=instances),
    )


@pytest.mark.asyncio
async def test_delete_instance_leaves_siblings_untouched(monkeypatch):
    service = _existing_service()
    target = _service_instance(id=21, state=CacheServiceStateEnum.ERROR)
    sibling = _service_instance(id=22, worker_id=6)
    monkeypatch.setattr(
        cache_services_route.CacheService, "one_by_id", AsyncMock(return_value=service)
    )
    monkeypatch.setattr(
        cache_services_route.CacheServiceInstance,
        "one_by_id",
        AsyncMock(return_value=target),
    )

    await cache_services_route.delete_cache_service_instance(
        session=MagicMock(), ctx=_user_ctx(), id=9, instance_id=21
    )

    target.delete.assert_awaited_once()
    sibling.delete.assert_not_called()
    service.update.assert_not_called()


@pytest.mark.asyncio
async def test_delete_instance_missing_service_is_not_found(monkeypatch):
    monkeypatch.setattr(
        cache_services_route.CacheService, "one_by_id", AsyncMock(return_value=None)
    )

    with pytest.raises(NotFoundException):
        await cache_services_route.delete_cache_service_instance(
            session=MagicMock(), ctx=_user_ctx(), id=9, instance_id=21
        )


@pytest.mark.asyncio
async def test_delete_instance_hidden_cross_tenant(monkeypatch):
    service = _existing_service(owner_principal_id=999)
    monkeypatch.setattr(
        cache_services_route.CacheService, "one_by_id", AsyncMock(return_value=service)
    )

    with pytest.raises(NotFoundException):
        await cache_services_route.delete_cache_service_instance(
            session=MagicMock(), ctx=_user_ctx(), id=9, instance_id=21
        )


@pytest.mark.asyncio
async def test_delete_instance_of_other_service_is_not_found(monkeypatch):
    service = _existing_service()
    foreign = _service_instance(id=21, cache_service_id=8)
    monkeypatch.setattr(
        cache_services_route.CacheService, "one_by_id", AsyncMock(return_value=service)
    )
    monkeypatch.setattr(
        cache_services_route.CacheServiceInstance,
        "one_by_id",
        AsyncMock(return_value=foreign),
    )

    with pytest.raises(NotFoundException):
        await cache_services_route.delete_cache_service_instance(
            session=MagicMock(), ctx=_user_ctx(), id=9, instance_id=21
        )
    foreign.delete.assert_not_called()


# ---- logs ----


def _patch_logs_session(monkeypatch):
    @asynccontextmanager
    async def fake_session():
        yield MagicMock()

    monkeypatch.setattr(cache_services_route, "async_session", fake_session)


def _logs_request() -> SimpleNamespace:
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                http_client=MagicMock(), http_client_no_proxy=MagicMock()
            )
        )
    )


@pytest.mark.asyncio
async def test_logs_rejects_service_without_instances(monkeypatch):
    _patch_logs_session(monkeypatch)
    service = _existing_service()
    monkeypatch.setattr(
        cache_services_route.CacheService, "one_by_id", AsyncMock(return_value=service)
    )
    _patch_service_instances(monkeypatch, [])

    with pytest.raises(BadRequestException) as exc_info:
        await cache_services_route.get_cache_service_logs(
            request=_logs_request(), ctx=_user_ctx(), id=9, log_options=LogOptions()
        )

    assert "no instances" in exc_info.value.message


@pytest.mark.asyncio
async def test_logs_rejects_multi_instance_service(monkeypatch):
    """A multi-instance service has no single log stream; callers must
    address one instance."""
    _patch_logs_session(monkeypatch)
    service = _existing_service()
    monkeypatch.setattr(
        cache_services_route.CacheService, "one_by_id", AsyncMock(return_value=service)
    )
    _patch_service_instances(
        monkeypatch,
        [_service_instance(id=21), _service_instance(id=22, worker_id=6)],
    )

    with pytest.raises(BadRequestException) as exc_info:
        await cache_services_route.get_cache_service_logs(
            request=_logs_request(), ctx=_user_ctx(), id=9, log_options=LogOptions()
        )

    assert "instances/{instance_id}/logs" in exc_info.value.message


@pytest.mark.asyncio
async def test_logs_proxies_single_instance_to_its_worker(monkeypatch):
    _patch_logs_session(monkeypatch)
    service = _existing_service()
    instance = _service_instance(id=21, worker_id=5)
    worker = SimpleNamespace(id=5, deleted_at=None)
    monkeypatch.setattr(
        cache_services_route.CacheService, "one_by_id", AsyncMock(return_value=service)
    )
    _patch_service_instances(monkeypatch, [instance])
    monkeypatch.setattr(
        cache_services_route.Worker, "one_by_id", AsyncMock(return_value=worker)
    )
    request_to_worker = AsyncMock(
        return_value=(SimpleNamespace(status=200), b"cache server log line\n")
    )
    monkeypatch.setattr(cache_services_route, "request_to_worker", request_to_worker)

    response = await cache_services_route.get_cache_service_logs(
        request=_logs_request(),
        ctx=_user_ctx(),
        id=9,
        log_options=LogOptions(tail=100, follow=False),
    )

    request_to_worker.assert_awaited_once()
    call_kwargs = request_to_worker.await_args.kwargs
    assert call_kwargs["worker"] is worker
    assert call_kwargs["path"] == "cacheServiceInstanceLogs/21"
    assert call_kwargs["params"] == {
        "tail": 100,
        "follow": False,
        "cache_service_id": 9,
    }
    assert response.status_code == 200
    assert response.body == b"cache server log line\n"


@pytest.mark.asyncio
async def test_instance_logs_proxy_to_instance_worker(monkeypatch):
    _patch_logs_session(monkeypatch)
    service = _existing_service()
    instance = _service_instance(id=22, worker_id=6)
    worker = SimpleNamespace(id=6, deleted_at=None)
    monkeypatch.setattr(
        cache_services_route.CacheService, "one_by_id", AsyncMock(return_value=service)
    )
    monkeypatch.setattr(
        cache_services_route.CacheServiceInstance,
        "one_by_id",
        AsyncMock(return_value=instance),
    )
    monkeypatch.setattr(
        cache_services_route.Worker, "one_by_id", AsyncMock(return_value=worker)
    )
    request_to_worker = AsyncMock(
        return_value=(SimpleNamespace(status=200), b"instance log line\n")
    )
    monkeypatch.setattr(cache_services_route, "request_to_worker", request_to_worker)

    response = await cache_services_route.get_cache_service_instance_logs(
        request=_logs_request(),
        ctx=_user_ctx(),
        id=9,
        instance_id=22,
        log_options=LogOptions(tail=50, follow=False),
    )

    call_kwargs = request_to_worker.await_args.kwargs
    assert call_kwargs["worker"] is worker
    assert call_kwargs["path"] == "cacheServiceInstanceLogs/22"
    assert call_kwargs["params"] == {
        "tail": 50,
        "follow": False,
        "cache_service_id": 9,
    }
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_instance_logs_of_other_service_is_not_found(monkeypatch):
    _patch_logs_session(monkeypatch)
    service = _existing_service()
    foreign = _service_instance(id=22, cache_service_id=8)
    monkeypatch.setattr(
        cache_services_route.CacheService, "one_by_id", AsyncMock(return_value=service)
    )
    monkeypatch.setattr(
        cache_services_route.CacheServiceInstance,
        "one_by_id",
        AsyncMock(return_value=foreign),
    )

    with pytest.raises(NotFoundException):
        await cache_services_route.get_cache_service_instance_logs(
            request=_logs_request(),
            ctx=_user_ctx(),
            id=9,
            instance_id=22,
            log_options=LogOptions(),
        )


# ---- instances listing ----


@pytest.mark.asyncio
async def test_service_instances_listed_ordered_by_worker(monkeypatch):
    service = _existing_service()
    monkeypatch.setattr(
        cache_services_route.CacheService, "one_by_id", AsyncMock(return_value=service)
    )
    from datetime import datetime

    now = datetime(2026, 7, 1)
    rows = [
        dict(
            id=22,
            name="svc-f6g7h",
            cache_service_id=9,
            worker_id=6,
            cluster_id=1,
            state=CacheServiceStateEnum.RUNNING,
            created_at=now,
            updated_at=now,
        ),
        dict(
            id=21,
            name="svc-a1b2c",
            cache_service_id=9,
            worker_id=5,
            cluster_id=1,
            state=CacheServiceStateEnum.PENDING,
            created_at=now,
            updated_at=now,
        ),
    ]
    _patch_service_instances(monkeypatch, [SimpleNamespace(**row) for row in rows])

    response = await cache_services_route.get_cache_service_instances_of_service(
        session=MagicMock(), ctx=_user_ctx(), id=9
    )

    assert [item.worker_id for item in response.items] == [5, 6]
    assert [item.id for item in response.items] == [21, 22]
    assert response.pagination.total == 2


@pytest.mark.asyncio
async def test_service_instances_hidden_cross_tenant(monkeypatch):
    service = _existing_service(owner_principal_id=999)
    monkeypatch.setattr(
        cache_services_route.CacheService, "one_by_id", AsyncMock(return_value=service)
    )

    with pytest.raises(NotFoundException):
        await cache_services_route.get_cache_service_instances_of_service(
            session=MagicMock(), ctx=_user_ctx(), id=9
        )


# ---- delete protection ----


def _model(name, cache_service_id, enabled=True, mode=KVCacheModeEnum.SHARED):
    return SimpleNamespace(
        name=name,
        deleted_at=None,
        extended_kv_cache=ExtendedKVCacheConfig(
            enabled=enabled, mode=mode, cache_service_id=cache_service_id
        ),
    )


@pytest.mark.asyncio
async def test_delete_blocked_while_models_reference_service(monkeypatch):
    service = _existing_service()
    monkeypatch.setattr(
        cache_services_route.CacheService, "one_by_id", AsyncMock(return_value=service)
    )
    monkeypatch.setattr(
        cache_services_route.Model,
        "all_by_fields",
        AsyncMock(return_value=[_model("llama3", cache_service_id=9)]),
    )

    with pytest.raises(BadRequestException) as exc_info:
        await cache_services_route.delete_cache_service(
            session=MagicMock(), ctx=_user_ctx(), id=9
        )
    assert "llama3" in exc_info.value.message
    service.delete.assert_not_called()


@pytest.mark.asyncio
async def test_delete_proceeds_without_shared_references(monkeypatch):
    service = _existing_service()
    monkeypatch.setattr(
        cache_services_route.CacheService, "one_by_id", AsyncMock(return_value=service)
    )
    # Other services' attachments, local-mode configs, and soft-deleted
    # models do not pin this service.
    other = _model("other", cache_service_id=8)
    local = _model("local", cache_service_id=None, mode=KVCacheModeEnum.LOCAL)
    removed = _model("removed", cache_service_id=9)
    removed.deleted_at = object()
    monkeypatch.setattr(
        cache_services_route.Model,
        "all_by_fields",
        AsyncMock(return_value=[other, local, removed]),
    )

    await cache_services_route.delete_cache_service(
        session=MagicMock(), ctx=_user_ctx(), id=9
    )
    service.delete.assert_awaited_once()


# ---- test-connection ----


def _referencing_model(id, name, cache_service_id, **overrides):
    fields = dict(
        id=id,
        name=name,
        replicas=2,
        ready_replicas=1,
        backend="vLLM",
        deleted_at=None,
        extended_kv_cache=ExtendedKVCacheConfig(
            enabled=True,
            mode=KVCacheModeEnum.SHARED,
            cache_service_id=cache_service_id,
        ),
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.mark.asyncio
async def test_models_lists_referencing_deployments_only(monkeypatch):
    service = _existing_service()
    monkeypatch.setattr(
        cache_services_route.CacheService, "one_by_id", AsyncMock(return_value=service)
    )
    referencing = _referencing_model(1, "llama3", cache_service_id=9)
    other_service = _referencing_model(2, "qwen3", cache_service_id=8)
    local = _referencing_model(
        3,
        "local-cache",
        cache_service_id=None,
        extended_kv_cache=ExtendedKVCacheConfig(
            enabled=True, mode=KVCacheModeEnum.LOCAL
        ),
    )
    removed = _referencing_model(4, "removed", cache_service_id=9, deleted_at=object())
    monkeypatch.setattr(
        cache_services_route.Model,
        "all_by_fields",
        AsyncMock(return_value=[referencing, other_service, local, removed]),
    )

    response = await cache_services_route.get_cache_service_models(
        session=MagicMock(), ctx=_user_ctx(), id=9
    )

    assert [(item.id, item.name) for item in response.items] == [(1, "llama3")]
    item = response.items[0]
    assert item.replicas == 2
    assert item.ready_replicas == 1
    assert item.backend == "vLLM"


# ---- visibility scoping ----


@pytest.mark.asyncio
async def test_models_listing_hides_cross_tenant_services(monkeypatch):
    service = _existing_service(owner_principal_id=999)
    monkeypatch.setattr(
        cache_services_route.CacheService, "one_by_id", AsyncMock(return_value=service)
    )

    with pytest.raises(NotFoundException):
        await cache_services_route.get_cache_service_models(
            session=MagicMock(), ctx=_user_ctx(), id=9
        )


# ---- cross-cluster create guard ----


@pytest.mark.asyncio
async def test_create_denies_cross_tenant_cluster(monkeypatch):
    victim_cluster = SimpleNamespace(id=1, deleted_at=None, owner_principal_id=999)
    monkeypatch.setattr(
        cache_services_route.Cluster,
        "one_by_id",
        AsyncMock(return_value=victim_cluster),
    )

    # A cluster the caller can't see is reported as missing (404).
    with pytest.raises(NotFoundException):
        await cache_services_route.create_cache_service(
            session=MagicMock(),
            ctx=_user_ctx(),
            cache_service_in=_managed_create(),
        )


# ---- dashboard ----


def _grafana_cfg(**overrides):
    fields = dict(
        grafana_url=None,
        server_external_url="http://server.example.com",
        grafana_cache_service_dashboard_uid="gpustack-cache-service",
        get_grafana_url=lambda: "http://127.0.0.1:13000",
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.mark.asyncio
async def test_dashboard_redirects_to_grafana(monkeypatch):
    service = _existing_service()
    monkeypatch.setattr(
        cache_services_route.CacheService, "one_by_id", AsyncMock(return_value=service)
    )
    monkeypatch.setattr(
        cache_services_route.Cluster,
        "one_by_id",
        AsyncMock(return_value=SimpleNamespace(name="default")),
    )
    monkeypatch.setattr(
        cache_services_route, "get_global_config", lambda: _grafana_cfg()
    )

    response = await cache_services_route.get_cache_service_dashboard(
        session=MagicMock(), ctx=_user_ctx(), id=9, request=MagicMock()
    )

    assert response.status_code == 302
    assert response.headers["location"] == (
        "http://server.example.com/grafana/d/gpustack-cache-service"
        "/gpustack-cache-service"
        "?var-cluster_name=default&var-cache_service_name=svc"
    )


@pytest.mark.asyncio
async def test_dashboard_requires_grafana_configuration(monkeypatch):
    service = _existing_service()
    monkeypatch.setattr(
        cache_services_route.CacheService, "one_by_id", AsyncMock(return_value=service)
    )
    monkeypatch.setattr(
        cache_services_route,
        "get_global_config",
        lambda: _grafana_cfg(get_grafana_url=lambda: None),
    )

    with pytest.raises(InternalServerErrorException):
        await cache_services_route.get_cache_service_dashboard(
            session=MagicMock(), ctx=_user_ctx(), id=9, request=MagicMock()
        )


@pytest.mark.asyncio
async def test_dashboard_invisible_service_is_not_found(monkeypatch):
    service = _existing_service(owner_principal_id=999)
    monkeypatch.setattr(
        cache_services_route.CacheService, "one_by_id", AsyncMock(return_value=service)
    )

    with pytest.raises(NotFoundException):
        await cache_services_route.get_cache_service_dashboard(
            session=MagicMock(), ctx=_user_ctx(), id=9, request=MagicMock()
        )


# ---- secret redaction ----


def _secretful_service(**overrides):
    """A stored service carrying both secret kinds: an L2 resp password
    and (via overrides) external endpoint params."""
    fields = dict(
        id=9,
        name="svc",
        provider_name="LMCache",
        cluster_id=1,
        owner_principal_id=ORG_PRINCIPAL,
        deleted_at=None,
        created_at=datetime(2026, 8, 17, tzinfo=timezone.utc),
        updated_at=datetime(2026, 8, 17, tzinfo=timezone.utc),
        state=CacheServiceStateEnum.RUNNING,
        config=_l2_config(
            "resp", host="10.0.0.2", port=6379, username="kv", password="hunter2"
        ),
        update=AsyncMock(),
    )
    fields.update(overrides)
    return SimpleNamespace(
        **fields,
        # mirrors pydantic's recursive model_dump so the detached-copy
        # guarantee of the redaction path is actually exercised
        model_dump=lambda **kwargs: {
            k: (v.model_dump() if hasattr(v, "model_dump") else v)
            for k, v in fields.items()
            if k not in ("update", "delete")
        },
    )


@pytest.mark.asyncio
async def test_get_redacts_password_fields_for_users(monkeypatch):
    _patch_provider(monkeypatch, _provider_with_l2())
    service = _secretful_service()
    monkeypatch.setattr(
        cache_services_route.CacheService, "one_by_id", AsyncMock(return_value=service)
    )

    public = await cache_services_route.get_cache_service(
        session=MagicMock(), ctx=_user_ctx(), id=9
    )

    params = public.config.l2_storages[0].params
    assert params["password"] == cache_services_route.SECRET_PLACEHOLDER
    # non-secret fields survive verbatim, and the stored row is untouched
    assert params["username"] == "kv"
    assert service.config.l2_storages[0].params["password"] == "hunter2"


@pytest.mark.asyncio
async def test_get_returns_raw_secrets_to_system_callers(monkeypatch):
    """Workers render the real credentials into the cache server's env;
    redacting their read path would break L2 backends outright."""
    _patch_provider(monkeypatch, _provider_with_l2())
    service = _secretful_service()
    monkeypatch.setattr(
        cache_services_route.CacheService, "one_by_id", AsyncMock(return_value=service)
    )

    result = await cache_services_route.get_cache_service(
        session=MagicMock(), ctx=_system_ctx(), id=9
    )

    assert result.config.l2_storages[0].params["password"] == "hunter2"


@pytest.mark.asyncio
async def test_update_placeholder_restores_stored_secret(monkeypatch):
    """An untouched edit round-trips the redacted GET: the placeholder
    is swapped back for the stored secret before the write."""
    _patch_provider(monkeypatch, _provider_with_l2())
    _patch_worker_lookup(monkeypatch)
    service = _secretful_service()
    monkeypatch.setattr(
        cache_services_route.CacheService, "one_by_id", AsyncMock(return_value=service)
    )

    update_in = _update_in(
        config=_l2_config(
            "resp",
            host="10.0.0.2",
            port=6379,
            username="kv",
            password=cache_services_route.SECRET_PLACEHOLDER,
        )
    )
    await cache_services_route.update_cache_service(
        session=MagicMock(), ctx=_user_ctx(), id=9, cache_service_in=update_in
    )

    service.update.assert_awaited_once()
    written = service.update.await_args[0][1]
    assert written.config.l2_storages[0].params["password"] == "hunter2"


@pytest.mark.asyncio
async def test_update_new_secret_value_wins_and_placeholder_without_stored_drops(
    monkeypatch,
):
    _patch_provider(monkeypatch, _provider_with_l2())
    _patch_worker_lookup(monkeypatch)
    service = _secretful_service()
    monkeypatch.setattr(
        cache_services_route.CacheService, "one_by_id", AsyncMock(return_value=service)
    )

    # a real new value passes through; a placeholder on a second, freshly
    # added entry has no stored counterpart and is dropped
    update_in = _update_in(
        config=_l2_cascade_config(
            CacheServiceL2Storage(
                backend="resp",
                params=dict(host="10.0.0.2", port=6379, password="rotated"),
            ),
            CacheServiceL2Storage(
                backend="resp",
                params=dict(
                    host="10.0.0.3",
                    port=6379,
                    password=cache_services_route.SECRET_PLACEHOLDER,
                ),
            ),
        )
    )
    await cache_services_route.update_cache_service(
        session=MagicMock(), ctx=_user_ctx(), id=9, cache_service_in=update_in
    )

    written = service.update.await_args[0][1]
    first, second = written.config.l2_storages
    assert first.params["password"] == "rotated"
    assert "password" not in second.params


@pytest.mark.asyncio
async def test_create_rejects_placeholder_secret(monkeypatch):
    _patch_provider(monkeypatch, _provider_with_l2())
    _patch_create_prereqs(
        monkeypatch, worker=SimpleNamespace(id=5, cluster_id=1, deleted_at=None)
    )

    create_in = _managed_create(
        config=_l2_config(
            "resp",
            host="10.0.0.2",
            port=6379,
            password=cache_services_route.SECRET_PLACEHOLDER,
        )
    )
    with pytest.raises(BadRequestException) as exc_info:
        await cache_services_route.create_cache_service(
            session=MagicMock(), ctx=_user_ctx(), cache_service_in=create_in
        )
    assert "placeholder" in exc_info.value.message


# ---- update runs the create validation set ----


@pytest.mark.asyncio
async def test_update_rejects_unknown_provider_version(monkeypatch):
    """Without this, a bad version only surfaces as a worker-side start
    failure parking the instance in ERROR; it should be a 400."""
    service = _existing_service()
    monkeypatch.setattr(
        cache_services_route.CacheService, "one_by_id", AsyncMock(return_value=service)
    )
    _patch_provider(monkeypatch, _provider())
    _patch_worker_lookup(monkeypatch)

    with pytest.raises(BadRequestException):
        await cache_services_route.update_cache_service(
            session=MagicMock(),
            ctx=_user_ctx(),
            id=9,
            cache_service_in=_update_in(provider_version="v999"),
        )
    service.update.assert_not_called()


@pytest.mark.asyncio
async def test_update_rejects_cross_cluster_worker(monkeypatch):
    service = _existing_service()
    monkeypatch.setattr(
        cache_services_route.CacheService, "one_by_id", AsyncMock(return_value=service)
    )
    _patch_provider(monkeypatch, _provider())
    _patch_worker_lookup(monkeypatch, cluster_id=2)

    with pytest.raises(BadRequestException) as exc_info:
        await cache_services_route.update_cache_service(
            session=MagicMock(), ctx=_user_ctx(), id=9, cache_service_in=_update_in()
        )
    assert "does not belong" in exc_info.value.message
    service.update.assert_not_called()


@pytest.mark.asyncio
async def test_update_requires_ram_size(monkeypatch):
    """A cleared capacity would silently fall through to an engine-internal
    default the platform can't see."""
    service = _existing_service()
    monkeypatch.setattr(
        cache_services_route.CacheService, "one_by_id", AsyncMock(return_value=service)
    )
    _patch_provider(monkeypatch, _provider())
    _patch_worker_lookup(monkeypatch)

    with pytest.raises(BadRequestException) as exc_info:
        await cache_services_route.update_cache_service(
            session=MagicMock(),
            ctx=_user_ctx(),
            id=9,
            cache_service_in=_update_in(
                config=CacheServiceConfig(fields={"ram_size": None})
            ),
        )
    assert "ram_size" in exc_info.value.message
    service.update.assert_not_called()


# ---- fields value validation ----


def _fielded_provider():
    from gpustack.schemas.cache_providers import CacheProviderField

    provider = _provider()
    provider.fields = provider.fields + [
        CacheProviderField(
            name="eviction_policy",
            default="LRU",
            options=["LRU", "IsolatedLRU", "noop"],
        ),
        CacheProviderField(
            name="eviction_ratio", type="number", default=0.2, min=0, max=1
        ),
    ]
    return provider


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fields,fragment",
    [
        ({"no_such_knob": 1}, "not declared"),
        ({"eviction_ratio": "abc"}, "must be a number"),
        ({"eviction_ratio": 1.5}, "must be between"),
        ({"eviction_policy": "LRUU"}, "must be one of"),
    ],
)
async def test_create_rejects_invalid_declared_field_values(
    monkeypatch, fields, fragment
):
    """A bad field value would render into the server command and
    crash-loop the container through its restart budget; an unknown name
    would be silently ignored by the worker."""
    _patch_provider(monkeypatch, _fielded_provider())
    _patch_create_prereqs(
        monkeypatch, worker=SimpleNamespace(id=5, cluster_id=1, deleted_at=None)
    )

    create_in = _managed_create(
        config=CacheServiceConfig(fields={"ram_size": 20, **fields})
    )
    with pytest.raises(BadRequestException) as exc_info:
        await cache_services_route.create_cache_service(
            session=MagicMock(), ctx=_user_ctx(), cache_service_in=create_in
        )
    assert fragment in exc_info.value.message


# ---- template-breaking characters in external params ----


def test_a_password_typed_declared_field_is_redacted(monkeypatch):
    """A declaration types a value as a password wherever it declares it,
    not only inside an L2 backend — a provider a plugin registers brings
    fields core's catalog never saw. Reading one back in the clear is the
    same leak either way."""
    provider = CacheProvider(
        name="LMCache",
        default_image="repo/cache:{{version}}",
        versions={"v1.0": {}},
        fields=[{"name": "token", "type": "password"}],
    )
    _patch_provider(monkeypatch, provider)

    service = SimpleNamespace(
        id=9,
        name="svc",
        provider_name="LMCache",
        provider_version="v1.0",
        cluster_id=1,
        worker_id=None,
        config=CacheServiceConfig(fields={"token": "s3cret"}),
        state=CacheServiceStateEnum.RUNNING,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    public = cache_services_route._redacted_for_user(service)

    assert public.config.fields["token"] == cache_services_route.SECRET_PLACEHOLDER
    # the row itself keeps the secret: redaction is for the reader
    assert service.config.fields["token"] == "s3cret"
