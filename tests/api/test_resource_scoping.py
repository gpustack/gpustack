"""Tenant resource-scoping checks for handlers that fetch by primary key.

Each handler that loads a resource by id applies the same
visibility/ownership check its sibling primary endpoint uses. These tests
drive each one as a non-owning, non-admin caller and assert it is scoped
to the caller's tenant, plus an owner/allowed case per handler to confirm
the scoping does not over-block.
"""

import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gpustack.api.exceptions import (
    BadRequestException,
    ForbiddenException,
    NotFoundException,
    ServiceUnavailableException,
)
from gpustack.api.tenant import TenantContext
from gpustack.routes import clusters as clusters_route
from gpustack.routes import cloud_credentials as cloud_credentials_route
from gpustack.routes import gpu_instances as gpu_instances_route
from gpustack.routes import model_instances as model_instances_route
from gpustack.routes import models as models_route
from gpustack.schemas.clusters import ClusterProvider
from gpustack.schemas.principals import PrincipalType
from gpustack.server import services as services_module


OWNER_PRINCIPAL = 999
CALLER_PRINCIPAL = 7


def _user_ctx(principal_id: int = CALLER_PRINCIPAL, org_role=None) -> TenantContext:
    """A non-admin USER caller operating as ``principal_id`` with no
    granted clusters — the default cross-tenant attacker shape."""
    user = MagicMock()
    user.kind = PrincipalType.USER
    return TenantContext(
        user=user,
        is_platform_admin=False,
        current_principal_id=principal_id,
        org_role=org_role,
    )


def _async_session(session):
    """Return a callable standing in for ``async_session`` that yields
    ``session`` from an async context manager."""

    @contextlib.asynccontextmanager
    async def _cm():
        yield session

    return _cm


# ---- cluster API-server proxy ----


@pytest.mark.asyncio
async def test_apiserver_proxy_denies_cross_tenant(monkeypatch):
    victim = SimpleNamespace(
        id=2,
        name="victim",
        deleted_at=None,
        owner_principal_id=OWNER_PRINCIPAL,
        provider=ClusterProvider.Kubernetes,
    )
    monkeypatch.setattr(clusters_route, "async_session", _async_session(MagicMock()))
    monkeypatch.setattr(
        clusters_route.Cluster, "one_by_id", AsyncMock(return_value=victim)
    )
    # No reachable workers so that, if the visibility check were removed,
    # the handler would raise ServiceUnavailable — a different error than
    # the NotFound the check produces.
    monkeypatch.setattr(
        clusters_route.Worker, "all_by_fields", AsyncMock(return_value=[])
    )

    with pytest.raises(NotFoundException):
        await clusters_route.cluster_apiserver_proxy(
            request=MagicMock(),
            ctx=_user_ctx(),
            id=2,
            path="api/v1/namespaces/kube-system/secrets",
        )


@pytest.mark.asyncio
async def test_apiserver_proxy_allows_owner(monkeypatch):
    owned = SimpleNamespace(
        id=2,
        name="mine",
        deleted_at=None,
        owner_principal_id=CALLER_PRINCIPAL,
        provider=ClusterProvider.Kubernetes,
    )
    monkeypatch.setattr(clusters_route, "async_session", _async_session(MagicMock()))
    monkeypatch.setattr(
        clusters_route.Cluster, "one_by_id", AsyncMock(return_value=owned)
    )
    monkeypatch.setattr(
        clusters_route.Worker, "all_by_fields", AsyncMock(return_value=[])
    )

    # Passes visibility and fails later on "no workers" — proving the owner
    # is not blocked by the tenant gate.
    with pytest.raises(ServiceUnavailableException):
        await clusters_route.cluster_apiserver_proxy(
            request=MagicMock(), ctx=_user_ctx(), id=2, path="version"
        )


# ---- cluster manifests ----


@pytest.mark.asyncio
async def test_manifests_denies_cross_tenant(monkeypatch):
    victim = SimpleNamespace(
        id=2,
        name="victim",
        deleted_at=None,
        owner_principal_id=OWNER_PRINCIPAL,
        provider=ClusterProvider.Kubernetes,
    )
    monkeypatch.setattr(
        clusters_route.Cluster, "one_by_id", AsyncMock(return_value=victim)
    )

    # The token is write-class, so a non-owner is a 403 (assert_cluster_writable).
    with pytest.raises(ForbiddenException):
        await clusters_route.get_cluster_manifests(
            request=MagicMock(),
            session=MagicMock(),
            ctx=_user_ctx(),
            id=2,
        )


# ---- GPU instance create ----


@pytest.mark.asyncio
async def test_gpu_instance_create_denies_cross_tenant_cluster(monkeypatch):
    victim_cluster = SimpleNamespace(
        id=2,
        name="victim",
        deleted_at=None,
        owner_principal_id=OWNER_PRINCIPAL,
        provider=ClusterProvider.Kubernetes,
    )
    monkeypatch.setattr(
        models_route.Cluster, "one_by_id", AsyncMock(return_value=victim_cluster)
    )
    # owner_principal_id matches the caller so owner validation passes and
    # execution reaches the cluster-ownership check.
    create_obj = SimpleNamespace(
        owner_principal_id=CALLER_PRINCIPAL, cluster_id=2, name="x"
    )

    with pytest.raises(NotFoundException):
        await gpu_instances_route.create_gpu_instance(
            session=MagicMock(), ctx=_user_ctx(), create_obj=create_obj
        )


# ---- cloud-credential provider proxy ----


@pytest.mark.asyncio
async def test_provider_proxy_rejects_host_override(monkeypatch):
    credential = SimpleNamespace(
        id=1,
        deleted_at=None,
        owner_principal_id=CALLER_PRINCIPAL,
        provider=ClusterProvider.DigitalOcean,
        key="ak",
        secret="sk",
        options={},
    )
    monkeypatch.setattr(
        cloud_credentials_route.CloudCredential,
        "one_by_id",
        AsyncMock(return_value=credential),
    )
    # Decouple from the real provider registry: stand in a provider whose
    # endpoint host is fixed. factory is a dict, so replace the module
    # attribute rather than setting an attribute on the dict.
    mock_provider = MagicMock()
    mock_provider.get_api_endpoint.return_value = "https://api.digitalocean.com/"
    monkeypatch.setattr(
        cloud_credentials_route,
        "factory",
        MagicMock(get=MagicMock(return_value=[mock_provider])),
    )
    # If the host check were removed, proxy_to would run and return 200 —
    # so a raised BadRequest proves the guard fired.
    monkeypatch.setattr(
        cloud_credentials_route,
        "proxy_to",
        AsyncMock(return_value=SimpleNamespace(status_code=200, headers=MagicMock())),
    )

    request = MagicMock()
    request.query_params = {}
    with pytest.raises(BadRequestException):
        await cloud_credentials_route.proxy_cluster_provider_api(
            request=request,
            session=MagicMock(),
            ctx=_user_ctx(),
            id=1,
            path="//evil.example.com/steal",
        )


@pytest.mark.asyncio
async def test_provider_proxy_allows_provider_path(monkeypatch):
    credential = SimpleNamespace(
        id=1,
        deleted_at=None,
        owner_principal_id=CALLER_PRINCIPAL,
        provider=ClusterProvider.DigitalOcean,
        key="ak",
        secret="sk",
        options={},
    )
    monkeypatch.setattr(
        cloud_credentials_route.CloudCredential,
        "one_by_id",
        AsyncMock(return_value=credential),
    )
    # Decouple from the real provider registry: stand in a provider whose
    # endpoint host is fixed. factory is a dict, so replace the module
    # attribute rather than setting an attribute on the dict.
    mock_provider = MagicMock()
    mock_provider.get_api_endpoint.return_value = "https://api.digitalocean.com/"
    monkeypatch.setattr(
        cloud_credentials_route,
        "factory",
        MagicMock(get=MagicMock(return_value=[mock_provider])),
    )
    sent = SimpleNamespace(status_code=200, headers=MagicMock())
    monkeypatch.setattr(
        cloud_credentials_route, "proxy_to", AsyncMock(return_value=sent)
    )

    request = MagicMock()
    request.query_params = {}
    result = await cloud_credentials_route.proxy_cluster_provider_api(
        request=request,
        session=MagicMock(),
        ctx=_user_ctx(),
        id=1,
        path="v2/account",
    )
    assert result is sent


# ---- model instance serving logs ----


@pytest.mark.asyncio
async def test_serving_logs_denies_cross_tenant(monkeypatch):
    victim = SimpleNamespace(
        id=3,
        name="victim-instance",
        owner_principal_id=OWNER_PRINCIPAL,
        worker_id=1,
        cluster_id=5,
    )
    monkeypatch.setattr(
        model_instances_route, "async_session", _async_session(MagicMock())
    )
    monkeypatch.setattr(
        model_instances_route.ModelInstance,
        "one_by_id_with_model_files",
        AsyncMock(return_value=victim),
    )
    # A present worker means removing the visibility check would fall through
    # to streaming rather than re-raising NotFound for an unrelated reason.
    monkeypatch.setattr(
        model_instances_route.Worker,
        "one_by_id",
        AsyncMock(return_value=SimpleNamespace(id=1)),
    )

    with pytest.raises(NotFoundException):
        await model_instances_route.get_serving_logs(
            request=MagicMock(),
            ctx=_user_ctx(),
            id=3,
            log_options=MagicMock(),
        )


# ---- unprefixed model-name route resolution ----


@pytest.mark.asyncio
async def test_raw_name_resolution_scopes_to_platform(monkeypatch):
    platform_route_id = 10
    victim_route_id = 20
    targets = [
        SimpleNamespace(
            route_id=platform_route_id,
            model_id=111,
            overridden_model_name=None,
            weight=1,
        ),
        SimpleNamespace(
            route_id=victim_route_id,
            model_id=222,
            overridden_model_name=None,
            weight=1,
        ),
    ]
    monkeypatch.setattr(services_module, "platform_principal_id", lambda: 1)
    monkeypatch.setattr(
        services_module.ModelRouteTarget,
        "all_by_fields",
        AsyncMock(return_value=targets),
    )
    # Only the platform-owned route is returned by the owner-scoped lookup.
    monkeypatch.setattr(
        services_module.ModelRoute,
        "all_by_fields",
        AsyncMock(return_value=[SimpleNamespace(id=platform_route_id)]),
    )

    svc = services_module.ModelRouteService(MagicMock())
    # Unique unprefixed name to avoid the in-memory resolution cache.
    resolutions = await svc.resolve_route_targets("collide-raw-name-f6")

    model_ids = {r.model_id for r in resolutions}
    assert model_ids == {111}


@pytest.mark.asyncio
async def test_serving_logs_visibility_before_worker_check(monkeypatch):
    """A cross-tenant instance that has no worker still returns the generic
    "not found" — the worker-assignment message must not distinguish it."""
    victim = SimpleNamespace(
        id=3,
        name="victim-instance",
        owner_principal_id=OWNER_PRINCIPAL,
        worker_id=None,
        cluster_id=5,
    )
    monkeypatch.setattr(
        model_instances_route, "async_session", _async_session(MagicMock())
    )
    monkeypatch.setattr(
        model_instances_route.ModelInstance,
        "one_by_id_with_model_files",
        AsyncMock(return_value=victim),
    )

    with pytest.raises(NotFoundException) as exc:
        await model_instances_route.get_serving_logs(
            request=MagicMock(),
            ctx=_user_ctx(),
            id=3,
            log_options=MagicMock(),
        )
    assert "not assigned" not in str(exc.value)


@pytest.mark.asyncio
async def test_log_options_denies_cross_tenant(monkeypatch):
    victim = SimpleNamespace(
        id=3,
        name="victim-instance",
        owner_principal_id=OWNER_PRINCIPAL,
        worker_id=1,
        cluster_id=5,
    )
    monkeypatch.setattr(
        model_instances_route.ModelInstance,
        "one_by_id_with_model_files",
        AsyncMock(return_value=victim),
    )

    with pytest.raises(NotFoundException):
        await model_instances_route.get_model_instance_log_options(
            request=MagicMock(), session=MagicMock(), ctx=_user_ctx(), id=3
        )


@pytest.mark.asyncio
async def test_get_by_name_scopes_raw_name_to_platform(monkeypatch):
    seen = {}

    async def _one_by_fields(session, fields):
        seen.update(fields)
        return SimpleNamespace(id=10)

    monkeypatch.setattr(services_module, "platform_principal_id", lambda: 1)
    monkeypatch.setattr(services_module.ModelRoute, "one_by_fields", _one_by_fields)

    svc = services_module.ModelRouteService(MagicMock())
    await svc.get_by_name("raw-name-getbyname-f6")

    assert seen == {"name": "raw-name-getbyname-f6", "owner_principal_id": 1}
