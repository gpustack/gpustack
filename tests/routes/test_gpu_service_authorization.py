"""GPU Service management is available to users, not service accounts."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from gpustack.api.auth import get_current_user
from gpustack.api.exceptions import register_handlers
from gpustack.api.tenant import TenantContext, get_tenant_context
from gpustack.routes import (
    gpu_instance_persistent_volume_types,
    gpu_instance_persistent_volumes,
    gpu_instance_ssh_public_keys,
    gpu_instance_templates,
    gpu_instance_type_flavors,
    gpu_instance_types,
    gpu_instances,
)
from gpustack.routes.routes import management_router
from gpustack.schemas.clusters import ClusterProvider, GpuInstanceOptions, K8sOptions
from gpustack.schemas.principals import OrgRole, PrincipalType
from gpustack.server.db import get_session

GPU_ROUTERS = [
    (gpu_instances, "/gpu-instances", gpu_instances.GPUInstance),
    (
        gpu_instance_persistent_volumes,
        "/gpu-instance-persistent-volumes",
        gpu_instance_persistent_volumes.GPUInstancePersistentVolume,
    ),
    (
        gpu_instance_persistent_volume_types,
        "/gpu-instance-persistent-volume-types",
        gpu_instance_persistent_volume_types.GPUInstancePersistentVolumeType,
    ),
    (
        gpu_instance_ssh_public_keys,
        "/gpu-instance-ssh-public-keys",
        gpu_instance_ssh_public_keys.GPUInstanceSSHPublicKey,
    ),
    (
        gpu_instance_templates,
        "/gpu-instance-templates",
        gpu_instance_templates.GPUInstanceTemplate,
    ),
    (gpu_instance_types, "/gpu-instance-types", None),
    (gpu_instance_type_flavors, "/gpu-instance-type-flavors", None),
]
GPU_REQUESTS = [
    (method, "/v2" + prefix + re.sub(r"\{[^}]+\}", "2", route.path))
    for module, prefix, _ in GPU_ROUTERS
    for route in module.router.routes
    for method in sorted(route.methods)
]
GPU_REQUESTS += [
    (method, path + "?watch=true") for method, path in GPU_REQUESTS if method == "GET"
]


def _context(kind):
    system = kind in ("cluster", "worker", "unlinked", "server")
    return TenantContext(
        user=SimpleNamespace(
            id=None if kind == "server" else 7,
            name="system/worker/test" if system else "user",
            kind=PrincipalType.SYSTEM if system else PrincipalType.USER,
            is_admin=kind == "admin",
        ),
        is_platform_admin=kind == "admin",
        current_principal_id=(
            None if system or kind == "admin" else 7 if kind == "personal" else 20
        ),
        org_role=(
            OrgRole.OWNER
            if kind == "owner"
            else OrgRole.MEMBER if kind == "member" else None
        ),
        current_is_personal_scope=kind == "personal",
        accessible_cluster_ids={1},
        accessible_cluster_owner_ids={10},
        scoped_cluster_id=1 if kind in ("cluster", "worker") else None,
        scoped_cluster_owner_id=10 if kind in ("cluster", "worker") else None,
    )


@pytest.fixture(scope="module")
def gpu_app():
    app = FastAPI()
    register_handlers(app)
    app.include_router(management_router)
    return app


@pytest.fixture
def gpu_api(gpu_app, monkeypatch):
    session = AsyncMock()
    lookup = AsyncMock(return_value=None)
    query = AsyncMock(side_effect=AssertionError("Unexpected resource query"))

    @asynccontextmanager
    async def open_session():
        yield session

    for module, _, model in GPU_ROUTERS:
        monkeypatch.setattr(module, "async_session", open_session)
        if model is not None:
            monkeypatch.setattr(model, "one_by_id", lookup)
            monkeypatch.setattr(model, "paginated_by_query", query)
    monkeypatch.setattr(gpu_instances.Cluster, "one_by_id", lookup)
    monkeypatch.setattr(gpu_instances.Cluster, "all_by_fields", query)
    gpu_app.dependency_overrides[get_session] = lambda: session
    yield SimpleNamespace(app=gpu_app, lookup=lookup, query=query, session=session)
    gpu_app.dependency_overrides.clear()


def _authenticate(api, kind):
    ctx = _context(kind)
    api.app.dependency_overrides[get_current_user] = lambda: ctx.user
    api.app.dependency_overrides[get_tenant_context] = lambda: ctx
    return ctx


@pytest.mark.parametrize("kind", ["cluster", "worker", "unlinked"])
@pytest.mark.parametrize("method,path", GPU_REQUESTS)
def test_service_accounts_cannot_reach_gpu_service(gpu_api, kind, method, path):
    _authenticate(gpu_api, kind)
    with TestClient(gpu_api.app) as client:
        response = client.request(method, path, json={})
    assert response.status_code == 403
    gpu_api.lookup.assert_not_awaited()
    gpu_api.query.assert_not_awaited()
    gpu_api.session.exec.assert_not_awaited()
    gpu_api.session.commit.assert_not_awaited()


@pytest.mark.parametrize("kind", ["owner", "member", "personal", "admin", "server"])
@pytest.mark.parametrize(
    "prefix", [prefix for _, prefix, model in GPU_ROUTERS if model]
)
def test_user_roles_and_platform_token_reach_resource_checks(gpu_api, kind, prefix):
    _authenticate(gpu_api, kind)
    with TestClient(gpu_api.app) as client:
        response = client.get(f"/v2{prefix}/2")
    assert response.status_code == 404
    gpu_api.lookup.assert_awaited_once()


@pytest.mark.parametrize("kind", ["owner", "member", "personal"])
@pytest.mark.parametrize("shared", [True, False])
def test_users_deploy_to_granted_clusters_in_their_own_scope(
    gpu_api, monkeypatch, kind, shared
):
    ctx = _authenticate(gpu_api, kind)
    if not shared:
        ctx.accessible_cluster_ids.clear()
    gpu_api.lookup.return_value = SimpleNamespace(
        id=1,
        owner_principal_id=10,
        deleted_at=None,
        provider=ClusterProvider.Kubernetes,
        k8s_options=K8sOptions(gpu_instance_options=GpuInstanceOptions()),
    )
    monkeypatch.setattr(
        gpu_instances.GPUInstance, "exist_by_fields", AsyncMock(return_value=False)
    )
    monkeypatch.setattr(
        gpu_instances, "_resolve_type_snapshot", AsyncMock(return_value="snapshot")
    )
    monkeypatch.setattr(gpu_instances, "_types_by_snapshot", AsyncMock(return_value={}))

    async def create_instance(*, session, source):
        return gpu_instances.GPUInstance(
            id=2,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            **source,
        )

    create = AsyncMock(side_effect=create_instance)
    monkeypatch.setattr(gpu_instances.GPUInstance, "create", create)
    with TestClient(gpu_api.app) as client:
        response = client.post(
            "/v2/gpu-instances",
            json={
                "name": "shared-instance",
                "cluster_id": 1,
                "spec": {
                    "type": "gpu",
                    "image": "busybox",
                    "volume": {"ephemeral": {}},
                },
            },
        )
    if shared:
        assert response.status_code == 200, response.text
        create.assert_awaited_once()
        assert (
            create.call_args.kwargs["source"]["owner_principal_id"]
            == ctx.current_principal_id
        )
        assert response.json()["owner_principal_id"] == ctx.current_principal_id
    else:
        assert response.status_code == 404
        create.assert_not_awaited()
