"""Service credentials cannot manage another cluster or org-wide secrets."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from gpustack.api.auth import get_current_user
from gpustack.api.exceptions import ForbiddenException, register_handlers
from gpustack.api.tenant import TenantContext, get_tenant_context
from gpustack.routes import cloud_credentials, clusters, inference_backend
from gpustack.routes.routes import management_router
from gpustack.schemas.clusters import CloudCredential, Cluster, WorkerPool
from gpustack.schemas.inference_backend import InferenceBackend, InferenceBackendUpdate
from gpustack.schemas.principals import OrgRole, PrincipalType
from gpustack.server.db import get_session


def _principal(kind):
    return SimpleNamespace(
        id=None if kind == "server" else 7,
        name="system/worker/test",
        kind=PrincipalType.SYSTEM,
        is_admin=False,
        worker=(
            SimpleNamespace(cluster_id=1, owner_principal_id=10)
            if kind == "worker"
            else None
        ),
        cluster=(
            SimpleNamespace(id=1, owner_principal_id=10) if kind == "cluster" else None
        ),
    )


@pytest.fixture
def infrastructure(monkeypatch):
    app = FastAPI()
    register_handlers(app)
    app.include_router(management_router)
    session = AsyncMock()
    app.dependency_overrides[get_session] = lambda: session
    row = SimpleNamespace(
        id=2,
        cluster_id=2,
        owner_principal_id=20,
        deleted_at=None,
        name="target",
        update=AsyncMock(),
        delete=AsyncMock(),
    )
    lookup = AsyncMock(return_value=row)
    mutations = []
    for model in (Cluster, WorkerPool, CloudCredential, InferenceBackend):
        monkeypatch.setattr(model, "one_by_id", lookup)
        for method in ("create", "update", "delete"):
            mutation = AsyncMock()
            monkeypatch.setattr(model, method, mutation)
            mutations.append(mutation)
    render_token = MagicMock(
        return_value=dict(
            token="test-registration-token",
            server_url="http://server.test",
            image="test-image",
            env={},
            args=[],
        )
    )
    monkeypatch.setattr(clusters, "get_registration_from_cluster", render_token)
    proxy = AsyncMock()
    monkeypatch.setattr(cloud_credentials, "proxy_to", proxy)
    yield SimpleNamespace(
        app=app,
        row=row,
        lookup=lookup,
        session=session,
        render_token=render_token,
        proxy=proxy,
    )
    for mutation in mutations:
        mutation.assert_not_awaited()
    row.update.assert_not_awaited()
    row.delete.assert_not_awaited()
    session.commit.assert_not_awaited()
    proxy.assert_not_awaited()


@pytest.mark.parametrize("kind", ["cluster", "worker"])
@pytest.mark.parametrize("owner_id", [10, 20])
@pytest.mark.parametrize(
    "method,path,payload",
    [
        ("GET", "registration-token", None),
        ("GET", "manifests", None),
        ("PUT", "", {"name": "target"}),
        ("DELETE", "", None),
        ("POST", "set-default", None),
        (
            "POST",
            "worker-pools",
            {
                "name": "pool",
                "instance_type": "test",
                "os_image": "test",
                "image_name": "test",
            },
        ),
    ],
)
def test_system_denied_other_cluster(
    infrastructure, kind, owner_id, method, path, payload
):
    infrastructure.row.owner_principal_id = owner_id
    infrastructure.app.dependency_overrides[get_current_user] = lambda: _principal(kind)
    url = "/v2/clusters/2" + (f"/{path}" if path else "")
    with TestClient(infrastructure.app) as client:
        response = client.request(
            method, url, json=payload, headers={"X-Organization-Id": str(owner_id)}
        )
    expected = 404 if kind == "cluster" and path == "manifests" else 403
    assert response.status_code == expected
    infrastructure.render_token.assert_not_called()


@pytest.mark.parametrize("kind", ["cluster", "worker"])
@pytest.mark.parametrize("method", ["PUT", "DELETE"])
def test_system_denied_other_cluster_worker_pool(infrastructure, kind, method):
    infrastructure.app.dependency_overrides[get_current_user] = lambda: _principal(kind)
    with TestClient(infrastructure.app) as client:
        response = client.request(method, "/v2/worker-pools/2", json={"name": "target"})
    assert response.status_code == 403


@pytest.mark.parametrize("kind", ["cluster", "worker"])
@pytest.mark.parametrize("owner_id", [None, 10, 20])
@pytest.mark.parametrize(
    "method,suffix", [("PUT", ""), ("DELETE", ""), ("PUT", "/from-yaml")]
)
def test_system_cannot_modify_inference_backends(
    infrastructure, kind, owner_id, method, suffix
):
    del infrastructure.row.cluster_id
    infrastructure.row.owner_principal_id = owner_id
    infrastructure.app.dependency_overrides[get_current_user] = lambda: _principal(kind)
    with TestClient(infrastructure.app) as client:
        response = client.request(
            method,
            f"/v2/inference-backends/2{suffix}",
            json={"backend_name": "custom", "content": "backend_name: custom"},
            headers={"X-Organization-Id": "20"},
        )
    assert response.status_code == 403


@pytest.mark.parametrize("kind", ["cluster", "worker", "unlinked"])
@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", ""),
        ("GET", "?watch=true"),
        ("GET", "/2"),
        ("GET", "/2/provider-proxy/v2/account"),
        ("POST", ""),
        ("PUT", "/2"),
        ("DELETE", "/2"),
    ],
)
def test_system_cannot_access_cloud_credentials(infrastructure, kind, method, path):
    infrastructure.app.dependency_overrides[get_current_user] = lambda: _principal(kind)
    with TestClient(infrastructure.app) as client:
        response = client.request(
            method,
            f"/v2/cloud-credentials{path}",
            json={"name": "target"},
            headers={"X-Organization-Id": "20"},
        )
    assert response.status_code == 403
    infrastructure.lookup.assert_not_awaited()
    infrastructure.session.exec.assert_not_awaited()


@pytest.mark.parametrize("kind", ["cluster", "worker", "server"])
def test_registration_token_requires_a_management_identity(infrastructure, kind):
    infrastructure.row.id = 1
    infrastructure.row.owner_principal_id = 10
    infrastructure.app.dependency_overrides[get_current_user] = lambda: _principal(kind)
    with TestClient(infrastructure.app) as client:
        response = client.get("/v2/clusters/1/registration-token")
    if kind == "server":
        assert response.status_code == 200
        assert response.json()["token"] == "test-registration-token"
    else:
        assert response.status_code == 403
        infrastructure.lookup.assert_not_awaited()
        infrastructure.render_token.assert_not_called()


@pytest.mark.parametrize("kind", ["admin", "owner", "member", "server"])
def test_management_identities_can_read_cloud_credentials(infrastructure, kind):
    ctx = TenantContext(
        user=(
            _principal("server")
            if kind == "server"
            else SimpleNamespace(kind=PrincipalType.USER)
        ),
        is_platform_admin=kind == "admin",
        current_principal_id=20 if kind in ("owner", "member") else None,
        org_role=(
            OrgRole.OWNER
            if kind == "owner"
            else OrgRole.MEMBER if kind == "member" else None
        ),
    )
    infrastructure.app.dependency_overrides[get_tenant_context] = lambda: ctx
    infrastructure.app.dependency_overrides[get_current_user] = lambda: ctx.user
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    infrastructure.lookup.return_value = CloudCredential(
        id=2,
        name="target",
        owner_principal_id=20,
        secret="test-cloud-secret",
        created_at=timestamp,
        updated_at=timestamp,
    )
    with TestClient(infrastructure.app) as client:
        response = client.get("/v2/cloud-credentials/2")
    assert response.status_code == 200
    assert response.json()["name"] == "target"
    assert "test-cloud-secret" not in response.text


@pytest.mark.asyncio
async def test_system_backend_edit_cannot_create_an_org_override(monkeypatch):
    ctx = TenantContext(
        user=_principal("cluster"),
        is_platform_admin=False,
        current_principal_id=20,
        org_role=OrgRole.OWNER,
        scoped_cluster_id=1,
        scoped_cluster_owner_id=10,
    )
    backend = InferenceBackend(id=2, backend_name="custom", owner_principal_id=None)
    monkeypatch.setattr(InferenceBackend, "one_by_id", AsyncMock(return_value=backend))
    find_override = AsyncMock(return_value=None)
    create_override = AsyncMock()
    monkeypatch.setattr(InferenceBackend, "one_by_fields", find_override)
    monkeypatch.setattr(InferenceBackend, "create", create_override)
    with pytest.raises(ForbiddenException):
        await inference_backend.update_inference_backend(
            session=AsyncMock(),
            ctx=ctx,
            id=2,
            backend_in=InferenceBackendUpdate(backend_name="custom"),
        )
    find_override.assert_not_awaited()
    create_override.assert_not_awaited()
