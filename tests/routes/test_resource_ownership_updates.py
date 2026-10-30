"""API updates preserve tenant bindings while accepting unchanged echoes."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from gpustack.api.exceptions import register_handlers
from gpustack.api.tenant import TenantContext
from gpustack.routes import (
    cloud_credentials,
    gpu_instance_persistent_volume_types,
    gpu_instance_persistent_volumes,
    gpu_instance_ssh_public_keys,
    gpu_instance_templates,
    model_instances,
    models,
)
from gpustack.schemas import (
    GPUInstancePersistentVolume,
    GPUInstancePersistentVolumeType,
    GPUInstanceSSHPublicKey,
    GPUInstanceTemplate,
)
from gpustack.schemas.clusters import CloudCredential
from gpustack.schemas.models import Model, ModelInstance
from gpustack.schemas.principals import OrgRole, PrincipalType
from gpustack.server import services
from gpustack.server.deps import get_session, get_tenant_context

CASES = [
    pytest.param(
        GPUInstanceTemplate,
        gpu_instance_templates.update_gpu_instance_template,
        {"spec": {"image": "example.invalid/image:latest"}},
        id="template",
    ),
    pytest.param(
        GPUInstancePersistentVolumeType,
        gpu_instance_persistent_volume_types.update_gpu_instance_persistent_volume_type,
        {},
        id="volume-type",
    ),
    pytest.param(
        GPUInstancePersistentVolume,
        gpu_instance_persistent_volumes.update_gpu_instance_persistent_volume,
        {},
        id="volume",
    ),
    pytest.param(
        GPUInstanceSSHPublicKey,
        gpu_instance_ssh_public_keys.update_gpu_instance_ssh_public_key,
        {"spec": {"data": "ssh-ed25519 test"}},
        id="ssh-key",
    ),
    pytest.param(CloudCredential, cloud_credentials.update, {}, id="credential"),
    pytest.param(
        Model,
        models.update_model,
        {"source": "huggingface", "huggingface_repo_id": "org/model"},
        id="model",
    ),
    pytest.param(
        ModelInstance,
        model_instances.update_model_instance,
        {
            "source": "huggingface",
            "huggingface_repo_id": "org/model",
            "model_name": "test",
        },
        id="model-instance",
    ),
]


def context(actor):
    return TenantContext(
        user=SimpleNamespace(
            id=7,
            kind=PrincipalType.SYSTEM if actor == "worker" else PrincipalType.USER,
        ),
        is_platform_admin=actor in ("admin", "admin-in-org"),
        current_principal_id=None if actor == "admin" else 8,
        org_role=OrgRole.MEMBER if actor == "member" else OrgRole.OWNER,
        scoped_cluster_id=10 if actor == "worker" else None,
        scoped_cluster_owner_id=8 if actor == "worker" else None,
    )


@pytest.fixture
def api(monkeypatch, resource_type, endpoint, payload):
    resource = resource_type(
        **{
            "id": 1,
            "name": "test",
            "owner_principal_id": 8,
            "cluster_id": 10,
            "model_id": 2,
            **payload,
        }
    )
    session = MagicMock(info={})
    session.exec = AsyncMock(return_value=MagicMock())
    session.exec.return_value.one_or_none.return_value = resource
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    save = AsyncMock()
    monkeypatch.setattr(resource_type, "one_by_id", AsyncMock(return_value=resource))
    monkeypatch.setattr(resource_type, "save", save)
    monkeypatch.setattr(resource_type, "_publish_event_after_commit", MagicMock())

    baseline = AsyncMock(return_value=None)
    monkeypatch.setattr(models, "ensure_baseline", baseline)
    monkeypatch.setattr(models, "assert_cluster_belongs_to_org", AsyncMock())
    monkeypatch.setattr(models, "validate_model_in", AsyncMock())
    monkeypatch.setattr(models, "validate_shared_kv_cache", AsyncMock())
    monkeypatch.setattr(
        models.ModelRoute, "one_by_fields", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(models, "revoke_model_access_cache", AsyncMock())
    monkeypatch.setattr(services, "delete_cache_by_key", AsyncMock())
    monkeypatch.setattr(services, "invalidate_workers_allocated", AsyncMock())

    ctx = context("owner")
    app = FastAPI()
    register_handlers(app)
    app.add_api_route("/{id}", endpoint, methods=["PUT"])
    app.dependency_overrides[get_session] = lambda: session
    app.dependency_overrides[get_tenant_context] = lambda: ctx
    return SimpleNamespace(
        app=app,
        ctx=ctx,
        resource=resource,
        payload={"name": "test", **payload},
        save=save,
        session=session,
    )


@pytest_asyncio.fixture
async def client(api):
    async with AsyncClient(
        transport=ASGITransport(app=api.app), base_url="http://test"
    ) as client:
        yield client


@pytest.mark.asyncio
@pytest.mark.parametrize("resource_type,endpoint,payload", CASES)
@pytest.mark.parametrize("actor", ["owner", "admin", "admin-in-org", "worker"])
@pytest.mark.parametrize("new_owner", [None, 99])
async def test_owner_changes_rejected_before_persistence(
    api, client, actor, new_owner, resource_type
):
    api.ctx.__dict__.update(context(actor).__dict__)
    response = await client.put(
        "/1", json={**api.payload, "owner_principal_id": new_owner}
    )
    # Worker credentials can update model runtime resources, not infrastructure.
    expected_status = (
        403
        if actor == "worker" and resource_type not in (Model, ModelInstance)
        else 422
    )
    assert response.status_code == expected_status, response.text
    assert api.resource.owner_principal_id == 8
    api.save.assert_not_awaited()
    api.session.commit.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("resource_type,endpoint,payload", CASES)
@pytest.mark.parametrize("actor", ["owner", "admin", "admin-in-org", "worker"])
@pytest.mark.parametrize("echo_owner", [False, True])
async def test_updates_preserve_owner_and_require_write_access(
    api, client, actor, echo_owner, resource_type
):
    api.ctx.__dict__.update(context(actor).__dict__)
    field, value = (
        ("pid", 42)
        if isinstance(api.resource, ModelInstance)
        else ("description", "updated")
    )
    payload = {**api.payload, field: value}
    if echo_owner:
        payload["owner_principal_id"] = 8
    response = await client.put("/1", json=payload)
    if actor == "worker" and resource_type not in (Model, ModelInstance):
        assert response.status_code == 403, response.text
        assert api.resource.owner_principal_id == 8
        assert getattr(api.resource, field) != value
        api.save.assert_not_awaited()
        api.session.commit.assert_not_awaited()
        return
    assert response.status_code == 200, response.text
    assert api.resource.owner_principal_id == 8
    assert getattr(api.resource, field) == value
    api.save.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("resource_type,endpoint,payload", CASES)
async def test_other_tenant_cannot_claim_resource(api, client):
    api.resource.owner_principal_id = 99
    response = await client.put("/1", json={**api.payload, "owner_principal_id": 8})
    assert response.status_code in (403, 404), response.text
    assert api.resource.owner_principal_id == 99
    api.save.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "resource_type,endpoint,payload", [CASES[0], CASES[2], CASES[3]]
)
@pytest.mark.parametrize("new_owner", [None, 99, 8])
async def test_member_updates_preserve_owner(api, client, new_owner):
    api.ctx.__dict__.update(context("member").__dict__)
    response = await client.put(
        "/1", json={**api.payload, "owner_principal_id": new_owner}
    )
    assert response.status_code == (200 if new_owner == 8 else 422), response.text
    assert api.resource.owner_principal_id == 8
    assert api.save.await_count == (1 if new_owner == 8 else 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("resource_type,endpoint,payload", [CASES[0]])
@pytest.mark.parametrize("echo_owner", [False, True])
async def test_admin_can_edit_global_template_without_transferring_it(
    api, client, echo_owner
):
    api.ctx.__dict__.update(context("admin").__dict__)
    api.resource.owner_principal_id = None
    payload = {**api.payload, "description": "updated"}
    if echo_owner:
        payload["owner_principal_id"] = None
    response = await client.put("/1", json=payload)
    assert response.status_code == 200, response.text
    assert api.resource.owner_principal_id is None
    assert api.resource.description == "updated"
    api.save.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("resource_type,endpoint,payload", [CASES[-1]])
@pytest.mark.parametrize("actor", ["owner", "admin", "worker"])
@pytest.mark.parametrize("field", ["model_id", "cluster_id"])
@pytest.mark.parametrize("new_value", [None, 99])
async def test_instance_parent_bindings_cannot_change(
    api, client, actor, field, new_value
):
    api.ctx.__dict__.update(context(actor).__dict__)
    response = await client.put("/1", json={**api.payload, field: new_value})
    assert response.status_code == 422, response.text
    assert api.resource.model_id == 2
    assert api.resource.cluster_id == 10
    api.save.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("resource_type,endpoint,payload", [CASES[-1]])
@pytest.mark.parametrize("cluster_id", [None, 10])
async def test_worker_can_echo_instance_bindings(api, client, cluster_id):
    api.ctx.__dict__.update(context("worker").__dict__)
    api.resource.cluster_id = cluster_id
    payload = api.resource.model_dump(mode="json")
    payload["pid"] = 42
    response = await client.put("/1", json=payload)
    if cluster_id is None:
        assert response.status_code == 404, response.text
        api.save.assert_not_awaited()
        return
    assert response.status_code == 200, response.text
    assert api.resource.pid == 42
    assert api.resource.owner_principal_id == 8
    assert api.resource.model_id == 2
    assert api.resource.cluster_id == cluster_id
    api.save.assert_awaited_once()
