from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI

from gpustack.api.exceptions import register_handlers
from gpustack.api.tenant import TenantContext
from gpustack.routes import model_instances as routes
from gpustack.routes.routes import model_routers
from gpustack.schemas.models import (
    ModelConfigSnapshot,
    ModelInstance,
    ModelInstancePublic,
)
from gpustack.schemas.principals import OrgRole, PrincipalType
from gpustack.server.deps import get_session, get_tenant_context
from gpustack.server.bus import Event, EventType


@pytest.fixture
def api(monkeypatch):
    ctx = TenantContext(
        user=SimpleNamespace(kind=PrincipalType.USER),
        is_platform_admin=False,
        current_principal_id=10,
        org_role=OrgRole.OWNER,
        accessible_cluster_ids={1},
    )
    parent = SimpleNamespace(id=100, owner_principal_id=10, cluster_id=1)
    payload = {
        "name": "test-instance",
        "model_name": "test-model",
        "model_id": parent.id,
        "source": "huggingface",
        "huggingface_repo_id": "test/model",
        "state": "running",
    }
    now = datetime.now(timezone.utc)
    instance = ModelInstance(
        **payload,
        id=1,
        owner_principal_id=parent.owner_principal_id,
        cluster_id=parent.cluster_id,
        created_at=now,
        updated_at=now,
    )
    session = MagicMock(info={})
    parent_lookup = AsyncMock(return_value=parent)
    monkeypatch.setattr(routes.Model, "one_by_id", parent_lookup)
    monkeypatch.setattr(
        routes.ModelInstance, "one_by_id", AsyncMock(return_value=instance)
    )

    async def create(session, source):
        return ModelInstancePublic(
            **source.model_dump(), id=2, created_at=now, updated_at=now
        )

    create_mock = AsyncMock(side_effect=create)
    monkeypatch.setattr(routes.ModelInstance, "create", create_mock)
    save = AsyncMock()
    monkeypatch.setattr(routes.ModelInstance, "save", save)

    async def update(current, source):
        await current.update(session, source)

    service = MagicMock()
    service.update = AsyncMock(side_effect=update)
    monkeypatch.setattr(routes, "ModelInstanceService", lambda session: service)
    app = FastAPI()
    router_config = next(r for r in model_routers if r["router"] is routes.router)
    app.include_router(
        routes.router,
        prefix="/v2" + router_config["prefix"],
        dependencies=router_config["dependencies"],
    )
    app.dependency_overrides[get_session] = lambda: session
    app.dependency_overrides[get_tenant_context] = lambda: ctx
    register_handlers(app)
    return SimpleNamespace(
        app=app,
        ctx=ctx,
        parent=parent,
        parent_lookup=parent_lookup,
        instance=instance,
        payload=payload,
        create=create_mock,
        update=service.update,
        save=save,
    )


async def _request(api, method, **fields):
    path = "/v2/model-instances" + ("/1" if method == "PUT" else "")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(api.app), base_url="http://server"
    ) as client:
        return await client.request(method, path, json={**api.payload, **fields})


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["other_tenant", "missing", "missing_id"])
async def test_create_rejects_inaccessible_parent(api, case):
    if case == "other_tenant":
        api.parent.owner_principal_id = 20
    elif case == "missing":
        api.parent_lookup.return_value = None
    else:
        api.payload.pop("model_id")
    response = await _request(api, "POST")
    assert response.status_code == 404
    api.create.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_binding", [False, True])
@pytest.mark.parametrize("cluster_id", [1, None])
async def test_create_inherits_parent_binding(api, explicit_binding, cluster_id):
    api.parent.cluster_id = cluster_id
    fields = (
        {"owner_principal_id": 10, "cluster_id": cluster_id} if explicit_binding else {}
    )
    response = await _request(api, "POST", **fields)
    assert response.status_code == 200
    assert response.json()["owner_principal_id"] == 10
    assert response.json()["cluster_id"] == cluster_id
    api.create.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fields",
    [{"owner_principal_id": 20}, {"cluster_id": 2}, {"cluster_id": None}],
)
async def test_create_rejects_conflicting_binding(api, fields):
    response = await _request(api, "POST", **fields)
    assert response.status_code == 422
    api.create.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["POST", "PUT"])
async def test_org_member_cannot_write_instances(api, method):
    api.ctx.org_role = OrgRole.MEMBER
    response = await _request(api, method)
    assert response.status_code == 403
    api.create.assert_not_awaited()
    api.update.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "identity, allowed",
    [
        ("admin", True),
        ("admin_act_as", False),
        ("legacy_system", True),
        ("same_cluster_system", True),
        ("other_cluster_system", False),
    ],
)
@pytest.mark.parametrize("method", ["POST", "PUT"])
async def test_privileged_identity_scope(api, identity, allowed, method):
    api.parent.owner_principal_id = 20
    api.instance.owner_principal_id = 20
    if identity.startswith("admin"):
        api.ctx.is_platform_admin = True
        if identity == "admin":
            api.ctx.current_principal_id = None
    else:
        api.ctx.user.kind = PrincipalType.SYSTEM
        api.ctx.current_principal_id = None
        if identity != "legacy_system":
            api.ctx.scoped_cluster_id = 1 if identity == "same_cluster_system" else 2
            api.ctx.scoped_cluster_owner_id = 20
    response = await _request(api, method)
    assert response.status_code == (200 if allowed else 404)
    if not allowed:
        api.create.assert_not_awaited()
        api.update.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fields",
    [
        {"model_id": 200},
        {"owner_principal_id": 20},
        {"cluster_id": 2},
        {"cluster_id": None},
    ],
)
async def test_update_rejects_binding_changes(api, fields):
    response = await _request(api, "PUT", **fields)
    assert response.status_code == 422
    api.save.assert_not_awaited()
    assert api.instance.model_id == 100
    assert api.instance.owner_principal_id == 10
    assert api.instance.cluster_id == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("full_report", [False, True])
async def test_update_accepts_unchanged_or_omitted_binding(api, full_report):
    if full_report:
        api.ctx.user.kind = PrincipalType.SYSTEM
        api.ctx.scoped_cluster_id = 1
        api.ctx.scoped_cluster_owner_id = 10
        api.payload = ModelInstancePublic.model_validate(api.instance).model_dump(
            mode="json"
        )
    else:
        api.payload.pop("model_id")
    response = await _request(api, "PUT", state="error", state_message="test error")
    assert response.status_code == 200
    assert response.json()["model_id"] == 100
    assert response.json()["owner_principal_id"] == 10
    assert response.json()["cluster_id"] == 1
    assert response.json()["state"] == "error"
    api.update.assert_awaited_once()
    api.save.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_rejects_other_tenants_instance(api):
    api.instance.owner_principal_id = 20
    response = await _request(api, "PUT", owner_principal_id=10)
    assert response.status_code == 404
    api.update.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("admin", [False, True])
@pytest.mark.parametrize("snapshot", [{}, {"env": {"TOKEN": "forged"}}])
async def test_user_cannot_forge_startup_configuration(api, admin, snapshot):
    api.ctx.is_platform_admin = admin
    response = await _request(api, "PUT", model_config_snapshot=snapshot)
    assert response.status_code == 403
    assert "forged" not in response.text
    api.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_user_status_update_preserves_private_snapshot(api):
    recorded = ModelConfigSnapshot(env={"TOKEN": "private-config-value"})
    api.instance.model_config_snapshot = recorded
    response = await _request(api, "PUT", model_config_snapshot=None, state="error")
    assert response.status_code == 200
    assert api.instance.model_config_snapshot == recorded
    assert "private-config-value" not in response.text


@pytest.mark.asyncio
async def test_cluster_system_can_write_snapshot_without_exposing_it(api):
    api.ctx.user.kind = PrincipalType.SYSTEM
    api.ctx.current_principal_id = None
    api.ctx.scoped_cluster_id = 1
    api.ctx.scoped_cluster_owner_id = 10
    response = await _request(
        api, "PUT", model_config_snapshot={"env": {"TOKEN": "private-config-value"}}
    )
    assert response.status_code == 200
    assert api.instance.model_config_snapshot.env == {"TOKEN": "private-config-value"}
    assert "model_config_snapshot" not in response.json()
    assert "private-config-value" not in response.text


@pytest.mark.asyncio
async def test_other_cluster_system_cannot_write_snapshot(api):
    api.ctx.user.kind = PrincipalType.SYSTEM
    api.ctx.current_principal_id = None
    api.ctx.scoped_cluster_id = 2
    api.ctx.scoped_cluster_owner_id = 10
    response = await _request(
        api, "PUT", model_config_snapshot={"backend_parameters": ["--forged"]}
    )
    assert response.status_code == 404
    api.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_does_not_accept_user_supplied_startup_snapshot(api):
    response = await _request(
        api, "POST", model_config_snapshot={"env": {"TOKEN": "forged"}}
    )
    assert response.status_code == 200
    assert "model_config_snapshot" not in api.create.await_args.args[1].model_dump()
    assert "forged" not in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize("list_response", [False, True])
async def test_instance_read_responses_do_not_expose_startup_config(
    api, monkeypatch, list_response
):
    api.instance.model_config_snapshot = ModelConfigSnapshot(
        env={"TOKEN": "private-config-value"}
    )
    if list_response:
        monkeypatch.setattr(
            routes.ModelInstance,
            "paginated_by_query",
            AsyncMock(
                return_value={
                    "items": [api.instance],
                    "pagination": {
                        "page": 1,
                        "perPage": 20,
                        "total": 1,
                        "totalPage": 1,
                    },
                }
            ),
        )
        manager = MagicMock()
        manager.__aenter__ = AsyncMock(return_value=MagicMock())
        manager.__aexit__ = AsyncMock(return_value=False)
        monkeypatch.setattr(routes, "async_session", lambda: manager)
    path = "/v2/model-instances" + ("" if list_response else "/1")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(api.app), base_url="http://server"
    ) as client:
        response = await client.get(path)
    assert response.status_code == 200
    assert "model_config_snapshot" not in response.text
    assert "private-config-value" not in response.text


@pytest.mark.asyncio
async def test_instance_watch_hides_private_configuration_and_change_values(
    api, monkeypatch
):
    api.instance.model_config_snapshot = ModelConfigSnapshot(
        env={"TOKEN": "private-config-value"}
    )
    changes = {
        "model_config_snapshot": (
            None,
            api.instance.model_config_snapshot.model_dump(),
        ),
        "state": ("starting", "running"),
    }

    async def events():
        yield Event(type=EventType.UPDATED, data=api.instance, changed_fields=changes)

    monkeypatch.setattr(routes.ModelInstance, "subscribe", lambda **kwargs: events())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(api.app), base_url="http://server"
    ) as client:
        response = await client.get("/v2/model-instances", params={"watch": "true"})
    assert response.status_code == 200
    assert "model_config_snapshot" not in response.text
    assert "private-config-value" not in response.text
    assert '"changed_fields":{"state":' in response.text
    assert "model_config_snapshot" in changes
