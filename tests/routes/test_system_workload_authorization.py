"""Runtime API admission does not grant access to another Org's workloads."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from sqlalchemy.orm.evaluator import _EvaluatorCompiler

from gpustack.api.auth import get_current_user
from gpustack.api.exceptions import register_handlers
from gpustack.api.tenant import TenantContext, get_tenant_context
from gpustack.routes import model_instances
from gpustack.routes.routes import management_router
from gpustack.schemas.common import Pagination
from gpustack.schemas.models import ModelInstance, ModelInstancesPublic
from gpustack.schemas.principals import OrgRole, PrincipalType
from gpustack.server import services
from gpustack.server.db import get_session

PAYLOAD = dict(
    name="instance",
    model_name="model",
    model_id=100,
    source="huggingface",
    huggingface_repo_id="org/model",
)


def _row(cluster_id=1, owner_id=10, id=2):
    now = datetime.now(timezone.utc)
    return ModelInstance(
        id=id,
        **PAYLOAD,
        cluster_id=cluster_id,
        owner_principal_id=owner_id,
        created_at=now,
        updated_at=now,
    )


@pytest.fixture(scope="module")
def workload_app():
    app = FastAPI()
    register_handlers(app)
    app.include_router(management_router)
    return app


@pytest.fixture
def workload_api(workload_app, monkeypatch):
    user = SimpleNamespace(
        id=7,
        name="system/worker/test",
        kind=PrincipalType.SYSTEM,
        is_admin=False,
        worker=SimpleNamespace(id=11, cluster_id=1),
        cluster=None,
    )
    ctx = TenantContext(
        user=user,
        is_platform_admin=False,
        current_principal_id=None,
        org_role=None,
        scoped_cluster_id=1,
        scoped_cluster_owner_id=10,
    )
    instance = _row()
    lookup = AsyncMock(return_value=instance)
    parent = AsyncMock(
        return_value=SimpleNamespace(id=100, cluster_id=1, owner_principal_id=10)
    )
    save, delete = AsyncMock(), AsyncMock()
    session = MagicMock(info={})
    monkeypatch.setattr(ModelInstance, "one_by_id", lookup)
    monkeypatch.setattr(ModelInstance, "save", save)
    monkeypatch.setattr(ModelInstance, "delete", delete)
    monkeypatch.setattr(ModelInstance, "_publish_event_after_commit", MagicMock())
    monkeypatch.setattr(model_instances.Model, "one_by_id", parent)
    monkeypatch.setattr(services, "delete_cache_by_key", AsyncMock())
    monkeypatch.setattr(services, "invalidate_workers_allocated", AsyncMock())

    async def create_instance(session, source):
        now = datetime.now(timezone.utc)
        return ModelInstance(
            id=3, created_at=now, updated_at=now, **source.model_dump()
        )

    create = AsyncMock(side_effect=create_instance)
    monkeypatch.setattr(ModelInstance, "create", create)

    @asynccontextmanager
    async def open_session():
        yield session

    monkeypatch.setattr(model_instances, "async_session", open_session)
    workload_app.dependency_overrides[get_current_user] = lambda: user
    workload_app.dependency_overrides[get_tenant_context] = lambda: ctx
    workload_app.dependency_overrides[get_session] = lambda: session
    yield SimpleNamespace(
        app=workload_app,
        ctx=ctx,
        lookup=lookup,
        parent=parent,
        create=create,
        save=save,
        delete=delete,
    )
    workload_app.dependency_overrides.clear()


@pytest.mark.parametrize("method", ["GET", "POST", "PUT", "DELETE"])
@pytest.mark.parametrize("cluster_id", [1, 2, None])
@pytest.mark.parametrize("owner_id", [10, 20, None])
def test_system_instance_api_requires_both_bindings(
    workload_api, method, cluster_id, owner_id
):
    api = workload_api
    api.lookup.return_value = _row(cluster_id, owner_id)
    api.parent.return_value = SimpleNamespace(
        id=100, cluster_id=cluster_id, owner_principal_id=owner_id
    )
    path = "/v2/model-instances" + ("" if method == "POST" else "/2")
    with TestClient(api.app) as client:
        response = client.request(
            method, path, json={**PAYLOAD, "state_message": "runtime report"}
        )
    allowed = cluster_id == 1 and owner_id == 10
    assert response.status_code == (200 if allowed else 404), response.text
    assert api.create.await_count == int(allowed and method == "POST")
    assert api.save.await_count == int(allowed and method == "PUT")
    assert api.delete.await_count == int(allowed and method == "DELETE")


@pytest.mark.parametrize("method", ["GET", "POST", "PUT", "DELETE"])
@pytest.mark.parametrize("kind", ["user", "server"])
def test_org_users_and_platform_token_keep_their_resource_access(
    workload_api, method, kind
):
    api = workload_api
    api.lookup.return_value = _row(1, 20)
    api.parent.return_value = SimpleNamespace(
        id=100, cluster_id=1, owner_principal_id=20
    )
    api.ctx.scoped_cluster_id = None
    api.ctx.scoped_cluster_owner_id = None
    if kind == "server":
        api.ctx.user.id = None
    else:
        api.ctx.user.kind = PrincipalType.USER
        api.ctx.current_principal_id = 20
        api.ctx.org_role = OrgRole.OWNER
        api.ctx.accessible_cluster_ids = {1}
    path = "/v2/model-instances" + ("" if method == "POST" else "/2")
    with TestClient(api.app) as client:
        response = client.request(method, path, json=PAYLOAD)
    assert response.status_code == 200, response.text


def test_instance_list_and_watch_apply_the_same_system_scope(workload_api, monkeypatch):
    rows = [
        _row(1, 10, 1),
        _row(1, 20, 2),
        _row(None, 10, 3),
        _row(None, 20, 4),
        _row(2, 10, 5),
    ]

    async def page(**kwargs):
        matches = _EvaluatorCompiler(ModelInstance).process(*kwargs["extra_conditions"])
        visible = [row for row in rows if matches(row) is True]
        return ModelInstancesPublic(
            items=visible,
            pagination=Pagination(page=1, perPage=20, total=len(visible), totalPage=1),
        )

    monkeypatch.setattr(
        ModelInstance, "paginated_by_query", AsyncMock(side_effect=page)
    )
    captured = {}

    def streaming(**kwargs):
        captured["filter"] = kwargs["filter_func"]

        async def events():
            if False:
                yield ""

        return events()

    monkeypatch.setattr(ModelInstance, "streaming", streaming)
    with TestClient(workload_api.app) as client:
        response = client.get("/v2/model-instances")
        assert response.status_code == 200, response.text
        assert [row["id"] for row in response.json()["items"]] == [1]
        response = client.get("/v2/model-instances?watch=true")
        assert response.status_code == 200, response.text
    assert [row.id for row in rows if captured["filter"](row)] == [1]
