from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from gpustack.api.exceptions import (
    BadRequestException,
    ConflictException,
    InternalServerErrorException,
    NotFoundException,
    register_handlers,
)
from gpustack.routes import model_revisions as routes
from gpustack.routes import models as model_routes
from gpustack.schemas.model_revisions import ModelRevision, ModelRollbackRequest
from gpustack.schemas.models import Model, ModelCreate, ModelUpdate
from gpustack.schemas.principals import PrincipalType
from gpustack.server.model_revisions import deployment_spec
from gpustack.server.deps import get_session, get_tenant_context


@pytest.fixture
def setup(monkeypatch):
    model = Model(
        id=1,
        name="current-name",
        source="huggingface",
        huggingface_repo_id="org/current",
        owner_principal_id=5,
        cluster_id=7,
        revision_history_limit=2,
        description="current description",
        env={"EXAMPLE": "current"},
    )
    session = SimpleNamespace(
        commit=AsyncMock(), rollback=AsyncMock(), execute=AsyncMock()
    )
    ctx = SimpleNamespace(user=SimpleNamespace(id=99, kind=PrincipalType.USER))
    latest = ModelRevision(id=3, model_id=1, revision=3, spec=deployment_spec(model))
    target = ModelRevision(
        id=1,
        model_id=1,
        revision=1,
        spec={**deployment_spec(model), "env": {"EXAMPLE": "historical"}},
    )
    monkeypatch.setattr(routes, "lock_model", AsyncMock(return_value=model))
    monkeypatch.setattr(routes, "prepare_history_read", AsyncMock())
    monkeypatch.setattr(routes, "ensure_baseline", AsyncMock(return_value=latest))
    monkeypatch.setattr(routes, "_find_revision", AsyncMock(return_value=target))
    return SimpleNamespace(
        model=model, session=session, ctx=ctx, latest=latest, target=target
    )


@pytest.mark.asyncio
async def test_preview_only_compares_configuration(setup, monkeypatch):
    validate = AsyncMock(
        side_effect=AssertionError("preview must not validate deployment")
    )
    monkeypatch.setattr(model_routes, "validate_model_in", validate)
    result = await routes.preview_model_rollback(
        setup.session, setup.ctx, 1, ModelRollbackRequest(target_revision=1)
    )
    assert result.changed
    assert [change.field for change in result.changes] == ["env"]
    assert result.current_revision == 3
    assert result.target_revision == 1
    validate.assert_not_awaited()
    assert "valid" not in result.model_dump()
    assert "errors" not in result.model_dump()


@pytest.mark.asyncio
async def test_display_omits_nulls_but_rollback_clears_from_stored_snapshot(
    setup, monkeypatch
):
    setup.target.spec["env"] = None
    original = deepcopy(setup.target.spec)
    detail = await routes.get_model_revision(setup.session, setup.ctx, 1, 1)
    assert "env" not in detail.spec
    assert "image_name" not in detail.spec
    preview = await routes.preview_model_rollback(
        setup.session, setup.ctx, 1, ModelRollbackRequest(target_revision=1)
    )
    assert preview.current["env"] == {"EXAMPLE": "current"}
    assert "env" not in preview.desired
    assert [(change.field, change.desired) for change in preview.changes] == [
        ("env", None)
    ]
    assert preview.changed
    assert setup.target.spec == original

    save = AsyncMock(return_value=setup.model)
    monkeypatch.setattr(model_routes, "save_model_update", save)
    await routes.rollback_model(
        setup.session, setup.ctx, 1, ModelRollbackRequest(target_revision=1)
    )
    assert save.await_args.args[3].env is None
    assert setup.target.spec == original


@pytest.mark.asyncio
async def test_rollback_uses_stored_spec_preserving_live_identity(setup, monkeypatch):
    save = AsyncMock(return_value=setup.model)
    monkeypatch.setattr(model_routes, "save_model_update", save)
    await routes.rollback_model(
        setup.session, setup.ctx, 1, ModelRollbackRequest(target_revision=1)
    )
    update = save.await_args.args[3]
    assert update.env == {"EXAMPLE": "historical"}
    assert update.name == "current-name"
    assert update.owner_principal_id == 5
    assert update.cluster_id == 7
    assert update.description == "current description"
    assert update.revision_history_limit == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["generic_proxy", "env", "replicas"])
async def test_missing_snapshot_field_preserves_live_value_in_preview_and_rollback(
    setup, monkeypatch, missing
):
    setup.model.generic_proxy = True
    setup.model.replicas = 3
    setup.target.spec = deployment_spec(setup.model)
    setup.target.spec.pop(missing)
    original = deepcopy(setup.target.spec)
    preview = await routes.preview_model_rollback(
        setup.session, setup.ctx, 1, ModelRollbackRequest(target_revision=1)
    )
    assert not preview.changed
    assert preview.current == preview.desired
    save = AsyncMock(return_value=setup.model)
    monkeypatch.setattr(model_routes, "save_model_update", save)
    await routes.rollback_model(
        setup.session, setup.ctx, 1, ModelRollbackRequest(target_revision=1)
    )
    assert getattr(save.await_args.args[3], missing) == getattr(setup.model, missing)
    assert setup.target.spec == original


@pytest.mark.asyncio
async def test_generic_proxy_is_previewed_and_restored(setup, monkeypatch):
    setup.target.spec["generic_proxy"] = True
    preview = await routes.preview_model_rollback(
        setup.session, setup.ctx, 1, ModelRollbackRequest(target_revision=1)
    )
    assert preview.desired["generic_proxy"] is True
    assert "generic_proxy" in {change.field for change in preview.changes}
    save = AsyncMock(return_value=setup.model)
    monkeypatch.setattr(model_routes, "save_model_update", save)
    await routes.rollback_model(
        setup.session, setup.ctx, 1, ModelRollbackRequest(target_revision=1)
    )
    assert save.await_args.args[3].generic_proxy is True


@pytest.mark.asyncio
async def test_direct_rollback_initializes_baseline_before_target_lookup(
    setup, monkeypatch
):
    setup.latest.revision = 1
    calls = []

    async def baseline(*args):
        calls.append("baseline")
        return setup.latest

    async def find(*args, **kwargs):
        assert calls == ["baseline"]
        assert kwargs == {"for_update": True}
        return setup.latest

    monkeypatch.setattr(routes, "ensure_baseline", baseline)
    monkeypatch.setattr(routes, "_find_revision", find)
    save = AsyncMock(return_value=setup.model)
    monkeypatch.setattr(model_routes, "save_model_update", save)
    await routes.rollback_model(
        setup.session, setup.ctx, 1, ModelRollbackRequest(target_revision=1)
    )
    save.assert_awaited_once()


@pytest.fixture
def history_app(setup):
    app = FastAPI()
    app.include_router(routes.router, prefix="/models")
    register_handlers(app)
    app.dependency_overrides[get_tenant_context] = lambda: setup.ctx
    app.dependency_overrides[get_session] = lambda: setup.session
    return app


@pytest.mark.asyncio
@pytest.mark.parametrize("scoped_cluster_id", [None, 7])
@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", "/models/1/revisions"),
        ("GET", "/models/1/revisions/1"),
        ("POST", "/models/1/rollback-preview"),
        ("POST", "/models/1/rollback"),
        ("DELETE", "/models/1/revisions/1"),
    ],
)
async def test_system_principals_cannot_access_history(
    setup, history_app, method, path, scoped_cluster_id
):
    setup.ctx.user.kind = PrincipalType.SYSTEM
    setup.ctx.scoped_cluster_id = scoped_cluster_id
    async with AsyncClient(
        transport=ASGITransport(app=history_app), base_url="http://test"
    ) as client:
        response = await client.request(method, path, json={"target_revision": 1})
    assert response.status_code == 403
    routes.prepare_history_read.assert_not_awaited()
    routes.lock_model.assert_not_awaited()
    routes._find_revision.assert_not_awaited()
    setup.session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_authorized_user_can_read_saved_configuration(setup, history_app):
    setup.target.created_by = 99
    setup.session.exec = AsyncMock()
    async with AsyncClient(
        transport=ASGITransport(app=history_app), base_url="http://test"
    ) as client:
        response = await client.get("/models/1/revisions/1")
    assert response.status_code == 200
    assert response.json()["spec"]["env"] == {"EXAMPLE": "historical"}
    assert response.json()["created_by"] == 99
    assert set(response.json()) == {
        "id",
        "model_id",
        "revision",
        "created_at",
        "created_by",
        "spec",
    }
    setup.session.exec.assert_not_awaited()
    routes.prepare_history_read.assert_awaited_once_with(setup.session, setup.ctx, 1)
    routes.lock_model.assert_not_awaited()
    routes.ensure_baseline.assert_not_awaited()
    routes._find_revision.assert_awaited_once_with(setup.session, 1, 1)
    setup.session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_reading_pruned_revision_returns_not_found(setup, monkeypatch):
    monkeypatch.setattr(
        routes, "_find_revision", AsyncMock(side_effect=NotFoundException())
    )
    with pytest.raises(NotFoundException):
        await routes.get_model_revision(setup.session, setup.ctx, 1, 1)
    routes.lock_model.assert_not_awaited()
    routes.ensure_baseline.assert_not_awaited()
    setup.session.commit.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("for_update", [False, True])
async def test_revision_lookup_locks_only_when_requested(for_update):
    from sqlalchemy.dialects import mysql, postgresql

    item = ModelRevision(model_id=1, revision=2, spec={})
    session = SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(one_or_none=lambda: item))
    )
    assert await routes._find_revision(session, 1, 2, for_update=for_update) is item
    statement = session.exec.await_args.args[0]
    for dialect in (mysql.dialect(), postgresql.dialect()):
        assert ("FOR UPDATE" in str(statement.compile(dialect=dialect))) == for_update


def test_history_is_mounted_only_in_user_management_routes():
    from gpustack.routes.routes import model_routers, tenant_routers

    assert all(item["router"] is not routes.router for item in model_routers)
    assert any(item["router"] is routes.router for item in tenant_routers)
    assert all(
        getattr(route, "endpoint", None)
        not in {item.endpoint for item in routes.router.routes}
        for route in model_routes.router.routes
    )


@pytest.mark.asyncio
async def test_noop_rollback_still_uses_update_validation(setup, monkeypatch):
    setup.target.spec = deployment_spec(setup.model)
    save = AsyncMock(return_value=setup.model)
    monkeypatch.setattr(model_routes, "save_model_update", save)
    await routes.rollback_model(
        setup.session, setup.ctx, 1, ModelRollbackRequest(target_revision=1)
    )
    save.assert_awaited_once()


@pytest.mark.asyncio
async def test_missing_target_does_not_update(setup, monkeypatch):
    monkeypatch.setattr(
        routes, "_find_revision", AsyncMock(side_effect=NotFoundException())
    )
    save = AsyncMock()
    monkeypatch.setattr(model_routes, "save_model_update", save)
    with pytest.raises(NotFoundException):
        await routes.rollback_model(
            setup.session, setup.ctx, 1, ModelRollbackRequest(target_revision=1)
        )
    save.assert_not_awaited()
    setup.session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_malformed_snapshot_error_does_not_include_input(setup, monkeypatch):
    setup.target.spec["env"] = ["fixture-sensitive-input"]
    with pytest.raises(BadRequestException) as error:
        await routes.rollback_model(
            setup.session, setup.ctx, 1, ModelRollbackRequest(target_revision=1)
        )
    assert "fixture-sensitive-input" not in error.value.message


@pytest.mark.asyncio
async def test_latest_cannot_be_deleted(setup, monkeypatch):
    monkeypatch.setattr(routes, "_find_revision", AsyncMock(return_value=setup.latest))
    with pytest.raises(ConflictException):
        await routes.delete_model_revision(setup.session, setup.ctx, 1, 3)
    setup.session.execute.assert_not_awaited()
    setup.session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_old_revision_is_deleted_and_committed(setup):
    result = await routes.delete_model_revision(setup.session, setup.ctx, 1, 1)
    assert result.status_code == 204
    assert setup.session.execute.await_args.args[0].compile().params == {"id_1": 1}
    setup.session.commit.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "handler,args",
    [
        (routes.list_model_revisions, {"page": 1, "perPage": 10}),
        (routes.get_model_revision, {"revision": 1}),
        (
            routes.preview_model_rollback,
            {"body": ModelRollbackRequest(target_revision=1)},
        ),
        (routes.rollback_model, {"body": ModelRollbackRequest(target_revision=1)}),
        (routes.delete_model_revision, {"revision": 1}),
    ],
)
async def test_every_history_endpoint_checks_parent_before_reading_history(
    setup, monkeypatch, handler, args
):
    monkeypatch.setattr(
        routes, "lock_model", AsyncMock(side_effect=NotFoundException())
    )
    monkeypatch.setattr(
        routes, "prepare_history_read", AsyncMock(side_effect=NotFoundException())
    )
    with pytest.raises(NotFoundException):
        await handler(setup.session, setup.ctx, 1, **args)
    routes.ensure_baseline.assert_not_awaited()
    routes._find_revision.assert_not_awaited()


@pytest.mark.asyncio
async def test_list_uses_standard_pagination_without_spec_or_extra_metadata(
    setup, monkeypatch
):
    record = setup.latest.model_dump(exclude={"spec"})
    record["created_by"] = 99
    results = [
        SimpleNamespace(one=lambda: 1),
        SimpleNamespace(all=lambda: [SimpleNamespace(_mapping=record)]),
    ]
    setup.session.exec = AsyncMock(side_effect=results)
    result = await routes.list_model_revisions(
        setup.session, setup.ctx, 1, page=1, perPage=10
    )
    assert set(result.model_dump()) == {"items", "pagination"}
    assert set(result.items[0].model_dump()) == {
        "id",
        "model_id",
        "revision",
        "created_at",
        "created_by",
    }
    assert result.pagination.total == 1
    assert result.items[0].created_by == 99
    assert "model_revisions.spec" not in str(setup.session.exec.await_args.args[0])
    assert "principals" not in str(setup.session.exec.await_args.args[0])
    routes.prepare_history_read.assert_awaited_once_with(setup.session, setup.ctx, 1)
    routes.lock_model.assert_not_awaited()
    routes.ensure_baseline.assert_not_awaited()
    setup.session.commit.assert_not_awaited()
    for call in setup.session.exec.await_args_list:
        assert "FOR UPDATE" not in str(call.args[0])


@pytest.fixture
def save_boundary(setup, monkeypatch):
    monkeypatch.setattr(
        model_routes, "ensure_baseline", AsyncMock(return_value=setup.latest)
    )
    monkeypatch.setattr(
        model_routes, "_apply_model_update", AsyncMock(return_value=setup.model)
    )
    monkeypatch.setattr(model_routes, "record_update", AsyncMock())
    monkeypatch.setattr(model_routes, "revoke_model_access_cache", AsyncMock())
    return ModelUpdate.model_validate(setup.model.model_dump())


@pytest.mark.asyncio
async def test_save_records_history_before_commit(setup, save_boundary, monkeypatch):
    calls = []

    async def record(*args):
        calls.append("history")

    async def commit():
        calls.append("commit")

    monkeypatch.setattr(model_routes, "record_update", record)
    setup.session.commit.side_effect = commit
    await model_routes.save_model_update(
        setup.session, setup.ctx, setup.model, save_boundary
    )
    assert calls == ["history", "commit"]
    setup.session.rollback.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_stage", ["validation", "history", "commit"])
async def test_failed_update_rolls_back_without_logging_payload(
    setup, save_boundary, monkeypatch, failing_stage, caplog
):
    failure = ValueError("fixture-sensitive-input")
    if failing_stage == "validation":
        model_routes._apply_model_update.side_effect = BadRequestException(
            message="Invalid configuration"
        )
        expected = BadRequestException
    elif failing_stage == "history":
        model_routes.record_update.side_effect = failure
        expected = InternalServerErrorException
    else:
        setup.session.commit.side_effect = failure
        expected = InternalServerErrorException
    with pytest.raises(expected) as error:
        await model_routes.save_model_update(
            setup.session, setup.ctx, setup.model, save_boundary
        )
    setup.session.rollback.assert_awaited_once()
    model_routes.revoke_model_access_cache.assert_not_awaited()
    assert "fixture-sensitive-input" not in error.value.message
    assert "fixture-sensitive-input" not in caplog.text
    if failing_stage == "validation":
        model_routes.record_update.assert_not_awaited()
    else:
        assert "in save_model_update" in caplog.text
        assert "ValueError" in caplog.text


@pytest.mark.asyncio
async def test_create_failure_logs_stack_without_payload(setup, monkeypatch, caplog):
    monkeypatch.setattr(
        model_routes, "_resolve_target_org", AsyncMock(return_value=(5, None))
    )
    monkeypatch.setattr(model_routes, "_check_model_create", AsyncMock())
    monkeypatch.setattr(
        model_routes,
        "_persist_model_create",
        AsyncMock(side_effect=ValueError("fixture-sensitive-input")),
    )
    with pytest.raises(InternalServerErrorException) as error:
        await model_routes.create_model(
            setup.session,
            setup.ctx,
            ModelCreate.model_validate(setup.model.model_dump()),
        )
    setup.session.rollback.assert_awaited_once()
    assert "fixture-sensitive-input" not in caplog.text
    assert "fixture-sensitive-input" not in error.value.message
    assert "in create_model" in caplog.text
    assert "ValueError" in caplog.text


@pytest.mark.asyncio
async def test_worker_runtime_write_does_not_create_history(
    setup, save_boundary, monkeypatch
):
    setup.ctx.user.kind = PrincipalType.SYSTEM
    projection = AsyncMock(
        side_effect=AssertionError("runtime writes need no snapshot")
    )
    monkeypatch.setattr(model_routes, "deployment_spec", projection)
    await model_routes.save_model_update(
        setup.session, setup.ctx, setup.model, save_boundary
    )
    model_routes.ensure_baseline.assert_not_awaited()
    model_routes.record_update.assert_not_awaited()
    projection.assert_not_called()
    setup.session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_import_update_records_before_route_side_effects(setup, monkeypatch):
    monkeypatch.setattr(
        model_routes, "ensure_baseline", AsyncMock(return_value=setup.latest)
    )
    record = AsyncMock()
    monkeypatch.setattr(model_routes, "record_update", record)

    async def update(model, source, **kwargs):
        for key, value in source.items():
            setattr(model, key, value)

    service = SimpleNamespace(update=AsyncMock(side_effect=update))
    monkeypatch.setattr(model_routes, "ModelService", lambda _: service)
    monkeypatch.setattr(model_routes, "_own_model_routes", AsyncMock(return_value=[]))
    from gpustack.schemas.models import ModelCreate

    request = ModelCreate.model_validate(
        {**setup.model.model_dump(), "env": {"EXAMPLE": "new"}}
    )
    before = deepcopy(deployment_spec(setup.model))
    await model_routes._persist_model_update(setup.session, setup.model, request, 99)
    assert record.await_args.args[2] == before
    assert record.await_args.args[4] == 99
    assert setup.model.env == {"EXAMPLE": "new"}
    setup.session.commit.assert_not_awaited()
