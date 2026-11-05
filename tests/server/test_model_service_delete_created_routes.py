"""Deleting a model tears down the route it created, in the same transaction.

A route with ``created_model_id`` must not outlive the model that created it:
while one exists, the unique-name lookup in ``create_model`` resolves it and
rejects a new model of that name. A route that other targets still point at is
kept instead, with ``created_model_id`` intact.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from sqlalchemy import event
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession
import pytest
import pytest_asyncio

import gpustack.schemas  # noqa: F401  (registers every table)
from gpustack.api.exceptions import (
    ConflictException,
    InternalServerErrorException,
    NotFoundException,
)
from gpustack.routes import models as model_routes
from gpustack.schemas.model_routes import (
    ModelRoute,
    ModelRouteTarget,
    TargetStateEnum,
)
from gpustack.schemas.models import (
    DistributedServers,
    Model,
    ModelInstance,
    ModelInstanceStateEnum,
    ModelInstanceWorkloadStateEnum,
    ModelInstanceLegacyStop,
    ModelInstanceSubordinateWorker,
    ModelInstanceWorkload,
    ModelInstanceWorkloadUpdate,
    RoleSpec,
    SourceEnum,
)
from gpustack.schemas.clusters import Cluster, ClusterProvider
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.schemas.principals import (
    Principal,
    PrincipalType,
    platform_principal_id,
)
from gpustack.server.services import ModelService, ModelInstanceService, WorkerService
from gpustack.server.bus import EventType, event_bus
from tests.utils.model import new_model_instance_workload

ORG_ID = platform_principal_id()


@pytest_asyncio.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite://")

    @event.listens_for(engine.sync_engine, "connect")
    def _enforce_foreign_keys(dbapi_connection, _record):
        # The production databases cascade target deletion through foreign
        # keys; SQLite only does so with the pragma switched on.
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    async with AsyncSession(engine, expire_on_commit=False) as s:
        s.add(Principal(id=ORG_ID, name="default", kind=PrincipalType.ORG))
        await s.commit()
        await Cluster.create(
            s,
            Cluster(
                id=1,
                name='cluster',
                provider=ClusterProvider.Docker,
                owner_principal_id=ORG_ID,
            ),
        )
        await Worker.create(
            s,
            Worker(
                id=1,
                name='worker-1',
                hostname='worker-1',
                ip='192.0.2.1',
                ifname='eth0',
                port=10150,
                worker_uuid='worker-1',
                cluster_id=1,
                state=WorkerStateEnum.READY,
                heartbeat_time=datetime.now(timezone.utc),
            ),
        )
        yield s
    await engine.dispose()


async def _model_with_created_route(session: AsyncSession, name: str) -> Model:
    model = await Model.create(
        session,
        source=Model(
            name=name,
            source=SourceEnum.HUGGING_FACE,
            huggingface_repo_id="org/repo",
            owner_principal_id=ORG_ID,
            enable_model_route=True,
        ),
    )
    route = await ModelRoute.create(
        session,
        source=ModelRoute(
            name=name, created_model_id=model.id, owner_principal_id=ORG_ID
        ),
        auto_commit=False,
    )
    await ModelRouteTarget.create(
        session,
        source=ModelRouteTarget(
            name=f"{name}-deployment",
            route_name=route.name,
            model_route=route,
            model=model,
            weight=100,
            state=TargetStateEnum.UNAVAILABLE,
        ),
    )
    return model


async def _route(session: AsyncSession, name: str):
    return await ModelRoute.one_by_fields(
        session, {"name": name, "owner_principal_id": ORG_ID}
    )


@pytest.mark.asyncio
async def test_delete_model_removes_the_route_it_created(session):
    model = await _model_with_created_route(session, "m1")
    assert await _route(session, "m1") is not None

    await ModelService(session).delete(model)

    # This is the lookup create_model uses for its 409 check.
    assert await _route(session, "m1") is None
    assert await ModelRouteTarget.all_by_fields(session, {"route_name": "m1"}) == []


@pytest.mark.asyncio
async def test_delete_model_keeps_a_created_route_other_targets_still_use(session):
    model = await _model_with_created_route(session, "shared")
    other = await Model.create(
        session,
        source=Model(
            name="other",
            source=SourceEnum.HUGGING_FACE,
            huggingface_repo_id="org/other",
            owner_principal_id=ORG_ID,
        ),
    )
    route = await _route(session, "shared")
    await ModelRouteTarget.create(
        session,
        source=ModelRouteTarget(
            name="other-deployment",
            route_name=route.name,
            model_route=route,
            model=other,
            weight=100,
            state=TargetStateEnum.UNAVAILABLE,
        ),
    )

    await ModelService(session).delete(model)

    kept = await _route(session, "shared")
    assert kept is not None and kept.created_model_id == model.id
    remaining = await ModelRouteTarget.all_by_fields(session, {"route_name": "shared"})
    assert [target.model_id for target in remaining] == [other.id]


@pytest.mark.asyncio
async def test_delete_model_ignores_soft_deleted_targets_of_other_models(session):
    model = await _model_with_created_route(session, "stale")
    other = await Model.create(
        session,
        source=Model(
            name="other-stale",
            source=SourceEnum.HUGGING_FACE,
            huggingface_repo_id="org/other",
            owner_principal_id=ORG_ID,
        ),
    )
    route = await _route(session, "stale")
    stale_target = await ModelRouteTarget.create(
        session,
        source=ModelRouteTarget(
            name="other-stale-deployment",
            route_name=route.name,
            model_route=route,
            model=other,
            weight=100,
            state=TargetStateEnum.UNAVAILABLE,
        ),
    )
    await stale_target.delete(session, soft=True)

    await ModelService(session).delete(model)

    assert await _route(session, "stale") is None


async def _managed_instance(session, model, state="running", group_id="pd-group"):
    return await ModelInstance.create(
        session,
        ModelInstance(
            name=f"{model.name}-instance",
            group_id=group_id,
            model_id=model.id,
            model_name=model.name,
            owner_principal_id=ORG_ID,
            source=SourceEnum.HUGGING_FACE,
            huggingface_repo_id="org/repo",
            worker_id=1,
            cluster_id=1,
            gpu_indexes=[0],
            state=ModelInstanceStateEnum.RUNNING,
            distributed_servers=DistributedServers(
                workloads=[new_model_instance_workload(state=state)]
            ),
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("task_state", ["running", "pending", "failed", "unknown"])
async def test_managed_model_delete_preserves_children_and_routes_until_stop(
    session, task_state
):
    model = await _model_with_created_route(session, "managed")
    instance = await _managed_instance(session, model, task_state)

    pending = await ModelService(session).delete(model)
    assert pending is False

    assert model.deleted_at is not None
    assert model.replicas == 0
    assert await Model.one_by_id(session, model.id) is model
    assert await ModelInstance.one_by_id(session, instance.id) is instance
    assert instance.draining_since is not None
    assert instance.distributed_servers.workloads[0].state == task_state
    assert await _route(session, model.name) is not None
    from unittest.mock import AsyncMock, patch

    service = ModelService(session)
    with patch.object(service, "stop_instances", AsyncMock()) as request_stop:
        await service.delete(model)
    request_stop.assert_not_awaited()

    servers = instance.distributed_servers.model_copy(deep=True)
    servers.workloads[0].state = ModelInstanceWorkloadStateEnum.STOPPED
    await instance.update(session, {"distributed_servers": servers})
    completed = await ModelService(session).delete(model)
    assert completed is True

    assert await Model.one_by_id(session, model.id) is None
    assert await ModelInstance.one_by_id(session, instance.id) is None
    assert await _route(session, model.name) is None


@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['stop', 'delete'])
@pytest.mark.parametrize('lost', ['missing', 'deleted', 'heartbeat', 'unreachable'])
@pytest.mark.parametrize('legacy', [False, True])
async def test_deployment_teardown_retires_lost_workers_after_live_peers_stop(
    session, operation, lost, legacy
):
    from uuid import uuid4
    from gpustack.server.controllers import sync_replicas

    model = await _model_with_created_route(session, 'lost-worker')
    await model.update(
        session,
        {
            'cluster_id': 1,
            'roles': [RoleSpec(name='prefill'), RoleSpec(name='decode')],
        },
    )
    instance = await _managed_instance(session, model, 'unknown')
    worker = await Worker.one_by_id(session, 1)
    if lost == 'missing':
        await worker.delete(session)
    elif lost == 'deleted':
        await worker.update(session, {'deleted_at': datetime.now(timezone.utc)})
    elif lost == 'heartbeat':
        await worker.update(
            session, {'heartbeat_time': datetime.now(timezone.utc) - timedelta(days=1)}
        )
    else:
        await worker.update(session, {'unreachable': True})
    await Worker.create(
        session,
        Worker(
            id=2,
            name='worker-2',
            hostname='worker-2',
            ip='192.0.2.2',
            ifname='eth0',
            port=10150,
            worker_uuid='worker-2',
            cluster_id=1,
            state=WorkerStateEnum.READY,
            heartbeat_time=datetime.now(timezone.utc),
        ),
    )
    servers = instance.distributed_servers.model_copy(deep=True)
    servers.subordinate_workers = [
        ModelInstanceSubordinateWorker(worker_id=2, gpu_indexes=[0])
    ]
    if legacy:
        servers.workloads = None
        servers.orchestration_mode = "legacy"
        servers.legacy_stop = ModelInstanceLegacyStop(
            token=uuid4(),
            restart_count=0,
            worker_ids=[1, 2],
            worker_principal_ids={1: None, 2: None},
        )
    else:
        servers.workloads.append(
            ModelInstanceWorkload.model_validate(
                new_model_instance_workload(worker_id=2, node_rank=1, state='running')
            )
        )
    await instance.update(session, {'distributed_servers': servers})

    async def reconcile():
        if operation == 'delete':
            await ModelService(session).delete(model)
        else:
            await sync_replicas(session, model)

    if operation == 'stop':
        await model.update(session, {'replicas': 0})
    await reconcile()
    assert await ModelInstance.one_by_id(session, instance.id) is instance
    assert instance.draining_since is not None
    servers = instance.distributed_servers.model_copy(deep=True)
    if legacy:
        servers.legacy_stop.confirmed_worker_ids = [2]
    else:
        assert servers.workloads[0].state == ModelInstanceWorkloadStateEnum.UNKNOWN
        servers.workloads[1].state = ModelInstanceWorkloadStateEnum.STOPPED
    await instance.update(session, {'distributed_servers': servers})
    instance_id = instance.id
    await reconcile()
    assert await ModelInstance.one_by_id(session, instance_id) is None

    if operation == 'delete':
        assert await Model.one_by_id(session, model.id) is None
        assert await _route(session, model.name) is None
        replacement = await _model_with_created_route(session, 'lost-worker')
        assert replacement is not None
    else:
        assert await Model.one_by_id(session, model.id) is model
        await model.update(session, {'replicas': 1})
        await sync_replicas(session, model)
        replacements = await ModelInstance.all_by_field(session, 'model_id', model.id)
        assert len(replacements) == 2
        assert all(
            item.state == ModelInstanceStateEnum.PENDING for item in replacements
        )
        assert all(item.draining_since is None for item in replacements)


@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['stop', 'delete'])
async def test_deployment_teardown_with_no_reachable_worker_releases_worker_deletion(
    session, operation, monkeypatch
):
    from gpustack.server.controllers import sync_replicas
    from gpustack.routes import model_instances as instance_routes

    model = await _model_with_created_route(session, 'all-lost')
    instance = await _managed_instance(session, model, 'unknown')
    worker = await Worker.one_by_id(session, 1)
    await worker.update(session, {'unreachable': True})
    instance_id = instance.id
    late_report = ModelInstanceWorkloadUpdate(
        restart_count=instance.restart_count or 0,
        workloads=[w.model_dump() for w in instance.distributed_servers.workloads],
    )
    if operation == 'delete':
        pending = await ModelService(session).delete(model)
        assert pending is False
        completed = await ModelService(session).delete(model)
        assert completed is True
        assert await Model.one_by_id(session, model.id) is None
    else:
        await model.update(session, {'replicas': 0})
        await sync_replicas(session, model)
    assert await ModelInstance.one_by_id(session, instance_id) is None
    monkeypatch.setattr(
        instance_routes, '_locked_reporting_worker', AsyncMock(return_value=worker)
    )
    ctx = SimpleNamespace(user=SimpleNamespace(worker=worker))
    with pytest.raises(NotFoundException):
        await instance_routes.report_model_instance_workloads(
            session, ctx, instance_id, late_report
        )
    await WorkerService(session).delete(worker)
    assert await Worker.one_by_id(session, 1) is None


@pytest.mark.asyncio
async def test_stopping_one_deployment_preserves_other_lost_worker_instances(session):
    from gpustack.server.controllers import sync_replicas

    model = await _model_with_created_route(session, 'stopped')
    instance = await _managed_instance(session, model, 'unknown')
    other = await _model_with_created_route(session, 'unaffected')
    other_instance = await _managed_instance(session, other, 'unknown')
    worker = await Worker.one_by_id(session, 1)
    await worker.update(session, {'unreachable': True})
    await model.update(session, {'replicas': 0})
    await sync_replicas(session, model)
    assert await ModelInstance.one_by_id(session, instance.id) is None
    assert await ModelInstance.one_by_id(session, other_instance.id) is other_instance
    assert other_instance.draining_since is None
    assert other.replicas == 1
    with pytest.raises(ConflictException):
        await WorkerService(session).delete(worker)


@pytest.mark.asyncio
async def test_instance_delete_keeps_manifest_until_confirmed_stopped(session):
    model = await _model_with_created_route(session, "single-stop")
    instance = await _managed_instance(session, model)
    service = ModelInstanceService(session)

    assert await service.delete(instance) is False
    since = instance.draining_since
    assert since is not None
    assert instance.distributed_servers.stop_requested is True
    assert await service.delete(instance) is False
    assert instance.draining_since == since
    assert instance.gpu_indexes == [0]
    assert instance.distributed_servers.workloads
    assert await service.get_running_instances(model.id) == []

    servers = instance.distributed_servers.model_copy(deep=True)
    servers.workloads[0].state = ModelInstanceWorkloadStateEnum.STOPPED
    await instance.update(session, {"distributed_servers": servers})
    assert await service.delete(instance) is True
    assert await ModelInstance.one_by_id(session, instance.id) is None


@pytest.mark.asyncio
async def test_all_stop_batch_retains_legacy_member_while_managed_task_is_unknown(
    session,
):
    model = await _model_with_created_route(session, "all-stop")
    instance = await _managed_instance(session, model, "unknown")
    legacy = await ModelInstance.create(
        session,
        ModelInstance(
            name="legacy-router",
            model_id=model.id,
            model_name=model.name,
            owner_principal_id=ORG_ID,
            source=SourceEnum.HUGGING_FACE,
            huggingface_repo_id="org/repo",
        ),
    )

    assert (
        await ModelInstanceService(session).batch_delete(
            [legacy, instance], require_all_stopped=True
        )
        == []
    )
    assert await ModelInstance.one_by_id(session, legacy.id) is legacy
    assert await ModelInstance.one_by_id(session, instance.id) is instance


@pytest.mark.asyncio
async def test_batch_stop_conflict_rolls_back_without_erasing_task_evidence(session):
    model = await _model_with_created_route(session, 'stop-conflict')
    managed = await _managed_instance(session, model, 'unknown')
    legacy = await ModelInstance.create(
        session,
        ModelInstance(
            name='unassigned-legacy',
            model_id=model.id,
            model_name=model.name,
            owner_principal_id=ORG_ID,
            source=SourceEnum.HUGGING_FACE,
            huggingface_repo_id='org/repo',
            state=ModelInstanceStateEnum.RUNNING,
            pid=123,
        ),
    )
    managed_id, legacy_id = managed.id, legacy.id
    with pytest.raises(ConflictException) as error:
        await ModelInstanceService(session).batch_delete([managed, legacy])
    assert error.value.status_code == 409
    current = await ModelInstance.one_by_id(session, managed_id, refresh=True)
    assert current.draining_since is None
    assert not current.distributed_servers.stop_requested
    assert current.distributed_servers.workloads[0].state == 'unknown'
    current = await ModelInstance.one_by_id(session, legacy_id, refresh=True)
    assert current.draining_since is None
    assert current.distributed_servers is None
    assert current.pid == 123


@pytest.mark.asyncio
async def test_model_restart_marks_parent_and_all_children_in_one_commit(session):
    from datetime import datetime, timezone
    from unittest.mock import patch

    model = await _model_with_created_route(session, "atomic-stop")
    instance = await _managed_instance(session, model)
    with patch.object(session, "commit", wraps=session.commit) as commit:
        await ModelService(session).stop_instances(
            model, [instance], {"restarting_since": datetime.now(timezone.utc)}
        )
    assert commit.await_count == 1
    assert model.restarting_since is not None
    assert instance.draining_since is not None
    assert instance.distributed_servers.stop_requested is True


@pytest.mark.asyncio
async def test_stale_delete_snapshot_preserves_the_new_generation(session):
    model = await _model_with_created_route(session, "stale-delete")
    instance = await _managed_instance(session, model)
    snapshot = instance.model_copy(deep=True)
    await instance.update(
        session,
        {"restart_count": 1, "distributed_servers": DistributedServers(workloads=[])},
    )
    assert await ModelInstanceService(session).batch_delete([snapshot]) == []
    assert await ModelInstance.one_by_id(session, instance.id) is instance
    assert instance.restart_count == 1
    assert instance.draining_since is None


@pytest.mark.asyncio
async def test_savepoint_rollback_preserves_outer_relationship_snapshot(
    session, monkeypatch
):
    model = await _model_with_created_route(session, "relationship-snapshot")
    route = await _route(session, model.name)
    route = await session.get(
        ModelRoute,
        route.id,
        populate_existing=True,
        options=[selectinload(ModelRoute.route_targets)],
    )
    await asyncio.sleep(0)
    published = AsyncMock()
    monkeypatch.setattr(event_bus, "publish", published)
    await route.update(session, {"description": "outer-saved"}, auto_commit=False)
    nested = await session.begin_nested()
    target = ModelRouteTarget(
        name="discarded",
        route_id=route.id,
        route_name=route.name,
        model_id=model.id,
        weight=100,
        state=TargetStateEnum.UNAVAILABLE,
    )
    session.add(target)
    await route.update(
        session, {"route_targets": [*route.route_targets, target]}, auto_commit=False
    )
    assert "route_targets" in session.info["pending_events"][-1].event.changed_fields
    await nested.rollback()
    await session.commit()
    await asyncio.sleep(0)
    published.assert_awaited_once()
    notification = published.await_args.args[1]
    assert notification.data.description == "outer-saved"
    assert [item.name for item in notification.data.route_targets] == [
        "relationship-snapshot-deployment"
    ]
    assert "route_targets" not in notification.changed_fields


@pytest.mark.asyncio
async def test_commit_payload_includes_later_changes_to_loaded_related_rows(
    session, monkeypatch
):
    model = await _model_with_created_route(session, "related-update")
    route = await _route(session, model.name)
    route = await session.get(
        ModelRoute,
        route.id,
        populate_existing=True,
        options=[selectinload(ModelRoute.route_targets)],
    )
    await asyncio.sleep(0)
    published = AsyncMock()
    monkeypatch.setattr(event_bus, "publish", published)
    await route.update(session, {"description": "saved"}, auto_commit=False)
    await route.route_targets[0].update(session, {"name": "renamed"}, auto_commit=False)
    await session.commit()
    await asyncio.sleep(0)
    notifications = {
        topic: item for topic, item in (call.args for call in published.await_args_list)
    }
    assert notifications["modelroute"].data.route_targets[0].name == "renamed"


@pytest.mark.asyncio
@pytest.mark.parametrize("newer_restart", [False, True])
async def test_legacy_restart_failure_clears_guard_without_rolled_back_deletions(
    session, monkeypatch, newer_restart
):
    model = await _model_with_created_route(session, "restart-failure")
    instance = await _managed_instance(session, model)
    await instance.update(session, {"distributed_servers": None})
    model_id, instance_id = model.id, instance.id
    await asyncio.sleep(0)
    published = AsyncMock()
    monkeypatch.setattr(event_bus, "publish", published)
    monkeypatch.setattr(model_routes, "assert_resource_visible", MagicMock())
    monkeypatch.setattr(
        model_routes, "model_spec_digest", AsyncMock(return_value="sha1:test")
    )
    original_delete = ModelInstance.delete
    failed_deletions = []

    async def fail_delete(row, current_session, **kwargs):
        await original_delete(row, current_session, **kwargs)
        await current_session.flush()
        failed_deletions.append(instance_id)
        raise RuntimeError("delete failed")

    monkeypatch.setattr(ModelInstance, "delete", fail_delete)
    replacement_guard = datetime(2040, 1, 1, tzinfo=timezone.utc)
    original_batch_delete = ModelInstanceService.batch_delete

    async def delete_and_replace_guard(service, instances):
        try:
            return await original_batch_delete(service, instances)
        except InternalServerErrorException:
            if newer_restart:
                async with AsyncSession(session.bind, expire_on_commit=False) as other:
                    current = await Model.one_by_id(other, model_id)
                    await current.update(other, {"restarting_since": replacement_guard})
            raise

    monkeypatch.setattr(ModelInstanceService, "batch_delete", delete_and_replace_guard)
    with pytest.raises(InternalServerErrorException) as error:
        await model_routes.restart_model(session, SimpleNamespace(), model_id)
    assert error.value.message.startswith("Failed to delete model instances")
    assert failed_deletions == [instance_id]
    await asyncio.sleep(0)
    await model.refresh(session)
    assert model.restarting_since == (replacement_guard if newer_restart else None)
    assert await ModelInstance.one_by_id(session, instance_id) is not None
    notifications = [call.args[1] for call in published.await_args_list]
    assert not any(item.type == EventType.DELETED for item in notifications)
    if not newer_restart:
        assert any(
            item.type == EventType.UPDATED and item.data.restarting_since is None
            for item in notifications
        )


@pytest.mark.asyncio
async def test_failed_stop_request_rolls_back_parent_and_children(session):
    from datetime import datetime, timezone
    from unittest.mock import AsyncMock, patch

    model = await _model_with_created_route(session, "rollback-stop")
    instance = await _managed_instance(session, model)
    with patch.object(
        ModelInstance, "update", AsyncMock(side_effect=RuntimeError("write failed"))
    ):
        with pytest.raises(RuntimeError):
            await ModelService(session).stop_instances(
                model, [instance], {"restarting_since": datetime.now(timezone.utc)}
            )
    await session.refresh(model)
    await session.refresh(instance)
    assert model.restarting_since is None
    assert instance.draining_since is None


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["delete", "batch", "model"])
async def test_ordinary_managed_delete_does_not_wait_for_physical_stop(
    session, operation
):
    model = await _model_with_created_route(session, "ordinary-stop")
    instance = await _managed_instance(session, model, group_id=None)
    service = ModelInstanceService(session)
    if operation == "delete":
        await service.delete(instance)
    elif operation == "batch":
        assert await service.batch_delete([instance]) == [instance.name]
    else:
        await ModelService(session).delete(model)
        assert await Model.one_by_id(session, model.id) is None
    assert await ModelInstance.one_by_id(session, instance.id) is None
    assert instance.draining_since is None
    assert instance.distributed_servers.stop_requested is False


@pytest.mark.asyncio
async def test_ordinary_pause_removes_the_ready_instance_in_one_convergence(
    session, monkeypatch
):
    from gpustack.server import controllers

    model = await _model_with_created_route(session, "ordinary-pause")
    instance = await _managed_instance(session, model, group_id=None)
    await model.update(session, {"replicas": 0, "ready_replicas": 1})
    monkeypatch.setattr(
        controllers,
        'find_scale_down_candidates',
        AsyncMock(return_value=[SimpleNamespace(model_instance=instance)]),
    )
    await controllers.sync_replicas(session, model)
    assert await ModelInstance.one_by_id(session, instance.id) is None
    await controllers.sync_model_status(session, model)
    assert model.replicas == 0
    assert model.ready_replicas == 0
    assert instance.draining_since is None
    assert not instance.distributed_servers.stop_requested
