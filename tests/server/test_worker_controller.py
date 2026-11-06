from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

import pytest

from gpustack.schemas.cache_services import CacheServiceStateEnum
from gpustack.schemas.models import (
    DistributedServerCoordinateModeEnum,
    DistributedServers,
    ModelInstanceSubordinateWorker,
    ModelInstanceStateEnum,
    ModelInstance,
    Model,
    ModelInstanceWorkloadStateEnum,
    ModelInstanceWorkload,
    ModelInstanceComponent,
    ComputedResourceClaim,
)
from gpustack.schemas.workers import Maintenance, Worker, WorkerStateEnum
from gpustack.server.bus import Event, EventType, Subscriber
from gpustack.server.controllers import WorkerController
from gpustack.server.worker_syncer import WorkerSyncer
from gpustack.server.services import WorkerService, ModelInstanceService
from gpustack.api.exceptions import ConflictException
from gpustack.utils.model_instance_workers import get_model_instance_worker_match
from tests.fixtures.workers.fixtures import (
    linux_nvidia_1_4090_24gx1,
    linux_nvidia_2_4080_16gx2,
)
from tests.utils.mock import mock_async_session
from tests.utils.model import new_model, new_model_instance, new_model_instance_workload


@pytest.fixture
def managed_worker():
    main, sub = linux_nvidia_1_4090_24gx1(), linux_nvidia_2_4080_16gx2()
    for worker in (main, sub):
        worker.cluster_id = 1
        worker.state = WorkerStateEnum.READY
        worker.unreachable = False
        worker.heartbeat_time = datetime.now(timezone.utc)
    instance = new_model_instance(
        1,
        "managed",
        1,
        worker_id=main.id,
        state=ModelInstanceStateEnum.RUNNING,
        gpu_indexes=[0],
        computed_resource_claim=ComputedResourceClaim(vram={0: 100}),
    )
    instance.cluster_id = 1
    instance.worker_name = main.name
    instance.restart_count = 3
    instance.distributed_servers = DistributedServers(
        workloads=[
            new_model_instance_workload(
                worker_id=worker.id, node_rank=index, state="running"
            )
            for index, worker in enumerate((main, sub))
        ],
        subordinate_workers=[
            ModelInstanceSubordinateWorker(
                worker_id=sub.id,
                worker_name=sub.name,
                gpu_indexes=[0],
                computed_resource_claim=ComputedResourceClaim(vram={0: 200}),
            )
        ],
    )
    instance.distributed_servers.workloads[0].dp_rank_placements[0].ready = True
    cm = mock_async_session()
    session = cm.__aenter__.return_value
    session.get.side_effect = lambda cls, key, **kwargs: (
        instance
        if cls is ModelInstance
        else (main if key == main.id else sub if key == sub.id else None)
    )
    return SimpleNamespace(
        main=main, sub=sub, instance=instance, cm=cm, session=session
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("components", [False, True])
@pytest.mark.parametrize("headless", [False, True])
@pytest.mark.parametrize(
    "event_type", [EventType.CREATED, EventType.UPDATED, EventType.DELETED]
)
async def test_managed_worker_loss_retains_the_complete_instance(
    managed_worker, headless, event_type, components
):
    state = managed_worker
    worker = state.sub if headless else state.main
    if components:
        state.instance.distributed_servers.workloads = [
            ModelInstanceWorkload(
                name='engine',
                worker_id=state.main.id,
                state='running',
                component=dict(
                    kind='sglang_engine', api_port=41000, dp_size=2, ready=True
                ),
            ),
            ModelInstanceWorkload(
                name='smg',
                worker_id=state.main.id,
                state='running',
                component=dict(
                    kind='sglang_smg', api_port=41002, metrics_port=41003, ready=True
                ),
            ),
            ModelInstanceWorkload(
                name='headless',
                worker_id=state.sub.id,
                state='running',
                component=dict(kind='sglang_engine'),
            ),
        ]
        state.instance.distributed_servers.orchestration_mode = "managed"
    worker.unreachable = True
    worker.state = WorkerStateEnum.UNREACHABLE
    unaffected = [
        item.model_copy(deep=True)
        for item in state.instance.distributed_servers.workloads
        if item.worker_id != worker.id
    ]
    if event_type == EventType.DELETED:
        state.session.get.side_effect = lambda cls, key, **kwargs: (
            state.instance if cls is ModelInstance else None
        )
    with (
        patch("gpustack.server.controllers.async_session", return_value=state.cm),
        patch.object(
            ModelInstance, "all_by_field", AsyncMock(return_value=[state.instance])
        ),
        patch(
            "gpustack.server.controllers.ModelInstanceService.update", AsyncMock()
        ) as update,
        patch(
            "gpustack.server.controllers.ModelInstanceService.batch_delete", AsyncMock()
        ) as delete,
    ):
        await WorkerController(MagicMock())._reconcile(
            Event(
                type=event_type,
                data=worker,
                changed_fields={"labels": ({}, {"a": "b"})},
            )
        )
    delete.assert_not_awaited()
    update.assert_awaited_once()
    patch_dict = update.await_args.args[1]
    assert patch_dict["state"] == ModelInstanceStateEnum.UNREACHABLE
    assert patch_dict["state_message"] == f"Worker {worker.id} is unavailable."
    local = [
        item
        for item in patch_dict["distributed_servers"].workloads
        if item.worker_id == worker.id
    ]
    assert all(item.state == ModelInstanceWorkloadStateEnum.UNKNOWN for item in local)
    assert all(
        not placement.ready for item in local for placement in item.dp_rank_placements
    )
    assert [
        item
        for item in patch_dict["distributed_servers"].workloads
        if item.worker_id != worker.id
    ] == unaffected
    for item in local:
        if item.component is not None:
            assert item.component.ready is (False if item.component.api_port else None)
    DistributedServers.model_validate_json(
        patch_dict["distributed_servers"].model_dump_json()
    )
    assert state.instance.worker_id == state.main.id
    assert state.instance.restart_count == 3
    assert state.instance.draining_since is None
    assert state.instance.computed_resource_claim.vram == {0: 100}


@pytest.mark.parametrize("managed", [False, True])
def test_managed_worker_membership_uses_id_instead_of_reused_name(
    managed_worker, managed
):
    state = managed_worker
    if not managed:
        state.instance.distributed_servers.workloads = None
        state.instance.distributed_servers.orchestration_mode = "legacy"
    match = get_model_instance_worker_match(
        state.instance, worker_id=999, worker_name=state.sub.name
    )
    assert match.matched is (not managed)


@pytest.mark.asyncio
@pytest.mark.parametrize("original", [False, True])
async def test_worker_sync_persists_reachability_changes_during_maintenance(
    managed_worker, original
):
    state = managed_worker
    worker = state.main
    worker.state = WorkerStateEnum.MAINTENANCE
    worker.state_message = "planned"
    worker.maintenance = Maintenance(enabled=True, message="planned")
    worker.unreachable = original
    stored = worker.model_copy(deep=True)

    async def probe(current):
        current.unreachable = not original

    syncer = WorkerSyncer(lambda: MagicMock(), lambda: MagicMock())
    with (
        patch("gpustack.server.worker_syncer.async_session", return_value=state.cm),
        patch.object(Worker, "all", AsyncMock(return_value=[worker])),
        patch.object(syncer, "_should_check_unreachable", return_value=True),
        patch.object(syncer, "_set_worker_unreachable", AsyncMock(side_effect=probe)),
        patch.object(Worker, "all_by_ids", AsyncMock(return_value=[stored])),
        patch.object(WorkerService, "batch_update", AsyncMock()) as update,
    ):
        await syncer._sync_workers_states()
    update.assert_awaited_once()
    assert stored.unreachable is (not original)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "manifest", [None, [], [new_model_instance_workload(state="unknown")]]
)
@pytest.mark.parametrize("location", ["main", "headless", "unrelated"])
async def test_worker_identity_is_kept_until_managed_tasks_stop(
    managed_worker, manifest, location
):
    state = managed_worker
    target = state.main if location == "main" else state.sub
    if location == "unrelated":
        target = state.main.model_copy(deep=True)
        target.id = 999
    state.session.get.side_effect = lambda cls, key, **kwargs: (
        target if cls is Worker else state.instance
    )
    state.instance.distributed_servers = DistributedServers(
        workloads=manifest,
        subordinate_workers=state.instance.distributed_servers.subordinate_workers,
    )
    state.session.exec.return_value.all.return_value = [state.instance]
    with (
        patch.object(Worker, "delete", AsyncMock()) as delete,
        patch("gpustack.server.services.delete_cache_by_key", AsyncMock()),
    ):
        if manifest and location != "unrelated":
            with pytest.raises(ConflictException):
                await WorkerService(state.session).delete(target)
            delete.assert_not_awaited()
        else:
            await WorkerService(state.session).delete(target)
            delete.assert_awaited_once()


@pytest.fixture(autouse=True)
def cache_instances(monkeypatch):
    lookup = AsyncMock(return_value=[])
    monkeypatch.setattr(
        "gpustack.server.controllers.CacheServiceInstance.all_by_field", lookup
    )
    return lookup


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "worker_state, unreachable, changed_fields",
    [
        (
            WorkerStateEnum.NOT_READY,
            False,
            {"state": (WorkerStateEnum.READY, WorkerStateEnum.NOT_READY)},
        ),
        (
            WorkerStateEnum.UNREACHABLE,
            True,
            {"state": (WorkerStateEnum.READY, WorkerStateEnum.UNREACHABLE)},
        ),
        (WorkerStateEnum.READY, True, {"unreachable": (False, True)}),
    ],
)
@pytest.mark.parametrize("model_location", ["none", "other_worker", "same_worker"])
async def test_worker_offline_marks_cache_instances_without_model_dependency(
    cache_instances, worker_state, unreachable, changed_fields, model_location
):
    worker = linux_nvidia_1_4090_24gx1()
    worker.state = worker_state
    worker.unreachable = unreachable
    cache_instance = SimpleNamespace(
        name="cache-instance",
        worker_id=worker.id,
        state=CacheServiceStateEnum.RUNNING,
        update=AsyncMock(),
    )
    cache_instances.return_value = [cache_instance]
    models = []
    if model_location != "none":
        model = new_model_instance(
            1,
            "model-instance",
            1,
            worker_id=worker.id if model_location == "same_worker" else worker.id + 1,
            state=ModelInstanceStateEnum.RUNNING,
        )
        models.append(model)

    event = Event(type=EventType.UPDATED, data=worker, changed_fields=changed_fields)
    with (
        patch(
            "gpustack.server.controllers.async_session",
            return_value=mock_async_session(),
        ),
        patch(
            "gpustack.server.controllers.ModelInstance.all_by_field",
            AsyncMock(return_value=models),
        ),
        patch(
            "gpustack.server.controllers.ModelInstanceService.update",
            new_callable=AsyncMock,
        ) as update_model,
    ):
        await WorkerController(MagicMock())._reconcile(event)

    cache_instances.assert_awaited_once()
    assert cache_instances.await_args.args[1:] == ("worker_id", worker.id)
    cache_instance.update.assert_awaited_once()
    assert cache_instance.update.await_args.args[1] == {
        "state": CacheServiceStateEnum.UNREACHABLE,
        "healthy": False,
        "state_message": "Worker is unreachable from the server",
    }
    if model_location == "same_worker":
        update_model.assert_awaited_once()
        assert (
            update_model.await_args.args[1]["state"]
            == ModelInstanceStateEnum.UNREACHABLE
        )
    else:
        update_model.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state",
    [
        CacheServiceStateEnum.PENDING,
        CacheServiceStateEnum.STARTING,
        CacheServiceStateEnum.ERROR,
        CacheServiceStateEnum.UNREACHABLE,
    ],
)
async def test_worker_offline_preserves_non_running_cache_instances(
    cache_instances, state
):
    worker = linux_nvidia_1_4090_24gx1()
    worker.state = WorkerStateEnum.NOT_READY
    instance = SimpleNamespace(worker_id=worker.id, state=state, update=AsyncMock())
    cache_instances.return_value = [instance]
    event = Event(
        type=EventType.UPDATED,
        data=worker,
        changed_fields={"state": (WorkerStateEnum.READY, worker.state)},
    )
    with (
        patch(
            "gpustack.server.controllers.async_session",
            return_value=mock_async_session(),
        ),
        patch(
            "gpustack.server.controllers.ModelInstance.all_by_field",
            AsyncMock(return_value=[]),
        ),
    ):
        await WorkerController(MagicMock())._reconcile(event)

    cache_instances.assert_awaited_once()
    instance.update.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "event_type, worker_state, changed_fields",
    [
        (
            EventType.UPDATED,
            WorkerStateEnum.READY,
            {"state": (WorkerStateEnum.NOT_READY, WorkerStateEnum.READY)},
        ),
        (
            EventType.UPDATED,
            WorkerStateEnum.MAINTENANCE,
            {"state": (WorkerStateEnum.READY, WorkerStateEnum.MAINTENANCE)},
        ),
        (EventType.DELETED, WorkerStateEnum.NOT_READY, None),
        (EventType.UPDATED, WorkerStateEnum.READY, {"labels": ({}, {"a": "b"})}),
    ],
)
async def test_worker_events_leave_cache_lifecycle_to_its_controllers(
    cache_instances, event_type, worker_state, changed_fields
):
    worker = linux_nvidia_1_4090_24gx1()
    worker.state = worker_state
    worker.unreachable = False
    event = Event(type=event_type, data=worker, changed_fields=changed_fields)
    with (
        patch(
            "gpustack.server.controllers.async_session",
            return_value=mock_async_session(),
        ),
        patch(
            "gpustack.server.controllers.ModelInstance.all_by_field",
            AsyncMock(return_value=[]),
        ),
    ):
        await WorkerController(MagicMock())._reconcile(event)

    cache_instances.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "final_state",
    [WorkerStateEnum.NOT_READY, WorkerStateEnum.UNREACHABLE, WorkerStateEnum.READY],
)
@pytest.mark.parametrize("changed_fields", [{"labels": ({}, {"zone": "a"})}, {}])
async def test_cache_availability_uses_latest_coalesced_worker_snapshot(
    cache_instances, final_state, changed_fields
):
    worker = linux_nvidia_1_4090_24gx1()
    worker.state = (
        final_state
        if final_state != WorkerStateEnum.READY
        else WorkerStateEnum.NOT_READY
    )
    worker.unreachable = False
    subscriber = Subscriber(topic="worker", source="worker_controller")
    await subscriber.enqueue(
        Event(
            type=EventType.UPDATED,
            data=worker.model_copy(deep=True),
            changed_fields={"state": (WorkerStateEnum.READY, worker.state)},
        )
    )
    if final_state == WorkerStateEnum.READY:
        previous_state = worker.state
        worker.state = final_state
        await subscriber.enqueue(
            Event(
                type=EventType.UPDATED,
                data=worker.model_copy(deep=True),
                changed_fields={"state": (previous_state, final_state)},
            )
        )
    if "labels" in changed_fields:
        worker.labels = changed_fields["labels"][1]
    await subscriber.enqueue(
        Event(type=EventType.UPDATED, data=worker, changed_fields=changed_fields)
    )
    event = await subscriber.receive()
    assert event.changed_fields == changed_fields
    assert subscriber.queue.empty()

    instance = SimpleNamespace(
        worker_id=worker.id,
        name="cache-instance",
        state=CacheServiceStateEnum.RUNNING,
        update=AsyncMock(),
    )
    cache_instances.return_value = [instance]
    with (
        patch(
            "gpustack.server.controllers.async_session",
            return_value=mock_async_session(),
        ),
        patch(
            "gpustack.server.controllers.ModelInstance.all_by_field",
            AsyncMock(return_value=[]),
        ),
    ):
        await WorkerController(MagicMock())._reconcile(event)

    if final_state == WorkerStateEnum.READY:
        instance.update.assert_not_awaited()
    else:
        instance.update.assert_awaited_once()
        assert instance.update.await_args.args[1] == {
            "state": CacheServiceStateEnum.UNREACHABLE,
            "healthy": False,
            "state_message": "Worker is unreachable from the server",
        }


@pytest.mark.asyncio
async def test_worker_controller_marks_distributed_subordinate_unreachable_when_worker_unreachable():
    main_worker = linux_nvidia_1_4090_24gx1()
    subordinate_worker = linux_nvidia_2_4080_16gx2()
    subordinate_worker.state = WorkerStateEnum.UNREACHABLE
    subordinate_worker.unreachable = True

    instance = new_model_instance(
        1,
        "distributed-instance",
        1,
        worker_id=main_worker.id,
        state=ModelInstanceStateEnum.RUNNING,
    )
    instance.worker_name = main_worker.name
    instance.distributed_servers = DistributedServers(
        mode=DistributedServerCoordinateModeEnum.RUN_FIRST,
        subordinate_workers=[
            ModelInstanceSubordinateWorker(
                worker_id=subordinate_worker.id,
                worker_name=subordinate_worker.name,
                worker_ip=subordinate_worker.ip,
                state=ModelInstanceStateEnum.RUNNING,
            )
        ],
    )

    update = AsyncMock(return_value=instance.name)
    event = Event(
        type=EventType.UPDATED,
        data=subordinate_worker,
        changed_fields={"state": (WorkerStateEnum.READY, WorkerStateEnum.UNREACHABLE)},
    )

    with (
        patch(
            "gpustack.server.controllers.async_session",
            return_value=mock_async_session(),
        ),
        patch(
            "gpustack.server.controllers.ModelInstance.all_by_field",
            AsyncMock(return_value=[instance]),
        ),
        patch(
            "gpustack.server.controllers.ModelInstanceService.update",
            update,
        ),
    ):
        controller = WorkerController(MagicMock())
        await controller._reconcile(event)

    update.assert_awaited_once()
    updated_instance = update.await_args.args[-2]
    patch_dict = update.await_args.args[-1]
    assert updated_instance.name == instance.name
    assert "state" not in patch_dict
    assert "state_message" not in patch_dict
    subordinate_patch = patch_dict["distributed_servers"].subordinate_workers[0]
    assert subordinate_patch.state == ModelInstanceStateEnum.UNREACHABLE
    assert subordinate_patch.state_message == "Worker is unreachable from the server"


@pytest.mark.asyncio
async def test_worker_controller_deletes_distributed_instance_when_subordinate_worker_deleted():
    main_worker = linux_nvidia_1_4090_24gx1()
    subordinate_worker = linux_nvidia_2_4080_16gx2()
    subordinate_worker.state = WorkerStateEnum.READY

    instance = new_model_instance(
        1,
        "distributed-instance",
        1,
        worker_id=main_worker.id,
        state=ModelInstanceStateEnum.RUNNING,
    )
    instance.worker_name = main_worker.name
    instance.distributed_servers = DistributedServers(
        mode=DistributedServerCoordinateModeEnum.RUN_FIRST,
        subordinate_workers=[
            ModelInstanceSubordinateWorker(
                worker_id=subordinate_worker.id,
                worker_name=subordinate_worker.name,
                worker_ip=subordinate_worker.ip,
                state=ModelInstanceStateEnum.RUNNING,
            )
        ],
    )

    batch_delete = AsyncMock(return_value=[instance.name])
    event = Event(type=EventType.DELETED, data=subordinate_worker)

    with (
        patch(
            "gpustack.server.controllers.async_session",
            return_value=mock_async_session(),
        ),
        patch(
            "gpustack.server.controllers.ModelInstance.all_by_field",
            AsyncMock(return_value=[instance]),
        ),
        patch(
            "gpustack.server.controllers.ModelInstanceService.batch_delete",
            batch_delete,
        ),
    ):
        controller = WorkerController(MagicMock())
        await controller._reconcile(event)

    batch_delete.assert_awaited_once()
    deleted_instances = batch_delete.await_args.args[-1]
    assert [item.name for item in deleted_instances] == [instance.name]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["new_generation", "worker_recovered", "stale_delete"])
async def test_managed_loss_uses_current_generation_and_worker_state(
    managed_worker, case
):
    state = managed_worker
    hint = state.instance.model_copy(deep=True)
    old_worker = state.main.model_copy(deep=True)
    old_worker.state = WorkerStateEnum.UNREACHABLE
    old_worker.unreachable = True
    if case == "new_generation":
        state.instance.restart_count += 1
    with patch(
        "gpustack.server.controllers.ModelInstanceService.update", AsyncMock()
    ) as update:
        await WorkerController(MagicMock())._mark_managed_instance_unreachable(
            state.session, hint, old_worker, deleted=case == "stale_delete"
        )
    update.assert_not_awaited()
    state.session.commit.assert_awaited_once()
    assert state.instance.state == ModelInstanceStateEnum.RUNNING
    assert state.session.get.call_args_list[0].kwargs == {
        "with_for_update": True,
        "populate_existing": True,
    }


@pytest.mark.parametrize("components", [False, True])
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "task_state",
    [ModelInstanceWorkloadStateEnum.FAILED, ModelInstanceWorkloadStateEnum.STOPPED],
)
async def test_worker_loss_preserves_known_failure_and_stop_confirmation(
    managed_worker, task_state, components
):
    state = managed_worker
    state.main.unreachable = True
    state.instance.state = ModelInstanceStateEnum.ERROR
    state.instance.state_message = "first rank failure"
    task = state.instance.distributed_servers.workloads[0]
    if components:
        task.dp_rank_placements = []
        task.component = ModelInstanceComponent(
            kind='sglang_engine', api_port=41000, dp_size=1, ready=True
        )
        other = state.instance.distributed_servers.workloads[1]
        other.dp_rank_placements = []
        other.component = ModelInstanceComponent(kind='sglang_engine')
    task.state = task_state
    if task.dp_rank_placements:
        task.dp_rank_placements[0].ready = False
    with patch(
        "gpustack.server.controllers.ModelInstanceService.update", AsyncMock()
    ) as update:
        await WorkerController(MagicMock())._mark_managed_instance_unreachable(
            state.session, state.instance, state.main
        )
    if components:
        update.assert_awaited_once()
        servers = update.await_args.args[1]['distributed_servers']
        assert servers.workloads[0].component.ready is False
        DistributedServers.model_validate_json(servers.model_dump_json())
    else:
        update.assert_not_awaited()
    assert task.state == task_state
    assert state.instance.state_message == "first rank failure"


@pytest.mark.asyncio
async def test_worker_ready_does_not_promote_unknown_managed_tasks(managed_worker):
    state = managed_worker
    state.instance.state = ModelInstanceStateEnum.UNREACHABLE
    state.instance.distributed_servers.workloads[1].state = (
        ModelInstanceWorkloadStateEnum.UNKNOWN
    )
    with (
        patch("gpustack.server.controllers.async_session", return_value=state.cm),
        patch.object(
            ModelInstance, "all_by_field", AsyncMock(return_value=[state.instance])
        ),
        patch(
            "gpustack.server.controllers.ModelInstanceService.update", AsyncMock()
        ) as update,
    ):
        await WorkerController(MagicMock())._reconcile(
            Event(
                type=EventType.UPDATED,
                data=state.sub,
                changed_fields={
                    "state": (WorkerStateEnum.UNREACHABLE, WorkerStateEnum.READY)
                },
            )
        )
    update.assert_not_awaited()
    assert state.instance.state == ModelInstanceStateEnum.UNREACHABLE
    assert state.instance.restart_count == 3


@pytest.mark.asyncio
async def test_repeated_worker_loss_does_not_republish_unchanged_state(managed_worker):
    state = managed_worker
    state.sub.unreachable = True

    async def apply(row, patch):
        for key, value in patch.items():
            setattr(row, key, value)

    with patch(
        "gpustack.server.controllers.ModelInstanceService.update",
        AsyncMock(side_effect=apply),
    ) as update:
        controller = WorkerController(MagicMock())
        await controller._mark_managed_instance_unreachable(
            state.session, state.instance, state.sub
        )
        await controller._mark_managed_instance_unreachable(
            state.session, state.instance, state.sub
        )
    update.assert_awaited_once()
    assert state.instance.restart_count == 3


@pytest.mark.asyncio
async def test_stale_ready_event_still_observes_current_worker_loss(managed_worker):
    state = managed_worker
    hint = state.main.model_copy(deep=True)
    state.main.unreachable = True
    with (
        patch("gpustack.server.controllers.async_session", return_value=state.cm),
        patch.object(
            ModelInstance, "all_by_field", AsyncMock(return_value=[state.instance])
        ),
        patch(
            "gpustack.server.controllers.ModelInstanceService.update", AsyncMock()
        ) as update,
    ):
        await WorkerController(MagicMock())._reconcile(
            Event(
                type=EventType.UPDATED,
                data=hint,
                changed_fields={
                    "state": (WorkerStateEnum.UNREACHABLE, WorkerStateEnum.READY)
                },
            )
        )
    update.assert_awaited_once()
    assert update.await_args.args[1]["state"] == ModelInstanceStateEnum.UNREACHABLE


@pytest.mark.asyncio
async def test_managed_maintenance_does_not_hide_a_lost_heartbeat(managed_worker):
    from datetime import timedelta

    state = managed_worker
    state.sub.state = WorkerStateEnum.MAINTENANCE
    state.sub.heartbeat_time = datetime.now(timezone.utc) - timedelta(days=1)
    with (
        patch("gpustack.server.controllers.async_session", return_value=state.cm),
        patch.object(
            ModelInstance, "all_by_field", AsyncMock(return_value=[state.instance])
        ),
        patch(
            "gpustack.server.controllers.ModelInstanceService.update", AsyncMock()
        ) as update,
    ):
        await WorkerController(MagicMock())._reconcile(
            Event(type=EventType.CREATED, data=state.sub)
        )
    update.assert_awaited_once()
    assert update.await_args.args[1]["state"] == ModelInstanceStateEnum.UNREACHABLE


@pytest.mark.asyncio
@pytest.mark.parametrize('event_type', [EventType.UPDATED, EventType.DELETED])
@pytest.mark.parametrize('recovering', [False, True])
async def test_guarded_legacy_worker_loss_cannot_retire_the_stop_barrier(
    managed_worker, event_type, recovering
):
    from uuid import uuid4
    from gpustack.schemas.models import ModelInstanceLegacyStop

    s = managed_worker
    s.instance.distributed_servers.workloads = None
    s.instance.distributed_servers.orchestration_mode = "legacy"
    if recovering:
        s.instance.state = ModelInstanceStateEnum.ERROR
        s.instance.state_message = 'engine failed'
    s.instance.distributed_servers.legacy_stop = ModelInstanceLegacyStop(
        token=uuid4(),
        restart_count=3,
        worker_ids=[s.main.id, s.sub.id],
        worker_principal_ids={s.main.id: 10, s.sub.id: 20},
    )
    before = s.instance.model_dump()
    s.main.state, s.main.unreachable = WorkerStateEnum.UNREACHABLE, True
    with (
        patch('gpustack.server.controllers.async_session', return_value=s.cm),
        patch.object(
            ModelInstance, 'all_by_field', AsyncMock(return_value=[s.instance])
        ),
        patch.object(ModelInstanceService, 'batch_delete', AsyncMock()) as remove,
        patch.object(ModelInstanceService, 'update', AsyncMock()) as update,
    ):
        await WorkerController(MagicMock())._reconcile(
            Event(
                type=event_type,
                data=s.main,
                changed_fields={'state': (None, s.main.state)},
            )
        )
    remove.assert_not_awaited()
    update.assert_not_awaited()
    assert s.instance.model_dump() == before
    s.session.exec.return_value.all.return_value = [s.instance]
    with patch.object(Worker, 'delete', AsyncMock()) as delete:
        with pytest.raises(ConflictException):
            await WorkerService(s.session).delete(s.main)
        delete.assert_not_awaited()


def test_worker_status_does_not_expose_orchestration_admission_fields():
    from gpustack.schemas.workers import WorkerStatus

    status = WorkerStatus.model_validate(
        {'orchestration': {'protocol_version': 1, 'admission_blocked': True}}
    )
    assert 'orchestration' not in status.model_dump()
    assert 'orchestration' not in WorkerStatus.model_json_schema()['properties']


@pytest.mark.parametrize('version', [None, 'unknown', '2.3.0', '2.3.1', '2.3.1-ent2'])
@pytest.mark.parametrize('maintenance', [False, True])
def test_version_mismatch_keeps_normal_state_rules(version, maintenance, monkeypatch):
    monkeypatch.setattr(
        'gpustack.extension.resolve_version_info', lambda: ('2.3.1-ent1', 'test')
    )
    worker = linux_nvidia_1_4090_24gx1()
    worker.worker_version = version
    worker.heartbeat_time = datetime.now(timezone.utc)
    worker.unreachable = False
    worker.maintenance = Maintenance(enabled=maintenance, message='planned')
    worker.state, worker.state_message = WorkerStateEnum.READY, None
    worker.compute_state()
    assert worker.state == (
        WorkerStateEnum.MAINTENANCE if maintenance else WorkerStateEnum.READY
    )
    assert worker.state_message == ('planned' if maintenance else None)


@pytest.mark.parametrize('version', [None, 'unknown', '2.3.0', '2.3.1', '2.3.1-ent2'])
@pytest.mark.parametrize('bound', [None, 11])
def test_managed_worker_requires_identity_regardless_of_version(version, bound):
    from gpustack.utils.model_instance_workers import (
        worker_supports_model_instance_workloads,
    )

    worker = linux_nvidia_1_4090_24gx1()
    worker.system_principal_id = bound
    worker.worker_version = version
    assert worker_supports_model_instance_workloads(worker) is bool(bound)


def test_managed_protocol_cohort_distinguishes_unassigned_pending_and_assigned_router():
    from gpustack.utils.model_instance_workers import (
        model_requires_managed_worker_binding,
        assigned_managed_workload_worker_ids,
    )

    pending = new_model_instance(1, 'engine', model_id=1)
    pending.worker_id = None
    pending.distributed_servers = DistributedServers(workloads=[])
    router = new_model_instance(2, 'router', model_id=1, worker_id=7)
    legacy = new_model_instance(3, 'legacy', model_id=2, worker_id=8)
    assert model_requires_managed_worker_binding([pending, router, legacy], 1)
    assert not model_requires_managed_worker_binding([pending, router, legacy], 2)
    assert assigned_managed_workload_worker_ids([pending, router, legacy]) == set()
    pending.worker_id = 9
    pending.distributed_servers.subordinate_workers = [
        ModelInstanceSubordinateWorker(worker_id=10)
    ]
    assert assigned_managed_workload_worker_ids([pending, router, legacy]) == {7, 9, 10}


@pytest.mark.parametrize('maintenance', [False, True])
def test_matching_worker_version_keeps_normal_state_rules(maintenance, monkeypatch):
    monkeypatch.setattr(
        'gpustack.extension.resolve_version_info', lambda: ('2.3.1', 'test')
    )
    worker = linux_nvidia_1_4090_24gx1()
    worker.worker_version = '2.3.1'
    worker.state, worker.state_message = WorkerStateEnum.READY, None
    worker.heartbeat_time = datetime.now(timezone.utc)
    worker.unreachable = False
    worker.maintenance = Maintenance(enabled=maintenance, message='planned')
    worker.compute_state()
    assert worker.state == (
        WorkerStateEnum.MAINTENANCE if maintenance else WorkerStateEnum.READY
    )


@pytest.mark.parametrize(
    'state', [state for state in WorkerStateEnum if state.is_provisioning]
)
def test_worker_provisioning_states_ignore_version(state, monkeypatch):
    monkeypatch.setattr(
        'gpustack.extension.resolve_version_info', lambda: ('2.3.1', 'test')
    )
    worker = linux_nvidia_1_4090_24gx1()
    worker.worker_version = None
    worker.state, worker.state_message = state, 'provisioning'
    worker.compute_state()
    assert worker.state == state
    assert worker.state_message == 'provisioning'


def test_assigned_legacy_stop_requires_the_complete_model_cohort():
    from uuid import uuid4
    from gpustack.schemas.models import ModelInstanceLegacyStop
    from gpustack.utils.model_instance_workers import (
        assigned_managed_workload_worker_ids,
        model_requires_managed_worker_binding,
    )

    engine = new_model_instance(1, 'engine', model_id=1, worker_id=7)
    engine.distributed_servers = DistributedServers(
        legacy_stop=ModelInstanceLegacyStop(
            token=uuid4(), restart_count=0, worker_ids=[7], worker_principal_ids={7: 11}
        )
    )
    router = new_model_instance(2, 'router', model_id=1, worker_id=8)
    unrelated = new_model_instance(3, 'legacy', model_id=2, worker_id=9)
    assert model_requires_managed_worker_binding([engine, router], 1)
    assert assigned_managed_workload_worker_ids([engine, router, unrelated]) == {7, 8}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'changed_identity',
    [None, 'ip', 'cluster_id', 'system_principal_id', 'token', 'proxy_address'],
)
async def test_worker_sync_ignores_late_identity_and_preserves_registered_version(
    managed_worker, changed_identity, monkeypatch
):
    monkeypatch.setattr(
        'gpustack.extension.resolve_version_info', lambda: ('2.3.1', 'test')
    )

    s = managed_worker
    probed = s.main
    probed.unreachable = True
    probed.state = WorkerStateEnum.UNREACHABLE
    stored = probed.model_copy(deep=True)
    stored.worker_version = '2.3.0'
    stored.state = WorkerStateEnum.NOT_READY
    stored.state_message = None
    if changed_identity is not None:
        value = getattr(stored, changed_identity)
        setattr(
            stored,
            changed_identity,
            (
                (value or 0) + 1
                if changed_identity in ('cluster_id', 'system_principal_id')
                else (
                    'http://new-registration'
                    if changed_identity == 'proxy_address'
                    else 'new-registration'
                )
            ),
        )

    async def probe(worker):
        worker.unreachable = False

    syncer = WorkerSyncer(lambda: MagicMock(), lambda: MagicMock())
    with (
        patch('gpustack.server.worker_syncer.async_session', return_value=s.cm),
        patch.object(Worker, 'all', AsyncMock(return_value=[probed])),
        patch.object(Worker, 'all_by_ids', AsyncMock(return_value=[stored])) as locked,
        patch.object(
            WorkerService,
            'get_by_id',
            AsyncMock(side_effect=AssertionError('cache is not authority')),
        ),
        patch.object(
            WorkerService,
            'update',
            AsyncMock(side_effect=AssertionError('must not commit per Worker')),
        ),
        patch.object(WorkerService, 'batch_update', AsyncMock()) as persist,
        patch.object(syncer, '_should_check_unreachable', return_value=True),
        patch.object(syncer, '_set_worker_unreachable', AsyncMock(side_effect=probe)),
    ):
        await syncer._sync_workers_states()
    locked.assert_awaited_once()
    if changed_identity is None:
        assert persist.await_args.args[0] == [stored]
        assert not stored.unreachable
        assert stored.state == WorkerStateEnum.READY
        assert stored.worker_version == '2.3.0'
    else:
        assert persist.await_args.args[0] == []
        assert stored.unreachable


@pytest.mark.asyncio
async def test_worker_sync_locks_and_changes_whole_batch_before_single_save(
    managed_worker,
):
    s = managed_worker
    probed = [s.main, s.sub]
    for worker in probed:
        worker.state = WorkerStateEnum.UNREACHABLE
        worker.unreachable = True
    current = [w.model_copy(deep=True) for w in probed]

    async def probe(worker):
        worker.unreachable = False

    async def save(service_workers):
        assert service_workers == current
        assert all(
            not w.unreachable and w.state == WorkerStateEnum.READY for w in current
        )

    syncer = WorkerSyncer(lambda: MagicMock(), lambda: MagicMock())
    with (
        patch('gpustack.server.worker_syncer.async_session', return_value=s.cm),
        patch.object(Worker, 'all', AsyncMock(return_value=probed)),
        patch.object(Worker, 'all_by_ids', AsyncMock(return_value=current)) as locked,
        patch.object(
            WorkerService,
            'update',
            AsyncMock(side_effect=AssertionError('per-row save releases batch locks')),
        ),
        patch.object(
            WorkerService, 'batch_update', AsyncMock(side_effect=save)
        ) as saved,
        patch.object(syncer, '_should_check_unreachable', return_value=True),
        patch.object(syncer, '_set_worker_unreachable', AsyncMock(side_effect=probe)),
    ):
        await syncer._sync_workers_states()
    locked.assert_awaited_once()
    assert locked.await_args.kwargs['for_update']
    saved.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'current_state',
    [state for state in WorkerStateEnum if state.is_provisioning],
)
async def test_worker_sync_preserves_concurrent_provisioning_states(
    managed_worker, current_state
):
    s = managed_worker
    probed = [s.main, s.sub]
    for worker in probed:
        worker.state = WorkerStateEnum.UNREACHABLE
        worker.unreachable = True
    current = [worker.model_copy(deep=True) for worker in probed]
    current[0].state = current_state
    current[0].state_message = 'Provisioning controller owns this state'

    async def probe(worker):
        worker.unreachable = False

    syncer = WorkerSyncer(lambda: MagicMock(), lambda: MagicMock())
    with (
        patch('gpustack.server.worker_syncer.async_session', return_value=s.cm),
        patch.object(Worker, 'all', AsyncMock(return_value=probed)),
        patch.object(Worker, 'all_by_ids', AsyncMock(return_value=current)) as locked,
        patch.object(WorkerService, 'batch_update', AsyncMock()) as saved,
        patch.object(syncer, '_should_check_unreachable', return_value=True),
        patch.object(syncer, '_set_worker_unreachable', AsyncMock(side_effect=probe)),
    ):
        await syncer._sync_workers_states()
    assert locked.await_args.kwargs['for_update']
    assert current[0].state == current_state
    assert current[0].state_message == 'Provisioning controller owns this state'
    assert saved.await_args.args[0] == [current[1]]
    assert current[0].unreachable
    assert current[1].state == WorkerStateEnum.READY
    assert not current[1].unreachable


@pytest.mark.asyncio
@pytest.mark.parametrize('stopped', [False, True])
async def test_mixed_router_worker_deletion_requires_its_own_stop_proof(
    managed_worker, stopped
):
    from uuid import uuid4
    from gpustack.schemas.models import ModelInstanceLegacyStop

    s = managed_worker
    engine = s.instance
    engine.distributed_servers.subordinate_workers = []
    engine.distributed_servers.workloads = [engine.distributed_servers.workloads[0]]
    engine.distributed_servers.orchestration_mode = "managed"
    router = new_model_instance(
        2, 'router', model_id=engine.model_id, worker_id=s.sub.id
    )
    if stopped:
        router.distributed_servers = DistributedServers(
            legacy_stop=ModelInstanceLegacyStop(
                token=uuid4(),
                restart_count=0,
                worker_ids=[s.sub.id],
                worker_principal_ids={s.sub.id: 30},
                confirmed_worker_ids=[s.sub.id],
            )
        )
    with (
        patch.object(
            ModelInstance,
            'all_by_field',
            AsyncMock(return_value=[engine, router]),
        ),
        patch.object(Worker, 'delete', AsyncMock()) as deleted,
        patch('gpustack.server.services.delete_cache_by_key', AsyncMock()),
    ):
        if stopped:
            await WorkerService(s.session).delete(s.sub)
            deleted.assert_awaited_once()
        else:
            with pytest.raises(ConflictException):
                await WorkerService(s.session).delete(s.sub)
            deleted.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('batch', [False, True])
async def test_worker_cache_keys_invalidate_old_and_new_cluster_name(batch):
    from sqlalchemy.orm.attributes import set_committed_value
    from gpustack.schemas.workers import WorkerStatus

    worker = Worker(
        id=7,
        name='old',
        hostname='host',
        ip='10.0.0.1',
        cluster_id=1,
        status=WorkerStatus(),
    )
    set_committed_value(worker, 'name', 'old')
    set_committed_value(worker, 'cluster_id', 1)
    service = WorkerService(MagicMock())

    async def update(_session, source, **kwargs):
        for key, value in source.items():
            setattr(worker, key, value)

    with patch(
        'gpustack.server.services.delete_cache_by_key', AsyncMock()
    ) as invalidated:
        if batch:
            worker.name, worker.cluster_id = 'new', 2
            with patch.object(Worker, 'batch_update', AsyncMock(return_value=1)):
                await service.batch_update([worker])
        else:
            with patch.object(Worker, 'update', AsyncMock(side_effect=update)):
                await service.update(worker, {'name': 'new', 'cluster_id': 2})
    keys = {
        call.args[1:]
        for call in invalidated.await_args_list
        if call.args[0].__name__ == 'get_by_cluster_id_name'
    }
    assert keys == {(1, 'old'), (2, 'new')}


@pytest.mark.asyncio
@pytest.mark.parametrize('soft', [False, True])
async def test_worker_delete_locks_worker_before_instances_and_invalidates_after_delete(
    managed_worker,
    soft,
):
    s = managed_worker
    order = []

    async def lock_worker(session, worker_id, **kwargs):
        assert session is s.session and worker_id == s.main.id
        assert kwargs['for_update'] is True
        assert kwargs['options'][0].path[1].key == 'system_principal'
        order.append('worker')
        return s.main

    async def lock_instances(session, field, value, **kwargs):
        assert order == ['worker']
        assert (session, field, value) == (s.session, 'cluster_id', s.main.cluster_id)
        assert kwargs == {'for_update': True, 'refresh': True}
        order.append('instances')
        return []

    async def delete(*args, **kwargs):
        order.append('delete')

    async def invalidate(*args, **kwargs):
        assert 'delete' in order
        order.append('cache')

    with (
        patch.object(Worker, 'one_by_id', side_effect=lock_worker),
        patch.object(ModelInstance, 'all_by_field', side_effect=lock_instances),
        patch.object(Worker, 'delete', side_effect=delete),
        patch(
            'gpustack.server.services.delete_cache_by_key', side_effect=invalidate
        ) as invalidated,
    ):
        await WorkerService(s.session).delete(s.main, soft=soft)
    assert order == ['worker', 'instances', 'delete', 'cache', 'cache', 'cache']
    key_calls = [
        call
        for call in invalidated.await_args_list
        if call.args[0].__name__ == 'get_by_cluster_id_name'
    ]
    assert len(key_calls) == 1
    assert key_calls[0].args[1:] == (s.main.cluster_id, s.main.name)
    assert key_calls[0].kwargs == {'sync_coordinator': False}


@pytest.mark.asyncio
@pytest.mark.parametrize('field', ['ip', 'unreachable', 'proxy_address'])
@pytest.mark.parametrize('matched', [False, True])
async def test_worker_notifications_load_models_only_for_matched_instances(
    managed_worker, field, matched
):
    s = managed_worker
    instance = s.instance
    model = new_model(instance.model_id, 'affected', huggingface_repo_id='org/model')
    unrelated = new_model_instance(99, 'unrelated', 999, worker_id=99)
    duplicate = new_model_instance(2, 'same-model', model.id, worker_id=s.sub.id)
    instances = [unrelated] + ([instance, duplicate] if matched else [])
    with (
        patch('gpustack.server.controllers.async_session', return_value=s.cm),
        patch.object(
            ModelInstance, 'all_by_fields', AsyncMock(return_value=instances)
        ) as rows,
        patch.object(Model, 'all_by_fields', AsyncMock(return_value=[model])) as models,
        patch('gpustack.server.controllers.event_bus.publish', AsyncMock()) as publish,
        patch(
            'gpustack.server.controllers.notify_model_route_target', AsyncMock()
        ) as notify,
    ):
        await WorkerController(MagicMock())._notify_relatives(
            Event(
                type=EventType.UPDATED,
                data=s.sub,
                changed_fields={field: (None, 'changed')},
            )
        )
    assert not rows.await_args.kwargs.get('options')
    assert models.await_count == int(matched)
    if matched:
        condition = models.await_args.kwargs['extra_conditions'][0]
        assert list(condition.compile().params.values()) == [[model.id]]
    assert publish.await_count == int(matched)
    assert notify.await_count == int(matched)


@pytest.mark.asyncio
async def test_worker_delete_failure_does_not_invalidate_name_caches(managed_worker):
    s = managed_worker
    with (
        patch.object(ModelInstance, 'all_by_field', AsyncMock(return_value=[])),
        patch.object(
            Worker, 'delete', AsyncMock(side_effect=RuntimeError('commit failed'))
        ),
        patch(
            'gpustack.server.services.delete_cache_by_key', AsyncMock()
        ) as invalidated,
    ):
        with pytest.raises(RuntimeError, match='commit failed'):
            await WorkerService(s.session).delete(s.main)
    invalidated.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('components', [False, True])
async def test_worker_ready_rechecks_tasks_that_reported_before_the_worker(
    managed_worker, components
):
    state = managed_worker
    state.instance.state = ModelInstanceStateEnum.UNREACHABLE
    if components:
        state.instance.distributed_servers.workloads = [
            ModelInstanceWorkload(
                name=f'engine-{worker.id}',
                worker_id=worker.id,
                state='running',
                component=dict(
                    kind='sglang_engine',
                    **(
                        dict(api_port=41000, dp_size=2, ready=True)
                        if worker.id == state.main.id
                        else {}
                    ),
                ),
            )
            for worker in (state.main, state.sub)
        ] + [
            ModelInstanceWorkload(
                name='smg',
                worker_id=state.main.id,
                state='running',
                component=dict(
                    kind='sglang_smg', api_port=41002, metrics_port=41003, ready=True
                ),
            )
        ]
        state.instance.distributed_servers.orchestration_mode = (
            "managed"
            if state.instance.distributed_servers.workloads is not None
            else "legacy"
        )
    state.sub.state = WorkerStateEnum.UNREACHABLE
    state.sub.unreachable = True

    async def apply(instance, source):
        for key, value in source.items():
            setattr(instance, key, value)

    with (
        patch('gpustack.server.controllers.async_session', return_value=state.cm),
        patch.object(
            Worker,
            'all_by_ids',
            AsyncMock(return_value=[state.main, state.sub]),
        ),
        patch.object(
            ModelInstance, 'all_by_field', AsyncMock(return_value=[state.instance])
        ),
        patch.object(ModelInstanceService, 'update', AsyncMock(side_effect=apply)),
    ):
        await ModelInstanceService(state.session).reconcile_managed_instance(
            state.instance.id, 3, allow_restart=False
        )
        assert state.instance.state == ModelInstanceStateEnum.UNREACHABLE
        state.sub.state, state.sub.unreachable = WorkerStateEnum.READY, False
        await WorkerController(MagicMock())._reconcile(
            Event(
                type=EventType.UPDATED,
                data=state.sub,
                changed_fields={
                    'state': (WorkerStateEnum.UNREACHABLE, WorkerStateEnum.READY)
                },
            )
        )
    assert state.instance.state == ModelInstanceStateEnum.RUNNING
    assert state.instance.restart_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize('worker_id', [1, 2, 3, 4])
async def test_hybrid_worker_loss_withdraws_four_node_group_without_confirming_stop(
    managed_worker, worker_id
):
    from gpustack.utils.model_instance_workers import model_instance_endpoints
    from tests.utils.model import new_hybrid_workload

    state = managed_worker
    instance = state.instance
    instance.worker_id = 1
    instance.worker_ip = '10.0.0.1'
    instance.distributed_servers.subordinate_workers = [
        ModelInstanceSubordinateWorker(
            worker_id=w, worker_ip=f'10.0.0.{w}', gpu_indexes=[0]
        )
        for w in (2, 3, 4)
    ]
    instance.distributed_servers.workloads = [
        ModelInstanceWorkload.model_validate(
            new_hybrid_workload(w, headless=w in (2, 4), state='running')
        )
        for w in (1, 2, 3, 4)
    ]
    instance.distributed_servers.orchestration_mode = "managed"
    for task in instance.distributed_servers.workloads:
        if task.component.api_port:
            task.component.dp_size, task.component.ready = 1, True
    assert len(model_instance_endpoints(instance)) == 2
    lost = SimpleNamespace(
        id=worker_id,
        cluster_id=instance.cluster_id,
        deleted_at=None,
        unreachable=True,
        state=WorkerStateEnum.UNREACHABLE,
    )
    other = [
        w.model_dump()
        for w in instance.distributed_servers.workloads
        if w.worker_id != worker_id
    ]
    with (
        patch.object(Worker, 'one_by_id', AsyncMock(return_value=lost)),
        patch.object(ModelInstance, 'one_by_id', AsyncMock(return_value=instance)),
        patch.object(ModelInstanceService, 'update', AsyncMock()) as update,
    ):
        await WorkerController(MagicMock())._mark_managed_instance_unreachable(
            state.session, instance, lost
        )
    patch_dict = update.await_args.args[1]
    updated = instance.model_copy(deep=True, update=patch_dict)
    assert updated.state == ModelInstanceStateEnum.UNREACHABLE
    assert updated.restart_count == 3
    assert model_instance_endpoints(updated) == []
    local = next(
        w for w in updated.distributed_servers.workloads if w.worker_id == worker_id
    )
    assert local.state == ModelInstanceWorkloadStateEnum.UNKNOWN
    assert local.component.ready is (False if local.component.api_port else None)
    assert [
        w.model_dump()
        for w in updated.distributed_servers.workloads
        if w.worker_id != worker_id
    ] == other
    assert updated.computed_resource_claim == instance.computed_resource_claim
