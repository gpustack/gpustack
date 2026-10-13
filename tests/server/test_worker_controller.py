from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gpustack.schemas.cache_services import CacheServiceStateEnum
from gpustack.schemas.models import (
    DistributedServerCoordinateModeEnum,
    DistributedServers,
    ModelInstanceSubordinateWorker,
    ModelInstanceStateEnum,
)
from gpustack.schemas.workers import WorkerStateEnum
from gpustack.server.bus import Event, EventType, Subscriber
from gpustack.server.controllers import WorkerController
from tests.fixtures.workers.fixtures import (
    linux_nvidia_1_4090_24gx1,
    linux_nvidia_2_4080_16gx2,
)
from tests.utils.mock import mock_async_session
from tests.utils.model import new_model_instance


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
