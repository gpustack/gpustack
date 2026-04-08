from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gpustack.schemas.models import (
    DistributedServerCoordinateModeEnum,
    DistributedServers,
    ModelInstanceSubordinateWorker,
    ModelInstanceStateEnum,
)
from gpustack.schemas.workers import WorkerStateEnum
from gpustack.server.bus import Event, EventType
from gpustack.server.controllers import WorkerController
from tests.fixtures.workers.fixtures import (
    linux_nvidia_1_4090_24gx1,
    linux_nvidia_2_4080_16gx2,
)
from tests.utils.mock import mock_async_session
from tests.utils.model import new_model_instance


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
