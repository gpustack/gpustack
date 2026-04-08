from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from gpustack.schemas.models import (
    DistributedServerCoordinateModeEnum,
    DistributedServers,
    ModelInstanceSubordinateWorker,
    ModelInstanceStateEnum,
)
from gpustack.schemas.workers import WorkerStateEnum
from gpustack.server.worker_instance_cleaner import WorkerInstanceCleaner
from tests.fixtures.workers.fixtures import (
    linux_nvidia_1_4090_24gx1,
    linux_nvidia_2_4080_16gx2,
)
from tests.utils.mock import mock_async_session
from tests.utils.model import new_model_instance


def _offline_worker(worker, *, heartbeat_age_seconds: int = 600):
    worker.state = WorkerStateEnum.NOT_READY
    worker.maintenance = None
    worker.heartbeat_time = datetime.now(timezone.utc) - timedelta(
        seconds=heartbeat_age_seconds
    )
    return worker


@pytest.mark.asyncio
async def test_cleanup_offline_worker_instances_deletes_main_worker_instances():
    offline_worker = _offline_worker(linux_nvidia_1_4090_24gx1())
    other_worker = linux_nvidia_2_4080_16gx2()

    instance = new_model_instance(
        1,
        "main-worker-instance",
        1,
        worker_id=offline_worker.id,
        state=ModelInstanceStateEnum.RUNNING,
    )
    instance.worker_name = offline_worker.name

    batch_delete = AsyncMock(return_value=[instance.name])

    with (
        patch(
            "gpustack.server.worker_instance_cleaner.async_session",
            return_value=mock_async_session(),
        ),
        patch(
            "gpustack.server.worker_instance_cleaner.Worker.all",
            AsyncMock(return_value=[offline_worker, other_worker]),
        ),
        patch(
            "gpustack.server.worker_instance_cleaner.ModelInstance.all",
            AsyncMock(return_value=[instance]),
        ),
        patch(
            "gpustack.server.worker_instance_cleaner.ModelInstanceService.batch_delete",
            batch_delete,
        ),
        patch(
            "gpustack.server.worker_instance_cleaner.envs.MODEL_INSTANCE_RESCHEDULE_GRACE_PERIOD",
            300,
        ),
    ):
        cleaner = WorkerInstanceCleaner()
        await cleaner._cleanup_offline_worker_instances()

    batch_delete.assert_awaited_once()
    deleted_instances = batch_delete.await_args.args[-1]
    assert [item.name for item in deleted_instances] == [instance.name]


@pytest.mark.asyncio
async def test_cleanup_offline_worker_instances_deletes_distributed_instances_with_offline_subordinate():
    main_worker = linux_nvidia_1_4090_24gx1()
    offline_subordinate = _offline_worker(linux_nvidia_2_4080_16gx2())

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
                worker_id=offline_subordinate.id,
                worker_name=offline_subordinate.name,
                worker_ip=offline_subordinate.ip,
                state=ModelInstanceStateEnum.RUNNING,
            )
        ],
    )

    batch_delete = AsyncMock(return_value=[instance.name])

    with (
        patch(
            "gpustack.server.worker_instance_cleaner.async_session",
            return_value=mock_async_session(),
        ),
        patch(
            "gpustack.server.worker_instance_cleaner.Worker.all",
            AsyncMock(return_value=[main_worker, offline_subordinate]),
        ),
        patch(
            "gpustack.server.worker_instance_cleaner.ModelInstance.all",
            AsyncMock(return_value=[instance]),
        ),
        patch(
            "gpustack.server.worker_instance_cleaner.ModelInstanceService.batch_delete",
            batch_delete,
        ),
        patch(
            "gpustack.server.worker_instance_cleaner.envs.MODEL_INSTANCE_RESCHEDULE_GRACE_PERIOD",
            300,
        ),
    ):
        cleaner = WorkerInstanceCleaner()
        await cleaner._cleanup_offline_worker_instances()

    batch_delete.assert_awaited_once()
    deleted_instances = batch_delete.await_args.args[-1]
    assert [item.name for item in deleted_instances] == [instance.name]
