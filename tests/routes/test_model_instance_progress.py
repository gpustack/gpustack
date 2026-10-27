from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gpustack.api.exceptions import NotFoundException
from gpustack.api.tenant import TenantContext
from gpustack.routes.model_instances import update_model_instance
from gpustack.schemas.model_files import ModelFileStateEnum
from gpustack.schemas.models import (
    DistributedServers,
    ModelInstance,
    ModelInstanceStateEnum,
    ModelInstanceSubordinateWorker,
    ModelInstanceUpdate,
    SourceEnum,
)
from gpustack.server.controllers import (
    sync_instance_files_state,
    sync_main_worker_model_file_state,
)
from tests.server.test_model_file_state_sync import _distributed_instance, _patched
from tests.utils.model import new_model, new_model_instance


async def _apply_report(instance, report):
    session = MagicMock(info={})

    async def persist(current, source):
        await current.update(session, source)

    service = MagicMock()
    service.update = AsyncMock(side_effect=persist)
    ctx = TenantContext(
        user=MagicMock(),
        is_platform_admin=True,
        current_principal_id=None,
        org_role=None,
    )
    with (
        patch.object(ModelInstance, "one_by_id", AsyncMock(return_value=instance)),
        patch.object(ModelInstance, "save", AsyncMock()),
        patch(
            "gpustack.routes.model_instances.ModelInstanceService", return_value=service
        ),
    ):
        await update_model_instance(session, ctx, instance.id, report)


@pytest.mark.asyncio
@pytest.mark.parametrize("progress", [0, 99, 100])
async def test_state_report_preserves_controller_download_progress(progress):
    instance = new_model_instance(
        13, "qwen3-8b", 2, worker_id=2, state=ModelInstanceStateEnum.DOWNLOADING
    )
    instance.source = SourceEnum.HUGGING_FACE
    instance.huggingface_repo_id = "Qwen/Qwen3-8B"
    instance.download_progress = 97.95
    instance.draft_model_download_progress = 79
    instance.distributed_servers = DistributedServers(
        subordinate_workers=[
            ModelInstanceSubordinateWorker(worker_id=1, download_progress=87.03),
            ModelInstanceSubordinateWorker(worker_id=3, download_progress=50),
        ]
    )
    report = ModelInstanceUpdate.model_validate(instance.model_dump())
    report.distributed_servers.subordinate_workers.reverse()
    subordinate = report.distributed_servers.subordinate_workers[1]
    subordinate.state = ModelInstanceStateEnum.ERROR
    subordinate.state_message = "Inference server exited or unhealthy."
    instance.download_progress = progress
    instance.draft_model_download_progress = progress
    instance.distributed_servers.subordinate_workers[0].download_progress = progress
    instance.distributed_servers.subordinate_workers[1].download_progress = 60

    session = MagicMock()
    ctx = TenantContext(
        user=MagicMock(),
        is_platform_admin=True,
        current_principal_id=None,
        org_role=None,
    )
    service = MagicMock()
    service.update = AsyncMock()
    with (
        patch.object(
            ModelInstance, "one_by_id", AsyncMock(return_value=instance)
        ) as get,
        patch(
            "gpustack.routes.model_instances.ModelInstanceService", return_value=service
        ),
    ):
        await update_model_instance(session, ctx, instance.id, report)

    get.assert_awaited_once_with(session, instance.id, for_update=True)
    service.update.assert_awaited_once()
    updated = service.update.call_args.args[1]
    assert updated.download_progress == progress
    assert updated.draft_model_download_progress == progress
    assert updated.distributed_servers.subordinate_workers[0].download_progress == 60
    assert (
        updated.distributed_servers.subordinate_workers[1].download_progress == progress
    )
    assert updated.distributed_servers.subordinate_workers[1].state == (
        ModelInstanceStateEnum.ERROR
    )


@pytest.mark.asyncio
async def test_instance_update_rejects_another_tenants_instance():
    instance = new_model_instance(13, "qwen3-8b", 2, worker_id=2)
    instance.source = SourceEnum.HUGGING_FACE
    instance.huggingface_repo_id = "Qwen/Qwen3-8B"
    instance.owner_principal_id = 1
    report = ModelInstanceUpdate.model_validate(instance.model_dump())
    ctx = TenantContext(
        user=MagicMock(),
        is_platform_admin=False,
        current_principal_id=2,
        org_role=None,
    )
    with (
        patch.object(ModelInstance, "one_by_id", AsyncMock(return_value=instance)),
        patch("gpustack.routes.model_instances.ModelInstanceService") as service,
        pytest.raises(NotFoundException),
    ):
        await update_model_instance(MagicMock(), ctx, instance.id, report)
    service.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "snapshot_state",
    [ModelInstanceStateEnum.INITIALIZING, ModelInstanceStateEnum.DOWNLOADING],
)
@pytest.mark.parametrize(
    "current_state", [ModelInstanceStateEnum.STARTING, ModelInstanceStateEnum.RUNNING]
)
async def test_stale_put_preserves_download_completion(snapshot_state, current_state):
    instance = _distributed_instance(sub_download_progress=100)
    instance.source = SourceEnum.HUGGING_FACE
    instance.huggingface_repo_id = "Qwen/Qwen3-0.6B"
    instance.state = snapshot_state
    report = ModelInstanceUpdate.model_validate(instance.model_dump())
    report.pid = 42
    report.distributed_servers.subordinate_workers[0].state = (
        ModelInstanceStateEnum.RUNNING
    )

    model = new_model(1, "test", huggingface_repo_id="Qwen/Qwen3-0.6B")
    session = MagicMock(info={})
    with _patched(instance, model):
        await sync_main_worker_model_file_state(
            session, instance.model_files[0], instance
        )
    assert instance.state == ModelInstanceStateEnum.STARTING
    instance.state = current_state
    instance.state_message = "Current status"
    await _apply_report(instance, report)

    assert instance.state == current_state
    assert instance.state_message == "Current status"
    assert instance.download_progress == 100
    assert instance.resolved_path == "/cache/main"
    assert instance.mounted_loras == []
    assert instance.pid == 42
    assert instance.distributed_servers.subordinate_workers[0].state == (
        ModelInstanceStateEnum.RUNNING
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "snapshot_state",
    [ModelInstanceStateEnum.INITIALIZING, ModelInstanceStateEnum.DOWNLOADING],
)
@pytest.mark.parametrize("failed_file_index", [0, 1], ids=["main", "subordinate"])
async def test_stale_put_preserves_download_failure(snapshot_state, failed_file_index):
    instance = _distributed_instance(sub_download_progress=79)
    instance.source = SourceEnum.HUGGING_FACE
    instance.huggingface_repo_id = "Qwen/Qwen3-0.6B"
    instance.state = snapshot_state
    report = ModelInstanceUpdate.model_validate(instance.model_dump())
    report.pid = 42

    failed_file = instance.model_files[failed_file_index]
    failed_file.state = ModelFileStateEnum.ERROR
    failed_file.state_message = "Download failed: insufficient disk space"
    with _patched(instance):
        await sync_instance_files_state(MagicMock(), instance, [failed_file])
    assert instance.state == ModelInstanceStateEnum.ERROR

    await _apply_report(instance, report)

    assert instance.state == ModelInstanceStateEnum.ERROR
    assert instance.state_message == failed_file.state_message
    assert instance.pid == 42


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "current_state, reported_state",
    [
        (ModelInstanceStateEnum.STARTING, ModelInstanceStateEnum.RUNNING),
        (ModelInstanceStateEnum.STARTING, ModelInstanceStateEnum.ERROR),
        (ModelInstanceStateEnum.RUNNING, ModelInstanceStateEnum.UNREACHABLE),
        (ModelInstanceStateEnum.ERROR, ModelInstanceStateEnum.SCHEDULED),
        (ModelInstanceStateEnum.SCHEDULED, ModelInstanceStateEnum.INITIALIZING),
    ],
)
async def test_worker_runtime_and_restart_transitions_are_allowed(
    current_state, reported_state
):
    instance = new_model_instance(1, "test", 1, state=current_state)
    instance.source = SourceEnum.HUGGING_FACE
    instance.huggingface_repo_id = "Qwen/Qwen3-0.6B"
    report = ModelInstanceUpdate.model_validate(instance.model_dump())
    report.state = reported_state
    report.state_message = "Worker status"

    await _apply_report(instance, report)

    assert instance.state == reported_state
    assert instance.state_message == "Worker status"
