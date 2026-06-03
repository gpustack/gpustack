"""Regression tests for distributed download-progress → STARTING promotion.

Guards the fix for issue #5445 ("Multi-node distributed deployment struck in 0
download progress"): completion must be decided by ModelFile.state alone, not by
the display-only subordinate_workers[].download_progress mirror, and a subordinate
READY event must re-check completion even when its progress already reached 100.
"""

import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gpustack.schemas.model_files import ModelFile, ModelFileStateEnum
from gpustack.schemas.models import (
    DistributedServers,
    ModelInstance,
    ModelInstanceStateEnum,
    ModelInstanceSubordinateWorker,
    SourceEnum,
)
from gpustack.server.controllers import (
    sync_distributed_model_file_state,
    sync_main_worker_model_file_state,
)
from tests.utils.model import new_model_instance

MAIN_WORKER_ID = 1
SUB_WORKER_ID = 2


def _model_file(
    id: int,
    worker_id: int,
    state: ModelFileStateEnum,
    resolved_paths,
) -> ModelFile:
    return ModelFile(
        id=id,
        worker_id=worker_id,
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="Qwen/Qwen3-0.6B",
        state=state,
        download_progress=100,
        resolved_paths=resolved_paths,
        is_lora=False,
    )


def _distributed_instance(sub_download_progress: float) -> ModelInstance:
    instance = new_model_instance(
        1,
        "distributed-instance",
        1,
        worker_id=MAIN_WORKER_ID,
        state=ModelInstanceStateEnum.DOWNLOADING,
    )
    instance.download_progress = 0
    instance.distributed_servers = DistributedServers(
        download_model_files=True,
        subordinate_workers=[
            ModelInstanceSubordinateWorker(
                worker_id=SUB_WORKER_ID,
                download_progress=sub_download_progress,
            )
        ],
    )
    # Both workers' files are physically READY — the single source of truth.
    instance.model_files = [
        _model_file(10, MAIN_WORKER_ID, ModelFileStateEnum.READY, ["/cache/main"]),
        _model_file(20, SUB_WORKER_ID, ModelFileStateEnum.READY, ["/cache/sub"]),
    ]
    return instance


@contextlib.contextmanager
def _patched(instance: ModelInstance):
    """Patch every DB touchpoint so the sync runs in-memory, while letting
    model_instance_download_completed execute for real against instance.model_files."""
    service = MagicMock()
    service.return_value.update = AsyncMock()
    patches = [
        patch.object(
            ModelInstance,
            "one_by_id_with_model_files",
            AsyncMock(return_value=instance),
        ),
        patch.object(ModelInstance, "one_by_id", AsyncMock(return_value=instance)),
        patch("gpustack.server.controllers.ModelInstanceService", service),
        patch(
            "gpustack.server.controllers._build_mounted_loras_payload",
            AsyncMock(return_value=([], [])),
        ),
        patch("gpustack.server.controllers.flag_modified", MagicMock()),
        # logger.trace is a custom level not registered in the test runtime.
        patch("gpustack.server.controllers.logger", MagicMock()),
    ]
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        yield


@pytest.mark.asyncio
async def test_main_worker_ready_promotes_despite_lagging_subordinate_progress():
    """All ModelFiles are READY but the subordinate's display progress is stuck
    below 100. The final main-worker READY event must still promote to STARTING
    (the old code gated completion on subordinate_workers[].download_progress)."""
    instance = _distributed_instance(sub_download_progress=50)
    main_file = instance.model_files[0]

    with _patched(instance):
        await sync_main_worker_model_file_state(MagicMock(), main_file, instance)

    assert instance.state == ModelInstanceStateEnum.STARTING
    assert instance.download_progress == 100


@pytest.mark.asyncio
async def test_subordinate_ready_promotes_when_progress_already_100():
    """The subordinate's last DOWNLOADING report already pushed progress to 100,
    so the READY event finds progress == 100. It must still re-check completion
    and promote to STARTING (the old `!= 100` gate skipped this entirely)."""
    instance = _distributed_instance(sub_download_progress=100)
    sub_file = instance.model_files[1]  # worker_id == SUB_WORKER_ID, READY

    with _patched(instance):
        await sync_distributed_model_file_state(MagicMock(), sub_file, instance)

    assert instance.state == ModelInstanceStateEnum.STARTING
