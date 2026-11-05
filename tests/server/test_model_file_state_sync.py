"""Regression tests for distributed download-progress → STARTING promotion.

Completion is decided by ModelFile.state alone, not by the display-only
subordinate_workers[].download_progress mirror, and a subordinate
READY event must re-check completion even when its progress already reached 100.
"""

import contextlib
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.orm.attributes import set_committed_value

from gpustack.schemas.model_files import ModelFile, ModelFileStateEnum
from gpustack.schemas.models import (
    DistributedServers,
    LoraListEntry,
    Model,
    ModelInstance,
    ModelInstanceStateEnum,
    ModelInstanceSubordinateWorker,
    ModelSource,
    SourceEnum,
)
from gpustack.server.controllers import (
    _refresh_instance_download_progress,
    _sync_subordinate_worker,
    sync_distributed_model_file_state,
    sync_main_worker_model_file_state,
)
from tests.utils.model import new_model, new_model_instance

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


def _distributed_instance(sub_download_progress: Optional[float]) -> ModelInstance:
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


@pytest.mark.asyncio
@pytest.mark.parametrize('subordinate', [False, True])
async def test_discarded_download_bytes_reduce_displayed_progress(subordinate):
    instance = _distributed_instance(sub_download_progress=50)
    instance.download_progress = 50
    file = instance.model_files[1 if subordinate else 0]
    file.state = ModelFileStateEnum.DOWNLOADING
    file.download_progress = 30

    if subordinate:
        worker = instance.distributed_servers.subordinate_workers[0]
        changed = await _sync_subordinate_worker(MagicMock(), instance, worker, file)
        progress = worker.download_progress
    else:
        changed = _refresh_instance_download_progress(instance, file)
        progress = instance.download_progress

    assert changed
    assert progress == 30
    assert instance.state == ModelInstanceStateEnum.DOWNLOADING


@contextlib.contextmanager
def _patched(instance: ModelInstance, model=None):
    """Patch every DB touchpoint so the sync runs in-memory, while letting
    _download_completed execute for real against instance.model_files."""
    service = MagicMock()
    service.return_value.update = AsyncMock()
    stored_servers = instance.distributed_servers.model_copy(deep=True)

    async def load_instance(*args, **kwargs):
        # An ORM refresh replaces nested JSON with its persisted value.
        set_committed_value(
            instance, "distributed_servers", stored_servers.model_copy(deep=True)
        )
        return instance

    patches = [
        patch.object(
            ModelInstance,
            "one_by_id_with_model_files",
            AsyncMock(side_effect=load_instance),
        ),
        patch.object(ModelInstance, "one_by_id", AsyncMock(return_value=instance)),
        patch("gpustack.server.controllers.ModelInstanceService", service),
        patch.object(Model, "one_by_id", AsyncMock(return_value=model)),
    ]
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        yield service.return_value


@pytest.mark.asyncio
async def test_main_worker_ready_promotes_despite_lagging_subordinate_progress():
    """All ModelFiles are READY but the subordinate's display progress is stuck
    below 100. The final main-worker READY event must still promote to STARTING."""
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
    and promote to STARTING."""
    instance = _distributed_instance(sub_download_progress=100)
    sub_file = instance.model_files[1]  # worker_id == SUB_WORKER_ID, READY

    with _patched(instance):
        await sync_distributed_model_file_state(MagicMock(), sub_file, instance)

    assert instance.state == ModelInstanceStateEnum.STARTING


@pytest.mark.asyncio
async def test_promotion_backfills_resolved_path_from_ready_primary_file():
    """Reaching STARTING must always carry a resolved_path, otherwise the worker
    crashes with Path(None) ("expected str, bytes or os.PathLike object, not
    NoneType"). The subordinate READY path promotes without setting resolved_path,
    and a concurrent main-worker event may promote off a snapshot where it is
    still None. Promotion must backfill it from the READY primary ModelFile."""
    instance = _distributed_instance(sub_download_progress=100)
    instance.resolved_path = None
    sub_file = instance.model_files[1]  # subordinate file, never owns resolved_path

    with _patched(instance):
        await sync_distributed_model_file_state(MagicMock(), sub_file, instance)

    assert instance.state == ModelInstanceStateEnum.STARTING
    assert instance.resolved_path == "/cache/main"


@pytest.mark.asyncio
@pytest.mark.parametrize("progress", [None, 0, 50, 100])
@pytest.mark.parametrize("main_ready", [False, True])
@pytest.mark.parametrize("with_lora", [False, True])
async def test_subordinate_ready_persists_progress(progress, main_ready, with_lora):
    instance = _distributed_instance(sub_download_progress=progress)
    if not main_ready:
        instance.model_files[0].state = ModelFileStateEnum.DOWNLOADING
    model = new_model(1, "test", huggingface_repo_id="Qwen/Qwen3-0.6B")
    if with_lora:
        model.lora_list = [
            LoraListEntry(lora_name="adapter", lora_repo_name="test/adapter")
        ]
        lora_file = _model_file(
            30, MAIN_WORKER_ID, ModelFileStateEnum.READY, ["/cache/adapter"]
        )
        lora_file.is_lora = True
        lora_file.huggingface_repo_id = "test/adapter"
        instance.model_files.append(lora_file)

    with _patched(instance, model) as service:
        await sync_distributed_model_file_state(
            MagicMock(), instance.model_files[1], instance
        )

    assert instance.distributed_servers.subordinate_workers[0].download_progress == 100
    assert instance.state == (
        ModelInstanceStateEnum.STARTING
        if main_ready
        else ModelInstanceStateEnum.DOWNLOADING
    )
    if progress != 100 or main_ready:
        service.update.assert_awaited_once_with(instance)
    else:
        service.update.assert_not_awaited()
    if main_ready:
        assert instance.resolved_path == "/cache/main"
        assert len(instance.mounted_loras) == int(with_lora)
        if with_lora:
            assert instance.mounted_loras[0].path == "/cache/adapter"


@pytest.mark.asyncio
@pytest.mark.parametrize("draft_ready", [False, True])
async def test_subordinate_ready_preserves_progress_while_waiting_for_draft(
    draft_ready,
):
    instance = _distributed_instance(sub_download_progress=0)
    instance.draft_model_source = ModelSource(
        source=SourceEnum.HUGGING_FACE, huggingface_repo_id="test/draft"
    )
    draft_file = _model_file(
        30,
        MAIN_WORKER_ID,
        ModelFileStateEnum.READY if draft_ready else ModelFileStateEnum.DOWNLOADING,
        ["/cache/draft"] if draft_ready else None,
    )
    draft_file.huggingface_repo_id = "test/draft"
    instance.draft_model_files = [draft_file]

    with _patched(instance) as service:
        await sync_distributed_model_file_state(
            MagicMock(), instance.model_files[1], instance
        )

    service.update.assert_awaited_once_with(instance)
    assert instance.distributed_servers.subordinate_workers[0].download_progress == 100
    assert instance.state == (
        ModelInstanceStateEnum.STARTING
        if draft_ready
        else ModelInstanceStateEnum.DOWNLOADING
    )
    if draft_ready:
        assert instance.draft_model_resolved_path == "/cache/draft"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "file_state, instance_state, needs_files",
    [
        (ModelFileStateEnum.DOWNLOADING, ModelInstanceStateEnum.DOWNLOADING, False),
        (ModelFileStateEnum.ERROR, ModelInstanceStateEnum.DOWNLOADING, False),
        (ModelFileStateEnum.READY, ModelInstanceStateEnum.DOWNLOADING, True),
        (ModelFileStateEnum.READY, ModelInstanceStateEnum.INITIALIZING, True),
        (ModelFileStateEnum.READY, ModelInstanceStateEnum.STARTING, False),
        (ModelFileStateEnum.READY, ModelInstanceStateEnum.RUNNING, False),
    ],
)
async def test_subordinate_loads_files_only_for_completion(
    file_state, instance_state, needs_files
):
    instance = _distributed_instance(sub_download_progress=79)
    instance.state = instance_state
    file = instance.model_files[1]
    file.state = file_state
    file.download_progress = 99
    file.state_message = (
        "Download failed" if file_state == ModelFileStateEnum.ERROR else ""
    )

    with _patched(instance) as service:
        await sync_distributed_model_file_state(MagicMock(), file, instance)
        loader = ModelInstance.one_by_id_with_model_files
        if needs_files:
            loader.assert_awaited_once()
        else:
            loader.assert_not_awaited()

    service.update.assert_awaited_once_with(instance)
    if file_state == ModelFileStateEnum.ERROR:
        assert instance.state == ModelInstanceStateEnum.ERROR
        assert instance.state_message == "Download failed"
    else:
        subordinate = instance.distributed_servers.subordinate_workers[0]
        expected_progress = 100 if file_state == ModelFileStateEnum.READY else 99
        assert subordinate.download_progress == expected_progress


@pytest.mark.asyncio
@pytest.mark.parametrize('still_linked', [False, True])
async def test_file_event_rechecks_association_after_managed_recovery(
    monkeypatch, still_linked
):
    from gpustack.server import controllers
    from gpustack.server.bus import Event, EventType

    instance = _distributed_instance(sub_download_progress=0)
    old_snapshot = instance.model_copy(deep=True)
    file = instance.model_files[0]
    set_committed_value(file, 'instances', [old_snapshot])
    set_committed_value(file, 'draft_instances', [])
    if not still_linked:
        set_committed_value(instance, 'model_files', [])
    session = MagicMock()

    @contextlib.asynccontextmanager
    async def session_context():
        yield session

    monkeypatch.setattr(controllers, 'async_session', session_context)
    monkeypatch.setattr(ModelFile, 'one_by_id', AsyncMock(return_value=file))
    load = AsyncMock(return_value=instance)
    monkeypatch.setattr(ModelInstance, 'one_by_id_with_model_files', load)
    sync = AsyncMock()
    monkeypatch.setattr(controllers, 'sync_main_worker_model_file_state', sync)

    await controllers.ModelFileController()._reconcile(Event(EventType.UPDATED, file))

    load.assert_awaited_once_with(session, instance.id, for_update=True)
    assert sync.await_count == int(still_linked)
    if still_linked:
        sync.assert_awaited_once_with(session, file, instance)


@pytest.mark.asyncio
@pytest.mark.parametrize('retired', [False, True])
async def test_file_preparation_checks_generation_under_lock(monkeypatch, retired):
    from gpustack.server import controllers

    instance = _distributed_instance(sub_download_progress=0)
    instance.state = ModelInstanceStateEnum.INITIALIZING
    event = instance.model_copy(deep=True)
    if retired:
        instance.restart_count = (instance.restart_count or 0) + 1
    load = AsyncMock(return_value=instance)
    monkeypatch.setattr(ModelInstance, 'one_by_id', load)
    monkeypatch.setattr(Model, 'one_by_id', AsyncMock(return_value=None))
    sync = AsyncMock()
    monkeypatch.setattr(controllers, 'sync_instance_files_state', sync)
    await controllers.ensure_instance_model_file(MagicMock(), event)
    assert sync.await_count == int(not retired)
    assert load.await_args.kwargs.get('for_update') is True
    assert load.await_args.kwargs.get('refresh') is True


@pytest.mark.asyncio
async def test_file_sync_rechecks_generation_after_each_commit(monkeypatch):
    from gpustack.server import controllers

    instance = _distributed_instance(sub_download_progress=0)
    files = list(instance.model_files)
    expected = instance.restart_count or 0

    async def load(*args, **kwargs):
        return instance

    monkeypatch.setattr(
        ModelInstance, 'one_by_id_with_model_files', AsyncMock(side_effect=load)
    )

    async def recover(*args, **kwargs):
        instance.restart_count = expected + 1

    primary = AsyncMock(side_effect=recover)
    subordinate = AsyncMock()
    monkeypatch.setattr(controllers, 'sync_main_worker_model_file_state', primary)
    monkeypatch.setattr(controllers, 'sync_distributed_model_file_state', subordinate)
    await controllers.sync_instance_files_state(MagicMock(), instance, files)
    primary.assert_awaited_once()
    subordinate.assert_not_awaited()
