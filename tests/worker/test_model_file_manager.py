import asyncio
import threading
from unittest.mock import MagicMock

import pytest

from gpustack.schemas.model_files import ModelFile, ModelFileStateEnum
from gpustack.schemas.models import SourceEnum
from gpustack.worker import model_file_manager


@pytest.mark.asyncio
@pytest.mark.parametrize('blocked_operation', ['glob', 'delete', 'log_cleanup'])
async def test_model_file_deletion_keeps_event_loop_responsive(
    monkeypatch, tmp_path, config, blocked_operation
):
    model_dir = tmp_path / 'model'
    model_dir.mkdir()
    weights = model_dir / 'weights.safetensors'
    weights.write_bytes(b'weights')
    model_config = model_dir / 'config.json'
    model_config.write_text('{}')
    log_dir = tmp_path / 'logs'
    (log_dir / 'serve').mkdir(parents=True)
    download_log = log_dir / 'serve' / 'model_file_1.download.log'
    download_log.write_text('download complete')
    manager_config = config.model_copy(update={'log_dir': str(log_dir)})
    clientset = MagicMock()
    manager = model_file_manager.ModelFileManager(1, clientset, manager_config)
    model_file = ModelFile(
        id=1,
        worker_id=1,
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id='test/model',
        state=ModelFileStateEnum.READY,
        cleanup_on_delete=True,
        resolved_paths=[
            (
                str(model_dir / '*.safetensors')
                if blocked_operation == 'glob'
                else str(model_dir)
            )
        ],
    )
    target, name = {
        'glob': (model_file_manager.glob, 'glob'),
        'delete': (model_file_manager, 'delete_path'),
        'log_cleanup': (model_file_manager, '_cleanup_download_log'),
    }[blocked_operation]
    original = getattr(target, name)
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = threading.Event()
    responsive = []

    def slow_operation(*args, **kwargs):
        loop.call_soon_threadsafe(entered.set)
        responsive.append(release.wait(2))
        return original(*args, **kwargs)

    monkeypatch.setattr(target, name, slow_operation)
    deletion = asyncio.create_task(manager._handle_deletion(model_file))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
    finally:
        release.set()
        await asyncio.wait_for(deletion, timeout=5)

    assert responsive and all(responsive)
    assert not weights.exists()
    assert not download_log.exists()
    if blocked_operation == 'glob':
        assert model_config.exists()
    else:
        assert not model_dir.exists()
    clientset.model_files.update.assert_not_called()
