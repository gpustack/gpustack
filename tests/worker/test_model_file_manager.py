import asyncio
import json
import threading
from unittest.mock import MagicMock

import pytest

from gpustack.schemas.model_files import ModelFile, ModelFileStateEnum
from gpustack.schemas.models import SourceEnum
from gpustack.worker import model_file_manager


@pytest.fixture
def lora_task():
    task = object.__new__(model_file_manager.ModelFileDownloadTask)
    task._model_file = ModelFile(
        id=1,
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/adapter",
        base_model="Qwen/Qwen3-8B",
    )
    task._update_model_file = MagicMock()
    task._write_to_instance_download_logs = MagicMock()
    return task


@pytest.mark.parametrize(
    "adapter_config",
    [
        {"base_model_name_or_path": "/training/models/base"},
        {"base_model_name": "other/model"},
        {"base_model_name_or_path": None, "base_model_name": "other/model"},
        {"base_model_name_or_path": None},
        {},
    ],
)
def test_lora_resolution_accepts_base_metadata(lora_task, tmp_path, adapter_config):
    (tmp_path / "adapter_config.json").write_text(json.dumps(adapter_config))

    assert lora_task._resolve_lora_state([str(tmp_path)]) == {
        "is_lora": True,
        "base_model": "Qwen/Qwen3-8B",
    }
    assert lora_task._model_file.is_lora is True
    lora_task._update_model_file.assert_not_called()


@pytest.mark.parametrize(
    "adapter_config,expected",
    [
        ({"base_model_name_or_path": "org/base"}, "org/base"),
        ({"base_model_name_or_path": None, "base_model_name": "org/base"}, "org/base"),
        ({"base_model_name_or_path": "", "base_model_name": "org/base"}, "org/base"),
        ({"base_model_name_or_path": None}, None),
    ],
)
def test_lora_resolution_backfills_empty_base(
    lora_task, tmp_path, adapter_config, expected
):
    lora_task._model_file.base_model = None
    (tmp_path / "adapter_config.json").write_text(json.dumps(adapter_config))

    assert lora_task._resolve_lora_state([str(tmp_path)]) == {
        "is_lora": True,
        "base_model": expected,
    }
    assert lora_task._model_file.base_model == expected
    lora_task._update_model_file.assert_not_called()


@pytest.mark.parametrize("contents", ["{", "[]"])
def test_lora_resolution_rejects_invalid_config(lora_task, tmp_path, contents):
    (tmp_path / "adapter_config.json").write_text(contents)

    assert lora_task._resolve_lora_state([str(tmp_path)]) is None

    lora_task._update_model_file.assert_called_once()
    update = lora_task._update_model_file.call_args
    assert update.args == (1,)
    assert update.kwargs["state"] == ModelFileStateEnum.ERROR
    assert "adapter_config.json" in update.kwargs["state_message"]


@pytest.mark.parametrize("user_hint", [False, True])
def test_lora_resolution_without_adapter_config(lora_task, tmp_path, user_hint):
    lora_task._model_file.is_lora = user_hint

    result = lora_task._resolve_lora_state([str(tmp_path)])

    assert result == (
        {"is_lora": True, "base_model": "Qwen/Qwen3-8B"} if user_hint else {}
    )
    assert lora_task._model_file.is_lora is user_hint
    lora_task._update_model_file.assert_not_called()


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
