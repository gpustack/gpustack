import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpustack.schemas.model_files import ModelFile, ModelFileStateEnum
from gpustack.schemas.models import SourceEnum
from gpustack.worker.model_file_manager import ModelFileManager


def _manager(cache_dir: str) -> ModelFileManager:
    manager = ModelFileManager.__new__(ModelFileManager)
    manager._config = SimpleNamespace(cache_dir=cache_dir)
    return manager


def _lock_path(cache_dir: str, source_dir: str, group_or_owner: str, name: str) -> str:
    return os.path.join(cache_dir, source_dir, group_or_owner, f"{name}.lock")


@pytest.mark.asyncio
async def test_get_incomplete_model_files_modelscope_whole_repo_includes_lock_file(
    tmp_path,
):
    # Mirrors the SourceEnum.MODEL_SCOPE / no model_scope_file_path path: a full
    # repo download, deleted before it completed (issue #4018's exact shape).
    cache_dir = str(tmp_path)
    local_dir = os.path.join(cache_dir, "model_scope", "unsloth", "DeepSeek-R1-GGUF")
    lock_path = _lock_path(cache_dir, "model_scope", "unsloth", "DeepSeek-R1-GGUF")
    os.makedirs(local_dir, exist_ok=True)
    Path(lock_path).touch()

    model_file = ModelFile(
        source=SourceEnum.MODEL_SCOPE,
        model_scope_model_id="unsloth/DeepSeek-R1-GGUF",
        resolved_paths=[local_dir],
        state=ModelFileStateEnum.READY,
    )

    paths = await _manager(cache_dir)._get_incomplete_model_files(model_file)

    assert lock_path in paths


@pytest.mark.asyncio
async def test_get_incomplete_model_files_modelscope_single_file_includes_lock_file(
    tmp_path,
):
    # Mirrors the model_scope_file_path path: a single matched file inside the
    # repo dir was downloading when the model file was deleted.
    cache_dir = str(tmp_path)
    local_dir = os.path.join(cache_dir, "model_scope", "unsloth", "DeepSeek-R1-GGUF")
    lock_path = _lock_path(cache_dir, "model_scope", "unsloth", "DeepSeek-R1-GGUF")
    os.makedirs(local_dir, exist_ok=True)
    Path(lock_path).touch()

    model_file = ModelFile(
        source=SourceEnum.MODEL_SCOPE,
        model_scope_model_id="unsloth/DeepSeek-R1-GGUF",
        model_scope_file_path="model.gguf",
        resolved_paths=[os.path.join(local_dir, "model.gguf")],
        state=ModelFileStateEnum.READY,
    )

    paths = await _manager(cache_dir)._get_incomplete_model_files(model_file)

    assert lock_path in paths


@pytest.mark.asyncio
async def test_get_incomplete_model_files_huggingface_whole_repo_includes_lock_file(
    tmp_path,
):
    # Same coarse-lock leak on the HuggingFace side: downloaders.HfDownloader takes
    # out the identical HeartbeatSoftFileLock (get_lock_path) before its own
    # per-file lock/metadata cleanup (already handled below) ever runs.
    cache_dir = str(tmp_path)
    local_dir = os.path.join(cache_dir, "huggingface", "org", "repo")
    lock_path = _lock_path(cache_dir, "huggingface", "org", "repo")
    os.makedirs(local_dir, exist_ok=True)
    Path(lock_path).touch()

    model_file = ModelFile(
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/repo",
        resolved_paths=[local_dir],
        state=ModelFileStateEnum.READY,
    )

    paths = await _manager(cache_dir)._get_incomplete_model_files(model_file)

    assert lock_path in paths
