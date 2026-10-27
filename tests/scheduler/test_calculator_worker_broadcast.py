"""read_local_path_file_from_workers must not leave broadcast tasks running
after it has already returned a result to its caller."""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gpustack.schemas.models import Model, SourceEnum
from gpustack.scheduler.calculator import read_local_path_file_from_workers


def make_local_path_model() -> Model:
    return Model(
        name="test-model",
        replicas=1,
        source=SourceEnum.LOCAL_PATH,
        local_path="/mnt/models/foo",
    )


@pytest.mark.asyncio
async def test_read_local_path_file_from_workers_cancels_losers():
    model = make_local_path_model()
    workers = [SimpleNamespace(name=f"worker-{i}") for i in range(5)]

    async def fake_read_model_config(self, worker, path):
        if worker.name == "worker-0":
            return {"ok": True}
        await asyncio.sleep(2)
        return {"ok": True}

    before = asyncio.all_tasks()
    with patch(
        "gpustack.client.worker_filesystem_client.WorkerFilesystemClient.read_model_config",
        new=fake_read_model_config,
    ):
        result = await read_local_path_file_from_workers(model, "config.json", workers)

    spawned = asyncio.all_tasks() - before
    assert result == {"ok": True}
    assert all(t.done() for t in spawned)


@pytest.mark.asyncio
async def test_read_local_path_file_from_workers_reports_all_errors():
    model = make_local_path_model()
    workers = [SimpleNamespace(name=f"worker-{i}") for i in range(3)]

    async def fake_read_model_config(self, worker, path):
        raise RuntimeError(f"boom from {worker.name}")

    with patch(
        "gpustack.client.worker_filesystem_client.WorkerFilesystemClient.read_model_config",
        new=fake_read_model_config,
    ):
        with pytest.raises(ValueError, match="Failed to read 'config.json'"):
            await read_local_path_file_from_workers(model, "config.json", workers)
