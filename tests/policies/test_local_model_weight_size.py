import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpustack.policies import utils


@pytest.mark.asyncio
async def test_single_file_local_model_weight_size_uses_file_size(tmp_path: Path):
    weight = tmp_path / "model.ninfer"
    weight.write_bytes(b"weights")

    assert await utils.get_local_model_weight_size(str(weight)) == len(b"weights")


@pytest.mark.asyncio
async def test_local_model_weight_size_reports_worker_lookup_failure(monkeypatch):
    class WorkerFilesystem:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get_model_weight_size(self, worker, path):
            raise OSError("missing")

    monkeypatch.setattr(utils, "WorkerFilesystemClient", WorkerFilesystem)

    with pytest.raises(FileNotFoundError, match="any worker"):
        await utils.get_local_model_weight_size(
            "/missing/model.ninfer", workers=[SimpleNamespace(id=1)]
        )


@pytest.mark.asyncio
async def test_local_model_weight_size_cancels_slower_workers_after_first_result(
    monkeypatch,
):
    cancelled = asyncio.Event()

    class WorkerFilesystem:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get_model_weight_size(self, worker, path):
            if worker.id == 1:
                return 7
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.set()
                raise

    monkeypatch.setattr(utils, "WorkerFilesystemClient", WorkerFilesystem)

    result = await utils.get_local_model_weight_size(
        "/missing/model.ninfer",
        workers=[SimpleNamespace(id=1), SimpleNamespace(id=2)],
    )

    assert result == 7
    await asyncio.wait_for(cancelled.wait(), timeout=1)
