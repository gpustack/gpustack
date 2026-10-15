"""get_local_model_weight_size must raise instead of returning None when every
worker fails, so the callers' existing except/default handling takes over."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gpustack.policies.utils import get_local_model_weight_size

_MISSING_PATH = "/nonexistent/gpustack-test-model"
_CLIENT_METHOD = (
    "gpustack.client.worker_filesystem_client."
    "WorkerFilesystemClient.get_model_weight_size"
)


@pytest.mark.asyncio
async def test_get_local_model_weight_size_from_workers():
    workers = [SimpleNamespace(id=index) for index in range(3)]

    async def only_last_worker_has_it(self, worker, path):
        if worker.id == 2:
            return 4096
        raise RuntimeError(f"boom from worker {worker.id}")

    async def no_worker_has_it(self, worker, path):
        raise RuntimeError(f"boom from worker {worker.id}")

    with patch(_CLIENT_METHOD, new=only_last_worker_has_it):
        assert await get_local_model_weight_size(_MISSING_PATH, workers) == 4096

    with patch(_CLIENT_METHOD, new=no_worker_has_it):
        with pytest.raises(ValueError, match="Failed to get model weight size"):
            await get_local_model_weight_size(_MISSING_PATH, workers)
