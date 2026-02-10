import logging
from typing import List, Tuple

from gpustack.client.worker_filesystem_client import WorkerFilesystemClient
from gpustack.policies.base import WorkerFilter
from gpustack.schemas.models import Model, SourceEnum
from gpustack.schemas.workers import Worker

logger = logging.getLogger(__name__)


class LocalPathFilter(WorkerFilter):
    """
    Filter workers based on whether the local path exists on them.
    Only applies to LOCAL_PATH models.
    """

    def __init__(self, model: Model):
        self._model = model

    async def filter(self, workers: List[Worker]) -> Tuple[List[Worker], List[str]]:
        """
        Filter workers by validating that the local path exists on each worker.
        For non-LOCAL_PATH models, all workers pass through unchanged.
        """

        # Skip filtering for non-LOCAL_PATH models
        if self._model.source != SourceEnum.LOCAL_PATH:
            return workers, []

        # Skip if no GPU selector is specified
        if (
            self._model.gpu_selector is None
            or self._model.gpu_selector.gpu_ids is None
            or len(self._model.gpu_selector.gpu_ids) == 0
        ):
            return workers, []

        candidates = []
        invalid_workers = []

        # Validate local path existence on each worker
        async with WorkerFilesystemClient() as filesystem_client:
            for worker in workers:
                try:
                    exists_response = await filesystem_client.path_exists(
                        worker, self._model.local_path
                    )
                    if exists_response.exists:
                        candidates.append(worker)
                    else:
                        invalid_workers.append(worker.name)
                except Exception as e:
                    logger.warning(
                        f"Failed to check path {self._model.local_path} "
                        f"on worker {worker.name}: {e}"
                    )
                    invalid_workers.append(worker.name)

        messages = []
        if invalid_workers:
            sorted_workers = sorted(invalid_workers)
            if len(sorted_workers) > 3:
                display_workers = sorted_workers[:3]
                worker_list = (
                    f"{', '.join(display_workers)}... "
                    f"(+{len(sorted_workers) - 3} more)"
                )
            else:
                worker_list = ', '.join(sorted_workers)

            messages.append(
                f"The model file path '{self._model.local_path}' does not "
                f"exist on the following workers: "
                f"{worker_list}. "
                f"Please ensure the model file is accessible on all workers."
            )

        return candidates, messages
