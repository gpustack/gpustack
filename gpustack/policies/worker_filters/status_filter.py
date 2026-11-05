import logging
from typing import List, Tuple

from gpustack.schemas.models import Model
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.utils.model_instance_workers import (
    worker_supports_model_instance_workloads,
)

logger = logging.getLogger(__name__)


class StatusFilter:
    def __init__(self, model: Model, require_managed_worker_binding: bool = False):
        self._model = model
        self._require_managed_worker_binding = require_managed_worker_binding

    async def filter(self, workers: List[Worker]) -> Tuple[List[Worker], List[str]]:
        """
        Filter the workers with the worker selector.
        """

        candidates = []
        incompatible = False
        for worker in workers:
            if worker.state != WorkerStateEnum.READY:
                continue
            if (
                self._require_managed_worker_binding
                and not worker_supports_model_instance_workloads(worker)
            ):
                incompatible = True
                continue
            candidates.append(worker)

        messages = []
        if len(candidates) != len(workers):
            criterion = (
                "READY status and bound Worker identity"
                if self._require_managed_worker_binding
                else "READY status"
            )
            messages = [
                f"Matched {len(candidates)}/{len(workers)} workers by {criterion}."
            ]
        if incompatible:
            messages.append(
                "Managed workloads require a bound Worker identity; "
                "re-register unbound Workers."
            )
        return candidates, messages
