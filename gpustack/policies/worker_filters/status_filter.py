import logging
from typing import List, Tuple

from gpustack.schemas.models import Model
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.utils.model_instance_workers import (
    worker_supports_model_instance_workloads,
)

logger = logging.getLogger(__name__)


class StatusFilter:
    def __init__(self, model: Model, require_managed_workload_protocol: bool = False):
        self._model = model
        self._require_managed_workload_protocol = require_managed_workload_protocol

    async def filter(self, workers: List[Worker]) -> Tuple[List[Worker], List[str]]:
        """
        Filter the workers with the worker selector.
        """

        candidates = []
        incompatible = False
        for worker in workers:
            if worker.state != WorkerStateEnum.READY:
                continue
            if self._require_managed_workload_protocol and not (
                worker_supports_model_instance_workloads(worker)
            ):
                incompatible = True
                continue
            candidates.append(worker)

        messages = []
        if len(candidates) != len(workers):
            criterion = (
                "READY status and managed workload admission"
                if self._require_managed_workload_protocol
                else "READY status"
            )
            messages = [
                f"Matched {len(candidates)}/{len(workers)} workers by {criterion}."
            ]
        if incompatible:
            messages.append(
                "Managed workloads require a READY, bound Worker with orchestration "
                "protocol v1 and admission enabled; upgrade or re-register incompatible Workers."
            )
        return candidates, messages
