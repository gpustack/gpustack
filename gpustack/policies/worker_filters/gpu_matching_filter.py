import logging
from typing import List, Tuple
from gpustack.policies.base import WorkerFilter
from gpustack.schemas.models import Model
from gpustack.schemas.workers import Worker
from gpustack.server.db import get_engine
from gpustack.utils.gpu import parse_gpu_ids_by_worker

logger = logging.getLogger(__name__)


class GPUMatchingFilter(WorkerFilter):
    def __init__(self, model: Model):
        self._model = model
        self._engine = get_engine()

    async def filter(self, workers: List[Worker]) -> Tuple[List[Worker], List[str]]:
        """
        Filter the gpus with the gpu selector.
        """

        if (
            self._model.gpu_selector is None
            or self._model.gpu_selector.gpu_ids is None
            or len(self._model.gpu_selector.gpu_ids) == 0
        ):
            return workers, []

        gpu_ids_by_worker = parse_gpu_ids_by_worker(self._model.gpu_selector.gpu_ids)
        seleted_workers = gpu_ids_by_worker.keys()

        candidates = []
        for worker in workers:
            if worker.name not in seleted_workers:
                continue

            candidates.append(worker)

        return candidates, [
            f"Matched {len(candidates)} {'worker' if len(candidates) == 1 else 'workers'} by gpu selector."
        ]
