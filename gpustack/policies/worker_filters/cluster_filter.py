import logging
from typing import List, Tuple
from gpustack.policies.base import WorkerFilter
from gpustack.schemas.models import Model
from gpustack.schemas.workers import Worker
from gpustack.server.db import get_engine

logger = logging.getLogger(__name__)


class ClusterFilter(WorkerFilter):
    def __init__(self, model: Model):
        self._model = model
        self._engine = get_engine()

    async def filter(self, workers: List[Worker]) -> Tuple[List[Worker], List[str]]:
        """
        Filter the workers with the cluster selector.
        """
        if not hasattr(self._model, "cluster_id"):
            return workers, []
        candidates = []
        for worker in workers:
            if worker.cluster_id != self._model.cluster_id:
                continue

            candidates.append(worker)

        return candidates, [
            f"Matched {len(candidates)} {'worker' if len(candidates) == 1 else 'workers'} by cluster selector."
        ]
