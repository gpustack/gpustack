import logging
from typing import List, Tuple

from gpustack.schemas.models import Model
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.server.db import get_engine

logger = logging.getLogger(__name__)


class StatusFilter:
    def __init__(self, model: Model):
        self._engine = get_engine()
        self._model = model

    async def filter(self, workers: List[Worker]) -> Tuple[List[Worker], List[str]]:
        """
        Filter the workers with the worker selector.
        """

        candidates = []
        for worker in workers:
            if worker.state == WorkerStateEnum.READY:
                candidates.append(worker)

        messages = []
        if len(candidates) != len(workers):
            messages = [
                f"Matched {len(candidates)}/{len(workers)} workers by READY status."
            ]
        return candidates, messages
