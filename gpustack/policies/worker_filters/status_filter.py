import logging
from typing import List, Optional, Tuple

from gpustack.schemas.models import Model, ModelInstance
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.server.db import get_engine

logger = logging.getLogger(__name__)


class StatusFilter:
    def __init__(self, model: Model, model_instance: Optional[ModelInstance] = None):
        self._engine = get_engine()
        self._model = model
        self._model_instance = model_instance

    async def filter(self, workers: List[Worker]) -> Tuple[List[Worker], List[str]]:
        """
        Filter the workers with the worker selector.
        """

        logger.debug(
            f"model {self._model.name}, filter workers with status policy"
            + (
                f", instance {self._model_instance.name}"
                if self._model_instance
                else ""
            )
        )

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
