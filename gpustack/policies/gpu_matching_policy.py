import logging
from typing import List
from gpustack.schemas.models import Model, ModelInstance
from gpustack.schemas.workers import Worker
from gpustack.server.db import get_engine

logger = logging.getLogger(__name__)


class GPUMatchingPolicy:
    def __init__(self, model: Model, model_instance: ModelInstance):
        self._model = model
        self._model_instance = model_instance
        self._engine = get_engine()

    async def filter(self, workers: List[Worker]) -> List[Worker]:
        """
        Filter the gpus with the gpu selector.
        """

        logger.debug(
            f"model {self._model.name}, filter gpus with gpu matching policy, instance {self._model_instance.name}"
        )

        if self._model.gpu_selector is None:
            return workers

        candidates = []
        for worker in workers:
            if worker.status.gpu_devices is None or len(worker.status.gpu_devices) == 0:
                continue

            if self._model.gpu_selector.worker_name != worker.name:
                continue

            gpu_candidates = []
            for gpu in worker.status.gpu_devices:
                if self._model.gpu_selector.gpu_index != gpu.index:
                    continue

                if (
                    self._model.gpu_selector.gpu_name is not None
                    and self._model.gpu_selector.gpu_name != gpu.name
                ):
                    continue

                gpu_candidates.append(gpu)

            if len(gpu_candidates) != 0:
                worker.status.gpu_devices = gpu_candidates
                candidates.append(worker)

        return candidates
