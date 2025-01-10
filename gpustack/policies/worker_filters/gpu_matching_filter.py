import logging
from typing import List, Tuple
from gpustack.policies.base import WorkerFilter
from gpustack.schemas.models import Model, ModelInstance
from gpustack.schemas.workers import Worker
from gpustack.server.db import get_engine
from gpustack.utils.gpu import parse_gpu_ids_by_worker

logger = logging.getLogger(__name__)


class GPUMatchingFilter(WorkerFilter):
    def __init__(self, model: Model, model_instance: ModelInstance):
        self._model = model
        self._model_instance = model_instance
        self._engine = get_engine()

    async def filter(self, workers: List[Worker]) -> Tuple[List[Worker], List[str]]:
        """
        Filter the gpus with the gpu selector.
        """

        logger.debug(
            f"model {self._model.name}, filter gpus with gpu matching policy, instance {self._model_instance.name}"
        )

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

            gpu_candidates = []
            for gpu in worker.status.gpu_devices:
                id = f"{worker.name}:{gpu.type}:{gpu.index}"
                if id not in self._model.gpu_selector.gpu_ids:
                    continue

                gpu_candidates.append(gpu)

            worker.status.gpu_devices = gpu_candidates
            candidates.append(worker)

        return candidates, [f"Matched {len(candidates)} workers by gpu selector."]
