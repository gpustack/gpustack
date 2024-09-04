import logging
from typing import Dict, List
from gpustack.schemas.models import Model, ModelInstance
from gpustack.schemas.workers import Worker
from gpustack.server.db import get_engine

logger = logging.getLogger(__name__)


class LabelMatchingPolicy:
    def __init__(self, model: Model, model_instance: ModelInstance):
        self._model = model
        self._model_instance = model_instance
        self._engine = get_engine()

    async def filter(self, workers: List[Worker]) -> List[Worker]:
        """
        Filter the workers with the worker selector.
        """

        logger.debug(
            f"model {self._model.name}, filter workers with label matching policy, instance {self._model_instance.name}"
        )

        if self._model.worker_selector is None:
            return workers

        candidates = []
        for worker in workers:
            if label_matching(self._model.worker_selector, worker.labels):
                candidates.append(worker)
        return candidates


def label_matching(required_labels: Dict[str, str], current_labels) -> bool:
    """
    Check if the current labels matched the required labels.
    """

    for key, value in required_labels.items():
        if key not in current_labels or current_labels[key] != value:
            return False
    return True
