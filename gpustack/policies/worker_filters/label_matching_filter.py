import logging
from typing import Dict, List, Tuple
from gpustack.policies.base import WorkerFilter
from gpustack.schemas.models import BackendEnum, Model, get_backend
from gpustack.schemas.workers import Worker
from gpustack.server.db import get_engine

logger = logging.getLogger(__name__)


class LabelMatchingFilter(WorkerFilter):
    def __init__(self, model: Model):
        self._model = model
        self._engine = get_engine()

    async def filter(self, workers: List[Worker]) -> Tuple[List[Worker], List[str]]:
        """
        Filter the workers with the worker selector.
        """

        if self._model.worker_selector is None:
            return workers, []

        candidates = []
        for worker in workers:
            if label_matching(self._model.worker_selector, worker.labels):
                candidates.append(worker)

        messages = []
        if len(candidates) != len(workers):
            messages = [
                f"Matched {len(candidates)}/{len(workers)} workers by label selector: {self._model.worker_selector}."
            ]
            if get_backend(self._model) == BackendEnum.VLLM:
                messages[0] += " (Note: The vLLM backend supports Linux only.)"
            elif get_backend(self._model) == BackendEnum.ASCEND_MINDIE:
                messages[0] += " (Note: The Ascend MindIE backend supports Linux only.)"

        return candidates, messages


def label_matching(required_labels: Dict[str, str], current_labels) -> bool:
    """
    Check if the current labels matched the required labels.
    """

    for key, value in required_labels.items():
        if key not in current_labels or current_labels[key] != value:
            return False
    return True
