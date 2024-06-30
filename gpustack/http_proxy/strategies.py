from abc import ABC, abstractmethod
import logging
from typing import Dict, List
import itertools

from gpustack.schemas.models import ModelInstance

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(ABC):

    @abstractmethod
    async def select_instance(self, instances: List[ModelInstance]) -> ModelInstance:
        pass


class RoundRobinStrategy(LoadBalancingStrategy):
    def __init__(self):
        self._iterators: Dict[int, itertools.cycle] = {}
        self._instance_lists: Dict[int, List[ModelInstance]] = {}

    async def select_instance(self, instances: List[ModelInstance]) -> ModelInstance:
        if len(instances) == 0:
            raise Exception("No instances available")
        model_id = instances[0].model_id
        if (
            model_id not in self._iterators
            or self._instance_lists[model_id] != instances
        ):
            logger.debug(f"Creating new iterator for model {model_id}")
            self._iterators[model_id] = itertools.cycle(instances)
            self._instance_lists[model_id] = instances

        return next(self._iterators[model_id])
