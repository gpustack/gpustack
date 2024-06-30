from typing import List

from gpustack.http_proxy.strategies import LoadBalancingStrategy, RoundRobinStrategy
from gpustack.schemas.models import ModelInstance


class LoadBalancer:
    def __init__(self, strategy: LoadBalancingStrategy = None):
        if strategy is None:
            strategy = RoundRobinStrategy()
        self._strategy = strategy

    def set_strategy(self, strategy: LoadBalancingStrategy):
        self._strategy = strategy

    async def get_instance(self, instances: List[ModelInstance]) -> ModelInstance:
        return await self._strategy.select_instance(instances)
