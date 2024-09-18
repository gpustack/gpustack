from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from typing import Dict, List, Optional
from gpustack.schemas.models import (
    ComputedResourceClaim,
    ModelInstance,
    ModelInstanceRPCServer,
)
from gpustack.schemas.workers import Worker

logger = logging.getLogger(__name__)


@dataclass
class ModelInstanceScore:
    model_instance: ModelInstance
    score: Optional[float] = None


@dataclass
class ModelInstanceScheduleCandidate:
    worker: Worker
    gpu_indexes: Optional[List[int]]
    computed_resource_claim: ComputedResourceClaim
    score: Optional[float] = None

    # for scale down
    instance: Optional[ModelInstance] = None

    # for rpc server scheduling
    rpc_servers: Optional[List[ModelInstanceRPCServer]] = None


@dataclass
class AllocationResource:
    ram: int
    vram: Dict[int, int]


@dataclass
class Allocatable(AllocationResource):
    pass


@dataclass
class Allocated(AllocationResource):
    pass


class WorkerFilter(ABC):
    @abstractmethod
    def filter(self, workers: List[Worker]) -> List[Worker]:
        """
        Filter workers suitable for scheduling.
        """
        pass


class WorkerFilterChain:
    def __init__(self, filters: List[WorkerFilter]):
        self.filters = filters

    async def filter(self, workers):
        for policy in self.filters:
            workers = await policy.filter(workers)
        return workers


class ScheduleCandidatesSelector(ABC):
    @abstractmethod
    async def select_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Get schedule candidates.
        """
        pass


class ModelInstanceScorer(ABC):
    @abstractmethod
    async def score_instances(
        self, instances: List[ModelInstance]
    ) -> List[ModelInstanceScore]:
        """
        Score the instances.
        """
        pass


class ScheduleCandidatesScorer(ABC):
    @abstractmethod
    async def score(
        self, candidates: List[ModelInstanceScheduleCandidate]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Score the candidates.
        """
        pass
