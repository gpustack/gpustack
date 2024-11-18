from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from typing import Dict, List, Optional, Tuple
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
    def filter(self, workers: List[Worker]) -> Tuple[List[Worker], List[str]]:
        """
        Filter workers suitable for scheduling.
        :return: A tuple containing:
                 - A list of workers that pass the filter.
                 - A list of messages why certain workers were filtered out.
        """
        pass


class WorkerFilterChain:
    def __init__(self, filters: List[WorkerFilter]):
        self.filters = filters

    async def filter(self, workers) -> Tuple[List[Worker], List[str]]:
        """
        Applies all filters sequentially to the list of workers.
        :param workers: The initial list of workers.
        :return: A tuple containing:
                 - The final list of workers that pass all filters.
                 - A list of messages for all workers filtered out across all filters.
        """
        messages = []
        for policy in self.filters:
            workers, filter_messages = await policy.filter(workers)
            messages.extend(filter_messages)
        return workers, messages


class ScheduleCandidatesSelector(ABC):
    @abstractmethod
    async def select_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Get schedule candidates.
        :param workers: The list of workers to select from.
        :return: A list of schedule candidates.
        """
        pass


class ModelInstanceScorer(ABC):
    @abstractmethod
    async def score_instances(
        self, instances: List[ModelInstance]
    ) -> List[ModelInstanceScore]:
        """
        Score the instances.
        :param instances: The list of instances to score.
        :return: A list of scored instances.
        """
        pass


class ScheduleCandidatesScorer(ABC):
    @abstractmethod
    async def score(
        self, candidates: List[ModelInstanceScheduleCandidate]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Score the candidates.
        :param candidates: The list of candidates to score.
        :return: A list of scored candidates.
        """
        pass
