from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from typing import Dict, List, Optional, Tuple
from gpustack.schemas.models import (
    ComputedResourceClaim,
    ModelInstance,
    ModelInstanceRPCServer,
    RayActor,
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
    overcommit: Optional[bool] = None

    # for multi-worker distributed scheduling
    rpc_servers: Optional[List[ModelInstanceRPCServer]] = None
    ray_actors: Optional[List[RayActor]] = None

    def to_log_string(self) -> str:
        log_dict = {
            "worker": f"worker: {self.worker.name}",
            "gpu_indexes": (
                f"gpu_indexes: {self.gpu_indexes}"
                if self.gpu_indexes is not None
                else None
            ),
            "offload_layers": (
                f"offload layers: {self.computed_resource_claim.offload_layers}"
                if self.computed_resource_claim.offload_layers is not None
                else None
            ),
            "tensor_split": (
                f"tensor_split: {self.computed_resource_claim.tensor_split}"
                if self.computed_resource_claim.tensor_split is not None
                else None
            ),
            "rpcs": None,
        }

        if self.rpc_servers:
            rpcs_string = ', '.join(
                [
                    f"(worker_id: {rpc.worker_id}, gpu_index:{rpc.gpu_index}, offload layers:{rpc.computed_resource_claim.offload_layers})"
                    for rpc in self.rpc_servers
                ]
            )
            log_dict["rpcs"] = f"rpcs: [{rpcs_string}]"

        log_parts = [value for value in log_dict.values() if value is not None]

        return ', '.join(log_parts)


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
