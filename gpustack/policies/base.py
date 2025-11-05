from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from typing import Dict, List, Optional, Tuple
from gpustack.schemas.models import (
    ComputedResourceClaim,
    ModelInstance,
    ModelInstanceSubordinateWorker,
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
    gpu_addresses: Optional[List[str]] = None
    score: Optional[float] = None
    overcommit: Optional[bool] = None

    # for multi-worker distributed scheduling
    subordinate_workers: Optional[List[ModelInstanceSubordinateWorker]] = None

    def to_log_string(self) -> str:
        log_entries = [
            f"worker: '{self.worker.name}'",
        ]
        if self.gpu_indexes:
            log_entries.append(f"gpu_indexes: {self.gpu_indexes}")
        if self.gpu_addresses:
            log_entries.append(f"gpu_addresses: {self.gpu_addresses}")
        if self.computed_resource_claim.offload_layers:
            log_entries.append(
                f"offload_layers: {self.computed_resource_claim.offload_layers}"
            )
        if self.computed_resource_claim.tensor_split:
            log_entries.append(
                f"tensor_split: {self.computed_resource_claim.tensor_split}"
            )
        if self.overcommit:
            log_entries.append("overcommit: true")

        if self.subordinate_workers:
            sw_str = '), ('.join(
                [
                    f"worker_id: {sw.worker_id}, "
                    f"worker_name: {sw.worker_name}, "
                    f"worker_ip: {sw.worker_ip}, "
                    f"worker_ifname {sw.worker_ifname}, "
                    f"total_gpus: {sw.total_gpus}, "
                    f"gpu_indexes: {sw.gpu_indexes}, "
                    f"gpu_addresses: {sw.gpu_addresses}"
                    for sw in self.subordinate_workers
                ]
            )
            log_entries.append(f"subordinate_workers: [{sw_str}]")

        return ', '.join(log_entries)


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
