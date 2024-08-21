from dataclasses import dataclass
import logging
from typing import Dict, List, Optional
from gpustack.scheduler.calculator import estimate
from gpustack.schemas.models import ComputedResourceClaim, ModelInstance
from gpustack.schemas.workers import Worker
from sqlmodel.ext.asyncio.session import AsyncSession
from gpustack.server.db import get_engine

logger = logging.getLogger(__name__)


@dataclass
class ModelInstanceScheduleCandidate:
    worker: Worker
    gpu_index: Optional[int]
    computed_resource_claim: ComputedResourceClaim


@dataclass
class AllocationResource:
    memory: int
    gpu_memory: Dict[int, int]


@dataclass
class Allocatable(AllocationResource):
    pass


@dataclass
class Allocated(AllocationResource):
    pass


class ResourceFitPolicy:
    def __init__(self, estimate: estimate):
        self._estimate = estimate
        self._engine = get_engine()

    async def filter(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Filter the workers with the resource fit claim.
        """

        candidates = []
        for worker in workers:
            filterd_candidates = await self._filter_one(worker)
            if len(filterd_candidates) != 0:
                candidates.extend(filterd_candidates)
        return candidates

    async def _filter_one(self, worker: Worker) -> List[ModelInstanceScheduleCandidate]:
        """
        Find a candidate worker for the model instance.
        """

        candidates = []
        total_layers = self._estimate.memory[-1].offloadLayers
        is_unified_memory = worker.status.memory.is_unified_memory

        allocatable = await self._get_worker_allocatable_resource(worker)

        arr = []
        estimate_arr = []
        for memory in self._estimate.memory:
            if is_unified_memory:
                arr.append(memory.uma.vram)
                estimate_arr.append(memory.uma)
            else:
                arr.append(memory.nonUMA.vram)
                estimate_arr.append(memory.nonUMA)

        for gpu_index in allocatable.gpu_memory:
            index = binary_search(arr, allocatable.gpu_memory[gpu_index])
            if index == -1:
                continue

            # For UMA, we need to remove the claim of gpu memory before check if the memory.
            if (
                is_unified_memory
                and (
                    self._estimate.memory[index].uma.ram
                    > allocatable.memory - arr[index]
                )
                or (self._estimate.memory[index].nonUMA.ram > allocatable.memory)
            ):
                continue

            offload_layers = self._estimate.memory[index].offloadLayers
            candidates.append(
                ModelInstanceScheduleCandidate(
                    worker=worker,
                    gpu_index=gpu_index,
                    computed_resource_claim=ComputedResourceClaim(
                        is_unified_memory=is_unified_memory,
                        offload_layers=offload_layers,
                        gpu_memory=estimate_arr[index].vram,
                        memory=estimate_arr[index].ram,
                        total_layers=total_layers,
                    ),
                )
            )
        return candidates

    async def _get_worker_model_instances(self, worker: Worker) -> List[ModelInstance]:
        async with AsyncSession(self._engine) as session:
            model_instances = await ModelInstance.all_by_field(
                session, "worker_id", worker.id
            )
            return model_instances

    async def _get_worker_allocatable_resource(self, worker: Worker) -> Allocatable:
        """
        Get the worker with the latest allocatable resources.
        """

        is_unified_memory = worker.status.memory.is_unified_memory
        model_instances = await self._get_worker_model_instances(worker)

        allocated = Allocated(memory=0, gpu_memory={})
        for model_instance in model_instances:
            if model_instance.worker_id != worker.id:
                continue

            allocated.memory += model_instance.computed_resource_claim.memory or 0
            gpu_index = model_instance.gpu_index
            if gpu_index is not None:
                allocated.gpu_memory[gpu_index] = (
                    allocated.gpu_memory.get(gpu_index, 0)
                ) + (model_instance.computed_resource_claim.gpu_memory or 0)

        allocatable = Allocatable(memory=0, gpu_memory={})
        for gpu_index, gpu in enumerate(worker.status.gpu_devices or []):
            if gpu.memory is None or gpu.memory.total is None:
                continue

            allocatable_gpu_memory = (
                gpu.memory.total
                - allocated.gpu_memory.get(gpu_index, 0)
                - worker.system_reserved.gpu_memory
            )

            allocatable.gpu_memory[gpu_index] = allocatable_gpu_memory

        allocatable.memory = (
            worker.status.memory.total
            - allocated.memory
            - worker.system_reserved.memory
        )

        if is_unified_memory:
            allocatable.memory = (
                allocatable.memory
                - worker.system_reserved.gpu_memory
                - sum(allocated.gpu_memory.values())
            )

            # For UMA, we need to set the gpu memory to the minimum of
            # the caculated with max allow gpu memory and the allocatable memory.
            allocatable.gpu_memory[0] = min(
                allocatable.memory, allocatable.gpu_memory[0]
            )

        logger.debug(
            f"Worker {worker.name} reserved memory: {worker.system_reserved.memory}, "
            f"reserved gpu memory: {worker.system_reserved.gpu_memory}, "
            f"allocatable memory: {allocatable.memory}, "
            f"allocatable gpu memory: {allocatable.gpu_memory}"
        )
        return allocatable


# arr is a sorted list from smallest to largest
def binary_search(arr, target):
    """
    Binary search the target in the arr.
    """
    if len(arr) == 0:
        return -1

    if arr[0] > target:
        return -1

    if arr[-1] < target:
        return len(arr) - 1

    low, high = 0, len(arr) - 1

    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1

    return high
