from dataclasses import dataclass
import logging
from typing import Dict, List, Optional
from gpustack.scheduler.calculator import binary_search, estimate
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


@dataclass
class SystemReservedResource:
    memory: int
    gpu_memory: int


class ResourceFitPolicy:
    def __init__(self, estimate: estimate, system_reserved: SystemReservedResource):
        self._estimate = estimate
        self._system_reserved = system_reserved
        self._engine = get_engine()

    async def filter(
        self, workers: List[ModelInstanceScheduleCandidate]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Filter the workers with the resource fit claim.
        """

        candidates = []
        for worker in workers:
            candidate = await self._filterOne(worker.worker)
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    async def _filterOne(self, worker: Worker) -> ModelInstanceScheduleCandidate:
        """
        Find a candidate worker for the model instance.
        """

        is_unified_memory = worker.status.memory.is_unified_memory
        candidate: ModelInstanceScheduleCandidate = None

        allocatable = await self._get_worker_allocatable_resource(worker)

        if is_unified_memory:
            arr = []
            for memory in self._estimate.memory:
                arr.append(memory.uma)

            index = binary_search(arr, allocatable.memory)
            if index != -1:
                candidate = ModelInstanceScheduleCandidate(
                    worker=worker,
                    gpu_index=0,
                    computed_resource_claim=ComputedResourceClaim(
                        is_unified_memory=True,
                        offload_layers=index,
                        memory=self._estimate.memory[index].uma,
                    ),
                )

            return candidate

        else:
            arr = []
            for memory in self._estimate.memory:
                arr.append(memory.nonUMA.vram)

            for gpu_index in allocatable.gpu_memory:
                index = binary_search(arr, allocatable.gpu_memory[gpu_index])
                if (
                    index != -1
                    and allocatable.memory > self._estimate.memory[index].nonUMA.ram
                ):
                    candidate = ModelInstanceScheduleCandidate(
                        worker=worker,
                        gpu_index=gpu_index,
                        computed_resource_claim=ComputedResourceClaim(
                            is_unified_memory=False,
                            offload_layers=index,
                            gpu_memory=self._estimate.memory[index].nonUMA.vram,
                            memory=self._estimate.memory[index].nonUMA.ram,
                        ),
                    )
                    break
            return candidate

    async def _get_worker_allocatable_resource(self, worker: Worker) -> Allocatable:
        """
        Get the worker with the latest allocatable resources.
        """

        async with AsyncSession(self._engine) as session:
            model_instances = await ModelInstance.all_by_field(
                session, "worker_id", worker.id
            )

        allocated = Allocated(memory=0, gpu_memory={})
        for model_instance in model_instances:
            allocated.memory += model_instance.computed_resource_claim.memory or 0
            gpu_index = model_instance.gpu_index
            if gpu_index is not None:
                allocated.gpu_memory[gpu_index] = (
                    allocated.gpu_memory.get(gpu_index) or 0
                ) + (model_instance.computed_resource_claim.gpu_memory or 0)

        allocatable = Allocatable(memory=0, gpu_memory={})
        allocatable.memory = (
            worker.status.memory.total - self._system_reserved.memory - allocated.memory
        )

        for gpu_index, gpu in enumerate(worker.status.gpu):
            if gpu.memory is None or gpu.memory.total is None:
                continue

            allocatable.gpu_memory[gpu_index] = gpu.memory.total - (
                allocated.gpu_memory.get(gpu_index) or 0
            )

            if gpu_index == 0 and not worker.status.memory.is_unified_memory:
                allocatable.gpu_memory[gpu_index] = (
                    allocatable.gpu_memory.get(gpu_index)
                    - self._system_reserved.gpu_memory
                )

        return allocatable
