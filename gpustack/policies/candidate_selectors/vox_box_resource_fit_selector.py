import logging
from typing import List

from gpustack.policies.base import (
    ModelInstanceScheduleCandidate,
    ScheduleCandidatesSelector,
)
from gpustack.policies.utils import (
    get_worker_allocatable_resource,
)
from gpustack.schemas.models import (
    ComputedResourceClaim,
    Model,
    ModelInstance,
)
from gpustack.schemas.workers import VendorEnum, Worker

from gpustack.server.db import get_engine

logger = logging.getLogger(__name__)


class VoxBoxResourceFitSelector(ScheduleCandidatesSelector):
    def __init__(
        self,
        model: Model,
        model_instance: ModelInstance,
    ):
        self._engine = get_engine()
        self._model = model
        self._model_instance = model_instance

        # TODO(michelia): set ram/vram by estimating model
        self._ram_claim = 0
        self._vram_claim = 0

    async def select_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Get schedule candidates that fit the GPU resources requirement.
        """

        candidate_functions = [
            self.find_single_worker_single_gpu_candidates,
            self.find_single_worker_cpu_candidates,
        ]

        for candidate_func in candidate_functions:
            logger.debug(
                f"model {self._model.name}, filter candidates with resource fit selector: {candidate_func.__name__}, instance {self._model_instance.name}"
            )

            candidates = await candidate_func(workers)
            if candidates:
                return candidates

        return []

    async def find_single_worker_single_gpu_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find single worker single gpu candidates for the model instance with workers.
        """

        candidates = []
        for worker in workers:
            if not worker.status.gpu_devices:
                continue

            result = await self._find_single_worker_single_gpu_candidates(worker)
            if result:
                candidates.extend(result)

        return candidates

    async def _find_single_worker_single_gpu_candidates(
        self, worker: Worker
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find single worker single gpu candidates for the model instance with worker.
        requires: worker.status.gpu_devices is not None
        """
        candidates = []

        allocatable = await get_worker_allocatable_resource(self._engine, worker)
        is_unified_memory = worker.status.memory.is_unified_memory

        if self._ram_claim > allocatable.ram:
            return []

        if worker.status.gpu_devices:
            for _, gpu in enumerate(worker.status.gpu_devices):
                if gpu.vendor != VendorEnum.NVIDIA.value:
                    continue

                gpu_index = gpu.index
                allocatable_vram = allocatable.vram.get(gpu_index, 0)

                if gpu.memory is None or gpu.memory.total == 0:
                    continue

                if self._vram_claim > allocatable_vram:
                    continue

                candidates.append(
                    ModelInstanceScheduleCandidate(
                        worker=worker,
                        gpu_indexes=[gpu_index],
                        computed_resource_claim=ComputedResourceClaim(
                            vram={gpu_index: int(self._vram_claim)},
                            ram=self._ram_claim,
                            is_unified_memory=is_unified_memory,
                        ),
                    )
                )

        return candidates

    async def find_single_worker_cpu_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find single worker without offloading candidates for the model instance with workers.
        """
        candidates = []
        for worker in workers:
            result = await self._find_single_worker_with_cpu_candidates(worker)
            if result:
                candidates.extend(result)
        return candidates

    async def _find_single_worker_with_cpu_candidates(
        self, worker: Worker
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find single worker without offloading candidates for the model instance.
        """

        allocatable = await get_worker_allocatable_resource(self._engine, worker)
        is_unified_memory = worker.status.memory.is_unified_memory

        if self._ram_claim > allocatable.ram:
            return []

        return [
            ModelInstanceScheduleCandidate(
                worker=worker,
                gpu_indexes=None,
                computed_resource_claim=ComputedResourceClaim(
                    is_unified_memory=is_unified_memory,
                    vram=None,
                    ram=self._ram_claim,
                ),
            )
        ]
