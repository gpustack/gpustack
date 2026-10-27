"""Resource accounting for one scheduling view, independent of workload storage."""

import logging
from typing import Dict, List, Optional

from gpustack.policies.base import Allocated, Allocatable
from gpustack.schemas.models import ModelInstance
from gpustack.schemas.workers import Worker

logger = logging.getLogger(__name__)


def compute_worker_allocated(
    all_model_instances: List[ModelInstance],
    worker_id: int,
    gpu_type: Optional[str] = None,
) -> Allocated:
    """Aggregate (ram, {gpu_index: vram}) for ``worker_id`` from current
    ModelInstance assignments — main worker + distributed subordinates.

    For the main worker, both ram and per-GPU vram are counted. For
    distributed subordinate workers, only vram is counted — the rpc-server
    side doesn't consume the model's RAM.
    """
    allocated = Allocated(ram=0, vram={})

    def add_vram(claim):
        for gpu_index, vram in (claim.vram or {}).items():
            allocated.vram[gpu_index] = allocated.vram.get(gpu_index, 0) + vram

    for mi in all_model_instances:
        if mi.worker_id == worker_id and (
            gpu_type is None or mi.gpu_type is None or mi.gpu_type == gpu_type
        ):
            claim = mi.computed_resource_claim
            if claim is not None:
                allocated.ram += claim.ram or 0
                if mi.gpu_indexes:
                    add_vram(claim)

        if mi.distributed_servers and mi.distributed_servers.subordinate_workers:
            for sw in mi.distributed_servers.subordinate_workers:
                if sw.worker_id != worker_id:
                    continue
                if sw.computed_resource_claim and (
                    gpu_type is None or mi.gpu_type == gpu_type
                ):
                    add_vram(sw.computed_resource_claim)

    return allocated


# TODO: With unified workloads, adapt every workload to a common claim input.
# Keep model placement metadata separate from resource accounting.
class ResourceView:
    """Combine model bindings and other workload reservations without querying storage.

    Derived views replace the model bindings for retries and capacity simulations,
    while retaining the same non-model reservations.
    """

    def __init__(
        self,
        model_instances: List[ModelInstance],
        reservations: Optional[Dict[int, Allocated]] = None,
    ) -> None:
        self._model_instances = model_instances
        self._reservations = {
            worker_id: Allocated(ram=claim.ram, vram=dict(claim.vram))
            for worker_id, claim in (reservations or {}).items()
        }

    def with_model_instances(self, instances: List[ModelInstance]) -> "ResourceView":
        """Replace model bindings, including any simulated assignments."""
        return ResourceView(instances, self._reservations)

    def has_reservation(self, worker_id: int) -> bool:
        """Whether this worker has a non-model resource reservation."""
        claim = self._reservations.get(worker_id)
        return bool(claim and (claim.ram or any(claim.vram.values())))

    def reservation_key(self) -> Dict[int, dict]:
        """Return the external reservations that affect an evaluation cache key."""
        return {
            worker_id: {"ram": claim.ram, "vram": dict(claim.vram)}
            for worker_id, claim in self._reservations.items()
        }

    def allocated(self, worker_id: int, gpu_type: Optional[str] = None) -> Allocated:
        """Sum model and non-model claims for a worker."""
        allocated = compute_worker_allocated(self._model_instances, worker_id, gpu_type)
        reservation = self._reservations.get(worker_id)
        if reservation:
            allocated.ram += reservation.ram
            for index, vram in reservation.vram.items():
                allocated.vram[index] = allocated.vram.get(index, 0) + vram
        return allocated

    def allocatable(
        self, worker: Worker, gpu_type: Optional[str] = None
    ) -> Allocatable:
        """Subtract reservations and system memory, including shared UMA capacity."""
        is_unified_memory = worker.status.memory.is_unified_memory
        allocated = self.allocated(worker.id, gpu_type)

        allocatable = Allocatable(ram=0, vram={})
        if worker.status.gpu_devices:
            for _, gpu in enumerate(worker.status.gpu_devices):
                gpu_index = gpu.index

                if (
                    gpu.memory is None
                    or gpu.memory.total is None
                    or (gpu_type is not None and gpu.type != gpu_type)
                ):
                    continue
                allocatable_vram = max(
                    (
                        gpu.memory.total
                        - allocated.vram.get(gpu_index, 0)
                        - worker.system_reserved.vram
                    ),
                    0,
                )
                allocatable.vram[gpu_index] = allocatable_vram

        allocatable.ram = max(
            (worker.status.memory.total - allocated.ram - worker.system_reserved.ram), 0
        )

        if is_unified_memory:
            allocatable.ram = max(
                allocatable.ram
                - worker.system_reserved.vram
                - sum(allocated.vram.values()),
                0,
            )

            # For UMA, we need to set the gpu memory to the minimum of
            # the calculated with max allow gpu memory and the allocatable memory.
            if allocatable.vram:
                allocatable.vram[0] = min(allocatable.ram, allocatable.vram[0])

        logger.debug(
            f"Worker {worker.name} gpu_type {gpu_type}, "
            f"reserved memory: {worker.system_reserved.ram}, "
            f"reserved gpu memory: {worker.system_reserved.vram}, "
            f"allocatable memory: {allocatable.ram}, "
            f"allocatable gpu memory: {allocatable.vram}"
        )
        return allocatable
