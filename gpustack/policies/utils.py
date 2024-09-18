import logging
from typing import List
from gpustack.policies.base import (
    Allocatable,
    Allocated,
)
from gpustack.schemas.models import (
    ModelInstance,
)
from gpustack.schemas.workers import Worker
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)


async def get_worker_allocatable_resource(  # noqa: C901
    engine: AsyncEngine, worker: Worker
) -> Allocatable:
    """
    Get the worker with the latest allocatable resources.
    """

    def update_allocated_vram(allocated, resource_claim):
        for gpu_index, vram in resource_claim.vram.items():
            allocated.vram[gpu_index] = allocated.vram.get(gpu_index, 0) + vram

    is_unified_memory = worker.status.memory.is_unified_memory
    model_instances = await get_worker_model_instances(engine, worker)
    allocated = Allocated(ram=0, vram={})

    for model_instance in model_instances:
        if model_instance.worker_id != worker.id:
            continue
        allocated.ram += model_instance.computed_resource_claim.ram or 0
        if model_instance.gpu_indexes:
            update_allocated_vram(allocated, model_instance.computed_resource_claim)

        if (
            model_instance.distributed_servers
            and model_instance.distributed_servers.rpc_servers
        ):
            for rpc_server in model_instance.distributed_servers.rpc_servers:
                if rpc_server.computed_resource_claim:
                    # rpc server only consider the vram
                    update_allocated_vram(allocated, rpc_server.computed_resource_claim)

    allocatable = Allocatable(ram=0, vram={})
    if worker.status.gpu_devices:
        for _, gpu in enumerate(worker.status.gpu_devices):
            gpu_index = gpu.index

            if gpu.memory is None or gpu.memory.total is None:
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
        # the caculated with max allow gpu memory and the allocatable memory.
        if allocatable.vram:
            allocatable.vram[0] = min(allocatable.ram, allocatable.vram[0])

    logger.debug(
        f"Worker {worker.name} reserved memory: {worker.system_reserved.ram}, "
        f"reserved gpu memory: {worker.system_reserved.vram}, "
        f"allocatable memory: {allocatable.ram}, "
        f"allocatable gpu memory: {allocatable.vram}"
    )
    return allocatable


async def get_worker_model_instances(
    engine: AsyncEngine, worker: Worker
) -> List[ModelInstance]:
    async with AsyncSession(engine) as session:
        model_instances = await ModelInstance.all_by_field(
            session, "worker_id", worker.id
        )
        return model_instances
