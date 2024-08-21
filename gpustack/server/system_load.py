import asyncio
import logging

from typing import Tuple
from sqlmodel.ext.asyncio.session import AsyncSession
from gpustack.schemas.workers import Worker
from gpustack.schemas.system_load import SystemLoad
from gpustack.server.db import get_engine

logger = logging.getLogger(__name__)


def compute_avg_cpu_memory_utilization_rate(
    workers: list[Worker],
) -> Tuple[float, float]:
    count = len(workers)
    cpu_sum_value = sum(worker.status.cpu.utilization_rate for worker in workers)
    memory_sum_value = sum(worker.status.memory.utilization_rate for worker in workers)

    if count == 0:
        return 0, 0

    return cpu_sum_value / count, memory_sum_value / count


def compute_avg_gpu_utilization_rate(workers: list[Worker]) -> Tuple[float, float]:
    count = sum(len(worker.status.gpu_devices or []) for worker in workers)
    util_sum_value = sum(
        gpu.core.utilization_rate
        for worker in workers
        for gpu in worker.status.gpu_devices or []
    )
    memory_sum_value = sum(
        gpu.memory.utilization_rate
        for worker in workers
        for gpu in worker.status.gpu_devices or []
    )

    if count == 0:
        return 0, 0

    return util_sum_value / count, memory_sum_value / count


def compute_system_load(workers: list[Worker]) -> SystemLoad:
    cpu, memory = compute_avg_cpu_memory_utilization_rate(workers)
    gpu, gpu_memory = compute_avg_gpu_utilization_rate(workers)

    load = SystemLoad()
    load.cpu = cpu
    load.memory = memory
    load.gpu = gpu
    load.gpu_memory = gpu_memory
    return load


class SystemLoadCollector:
    def __init__(self, interval=60):
        self.interval = interval
        self._engine = get_engine()

    async def start(self):
        while True:
            await asyncio.sleep(self.interval)
            try:
                async with AsyncSession(self._engine) as session:
                    workers = await Worker.all(session=session)
                    system_load = compute_system_load(workers)
                    await SystemLoad.create(session, system_load)
            except Exception as e:
                logger.error(f"failed to collect system load: {e}")
