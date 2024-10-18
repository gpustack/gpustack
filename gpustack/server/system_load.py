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
    util_count = sum(
        1
        for worker in workers
        for gpu in worker.status.gpu_devices or []
        if gpu.core and gpu.core.utilization_rate is not None
    )

    memory_count = sum(
        1
        for worker in workers
        for gpu in worker.status.gpu_devices or []
        if gpu.memory and gpu.memory.utilization_rate is not None
    )

    util_sum_value = sum(
        gpu.core.utilization_rate
        for worker in workers
        for gpu in worker.status.gpu_devices or []
        if gpu.core and gpu.core.utilization_rate is not None
    )

    memory_sum_value = sum(
        gpu.memory.utilization_rate
        for worker in workers
        for gpu in worker.status.gpu_devices or []
        if gpu.memory and gpu.memory.utilization_rate is not None
    )

    util_rate = util_sum_value / util_count if util_count > 0 else 0
    memory_rate = memory_sum_value / memory_count if memory_count > 0 else 0

    return util_rate, memory_rate


def compute_system_load(workers: list[Worker]) -> SystemLoad:
    cpu, ram = compute_avg_cpu_memory_utilization_rate(workers)
    gpu, vram = compute_avg_gpu_utilization_rate(workers)

    load = SystemLoad()
    load.cpu = cpu
    load.ram = ram
    load.gpu = gpu
    load.vram = vram
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
