import asyncio
import logging

from sqlmodel.ext.asyncio.session import AsyncSession
from gpustack.schemas.dashboard import (
    GPUUtilizationInfo,
    MaxMinUtilizationInfo,
    WorkerUtilizationInfo,
)
from gpustack.schemas.gpu_devices import GPUDevice
from gpustack.schemas.workers import UtilizationInfo, Worker
from gpustack.schemas.system_load import SystemLoad
from gpustack.server.db import get_engine

logger = logging.getLogger(__name__)


def compute_cpu_load(workers: list[Worker]) -> UtilizationInfo:
    total_cpu = sum(worker.status.cpu.total for worker in workers)
    used_cpu = sum(
        worker.status.cpu.total * worker.status.cpu.utilization_rate / 100
        for worker in workers
        if worker.status.cpu.total >= 0
    )

    if total_cpu == 0:
        return UtilizationInfo(total=0, used=0, utilization_rate=0)

    return UtilizationInfo(
        total=total_cpu, used=used_cpu, utilization_rate=used_cpu / total_cpu * 100
    )


def compute_memory_load(workers: list[Worker]) -> UtilizationInfo:
    total_memory = sum(worker.status.memory.total for worker in workers)
    used_memory = sum(
        worker.status.memory.used for worker in workers if worker.status.memory.used
    )

    if total_memory == 0:
        return UtilizationInfo(total=0, used=0, utilization_rate=0)

    return UtilizationInfo(
        total=total_memory,
        used=used_memory,
        utilization_rate=used_memory / total_memory * 100,
    )


def compute_gpu_load(workers: list[Worker]) -> UtilizationInfo:
    total_gpu = sum(
        gpu.core.total
        for worker in workers
        for gpu in worker.status.gpu_devices
        if gpu.core.total >= 0
    )
    used_gpu = sum(
        gpu.core.total * gpu.core.utilization_rate / 100
        for worker in workers
        for gpu in worker.status.gpu_devices
        if gpu.core.total >= 0
    )

    if total_gpu == 0:
        return UtilizationInfo(total=0, used=0, utilization_rate=0)

    return UtilizationInfo(
        total=total_gpu, used=used_gpu, utilization_rate=used_gpu / total_gpu * 100
    )


def compute_gpu_memory_load(workers: list[Worker]) -> UtilizationInfo:
    total_gpu_memory = sum(
        gpu.memory.total
        for worker in workers
        for gpu in worker.status.gpu_devices
        if gpu.memory.total > 0
    )
    used_gpu_memory = sum(
        gpu.memory.used
        for worker in workers
        for gpu in worker.status.gpu_devices
        if gpu.memory.used
    )

    if total_gpu_memory == 0:
        return UtilizationInfo(total=0, used=0, utilization_rate=0)

    return UtilizationInfo(
        total=total_gpu_memory,
        used=used_gpu_memory,
        utilization_rate=used_gpu_memory / total_gpu_memory * 100,
    )


def compute_system_load(workers: list[Worker]) -> SystemLoad:
    load = SystemLoad()
    load.cpu = compute_cpu_load(workers)
    load.memory = compute_memory_load(workers)
    load.gpu = compute_gpu_load(workers)
    load.gpu_memory = compute_gpu_memory_load(workers)
    return load


def get_max_min_utilization_info(
    workers: list[Worker],
) -> MaxMinUtilizationInfo:
    if len(workers) <= 1:
        return None

    gpu_utils = []
    gpu_memory_utils = []
    util_info = MaxMinUtilizationInfo()
    util_info.min_cpu = util_info.max_cpu = worker_util_info(
        workers[0], is_memory=False
    )
    util_info.min_memory = util_info.max_memory = worker_util_info(workers[0])

    for worker in workers:
        if worker.status.gpu_devices:
            for gpu in worker.status.gpu_devices:
                gpu_utils.append(gpu_util_info(worker.name, gpu, is_memory=False))
                gpu_memory_utils.append(gpu_util_info(worker.name, gpu))

        if worker.status.cpu.utilization_rate < util_info.min_cpu.utilization_rate:
            util_info.min_cpu = worker_util_info(worker, is_memory=False)

        if (
            worker.status.memory.utilization_rate
            < util_info.min_memory.utilization_rate
        ):
            util_info.min_memory = worker_util_info(worker)

        if worker.status.cpu.utilization_rate > util_info.max_cpu.utilization_rate:
            util_info.max_cpu = worker_util_info(worker, is_memory=False)

        if (
            worker.status.memory.utilization_rate
            > util_info.max_memory.utilization_rate
        ):
            util_info.max_memory = worker_util_info(worker)

    if gpu_utils:
        sorted_gpu_utils = sorted(
            gpu_utils, key=lambda gpu_util: gpu_util.utilization_rate
        )

        sorted_gpu_memory_utils = sorted(
            gpu_memory_utils,
            key=lambda gpu_memory_util: gpu_memory_util.utilization_rate,
        )

        util_info.min_gpu = sorted_gpu_utils[0]
        util_info.min_gpu_memory = sorted_gpu_memory_utils[0]
        util_info.max_gpu = sorted_gpu_utils[-1]
        util_info.max_gpu_memory = sorted_gpu_memory_utils[-1]

    return util_info


def worker_util_info(worker: Worker, is_memory=True) -> WorkerUtilizationInfo:
    if is_memory:
        return WorkerUtilizationInfo(
            worker_name=worker.name,
            utilization_rate=worker.status.memory.utilization_rate,
            total=worker.status.memory.total,
        )

    return WorkerUtilizationInfo(
        worker_name=worker.name,
        utilization_rate=worker.status.cpu.utilization_rate,
        total=worker.status.cpu.total,
    )


def gpu_util_info(
    worker_name: str, gpu: GPUDevice, is_memory=True
) -> GPUUtilizationInfo:
    if is_memory:
        return GPUUtilizationInfo(
            worker_name=worker_name,
            index=gpu.index,
            utilization_rate=gpu.memory.utilization_rate,
            total=gpu.memory.total,
        )

    return GPUUtilizationInfo(
        worker_name=worker_name,
        index=gpu.index,
        utilization_rate=gpu.core.utilization_rate,
        total=gpu.core.total,
    )


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
