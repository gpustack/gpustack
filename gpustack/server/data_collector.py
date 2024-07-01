import asyncio
from datetime import date
import logging

from sqlmodel.ext.asyncio.session import AsyncSession
from gpustack.schemas.model_usage import ModelUsage, ResourceClaim
from gpustack.schemas.models import ModelInstance
from gpustack.schemas.workers import UtilizationInfo, Worker
from gpustack.schemas.system_load import SystemLoad
from gpustack.server.db import get_engine

logger = logging.getLogger(__name__)


def compute_cpu_load(workers: list[Worker]) -> UtilizationInfo:
    total_cpu = sum(worker.status.cpu.total for worker in workers)
    used_cpu = sum(
        # TODO use cpu.used when available
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
        # TODO use gpu.core.used when available
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


class DataCollector:
    def __init__(self, interval=60):
        self.interval = interval
        self._engine = get_engine()

    async def start(self):
        while True:
            await asyncio.sleep(self.interval)
            try:
                await self.collect_data()
            except Exception as e:
                logger.error(f"Failed to collect data: {e}")

    async def collect_data(self):
        tasks = [
            self.collect_system_load(),
            self.collect_model_usage(),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for task, result in zip(
            ["collect_system_load", "collect_model_usage"], results
        ):
            if isinstance(result, Exception):
                raise Exception(f"Failed to run {task}: {result}")

    async def collect_system_load(self):
        async with AsyncSession(self._engine) as session:
            workers = await Worker.all(session=session)
            system_load = compute_system_load(workers)
            await SystemLoad.create(session, system_load)

    async def collect_model_usage(self):
        async with AsyncSession(self._engine) as session:
            model_usages = await ModelUsage.all_by_field(
                session=session, field="date", value=date.today()
            )

        if len(model_usages) == 0:
            return

        for model_usage in model_usages:
            model_instances = await ModelInstance.all_by_field(
                session=session, field="model_id", value=model_usage.model_id
            )

            claim_memory = 0
            claim_gpu_memory = 0
            for instance in model_instances:
                if instance.computed_resource_claim is None:
                    continue

                if instance.computed_resource_claim.memory is not None:
                    claim_memory += instance.computed_resource_claim.memory
                if instance.computed_resource_claim.gpu_memory is not None:
                    claim_gpu_memory += instance.computed_resource_claim.gpu_memory

            if (
                model_usage.resource_claim is not None
                and model_usage.resource_claim.memory == claim_memory
                and model_usage.resource_claim.gpu_memory == claim_gpu_memory
            ):
                continue

            model_usage.resource_claim = ResourceClaim(
                memory=claim_memory, gpu_memory=claim_gpu_memory
            )
            await model_usage.update(session)
