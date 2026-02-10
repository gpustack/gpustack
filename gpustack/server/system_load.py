import asyncio
import logging

from typing import Tuple, Dict, List
from gpustack.schemas.workers import Worker
from gpustack.schemas.system_load import SystemLoad
from gpustack.server.db import async_session

logger = logging.getLogger(__name__)


def workers_by_cluster_id(workers: List[Worker]) -> Dict[int, List[Worker]]:
    rtn: Dict[int, List[Worker]] = {}
    for worker in workers:
        if worker.cluster_id not in rtn:
            rtn[worker.cluster_id] = []
        rtn[worker.cluster_id].append(worker)
    return rtn


def _safe_cpu_rate(worker: Worker) -> float:
    if worker.status and worker.status.cpu and worker.status.cpu.utilization_rate:
        return worker.status.cpu.utilization_rate
    return 0.0


def _safe_memory_rate(worker: Worker) -> float:
    if worker.status and worker.status.memory and worker.status.memory.utilization_rate:
        return worker.status.memory.utilization_rate
    return 0.0


def compute_avg_cpu_memory_utilization_rate(
    workers: List[Worker],
) -> Dict[int | None, Tuple[float, float]]:
    rtn: Dict[int | None, Tuple[float, float]] = {
        None: (0, 0),
    }
    by_cluster = workers_by_cluster_id(workers)
    cpu_sum_value = 0
    memory_sum_value = 0
    for cluster_id, cluster_workers in by_cluster.items():
        cpu_value = sum(_safe_cpu_rate(worker) for worker in cluster_workers)
        memory_value = sum(_safe_memory_rate(worker) for worker in cluster_workers)
        rtn[cluster_id] = (
            cpu_value / len(cluster_workers),
            memory_value / len(cluster_workers),
        )
        cpu_sum_value += cpu_value
        memory_sum_value += memory_value

    if len(workers) > 0:
        cpu_rate = cpu_sum_value / len(workers)
        memory_rate = memory_sum_value / len(workers)
        rtn[None] = (cpu_rate, memory_rate)

    return rtn


def compute_avg_gpu_utilization_rate(
    workers: List[Worker],
) -> Dict[int | None, Tuple[float, float]]:
    by_cluster = workers_by_cluster_id(workers)
    rtn: Dict[int | None, Tuple[float, float]] = {}
    all_util_count = 0
    all_memory_count = 0
    all_util_sum_value = 0
    all_memory_sum_value = 0
    for cluster_id, cluster_workers in by_cluster.items():
        util_count = sum(
            1
            for worker in cluster_workers
            for gpu in worker.status.gpu_devices or []
            if gpu.core and gpu.core.utilization_rate is not None
        )

        memory_count = sum(
            1
            for worker in cluster_workers
            for gpu in worker.status.gpu_devices or []
            if gpu.memory and gpu.memory.utilization_rate is not None
        )

        util_sum_value = sum(
            gpu.core.utilization_rate
            for worker in cluster_workers
            for gpu in worker.status.gpu_devices or []
            if gpu.core and gpu.core.utilization_rate is not None
        )

        memory_sum_value = sum(
            gpu.memory.utilization_rate
            for worker in cluster_workers
            for gpu in worker.status.gpu_devices or []
            if gpu.memory and gpu.memory.utilization_rate is not None
        )
        util_rate = util_sum_value / util_count if util_count > 0 else 0
        memory_rate = memory_sum_value / memory_count if memory_count > 0 else 0
        rtn[cluster_id] = (util_rate, memory_rate)

        all_util_count += util_count
        all_memory_count += memory_count
        all_util_sum_value += util_sum_value
        all_memory_sum_value += memory_sum_value

    rtn[None] = (
        all_util_sum_value / all_util_count if all_util_count > 0 else 0,
        all_memory_sum_value / all_memory_count if all_memory_count > 0 else 0,
    )
    return rtn


def compute_system_load(workers: List[Worker]) -> List[SystemLoad]:
    workers = [worker for worker in workers if not worker.state.is_provisioning]
    cpu_memory_by_cluster = compute_avg_cpu_memory_utilization_rate(workers)
    gpu_vram_by_cluster = compute_avg_gpu_utilization_rate(workers)
    rtn: List[SystemLoad] = [
        SystemLoad(
            cluster_id=cluster_id,
            cpu=cpu_memory_by_cluster.get(cluster_id, (0, 0))[0],
            ram=cpu_memory_by_cluster.get(cluster_id, (0, 0))[1],
            gpu=gpu_vram_by_cluster.get(cluster_id, (0, 0))[0],
            vram=gpu_vram_by_cluster.get(cluster_id, (0, 0))[1],
        )
        for cluster_id in set(cpu_memory_by_cluster) | set(gpu_vram_by_cluster)
    ]
    return rtn


class SystemLoadCollector:
    def __init__(self, interval=60):
        self.interval = interval

    async def start(self):
        while True:
            await asyncio.sleep(self.interval)
            try:
                async with async_session() as session:
                    workers = await Worker.all(session=session)
                    system_loads = compute_system_load(workers)
                    session.add_all(system_loads)
                    await session.commit()
            except Exception as e:
                logger.error(f"Failed to collect system load: {e}")
