import asyncio
from collections import defaultdict
import logging

from gpustack import envs
from gpustack.schemas.models import ModelInstance
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.server.db import async_session

from gpustack.server.services import ModelInstanceService
from gpustack.utils.model_instance_workers import get_model_instance_worker_match
from gpustack.utils.network import is_offline


logger = logging.getLogger(__name__)


class WorkerInstanceCleaner:
    """
    Periodically check offline workers and delete model instances.
    """

    def __init__(self, interval=30):
        """
        :param interval: loop interval in seconds
        """
        self._interval = interval

    async def start(self):
        while True:
            await asyncio.sleep(self._interval)
            try:
                await self._cleanup_offline_worker_instances()
            except Exception as e:
                logger.error(f"Failed to cleanup worker instances: {e}")

    async def _cleanup_offline_worker_instances(self):
        """
        Delete model instances assigned to offline workers.
        """
        async with async_session() as session:
            workers = await Worker.all(session)
            if not workers:
                return

            offline_workers = {}
            for worker in workers:
                if worker.state == WorkerStateEnum.NOT_READY and (
                    not worker.maintenance or not worker.maintenance.enabled
                ):
                    offline, last_heartbeat_str = is_offline(
                        worker.heartbeat_time,
                        envs.MODEL_INSTANCE_RESCHEDULE_GRACE_PERIOD,
                    )
                    if offline:
                        offline_workers[worker.name] = {
                            "id": worker.id,
                            "cluster_id": worker.cluster_id,
                            "last_heartbeat_str": last_heartbeat_str,
                        }

            if not offline_workers:
                return

            cluster_ids = {
                w["cluster_id"] for w in offline_workers.values() if w["cluster_id"]
            }
            if cluster_ids:
                instances = await ModelInstance.all_by_fields(
                    session,
                    extra_conditions=[ModelInstance.cluster_id.in_(cluster_ids)],
                )
            else:
                instances = await ModelInstance.all(session)
            if not instances:
                return

            instances_to_delete = []
            impacted_instances_by_worker = defaultdict(list)
            for instance in instances:
                impacted_worker_names = []
                for worker_name, worker_info in offline_workers.items():
                    match = get_model_instance_worker_match(
                        instance,
                        worker_name=worker_name,
                        worker_id=worker_info["id"],
                    )
                    if match.matched:
                        impacted_worker_names.append(worker_name)
                if not impacted_worker_names:
                    continue

                instances_to_delete.append(instance)
                for worker_name in impacted_worker_names:
                    impacted_instances_by_worker[worker_name].append(instance.name)

            if not instances_to_delete:
                return

            await ModelInstanceService(session).batch_delete(instances_to_delete)

            reschedule_minutes = envs.MODEL_INSTANCE_RESCHEDULE_GRACE_PERIOD / 60
            for worker_name, instance_names in impacted_instances_by_worker.items():
                last_heartbeat_str = offline_workers[worker_name]["last_heartbeat_str"]
                logger.info(
                    f"Worker {worker_name} is in NOT_READY state for more than "
                    f"{reschedule_minutes:.1f} minutes (last heartbeat at {last_heartbeat_str}) "
                    "and is not in maintenance mode. "
                    f"The following instances {', '.join(instance_names)} have been deleted and will be automatically rescheduled on other available nodes."
                )
