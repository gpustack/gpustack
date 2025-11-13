import asyncio
import logging

from gpustack import envs
from gpustack.schemas.models import ModelInstance
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.server.db import get_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.server.services import ModelInstanceService
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
        self._engine = get_engine()
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
        async with AsyncSession(self._engine) as session:
            workers = await Worker.all(session)
            if not workers:
                return

            offline_worker_names = {}
            for worker in workers:
                if worker.state == WorkerStateEnum.NOT_READY and (
                    not worker.maintenance or not worker.maintenance.enabled
                ):
                    offline, last_heartbeat_str = is_offline(
                        worker.heartbeat_time,
                        envs.MODEL_INSTANCE_RESCHEDULE_GRACE_PERIOD,
                    )
                    if offline:
                        offline_worker_names[worker.name] = last_heartbeat_str

            for worker_name, last_heartbeat_str in offline_worker_names.items():
                instances = await ModelInstance.all_by_field(
                    session, "worker_name", worker_name
                )
                if not instances:
                    continue

                instance_names = await ModelInstanceService(session).batch_delete(
                    instances
                )

                reschedule_minutes = envs.MODEL_INSTANCE_RESCHEDULE_GRACE_PERIOD / 60
                logger.info(
                    f"Worker {worker_name} is in NOT_READY state for more than "
                    f"{reschedule_minutes:.1f} minutes (last heartbeat at {last_heartbeat_str}) "
                    "and is not in maintenance mode. "
                    f"The following instances {', '.join(instance_names)} have been deleted and will be automatically rescheduled on other available nodes."
                )
