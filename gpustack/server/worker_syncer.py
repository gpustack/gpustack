import asyncio
from datetime import datetime, timedelta, timezone
import logging
from sqlmodel.ext.asyncio.session import AsyncSession
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.server.db import get_engine
from gpustack.utils.network import is_url_reachable

logger = logging.getLogger(__name__)


class WorkerSyncer:
    """
    WorkerSyncer syncs worker status periodically.
    """

    def __init__(
        self, interval=60, worker_offline_timeout=180, worker_unreachable_timeout=10
    ):
        self._engine = get_engine()
        self._interval = interval
        self._worker_offline_timeout = worker_offline_timeout
        self._worker_unreachable_timeout = worker_unreachable_timeout

    async def start(self):
        while True:
            await asyncio.sleep(self._interval)
            try:
                await self._sync_worker_connectivity()
            except Exception as e:
                logger.error(f"Failed to sync workers: {e}")

    async def _sync_worker_connectivity(self):
        """
        Mark offline workers to not_ready state.
        """
        async with AsyncSession(self._engine) as session:
            now = datetime.now(timezone.utc)
            three_minutes_ago = now - timedelta(seconds=self._worker_offline_timeout)

            workers = await Worker.all(session)
            if not workers:
                return

            should_update_workers = []
            offline_worker_names = []
            unreachable_worker_names = []
            for worker in workers:
                if worker.updated_at < three_minutes_ago:
                    worker.state = WorkerStateEnum.NOT_READY
                    worker.state_message = "Heartbeat lost"

                    should_update_workers.append(worker)
                    offline_worker_names.append(worker.name)
                else:
                    worker_reachable = await self.is_worker_reachable(worker)
                    if not worker_reachable:
                        worker.state = WorkerStateEnum.UNREACHABLE
                        worker.state_message = "Worker is unreachable"

                        should_update_workers.append(worker)
                        unreachable_worker_names.append(worker.name)

            for worker in should_update_workers:
                await worker.update(session, worker)

            (
                logger.debug(
                    f"Marked worker {', '.join(offline_worker_names)} as not_ready"
                )
                if offline_worker_names
                else None
            )

            (
                logger.debug(
                    f"Marked worker {', '.join(unreachable_worker_names)} as unreachable"
                )
                if unreachable_worker_names
                else None
            )

    async def is_worker_reachable(
        self,
        worker: Worker,
    ) -> bool:
        healthz_url = f"http://{worker.ip}:{worker.port}/healthz"
        reachable = await is_url_reachable(
            healthz_url,
            self._worker_unreachable_timeout,
        )
        return reachable
