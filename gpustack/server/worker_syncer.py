import asyncio
from datetime import datetime, timedelta, timezone
import logging
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.server.db import get_engine

logger = logging.getLogger(__name__)


class WorkerSyncer:
    """
    WorkerSyncer syncs worker status periodically.
    """

    def __init__(self, interval=60, worker_offline_timeout=180):
        self._engine = get_engine()
        self._interval = interval
        self._worker_offline_timeout = worker_offline_timeout

    async def start(self):
        while True:
            await asyncio.sleep(self._interval)
            try:
                await self._sync_offline_workers()
            except Exception as e:
                logger.error(f"Failed to sync workers: {e}")

    async def _sync_offline_workers(self):
        """
        Mark offline workers to not_ready state.
        """
        async with AsyncSession(self._engine) as session:
            now = datetime.now(timezone.utc)
            three_minutes_ago = now - timedelta(seconds=self._worker_offline_timeout)
            statement = select(Worker).where(
                Worker.updated_at < three_minutes_ago,
                Worker.state == WorkerStateEnum.READY,
            )

            workers = (await session.exec(statement)).all()

            if not workers:
                return

            offline_worker_names = []
            for worker in workers:
                offline_worker_names.append(worker.name)
                worker.state = WorkerStateEnum.NOT_READY
                worker.state_message = "Heartbeat lost"
                await worker.update(session, worker)

            logger.debug(
                f"Marked worker {', '.join(offline_worker_names)} as not_ready"
            )
