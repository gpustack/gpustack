import asyncio
from datetime import datetime, timedelta, UTC
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

    def __init__(self, interval=60, worker_unknown_timeout=180):
        self._engine = get_engine()
        self._interval = interval
        self._worker_unknown_timeout = worker_unknown_timeout

    async def start(self):
        while True:
            await asyncio.sleep(self._interval)
            try:
                await self._sync_unresponsive_workers()
            except Exception as e:
                logger.error(f"Failed to sync workers: {e}")

    async def _sync_unresponsive_workers(self):
        """
        Mark workers which are not updated for a while to unknown state.
        """
        async with AsyncSession(self._engine) as session:
            now = datetime.now(UTC)
            three_minutes_ago = now - timedelta(seconds=self._worker_unknown_timeout)
            statement = select(Worker).where(
                Worker.updated_at < three_minutes_ago,
                Worker.state == WorkerStateEnum.running,
            )

            workers = (await session.exec(statement)).all()

            if not workers:
                return

            for worker in workers:
                worker.state = WorkerStateEnum.unknown
                session.add(worker)

            await session.commit()
            logger.debug(f"Marked {len(workers)} worker as unknown")
