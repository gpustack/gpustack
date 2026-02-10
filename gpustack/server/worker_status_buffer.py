import asyncio
import datetime
import logging
from typing import Dict, Set

from sqlalchemy import update

from gpustack.schemas.workers import Worker
from gpustack.server.db import async_session
from gpustack.server.services import WorkerService

logger = logging.getLogger(__name__)

FLUSH_INTERVAL_SECONDS = 5

# Buffer to store worker IDs that need heartbeat update
heartbeat_flush_buffer: Set[int] = set()
heartbeat_flush_buffer_lock = asyncio.Lock()

# Buffer to store worker status updates: {worker_id: input_dict}
worker_status_flush_buffer: Dict[int, dict] = {}
worker_status_flush_buffer_lock = asyncio.Lock()


async def flush_heartbeats():
    """
    Flush worker heartbeat updates to the database periodically.
    Uses a single UPDATE statement to update all workers with the same timestamp.
    """
    if not heartbeat_flush_buffer:
        return

    # Copy buffer and clear it atomically
    async with heartbeat_flush_buffer_lock:
        local_buffer = set(heartbeat_flush_buffer)
        heartbeat_flush_buffer.clear()

    try:
        async with async_session() as session:
            # Single UPDATE for all workers with the same timestamp
            # UPDATE workers SET heartbeat_time = '2024-01-27 10:00:00' WHERE id IN (1, 2, 3, ...)
            heartbeat_time = datetime.datetime.now(datetime.timezone.utc).replace(
                microsecond=0
            )

            stmt = (
                update(Worker)
                .where(Worker.id.in_(local_buffer))
                .values(heartbeat_time=heartbeat_time)
            )

            await session.execute(stmt)
            await session.commit()
    except Exception as e:
        logger.error(f"Error flushing heartbeats to DB: {e}")


async def flush_worker_status():
    """
    Flush worker status updates to the database periodically.
    Uses batch_update to update multiple workers with different status data.
    """
    if not worker_status_flush_buffer:
        return

    async with worker_status_flush_buffer_lock:
        to_update_worker_ids = list(worker_status_flush_buffer.keys())
        to_update_worker_status = dict(worker_status_flush_buffer)
        worker_status_flush_buffer.clear()

    try:
        async with async_session() as session:
            # Query workers by ids
            workers = await Worker.all_by_fields(
                session=session, extra_conditions=[Worker.id.in_(to_update_worker_ids)]
            )

            for worker in workers:
                for key, value in to_update_worker_status.get(worker.id, {}).items():
                    setattr(worker, key, value)
                worker.compute_state()

            await WorkerService(session).batch_update(workers)
    except Exception as e:
        logger.error(f"Error flushing worker status to DB: {e}")


async def flush_worker_status_to_db():
    """
    Flush both worker heartbeats and status updates to the database periodically.
    """
    while True:
        await asyncio.sleep(FLUSH_INTERVAL_SECONDS)

        await flush_heartbeats()
        await flush_worker_status()
