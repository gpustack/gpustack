import asyncio
import datetime
import logging
from typing import Set

from sqlalchemy import update

from gpustack.schemas.workers import Worker
from gpustack.server.db import async_session

logger = logging.getLogger(__name__)

# Buffer to store worker IDs that need heartbeat update
heartbeat_flush_buffer: Set[int] = set()
heartbeat_flush_buffer_lock = asyncio.Lock()


async def flush_heartbeats_to_db():
    """
    Flush worker heartbeat updates to the database periodically.
    Uses a single UPDATE statement to update all workers with the same timestamp.
    """
    while True:
        await asyncio.sleep(5)

        if not heartbeat_flush_buffer:
            continue

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
