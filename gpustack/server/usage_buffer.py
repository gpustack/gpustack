import asyncio
import logging
from typing import Dict
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.model_usage import ModelUsage
from gpustack.server.db import get_engine

logger = logging.getLogger(__name__)

usage_flush_buffer: Dict[str, ModelUsage] = {}


async def flush_usage_to_db():
    """
    Flush model usage records to the database periodically.
    """
    while True:
        await asyncio.sleep(5)

        if not usage_flush_buffer:
            continue

        local_buffer = dict(usage_flush_buffer)
        usage_flush_buffer.clear()

        try:
            async with AsyncSession(get_engine()) as session:
                for key, usage in local_buffer.items():
                    to_update = await ModelUsage.one_by_id(session, usage.id)
                    to_update.prompt_token_count = usage.prompt_token_count
                    to_update.completion_token_count = usage.completion_token_count
                    to_update.request_count = usage.request_count
                    session.add(to_update)

                await session.commit()
                logger.debug(f"Flushed {len(local_buffer)} usage records to DB")
        except Exception as e:
            logger.error(f"Error flushing usage to DB: {e}")
