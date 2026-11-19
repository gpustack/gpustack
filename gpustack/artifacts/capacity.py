"""Shared Benchmark archive capacity, accounted in compressed bytes."""

from sqlalchemy import update
from sqlmodel import select

from gpustack.config.config import get_global_config
from gpustack.schemas.artifact_storage import ArtifactStorageCapacity as Capacity
from gpustack.server.db import async_session

_WARNING_THRESHOLD = 80


async def lock_capacity(session):
    """Lock the installation policy inside the caller's transaction."""
    # The schema migration seeds this singleton. UPDATE also serializes writes
    # on engines that do not implement SELECT FOR UPDATE.
    await session.exec(
        update(Capacity).where(Capacity.id == 1).values(used_bytes=Capacity.used_bytes)
    )
    capacity = await _read_capacity(session)
    if not capacity.initialized:
        config = get_global_config()
        capacity.max_bytes = getattr(config, 'artifact_storage_max_total_bytes', 0)
        capacity.initialized = True
        session.add(capacity)
    return capacity


async def _read_capacity(session):
    result = await session.exec(
        select(Capacity)
        .where(Capacity.id == 1)
        .execution_options(populate_existing=True)
    )
    return result.one()


def _usage(capacity):
    used, maximum = capacity.used_bytes, capacity.max_bytes
    percent = used * 100 / maximum if maximum else None
    state = 'unlimited'
    if maximum:
        state = 'normal'
        if used >= maximum:
            state = 'full'
        elif percent >= _WARNING_THRESHOLD:
            state = 'warning'
    return {
        'max_bytes': maximum,
        'unlimited': maximum == 0,
        'used_bytes': used,
        'remaining_bytes': max(0, maximum - used) if maximum else None,
        'usage_percent': round(percent, 2) if percent is not None else None,
        'state': state,
        'warning_threshold_percent': _WARNING_THRESHOLD,
    }


class CapacityManager:
    async def usage(self):
        async with async_session() as session:
            capacity = await _read_capacity(session)
            if not capacity.initialized:
                capacity = await lock_capacity(session)
                usage = _usage(capacity)
                await session.commit()
                return usage
            return _usage(capacity)

    async def set_limit(self, maximum):
        async with async_session() as session:
            capacity = await lock_capacity(session)
            capacity.max_bytes = maximum
            result = _usage(capacity)
            await session.commit()
            return result
