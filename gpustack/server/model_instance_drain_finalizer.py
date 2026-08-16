"""Lightweight finalizer for gracefully drained model replicas.

When a model instance enters DRAINING (scale-down or DELETE), this loop waits
until the worker reports ``drain_idle`` or ``MODEL_INSTANCE_DRAIN_TIMEOUT``
elapses, then hard-deletes the row so ServeManager's existing reap path tears
down the workload. Intentionally not a CR-style multi-cluster finalizer.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import List

from gpustack import envs
from gpustack.schemas.models import ModelInstance, ModelInstanceStateEnum
from gpustack.server.db import async_session
from gpustack.server.services import ModelInstanceService

logger = logging.getLogger(__name__)


def _drain_timed_out(mi: ModelInstance, now: datetime, timeout_seconds: int) -> bool:
    if not mi.drain_started_at:
        # Missing timestamp: treat as already expired so we cannot stick forever.
        return True
    started = mi.drain_started_at
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    return (now - started).total_seconds() >= timeout_seconds


class ModelInstanceDrainFinalizer:
    """Periodically hard-deletes DRAINING instances that are idle or timed out."""

    def __init__(
        self,
        interval: float | None = None,
        timeout_seconds: int | None = None,
    ):
        self._interval = (
            interval
            if interval is not None
            else float(envs.MODEL_INSTANCE_DRAIN_FINALIZER_INTERVAL)
        )
        self._timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else envs.MODEL_INSTANCE_DRAIN_TIMEOUT
        )

    async def start(self):
        logger.debug("Model instance drain finalizer started.")
        while True:
            try:
                await self.reconcile()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Model instance drain finalizer error: {e}")
            await asyncio.sleep(self._interval)

    async def reconcile(self) -> List[str]:
        """Hard-delete draining instances that are ready. Returns deleted names."""
        deleted: List[str] = []
        now = datetime.now(timezone.utc)

        async with async_session() as session:
            draining = await ModelInstance.all_by_field(
                session,
                "state",
                ModelInstanceStateEnum.DRAINING,
                for_update=True,
            )
            if not draining:
                return deleted

            service = ModelInstanceService(session)
            for mi in draining:
                idle = bool(mi.drain_idle)
                timed_out = _drain_timed_out(mi, now, self._timeout_seconds)
                if not idle and not timed_out:
                    continue

                reason = "idle" if idle else "timeout"
                logger.info(
                    f"Hard-deleting draining model instance {mi.name} "
                    f"(id={mi.id}, reason={reason})"
                )
                await service.delete(mi)
                deleted.append(mi.name)

        return deleted
