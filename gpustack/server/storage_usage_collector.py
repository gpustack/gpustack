"""Aggregate ``resource_events`` for persistent volumes into ``metered_usage``.

PV is lifecycle-gated: from CREATED to DELETED the volume occupies its
capacity **regardless of attachment** (created ⇒ metered, like AWS EBS). So
there is no phase state machine and no attached/dangling split — one
``storage.capacity`` row per (volume, day), ``quantity`` = capacity_mib ×
metered seconds.

Idempotency mirrors ``ResourceUsageCollector``: each row carries a
``settled_until`` high-water mark, so replay / tick overlap / restart never
double-counts.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

from sqlalchemy import func
from sqlmodel import select

from gpustack import envs
from gpustack.schemas.metered_usage import (
    METER_STORAGE_CAPACITY,
    RESOURCE_TYPE_PERSISTENT_VOLUME,
    UNIT_MIB_SECONDS,
    MeteredUsage,
)
from gpustack.schemas.resource_events import (
    EVENT_TYPE_CREATED,
    EVENT_TYPE_DELETED,
    RESOURCE_TYPE_PERSISTENT_VOLUME as EVENT_RESOURCE_TYPE_PV,
    ResourceEvent,
)
from gpustack.server.bus import EventType
from gpustack.server.db import async_session
from gpustack.server.resource_usage_collector import (
    _clamped_seconds,
    _naive_utc,
    _snapshot_dict,
)
from gpustack.utils.resource_usage import iter_utc_hour_segments, parse_quantity_to_mib

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


@dataclass
class _OpenVolume:
    volume_id: int
    volume_name: str
    volume_display_name: Optional[str]
    storage_type: Optional[str]
    owner_principal_id: Optional[int]
    owner_name: Optional[str]
    consumer_principal_id: Optional[int]
    consumer_name: Optional[str]
    creator_id: Optional[int]
    creator_name: Optional[str]
    window_start: datetime
    capacity_mib: int
    settled_through: Optional[datetime] = None


def _open_volume_from_event(evt: ResourceEvent) -> Optional[_OpenVolume]:
    if evt.resource_id is None:
        return None
    snap = _snapshot_dict(evt.spec_snapshot)
    spec = snap.get("spec") or {}
    return _OpenVolume(
        volume_id=evt.resource_id,
        volume_name=evt.resource_name or snap.get("name") or "",
        volume_display_name=snap.get("display_name"),
        storage_type=spec.get("type_") or spec.get("type"),
        owner_principal_id=evt.owner_principal_id,
        owner_name=evt.owner_name,
        consumer_principal_id=evt.consumer_principal_id,
        consumer_name=evt.consumer_name,
        creator_id=evt.creator_id,
        creator_name=evt.creator_name,
        window_start=_naive_utc(evt.occurred_at),
        capacity_mib=parse_quantity_to_mib(spec.get("capacity")),
    )


class StorageUsageCollector:
    """Long-running task: settle PV-capacity windows into ``metered_usage``."""

    SOURCE = "storage_usage_collector"

    def __init__(self) -> None:
        self._open: Dict[int, _OpenVolume] = {}
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        await self._reconcile_open_volumes()
        await asyncio.gather(self._run_events(), self._run_tick())

    async def _reconcile_open_volumes(self) -> None:
        """Rebuild open PV windows from ``resource_events`` on startup — a volume
        whose latest lifecycle event is ``created`` (no later ``deleted``) is
        still metering. Per-row ``settled_until`` keeps re-settle idempotent."""
        try:
            async with async_session() as session:
                events = (
                    await session.exec(
                        select(ResourceEvent)
                        .where(ResourceEvent.resource_type == EVENT_RESOURCE_TYPE_PV)
                        .where(
                            ResourceEvent.event_type.in_(
                                [EVENT_TYPE_CREATED, EVENT_TYPE_DELETED]
                            )
                        )
                        .order_by(ResourceEvent.resource_id, ResourceEvent.occurred_at)
                    )
                ).all()
            latest: Dict[int, ResourceEvent] = {}
            for e in events:
                if e.resource_id is not None:
                    latest[e.resource_id] = e
            for rid, e in latest.items():
                if e.event_type == EVENT_TYPE_CREATED:
                    vol = _open_volume_from_event(e)
                    if vol is not None and vol.capacity_mib > 0:
                        self._open[rid] = vol
            if self._open:
                await self._seed_settled_through(session)
                logger.info(
                    "storage_usage_collector: reconciled %d open volume(s) on startup",
                    len(self._open),
                )
        except Exception:
            logger.exception("storage_usage_collector: startup reconcile failed")

    async def _seed_settled_through(self, session) -> None:
        """Seed each rebuilt volume's ``settled_through`` from its row's persisted
        high-water mark (``MAX(settled_until)``), so a long-lived volume resumes
        from where it left off instead of re-iterating every hour back to
        ``window_start`` on restart. Pure perf hint — correctness still rests on
        the per-row ``settled_until`` clamp, so a missing value is safe."""
        hwm_rows = (
            await session.exec(
                select(
                    MeteredUsage.resource_id,
                    func.max(MeteredUsage.settled_until),
                )
                .where(
                    MeteredUsage.meter_key == METER_STORAGE_CAPACITY,
                    MeteredUsage.resource_id.in_(list(self._open.keys())),
                )
                .group_by(MeteredUsage.resource_id)
            )
        ).all()
        for rid, hwm in hwm_rows:
            vol = self._open.get(rid)
            if vol is not None and hwm is not None:
                vol.settled_through = _naive_utc(hwm)

    async def _run_events(self) -> None:
        async for event in ResourceEvent.subscribe(source=self.SOURCE):
            if event.type == EventType.HEARTBEAT:
                continue
            if event.data is None:
                continue
            resource_event: ResourceEvent = event.data
            if resource_event.resource_type != EVENT_RESOURCE_TYPE_PV:
                continue
            try:
                await self._handle_event(resource_event)
            except Exception:
                logger.exception(
                    "storage_usage_collector: failed to handle event id=%s "
                    "event_type=%s",
                    resource_event.id,
                    resource_event.event_type,
                )

    async def _handle_event(self, evt: ResourceEvent) -> None:
        async with self._lock:
            if evt.event_type == EVENT_TYPE_CREATED:
                vol = _open_volume_from_event(evt)
                if vol is not None and vol.capacity_mib > 0:
                    self._open[vol.volume_id] = vol
                return
            if evt.event_type == EVENT_TYPE_DELETED:
                if evt.resource_id is None:
                    return
                vol = self._open.pop(evt.resource_id, None)
                if vol is not None:
                    await self._settle_locked(vol, _naive_utc(evt.occurred_at))
            # UPDATED / attached / detached — audit-only, no rollup effect.

    async def _run_tick(self) -> None:
        interval = max(60, envs.STORAGE_USAGE_TICK_SECONDS)
        while True:
            await asyncio.sleep(interval)
            try:
                await self._tick_once()
            except Exception:
                logger.exception("storage_usage_collector: tick failed")

    async def _tick_once(self) -> None:
        async with self._lock:
            now = _utc_now()
            for volume_id, vol in list(self._open.items()):
                try:
                    await self._settle_locked(vol, now)
                except Exception:
                    logger.exception(
                        "storage_usage_collector: tick settle failed volume_id=%s",
                        volume_id,
                    )
        # Seal fully-elapsed buckets after settling (see MeteredUsage.seal_due).
        await self._seal_due(now)

    async def _seal_due(self, now: datetime) -> None:
        try:
            async with async_session() as session:
                await MeteredUsage.seal_due(
                    session,
                    METER_STORAGE_CAPACITY,
                    now,
                    envs.METERED_USAGE_SEAL_GRACE_SECONDS,
                )
        except Exception:
            logger.exception("storage_usage_collector: seal failed")

    async def _settle_locked(self, vol: _OpenVolume, end_ts: datetime) -> None:
        # All hour-segments of one settle share a single session/transaction;
        # the per-row settled_until clamp keeps it idempotent on retry.
        start = vol.window_start
        if vol.settled_through is not None and vol.settled_through > start:
            start = vol.settled_through
        segments = iter_utc_hour_segments(start, end_ts)
        if segments:
            async with async_session() as session:
                for bucket_start, seg_start, seg_end in segments:
                    await self._upsert_bucket(
                        session, vol, bucket_start, seg_start, seg_end
                    )
                await session.commit()
        if vol.settled_through is None or end_ts > vol.settled_through:
            vol.settled_through = end_ts

    async def _upsert_bucket(
        self,
        session,
        vol: _OpenVolume,
        bucket_start: datetime,
        seg_start: datetime,
        seg_end: datetime,
    ) -> None:
        if vol.capacity_mib <= 0:
            return
        row = (
            await session.exec(
                select(MeteredUsage).where(
                    MeteredUsage.meter_key == METER_STORAGE_CAPACITY,
                    MeteredUsage.resource_id == vol.volume_id,
                    MeteredUsage.bucket_start == bucket_start,
                )
            )
        ).first()

        prior = _naive_utc(row.settled_until) if row is not None else None
        add_seconds = _clamped_seconds(seg_start, seg_end, prior)
        delta_capacity = add_seconds * vol.capacity_mib

        if row is not None:
            # Sealed buckets are final — drop any late segment (see
            # MeteredUsage.seal_due) rather than mutate a metered row.
            if row.sealed_at is not None:
                if add_seconds > 0:
                    logger.warning(
                        "storage_usage_collector: dropping %ss for sealed "
                        "bucket volume_id=%s bucket_start=%s",
                        add_seconds,
                        vol.volume_id,
                        bucket_start,
                    )
                return
            if add_seconds > 0:
                row.quantity += delta_capacity
                row.settled_until = seg_end
            row.resource_name = vol.volume_name or row.resource_name
            if vol.volume_display_name is not None:
                row.resource_display_name = vol.volume_display_name
            if vol.owner_name is not None:
                row.owner_name = vol.owner_name
            if vol.consumer_name is not None:
                row.consumer_name = vol.consumer_name
            if vol.creator_name is not None:
                row.creator_name = vol.creator_name
            if vol.storage_type is not None:
                row.sku = vol.storage_type
            row.dimensions = {
                "storage_type": vol.storage_type,
                "capacity_mib": vol.capacity_mib,
            }
            session.add(row)
            return

        if add_seconds <= 0:
            return
        session.add(
            MeteredUsage(
                owner_principal_id=vol.owner_principal_id,
                owner_name=vol.owner_name,
                consumer_principal_id=vol.consumer_principal_id,
                consumer_name=vol.consumer_name,
                creator_id=vol.creator_id,
                creator_name=vol.creator_name,
                cluster_id=None,
                cluster_name=None,
                meter_key=METER_STORAGE_CAPACITY,
                resource_type=RESOURCE_TYPE_PERSISTENT_VOLUME,
                resource_id=vol.volume_id,
                resource_name=vol.volume_name or "",
                resource_display_name=vol.volume_display_name,
                sku=vol.storage_type,
                sku_count=1,
                dimensions={
                    "storage_type": vol.storage_type,
                    "capacity_mib": vol.capacity_mib,
                },
                bucket_start=bucket_start,
                quantity=delta_capacity,
                unit=UNIT_MIB_SECONDS,
                settled_until=seg_end,
            )
        )
