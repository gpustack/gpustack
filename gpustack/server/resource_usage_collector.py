"""Aggregate ``resource_events`` for GPU / CPU instances into ``metered_usage``.

Each instance's currently-open metered window is tracked in memory:
``phase_to_metered`` opens it, ``phase_left_metered`` / ``deleted`` closes
it. Settling a window writes its elapsed seconds (split across UTC midnights)
into the single ``instance.uptime`` meter — one row per (instance, day).
``quantity`` is wall-clock seconds (whole-machine SKU, NOT × card
count); ``sku_count`` is carried as a column (GPU card count, 1 for CPU) so
GPU-Hours can be derived as SUM(quantity × sku_count).

A periodic tick keeps "today so far" fresh for instances that stay metered
for hours/days without a phase transition.

Idempotency / recovery
-----------------------
Each rollup row carries a ``settled_until`` high-water mark. A settlement only
adds the slice of a day-segment *after* the row's ``settled_until``, so
re-processing the same window (event replay, tick overlap, restart, stop→start
within a day) never double-counts — the durable cursor lives on the row, not
only in memory.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from sqlalchemy import func
from sqlmodel import select

from gpustack import envs
from gpustack.schemas.gpu_instance_persistent_volumes import (
    GPUInstancePersistentVolume,
)
from gpustack.schemas.metered_usage import (
    METER_INSTANCE_UPTIME,
    UNIT_SECONDS,
    MeteredUsage,
)
from gpustack.schemas.resource_events import (
    EVENT_TYPE_DELETED,
    EVENT_TYPE_PHASE_LEFT_METERED,
    EVENT_TYPE_PHASE_TO_METERED,
    RESOURCE_TYPE_CPU_INSTANCE,
    RESOURCE_TYPE_GPU_INSTANCE,
    ResourceEvent,
)
from gpustack.server.bus import EventType
from gpustack.server.db import async_session
from gpustack.utils.resource_usage import (
    instance_sku,
    iter_utc_hour_segments,
    parse_accelerator_count,
    parse_gpu_descriptor,
    parse_gpu_type,
    parse_quantity_to_mib,
    parse_quantity_to_millicores,
)

logger = logging.getLogger(__name__)

_INSTANCE_RESOURCE_TYPES = (RESOURCE_TYPE_GPU_INSTANCE, RESOURCE_TYPE_CPU_INSTANCE)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _naive_utc(value: Optional[datetime]) -> Optional[datetime]:
    """Coerce a (possibly tz-aware, from ``UTCDateTime``) datetime to naive UTC
    so it compares cleanly against ``_utc_now()``."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _snapshot_dict(snap: Any) -> dict:
    """Coerce a ``spec_snapshot`` to a dict. It's a JSON column (normally a
    dict), but some drivers / bus replay paths can surface it as a raw JSON
    string — parse defensively so window extraction never AttributeErrors."""
    if isinstance(snap, str):
        try:
            snap = json.loads(snap)
        except (ValueError, TypeError):
            return {}
    return snap if isinstance(snap, dict) else {}


@dataclass
class _OpenWindow:
    """In-memory snapshot needed to settle one instance's open window.

    Captured at ``phase_to_metered`` so mid-flight spec changes don't
    retroactively re-rate. ``settled_through`` is an in-memory perf hint (skip
    already-iterated days); correctness comes from the per-row ``settled_until``
    clamp, so a stale/missing value is always safe.
    """

    resource_id: int
    resource_type: str
    resource_name: str
    resource_display_name: Optional[str]
    owner_principal_id: Optional[int]
    owner_name: Optional[str]
    consumer_principal_id: Optional[int]
    consumer_name: Optional[str]
    creator_id: Optional[int]
    creator_name: Optional[str]
    cluster_id: Optional[int]
    cluster_name: Optional[str]
    window_start: datetime
    sku: str
    gpu_count: int
    dimensions: Dict[str, Any]
    settled_through: Optional[datetime] = field(default=None)


def _clamped_seconds(
    seg_start: datetime, seg_end: datetime, prior_settled: Optional[datetime]
) -> int:
    """Seconds to add for a day-segment, clamped to the row's high-water mark.

    Only the slice of ``[seg_start, seg_end]`` *after* ``prior_settled`` counts,
    so re-processing an already-settled window (replay / tick overlap / restart /
    stop→start within a day) adds 0. Pure (no DB) so it's unit-testable.

    The count is the difference of two integer offsets from the hour's
    ``bucket_start`` rather than ``int(seg_end - effective_start)``. A bucket is
    filled by many ticks landing on sub-second boundaries; truncating each
    segment independently drops the fractional remainder every tick, so a full
    hour under-counts (e.g. 3599 instead of 3600). Anchoring the truncation to a
    fixed origin makes the per-segment losses telescope: the offsets cancel and
    a fully-covered hour sums to exactly 3600 (see issue #5710).
    """
    effective_start = seg_start
    if prior_settled is not None and prior_settled > effective_start:
        effective_start = prior_settled
    if seg_end <= effective_start:
        return 0
    bucket_start = seg_start.replace(minute=0, second=0, microsecond=0)
    end_offset = int((seg_end - bucket_start).total_seconds())
    start_offset = int((effective_start - bucket_start).total_seconds())
    seconds = end_offset - start_offset
    return seconds if seconds > 0 else 0


def _open_window_from_event(evt: ResourceEvent) -> Optional[_OpenWindow]:
    """Build an ``_OpenWindow`` from a ``phase_to_metered`` event row."""
    if evt.resource_id is None:
        return None
    snap = _snapshot_dict(evt.spec_snapshot)
    spec = snap.get("spec") or {}
    resources = spec.get("resources") or {}
    volume = spec.get("volume") or {}
    ephemeral = volume.get("ephemeral") or {}
    persistent = volume.get("persistent") or {}

    # model_dump may serialize the field by name (``type_``) or alias
    # (``type``) depending on by_alias — read both for robustness.
    instance_type = spec.get("type_") or spec.get("type")
    gpu_type, _ = parse_gpu_type(instance_type)
    gpu_count = parse_accelerator_count(resources.get("accelerator"))
    cpu_milli = parse_quantity_to_millicores(resources.get("cpu"))
    mem_mib = parse_quantity_to_mib(resources.get("ram"))
    ephemeral_mib = parse_quantity_to_mib(ephemeral.get("capacity"))
    # System (OS) disk — the GPU Instances list shows it under Disk → System.
    # The snapshot is model_dump(mode="json") (by field name), so read the
    # snake_case field; fall back to the camelCase alias for robustness.
    local_storage_mib = parse_quantity_to_mib(
        resources.get("local_storage") or resources.get("localStorage")
    )
    # Pretty product name + per-card cpu/mem/vram for the "Instance Type" display
    # (so Usage matches the GPU Instances list instead of the raw flavor slug).
    # Per-card VRAM rides in the same descriptor blob, so read it from there
    # rather than parsing spec.memory a second time.
    descriptor = parse_gpu_descriptor(snap.get("description"))
    vram_mib = descriptor.get("vram_mib", 0)

    dimensions = {
        "gpu_type": gpu_type,
        "gpu_count": gpu_count,
        "vram_mib": vram_mib,
        "cpu_milli": cpu_milli,
        "memory_mib": mem_mib,
        "ephemeral_mib": ephemeral_mib,
        "local_storage_mib": local_storage_mib,
    }
    # Persistent data disk is a reference to a separate PV resource (only its
    # name is in the instance spec) — store the name so the breakdown can
    # resolve its provisioned capacity for the Disk → Persistent row.
    if persistent.get("name"):
        dimensions["persistent_name"] = persistent["name"]
    if descriptor.get("product"):
        dimensions["product"] = descriptor["product"]
    if descriptor.get("unit_cpu_milli"):
        dimensions["unit_cpu_milli"] = descriptor["unit_cpu_milli"]
    if descriptor.get("unit_memory_mib"):
        dimensions["unit_memory_mib"] = descriptor["unit_memory_mib"]

    return _OpenWindow(
        resource_id=evt.resource_id,
        resource_type=evt.resource_type,
        resource_name=evt.resource_name or snap.get("name") or "",
        resource_display_name=snap.get("display_name"),
        owner_principal_id=evt.owner_principal_id,
        owner_name=evt.owner_name,
        consumer_principal_id=evt.consumer_principal_id,
        consumer_name=evt.consumer_name,
        creator_id=evt.creator_id,
        creator_name=evt.creator_name,
        cluster_id=evt.cluster_id,
        cluster_name=evt.cluster_name,
        window_start=_naive_utc(evt.occurred_at),
        sku=instance_sku(instance_type, gpu_type, gpu_count, cpu_milli, mem_mib),
        gpu_count=gpu_count,
        dimensions=dimensions,
    )


async def _resolve_persistent_mib(session, window: "_OpenWindow") -> None:
    """Resolve the referenced persistent volume's capacity into the window's
    dimensions (``persistent_mib``), once, at metering time.

    The instance spec only references the PV by name; its size lives on the
    separate PV resource. Resolving it here — while the PV is guaranteed to
    exist — and snapshotting it onto the metered rows means the Usage breakdown
    keeps showing the size even after the PV is later deleted (unlike resolving
    lazily at read time). PV names are unique per principal, so match on owner.

    ``persistent_name`` is consumed (popped) here — it's only an internal lookup
    key, so it shouldn't bloat every persisted metered row.
    """
    name = window.dimensions.pop("persistent_name", None)
    if not name or window.owner_principal_id is None:
        return
    pv = (
        await session.exec(
            select(GPUInstancePersistentVolume).where(
                GPUInstancePersistentVolume.owner_principal_id
                == window.owner_principal_id,
                GPUInstancePersistentVolume.name == name,
            )
        )
    ).first()
    spec = getattr(pv, "spec", None) if pv else None
    cap = getattr(spec, "capacity", None)
    if cap is None and isinstance(spec, dict):
        cap = spec.get("capacity")
    mib = parse_quantity_to_mib(cap) if cap else 0
    if mib:
        window.dimensions["persistent_mib"] = mib


class ResourceUsageCollector:
    """Long-running task: settle instance uptime windows into ``metered_usage``."""

    SOURCE = "resource_usage_collector"

    def __init__(self) -> None:
        self._open: Dict[int, _OpenWindow] = {}
        # One mutex guards ``_open`` + rollup writes; tick and event paths can
        # both touch the same instance. Contention is low.
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        await self._reconcile_open_windows()
        await asyncio.gather(self._run_events(), self._run_tick())

    async def _reconcile_open_windows(self) -> None:
        """Rebuild in-memory open windows from ``resource_events`` on startup.

        An instance that was already metered before this process started won't
        produce a fresh event we're guaranteed to catch (subscribe-replay timing
        / restart races), so seed ``_open`` from the latest lifecycle event per
        resource: if it's a ``phase_to_metered`` with no later close, the window
        is open. Per-row ``settled_until`` keeps the subsequent settle idempotent.
        """
        try:
            # One session for the whole rebuild: the events query AND the
            # settled_through seeding share it (seeding on a closed session
            # would raise, get swallowed below, and silently skip the
            # high-water resume — the very thing it exists to do).
            async with async_session() as session:
                # Latest lifecycle event per instance, reduced in SQL: id is the
                # autoincrement append order, so MAX(id) per resource_id is the
                # most recent — one row per resource instead of the full history.
                # Instance ids are a single space here, so group by resource_id.
                latest_ids = (
                    select(func.max(ResourceEvent.id))
                    .where(ResourceEvent.resource_type.in_(_INSTANCE_RESOURCE_TYPES))
                    .where(
                        ResourceEvent.event_type.in_(
                            [
                                EVENT_TYPE_PHASE_TO_METERED,
                                EVENT_TYPE_PHASE_LEFT_METERED,
                                EVENT_TYPE_DELETED,
                            ]
                        )
                    )
                    .group_by(ResourceEvent.resource_id)
                )
                events = (
                    await session.exec(
                        select(ResourceEvent).where(ResourceEvent.id.in_(latest_ids))
                    )
                ).all()
                latest: Dict[int, ResourceEvent] = {}
                for e in events:
                    if e.resource_id is not None:
                        latest[e.resource_id] = e  # one row per resource (latest)
                for rid, e in latest.items():
                    if e.event_type == EVENT_TYPE_PHASE_TO_METERED:
                        window = _open_window_from_event(e)
                        if window is not None:
                            await _resolve_persistent_mib(session, window)
                            self._open[rid] = window
                if self._open:
                    await self._seed_settled_through(session)
                    logger.info(
                        "resource_usage_collector: reconciled %d open window(s) "
                        "on startup",
                        len(self._open),
                    )
        except Exception:
            logger.exception("resource_usage_collector: startup reconcile failed")

    async def _seed_settled_through(self, session) -> None:
        """Seed each rebuilt window's ``settled_through`` from its row's persisted
        high-water mark (``MAX(settled_until)``).

        Without this, a restart resets ``settled_through`` to ``None`` and the
        first settle re-iterates every hour back to ``window_start`` — for an
        instance that's been metered for weeks that's hundreds of redundant
        bucket lookups per restart. Resuming from the high-water mark skips the
        already-settled history. Purely a startup perf hint: correctness still
        rests on the per-row ``settled_until`` clamp, so a missing/stale value
        is always safe (we just re-scan a bit more)."""
        hwm_rows = (
            await session.exec(
                select(
                    MeteredUsage.resource_id,
                    func.max(MeteredUsage.settled_until),
                )
                .where(
                    MeteredUsage.meter_key == METER_INSTANCE_UPTIME,
                    MeteredUsage.resource_id.in_(list(self._open.keys())),
                )
                .group_by(MeteredUsage.resource_id)
            )
        ).all()
        for rid, hwm in hwm_rows:
            window = self._open.get(rid)
            if window is not None and hwm is not None:
                window.settled_through = _naive_utc(hwm)

    # ------------------------------------------------------------------
    # Event-driven path
    # ------------------------------------------------------------------

    async def _run_events(self) -> None:
        async for event in ResourceEvent.subscribe(source=self.SOURCE):
            if event.type == EventType.HEARTBEAT:
                continue
            if event.data is None:
                continue
            resource_event: ResourceEvent = event.data
            if resource_event.resource_type not in _INSTANCE_RESOURCE_TYPES:
                continue
            try:
                await self._handle_event(resource_event)
            except Exception:
                logger.exception(
                    "resource_usage_collector: failed to handle event id=%s "
                    "event_type=%s",
                    resource_event.id,
                    resource_event.event_type,
                )

    async def _handle_event(self, evt: ResourceEvent) -> None:
        async with self._lock:
            if evt.event_type == EVENT_TYPE_PHASE_TO_METERED:
                window = _open_window_from_event(evt)
                if window is not None:
                    # Snapshot the persistent-volume size while the PV still
                    # exists (only for instances that reference one).
                    if window.dimensions.get("persistent_name"):
                        async with async_session() as session:
                            await _resolve_persistent_mib(session, window)
                    # Replace any stale window (missed close during a crash);
                    # the per-row settled_until absorbs the older time safely.
                    self._open[window.resource_id] = window
                return

            if evt.event_type in (EVENT_TYPE_PHASE_LEFT_METERED, EVENT_TYPE_DELETED):
                if evt.resource_id is None:
                    return
                window = self._open.pop(evt.resource_id, None)
                if window is None:
                    return
                await self._settle_locked(window, _naive_utc(evt.occurred_at))

    # ------------------------------------------------------------------
    # Periodic tick
    # ------------------------------------------------------------------

    async def _run_tick(self) -> None:
        interval = max(60, envs.RESOURCE_USAGE_TICK_SECONDS)
        while True:
            await asyncio.sleep(interval)
            try:
                await self._tick_once()
            except Exception:
                logger.exception("resource_usage_collector: tick failed")

    async def _tick_once(self) -> None:
        async with self._lock:
            now = _utc_now()
            for resource_id, window in list(self._open.items()):
                try:
                    await self._settle_locked(window, now)
                except Exception:
                    logger.exception(
                        "resource_usage_collector: tick settle failed id=%s",
                        resource_id,
                    )
        # Seal fully-elapsed buckets *after* settling, so a still-running
        # instance's current hour is written before it becomes eligible.
        await self._seal_due(now)

    async def _seal_due(self, now: datetime) -> None:
        try:
            async with async_session() as session:
                await MeteredUsage.seal_due(
                    session,
                    METER_INSTANCE_UPTIME,
                    now,
                    envs.METERED_USAGE_SEAL_GRACE_SECONDS,
                )
        except Exception:
            logger.exception("resource_usage_collector: seal failed")

    # ------------------------------------------------------------------
    # Settlement core
    # ------------------------------------------------------------------

    async def _settle_locked(self, window: _OpenWindow, end_ts: datetime) -> None:
        """Settle ``[window_start, end_ts]`` into per-hour rollup rows, clamping
        each hour-segment against the row's persisted ``settled_until``.

        All hour-segments of one settle share a single session/transaction —
        a long backfill (e.g. restart after days down) is one commit, not one
        per hour. The per-row ``settled_until`` clamp keeps it idempotent if
        the transaction is retried."""
        start = window.window_start
        if window.settled_through is not None and window.settled_through > start:
            start = window.settled_through
        segments = iter_utc_hour_segments(start, end_ts)
        if segments:
            async with async_session() as session:
                for bucket_start, seg_start, seg_end in segments:
                    await self._upsert_bucket(
                        session, window, bucket_start, seg_start, seg_end
                    )
                await session.commit()
        if (
            end_ts > (window.settled_through or end_ts)
            or window.settled_through is None
        ):
            window.settled_through = end_ts

    async def _upsert_bucket(
        self,
        session,
        window: _OpenWindow,
        bucket_start: datetime,
        seg_start: datetime,
        seg_end: datetime,
    ) -> None:
        row = (
            await session.exec(
                select(MeteredUsage).where(
                    MeteredUsage.meter_key == METER_INSTANCE_UPTIME,
                    MeteredUsage.resource_id == window.resource_id,
                    MeteredUsage.bucket_start == bucket_start,
                )
            )
        ).first()

        # Clamp to the row's high-water mark — only count time after what's
        # already settled for this hour. Makes replay / overlap idempotent.
        prior = _naive_utc(row.settled_until) if row is not None else None
        add_seconds = _clamped_seconds(seg_start, seg_end, prior)

        if row is not None:
            # Sealed buckets are final — a late segment landing here would
            # corrupt an already-metered row, so drop it (and surface it).
            if row.sealed_at is not None:
                if add_seconds > 0:
                    logger.warning(
                        "resource_usage_collector: dropping %ss for sealed "
                        "bucket resource_id=%s bucket_start=%s",
                        add_seconds,
                        window.resource_id,
                        bucket_start,
                    )
                return
            if add_seconds > 0:
                row.quantity += add_seconds
                row.settled_until = seg_end
            # Refresh display snapshots from the latest window so renames /
            # spec changes show up without rewriting history.
            row.resource_name = window.resource_name or row.resource_name
            if window.resource_display_name is not None:
                row.resource_display_name = window.resource_display_name
            if window.owner_name is not None:
                row.owner_name = window.owner_name
            if window.consumer_name is not None:
                row.consumer_name = window.consumer_name
            if window.creator_name is not None:
                row.creator_name = window.creator_name
            if window.cluster_name is not None:
                row.cluster_name = window.cluster_name
            row.sku = window.sku or row.sku
            row.sku_count = window.gpu_count or 1
            row.dimensions = window.dimensions
            session.add(row)
            return

        if add_seconds <= 0:
            return
        session.add(
            MeteredUsage(
                owner_principal_id=window.owner_principal_id,
                owner_name=window.owner_name,
                consumer_principal_id=window.consumer_principal_id,
                consumer_name=window.consumer_name,
                creator_id=window.creator_id,
                creator_name=window.creator_name,
                cluster_id=window.cluster_id,
                cluster_name=window.cluster_name,
                meter_key=METER_INSTANCE_UPTIME,
                resource_type=window.resource_type,
                resource_id=window.resource_id,
                resource_name=window.resource_name or "",
                resource_display_name=window.resource_display_name,
                sku=window.sku,
                sku_count=window.gpu_count or 1,
                dimensions=window.dimensions,
                bucket_start=bucket_start,
                quantity=add_seconds,
                unit=UNIT_SECONDS,
                settled_until=seg_end,
            )
        )
