"""Subscribe to ``GPUInstance`` + ``GPUInstancePersistentVolume`` pub/sub and
translate row-level events into semantic ``resource_events`` rows.

This is the *source of truth* for downstream collectors:

* ``ResourceUsageCollector`` reads ``phase_to_metered`` /
  ``phase_left_metered`` pairs to compute instance-runtime rollups.
* ``StorageUsageCollector`` reads ``created`` / ``deleted`` pairs (PV is
  lifecycle-gated, no phase state machine).
* The resource-events UI reads everything for the audit trail.

Only **metering-relevant transitions** are recorded — an in-memory per-resource
state guard dedups the storm of status updates the reconciler emits while a
phase stays the same, so we don't append a ``phase_to_metered`` row on every
heartbeat. The guard is per-process, so on startup ``_warmup_state`` rehydrates
it from ``resource_events`` (latest event per resource): without that, a restart
would re-emit a spurious ``phase_to_metered`` for an already-metered instance
— a duplicate "Metering Started" in the audit log AND a window-start reset that
drops the pre-restart / downtime tail from the rollup.

Leader-only: one logger across the cluster, else each transition is written
twice. Events are written via ``ResourceEvent.create`` (NOT raw ``session.add``)
so the row publishes onto the event bus the collectors subscribe to.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set, Tuple

from sqlalchemy import func
from sqlmodel import select

from gpustack.schemas.gpu_instance_persistent_volumes import GPUInstancePersistentVolume
from gpustack.schemas.gpu_instances import GPUInstance
from gpustack.schemas.resource_events import (
    EVENT_TYPE_CREATED,
    EVENT_TYPE_DELETED,
    EVENT_TYPE_PHASE_LEFT_METERED,
    EVENT_TYPE_PHASE_TO_METERED,
    RESOURCE_TYPE_PERSISTENT_VOLUME,
    ResourceEvent,
)
from gpustack.server.bus import EventType
from gpustack.server.db import async_session
from gpustack.server.services import ClusterService, PrincipalService
from gpustack.utils.resource_usage import (
    instance_resource_type,
    is_metered_phase,
    parse_accelerator_count,
)

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    """Naive UTC, matching ``TimestampsMixin._datetime_func`` convention."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _phase_of(instance: GPUInstance) -> Optional[str]:
    """Read ``status.phase`` defensively. Cross-instance bus replay can deliver
    ``status`` as a plain dict instead of :class:`GPUInstanceStatus`."""
    status = instance.status
    if status is None:
        return None
    if isinstance(status, dict):
        return status.get("phase")
    return getattr(status, "phase", None)


def _instance_resource_type(instance: GPUInstance) -> str:
    """Derive ``gpu_instance`` vs ``cpu_instance`` from the requested
    accelerator count. Defensive against ``spec`` arriving as a dict."""
    spec = instance.spec
    if isinstance(spec, dict):
        resources = spec.get("resources") or {}
        accelerator = (
            resources.get("accelerator") if isinstance(resources, dict) else None
        )
    else:
        resources = getattr(spec, "resources", None)
        accelerator = getattr(resources, "accelerator", None)
    return instance_resource_type(parse_accelerator_count(accelerator))


async def _resolve_principals(
    session,
    consumer_principal_id: Optional[int],
    creator_id: Optional[int],
    cluster_id: Optional[int],
) -> tuple[Optional[int], Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Resolve the three-principal snapshot for an event.

    ``consumer_principal_id`` (the resource's owning tenant = consumer) and
    ``cluster_id`` come from the live entity; the PROVIDER (``owner``) is the
    cluster's owner — clusters are shareable, so the tenant running on a shared
    cluster (consumer) differs from the cluster owner (provider). Without a
    cluster (e.g. PV) provider == consumer.

    Returns ``(owner_principal_id, owner_name, consumer_name, creator_name,
    cluster_name)``. Resolved once here (the only place that sees the live
    entities) so the names ride ``resource_events`` into ``metered_usage``
    without re-resolving. ids that no longer resolve (delete race) yield
    ``None`` — the id is still preserved for audit.

    Cluster and principal lookups go through the cached ``ClusterService`` /
    ``PrincipalService`` so a reconciler storm of events for the same tenant /
    cluster hits the in-memory cache instead of the DB. Names are slowly
    changing, so cache-TTL staleness here is acceptable.
    """
    cluster_name = None
    owner_principal_id = consumer_principal_id  # default: self-owned (no cluster)
    if cluster_id is not None:
        c = await ClusterService(session).get_by_id(cluster_id)
        if c is not None:
            cluster_name = c.name
            owner_principal_id = c.owner_principal_id

    principals = PrincipalService(session)
    # Per-call memo of the principal ROW so owner == consumer (self-owned)
    # resolves once even on a cold cache; the service cache then makes it cheap
    # across events. Caching the row (not just the name) lets the consumer kind
    # come from the same lookup — no extra round-trip.
    principal_cache: Dict[int, Any] = {}

    async def principal_of(pid: Optional[int]):
        if pid is None:
            return None
        if pid not in principal_cache:
            principal_cache[pid] = await principals.get_by_id(pid)
        return principal_cache[pid]

    async def name_of(pid: Optional[int]) -> Optional[str]:
        p = await principal_of(pid)
        return p.name if p else None

    owner_name = await name_of(owner_principal_id)
    consumer_name = await name_of(consumer_principal_id)
    creator_name = await name_of(creator_id)

    # Consumer principal kind (``org`` / ``user`` / ``group``) snapshot, so the
    # Organization breakdown can tag a personal (USER) consumer. Reuses the
    # cached principal row resolved for ``consumer_name`` above.
    consumer_principal = await principal_of(consumer_principal_id)
    consumer_kind: Optional[str] = None
    if consumer_principal is not None:
        kind = consumer_principal.kind
        consumer_kind = kind.value if hasattr(kind, "value") else kind

    return (
        owner_principal_id,
        owner_name,
        consumer_name,
        creator_name,
        cluster_name,
        consumer_kind,
    )


def _model_dump_safe(obj: Any) -> Any:
    """Serialize ``obj`` to JSON-safe types for ``spec_snapshot``."""
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json")
        except Exception:  # pragma: no cover - defensive
            return str(obj)
    return obj


class ResourceEventLogger:
    """Long-running task: bridges DB pub/sub → ``resource_events`` rows."""

    SOURCE = "resource_event_logger"

    def __init__(self) -> None:
        # resource_id → last-known metered state (open window or not).
        self._instance_metered: Dict[int, bool] = {}
        # resource_ids we've already written a ``created`` row for this process.
        self._instance_created: Set[int] = set()
        self._volume_created: Set[int] = set()

    async def start(self) -> None:
        # Rebuild the dedup guards from the event log BEFORE subscribing, so a
        # restart doesn't re-emit transitions for resources that were already
        # metered / created.
        await self._warmup_state()
        await asyncio.gather(
            self._run_instances(),
            self._run_persistent_volumes(),
        )

    async def _warmup_state(self) -> None:
        """Seed ``_instance_metered`` / ``_instance_created`` / ``_volume_created``
        from ``resource_events`` on startup.

        The dedup guards are per-process in-memory state. Without rehydrating
        them, a restart sees ``prev_metered=False`` for an instance that is in
        fact still metered, so the first post-restart status update re-emits a
        spurious ``phase_to_metered`` — a second "Metering Started" in the audit
        log with no intervening "Metering Stopped", AND it overwrites the
        collector's reconciled window start (line ``_handle_event``), dropping
        the pre-restart / downtime tail from the rollup. Seeding from the latest
        event per resource makes a restart a no-op: no duplicate event, the
        collector keeps the window it reconciled from the original
        ``phase_to_metered``.
        """
        try:
            async with async_session() as session:
                # Latest lifecycle event per (resource_type, resource_id),
                # reduced in SQL instead of loading the whole history: id is the
                # autoincrement append order, so MAX(id) per group is the most
                # recent event — one row per resource. resource_type is in the
                # group key because instance and PV ids are separate spaces and
                # can collide.
                latest_ids = (
                    select(func.max(ResourceEvent.id))
                    .where(
                        ResourceEvent.event_type.in_(
                            [
                                EVENT_TYPE_CREATED,
                                EVENT_TYPE_PHASE_TO_METERED,
                                EVENT_TYPE_PHASE_LEFT_METERED,
                                EVENT_TYPE_DELETED,
                            ]
                        )
                    )
                    .group_by(ResourceEvent.resource_type, ResourceEvent.resource_id)
                )
                events = (
                    await session.exec(
                        select(ResourceEvent).where(ResourceEvent.id.in_(latest_ids))
                    )
                ).all()
        except Exception:
            logger.exception("resource_event_logger: startup warmup failed")
            return

        # One row per (resource_type, resource_id) already — the latest event.
        latest: Dict[Tuple[str, int], ResourceEvent] = {}
        for e in events:
            if e.resource_id is not None:
                latest[(e.resource_type, e.resource_id)] = e

        for (resource_type, rid), e in latest.items():
            # A deleted resource is gone (ids aren't reused) — nothing to dedup.
            if e.event_type == EVENT_TYPE_DELETED:
                continue
            if resource_type == RESOURCE_TYPE_PERSISTENT_VOLUME:
                self._volume_created.add(rid)
            else:
                self._instance_created.add(rid)
                self._instance_metered[rid] = (
                    e.event_type == EVENT_TYPE_PHASE_TO_METERED
                )

    # ------------------------------------------------------------------
    # GPU / CPU instances — state-gated time-based (Pending/Ready/...)
    # ------------------------------------------------------------------

    async def _run_instances(self) -> None:
        async for event in GPUInstance.subscribe(source=self.SOURCE):
            if event.type == EventType.HEARTBEAT:
                continue
            if event.data is None:
                continue
            try:
                await self._handle_instance(event)
            except Exception:
                logger.exception(
                    "resource_event_logger: failed to handle GPUInstance event "
                    "type=%s id=%s",
                    event.type,
                    event.id,
                )

    async def _handle_instance(self, event) -> None:
        instance: GPUInstance = event.data
        rid = instance.id
        if rid is None:
            return
        new_phase = _phase_of(instance)
        now_metered = is_metered_phase(new_phase)
        resource_type = _instance_resource_type(instance)

        def emit(event_type: str, message: Optional[str] = None):
            return self._write_event(
                resource_type=resource_type,
                resource_id=rid,
                resource_name=instance.name,
                # The instance's owning tenant is the consumer; the
                # provider (owner) is derived from the cluster downstream.
                consumer_principal_id=instance.owner_principal_id,
                creator_id=instance.creator_id,
                cluster_id=instance.cluster_id,
                event_type=event_type,
                phase=new_phase,
                spec_snapshot=_model_dump_safe(instance),
                message=message,
            )

        if event.type == EventType.DELETED:
            if self._instance_metered.get(rid):
                await emit(EVENT_TYPE_PHASE_LEFT_METERED)
            await emit(EVENT_TYPE_DELETED, "instance deleted")
            self._instance_metered.pop(rid, None)
            self._instance_created.discard(rid)
            return

        # CREATED / UPDATED — write ``created`` once, then only on metered flips.
        if event.type == EventType.CREATED and rid not in self._instance_created:
            self._instance_created.add(rid)
            await emit(EVENT_TYPE_CREATED, "instance created")

        prev_metered = self._instance_metered.get(rid, False)
        if now_metered and not prev_metered:
            self._instance_metered[rid] = True
            await emit(EVENT_TYPE_PHASE_TO_METERED)
        elif not now_metered and prev_metered:
            self._instance_metered[rid] = False
            await emit(EVENT_TYPE_PHASE_LEFT_METERED)

    # ------------------------------------------------------------------
    # Persistent volumes — lifecycle-gated (created → ... → deleted)
    # ------------------------------------------------------------------

    async def _run_persistent_volumes(self) -> None:
        async for event in GPUInstancePersistentVolume.subscribe(source=self.SOURCE):
            if event.type == EventType.HEARTBEAT:
                continue
            if event.data is None:
                continue
            try:
                await self._handle_volume(event)
            except Exception:
                logger.exception(
                    "resource_event_logger: failed to handle PV event type=%s id=%s",
                    event.type,
                    event.id,
                )

    async def _handle_volume(self, event) -> None:
        volume: GPUInstancePersistentVolume = event.data
        rid = volume.id
        if rid is None:
            return

        def emit(event_type: str, message: Optional[str] = None):
            return self._write_event(
                resource_type=RESOURCE_TYPE_PERSISTENT_VOLUME,
                resource_id=rid,
                resource_name=volume.name,
                consumer_principal_id=volume.owner_principal_id,
                creator_id=volume.creator_id,
                cluster_id=None,  # PVs aren't pinned to a cluster → owner==consumer
                event_type=event_type,
                phase=None,
                spec_snapshot=_model_dump_safe(volume),
                message=message,
            )

        if event.type == EventType.DELETED:
            await emit(EVENT_TYPE_DELETED, "persistent volume deleted")
            self._volume_created.discard(rid)
            return

        # PV meters from creation regardless of attach; only ``created`` matters
        # to the collector. ``updated`` (attach/detach/resize) is intentionally
        # not recorded — it would add audit noise without affecting metering.
        if rid not in self._volume_created:
            self._volume_created.add(rid)
            await emit(EVENT_TYPE_CREATED, "persistent volume created")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def _write_event(
        self,
        *,
        resource_type: str,
        resource_id: Optional[int],
        resource_name: Optional[str],
        consumer_principal_id: Optional[int],
        creator_id: Optional[int],
        cluster_id: Optional[int],
        event_type: str,
        phase: Optional[str],
        spec_snapshot: Optional[dict] = None,
        message: Optional[str] = None,
    ) -> None:
        async with async_session() as session:
            (
                owner_principal_id,
                owner_name,
                consumer_name,
                creator_name,
                cluster_name,
                consumer_principal_kind,
            ) = await _resolve_principals(
                session, consumer_principal_id, creator_id, cluster_id
            )
            row = ResourceEvent(
                occurred_at=_utc_now(),
                owner_principal_id=owner_principal_id,
                owner_name=owner_name,
                consumer_principal_id=consumer_principal_id,
                consumer_name=consumer_name,
                consumer_principal_kind=consumer_principal_kind,
                creator_id=creator_id,
                creator_name=creator_name,
                cluster_id=cluster_id,
                cluster_name=cluster_name,
                resource_type=resource_type,
                resource_id=resource_id,
                resource_name=resource_name or "",
                event_type=event_type,
                event_message=message,
                phase=phase,
                spec_snapshot=spec_snapshot,
            )
            # Must go through ``create`` (not raw session.add) so the row
            # publishes onto the event bus the collectors subscribe to.
            await ResourceEvent.create(session, row)
