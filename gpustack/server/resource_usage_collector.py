"""Aggregate ``resource_events`` for GPU / CPU instances into ``metered_usage``.

Each instance's currently-open metered window is tracked in memory:
``phase_to_metered`` opens it, ``phase_left_metered`` / ``deleted`` closes
it. Settling a window writes its elapsed seconds (split across UTC hours)
into the single ``instance.uptime`` meter — one row per (instance, hour, billed
shape). ``quantity`` is wall-clock seconds (whole-machine SKU, NOT × card
count); ``sku_count`` is carried as a column (accelerator card count — possibly
fractional for a sliced card — or base-flavor unit count for CPU) so GPU-Hours
can be derived as SUM(quantity × sku_count).

A periodic tick keeps "this hour so far" fresh for instances that stay metered
for hours/days without a phase transition.

Where the billed shape comes from
---------------------------------
``sku`` is the running instance type's identity snapshot
(``gpu_instances.type_snapshot`` -> ``gpu_instance_types.snapshot``) verbatim: a
real reference key, not a derived string. The type row it points at also
supplies ``definition_snapshot``, the display name, and the hardware facets in
``dimensions``. The row is looked up ignoring ``deleted_at`` — a type can be
retired while instances still run on it, which is exactly why the projection
soft-deletes.

``sku_count`` is the *sellable capacity share* the request occupies: a whole
card is 1, a soft slice is its VRAM percentage, and a hardware partition is its
``memoryMib`` over the card's VRAM (see ``utils.resource_usage``). A partition
whose share cannot be established yet is NOT settled — falling back to a whole
card would silently overcharge — so the window is retried on later ticks.

Legacy fallback: instances created before ``type_snapshot`` existed have it
NULL and it cannot be backfilled, so those fall back to the ``description``
descriptor blob and then to the raw type name, tagging
``dimensions.sku_source`` either way.

Idempotency / recovery
-----------------------
Each rollup row carries a ``settled_until`` high-water mark. A settlement only
adds the slice of an hour-segment *after* the row's ``settled_until``, so
re-processing the same window (event replay, tick overlap, restart, stop→start
within an hour) never double-counts — the durable cursor lives on the row, not
only in memory.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Optional
from sqlalchemy import func
from sqlmodel import select

from gpustack import envs
from gpustack.schemas.gpu_instance_persistent_volumes import (
    GPUInstancePersistentVolume,
)
from gpustack.schemas.gpu_instance_types import GPUInstanceType
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
    SLICE_MODE_PROFILE,
    SLICE_MODE_WHOLE,
    WHOLE_CARD_MILLI,
    instance_sku,
    iter_utc_hour_segments,
    parse_accelerator_count,
    parse_gpu_descriptor,
    parse_gpu_type,
    parse_quantity_to_mib,
    parse_quantity_to_millicores,
    profile_memory_mib,
    slice_mode_of,
    slice_share_milli,
    sliced_sku_count,
)

logger = logging.getLogger(__name__)

_INSTANCE_RESOURCE_TYPES = (RESOURCE_TYPE_GPU_INSTANCE, RESOURCE_TYPE_CPU_INSTANCE)

# ``dimensions.sku_source`` — which source produced ``sku`` / the hardware
# facets. Anything other than ``type_snapshot`` means the authoritative catalog
# row was unavailable and the row is degraded; it is logged when it happens.
SKU_SOURCE_TYPE_SNAPSHOT = "type_snapshot"
SKU_SOURCE_DESCRIPTION = "description"
SKU_SOURCE_TYPE_NAME = "type_name"

# How many ticks an incomplete type lookup is retried before it is treated as a
# standing problem rather than a pending backfill. ``status.detail`` is filled in
# asynchronously by the operator (an instance can reach a metered phase before it
# lands), so some retrying is required; a bound is what makes a real
# misconfiguration surface instead of being retried in silence forever.
#
# What the bound does at that point depends on whether the window is billable —
# see ``_give_up_or_retry``. It is NOT a deadline for billing: sizing it against
# ``METERED_USAGE_SEAL_GRACE_SECONDS`` (past which the earliest hours can no
# longer be recovered) was considered and is wrong, because giving up on a window
# with no ``sku_count`` costs its WHOLE life, not the hours already sealed. The
# number therefore only has to be long enough that a slow-but-working backfill is
# not reported as a fault: 20 ticks is ~100 minutes at the default tick.
_TYPE_LOOKUP_MAX_ATTEMPTS = 20


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
    consumer_principal_kind: Optional[str]
    creator_id: Optional[int]
    creator_name: Optional[str]
    cluster_id: Optional[int]
    cluster_name: Optional[str]
    window_start: datetime
    sku: str
    gpu_count: int
    # Unit multiplier billed for this instance: accelerator card count for GPU
    # instances — fractional when the cards are sliced — base-flavor unit count
    # (e.g. 2 for a 2c4g instance on a 1c2g flavor) for CPU instances. Stored as
    # ``metered_usage.sku_count``. ``None`` means "not established yet" (a
    # hardware partition whose share is still unresolvable): such a window is
    # NOT settled, because defaulting to a whole card would overcharge up to 8x.
    sku_count: Optional[Decimal]
    dimensions: Dict[str, Any]

    # —— Instance type identity ——
    # ``type_snapshot`` is the authoritative reference into ``gpu_instance_types``
    # and is what ``sku`` carries. NULL for pre-v2.3.0 instances (unbackfillable),
    # which is the one case the legacy fallbacks exist for.
    type_snapshot: Optional[str] = field(default=None)
    definition_snapshot: Optional[str] = field(default=None)
    instance_type_name: Optional[str] = field(default=None)

    # —— Accelerator slicing request ——
    slice_mode: str = field(default=SLICE_MODE_WHOLE)
    partitioned_profile: Optional[str] = field(default=None)
    # Per-card billed share in thousandths; ``None`` until resolvable (profile
    # mode needs the type's ``memoryMib`` + card VRAM, both on ``status.detail``,
    # which the operator backfills asynchronously).
    share_milli: Optional[int] = field(default=None)

    # —— Type-lookup state (not persisted) ——
    # True while the type row still owes us something (row not found yet, or its
    # ``status.detail`` not backfilled). ``_tick_once`` retries, bounded by
    # ``_TYPE_LOOKUP_MAX_ATTEMPTS`` once the window is billable at all;
    # ``_upsert_bucket`` rewrites dimensions from the latest window every time, so
    # an open row catches up automatically.
    needs_type_lookup: bool = field(default=False)
    type_lookup_attempts: int = field(default=0)
    # Log-once latches. Both of these conditions are re-evaluated on every tick
    # (300s), so logging them per evaluation produced an unbounded stream of
    # identical lines — which buries the one line that carried information. They
    # are logged on the state CHANGE instead: entering the state, and leaving it.
    lookup_bound_logged: bool = field(default=False)
    deferral_logged: bool = field(default=False)

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


def _resolve_sku_count(
    gpu_count: int,
    cpu_milli: int,
    mem_mib: int,
    unit_cpu_milli: Optional[int],
    unit_memory_mib: Optional[int],
    share_milli: Optional[int] = WHOLE_CARD_MILLI,
) -> Optional[Decimal]:
    """Unit multiplier billed for an instance (``metered_usage.sku_count``).

    GPU instances bill per accelerator card, scaled by the per-card sellable
    share (``share_milli``, 1000 = whole card) so a sliced card bills a
    fraction. ``share_milli=None`` means the share is not established yet and
    the caller must not settle — returns ``None`` rather than silently rounding
    a slice up to a whole card.

    CPU instances bill per base-flavor unit: a 2c4g instance on a 1c2g flavor is
    2 units. Derive it from the instance totals over the flavor's per-unit spec,
    preferring CPU and falling back to memory, then to 1 when the unit spec is
    unknown (legacy flavors with no ``unitResources`` descriptor)."""
    if gpu_count and gpu_count > 0:
        if share_milli is None:
            return None
        return sliced_sku_count(gpu_count, share_milli)
    for total, unit in ((cpu_milli, unit_cpu_milli), (mem_mib, unit_memory_mib)):
        if total and unit and unit > 0:
            return Decimal(max(1, round(total / unit)))
    return Decimal(1)


def _open_window_from_event(  # noqa: C901
    evt: ResourceEvent,
) -> Optional[_OpenWindow]:
    """Build an ``_OpenWindow`` from a ``phase_to_metered`` event row.

    Pure (no DB): everything derivable from the event snapshot alone. The
    catalog-backed facets and a hardware partition's billed share need the
    instance type row, which ``_resolve_instance_type`` fills in afterwards.
    """
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

    # A phase_to_metered event with no usable spec (empty / NULL snapshot — e.g.
    # seed rows or a malformed event) carries neither resources nor a type.
    # Metering it would mint a bogus "0 vCPU / 0 GB" window (sku "cpu-0vcpu-0g")
    # that pollutes the Instance Types breakdown and, for GPU instances (the
    # resource_type is inherited from the event), leaks into GPU-Hours. Skip it.
    if not resources and not instance_type:
        logger.warning(
            "resource_usage_collector: skipping metering for resource_id=%s "
            "(%s) — event has no spec resources/type (empty snapshot)",
            evt.resource_id,
            evt.resource_type,
        )
        return None

    type_snapshot = snap.get("type_snapshot") or None

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

    # Accelerator slicing request. ``memory_percentage == 0`` means slicing is
    # disabled (exclusive whole card), so it maps to "whole", not "zero share".
    sliced_memory_pct = resources.get(
        "accelerator_sliced_memory_percentage"
    ) or resources.get("acceleratorSlicedMemoryPercentage")
    sliced_cores_pct = resources.get(
        "accelerator_sliced_cores_percentage"
    ) or resources.get("acceleratorSlicedCoresPercentage")
    partitioned_profile = resources.get(
        "accelerator_partitioned_profile"
    ) or resources.get("acceleratorPartitionedProfile")
    slice_mode = slice_mode_of(sliced_memory_pct, partitioned_profile)
    # A partition's share needs the type row's ``memoryMib`` + card VRAM, so it
    # stays unresolved here and is filled by ``_resolve_instance_type``.
    share_milli = (
        None
        if slice_mode == SLICE_MODE_PROFILE
        else slice_share_milli(memory_percentage=sliced_memory_pct)
    )

    dimensions = {
        # Zero cards is a fact about the request, so it is stated rather than
        # omitted — unlike the facets below, which describe a card.
        "gpu_count": gpu_count,
        "cpu_milli": cpu_milli,
        "memory_mib": mem_mib,
        "ephemeral_mib": ephemeral_mib,
        "local_storage_mib": local_storage_mib,
    }
    # Card facets, only where there is a card. On a CPU instance every one of
    # them was noise or worse: ``gpu_type`` held the regex's leftovers from a
    # CPU flavor name (``-generic-linux``), ``vram_mib`` was 0, and
    # ``slice_mode: whole`` / ``slice_share_milli: 1000`` claimed a whole card
    # was held exclusively — of nothing. Absent reads as "not applicable";
    # a zero reads as a measurement.
    if gpu_count > 0:
        # Overwritten by ``_resolve_instance_type`` with the accurate
        # ``spec.accelerator_group``; this regex over the type name is the
        # fallback for instances predating ``type_snapshot``.
        dimensions["gpu_type"] = gpu_type
        if vram_mib:
            dimensions["vram_mib"] = vram_mib
        dimensions["slice_mode"] = slice_mode
        # Slicing facets, so a bill can be audited without re-deriving the
        # share: ``slice_share_milli`` × card count must reproduce ``sku_count``.
        if sliced_memory_pct:
            dimensions["sliced_memory_percentage"] = sliced_memory_pct
        if sliced_cores_pct:
            dimensions["sliced_cores_percentage"] = sliced_cores_pct
        if partitioned_profile:
            dimensions["partitioned_profile"] = partitioned_profile
        if share_milli is not None:
            dimensions["slice_share_milli"] = share_milli
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

    sku_count = _resolve_sku_count(
        gpu_count,
        cpu_milli,
        mem_mib,
        descriptor.get("unit_cpu_milli"),
        descriptor.get("unit_memory_mib"),
        share_milli,
    )

    # ``sku`` is the type snapshot verbatim — no transformation, so a metered row
    # joins straight onto ``gpu_instance_types.snapshot``. Only a pre-v2.3.0
    # instance (NULL, unbackfillable) falls back to the legacy name-based sku,
    # and that is recorded on the row.
    if type_snapshot:
        sku = type_snapshot
    else:
        sku = instance_sku(instance_type, gpu_type, gpu_count, cpu_milli, mem_mib)
        dimensions["sku_source"] = (
            SKU_SOURCE_DESCRIPTION if descriptor else SKU_SOURCE_TYPE_NAME
        )

    return _OpenWindow(
        resource_id=evt.resource_id,
        resource_type=evt.resource_type,
        resource_name=evt.resource_name or snap.get("name") or "",
        resource_display_name=snap.get("display_name"),
        owner_principal_id=evt.owner_principal_id,
        owner_name=evt.owner_name,
        consumer_principal_id=evt.consumer_principal_id,
        consumer_name=evt.consumer_name,
        consumer_principal_kind=getattr(evt, "consumer_principal_kind", None),
        creator_id=evt.creator_id,
        creator_name=evt.creator_name,
        cluster_id=evt.cluster_id,
        cluster_name=evt.cluster_name,
        window_start=_naive_utc(evt.occurred_at),
        sku=sku,
        gpu_count=gpu_count,
        sku_count=sku_count,
        dimensions=dimensions,
        type_snapshot=type_snapshot,
        instance_type_name=instance_type,
        slice_mode=slice_mode,
        partitioned_profile=partitioned_profile,
        share_milli=share_milli,
        # A type snapshot always warrants a catalog lookup (it carries the
        # definition snapshot and the authoritative hardware facets); without one
        # there is nothing to look up.
        needs_type_lookup=bool(type_snapshot),
    )


def _detail_of(row: GPUInstanceType) -> Any:
    """The type's observed hardware descriptor (``status.detail``), or ``None``.

    ``status`` is declared with ``pydantic_column_type``, so it arrives as the
    validated model — recursively, so ``detail`` and everything under it are
    models too (verified on a cold session and on a tuple select, i.e. not the
    identity map handing back what was written). Every writer assigns a validated
    ``GPUInstanceTypeStatusPublic`` as well. So the walk below reads attributes,
    with no dict form to accommodate.
    """
    status = row.status
    return getattr(status, "detail", None) if status is not None else None


def _detail_attr(detail: Any, name: str) -> Any:
    """One attribute off a nested detail model, tolerating an absent parent.

    The nesting is optional at every level (``exclude_none`` responses, and a
    type that reports no slicing at all), so this exists to keep the chain in
    :func:`_resolve_profile_share` from needing a guard per hop.
    """
    return getattr(detail, name, None) if detail is not None else None


async def _resolve_instance_type(session, window: "_OpenWindow") -> None:  # noqa: C901
    """Fill the window's identity + hardware facets from the instance type row.

    This is the authoritative replacement for parsing the user-writable
    ``description`` blob: ``type_snapshot`` is a real reference, resolved against
    the unique ``gpu_instance_types.snapshot`` index in one hit.

    ``deleted_at`` is deliberately NOT filtered: a type can be retired while
    instances still run on it, and keeping the soft-deleted row resolvable is the
    whole reason the projection soft-deletes instead of hard-deleting.

    Resolves, when available:

    ==========================================  ==================================
    ``definition_snapshot`` / type name         the row's own columns
    ``dimensions.gpu_type``                     ``spec.accelerator_group``
    ``dimensions.unit_cpu_milli`` / ``…mib``    ``spec.unit_resources``
    ``dimensions.product`` / ``manufacturer`` / ``family``   ``status.detail.*``
    card VRAM + a partition's ``memoryMib``     ``status.detail.*``
    ==========================================  ==================================

    ``status.detail`` is backfilled asynchronously by the operator, so an
    instance can reach a metered phase before it exists. ``needs_type_lookup``
    stays set until everything this window needs has landed; ``_tick_once``
    retries (bounded) and ``_upsert_bucket`` rewrites dimensions from the latest
    window, so an open row catches up on its own.
    """
    if not window.type_snapshot:
        window.needs_type_lookup = False
        return
    window.type_lookup_attempts += 1
    row = (
        await session.exec(
            select(GPUInstanceType).where(
                GPUInstanceType.snapshot == window.type_snapshot
            )
        )
    ).first()
    if row is None:
        # The catalog has not projected this type yet (or the deployment has no
        # operator at all, in which case the table stays empty forever). Keep the
        # legacy-derived facets and retry; the sku is already correct either way,
        # since it IS the snapshot.
        # Mark the row so a hash sku sitting next to legacy-derived facets is
        # recognizable as degraded rather than looking like corrupt data.
        window.dimensions["type_unresolved"] = True
        _give_up_or_retry(
            window,
            f"instance type {window.type_snapshot} not found in the catalog",
        )
        return
    window.dimensions.pop("type_unresolved", None)

    window.instance_type_name = row.name or window.instance_type_name
    # Fall back to computing it when the column predates this row (upgrade: the
    # migration adds it NULL and only active rows get backfilled by the watch
    # re-LIST; a soft-deleted row is never re-LISTed). It is a pure function of
    # (name, spec), so the derived value equals the persisted one.
    window.definition_snapshot = (
        row.definition_snapshot or row.compute_definition_snapshot()
    )
    window.dimensions["sku_source"] = SKU_SOURCE_TYPE_SNAPSHOT

    spec = row.spec
    if spec is not None:
        # An accelerated type's ``unit_resources`` describes "the resources that
        # come with one card", so folding a card-less request into a unit count
        # is meaningless. Refuse to meter it rather than emit a number nobody can
        # defend on an invoice.
        if getattr(spec, "acceleratable", False) and window.gpu_count <= 0:
            window.needs_type_lookup = False
            window.sku_count = None
            logger.error(
                "resource_usage_collector: refusing to meter resource_id=%s — it "
                "requests 0 accelerators on the accelerated instance type %s, so "
                "there is no defensible unit to bill",
                window.resource_id,
                row.name,
            )
            return
        # ``accelerator_group`` (e.g. ``nvidia-a10g``) is the accurate card-pool
        # key and is byte-stable across operator versions — unlike the regex over
        # the type name, which mangles both naming schemes.
        if getattr(spec, "accelerator_group", None):
            window.dimensions["gpu_type"] = spec.accelerator_group
        unit = getattr(spec, "unit_resources", None)
        if unit is not None:
            unit_cpu = parse_quantity_to_millicores(getattr(unit, "cpu", None))
            unit_ram = parse_quantity_to_mib(getattr(unit, "ram", None))
            if unit_cpu:
                window.dimensions["unit_cpu_milli"] = unit_cpu
            if unit_ram:
                window.dimensions["unit_memory_mib"] = unit_ram

    detail = _detail_of(row)
    card_memory_mib = parse_quantity_to_mib(_detail_attr(detail, "memory"))
    for key in ("product", "manufacturer", "family"):
        value = _detail_attr(detail, key)
        if value:
            window.dimensions[key] = value
    if card_memory_mib and window.slice_mode != SLICE_MODE_PROFILE:
        # Whole card / soft slice: the VRAM shown is what this instance occupies,
        # not the whole card — a 25% slice of an 80G card reads 20G.
        window.dimensions["vram_mib"] = (
            card_memory_mib * (window.share_milli or WHOLE_CARD_MILLI)
        ) // WHOLE_CARD_MILLI

    if window.slice_mode == SLICE_MODE_PROFILE:
        _resolve_profile_share(window, detail, card_memory_mib)
        if window.share_milli is None:
            return
        if card_memory_mib:
            window.dimensions["vram_mib"] = (
                card_memory_mib * window.share_milli // WHOLE_CARD_MILLI
            )
        window.dimensions["slice_share_milli"] = window.share_milli

    _refresh_sku_count(window)

    # ``status.detail`` is backfilled asynchronously (a MODIFIED event, not the
    # initial ADDED), so an instance can be metered before it lands. A whole-card
    # or soft-sliced request is already BILLABLE without it — its share does not
    # depend on the hardware — but the display facets (product / card VRAM /
    # manufacturer) are missing, so keep retrying until they arrive rather than
    # declaring the lookup done. ``_upsert_bucket`` rewrites dimensions from the
    # latest window on every settle, so an already-open row catches up with no
    # history rewrite.
    #
    # Card-less requests are exempt, because ``card_memory_mib`` reads
    # ``status.detail.memory`` — the ACCELERATOR's VRAM. A CPU-only type has no
    # such field to fill (``GPUInstanceTypeCPU`` does not define one), so waiting
    # on it is waiting for something that never arrives: every CPU instance then
    # burned 20 catalog lookups and 20 warnings across ~100 minutes before giving
    # up on a type that had in fact resolved completely on the first attempt.
    if window.gpu_count > 0 and not card_memory_mib:
        _give_up_or_retry(
            window,
            f"instance type {row.name} has no status.detail yet "
            "(display facets incomplete; metering is unaffected)",
            fatal=False,
        )
        return
    window.needs_type_lookup = False


def _resolve_profile_share(
    window: "_OpenWindow", detail: Any, card_memory_mib: int
) -> None:
    """Resolve a hardware-partition request's per-card share from the type's
    aggregated profile list. Leaves ``share_milli`` ``None`` (so the window is
    not settled) when the profile or the card VRAM is not yet resolvable."""
    sliced = _detail_attr(detail, "sliced_detail")
    physical = _detail_attr(sliced, "physical")
    profiles = _detail_attr(physical, "profiles")
    mib = profile_memory_mib(profiles, window.partitioned_profile)
    share = slice_share_milli(
        partitioned_profile=window.partitioned_profile,
        profile_memory_mib=mib,
        card_memory_mib=card_memory_mib,
    )
    if share is None:
        _give_up_or_retry(
            window,
            f"partition profile {window.partitioned_profile!r} of instance type "
            f"{window.type_snapshot} has no resolvable memoryMib "
            f"(profile_mib={mib}, card_mib={card_memory_mib})",
        )
        return
    window.share_milli = share


def _refresh_sku_count(window: "_OpenWindow") -> None:
    """Recompute ``sku_count`` after the type row supplied the missing inputs
    (partition share, or a CPU flavor's real ``unitResources``)."""
    window.sku_count = _resolve_sku_count(
        window.gpu_count,
        window.dimensions.get("cpu_milli") or 0,
        window.dimensions.get("memory_mib") or 0,
        window.dimensions.get("unit_cpu_milli"),
        window.dimensions.get("unit_memory_mib"),
        window.share_milli,
    )


def _give_up_or_retry(
    window: "_OpenWindow", reason: str, *, fatal: bool = True
) -> None:
    """Log an unresolved type lookup, and stop retrying only where that is safe.

    Never invents a ``sku_count``: billing a partition as a whole card is an
    up-to-8x overcharge. The ``sku`` is unaffected — it is the snapshot itself,
    which is known without the lookup.

    The bound applies to a window that is ALREADY BILLABLE. There it is what it
    was designed to be: a ceiling on how long a row may go on looking incomplete
    while its display facets are chased. ``fatal=False`` marks exactly those
    cases, so exhausting the retries is a warning, not an error.

    It deliberately does NOT apply to a window with no ``sku_count``. Such a
    window cannot be settled at all, and nothing outside this function will ever
    set one — so clearing ``needs_type_lookup`` there does not stop chasing a
    cosmetic field, it drops the instance out of the usage report for the rest of
    its life, leaving one log line as the only trace. A missing row is far harder
    to notice than a wrong number. Retrying instead costs one catalog query per
    tick and keeps the elapsed time recoverable: ``window_start`` and the row's
    ``settled_until`` both stay put, so whatever has not sealed yet is still
    billed once the share resolves.

    Logging follows the state change, not the tick. Every attempt used to log,
    which put 20 identical lines per instance in the log — and would now be
    unbounded. Instead: one line when the retry starts, one when the bound is
    reached (which is where a real misconfiguration surfaces — the reason the
    bound was introduced), then silence.
    """
    if window.type_lookup_attempts < _TYPE_LOOKUP_MAX_ATTEMPTS:
        if window.type_lookup_attempts <= 1:
            logger.warning(
                "resource_usage_collector: instance type not fully resolved for "
                "resource_id=%s — %s. Retrying every tick.",
                window.resource_id,
                reason,
            )
        return
    if window.sku_count is None:
        if not window.lookup_bound_logged:
            window.lookup_bound_logged = True
            logger.error(
                "resource_usage_collector: resource_id=%s is NOT being metered "
                "after %d attempts — %s. Still retrying, and its elapsed time is "
                "held; but hours already sealed cannot be recovered, so this needs "
                "attention rather than time.",
                window.resource_id,
                window.type_lookup_attempts,
                reason,
            )
        return
    window.needs_type_lookup = False
    logger.log(
        logging.ERROR if fatal else logging.WARNING,
        "resource_usage_collector: giving up resolving the instance type for "
        "resource_id=%s after %d attempts — %s. Billing is unaffected "
        "(sku_count=%s); only the display facets stay incomplete.",
        window.resource_id,
        window.type_lookup_attempts,
        reason,
        window.sku_count,
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
                            await _resolve_instance_type(session, window)
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
                    # Resolve the instance type (sku facets + the billed share)
                    # and snapshot the persistent-volume size while the PV still
                    # exists — one session for both.
                    if window.needs_type_lookup or window.dimensions.get(
                        "persistent_name"
                    ):
                        async with async_session() as session:
                            await _resolve_instance_type(session, window)
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
            await self._retry_type_lookups()
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

    async def _retry_type_lookups(self) -> None:
        """Re-resolve windows whose instance type was not fully readable yet.

        The operator backfills ``status.detail`` asynchronously, so an instance
        can enter a metered phase before the card VRAM / partition profiles are
        known. Retrying here (rather than at open time only) is what lets those
        windows start being billed as soon as the data lands — ``_upsert_bucket``
        rewrites dimensions from the latest window, so an already-open row picks
        up the corrected facets without any history rewrite.

        Bounded by ``_TYPE_LOOKUP_MAX_ATTEMPTS`` only for a window that is already
        settleable — there the bound caps how long a cosmetic field is chased. A
        window with no ``sku_count`` is retried indefinitely, because giving up on
        it would drop the instance out of the usage report for the rest of its
        life (see ``_give_up_or_retry``). Caller holds ``self._lock``.
        """
        pending = [w for w in self._open.values() if w.needs_type_lookup]
        if not pending:
            return
        try:
            async with async_session() as session:
                for window in pending:
                    await _resolve_instance_type(session, window)
        except Exception:
            logger.exception("resource_usage_collector: type lookup retry failed")

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
        the transaction is retried.

        A window with no established ``sku_count`` is skipped entirely: that
        means a hardware partition whose share is still unresolvable (or a
        card-less request on an accelerated type), and there is no safe default —
        billing it as a whole card overcharges up to 8x.

        Its seconds are held rather than dropped: ``window_start`` and the row's
        ``settled_until`` both stay put, so the first settle after the share
        resolves picks up whatever has NOT sealed by then. Which is worth stating
        plainly rather than as a footnote: the seal grace is 15 minutes by default,
        so a deferral lasting longer than that — the common case, since it waits on
        an operator backfill — permanently loses its earliest hours. A deferral is
        a problem to fix, not a state to live in; ``_give_up_or_retry`` escalates
        it to ERROR once the retry bound passes.

        Logged on the state change only (entering the deferral, and recovering
        from it). This is evaluated on every tick, so per-evaluation logging was
        288 identical lines a day per stuck instance, with no new information in
        any of them."""
        if window.sku_count is None:
            if not window.deferral_logged:
                window.deferral_logged = True
                logger.warning(
                    "resource_usage_collector: deferring settlement for "
                    "resource_id=%s — the billed share is not established yet "
                    "(slice_mode=%s, profile=%s). Its elapsed time is held until "
                    "the share resolves.",
                    window.resource_id,
                    window.slice_mode,
                    window.partitioned_profile,
                )
            return
        if window.deferral_logged:
            window.deferral_logged = False
            logger.info(
                "resource_usage_collector: resource_id=%s is settleable again "
                "(sku_count=%s); billing its held time now",
                window.resource_id,
                window.sku_count,
            )
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

    async def _upsert_bucket(  # noqa: C901
        self,
        session,
        window: _OpenWindow,
        bucket_start: datetime,
        seg_start: datetime,
        seg_end: datetime,
    ) -> None:
        # Match the full natural key, INCLUDING the billed shape: a mid-hour
        # reconfiguration (4 cards -> 1, 25% -> 50%, or a switch to another
        # instance type) must land in its own row so each segment is priced by the
        # shape that was actually running, instead of the whole hour inheriting
        # whichever shape settled last.
        row = (
            await session.exec(
                select(MeteredUsage).where(
                    MeteredUsage.meter_key == METER_INSTANCE_UPTIME,
                    MeteredUsage.resource_id == window.resource_id,
                    MeteredUsage.bucket_start == bucket_start,
                    MeteredUsage.sku == window.sku,
                    MeteredUsage.sku_count == window.sku_count,
                )
            )
        ).first()

        # Clamp to the row's high-water mark — only count time after what's
        # already settled for this hour AND this shape. Makes replay / overlap
        # idempotent. Two shapes in one hour never overlap in time (a
        # reconfiguration goes through a non-metered Stopped phase), so clamping
        # them independently stays correct.
        prior = _naive_utc(row.settled_until) if row is not None else None
        add_seconds = _clamped_seconds(seg_start, seg_end, prior)

        if row is not None:
            # Sealed buckets are final — a late segment landing here would
            # corrupt an already-metered row, so drop it (and surface it). This
            # check MUST stay ahead of every mutation below: it is what makes
            # pre-upgrade history immune to any new logic.
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
            # Refresh DISPLAY snapshots from the latest window so renames show up
            # without rewriting history. ``sku`` / ``sku_count`` are deliberately
            # NOT refreshed here — they are pricing inputs, not display fields,
            # and they are part of the row's identity now: a different shape
            # matched no row above and got its own row instead.
            row.resource_name = window.resource_name or row.resource_name
            if window.resource_display_name is not None:
                row.resource_display_name = window.resource_display_name
            if window.owner_name is not None:
                row.owner_name = window.owner_name
            if window.consumer_name is not None:
                row.consumer_name = window.consumer_name
            if window.consumer_principal_kind is not None:
                row.consumer_principal_kind = window.consumer_principal_kind
            if window.creator_name is not None:
                row.creator_name = window.creator_name
            if window.cluster_name is not None:
                row.cluster_name = window.cluster_name
            if window.instance_type_name is not None:
                row.instance_type_name = window.instance_type_name
            if window.definition_snapshot is not None:
                row.definition_snapshot = window.definition_snapshot
            # dimensions IS refreshed: it is the display blob, and this is how an
            # open row picks up facets the operator backfilled late.
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
                consumer_principal_kind=window.consumer_principal_kind,
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
                sku_count=window.sku_count,
                definition_snapshot=window.definition_snapshot,
                instance_type_name=window.instance_type_name,
                dimensions=window.dimensions,
                bucket_start=bucket_start,
                quantity=add_seconds,
                unit=UNIT_SECONDS,
                settled_until=seg_end,
            )
        )
