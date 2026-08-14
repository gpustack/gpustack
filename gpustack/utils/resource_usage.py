"""Helpers shared by the resource-usage collectors / event-logger.

Kept tiny and dependency-free so they can be unit-tested without spinning up
the rest of the server.
"""

from __future__ import annotations

import json
import math
import re
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Metered phase predicate
# ---------------------------------------------------------------------------

# Phases where the instance is NOT holding an accelerator, so metering must be
# off. Two groups (literal copies of ``schemas.gpu_instances.GPUInstancePhase``
# values — kept as literals to keep this layer import-light, no schema/server
# dependency; see test_non_metered_phases_match_schema):
#   * terminal create failures — the accelerator never came up / was released;
#   * the stop / delete lifecycle (Stopping/Stopped from an explicit stop,
#     Deleting on the way out) — the accelerator is released, so the user is
#     not charged while stopped.
# Everything else non-empty IS metered: Ready / NotReady (up, maybe degraded),
# Starting (coming up — reacquiring the reservation, like the initial bring-up),
# and Unknown (status unreadable → assume still allocated, conservative).
_NON_METERED_PHASES = frozenset(
    {
        "CreateFailed",
        "InitializeFailed",
        "SSHPublicKeyCreateFailed",
        "PersistentVolumeCreateFailed",
        "PersistentVolumeTypeCreateFailed",
        "Stopping",
        "Stopped",
        "Deleting",
    }
)


def is_metered_phase(phase: Optional[str]) -> bool:
    """Return True if ``phase`` represents a state that consumes GPU.

    The plan's "kueue admits → resource is reserved" model: any non-empty phase
    is metered unless it's in :data:`_NON_METERED_PHASES`. ``None`` covers
    brand-new rows whose reconciler hasn't run yet; the non-metered set covers
    failures and the stop/delete lifecycle, where the accelerator is released.
    """
    return phase is not None and phase not in _NON_METERED_PHASES


# ---------------------------------------------------------------------------
# GPU type parser
# ---------------------------------------------------------------------------

# Kueue queue names look like ``gpustack-nvidia-geforce-rtx-4090-c9bjn`` —
# a stable ``gpustack-{manufacturer}-{model}`` prefix plus a short random
# suffix kueue appends to make queue names unique per cluster. Suffix is
# always lowercase alphanum, 5-6 chars in practice; we match 4-8 to leave a
# little slack without swallowing legitimate model fragments.
_QUEUE_PREFIX = "gpustack-"
_QUEUE_SUFFIX_RE = re.compile(r"-[a-z0-9]{4,8}$")
_KNOWN_MANUFACTURERS = frozenset(
    {"nvidia", "amd", "ascend", "intel", "moore", "iluvatar"}
)


def parse_gpu_type(queue_name: Optional[str]) -> Tuple[str, Optional[str]]:
    """Parse a kueue queue name into ``(gpu_type, manufacturer)``.

    .. deprecated::
        Fallback only. The accurate card-pool key is the instance type's
        ``spec.accelerator_group`` (e.g. ``nvidia-a10g``, ``ascend-910b2``),
        which is byte-stable across operator versions — reach it via
        ``type_snapshot``.

        This regex was written for kueue queue names (``gpustack-<vendor>-<model>``
        plus a short random suffix) and produces garbage on an operator
        InstanceType name, which uses a ``--`` separator and no random suffix::

            >>> parse_gpu_type("gpustack--generic--ascend-910b2-linux-arm64")
            ('-generic--ascend-910b2-linux', None)

        Note the leading ``-``, the ``None`` manufacturer, and the swallowed
        ``arm64`` — the suffix stripper mistook the arch for a random suffix.
        Kept unchanged (rather than "fixed") because historical rows were
        written with exactly this behaviour and it is only reached when there is
        no type row to consult.

    Examples
    --------
    >>> parse_gpu_type("gpustack-nvidia-geforce-rtx-4090-c9bjn")
    ('nvidia-geforce-rtx-4090', 'nvidia')
    >>> parse_gpu_type("gpustack-amd-mi300x-ab12c")
    ('amd-mi300x', 'amd')
    >>> parse_gpu_type(None)
    ('unknown', None)
    """
    if not queue_name:
        return ("unknown", None)
    s = queue_name
    if s.startswith(_QUEUE_PREFIX):
        s = s[len(_QUEUE_PREFIX) :]
    s = _QUEUE_SUFFIX_RE.sub("", s)
    if not s:
        return ("unknown", None)
    head = s.split("-", 1)[0].lower()
    if head in _KNOWN_MANUFACTURERS:
        return (s, head)
    return (s, None)


def parse_gpu_vram_mib(description: Any) -> int:
    """Per-card GPU VRAM in MiB, parsed from a GPUInstance ``description`` blob.

    .. deprecated::
        Legacy fallback only. The authoritative source is the instance type row
        reached via ``gpu_instances.type_snapshot``
        (``status.detail.memory``); ``description`` is a 1024-char *user* text
        field that only carries this JSON because the UI chooses to write it.
        It stays because pre-upgrade instances have a NULL ``type_snapshot``
        that cannot be backfilled.

    ``description`` is the device descriptor — usually a JSON string (sometimes
    already a dict) shaped like ``{"spec": {"memory": "48Gi", ...}}``. Note the
    flat shape is the pre-v2.3.0 InstanceType layout: the operator moved these
    observed fields to ``status.detail``, and the UI flattens them back for
    compatibility. Returns 0 when absent / unparseable (e.g. CPU instances,
    or any client that creates instances through the API without the UI),
    matching the "0 = skip" convention of ``parse_quantity_to_mib``.
    """
    if not description:
        return 0
    data = description
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except (ValueError, TypeError):
            return 0
    if not isinstance(data, dict):
        return 0
    spec = data.get("spec") or {}
    if not isinstance(spec, dict):
        return 0
    return parse_quantity_to_mib(spec.get("memory"))


def parse_gpu_descriptor(description: Any) -> dict:
    """Display-flavor fields parsed from a GPUInstance ``description`` blob:
    ``{"spec": {"product": "NVIDIA-GeForce-RTX-5090-D", "memory": "32607Mi",
    "unitResourcesParsed": {"cpu": {"cores": 18}, "ram": {"value": 54,
    "unit": "Gi"}}}}``.

    Returns whichever of ``product`` / ``vram_mib`` / ``unit_cpu_milli`` /
    ``unit_memory_mib`` could be parsed (per-card specs) — used to enrich
    ``metered_usage.dimensions`` so the Usage "Instance Type" view can render the
    pretty product name + per-card specs, matching the GPU Instances list.
    Missing / unparseable keys are omitted (CPU instances → ``{}``).

    .. deprecated::
        Legacy fallback only — see :func:`parse_gpu_vram_mib`. The flat
        ``spec.{product,memory,unitResources}`` shape it expects is the
        pre-v2.3.0 InstanceType layout, kept alive by the UI writing a
        compatibility blob; ``product`` / ``memory`` now live on the type's
        ``status.detail`` and ``unitResources`` on its ``spec``. Instances
        created straight through the API carry no descriptor at all, so callers
        must have a real source (``type_snapshot``) and use this only when that
        is unavailable.
    """
    data = description
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except (ValueError, TypeError):
            return {}
    if not isinstance(data, dict):
        return {}
    spec = data.get("spec")
    if not isinstance(spec, dict):
        return {}
    out: dict = {}
    if spec.get("product"):
        out["product"] = spec["product"]
    vram = parse_quantity_to_mib(spec.get("memory"))
    if vram:
        out["vram_mib"] = vram
    cpu_milli, mem_mib = _parse_unit_resources(spec)
    if cpu_milli:
        out["unit_cpu_milli"] = cpu_milli
    if mem_mib:
        out["unit_memory_mib"] = mem_mib
    return out


def _parse_unit_resources(spec: dict) -> tuple:
    """Per-card ``(cpu_milli, mem_mib)`` from a descriptor spec.

    Prefers the raw ``unitResources`` quantity strings (e.g. ``"8000m"`` /
    ``"24576Mi"``) — unambiguous k8s quantities. Falls back to
    ``unitResourcesParsed``, whose ram ``value``/``unit`` can be inconsistent
    (observed ``{"value": 24, "unit": "Mi", "num": 24576}`` for a 24Gi card),
    so its ``num`` (the real amount in ``unit``) is trusted over ``value``.
    """
    raw = spec.get("unitResources")
    parsed = spec.get("unitResourcesParsed")
    cpu_milli = None
    mem_mib = None
    if isinstance(raw, dict):
        cpu_milli = parse_quantity_to_millicores(raw.get("cpu")) or None
        mem_mib = parse_quantity_to_mib(raw.get("ram")) or None
    if cpu_milli is None and isinstance(parsed, dict):
        cpu = parsed.get("cpu")
        if isinstance(cpu, dict) and cpu.get("cores"):
            try:
                cpu_milli = int(float(cpu["cores"]) * 1000)
            except (ValueError, TypeError):
                cpu_milli = None
    if mem_mib is None and isinstance(parsed, dict):
        ram = parsed.get("ram")
        if isinstance(ram, dict):
            amount = ram["num"] if ram.get("num") is not None else ram.get("value")
            if amount is not None:
                # ``or ""`` (not a get-default) so an explicit ``unit: None``
                # doesn't stringify into "<amount>None" and fail to parse.
                unit = ram.get("unit") or ""
                mem_mib = parse_quantity_to_mib(f"{amount}{unit}") or None
    return cpu_milli, mem_mib


# ---------------------------------------------------------------------------
# Kubernetes quantity parser
# ---------------------------------------------------------------------------

_BINARY_SUFFIX = {
    "Ki": 1.0 / 1024,  # 1 Ki = 1024 bytes = 1/1024 MiB
    "Mi": 1.0,
    "Gi": 1024.0,
    "Ti": 1024.0 * 1024,
    "Pi": 1024.0 * 1024 * 1024,
    "Ei": 1024.0 * 1024 * 1024 * 1024,
}
_DECIMAL_SUFFIX = {
    "": 1.0 / (1024 * 1024),  # raw bytes → MiB
    "k": 1000.0 / (1024 * 1024),
    "K": 1000.0 / (1024 * 1024),
    "M": 1_000_000.0 / (1024 * 1024),
    "G": 1_000_000_000.0 / (1024 * 1024),
    "T": 1_000_000_000_000.0 / (1024 * 1024),
}


def parse_quantity_to_mib(value: Optional[str | int | float]) -> int:
    """Parse a k8s resource quantity (memory / storage) to integer MiB.

    Accepts strings like ``"100Gi"``, ``"2048Mi"``, ``"512Ki"``, bare numbers
    (interpreted as bytes), or numeric types. Returns 0 for ``None`` / empty /
    unparseable inputs — callers treat 0 as "skip this resource".
    """
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return max(0, int(value / (1024 * 1024)))
    s = str(value).strip()
    if not s:
        return 0
    # Binary suffixes (Ki/Mi/Gi/...) take priority over decimal because the
    # binary form unambiguously ends in 'i'.
    for suffix, multiplier in _BINARY_SUFFIX.items():
        if s.endswith(suffix):
            numeric = s[: -len(suffix)]
            try:
                return max(0, int(float(numeric) * multiplier))
            except ValueError:
                return 0
    # Decimal suffixes — handle longest first so "M" doesn't shadow "Mi".
    for suffix in sorted(_DECIMAL_SUFFIX, key=len, reverse=True):
        if suffix and s.endswith(suffix):
            numeric = s[: -len(suffix)]
            try:
                return max(0, int(float(numeric) * _DECIMAL_SUFFIX[suffix]))
            except ValueError:
                return 0
    # Bare number → bytes.
    try:
        return max(0, int(float(s) / (1024 * 1024)))
    except ValueError:
        return 0


def parse_quantity_to_millicores(value: Optional[str | int | float]) -> int:
    """Parse a k8s CPU quantity to integer millicores.

    Accepts ``"2"`` (= 2000m), ``"500m"`` (= 500m), or numeric types (whole
    cores). Returns 0 for unparseable inputs.
    """
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return max(0, int(value * 1000))
    s = str(value).strip()
    if not s:
        return 0
    if s.endswith("m"):
        try:
            return max(0, int(float(s[:-1])))
        except ValueError:
            return 0
    try:
        return max(0, int(float(s) * 1000))
    except ValueError:
        return 0


def parse_accelerator_count(value: Optional[str | int | float]) -> int:
    """Parse the ``spec.resources.accelerator`` field to an integer card count.

    The schema declares it ``Optional[str]`` (e.g. ``"1"``); ``None`` / empty /
    unparseable → 0.
    """
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return max(0, int(value))
    s = str(value).strip()
    if not s:
        return 0
    try:
        if s.endswith("m"):  # millicards aren't a thing; defensive.
            return 0
        return max(0, int(float(s)))
    except ValueError:
        return 0


# ---------------------------------------------------------------------------
# Accelerator slicing weight
# ---------------------------------------------------------------------------

# ``dimensions.slice_mode`` — how the request carves up each card.
SLICE_MODE_WHOLE = "whole"  # exclusive whole card(s)
SLICE_MODE_RATIO = "ratio"  # software slice, a VRAM percentage per card
SLICE_MODE_PROFILE = "profile"  # hardware partition (e.g. an NVIDIA MIG profile)

# Per-card share is carried in thousandths so it stays an exact integer for both
# slicing modes (a percentage is a whole number of tenths of a percent).
WHOLE_CARD_MILLI = 1000


def slice_mode_of(
    memory_percentage: Optional[int], partitioned_profile: Optional[str]
) -> str:
    """Classify a request's per-card slicing mode.

    Hardware partitioning wins when both are present: the two are mutually
    exclusive by schema (one card cannot be both partitioned and soft-sliced),
    and the partition is the one the scheduler actually honours.
    ``memory_percentage == 0`` means "slicing disabled" — an exclusive whole-card
    request — not "zero share", so it maps to ``whole``.

    Coerced through ``int`` rather than tested for truthiness: the value reaches
    here from a spec snapshot, i.e. JSON, and a string ``"0"`` is truthy while
    meaning the opposite. That would classify an exclusive request as ``ratio``
    with a 0 share, and a share of 0 prices the instance at nothing.
    """
    if partitioned_profile:
        return SLICE_MODE_PROFILE
    try:
        percentage = int(memory_percentage or 0)
    except (TypeError, ValueError):
        percentage = 0
    if percentage > 0:
        return SLICE_MODE_RATIO
    return SLICE_MODE_WHOLE


def slice_share_milli(
    *,
    memory_percentage: Optional[int] = None,
    partitioned_profile: Optional[str] = None,
    profile_memory_mib: Optional[int] = None,
    card_memory_mib: Optional[int] = None,
) -> Optional[int]:
    """Per-card billed share in thousandths (1000 = a whole card).

    The metered share is the fraction of the card's **sellable capacity** the
    request occupies, and VRAM is what decides how many slices a card can host,
    so VRAM is the yardstick::

        whole    -> 1000
        ratio    -> memory_percentage * 10                       (exact)
        profile  -> min(1000, ceil(profile_mib * 1000 / card_mib))

    This is deliberately NOT the compute share. MIG ``1g.10gb`` and ``1g.20gb``
    have identical compute but the latter halves how many instances fit on a
    card, so it must cost twice as much; billing compute share would collect 57%
    of a fully-partitioned card. It also matches what the operator already
    charges Kueue for a MIG request (``MemoryMibToUnits`` — VRAM-anchored).

    CPU / RAM never enter the weight: on an accelerator node they are
    overcommitted (10x / 8x) and excluded from the accelerated ClusterQueue, so
    the 1c1g floor a tiny slice gets is given away for usability, not sold.

    Rounding is ``ceil`` — the opposite of the operator's quota-side ``floor``,
    and intentionally so: quota floors to never over-allocate physical capacity,
    billing ceils to never under-charge. Both lean toward protecting the
    platform and differ by at most 1 milli (0.1% of a card). The single division
    matters: going through ``MemoryMibToUnits(...) / 1600`` would truncate twice
    and lose an extra step. The ``min`` clamp guards against a detect-time skew
    reporting a partition at least as large as the card, which would otherwise
    price a slice above a whole card.

    Returns ``None`` when a profile request's share cannot be established
    (unknown profile, or ``memoryMib`` / card VRAM not yet backfilled). Callers
    MUST treat that as "not settleable yet" and retry — falling back to a whole
    card would silently overcharge up to 8x.

    Examples
    --------
    >>> slice_share_milli()
    1000
    >>> slice_share_milli(memory_percentage=25)
    250
    >>> slice_share_milli(memory_percentage=0)
    1000
    >>> slice_share_milli(partitioned_profile="1g.10gb", profile_memory_mib=9728,
    ...                   card_memory_mib=81920)
    119
    >>> slice_share_milli(partitioned_profile="1g.10gb", card_memory_mib=81920)
    """
    mode = slice_mode_of(memory_percentage, partitioned_profile)
    if mode == SLICE_MODE_WHOLE:
        return WHOLE_CARD_MILLI
    if mode == SLICE_MODE_RATIO:
        # ``ratio`` is only classified for a percentage that parses to >= 1, so
        # the clamp cannot produce a 0 share (which would price the slice at
        # nothing) — the floor is stated rather than relied on.
        pct = max(1, min(100, int(memory_percentage)))
        return pct * 10
    if not profile_memory_mib or not card_memory_mib or card_memory_mib <= 0:
        return None
    share = math.ceil(profile_memory_mib * WHOLE_CARD_MILLI / card_memory_mib)
    return min(WHOLE_CARD_MILLI, max(1, share))


def profile_memory_mib(profiles: Any, name: Optional[str]) -> Optional[int]:
    """Look up a partition profile's VRAM (MiB) by name in a type's aggregated
    ``status.detail.slicedDetail.physical.profiles`` list.

    Reads the reported ``memoryMib`` rather than parsing the ``<n>g.<m>gb``
    name, for four reasons: (1) ``memoryMib`` is the very value the operator's
    Pod webhook folds into the Kueue credit request, so using it keeps quota and
    bill on one number; (2) the name's ``<m>gb`` is derived from the partition
    geometry and rounds to the marketing size — an A100-80GB ``1g.10gb`` really
    has ~9728 MiB, so parsing the name over-states it by ~5%; (3) the name format
    is not guaranteed (any manufacturer can enable partitioning with its own
    naming), and a regex miss would silently fall back to a whole card; (4) the
    type row is already loaded, so it costs nothing.

    Accepts pydantic profile objects or plain dicts (either key style). Returns
    ``None`` when the list or the named profile is absent, or when its
    ``memoryMib`` is missing / non-positive.
    """
    if not name or not profiles:
        return None
    for profile in profiles:
        if isinstance(profile, dict):
            p_name = profile.get("name")
            p_mib = profile.get("memory_mib", profile.get("memoryMib"))
        else:
            p_name = getattr(profile, "name", None)
            p_mib = getattr(profile, "memory_mib", None)
        if p_name != name:
            continue
        try:
            mib = int(p_mib)
        except (TypeError, ValueError):
            return None
        return mib if mib > 0 else None
    return None


def sliced_sku_count(accelerator_count: int, share_milli: int) -> Decimal:
    """Billed unit multiplier for ``accelerator_count`` cards at ``share_milli``
    each — the value stored in ``metered_usage.sku_count``.

    Per-card share is computed first and multiplied by the card count (not the
    other way round) so one profile costs exactly N times as much on N cards,
    with no rounding tail from division order. ``Decimal`` keeps it exact:
    ``share_milli`` is an integer ≤ 1000, so the quotient always fits the
    column's 8 fraction digits.

    Not normalized: Decimal division already yields the shortest exact form for
    a terminating quotient (``4000/1000 -> 4``, ``500/1000 -> 0.5``), whereas
    ``normalize()`` would rewrite a whole ``10`` as ``1E+1``.

    Examples
    --------
    >>> sliced_sku_count(2, 250)
    Decimal('0.5')
    >>> sliced_sku_count(4, 1000)
    Decimal('4')
    >>> sliced_sku_count(10, 1000)
    Decimal('10')
    >>> sliced_sku_count(1, 119)
    Decimal('0.119')
    """
    return Decimal(accelerator_count) * Decimal(share_milli) / Decimal(WHOLE_CARD_MILLI)


# ---------------------------------------------------------------------------
# UTC midnight splitter
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Instance SKU / resource_type derivation
# ---------------------------------------------------------------------------

# Mirror of ``gpustack.schemas.metered_usage.RESOURCE_TYPE_*`` — kept as
# literals so this module stays import-light (no schema dependency).
_RESOURCE_TYPE_GPU_INSTANCE = "gpu_instance"
_RESOURCE_TYPE_CPU_INSTANCE = "cpu_instance"


def instance_resource_type(gpu_count: Optional[int]) -> str:
    """An instance with >=1 accelerator is a ``gpu_instance``; otherwise it is
    a ``cpu_instance``. Drives both the ``metered_usage.resource_type`` column
    and the Resource-tab breakdown bucket."""
    return (
        _RESOURCE_TYPE_GPU_INSTANCE
        if gpu_count and gpu_count > 0
        else _RESOURCE_TYPE_CPU_INSTANCE
    )


def instance_sku(
    instance_type: Optional[str],
    gpu_type: Optional[str],
    gpu_count: int,
    cpu_millicores: int,
    memory_mib: int,
) -> str:
    """Legacy sku derivation — the per-cluster instance type NAME.

    .. deprecated::
        The sku is now the instance type's identity snapshot
        (``gpu_instance_types.snapshot``, i.e. ``gpu_instances.type_snapshot``)
        verbatim. This function is the last-resort fallback for instances whose
        ``type_snapshot`` is NULL (created before v2.3.0, unbackfillable) or for
        deployments with no operator, and a row that falls back MUST be tagged
        ``dimensions.sku_source`` so it is distinguishable.

        The name is a poor key for three reasons the snapshot does not share:
        the operator renamed its derived types between v0.5 and v0.7 (so one
        hardware pool is two skus across an upgrade); the name does not encode
        ``unitResources``, so a CPU row's ``sku_count`` unit is not recoverable
        from it; and it is per-cluster only by convention.

    The sku is the instance spec's ``type`` (the flavor / queue name, e.g.
    ``gpustack--generic--nvidia-tesla-t4-linux-amd64`` on operator v0.7,
    ``gpustack--generic-ln-x64-4c-16g-98g--nvidia-tesla-t4-1d`` on v0.5)
    verbatim. The remaining args are a fallback for snapshots missing ``type``:
    GPU instances fall back to ``gpu_type`` (the card model), CPU instances to a
    ``cpu-{cores}vcpu-{gib}g`` flavor.

    Examples
    --------
    >>> instance_sku("gpustack--generic--nvidia-tesla-t4-linux-amd64", "x", 2, 8000, 1)
    'gpustack--generic--nvidia-tesla-t4-linux-amd64'
    >>> instance_sku(None, "nvidia-h100", 2, 8000, 128000)
    'nvidia-h100'
    >>> instance_sku(None, None, 0, 2000, 8192)
    'cpu-2vcpu-8g'
    """
    if instance_type:
        return instance_type
    if gpu_count and gpu_count > 0:
        return gpu_type or "unknown"
    cores = f"{cpu_millicores / 1000:g}"
    gib = memory_mib // 1024
    return f"cpu-{cores}vcpu-{gib}g"


def volume_sku(category: str, type_name: str) -> str:
    """The "Storage Type" breakdown dimension (the sku) for persistent volumes.

    Mirrors the instance ``gpustack--...`` flavor convention with a ``volume--``
    prefix so the resource kind and provisioner are recoverable from the sku
    string alone (issue #5716). Three ``--``-joined segments::

        volume--<category>--<type_name>

    where ``category`` is the provisioner kind (``nfs`` / ``s3``) derived from
    the volume type's spec and ``type_name`` is the user-defined volume-type
    name. Both are always resolvable for a live volume — its type can't be
    deleted while in use (FK ``RESTRICT``) and a valid type always has a
    provisioner — so callers pass concrete values; the collector skips the
    rollup if either is somehow missing rather than synthesizing a placeholder.
    All volumes of a type share one sku, so the "by type" breakdown still
    aggregates per storage type.

    Examples
    --------
    >>> volume_sku("nfs", "aws")
    'volume--nfs--aws'
    >>> volume_sku("s3", "minio")
    'volume--s3--minio'
    """
    return f"volume--{category}--{type_name}"


def split_delta_across_utc_midnight(
    start: datetime, end: datetime
) -> List[Tuple[date, int]]:
    """Split a window ``[start, end]`` into per-UTC-day segments.

    Both timestamps must be naive UTC (matching ``ModelUsage.date`` /
    ``TimestampsMixin`` convention). Returns ``[(utc_date, seconds), ...]`` in
    chronological order. Returns ``[]`` for non-positive windows.

    Examples
    --------
    >>> split_delta_across_utc_midnight(
    ...     datetime(2026, 5, 28, 23, 59, 30),
    ...     datetime(2026, 5, 29, 0, 0, 30),
    ... )
    [(datetime.date(2026, 5, 28), 30), (datetime.date(2026, 5, 29), 30)]
    """
    if end <= start:
        return []
    out: List[Tuple[date, int]] = []
    cursor = start
    while cursor < end:
        next_midnight = (cursor + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        segment_end = min(next_midnight, end)
        seconds = int((segment_end - cursor).total_seconds())
        if seconds > 0:
            out.append((cursor.date(), seconds))
        cursor = segment_end
    return out


def iter_utc_day_segments(
    start: datetime, end: datetime
) -> List[Tuple[date, datetime, datetime]]:
    """Like :func:`split_delta_across_utc_midnight` but yields the actual
    ``(utc_date, segment_start, segment_end)`` datetime bounds for each day.

    The collector needs the bounds (not just the second count) so it can clamp
    each segment against the row's persisted ``settled_until`` high-water mark
    — making settlement idempotent across restarts / event replay and across
    stop→start multi-window days.

    Examples
    --------
    >>> iter_utc_day_segments(
    ...     datetime(2026, 5, 28, 23, 59, 30),
    ...     datetime(2026, 5, 29, 0, 0, 30),
    ... )
    [(datetime.date(2026, 5, 28), datetime.datetime(2026, 5, 28, 23, 59, 30), datetime.datetime(2026, 5, 29, 0, 0)), (datetime.date(2026, 5, 29), datetime.datetime(2026, 5, 29, 0, 0), datetime.datetime(2026, 5, 29, 0, 0, 30))]
    """
    if end <= start:
        return []
    out: List[Tuple[date, datetime, datetime]] = []
    cursor = start
    while cursor < end:
        next_midnight = (cursor + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        segment_end = min(next_midnight, end)
        out.append((cursor.date(), cursor, segment_end))
        cursor = segment_end
    return out


def iter_utc_hour_segments(
    start: datetime, end: datetime
) -> List[Tuple[datetime, datetime, datetime]]:
    """Split a window ``[start, end]`` into per-UTC-hour segments, yielding
    ``(bucket_start, segment_start, segment_end)`` where ``bucket_start`` is the
    hour-truncated start of the segment's hour (the ``metered_usage`` bucket key).

    The collector clamps each segment against the row's ``settled_until`` so
    settlement is idempotent across restarts / replay / stop-start. Both inputs
    must be naive UTC. Returns ``[]`` for non-positive windows.

    Examples
    --------
    >>> iter_utc_hour_segments(
    ...     datetime(2026, 5, 26, 10, 59, 30),
    ...     datetime(2026, 5, 26, 11, 0, 30),
    ... )
    [(datetime.datetime(2026, 5, 26, 10, 0), datetime.datetime(2026, 5, 26, 10, 59, 30), datetime.datetime(2026, 5, 26, 11, 0)), (datetime.datetime(2026, 5, 26, 11, 0), datetime.datetime(2026, 5, 26, 11, 0), datetime.datetime(2026, 5, 26, 11, 0, 30))]
    """
    if end <= start:
        return []
    out: List[Tuple[datetime, datetime, datetime]] = []
    cursor = start
    while cursor < end:
        bucket_start = cursor.replace(minute=0, second=0, microsecond=0)
        next_hour = bucket_start + timedelta(hours=1)
        segment_end = min(next_hour, end)
        out.append((bucket_start, cursor, segment_end))
        cursor = segment_end
    return out
