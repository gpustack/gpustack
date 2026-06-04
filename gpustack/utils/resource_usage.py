"""Helpers shared by the resource-usage collectors / event-logger.

Kept tiny and dependency-free so they can be unit-tested without spinning up
the rest of the server.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta
from typing import Any, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Metered phase predicate
# ---------------------------------------------------------------------------

# Terminal failure phases from
# ``gpustack.server.controllers.GPUInstanceController.PHASE_*_FAILED``.
# Kept here as a literal set to avoid pulling the server module (and the
# whole reconcile graph) into the usage layer at import time.
_FAILED_PHASES = frozenset(
    {
        "CreateFailed",
        "SSHPublicKeyCreateFailed",
        "PersistentVolumeCreateFailed",
        "PersistentVolumeTypeCreateFailed",
    }
)


def is_metered_phase(phase: Optional[str]) -> bool:
    """Return True if ``phase`` represents a state that consumes GPU.

    The plan's "kueue admits → resource is reserved" model: any non-empty,
    non-failure phase is metered. ``None`` covers brand-new rows whose
    reconciler hasn't run yet; failure phases cover terminal states where the
    accelerator has already been released.
    """
    return phase is not None and phase not in _FAILED_PHASES


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

    ``description`` is the device descriptor — usually a JSON string (sometimes
    already a dict) shaped like ``{"spec": {"memory": "48Gi", ...}}``. Returns 0
    when absent / unparseable (e.g. CPU instances), matching the "0 = skip"
    convention of ``parse_quantity_to_mib``.
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
                mem_mib = (
                    parse_quantity_to_mib(f"{amount}{ram.get('unit', '')}") or None
                )
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
    gpu_type: Optional[str],
    gpu_count: int,
    cpu_millicores: int,
    memory_mib: int,
) -> str:
    """The "Instance Type" breakdown dimension (the sku), derived from spec.

    GPU instances are **per card** → sku = ``gpu_type`` (the card model);
    card count is carried separately in ``gpu_count`` and metered via GPU-Hours.
    CPU instances (no GPU) are **whole-machine** → sku = the cpu flavor.

    Examples
    --------
    >>> instance_sku("nvidia-h100", 2, 8000, 128000)
    'nvidia-h100'
    >>> instance_sku(None, 0, 2000, 8192)
    'cpu-2vcpu-8g'
    """
    if gpu_count and gpu_count > 0:
        return gpu_type or "unknown"
    cores = f"{cpu_millicores / 1000:g}"
    gib = memory_mib // 1024
    return f"cpu-{cores}vcpu-{gib}g"


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
