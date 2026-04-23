"""
Helpers to reconcile gpustack's configured port ranges (service, Ray) with the
Linux kernel's ephemeral port range.

When an outbound TCP connection is made, the kernel picks a local source port
from `net.ipv4.ip_local_port_range` (default 32768-60999). gpustack's default
`ray_port_range` (41000-41999) and `service_port_range` (40000-40063) fall
inside this window, so a worker-side outbound connection (e.g. gpustack-worker
talking to the server on :80) can transiently squat on a port that Ray or an
inference server later tries to bind(), causing "Address already in use".

The fix is to append the gpustack ranges to `ip_local_reserved_ports`. This
module detects the conflict and, when running with enough privilege, applies
the reservation automatically. Otherwise it logs the exact sysctl command the
operator must run.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from gpustack import envs
from gpustack.utils import platform

logger = logging.getLogger(__name__)

_LOCAL_PORT_RANGE_PATH = Path("/proc/sys/net/ipv4/ip_local_port_range")
_RESERVED_PORTS_PATH = Path("/proc/sys/net/ipv4/ip_local_reserved_ports")

Range = Tuple[int, int]


def _parse_ranges(text: str) -> List[Range]:
    """
    Parse a sysctl port list (comma-separated ports and N1-N2 ranges) into
    a list of inclusive (start, end) tuples. Returns [] on empty input.
    """
    ranges: List[Range] = []
    for token in text.replace(",", " ").split():
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            ranges.append((int(a), int(b)))
        else:
            p = int(token)
            ranges.append((p, p))
    return ranges


def _merge(ranges: List[Range]) -> List[Range]:
    """
    Merge overlapping/adjacent ranges. Result is sorted and disjoint.
    """
    if not ranges:
        return []
    ordered = sorted(ranges)
    merged: List[Range] = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _format_ranges(ranges: List[Range]) -> str:
    parts: List[str] = []
    for start, end in ranges:
        parts.append(str(start) if start == end else f"{start}-{end}")
    return ",".join(parts)


def _covered_by(target: Range, ranges: List[Range]) -> bool:
    start, end = target
    for rs, re_ in ranges:
        if rs <= start and re_ >= end:
            return True
    return False


def _overlaps(a: Range, b: Range) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def _read_port_range(path: Path) -> Optional[Range]:
    try:
        text = path.read_text().strip()
    except OSError as e:
        logger.debug("Cannot read %s: %s", path, e)
        return None
    parts = text.split()
    if len(parts) != 2:
        logger.debug("Unexpected content in %s: %r", path, text)
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        logger.debug("Non-numeric content in %s: %r", path, text)
        return None


def _read_reserved_ports(path: Path) -> Optional[List[Range]]:
    """
    Returns the currently reserved ranges, [] if the file is missing/empty,
    or None if the content is unparseable — in which case callers should
    abort rather than risk overwriting user-managed reservations.
    """
    try:
        text = path.read_text().strip()
    except OSError as e:
        logger.debug("Cannot read %s: %s", path, e)
        return []
    try:
        return _merge(_parse_ranges(text))
    except ValueError:
        logger.warning(
            "Cannot parse %s: %r. Skipping auto-reservation to avoid "
            "overwriting existing configuration.",
            path,
            text,
        )
        return None


def ensure_reserved_against_ephemeral(
    port_ranges: List[Tuple[str, Range]],
) -> None:
    """
    Ensure each given port range is reserved against the kernel's ephemeral
    range on Linux. No-op on other platforms or when /proc sysctls are
    unreadable (e.g., unprivileged container, unusual kernels).

    port_ranges: list of (human_name, (start, end)) tuples.
    """
    if envs.SKIP_RESERVE_EPHEMERAL_PORTS:
        logger.info(
            "Skipping ephemeral port reservation because "
            "GPUSTACK_SKIP_RESERVE_EPHEMERAL_PORTS is set."
        )
        return

    if platform.system() != "linux":
        return

    if not _LOCAL_PORT_RANGE_PATH.exists():
        return

    ephemeral = _read_port_range(_LOCAL_PORT_RANGE_PATH)
    if ephemeral is None:
        return

    reserved = _read_reserved_ports(_RESERVED_PORTS_PATH)
    if reserved is None:
        return

    conflicts: List[Tuple[str, Range]] = []
    for name, rng in port_ranges:
        if not _overlaps(rng, ephemeral):
            continue
        if _covered_by(rng, reserved):
            continue
        conflicts.append((name, rng))

    if not conflicts:
        return

    desired = _merge(reserved + [rng for _, rng in conflicts])
    payload = _format_ranges(desired)

    conflict_desc = ", ".join(
        f"{name}={start}-{end}" for name, (start, end) in conflicts
    )
    try:
        _RESERVED_PORTS_PATH.write_text(payload)
    except OSError as e:
        logger.warning(
            "gpustack port ranges (%s) overlap the kernel ephemeral port "
            "range %d-%d and are not reserved. Ray or inference servers may "
            "fail to bind when the kernel transiently assigns one of these "
            "ports as an outbound ephemeral port. Failed to auto-reserve "
            "(%s). Run on each worker host:\n"
            "    echo 'net.ipv4.ip_local_reserved_ports = %s' | "
            "sudo tee -a /etc/sysctl.conf && sudo sysctl -p",
            conflict_desc,
            ephemeral[0],
            ephemeral[1],
            e,
            payload,
        )
        return

    logger.info(
        "Reserved gpustack port ranges (%s) against kernel ephemeral range "
        "%d-%d via %s (now: %s).",
        conflict_desc,
        ephemeral[0],
        ephemeral[1],
        _RESERVED_PORTS_PATH,
        payload,
    )
