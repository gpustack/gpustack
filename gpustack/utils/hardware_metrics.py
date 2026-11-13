"""Stable hardware identities shared by exposition and resource queries."""

from collections import Counter
from typing import Iterable, Optional, Tuple
from gpustack.schemas.workers import GPUDeviceStatus


def gpu_identity(device: GPUDeviceStatus) -> Optional[Tuple[str, int]]:
    """Return a UUID/chip identity only for an addressable inventory device."""
    uuid = getattr(device, 'uuid', None)
    index = getattr(device, 'index', None)
    chip = getattr(device, 'device_chip_index', None)
    if not isinstance(uuid, str) or not uuid.strip():
        return None
    if uuid.strip().lower() in ('unknown', 'n/a', 'none'):
        return None
    if any(type(value) is not int or value < 0 for value in (index, chip)):
        return None
    return uuid.strip(), chip


def verified_gpu_devices(devices: Iterable[GPUDeviceStatus]) -> list[GPUDeviceStatus]:
    """Exclude ambiguous logical indexes and duplicate stable identities."""
    devices = list(devices)
    indexes = Counter(getattr(d, 'index', None) for d in devices)
    identities = Counter(gpu_identity(d) for d in devices)
    return [
        d
        for d in devices
        if gpu_identity(d) is not None
        and indexes[d.index] == 1
        and identities[gpu_identity(d)] == 1
    ]


def is_gpu_partition(device: GPUDeviceStatus) -> bool:
    """Recognize the MIG identities supplied by supported runtime detectors."""
    return (getattr(device, 'uuid', '') or '').upper().startswith('MIG-')
