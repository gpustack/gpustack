"""Bounded monitoring intervals and sanitized storage-policy hints."""

import asyncio
from dataclasses import dataclass
import math
import re
import time
from typing import Optional

import aiohttp

from gpustack.schemas.model_metrics import ModelMonitoringCapabilities
from . import prometheus as prometheus_query

MIN_QUERY_SECONDS = 300
MAX_QUERY_SECONDS = 604800
_STATUS_TIMEOUT_SECONDS = 2.0
_CACHE_SECONDS = 60.0
_POLICY_CACHE = {}
_DURATION_PART = re.compile(r'(\d+)(ms|y|w|d|h|m|s)')
_DURATION_UNITS = {
    'ms': 0.001,
    's': 1,
    'm': 60,
    'h': 3600,
    'd': 86400,
    'w': 604800,
    'y': 31536000,
}
_SIZE_PART = re.compile(r'(\d+(?:\.\d+)?)\s*(?:[KMGTPE]?i?B|bytes|%)', re.I)


@dataclass(frozen=True)
class MonitoringInterval:
    start: float
    end: float
    seconds: int


def monitoring_interval(
    window_seconds: int,
    start: Optional[float] = None,
    end: Optional[float] = None,
    *,
    now: Optional[float] = None,
) -> MonitoringInterval:
    """Resolve a relative interval or validate an exact whole-second range.

    The duration bounds limit work per request, not the age of stored data.
    A remote-read source can retain observations beyond the local TSDB policy.
    """
    current = time.time() if now is None else now
    if start is None and end is None:
        if type(window_seconds) is not int or not (
            MIN_QUERY_SECONDS <= window_seconds <= MAX_QUERY_SECONDS
        ):
            raise ValueError('Invalid monitoring window')
        return MonitoringInterval(current - window_seconds, current, window_seconds)
    if start is None or end is None:
        raise ValueError('Both start and end are required')
    for value in (start, end):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            or value != int(value)
        ):
            raise ValueError('Monitoring timestamps must be whole Unix seconds')
    seconds = int(end - start)
    if not MIN_QUERY_SECONDS <= seconds <= MAX_QUERY_SECONDS:
        raise ValueError('Monitoring range must span 5 minutes to 7 days')
    if end > current:
        raise ValueError('Monitoring range cannot end in the future')
    return MonitoringInterval(int(start), int(end), seconds)


def _duration_seconds(value):
    if not isinstance(value, str) or not value or len(value) > 128:
        return None
    parts = list(_DURATION_PART.finditer(value))
    if not parts or ''.join(part.group() for part in parts) != value:
        return None
    total = sum(int(part[1]) * _DURATION_UNITS[part[2]] for part in parts)
    return int(total) if math.isfinite(total) and total >= 1 else None


def _runtime_policy(data):
    if 'storageRetention' not in data:
        return None
    value = data.get('storageRetention')
    if not isinstance(value, str) or len(value) > 256:
        value = ''
    parts = value.strip().split(' or ')
    durations = [_duration_seconds(part.strip()) for part in parts]
    sizes = [_SIZE_PART.fullmatch(part.strip()) for part in parts]
    seconds = next((duration for duration in durations if duration is not None), None)
    size_based = any(match and float(match[1]) > 0 for match in sizes)
    # Do not substitute startup flags for an explicitly reported effective policy.
    return dict(
        retention_seconds=seconds,
        retention_size_based=bool(size_based),
        retention_known=seconds is not None or bool(size_based),
        retention_source='prometheus_runtimeinfo',
    )


def _flags_policy(data):
    seconds = _duration_seconds(data.get('storage.tsdb.retention.time'))
    size = data.get('storage.tsdb.retention.size')
    match = (
        _SIZE_PART.fullmatch(size.strip())
        or re.fullmatch(r'(\d+(?:\.\d+)?)', size.strip())
        if isinstance(size, str) and len(size) <= 128
        else None
    )
    size_based = bool(match and float(match[1]) > 0)
    return dict(
        retention_seconds=seconds,
        retention_size_based=size_based,
        retention_known=seconds is not None or size_based,
        retention_source='prometheus_flags',
    )


async def _status(client, url, endpoint):
    target = f'{prometheus_query.api_base(url)}/status/{endpoint}'
    async with client.get(
        target, timeout=aiohttp.ClientTimeout(total=_STATUS_TIMEOUT_SECONDS)
    ) as response:
        payload = await response.json(content_type=None)
        if (
            response.status != 200
            or not isinstance(payload, dict)
            or payload.get('status') != 'success'
            or not isinstance(payload.get('data'), dict)
        ):
            raise ValueError('Unavailable monitoring storage policy')
        return payload['data']


async def _storage_policy(client, url):
    cached = _POLICY_CACHE.get(url)
    if cached and cached[0] > time.monotonic():
        return cached[1]
    policy = None
    for endpoint in ('runtimeinfo', 'flags'):
        try:
            data = await asyncio.wait_for(
                _status(client, url, endpoint), _STATUS_TIMEOUT_SECONDS
            )
            policy = (
                _runtime_policy(data)
                if endpoint == 'runtimeinfo'
                else _flags_policy(data)
            )
        except (
            ValueError,
            TypeError,
            aiohttp.ClientError,
            OSError,
            asyncio.TimeoutError,
        ):
            continue
        if policy is not None:
            break
    policy = policy or {}
    _POLICY_CACHE[url] = (time.monotonic() + _CACHE_SECONDS, policy)
    return policy


async def collect_monitoring_capabilities(
    model_id: int, client: Optional[aiohttp.ClientSession] = None
) -> ModelMonitoringCapabilities:
    """Return limits and retention hints, never global TSDB contents or flags."""
    result = ModelMonitoringCapabilities(model_id=model_id)
    url = prometheus_query.prometheus_url()
    if not url:
        return result
    if client is None:
        async with aiohttp.ClientSession() as owned:
            policy = await _storage_policy(owned, url)
    else:
        policy = await _storage_policy(client, url)
    return result.model_copy(update=policy)


async def collect_storage_policy(client=None) -> dict:
    """Return only sanitized policy fields for directly authorized resources."""
    url = prometheus_query.prometheus_url()
    if not url:
        return {}
    if client is None:
        async with aiohttp.ClientSession() as owned:
            return await _storage_policy(owned, url)
    return await _storage_policy(client, url)
