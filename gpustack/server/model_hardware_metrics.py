"""Authorized history of stable hardware currently associated with a model."""

import asyncio
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import time
from typing import Dict, List, Optional

import aiohttp
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.exceptions import InvalidException
from gpustack.api.tenant import (
    TenantContext,
    tenant_list_conditions,
    tenant_stream_filter,
)
from gpustack.schemas.models import Model, ModelInstance, member_worker_ids
from gpustack.schemas.model_metrics import (
    MetricDimensions,
    MetricSeries,
    MetricSummary,
    ModelMetric,
    ModelResourceMetrics,
    ResourceDevice,
)
from gpustack.schemas.workers import Worker
from gpustack.server import prometheus_query
from gpustack.utils.hardware_metrics import (
    gpu_identity,
    is_gpu_partition,
    verified_gpu_devices,
)
from gpustack.utils.model_instance_workers import (
    get_model_instance_workloads,
)

HISTORY_SCOPE = 'owner_qualified_current_associated_resources'
COLLECTION_TIMEOUT_SECONDS = 12
FRESHNESS_SECONDS = 120
_DEFINITIONS = {
    'gpu_utilization': ('worker_gpu_utilization_percent', 'percent', 0, 100),
    'gpu_memory_used': ('worker_gpu_memory_used_bytes', 'bytes', 0, None),
    'gpu_memory_total': ('worker_gpu_memory_total_bytes', 'bytes', 0, None),
    'gpu_power': ('worker_gpu_power_watts', 'watts', 0, None),
    'gpu_temperature': ('worker_gpu_temperature_celsius', 'celsius', None, None),
    'host_cpu': ('worker_host_cpu_utilization_percent', 'percent', 0, 100),
    'host_memory_used': ('worker_host_memory_used_bytes', 'bytes', 0, None),
    'host_memory_total': ('worker_host_memory_total_bytes', 'bytes', 0, None),
}


@dataclass
class _Device:
    public: ResourceDevice
    worker: Worker
    identity: Optional[tuple]


def _number(value, minimum=None, maximum=None):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number):
        return None
    if minimum is not None and number < minimum:
        return None
    if maximum is not None and number > maximum:
        return None
    return number


def _worker_identity(worker) -> Optional[Dict[str, str]]:
    uuid = worker.worker_uuid
    owner = worker.owner_principal_id
    if not isinstance(uuid, str) or not uuid.strip():
        return None
    if owner is not None and (type(owner) is not int or owner <= 0):
        return None
    return {
        'worker_id': str(worker.id),
        'worker_uuid': uuid,
        'owner_principal_id': str(owner or ''),
    }


def _selector(family, labels):
    matchers = ','.join(f'{key}={json.dumps(value)}' for key, value in labels.items())
    return f'gpustack:{family}{{{matchers}}}'


def _fresh(selector):
    return f'({selector}) and (time() - timestamp({selector}) <= {FRESHNESS_SECONDS})'


def _matches(row, labels):
    actual = row.get('metric', {})
    return all(actual.get(key) == value for key, value in labels.items())


def _current(device):
    partition = device is not None and is_gpu_partition(device)
    values = {
        'gpu_utilization': getattr(
            getattr(device, 'core', None), 'utilization_rate', None
        ),
        'gpu_memory_used': getattr(getattr(device, 'memory', None), 'used', None),
        'gpu_memory_total': getattr(getattr(device, 'memory', None), 'total', None),
        'gpu_power': getattr(device, 'power_used', None),
        'gpu_temperature': getattr(device, 'temperature', None),
    }
    return {
        key: (
            None
            if partition and key in ('gpu_utilization', 'gpu_power', 'gpu_temperature')
            else _number(value, *_DEFINITIONS[key][2:])
        )
        for key, value in values.items()
    }


def _device(worker, index, instance_ids):
    candidates = (
        [d for d in (worker.status.gpu_devices or []) if d.index == index]
        if worker.status
        else []
    )
    device = candidates[0] if len(candidates) == 1 else None
    verified = (
        verified_gpu_devices(worker.status.gpu_devices or []) if worker.status else []
    )
    identity = (
        gpu_identity(device) if device is not None and device in verified else None
    )
    labels = _worker_identity(worker)
    if labels is None:
        identity = None
    key = [
        worker.id,
        worker.worker_uuid,
        worker.owner_principal_id,
        identity or ['unresolved', index],
    ]
    opaque = sha256(json.dumps(key).encode()).hexdigest()[:32]
    heartbeat = getattr(worker, 'heartbeat_time', None)
    reported_at = (
        _number(heartbeat.timestamp())
        if heartbeat is not None and heartbeat.utcoffset() is not None
        else None
    )
    public = ResourceDevice(
        id=opaque,
        worker_id=worker.id,
        worker_name=worker.name,
        name=device.name if device is not None else 'GPU',
        index=index,
        chip_index=device.device_chip_index if device is not None else None,
        scope=(
            'partition'
            if device is not None and is_gpu_partition(device)
            else 'physical_gpu'
        ),
        model_instance_ids=sorted(instance_ids),
        history_available=False,
        reason=('no_samples' if identity else 'device_identity_unresolved'),
        current=_current(device),
        last_reported_at=reported_at,
    )
    return _Device(public, worker, identity)


def _allocated_gpus(instance):
    """Read device allocation independently of listener readiness or ports."""
    allocations = {worker_id: set() for worker_id in member_worker_ids(instance)}
    if instance.worker_id is not None:
        allocations[instance.worker_id].update(instance.gpu_indexes or [])
    servers = instance.distributed_servers
    for member in (servers.subordinate_workers or []) if servers else []:
        if member.worker_id is not None:
            allocations.setdefault(member.worker_id, set()).update(
                member.gpu_indexes or []
            )
    for workload in get_model_instance_workloads(instance) or []:
        indexes = allocations.setdefault(workload.worker_id, set())
        for placement in workload.dp_rank_placements:
            indexes.update(placement.gpu_indexes)
    return allocations


async def _associated_devices(instances, ctx, session):
    associations, worker_ids = {}, set()
    for instance in instances:
        for worker_id, indexes in _allocated_gpus(instance).items():
            worker_ids.add(worker_id)
            for index in indexes:
                associations.setdefault((worker_id, index), set()).add(instance.id)
    if not worker_ids:
        return [], {}, associations, worker_ids
    result = await session.exec(
        select(Worker).where(
            Worker.id.in_(worker_ids),
            Worker.deleted_at.is_(None),
            *tenant_list_conditions(ctx, Worker),
        )
    )
    visible = tenant_stream_filter(ctx, Worker)
    workers = {
        w.id: w
        for w in result.all()
        if w.id in worker_ids and visible(w) and not w.deleted_at
    }
    devices = [
        _device(workers[worker_id], index, ids)
        for (worker_id, index), ids in sorted(associations.items())
        if worker_id in workers
    ]
    return devices, workers, associations, worker_ids


def _empty(key, status, reason, scope='physical_gpu'):
    return ModelMetric(
        kind='gauge',
        unit=_DEFINITIONS[key][1],
        measurement_scope=scope,
        status=status,
        reason=reason,
        supported_statistics=['current'],
    )


async def _identity_presence(devices, client, url, end, semaphore):
    selectors = {
        _selector('worker_gpu_identity_info', _worker_identity(d.worker))
        for d in devices
        if d.identity is not None
    }
    if not selectors:
        return
    async with semaphore:
        rows = await prometheus_query.query_instant(
            client, url, ' or '.join(_fresh(s) for s in sorted(selectors)), end
        )
    for device in devices:
        if device.identity is None:
            continue
        device.public.reason = 'legacy_history_unavailable'
        labels = {
            **_worker_identity(device.worker),
            'gpu_uuid': device.identity[0],
            'gpu_chip_index': str(device.identity[1]),
            'gpu_index': str(device.public.index),
        }
        if any(
            _matches(row, labels)
            and _number(row.get('value', [None, None])[1], 0) not in (None, 0)
            for row in rows
        ):
            device.public.history_available = True
            device.public.reason = None


def _samples(rows, labels, bounds, field):
    samples = {}
    for row in rows:
        if not _matches(row, labels):
            continue
        values = row.get(field, [])
        if field == 'value':
            values = [values]
        for pair in values:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            timestamp = _number(pair[0], 0)
            if timestamp is None:
                continue
            value = _number(pair[1], *bounds)
            if timestamp not in samples:
                samples[timestamp] = value
            elif samples[timestamp] != value:
                samples[timestamp] = None
    return sorted(samples.items())


async def _query_metric(
    key,
    labels,
    dimensions,
    scope,
    response,
    client,
    url,
    semaphore,
    empty_reason='no_samples',
):
    definition = _DEFINITIONS[key]
    expression = _fresh(_selector(definition[0], labels))

    async def query(kind):
        async with semaphore:
            if kind == 'range':
                return await prometheus_query.query_range(
                    client, url, expression, response.start, response.end, response.step
                )
            return await prometheus_query.query_instant(
                client, url, expression, response.end
            )

    try:
        tasks = [asyncio.create_task(query(kind)) for kind in ('range', 'instant')]
        try:
            history, current = await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        points = _samples(history, labels, definition[2:], 'values')
        latest = _samples(current, labels, definition[2:], 'value')
        value = latest[-1][1] if latest else None
        has_history = any(value is not None for _, value in points)
        status = (
            'available' if value is not None else 'stale' if has_history else 'no_data'
        )
        reason = (
            None
            if status == 'available'
            else 'no_recent_samples' if has_history else empty_reason
        )
        metric = _empty(key, status, reason, scope)
        metric.series = [
            MetricSeries(
                name=key, statistic='current', dimensions=dimensions, points=points
            )
        ]
        metric.summaries = [
            MetricSummary(statistic='current', dimensions=dimensions, value=value)
        ]
        response.metrics[key] = metric
    except Exception:
        response.metrics[key] = _empty(key, 'query_error', 'query_failed', scope)


async def _collect_history(devices, selected, worker, response, client, url):
    semaphore = asyncio.Semaphore(4)
    deadline = asyncio.get_running_loop().time() + COLLECTION_TIMEOUT_SECONDS
    try:
        await asyncio.wait_for(
            _identity_presence(devices, client, url, response.end, semaphore),
            timeout=COLLECTION_TIMEOUT_SECONDS,
        )
    except Exception:
        for device in devices:
            if device.identity is not None:
                device.public.reason = 'query_failed'
    jobs = []
    for key in _DEFINITIONS:
        gpu = key.startswith('gpu_')
        scope = selected.public.scope if gpu and selected else 'worker'
        if gpu and (selected is None or selected.identity is None):
            response.metrics[key] = _empty(
                key, 'unsupported', 'device_identity_unresolved', scope
            )
            continue
        if (
            gpu
            and selected.public.scope == 'partition'
            and key in ('gpu_power', 'gpu_temperature', 'gpu_utilization')
        ):
            response.metrics[key] = _empty(
                key, 'unsupported', 'metric_not_provided', scope
            )
            continue
        labels = _worker_identity(worker) if worker else None
        if labels is None:
            response.metrics[key] = _empty(
                key, 'unsupported', 'source_identity_unresolved', scope
            )
            continue
        dimensions = MetricDimensions(worker_id=worker.id)
        if gpu:
            labels.update(
                gpu_uuid=selected.identity[0], gpu_chip_index=str(selected.identity[1])
            )
            dimensions.device_id = selected.public.id
        response.metrics[key] = _empty(key, 'query_error', 'query_failed', scope)
        empty_reason = (
            'legacy_history_unavailable'
            if gpu and selected.public.reason == 'legacy_history_unavailable'
            else 'no_samples'
        )
        jobs.append((key, labels, dimensions, scope, empty_reason))
    remaining = deadline - asyncio.get_running_loop().time()
    if remaining <= 0:
        return
    try:
        await asyncio.wait_for(
            asyncio.gather(
                *(
                    _query_metric(
                        *job[:4],
                        response,
                        client,
                        url,
                        semaphore,
                        empty_reason=job[4],
                    )
                    for job in jobs
                )
            ),
            timeout=remaining,
        )
    except asyncio.TimeoutError:
        pass


async def collect_model_resource_metrics(
    model: Model,
    instances: List[ModelInstance],
    ctx: TenantContext,
    session: AsyncSession,
    *,
    window_seconds: int,
    device_id: Optional[str] = None,
    client: Optional[aiohttp.ClientSession] = None,
) -> ModelResourceMetrics:
    """Query owner-qualified history for current associated resources.

    Args:
        model: Model whose visibility the route has already authorized.
        instances: Authorized instance and role selection for this Model.
        ctx: Caller context used for independent Worker visibility checks.
        session: Database session for the current Worker inventory.
        window_seconds: Selected history interval, from 5 minutes to 7 days.
        device_id: Opaque device identifier from this authorized association set.
        client: Optional shared Prometheus HTTP session.

    Returns:
        Sanitized hardware observations, current inventory and local missing states.
    """
    if type(window_seconds) is not int or not 300 <= window_seconds <= 604800:
        raise InvalidException(message='Invalid monitoring window')
    end = time.time()
    devices, workers, associations, allocated_workers = await _associated_devices(
        instances, ctx, session
    )
    selected = (
        next((d for d in devices if d.public.id == device_id), None)
        if device_id
        else next(iter(devices), None)
    )
    if device_id and selected is None:
        raise InvalidException(message='Device is not associated with this model')
    response = ModelResourceMetrics(
        model_id=model.id,
        start=end - window_seconds,
        end=end,
        step=max(15, math.ceil(window_seconds / 240)),
        last_refreshed_at=end,
        history_scope=HISTORY_SCOPE,
        allocated_gpu_count=len(associations),
        allocated_worker_count=len(allocated_workers),
        devices=[d.public for d in devices],
        selected_device_id=selected.public.id if selected else None,
    )
    if not workers:
        response.reason = 'resource_not_visible' if allocated_workers else 'no_samples'
        response.metrics = {
            key: _empty(
                key,
                'restricted' if allocated_workers else 'not_applicable',
                response.reason,
                'worker' if key.startswith('host_') else 'physical_gpu',
            )
            for key in _DEFINITIONS
        }
        response.last_refreshed_at = time.time()
        return response
    worker = selected.worker if selected else workers[min(workers)]
    url = prometheus_query.prometheus_url()
    if url is None:
        response.reason = 'prometheus_unavailable'
        for device in devices:
            if device.identity is not None:
                device.public.reason = response.reason
        response.metrics = {
            key: _empty(
                key,
                'query_error',
                response.reason,
                (
                    'worker'
                    if key.startswith('host_')
                    else selected.public.scope if selected else 'physical_gpu'
                ),
            )
            for key in _DEFINITIONS
        }
        response.last_refreshed_at = time.time()
        return response
    if client is None:
        async with aiohttp.ClientSession() as owned_client:
            await _collect_history(
                devices, selected, worker, response, owned_client, url
            )
    else:
        await _collect_history(devices, selected, worker, response, client, url)
    response.available = any(
        m.status in ('available', 'stale') for m in response.metrics.values()
    )
    if selected and selected.identity is not None:
        gpu_metrics = [
            m for key, m in response.metrics.items() if key.startswith('gpu_')
        ]
        if any(m.status in ('available', 'stale') for m in gpu_metrics):
            selected.public.history_available = True
            selected.public.reason = None
        elif selected.public.history_available:
            selected.public.history_available = False
            selected.public.reason = (
                'query_failed'
                if any(m.status == 'query_error' for m in gpu_metrics)
                else 'no_samples'
            )
    response.reason = None if response.available else 'no_samples'
    response.last_refreshed_at = time.time()
    return response
