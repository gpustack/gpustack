"""Benchmark-time observations over recorded associations and measured phases."""

from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
import time
from types import SimpleNamespace
from typing import Optional

import aiohttp

from gpustack.api.exceptions import (
    BadRequestException,
    ForbiddenException,
    NotFoundException,
)
from gpustack.api.tenant import assert_resource_visible, tenant_stream_filter
from gpustack.schemas.benchmark import Benchmark, BenchmarkResult
from gpustack.schemas.benchmark_monitoring import (
    BenchmarkModelMetrics,
    BenchmarkMonitoringContext,
    BenchmarkMonitoringMember,
    BenchmarkMonitoringStage,
    BenchmarkMonitoringWindow,
    BenchmarkResourceMetrics,
    BenchmarkRunDevice,
    BenchmarkRunEvent,
    BenchmarkRunMember,
    BenchmarkRunSnapshot,
    BenchmarkRunWorker,
    BenchmarkSampleStatistics,
)
from gpustack.schemas.model_metrics import (
    MetricSummary,
    ModelMetric,
    ModelMonitoringCapabilities,
    ResourceDevice,
    ResourceWorker,
)
from gpustack.schemas.models import Model, ModelInstance
from gpustack.schemas.principals import OrgRole, PrincipalType
from gpustack.schemas.workers import Worker
from gpustack.utils.hardware_metrics import (
    gpu_identity,
    is_gpu_partition,
    verified_gpu_devices,
)
from gpustack.utils.model_instance_workers import (
    model_instance_metrics_source_inventory,
)
from . import hardware, prometheus
from .catalog import HARDWARE_KEYS, model_metric_definitions
from .config import get_query_config
from .models import _Collection, collect_model_metrics, parse_instance_ids
from .time_range import MAX_QUERY_SECONDS, MonitoringInterval, collect_storage_policy

HISTORY_SCOPE = 'benchmark_snapshot'


def _timestamp(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and value >= 0 else None


def benchmark_windows(results):
    """Read actual scheduler measurement and traffic bounds, retaining missing phases."""
    stages, windows = [], []
    for point in sorted(results, key=lambda p: (p.input_tokens or 0, p.sequence)):
        stage = BenchmarkMonitoringStage(
            id=f'{point.input_tokens or 0}:{point.sequence}',
            sequence=point.sequence,
            input_tokens=point.input_tokens,
            rate=point.rate,
        )
        raw = point.raw_metrics if isinstance(point.raw_metrics, dict) else {}
        timing = raw.get('scheduler_metrics')
        timing = timing if isinstance(timing, dict) else {}
        start, end, traffic_start, traffic_end = (
            _timestamp(timing.get(key))
            for key in (
                'measure_start_time',
                'measure_end_time',
                'request_start_time',
                'request_end_time',
            )
        )
        if start is not None and end is not None and start < end:
            stage.start, stage.end = start, end
            stage.status, stage.reason = 'available', None
        if (
            traffic_start is not None
            and traffic_end is not None
            and traffic_start < traffic_end
            and (start is None or traffic_start <= start)
            and (end is None or end <= traffic_end)
        ):
            stage.traffic_start, stage.traffic_end = traffic_start, traffic_end
            windows.append((traffic_start, traffic_end))
        stages.append(stage)
    # A partially timed run cannot be labelled a complete traffic interval.
    window = (
        BenchmarkMonitoringWindow(
            start=min(w[0] for w in windows), end=max(w[1] for w in windows)
        )
        if stages and len(windows) == len(stages)
        else None
    )
    return stages, window


def exact_interval(start, end) -> MonitoringInterval:
    """Resolve a measured sub-second boundary without widening a short phase."""
    start, end = _timestamp(start), _timestamp(end)
    if start is None or end is None or not 0.001 <= end - start <= MAX_QUERY_SECONDS:
        raise ValueError('Benchmark range must span 1 millisecond to 7 days')
    if end > time.time():
        raise ValueError('Benchmark range cannot end in the future')
    return MonitoringInterval(start, end, max(1, math.ceil(end - start)))


def sample_statistics(series, start, end, *, step, threshold=None):
    """Describe valid sampled values in [start, end), never a neighboring phase."""
    values = [(t, v) for t, v in series.points if start <= t < end and v is not None]
    expected = max(1, math.ceil((end - start) / step))
    result = BenchmarkSampleStatistics(
        dimensions=series.dimensions,
        expected_samples=expected,
        observed_samples=len(values),
        coverage=min(1, len(values) / expected),
        threshold=threshold,
    )
    if values:
        result.average = sum(v for _, v in values) / len(values)
        result.peak_at, result.peak = max(values, key=lambda pair: pair[1])
        result.last_at, result.last = max(values, key=lambda pair: pair[0])
        if threshold is not None:
            result.threshold_fraction = sum(v >= threshold for _, v in values) / len(
                values
            )
    return result


def _public_device(device, workers):
    return ResourceDevice(
        id=device.id,
        worker_id=device.worker_id,
        worker_name=workers[device.worker_id].name,
        name=device.name,
        index=device.index,
        chip_index=device.chip_index,
        scope=device.scope,
        model_instance_ids=device.model_instance_ids,
        history_available=False,
        reason='no_samples' if device.uuid else 'device_identity_unresolved',
    )


def _visible(ctx, resource, model):
    return resource is not None and tenant_stream_filter(ctx, model)(resource)


@dataclass
class BenchmarkHistory:
    context: BenchmarkMonitoringContext
    snapshot: Optional[BenchmarkRunSnapshot] = None
    model: object = None
    instances: list = field(default_factory=list)
    workers: dict = field(default_factory=dict)
    devices: list = field(default_factory=list)


async def _authorized_associations(history, session, ctx):
    snapshot, context = history.snapshot, history.context
    model = await Model.one_by_id(session, snapshot.model_id)
    if (
        not _visible(ctx, model, Model)
        or model.owner_principal_id != snapshot.owner_principal_id
    ):
        context.reason = 'model_not_visible'
        return
    for member in snapshot.instances:
        current = await ModelInstance.one_by_id(session, member.id)
        if (
            _visible(ctx, current, ModelInstance)
            and current.model_id == snapshot.model_id
        ):
            history.instances.append(member)
    visible_ids = {member.id for member in history.instances}
    allowed_workers = {id for member in history.instances for id in member.worker_ids}
    for worker in snapshot.workers:
        if worker.id not in allowed_workers:
            continue
        current = await Worker.one_by_id(session, worker.id)
        if (
            _visible(ctx, current, Worker)
            and current.owner_principal_id == worker.owner_principal_id
            and current.worker_uuid == worker.worker_uuid
        ):
            # Use captured identity and names; current rows only authorize reading it.
            history.workers[worker.id] = worker
    for device in snapshot.devices:
        members = sorted(set(device.model_instance_ids) & visible_ids)
        if device.worker_id in history.workers and members:
            history.devices.append(
                device.model_copy(update={'model_instance_ids': members})
            )
    context.instances = [
        BenchmarkMonitoringMember(
            **{
                'id': member.id,
                'name': member.name,
                'role': member.role,
                'worker_id': (
                    member.worker_id if member.worker_id in history.workers else None
                ),
                'worker_ids': [id for id in member.worker_ids if id in history.workers],
                'gpu_indexes': (
                    member.gpu_indexes if member.worker_id in history.workers else []
                ),
            }
        )
        for member in history.instances
    ]
    context.workers = [
        ResourceWorker(id=worker.id, name=worker.name)
        for _, worker in sorted(history.workers.items())
    ]
    context.devices = [
        _public_device(device, history.workers) for device in history.devices
    ]
    context.events = [
        event
        for event in snapshot.events
        if event.model_instance_id is None or event.model_instance_id in visible_ids
    ]
    context.available = bool(history.instances)
    context.reason = None if context.available else 'model_instance_not_visible'
    context.resource_status = 'available' if history.workers else 'restricted'
    context.resource_reason = None if history.workers else 'resource_not_visible'
    history.model = SimpleNamespace(
        id=snapshot.model_id,
        cluster_id=snapshot.cluster_id,
        backend=snapshot.backend,
        roles=snapshot.roles,
        disaggregation=snapshot.disaggregation,
    )


async def load_benchmark_history(session, ctx, id, *, client=None) -> BenchmarkHistory:
    """Check each resource independently before exposing historical identities."""
    if ctx is None or getattr(ctx.user, 'kind', None) == PrincipalType.SYSTEM:
        raise ForbiddenException(
            message='Benchmark monitoring requires a user principal'
        )
    ctx.assert_org_role(OrgRole.OWNER)
    benchmark = await Benchmark.one_by_id(session, id)
    assert_resource_visible(ctx, benchmark, not_found_message='Benchmark not found')
    results = await BenchmarkResult.all_by_fields(session, fields={'benchmark_id': id})
    stages, window = benchmark_windows(results)
    capabilities = ModelMonitoringCapabilities(
        model_id=benchmark.model_id or 0, min_query_seconds=1
    )
    history = BenchmarkHistory(
        context=BenchmarkMonitoringContext(
            benchmark_id=id,
            model_id=benchmark.model_id,
            run_window=window,
            stages=stages,
            capabilities=capabilities,
        ),
        snapshot=benchmark.snapshot.monitoring if benchmark.snapshot else None,
    )
    if history.snapshot is None:
        history.context.reason = 'historical_association_unavailable'
        history.context.resource_reason = 'historical_association_unavailable'
        return history
    await _authorized_associations(history, session, ctx)
    if history.model is not None:
        history.context.capabilities = capabilities.model_copy(
            update=await collect_storage_policy(client)
        )
    if history.snapshot.confirmed_at is None:
        history.context.available = False
        history.context.reason = 'run_association_unconfirmed'
    if window is None:
        history.context.available = False
        history.context.reason = 'run_timing_unavailable'
    if any(e.kind == 'generation_changed' for e in history.context.events):
        history.context.available = False
        history.context.reason = 'historical_association_changed'
        history.context.resource_status = 'unavailable'
        history.context.resource_reason = 'historical_association_changed'
    return history


def _selected_interval(history, phase_id, start, end):
    context = history.context
    if (start is None) != (end is None):
        raise BadRequestException(message='Both start and end are required')
    stage = None
    if phase_id is not None:
        stage = next((s for s in context.stages if s.id == phase_id), None)
        if stage is None:
            raise NotFoundException(message='Benchmark phase not found')
    window = stage if stage is not None else context.run_window
    if window is None or window.start is None or window.end is None:
        return None
    if start is None:
        start, end = window.start, window.end
    if start < window.start or end > window.end:
        raise BadRequestException(
            message='Monitoring range must stay within the measured interval'
        )
    try:
        interval = exact_interval(start, end)
    except ValueError as exc:
        raise BadRequestException(message=str(exc)) from exc
    return interval


def _statistics(response):
    if response.start is None or response.end is None:
        return response
    for key, metric in response.metrics.items():
        if metric.kind == 'gauge':
            response.sample_statistics[key] = [
                sample_statistics(
                    line,
                    response.start,
                    response.end,
                    step=response.step,
                    threshold=90 if key == 'gpu_utilization' else None,
                )
                for line in metric.series
            ]
    return response


def _empty_engine(history, interval, phase_id, reason):
    policy = get_query_config().time_windows.llm
    step = policy.step(interval.seconds if interval else 1)
    response = BenchmarkModelMetrics(
        model_id=history.context.model_id,
        start=interval.start if interval else None,
        end=interval.end if interval else None,
        step=step,
        rate_window_seconds=policy.rate_window(step),
        last_refreshed_at=time.time(),
        history_scope=HISTORY_SCOPE,
        phase_id=phase_id,
        reason=reason,
    )
    for definition in model_metric_definitions(
        history.snapshot.backend.lower() if history.snapshot else ''
    ):
        response.metrics[definition.key] = ModelMetric(
            kind=definition.kind,
            unit=definition.unit,
            measurement_scope=definition.measurement_scope,
            status='restricted' if reason.endswith('not_visible') else 'no_data',
            reason=reason,
        )
    return response


async def collect_benchmark_model_metrics(
    history,
    *,
    phase_id=None,
    start=None,
    end=None,
    instance_ids=None,
    role='all',
    view='aggregate',
    statistic='avg,p50,p95,p99',
    section='all',
    client=None,
):
    """Reuse role-aware normalized queries with frozen model and launch identities."""
    interval = _selected_interval(history, phase_id, start, end)
    if history.context.reason == 'historical_association_changed':
        return _empty_engine(history, interval, phase_id, history.context.reason)
    if interval is None or history.model is None or not history.snapshot.confirmed_at:
        return _empty_engine(
            history,
            interval,
            phase_id,
            history.context.reason or 'phase_timing_unavailable',
        )
    instances = history.instances
    if instance_ids is not None:
        requested = parse_instance_ids(instance_ids)
        if not requested.issubset({i.id for i in instances}):
            raise NotFoundException(message='Model instance not found')
        instances = [i for i in instances if i.id in requested]
    instances = [i for i in instances if role == 'all' or i.role == role]
    launches = tuple(
        (i.id, int(source['worker_id']), int(source['restart_count']))
        for i in instances
        for source in i.source_identities
    )
    instances = [i for i in instances if any(launch[0] == i.id for launch in launches)]
    if not instances:
        return _empty_engine(history, interval, phase_id, 'source_identity_unresolved')
    response = await collect_model_metrics(
        history.model,
        instances,
        window_seconds=interval.seconds,
        interval=interval,
        role=role,
        view=view,
        statistics=tuple(dict.fromkeys(statistic.split(','))),
        section=section,
        upstream_instances=history.instances,
        client=client,
        launches=launches,
        source_identities=tuple(
            source for i in instances for source in i.source_identities
        ),
    )
    response = BenchmarkModelMetrics(**response.model_dump(), phase_id=phase_id)
    response.history_scope = HISTORY_SCOPE
    _project_samples(response, history, phase_id)
    return _statistics(response)


def _hardware_devices(history):
    return [
        hardware._Device(
            _public_device(device, history.workers),
            history.workers[device.worker_id],
            (
                (device.uuid, device.chip_index)
                if device.uuid
                and hardware._worker_identity(history.workers[device.worker_id])
                else None
            ),
        )
        for device in history.devices
    ]


async def collect_benchmark_resource_metrics(
    history,
    *,
    phase_id=None,
    start=None,
    end=None,
    client=None,
):
    """Query every visible captured GPU/Worker using its stable owner identity."""
    interval = _selected_interval(history, phase_id, start, end)
    devices = _hardware_devices(history)
    if max(len(devices), len(history.workers)) > hardware.MAX_RESOURCE_TARGETS:
        raise BadRequestException(message='Too many associated resources for one query')
    response = BenchmarkResourceMetrics(
        model_id=history.context.model_id,
        start=interval.start if interval else None,
        end=interval.end if interval else None,
        step=get_query_config().time_windows.hardware.step(
            interval.seconds if interval else 1
        ),
        last_refreshed_at=time.time(),
        history_scope=HISTORY_SCOPE,
        phase_id=phase_id,
        all_devices=True,
        devices=[d.public for d in devices],
        workers=history.context.workers,
        allocated_gpu_count=len(devices),
        allocated_worker_count=len(history.workers),
    )
    reason = None
    if interval is None or history.model is None or not history.snapshot.confirmed_at:
        reason = history.context.reason or 'phase_timing_unavailable'
    elif history.context.resource_status != 'available':
        reason = history.context.resource_reason or 'resource_not_visible'
    url = prometheus.prometheus_url()
    if reason or not url:
        response.reason = reason or 'prometheus_unavailable'
        response.metrics = {
            key: hardware._empty(
                key,
                'restricted' if reason == 'resource_not_visible' else 'no_data',
                response.reason,
                'worker' if key.startswith('host_') else 'physical_gpu',
            )
            for key in HARDWARE_KEYS
        }
        return response
    if client is None:
        async with aiohttp.ClientSession() as owned:
            await hardware._collect_all_history(
                devices,
                history.workers,
                response,
                owned,
                url,
                not_before=interval.start,
            )
    else:
        await hardware._collect_all_history(
            devices,
            history.workers,
            response,
            client,
            url,
            not_before=interval.start,
        )
    _project_samples(response, history, phase_id)
    hardware._update_device_history(devices, response)
    response.available = any(
        m.status in ('available', 'stale') for m in response.metrics.values()
    )
    response.reason = (
        None
        if response.available
        else (
            'query_failed'
            if any(m.status == 'query_error' for m in response.metrics.values())
            else 'no_samples'
        )
    )
    return _statistics(response)


def _project_samples(response, history, phase_id):
    """Exclude boundary samples and rollups whose lookback crosses a phase."""
    for metric in response.metrics.values():
        observations = list(metric.series) + list(metric.summaries)
        if response.source == 'engine' and any(
            o.dimensions.worker_id is not None
            and o.dimensions.worker_id not in history.workers
            for o in observations
        ):
            metric.status, metric.reason = 'restricted', 'resource_not_visible'
            metric.series, metric.summaries, metric.coverage = [], [], None
            continue
        for series in metric.series:
            rolling = metric.kind != 'gauge'
            earliest = response.start + (response.rate_window_seconds if rolling else 0)
            series.points = [
                (t, v if t >= earliest else None)
                for t, v in series.points
                if response.start <= t < response.end
            ]
        if (
            metric.kind != 'gauge'
            and metric.series
            and metric.status not in ('query_error', 'unsupported', 'not_applicable')
            and response.end - response.start <= response.rate_window_seconds
        ):
            metric.reason = 'insufficient_phase_rate_window'
        if metric.kind == 'gauge':
            metric.summaries = []
            for line in metric.series:
                values = [(t, v) for t, v in line.points if v is not None]
                metric.summaries.append(
                    MetricSummary(
                        statistic='current',
                        dimensions=line.dimensions,
                        value=max(values)[1] if values else None,
                    )
                )
            if metric.status in ('available', 'stale') and not any(
                summary.value is not None for summary in metric.summaries
            ):
                metric.status, metric.reason = 'no_data', 'no_samples'
        for observation in list(metric.series) + list(metric.summaries):
            if observation.dimensions.worker_id not in history.workers:
                observation.dimensions.worker_id = None
                observation.dimensions.source_name = None
                if hasattr(observation, 'name'):
                    observation.name = (
                        ' · '.join(
                            str(v)
                            for v in (
                                observation.dimensions.role,
                                observation.dimensions.model_instance_name,
                            )
                            if v
                        )
                        or 'Deployment'
                    )


def _source_identities(member):
    return [
        {
            'model_instance_id': str(source.instance_id),
            'worker_id': str(source.worker_id),
            'restart_count': str(source.restart_count),
            'source_kind': source.kind,
            'source_name': source.name,
            'endpoint_dp_rank': (
                str(source.dp_rank) if source.dp_rank is not None else ''
            ),
            'component_attempt': (
                str(source.attempt) if source.attempt is not None else ''
            ),
        }
        for source in model_instance_metrics_source_inventory(member)
    ]


async def verify_benchmark_run(benchmark, session, *, confirm=False):
    """Observe fixed member generations without substituting current membership."""
    snapshot = benchmark.snapshot
    recorded = snapshot.monitoring if snapshot else None
    if recorded is None:
        return None
    recorded = recorded.model_copy(deep=True)
    now = time.time()
    changed = set()
    for member in recorded.instances:
        current = await ModelInstance.one_by_id(session, member.id)
        if (
            current is None
            or current.pid != member.pid
            or current.worker_id != member.worker_id
            or current.restart_count != member.restart_count
            or _source_identities(current) != member.source_identities
            or hardware._allocated_gpus(current)
            != {id: set(indexes) for id, indexes in member.allocations.items()}
        ):
            changed.add(member.id)
    seen = {
        e.model_instance_id for e in recorded.events if e.kind == 'generation_changed'
    }
    for worker in recorded.workers:
        current = await Worker.one_by_id(session, worker.id)
        if (
            current is None
            or current.worker_uuid != worker.worker_uuid
            or current.owner_principal_id != worker.owner_principal_id
        ):
            changed.update(
                m.id for m in recorded.instances if worker.id in m.worker_ids
            )
    recorded.events.extend(
        BenchmarkRunEvent(at=now, kind='generation_changed', model_instance_id=id)
        for id in sorted(changed - seen)
    )
    if confirm and not changed:
        recorded.confirmed_at = now
    return snapshot.model_copy(update={'monitoring': recorded})


async def capture_benchmark_run(snapshot, model, members, session, *, workers=None):
    """Freeze the selected deployment and hardware identities when creating the run."""
    captured_at = time.time()
    recorded = BenchmarkRunSnapshot(
        captured_at=captured_at,
        model_id=model.id,
        cluster_id=model.cluster_id,
        owner_principal_id=model.owner_principal_id,
        backend=str(getattr(model.backend, 'value', model.backend)),
        roles=model.roles or [],
        disaggregation=model.disaggregation,
        events=[BenchmarkRunEvent(at=captured_at, kind='association_captured')],
    )
    for member in members:
        allocations = hardware._allocated_gpus(member)
        recorded.instances.append(
            BenchmarkRunMember(
                id=member.id,
                name=member.name,
                role=member.role,
                worker_id=member.worker_id,
                worker_ip=member.worker_ip,
                port=member.port,
                pid=member.pid,
                restart_count=member.restart_count,
                gpu_indexes=member.gpu_indexes or [],
                worker_ids=sorted(allocations),
                source_identities=_source_identities(member),
                source_addresses=sorted(_Collection.instance_addresses(member)),
                allocations={
                    id: sorted(indexes) for id, indexes in allocations.items()
                },
            )
        )
    worker_ids = {id for member in recorded.instances for id in member.worker_ids}
    for worker_id in sorted(worker_ids):
        worker = workers.get(worker_id) if workers is not None else None
        if worker is None:
            worker = await Worker.one_by_id(session, worker_id)
        if worker is None:
            continue
        recorded.workers.append(
            BenchmarkRunWorker(
                id=worker_id,
                name=worker.name,
                worker_uuid=getattr(worker, 'worker_uuid', None),
                owner_principal_id=getattr(worker, 'owner_principal_id', None),
            )
        )
        indexes = {}
        for member in members:
            for index in hardware._allocated_gpus(member).get(worker_id, []):
                indexes.setdefault(index, []).append(member.id)
        status = getattr(worker, 'status', None)
        candidates = status.gpu_devices if status else []
        verified = verified_gpu_devices(candidates or [])
        for index in sorted(indexes):
            matching = [device for device in candidates or [] if device.index == index]
            device = matching[0] if len(matching) == 1 else None
            identity = gpu_identity(device) if device in verified else None
            opaque = sha256(
                json.dumps(
                    [
                        worker_id,
                        getattr(worker, 'worker_uuid', None),
                        getattr(worker, 'owner_principal_id', None),
                        identity or ['unresolved', index],
                    ]
                ).encode()
            ).hexdigest()[:32]
            recorded.devices.append(
                BenchmarkRunDevice(
                    id=opaque,
                    worker_id=worker_id,
                    name=device.name if device else 'GPU',
                    index=index,
                    uuid=identity[0] if identity else None,
                    chip_index=identity[1] if identity else 0,
                    scope=(
                        'partition'
                        if device and is_gpu_partition(device)
                        else 'physical_gpu'
                    ),
                    model_instance_ids=sorted(indexes[index]),
                )
            )
    snapshot.monitoring = recorded
    return snapshot
