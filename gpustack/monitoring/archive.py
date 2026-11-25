"""Enrollment and tenant-authorized replay of Benchmark monitoring archives."""

import asyncio
from collections import OrderedDict
import math
import time

from sqlmodel import select

from gpustack.api.exceptions import (
    ConflictException,
    ForbiddenException,
    InvalidException,
    NotFoundException,
    ServiceUnavailableException,
)
from gpustack.config.config import get_global_config
from gpustack.monitoring.benchmark_access import (
    assert_benchmark_monitoring_visible,
    require_benchmark_monitoring_user,
)
from gpustack.monitoring.benchmark_payload import (
    ARCHIVE_FORMAT_VERSION,
    BenchmarkPayloadError,
    BenchmarkPayloadStore,
    database_now,
    decode_payload,
    locked_payload,
)
from gpustack.monitoring.config import get_query_config
from gpustack.schemas.benchmark import (
    BenchmarkResult,
    BenchmarkStateEnum,
    benchmark_load_mode,
)
from gpustack.schemas.benchmark_monitoring import (
    BenchmarkArchiveQueryCapabilities,
    BenchmarkModelMetrics,
    BenchmarkMonitoringContext,
    BenchmarkObservationStorage,
    BenchmarkResourceMetrics,
)
from gpustack.schemas.benchmark_monitoring_payload import BenchmarkMonitoringPayload
from gpustack.server.benchmark_run_identity import assert_benchmark_run

ACTIVE_BENCHMARK_STATES = (
    BenchmarkStateEnum.QUEUED,
    BenchmarkStateEnum.PENDING,
    BenchmarkStateEnum.RUNNING,
)
MAX_PHASE_BATCH = 50
_CACHE_LIMIT = 16 * 1024 * 1024
_CACHE = OrderedDict()
_CACHE_SIZE = 0


def _scheduler_timing(point):
    raw = point.raw_metrics if isinstance(point.raw_metrics, dict) else {}
    values = raw.get('scheduler_metrics', {})
    if not isinstance(values, dict):
        return {}
    return {
        key: value
        for key in (
            'measure_start_time',
            'measure_end_time',
            'request_start_time',
            'request_end_time',
        )
        if isinstance((value := values.get(key)), (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    }


def _authorize(ctx, benchmark) -> None:
    require_benchmark_monitoring_user(ctx)
    assert_benchmark_monitoring_visible(ctx, benchmark)


def _hardware_allowed(archive, ctx) -> bool:
    return archive.hardware_visible and (
        ctx.is_platform_admin or ctx.current_principal_id == archive.owner_principal_id
    )


def _scope(context, phase_id, start, end):
    if phase_id is not None:
        window = next(
            (s for s in context.get('stages', []) if s['id'] == phase_id), None
        )
        if window is None:
            raise NotFoundException(message='Benchmark phase not found')
    else:
        window = context.get('run_window')
    if (start is None) != (end is None):
        raise InvalidException(message='Both start and end are required')
    if start is not None and (
        window is None or start != window.get('start') or end != window.get('end')
    ):
        raise InvalidException(
            message='Archived summaries support only the complete run or a recorded phase',
            details={
                'supported_scopes': ['run', 'phase'],
                'custom_range_summary': False,
            },
        )
    return f'phase:{phase_id}' if phase_id is not None else 'run'


def _filter_metrics(
    response, statistic, section, *, hide_hardware=False, sections=None
):
    from gpustack.monitoring.catalog import model_metric_definitions

    statistics = set(statistic.split(','))
    if not statistics or not statistics <= {'avg', 'p50', 'p95', 'p99'}:
        raise InvalidException(message='Unsupported latency statistic')
    if section not in ('all', 'llm', 'pd'):
        raise InvalidException(message='Unsupported monitoring section')
    if section != 'all':
        sections = sections or {m.key: m.section for m in model_metric_definitions('')}
        allowed = {key for key, value in sections.items() if value == section}
        response.metrics = {k: v for k, v in response.metrics.items() if k in allowed}
        response.sample_statistics = {
            k: v for k, v in response.sample_statistics.items() if k in allowed
        }
    for metric in response.metrics.values():
        metric.series = [
            line
            for line in metric.series
            if line.statistic in statistics or line.statistic in ('current', 'rate')
        ]
        if hide_hardware:
            for observation in list(metric.series) + list(metric.summaries):
                observation.dimensions.worker_id = None
                observation.dimensions.source_name = None
                observation.dimensions.device_id = None
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
    if hide_hardware:
        for samples in response.sample_statistics.values():
            for sample in samples:
                sample.dimensions.worker_id = None
                sample.dimensions.source_name = None
                sample.dimensions.device_id = None
    return response


def archive_content_changed():
    return ConflictException(
        message='Benchmark monitoring content changed; reload context',
        details={'code': 'benchmark_archive_content_changed'},
    )


async def _metadata(session, benchmark, run_id):
    if run_id is not None:
        assert_benchmark_run(benchmark, run_id)
    if not benchmark.run_id:
        return None
    result = await session.exec(
        select(BenchmarkMonitoringPayload)
        .where(
            BenchmarkMonitoringPayload.benchmark_id == benchmark.id,
            BenchmarkMonitoringPayload.run_id == benchmark.run_id,
        )
        .execution_options(populate_existing=True)
    )
    row = result.first()
    if row is not None and row.owner_principal_id != benchmark.owner_principal_id:
        raise ForbiddenException(message='Benchmark archive ownership does not match')
    return row


def _storage(row, ctx):
    version = str(row.write_seq) if row.payload_sha256 else None
    state = row.state
    if row.last_error and (row.repair_requested or not row.sealed):
        state = 'failed'
    elif row.repair_requested:
        state = 'finalizing'
    return {
        'run_id': row.run_id,
        'archive_state': state,
        'content_version': version,
        'compressed_bytes': (
            row.compressed_bytes
            if version is not None and _hardware_allowed(row, ctx)
            else None
        ),
        'captured_until': row.captured_until,
        'last_error': row.last_error,
        'sealed': row.sealed,
        'expires_at': row.expires_at,
        'data_coverage': row.data_coverage,
        'capture_complete': row.capture_complete,
        'retryable': not row.repair_requested
        and state in ('failed', 'partial', 'no_data')
        and row.last_error != 'archive_scope_budget_exceeded',
    }


async def _saved(session, ctx, benchmark, run_id, content_version, *, hardware=False):
    _authorize(ctx, benchmark)
    row = await _metadata(session, benchmark, run_id)
    if row is None or not row.payload_sha256:
        if content_version is not None:
            raise archive_content_changed()
        return None
    permitted = _hardware_allowed(row, ctx)
    if hardware and not permitted:
        raise ForbiddenException(
            message='Historical hardware is not visible in this organization'
        )
    metadata = _storage(row, ctx)
    if content_version is not None and content_version != metadata['content_version']:
        raise archive_content_changed()
    package = await _read_package(session, row)
    return package, metadata, permitted, row.plan


async def _read_package(session, row):
    global _CACHE_SIZE
    if row.format_version != ARCHIVE_FORMAT_VERSION:
        raise ServiceUnavailableException(
            message='Unsupported Benchmark archive format'
        )
    key = (row.benchmark_id, row.run_id, row.payload_sha256)
    limit = getattr(
        get_global_config(), 'benchmark_archive_max_run_bytes', 256 * 1024 * 1024
    )
    if max(row.compressed_bytes, row.uncompressed_bytes) > limit:
        raise ServiceUnavailableException(
            message='Benchmark monitoring archive exceeds the read limit',
            details={'code': 'archive_payload_too_large'},
        )
    cached = _CACHE.get(key)
    if cached is not None:
        _CACHE.move_to_end(key)
        return cached[0]

    result = await session.exec(
        select(BenchmarkMonitoringPayload.payload).where(
            BenchmarkMonitoringPayload.benchmark_id == row.benchmark_id,
            BenchmarkMonitoringPayload.run_id == row.run_id,
            BenchmarkMonitoringPayload.write_seq == row.write_seq,
            BenchmarkMonitoringPayload.payload_sha256 == row.payload_sha256,
        )
    )
    blob = result.first()
    if blob is None:
        raise archive_content_changed()
    size = row.uncompressed_bytes
    # These endpoints own a read-only session. Return its connection before
    # inflating or validating the archive.
    await session.close()
    try:
        package = await asyncio.to_thread(
            decode_payload,
            blob,
            sha256=key[-1],
            uncompressed_bytes=size,
            max_bytes=limit,
        )
        if not isinstance(package.get('context'), dict) or not isinstance(
            package.get('scopes'), dict
        ):
            raise BenchmarkPayloadError('archive_payload_corrupt')
    except BenchmarkPayloadError as exc:
        raise ServiceUnavailableException(
            message='Benchmark monitoring archive is unreadable',
            details={'code': exc.reason},
        ) from None
    if size <= _CACHE_LIMIT and key not in _CACHE:
        while _CACHE and (_CACHE_SIZE + size > _CACHE_LIMIT or len(_CACHE) >= 16):
            _, (_, evicted_size) = _CACHE.popitem(last=False)
            _CACHE_SIZE -= evicted_size
        _CACHE[key] = package, size
        _CACHE_SIZE += size
    return package


def _decorate(response, metadata):
    response.observation_storage = BenchmarkObservationStorage.model_validate(metadata)
    response.query_capabilities = BenchmarkArchiveQueryCapabilities()
    return response


def _restrict_context(context):
    context.workers, context.devices = [], []
    context.resource_status = 'restricted'
    context.resource_reason = 'resource_not_visible'
    for member in context.instances:
        member.worker_id, member.worker_ids, member.gpu_indexes = None, [], []
    return context


async def read_archive_context(
    session, ctx, benchmark, *, run_id=None, content_version=None
):
    saved = await _saved(session, ctx, benchmark, run_id, content_version)
    if saved is None:
        return None
    package, metadata, permitted, _ = saved
    context = BenchmarkMonitoringContext.model_validate(package['context'])
    if not permitted:
        _restrict_context(context)
    context.capabilities.supports_custom_range = False
    return _decorate(context, metadata)


def _entry(package, key, plan, phase_id, *, summaries_only=False):
    entry = package['scopes'].get(key)
    if entry is None:
        return {
            'model_id': package['context'].get('model_id'),
            'phase_id': phase_id,
            'step': plan.get('step', 5),
            'rate_window_seconds': plan.get('rate_window', 75),
            'last_refreshed_at': time.time(),
            'history_scope': 'benchmark_snapshot',
            'reason': 'archive_scope_unavailable',
        }
    if summaries_only:
        return {
            **entry,
            'metrics': {
                k: {**metric, 'series': []}
                for k, metric in entry.get('metrics', {}).items()
            },
        }
    return entry


async def read_archive_metrics(
    session,
    ctx,
    benchmark,
    *,
    phase_id=None,
    start=None,
    end=None,
    instance_ids=None,
    role='all',
    view='aggregate',
    statistic='avg,p50,p95,p99',
    section='all',
    run_id=None,
    content_version=None,
):
    saved = await _saved(session, ctx, benchmark, run_id, content_version)
    if saved is None:
        return None
    package, metadata, permitted, plan = saved
    if instance_ids is not None:
        raise InvalidException(
            message='Archived metrics do not support arbitrary instance subsets',
            details={'arbitrary_subset_summary': False},
        )
    scope = _scope(package['context'], phase_id, start, end)
    response = BenchmarkModelMetrics.model_validate(
        _entry(package, f'{scope}/runtime/{role}/{view}', plan, phase_id)
    )
    _filter_metrics(
        response,
        statistic,
        section,
        hide_hardware=not permitted,
        sections=plan.get('sections'),
    )
    return _decorate(response, metadata)


async def read_archive_resources(
    session,
    ctx,
    benchmark,
    *,
    phase_id=None,
    start=None,
    end=None,
    run_id=None,
    content_version=None,
):
    saved = await _saved(
        session, ctx, benchmark, run_id, content_version, hardware=True
    )
    if saved is None:
        return None
    package, metadata, _, plan = saved
    scope = _scope(package['context'], phase_id, start, end)
    response = BenchmarkResourceMetrics.model_validate(
        _entry(package, f'{scope}/resources/all/aggregate', plan, phase_id)
    )
    return _decorate(response, metadata)


async def read_archive_phase_observations(
    session,
    ctx,
    benchmark,
    *,
    phase_ids=None,
    run_id=None,
    content_version=None,
):
    saved = await _saved(session, ctx, benchmark, run_id, content_version)
    if saved is None:
        return None
    package, metadata, permitted, plan = saved
    ids = (
        phase_ids
        if phase_ids is not None
        else [s['id'] for s in package['context'].get('stages', [])]
    )
    if len(ids) > MAX_PHASE_BATCH:
        raise InvalidException(
            message=f'At most {MAX_PHASE_BATCH} phases may be read together'
        )
    phases = {}
    for phase_id in dict.fromkeys(ids):
        scope = _scope(package['context'], phase_id, None, None)
        runtime = BenchmarkModelMetrics.model_validate(
            _entry(
                package,
                f'{scope}/runtime/all/instances',
                plan,
                phase_id,
                summaries_only=True,
            )
        )
        _filter_metrics(
            runtime,
            'avg,p50,p95,p99',
            'all',
            hide_hardware=not permitted,
            sections=plan.get('sections'),
        )
        resources = None
        if permitted:
            resources = _decorate(
                BenchmarkResourceMetrics.model_validate(
                    _entry(
                        package,
                        f'{scope}/resources/all/aggregate',
                        plan,
                        phase_id,
                        summaries_only=True,
                    )
                ),
                metadata,
            )
        phases[phase_id] = {
            'runtime': _decorate(runtime, metadata),
            'resources': resources,
        }
    return {
        'run_id': metadata['run_id'],
        'content_version': metadata['content_version'],
        'phases': phases,
    }


async def archive_status(session, ctx, benchmark, *, run_id=None):
    _authorize(ctx, benchmark)
    row = await _metadata(session, benchmark, run_id)
    if row is None:
        return {'state': 'not_archived', 'run_id': benchmark.run_id, 'retryable': False}
    status = _storage(row, ctx)
    return {'state': status.pop('archive_state'), **status}


async def retry_archive(session, ctx, benchmark, *, run_id=None):
    _authorize(ctx, benchmark)
    row = await _metadata(session, benchmark, run_id)
    if row is None or not await BenchmarkPayloadStore().request_repair(
        session, benchmark.id, benchmark.run_id
    ):
        raise ConflictException(
            message='This monitoring archive is not eligible for retry'
        )
    await session.commit()
    return await archive_status(session, ctx, benchmark, run_id=run_id)


async def delete_archive(session, benchmark):
    await BenchmarkPayloadStore().delete(session, benchmark.id)


def _capture_snapshot(benchmark, monitoring, results):
    recorded = monitoring.model_dump(mode='json')
    recorded['roles'] = [{'name': role.name} for role in monitoring.roles]
    return {
        'monitoring': recorded,
        'target_mode': getattr(benchmark.target_mode, 'value', benchmark.target_mode),
        'via_gateway': bool(getattr(benchmark.snapshot, 'gateway_url', None)),
        'load_mode': benchmark_load_mode(benchmark),
        'results': [
            dict(
                input_tokens=point.input_tokens,
                sequence=point.sequence,
                rate=point.rate,
                raw_metrics={'scheduler_metrics': _scheduler_timing(point)},
            )
            for point in results
        ],
        'state': getattr(benchmark.state, 'value', benchmark.state),
        'observed_at': time.time(),
    }


def enroll_archive(session, benchmark):
    """Enroll only a newly created Benchmark in its creation transaction."""
    monitoring = getattr(benchmark.snapshot, 'monitoring', None)
    if monitoring is None or not benchmark.run_id:
        return
    cfg = get_global_config()
    catalog = get_query_config()
    policy = catalog.time_windows.llm
    source_interval = (
        getattr(cfg, 'benchmark_archive_source_interval_seconds', None) or 5
    )
    if not getattr(cfg, 'prometheus_url', None):
        source_interval = max(5, source_interval)
    step = max(
        getattr(cfg, 'benchmark_archive_sample_step_seconds', 5), source_interval
    )
    session.add(
        BenchmarkMonitoringPayload(
            benchmark_id=benchmark.id,
            run_id=benchmark.run_id,
            owner_principal_id=benchmark.owner_principal_id,
            snapshot=_capture_snapshot(benchmark, monitoring, []),
            plan={
                'step': step,
                'rate_window': max(
                    policy.rate_window(policy.min_step_seconds), source_interval * 4
                ),
                'sections': {
                    key: metric.section for key, metric in catalog.metrics.items()
                },
            },
        )
    )


async def update_archive_snapshot(session, benchmark, *, results=None):
    """Update the durable capture inputs for an enrolled execution."""
    if not benchmark.run_id:
        return
    row = await locked_payload(session, benchmark.id)
    if row is None or row.run_id != benchmark.run_id or row.sealed:
        return
    monitoring = getattr(benchmark.snapshot, 'monitoring', None)
    if monitoring is None:
        return
    if results is None:
        results = (
            await session.exec(
                select(BenchmarkResult)
                .where(
                    BenchmarkResult.benchmark_id == benchmark.id,
                    BenchmarkResult.deleted_at.is_(None),
                )
                .order_by(BenchmarkResult.sequence, BenchmarkResult.input_tokens)
            )
        ).all()
    snapshot = _capture_snapshot(benchmark, monitoring, results)
    was_confirmed = row.snapshot.get('monitoring', {}).get('confirmed_at')
    if not was_confirmed and monitoring.confirmed_at is not None:
        row.retry_at = 0
    if {k: v for k, v in snapshot.items() if k != 'observed_at'} != {
        k: v for k, v in row.snapshot.items() if k != 'observed_at'
    }:
        row.snapshot = snapshot
        row.snapshot_seq += 1
    if (
        benchmark.state not in ACTIVE_BENCHMARK_STATES
        and row.finalize_requested_at is None
    ):
        row.finalize_requested_at = await database_now(session)
        row.fencing_token += 1
        row.lease_owner = None
        row.lease_until = 0
        row.retry_at = 0
    session.add(row)
