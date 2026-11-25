"""Save Benchmark monitoring checkpoints in the shared database."""

import asyncio
from decimal import Decimal, ROUND_CEILING
import logging
import math
import time
from types import SimpleNamespace
from uuid import uuid4

import aiohttp
from sqlalchemy import or_
from sqlalchemy.orm import undefer
from sqlmodel import select

from gpustack.monitoring import benchmarks as collectors
from gpustack.monitoring.benchmark_payload import (
    BenchmarkPayloadError,
    BenchmarkPayloadStore,
    database_now,
    decode_payload,
    encode_payload,
)
from gpustack.monitoring.time_range import MonitoringInterval
from gpustack.routes.benchmarks import MAX_BENCHMARK_RESULT_POINTS
from gpustack.schemas.benchmark import Benchmark
from gpustack.schemas.benchmark_monitoring import (
    BenchmarkModelMetrics,
    BenchmarkMonitoringContext,
    BenchmarkMonitoringMember,
    BenchmarkMonitoringWindow,
    BenchmarkResourceMetrics,
    BenchmarkRunSnapshot,
)
from gpustack.schemas.benchmark_monitoring_payload import BenchmarkMonitoringPayload
from gpustack.schemas.model_metrics import (
    MetricSummary,
    ModelMonitoringCapabilities,
    ResourceWorker,
)
from gpustack.server.db import async_session

logger = logging.getLogger(__name__)
_TERMINAL = {'completed', 'stopped', 'error', 'unreachable'}
_LEASE_SECONDS = 30
_POLL_SECONDS = 5
# Budget enough scopes for MAX_BENCHMARK_RESULT_POINTS (500) stages,
# where each stage captures up to 9 scopes plus 9 for the run window.
_MAX_SCOPES = max(2048, 9 * (1 + MAX_BENCHMARK_RESULT_POINTS))
# Keep each range query below Prometheus' evaluation limit.
_MAX_RANGE_POINTS = 10000


class ArchiveLeaseLost(RuntimeError):
    pass


def _value(value):
    return getattr(value, 'value', value)


def _series_key(line):
    return line.statistic, tuple(sorted(line.dimensions.model_dump().items()))


def _has_observation(metric):
    return any(
        summary.value is not None or summary.lower_bound is not None
        for summary in metric.summaries
    ) or any(
        line.lower_bounds or any(value is not None for _, value in line.points)
        for line in metric.series
    )


def _merge_line(current, previous, start, end):
    """Use fresh observations while retaining durable values at missing ticks."""
    samples = dict(previous.points) if previous else {}
    bounds = dict(previous.lower_bounds or []) if previous else {}
    for at, value in current.points:
        if value is not None or samples.get(at) is None:
            samples[at] = value
    for at, bound in current.lower_bounds or []:
        bounds[at] = max(bound, bounds.get(at, bound))
    line = current.model_copy()
    line.points = sorted((at, v) for at, v in samples.items() if start <= at < end)
    line.lower_bounds = [
        (at, bounds[at]) for at, v in line.points if v is None and at in bounds
    ] or None
    return line


def _merge_metric(current, previous, start, end, *, same_window):
    if current.status == 'restricted':
        return current.model_copy(
            update={'series': [], 'summaries': [], 'coverage': None}
        )
    metric = current.model_copy()
    known = {_series_key(line): line for line in previous.series} if previous else {}
    metric.series = [
        _merge_line(line, known.pop(_series_key(line), None), start, end)
        for line in current.series
    ]
    metric.series.extend(_merge_line(line, None, start, end) for line in known.values())
    if same_window and previous:
        # Native summaries may only be carried across captures of the same
        # interval. Adjacent-window summaries never get combined.
        summaries = {_series_key(s): s for s in current.summaries}
        for saved in previous.summaries:
            latest = summaries.get(_series_key(saved))
            if latest is None or latest.value is None and latest.lower_bound is None:
                summaries[_series_key(saved)] = saved
        metric.summaries = list(summaries.values())
    if metric.status in ('query_error', 'no_data') and _has_observation(metric):
        if previous and previous.status in ('available', 'stale', 'no_observations'):
            metric.status, metric.reason = previous.status, previous.reason
    return metric


def _merge_scope(current, previous, start, end):
    result = current.model_copy(update={'start': start, 'end': end})
    old_metrics = previous.metrics if previous else {}
    same_window = previous is not None and (previous.start, previous.end) == (
        start,
        end,
    )
    result.metrics = {
        key: _merge_metric(
            metric, old_metrics.get(key), start, end, same_window=same_window
        )
        for key, metric in current.metrics.items()
    }
    for key, old in old_metrics.items():
        if key not in result.metrics:
            result.metrics[key] = _merge_metric(
                old.model_copy(update={'summaries': []}),
                old,
                start,
                end,
                same_window=same_window,
            )
    return result


def _capture_incomplete(response):
    if getattr(response, '_archive_empty_population', False):
        return False
    return response.reason == 'prometheus_unavailable' or any(
        metric.status == 'query_error'
        or metric.reason == 'historical_association_changed'
        for metric in response.metrics.values()
    )


def _scope_has_data(response):
    return any(_has_observation(metric) for metric in response.metrics.values())


def _expected_samples(start, end, step):
    if start is None or end is None:
        return 0
    interval = _evaluation_range(start, end, start, step)
    if interval is None:
        return 0
    span_ms = round(interval.end * 1000) - round(interval.start * 1000)
    return span_ms // (step * 1000) + 1


def _data_partial(response):
    if getattr(response, '_archive_empty_population', False):
        return False
    expected = _expected_samples(response.start, response.end, response.step)
    return any(
        metric.status in ('query_error', 'no_data', 'restricted')
        or metric.reason == 'insufficient_phase_rate_window'
        or bool(metric.coverage and metric.coverage.partial)
        or (
            metric.status in ('available', 'stale')
            and any(
                sum(v is not None for _, v in line.points)
                + len(line.lower_bounds or [])
                < expected
                for line in metric.series
            )
        )
        for metric in response.metrics.values()
    )


def _archive_outcome(scopes, errors, terminal):
    has_data = any(_scope_has_data(response) for response in scopes.values())
    coverage = (
        'none'
        if not has_data
        else (
            'partial'
            if errors or any(_data_partial(response) for response in scopes.values())
            else 'complete'
        )
    )
    if not terminal:
        return 'capturing', coverage, False
    if errors:
        return 'partial', coverage, False
    return ('ready' if has_data else 'no_data'), coverage, True


def _scope_model(key, data):
    model = BenchmarkModelMetrics if '/runtime/' in key else BenchmarkResourceMetrics
    return model.model_validate(data)


def _cutoff(history):
    return min(
        (e.at for e in history.context.events if e.kind == 'generation_changed'),
        default=None,
    )


def _mark_empty_roles(key, response, history):
    role = key.split('/')[-2]
    if role != 'all' and not any(i.role == role for i in history.instances):
        object.__setattr__(response, '_archive_empty_population', True)


def _refresh_scope(key, response, history):
    _mark_empty_roles(key, response, history)
    response.sample_statistics = {}
    collectors._statistics(response)
    for metric_key, statistics in response.sample_statistics.items():
        response.metrics[metric_key].summaries = [
            MetricSummary(statistic='current', dimensions=s.dimensions, value=s.last)
            for s in statistics
        ]
    response.available = _scope_has_data(response)
    if response.available and response.reason in (
        'no_samples',
        'prometheus_unavailable',
    ):
        response.reason = None
    elif not response.available and response.reason is None:
        response.reason = 'no_samples'
    if isinstance(response, BenchmarkResourceMetrics):
        device_ids = {
            line.dimensions.device_id
            for metric in response.metrics.values()
            for line in metric.series
            if any(v is not None for _, v in line.points)
        }
        for device in response.devices:
            if device.id in device_ids:
                device.history_available = True
                device.reason = None
            elif device.history_available:
                device.history_available = False
                device.reason = 'no_samples'
    return response


def frozen_history(benchmark, snapshot, results, *, now=None):
    """Build collector input solely from the execution snapshot, never current rows."""
    recorded = BenchmarkRunSnapshot.model_validate(snapshot['monitoring'])
    stages, window = collectors.benchmark_windows(results, mode=snapshot['load_mode'])
    if window is None and recorded.confirmed_at is not None:
        available = [s for s in stages if s.traffic_start is not None]
        start = min((s.traffic_start for s in available), default=recorded.confirmed_at)
        end = max(
            (s.traffic_end for s in available),
            default=(
                snapshot.get('observed_at', now or time.time())
                if _value(benchmark.state) in _TERMINAL
                else now or time.time()
            ),
        )
        if end > start:
            window = BenchmarkMonitoringWindow(start=start, end=end)
    if window is not None and _value(benchmark.state) not in _TERMINAL:
        window.end = max(window.end, now or time.time())
    if window is not None and _value(benchmark.state) in (
        'stopped',
        'error',
        'unreachable',
    ):
        window.end = max(window.end, snapshot.get('observed_at', window.end))
    workers = {w.id: w for w in recorded.workers}
    context = BenchmarkMonitoringContext(
        benchmark_id=benchmark.id,
        model_id=recorded.model_id,
        available=recorded.confirmed_at is not None and window is not None,
        reason=None if window is not None else 'run_timing_unavailable',
        run_window=window,
        stages=stages,
        instances=[
            BenchmarkMonitoringMember.model_validate(m.model_dump())
            for m in recorded.instances
        ],
        workers=[ResourceWorker(id=w.id, name=w.name) for w in recorded.workers],
        devices=[
            collectors._public_device(d, workers)
            for d in recorded.devices
            if d.worker_id in workers
        ],
        events=recorded.events,
        resource_status='available' if workers else 'unavailable',
        resource_reason=None if workers else 'historical_association_unavailable',
        capabilities=ModelMonitoringCapabilities(
            model_id=recorded.model_id, min_query_seconds=1, supports_custom_range=False
        ),
    )
    return collectors.BenchmarkHistory(
        context=context,
        snapshot=recorded,
        model=SimpleNamespace(
            id=recorded.model_id,
            cluster_id=recorded.cluster_id,
            backend=recorded.backend,
            roles=recorded.roles,
            disaggregation=recorded.disaggregation,
        ),
        instances=recorded.instances,
        workers=workers,
        devices=recorded.devices,
        target_mode=snapshot['target_mode'],
        gateway_url='captured' if snapshot['via_gateway'] else None,
    )


class BenchmarkMonitoringArchiver:
    """Collect enrolled runs under time-limited database leases."""

    def __init__(self, config):
        self.config = config
        self.worker_id = str(uuid4())
        self.store = BenchmarkPayloadStore(config, session_factory=async_session)
        self.flush_interval = getattr(
            config, 'benchmark_archive_flush_interval_seconds', 60
        )
        self.settle = getattr(config, 'benchmark_archive_settle_seconds', 30)
        self.max_bytes = getattr(
            config, 'benchmark_archive_max_run_bytes', 256 * 1024 * 1024
        )
        self.concurrency = getattr(config, 'benchmark_archive_max_concurrency', 2)
        self.retention_days = getattr(config, 'benchmark_archive_retention_days', None)

    async def run(self):
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(
                    'Benchmark monitoring archive scheduler failed (%s)',
                    type(exc).__name__,
                )
            await asyncio.sleep(min(self.flush_interval, _POLL_SECONDS))

    async def tick(self):
        await self._retire_payloads()
        row = BenchmarkMonitoringPayload
        async with async_session() as session:
            now = await database_now(session)
            due = (
                await session.exec(
                    select(row.benchmark_id)
                    .where(
                        row.retry_at <= now,
                        row.state != 'expired',
                        or_(row.expires_at.is_(None), row.expires_at > now),
                        or_(row.sealed.is_(False), row.repair_requested.is_(True)),
                        or_(row.lease_owner.is_(None), row.lease_until <= now),
                    )
                    .order_by(row.retry_at, row.benchmark_id)
                    .limit(self.concurrency)
                )
            ).all()
        async with asyncio.TaskGroup() as tasks:
            for benchmark_id in due:
                tasks.create_task(self._execute(benchmark_id))

    async def _retire_payloads(self):
        row = BenchmarkMonitoringPayload
        async with async_session() as session:
            now = await database_now(session)
            overdue = (
                await session.exec(
                    select(row.benchmark_id)
                    .where(row.expires_at <= now, row.state != 'expired')
                    .order_by(row.expires_at)
                    .limit(100)
                )
            ).all()
            orphans = (
                await session.exec(
                    select(row.benchmark_id)
                    .outerjoin(Benchmark, Benchmark.id == row.benchmark_id)
                    .where(
                        or_(Benchmark.id.is_(None), Benchmark.deleted_at.is_not(None))
                    )
                    .order_by(row.benchmark_id)
                    .limit(100)
                )
            ).all()
        for benchmark_id in overdue:
            await self.store.expire(benchmark_id)
        for benchmark_id in orphans:
            async with async_session() as session:
                await self.store.delete(session, benchmark_id)
                await session.commit()

    async def _execute(self, benchmark_id):
        try:
            lease = await self.store.claim(
                benchmark_id, self.worker_id, ttl=_LEASE_SECONDS
            )
        except Exception as exc:
            logger.warning(
                'Benchmark monitoring archive claim failed for %s (%s)',
                benchmark_id,
                type(exc).__name__,
            )
            return
        if lease is None:
            return
        heartbeat = asyncio.create_task(self._heartbeat(lease))
        capture = asyncio.create_task(self._capture(lease))
        try:
            done, _ = await asyncio.wait(
                (heartbeat, capture), return_when=asyncio.FIRST_COMPLETED
            )
            if capture in done:
                await capture
            else:
                await heartbeat
        except ArchiveLeaseLost:
            pass
        except Exception as exc:
            reason = (
                exc.reason
                if isinstance(exc, BenchmarkPayloadError)
                else 'capture_failed'
            )
            logger.warning(
                'Benchmark monitoring capture failed for %s (%s)',
                benchmark_id,
                reason,
            )
            try:
                retry_seconds = (
                    None
                    if reason == 'archive_scope_budget_exceeded'
                    else self.flush_interval
                )
                await self.store.fail(lease, reason, retry_seconds=retry_seconds)
            except Exception as exc:
                logger.warning(
                    'Benchmark monitoring failure could not be recorded for %s (%s)',
                    benchmark_id,
                    type(exc).__name__,
                )
        finally:
            for task in (heartbeat, capture):
                task.cancel()
            await asyncio.gather(heartbeat, capture, return_exceptions=True)
            try:
                await self.store.release(lease)
            except Exception as exc:
                logger.warning(
                    'Benchmark monitoring lease release failed for %s (%s)',
                    benchmark_id,
                    type(exc).__name__,
                )

    async def _heartbeat(self, lease):
        while True:
            await asyncio.sleep(_LEASE_SECONDS / 3)
            if not await self.store.renew(lease, ttl=_LEASE_SECONDS):
                raise ArchiveLeaseLost()

    async def _load_capture(self, lease):
        row = BenchmarkMonitoringPayload
        async with async_session() as session:
            record = (
                await session.exec(
                    select(row)
                    .options(undefer(row.payload))
                    .where(
                        row.benchmark_id == lease.benchmark_id,
                        row.run_id == lease.run_id,
                        row.lease_owner == lease.owner,
                        row.fencing_token == lease.token,
                        row.write_seq == lease.write_seq,
                        row.snapshot_seq == lease.snapshot_seq,
                    )
                )
            ).first()
            now = await database_now(session)
            if record is None or record.lease_until <= now:
                raise ArchiveLeaseLost()
        return record, now

    async def _capture(self, lease):
        record, now = await self._load_capture(lease)
        snapshot = record.snapshot
        terminal = lease.finalize_requested_at is not None or lease.repair_requested
        settle_remaining = (
            max(0, lease.finalize_requested_at + self.settle - now)
            if lease.finalize_requested_at is not None and not lease.repair_requested
            else 0
        )
        if settle_remaining:
            await self.store.release(lease, retry_seconds=math.ceil(settle_remaining))
            return
        if not (snapshot.get('monitoring') or {}).get('confirmed_at') and not terminal:
            await self.store.release(lease, retry_seconds=self.flush_interval)
            return
        corrupt = False
        try:
            previous = await self._read_previous(record)
        except BenchmarkPayloadError as exc:
            if not lease.repair_requested or exc.reason != 'archive_payload_corrupt':
                raise
            previous, corrupt = {}, True
        bundle, until, state, coverage, complete = await self._prepare(
            record, previous, now, terminal, lease.repair_requested
        )
        if corrupt and coverage == 'none':
            raise BenchmarkPayloadError('archive_payload_corrupt')
        if not terminal and until is None:
            await self.store.release(lease, retry_seconds=self.flush_interval)
            return
        encoded = await asyncio.to_thread(
            encode_payload, bundle, max_bytes=self.max_bytes
        )
        expires_at = (
            now + self.retention_days * 86400
            if terminal and self.retention_days and record.expires_at is None
            else None
        )
        published = await self.store.publish(
            lease,
            encoded,
            captured_until=until,
            state=state,
            data_coverage=coverage,
            capture_complete=complete,
            sealed=terminal,
            expires_at=expires_at,
            next_capture_seconds=self.flush_interval,
        )
        if not published:
            raise ArchiveLeaseLost()

    async def _read_previous(self, record):
        if record.payload is None:
            return {}
        return await asyncio.to_thread(
            decode_payload,
            record.payload,
            sha256=record.payload_sha256,
            uncompressed_bytes=record.uncompressed_bytes,
            max_bytes=self.max_bytes,
        )

    async def _prepare(self, record, previous, now, terminal, repair):
        snapshot = record.snapshot
        benchmark = SimpleNamespace(id=record.benchmark_id, state=snapshot.get('state'))
        history = frozen_history(
            benchmark,
            snapshot,
            [SimpleNamespace(**result) for result in snapshot.get('results', [])],
            now=now - self.settle,
        )
        plan = record.plan
        if 9 * (1 + len(history.context.stages)) > _MAX_SCOPES:
            raise BenchmarkPayloadError('archive_scope_budget_exceeded')
        async with aiohttp.ClientSession() as client:
            scopes, errors = await self._capture_phases(
                history, plan, previous, now, terminal, repair, client
            )
            run, run_errors, until = await self._capture_run_window(
                history, plan, record, previous, now, terminal, repair, client
            )
        scopes.update(run)
        errors.update(run_errors)
        if _cutoff(history) is not None:
            errors.add('historical_association_changed')
        state, coverage, complete = _archive_outcome(scopes, errors, terminal)
        return (
            {
                'context': history.context.model_dump(mode='json'),
                'scopes': {
                    key: value.model_dump(mode='json') for key, value in scopes.items()
                },
                'scope_errors': sorted(errors),
            },
            until,
            state,
            coverage,
            complete,
        )

    async def _capture_phases(
        self, history, plan, previous, now, terminal, repair, client
    ):
        scopes, errors = {}, set()
        for phase in history.context.stages:
            if phase.status != 'available':
                if terminal:
                    errors.add(f'phase:{phase.id}')
                continue
            if phase.end > now - self.settle and not terminal:
                continue
            prefix = f'phase:{phase.id}'
            saved = {
                key: value
                for key, value in previous.get('scopes', {}).items()
                if key.startswith(prefix + '/')
                and (value['start'], value['end']) == (phase.start, phase.end)
            }
            retrying = repair or any(
                key.startswith(prefix + '/') for key in previous.get('scope_errors', ())
            )
            outputs = (
                await self._collect_scope(
                    history, prefix, phase.id, plan, client=client
                )
                if not saved or retrying
                else {key: _scope_model(key, value) for key, value in saved.items()}
            )
            for key, response in outputs.items():
                if _capture_incomplete(response):
                    errors.add(key)
                old = _scope_model(key, saved[key]) if key in saved else None
                scopes[key] = self._finish_scope(
                    key, _merge_scope(response, old, phase.start, phase.end), history
                )
        return scopes, errors

    async def _capture_run_window(
        self, history, plan, record, previous, now, terminal, repair, client
    ):
        window = history.context.run_window
        if window is None:
            errors = {'run'} if terminal and history.context.stages else set()
            return {}, errors, None
        end = window.end if terminal else min(window.end, now - self.settle)
        if end - window.start < 0.001:
            return {}, set(), None
        window.end = end
        last_context = previous.get('context', {}).get('run_window') or {}
        same_start = last_context.get('start') == window.start
        cursor = record.captured_until if same_start else window.start
        prior_errors = {
            key for key in previous.get('scope_errors', ()) if key.startswith('run/')
        }
        requery = repair or terminal and prior_errors
        if requery or cursor is None or cursor > end:
            cursor = window.start
        scopes, errors = await self._capture_run(
            history,
            plan,
            previous.get('scopes', {}),
            start=window.start,
            end=end,
            cursor=cursor,
            client=client,
        )
        if not requery and cursor > window.start:
            errors.update(prior_errors)
        return scopes, errors, end

    async def _capture_run(self, history, plan, saved, *, start, end, cursor, client):
        stored = {
            key: _scope_model(key, value)
            for key, value in saved.items()
            if key.startswith('run/')
        }
        scopes = {}
        errors = set()
        plan = {**plan, 'anchor': start}
        # Extend the trend grid from the saved watermark. A shifted authoritative
        # run interval or a repair queries the whole interval instead.
        while Decimal(str(end)) - Decimal(str(cursor)) >= Decimal('.001'):
            stop = min(end, cursor + _MAX_RANGE_POINTS * plan['step'])
            outputs = await self._collect_scope(
                history, 'run', None, plan, start=cursor, end=stop, client=client
            )
            for key, response in outputs.items():
                if _capture_incomplete(response):
                    errors.add(key)
                for metric in response.metrics.values():
                    metric.summaries = []
                scopes[key] = _merge_scope(response, scopes.get(key), start, stop)
            cursor = stop
        # Capture the full result before merging stored points. A recoverable
        # gap in one query interval must not discard points in any other.
        for key, saved_response in stored.items():
            current = scopes.get(key, saved_response)
            scopes[key] = _merge_scope(current, saved_response, start, end)
        # Query native summaries over the exact whole interval. The summaries
        # returned by trend slices never represent the full run.
        native = await self._collect_scope(
            history,
            'run',
            None,
            plan,
            start=start,
            end=end,
            summary_only=True,
            client=client,
        )
        for key, response in native.items():
            if _capture_incomplete(response):
                errors.add(key)
            scopes[key] = _merge_scope(response, scopes.get(key), start, end)
        return (
            {
                key: self._finish_scope(key, value, history)
                for key, value in scopes.items()
            },
            errors,
        )

    @staticmethod
    def _finish_scope(key, response, history):
        cutoff = _cutoff(history)
        if cutoff is not None:
            BenchmarkMonitoringArchiver._isolate_generation(response, cutoff)
        return _refresh_scope(key, response, history)

    async def _collect_scope(
        self,
        history,
        scope,
        phase_id,
        plan,
        *,
        start=None,
        end=None,
        summary_only=False,
        client=None,
    ):
        window = (
            next(s for s in history.context.stages if s.id == phase_id)
            if phase_id is not None
            else history.context.run_window
        )
        start = start if start is not None else window.start
        end = end if end is not None else window.end
        anchor = plan.get('anchor', start)
        interval = _evaluation_range(start, end, anchor, plan['step'])
        summary_only = summary_only or interval is None
        options = dict(
            phase_id=phase_id,
            start=start,
            end=end,
            client=client,
            query_step=plan['step'],
            query_interval=interval,
            source_not_before=anchor,
        )
        outputs = {}
        for role in ('all', 'prefill', 'decode', 'router'):
            for view in ('aggregate', 'instances'):
                key = f'{scope}/runtime/{role}/{view}'
                response = await collectors.collect_benchmark_model_metrics(
                    history,
                    role=role,
                    view=view,
                    statistic='avg,p50,p95,p99',
                    section='all',
                    rate_window_seconds=plan['rate_window'],
                    summary_only=summary_only,
                    **options,
                )
                response.step = plan['step']
                response.rate_window_seconds = plan['rate_window']
                _mark_empty_roles(key, response, history)
                outputs[key] = response
        if not summary_only:
            outputs[
                f'{scope}/resources/all/aggregate'
            ] = await collectors.collect_benchmark_resource_metrics(history, **options)
        return outputs

    @staticmethod
    def _isolate_generation(response, cutoff):
        if response.end is None or response.end <= cutoff:
            return
        for metric in response.metrics.values():
            for line in metric.series:
                line.points = [(t, v if t < cutoff else None) for t, v in line.points]
                if line.lower_bounds:
                    line.lower_bounds = [
                        (t, v) for t, v in line.lower_bounds if t < cutoff
                    ]
            for summary in metric.summaries:
                summary.value, summary.lower_bound, summary.sample_count = (
                    None,
                    None,
                    None,
                )
            if metric.status not in ('unsupported', 'not_applicable', 'query_error'):
                metric.reason = 'historical_association_changed'
                metric.status = 'no_data'
        collectors._statistics(response)


def _evaluation_range(start, end, anchor, step):
    def milliseconds(value):
        return int(
            (Decimal(str(value)) * 1000).to_integral_value(rounding=ROUND_CEILING)
        )

    step_ms = step * 1000
    grid_start = milliseconds(anchor)
    first = (
        grid_start
        + (milliseconds(start) - grid_start + step_ms - 1) // step_ms * step_ms
    )
    last = milliseconds(end) - 1
    if first > last:
        return None
    return MonitoringInterval(first / 1000, last / 1000, math.ceil(end - start))
