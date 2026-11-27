"""Authorized engine-local model observations over a bounded history window."""

import asyncio
from collections import defaultdict
import logging
import math
import re
import time
from typing import List, Optional, Set, Tuple
from urllib.parse import urlsplit

import aiohttp

from gpustack.api.exceptions import BadRequestException

from gpustack.schemas.model_metrics import (
    MetricCoverage,
    MetricDimensions,
    MetricSeries,
    MetricSummary,
    ModelMetric,
    ModelMetrics,
    TrendStatistic,
)
from gpustack.schemas.models import Model, ModelInstance

from gpustack.utils.model_instance_workers import (
    model_instance_metrics_source_inventory,
)
from .catalog import model_metric_definitions
from .histogram_bounds import (
    QUANTILES,
    all_in_first_bucket,
    apply_lower_bounds,
    ceiling_scopes,
    first_bucket_ceiling,
    group_key,
    histogram_lower_bounds,
    is_first_bucket_estimate,
    requires_bound_check,
    trend_ceilings,
)
from .ingress import collect_ingress_metric
from . import config as query_config
from .queries import (
    HISTOGRAM_STATISTICS,
    PROFILE_LABELS,
    MetricQueryScope,
    build_counter_observations,
    build_counter_query,
    build_gauge_query,
    build_histogram_components,
    build_histogram_query,
    build_histogram_cdf_query,
    build_histogram_trend_components,
    build_ratio_query,
    cohort_identity,
    deduplicate_instant,
    histogram_statistics,
)
from .prometheus import (
    prometheus_url as _prometheus_url,
    query_instant as _query_instant,
    query_range as _query_range,
)
from .time_range import MonitoringInterval, monitoring_interval

logger = logging.getLogger(__name__)
_QUERY_ERRORS = (ValueError, aiohttp.ClientError, OSError, asyncio.TimeoutError)
_COLLECT_DEADLINE_SECONDS = 15.0
_HISTORY_SCOPE = 'current_instances_with_observed_source_history'


def parse_instance_ids(value: Optional[str]) -> Optional[Set[int]]:
    """Validate a bounded CSV population shared by monitoring endpoints."""
    if value is None:
        return None
    if not value or len(value) > 2048:
        raise BadRequestException(message="Invalid instance_ids")
    tokens = [token.strip() for token in value.split(',')]
    if any(not re.fullmatch(r'[1-9][0-9]{0,18}', token) for token in tokens):
        raise BadRequestException(message="instance_ids must contain positive integers")
    identities = {int(token) for token in tokens}
    if len(identities) > 32 or max(identities) > 2**63 - 1:
        raise BadRequestException(message="instance_ids accepts at most 32 valid IDs")
    return identities


def _finite(value) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _instant(entry) -> Optional[float]:
    value = entry.get('value') or []
    return _finite(value[1]) if len(value) == 2 else None


def _histogram_range_samples(rows):
    samples = defaultdict(lambda: defaultdict(list))
    for row in rows:
        labels = (row.get('metric') or {}).copy()
        part = labels.pop('histogram_component', None)
        if part is None:
            continue
        for timestamp, value in row.get('values') or ():
            at = _finite(timestamp)
            if at is not None:
                samples[at][part].append({'metric': labels, 'value': (at, value)})
    return samples


def _validate_result(entries, ranged):
    if not isinstance(entries, list):
        raise ValueError('Invalid metric result')
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get('metric', {}), dict):
            raise ValueError('Invalid metric series')
        if ranged:
            points = entry.get('values') or []
            if not isinstance(points, list) or any(
                not isinstance(point, (list, tuple)) or len(point) != 2
                for point in points
            ):
                raise ValueError('Invalid metric points')
        else:
            point = entry.get('value')
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError('Invalid metric point')
    return entries


def _upstream_address(value):
    value = re.sub(r'@\d+$', '', str(value).rstrip('/'))
    try:
        parsed = urlsplit(value if '://' in value else 'http://' + value)
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            return None
        if parsed.path not in ('', '/') or parsed.scheme not in ('http', 'https'):
            return None
        return (
            (parsed.hostname.lower(), parsed.port)
            if parsed.hostname and parsed.port
            else None
        )
    except ValueError:
        return None


def _engine_roles(model):
    declared = {
        getattr(item, 'name', None) for item in getattr(model, 'roles', None) or []
    }
    return {
        role
        for role in declared
        if isinstance(role, str) and re.fullmatch(r'[A-Za-z0-9_-]{1,64}', role)
    } or {''}


_PD_FIXED_ROLES = {
    'tpot': 'decode',
    'itl': 'decode',
    'decode': 'decode',
    'output_tokens': 'decode',
    'prefill': 'prefill',
    'input_tokens': 'prefill',
    'pd_prefill_bootstrap_queue': 'prefill',
    'pd_prefill_inflight_queue': 'prefill',
    'pd_decode_prealloc_queue': 'decode',
    'pd_decode_transfer_queue': 'decode',
    'pd_kv_allocation': 'decode',
    'router_health': 'router',
    'router_requests': 'router',
    'router_request_duration': 'router',
}
_PD_TRANSFER_KEYS = {
    'pd_kv_transfer',
    'pd_kv_transfer_post',
    'pd_kv_transfer_bytes',
    'pd_kv_transfer_speed',
    'pd_kv_completion',
    'pd_kv_throughput',
    'pd_kv_transfer_failures',
    'pd_kv_bootstrap_failures',
    'pd_kv_expired',
}
_PD_EVENT_KEYS = {
    'pd_kv_transfer_failures',
    'pd_kv_bootstrap_failures',
    'pd_kv_expired',
}
_SGLANG_PD_KEYS = {
    'pd_kv_allocation',
    'pd_prefill_bootstrap_queue',
    'pd_prefill_inflight_queue',
    'pd_decode_prealloc_queue',
    'pd_decode_transfer_queue',
    'pd_kv_transfer_speed',
}


def _transfer_selection(definition, model):
    from gpustack.server.pd_mode_catalog import get_pd_mode

    mode_name = model.disaggregation.mode
    mode = get_pd_mode(getattr(mode_name, 'value', mode_name))
    transfer = mode.transfer_metrics if mode else None
    if transfer is None:
        return set(), 'no_data', 'source_identity_unresolved'
    if transfer.connector == 'ascend-mooncake':
        return set(), 'unsupported', 'metric_not_provided'
    if definition.key == 'pd_kv_transfer_post' and transfer.connector != 'nixl':
        return set(), 'unsupported', 'metric_not_provided'
    sglang = transfer.connector.startswith('sglang-')
    if definition.key == 'pd_kv_expired':
        return (
            (set(), 'unsupported', 'metric_not_provided')
            if sglang
            else ({'prefill'}, None, None)
        )
    if definition.key == 'pd_kv_bootstrap_failures':
        return (
            ({'prefill', 'decode'}, None, None)
            if sglang
            else (set(), 'unsupported', 'metric_not_provided')
        )
    if definition.key == 'pd_kv_transfer_failures' and sglang:
        # Each side reports its own failures; retain role populations separately.
        return {'prefill', 'decode'}, None, None
    # SGLang's receiver records bootstrap/allocation unconditionally; only its
    # sender can report the optional transfer size and bandwidth histograms.
    sender_measurement = (
        definition.key
        in (
            'pd_kv_transfer_bytes',
            'pd_kv_transfer_speed',
            'pd_kv_completion',
            'pd_kv_throughput',
        )
        and sglang
    )
    selected = {'prefill' if sender_measurement else transfer.read_from_role}
    return selected, None, None


def _unsupported_pd_metric(definition, backend):
    key = definition.key
    if key in _SGLANG_PD_KEYS and backend != 'sglang':
        return True
    if key == 'router_health' and backend != 'sglang':
        return True
    return backend == 'sglang' and key in (
        'prefill',
        'decode',
        'external_cache_hit_rate',
    )


def _roles(definition, model, backend, role):
    if definition.measurement_scope == 'service_ingress':
        return (
            ((), None, None)
            if role == 'all'
            else ((), 'not_applicable', 'role_not_applicable')
        )
    pd = bool(getattr(model, 'disaggregation', None))
    if definition.section == 'pd' and not pd:
        return (), 'not_applicable', 'role_not_applicable'
    selected = {'prefill', 'decode'} if pd else _engine_roles(model)
    if pd and definition.key in _PD_FIXED_ROLES:
        selected = {_PD_FIXED_ROLES[definition.key]}
    if pd and definition.key == 'ttft' and backend == 'sglang':
        selected = {'decode'}
    if pd and definition.key in _PD_TRANSFER_KEYS:
        selected, status, reason = _transfer_selection(definition, model)
        if status:
            return (), status, reason
    if role != 'all':
        selected &= {role}
    if not selected:
        return (), 'not_applicable', 'role_not_applicable'
    if _unsupported_pd_metric(definition, backend):
        return (), 'unsupported', 'metric_not_provided'
    return tuple(sorted(selected)), None, None


def _statistics(definition):
    if definition.measurement_scope == 'service_ingress':
        return ['avg']
    if definition.kind == 'histogram':
        return list(HISTOGRAM_STATISTICS)
    return ['current' if definition.kind == 'gauge' else 'rate']


def _observation_roles(definition, model, backend, role, ingress_context):
    result = _roles(definition, model, backend, role)
    if definition.measurement_scope != 'service_ingress':
        return result
    absent = {
        'direct': ('not_applicable', 'gateway_bypassed'),
        'filtered': ('not_applicable', 'role_not_applicable'),
        'unknown': ('no_data', 'source_identity_unresolved'),
    }.get(ingress_context)
    return ((), *absent) if absent else result


class _Collection:
    def __init__(
        self,
        model,
        instances,
        client,
        url,
        start,
        end,
        step,
        rate_window,
        view,
        statistics,
        upstream_instances=None,
        allow_trend_lookback=False,
        query_interval=None,
        summary_only=False,
    ):
        self.model = model
        self.instances = {instance.id: instance for instance in instances}
        self.upstreams = {
            instance.id: instance
            for instance in (
                upstream_instances if upstream_instances is not None else instances
            )
        }
        self.client, self.url = client, url
        self.start, self.end, self.step = start, end, step
        self.window = end - start
        self.rate_window = rate_window
        self.view = view
        self.statistics = [s for s in HISTOGRAM_STATISTICS if s in statistics]
        self.allow_trend_lookback = allow_trend_lookback
        self.query_interval = query_interval
        self.summary_only = summary_only
        self.slots = asyncio.Semaphore(8)
        self.completed = set()
        self.reads = {}
        self.read_waiters = defaultdict(int)
        self.pending_reads = set()

    async def read(self, query, *, ranged=False):
        # Completion and legacy connector views can share an identical native
        # query. Coalesce concurrent reads within this authorized collection only.
        key = (query, ranged)
        if key not in self.reads:
            self.reads[key] = asyncio.create_task(
                self.read_native(query, ranged=ranged)
            )
        task = self.reads[key]
        waiter = asyncio.current_task()
        self.pending_reads.add(waiter)
        self.read_waiters[task] += 1
        try:
            # A metric can abandon its read while another metric still needs it.
            return await asyncio.shield(task)
        finally:
            try:
                self.read_waiters[task] -= 1
                if not self.read_waiters[task]:
                    del self.read_waiters[task]
                    if not task.done():
                        if self.reads.get(key) is task:
                            del self.reads[key]
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                if task.done() and (task.cancelled() or task.exception() is not None):
                    if self.reads.get(key) is task:
                        del self.reads[key]
            finally:
                self.pending_reads.discard(waiter)

    async def close(self, fills):
        """Drain collection-owned tasks before releasing their HTTP client."""
        # Stop metric producers first, then reads abandoned by failed gathers.
        for tasks in (fills, tuple(self.pending_reads), tuple(self.reads.values())):
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def read_native(self, query, *, ranged=False):
        if ranged and self.summary_only:
            return []
        async with self.slots:
            if ranged:
                start, end = (
                    (self.query_interval.start, self.query_interval.end)
                    if self.query_interval is not None
                    else (self.start, self.end)
                )
                entries = await _query_range(
                    self.client, self.url, query, start, end, self.step
                )
                return _validate_result(entries, True)
            entries = await _query_instant(self.client, self.url, query, self.end)
            return deduplicate_instant(_validate_result(entries, False))

    def instance_id(self, labels):
        try:
            identity = int(labels.get('model_instance_id', ''))
        except (TypeError, ValueError):
            return None
        return identity if identity in self.instances else None

    def dimensions(self, labels, *, instances=False, source=False):
        instance_id = self.instance_id(labels)
        fields = {'role': labels.get('role') or None}
        if (instances or source) and instance_id is not None:
            instance = self.instances[instance_id]
            fields.update(
                model_instance_id=instance_id, model_instance_name=instance.name
            )
        if source:
            fields.update(
                source_name=labels.get('source_name'),
                endpoint_dp_rank=labels.get('endpoint_dp_rank'),
                dp_rank=labels.get('dp_rank') or labels.get('engine'),
            )
            try:
                fields['worker_id'] = int(labels['worker_id'])
            except (KeyError, TypeError, ValueError):
                pass
        return MetricDimensions(**fields)

    @staticmethod
    def name(dimensions):
        return (
            ' · '.join(
                str(value)
                for value in (
                    dimensions.role,
                    dimensions.model_instance_name,
                    dimensions.source_name,
                    (
                        f'DP {dimensions.dp_rank}'
                        if dimensions.dp_rank is not None
                        else None
                    ),
                )
                if value
            )
            or 'Deployment'
        )

    def series(self, entries, statistic, *, scale=1, source=False):
        series = []
        for entry in entries:
            dimensions = self.dimensions(
                entry.get('metric') or {},
                instances=self.view == 'instances',
                source=source,
            )
            points = []
            for timestamp, value in entry.get('values') or []:
                timestamp, value = _finite(timestamp), _finite(value)
                if timestamp is not None:
                    points.append(
                        (
                            timestamp,
                            _finite(value * scale) if value is not None else None,
                        )
                    )
            if points:
                series.append(
                    MetricSeries(
                        name=self.name(dimensions),
                        statistic=statistic,
                        dimensions=dimensions,
                        points=points,
                    )
                )
        return series

    def coverage(self, entries, scope):
        observed = {
            self.instance_id(row.get('metric') or {})
            for row in entries
            if _instant(row) is not None
        }
        observed.discard(None)
        expected = len(set(scope.instance_ids))
        return MetricCoverage(
            expected_instances=expected,
            observed_instances=len(observed),
            partial=len(observed) < expected,
        )

    def grouped_values(self, entries):
        values = defaultdict(list)
        for entry in entries:
            labels = entry.get('metric') or {}
            value = _instant(entry)
            if self.instance_id(labels) is not None and value is not None:
                values[labels.get('role', '')].append(value)
        return values

    async def numeric(self, definition, metric, scope):
        if definition.denominator:
            return await self.ratio(definition, metric, scope)
        gauge = definition.kind == 'gauge'
        grouping = (
            PROFILE_LABELS if definition.per_source else ('role', 'model_instance_id')
        )
        if gauge:
            summary = build_gauge_query(
                definition.family,
                scope,
                group_by=grouping,
                profile=definition.query_profile,
            )
            trend = build_gauge_query(
                definition.family,
                scope,
                group_by=grouping if definition.per_source else None,
                profile=definition.query_profile,
            )
        else:
            summary = build_counter_query(
                definition.family,
                scope,
                self.window,
                summary=True,
                group_by=grouping,
                profile=definition.query_profile,
            )
            trend = build_counter_query(
                definition.family,
                scope,
                self.rate_window,
                profile=definition.query_profile,
            )
        rows, ranges = await asyncio.gather(
            self.read(summary), self.read(trend, ranged=True)
        )
        metric.coverage = self.coverage(rows, scope)
        metric.series = self.series(
            ranges,
            _statistics(definition)[0],
            scale=definition.scale,
            source=definition.per_source,
        )
        if definition.key in _PD_EVENT_KEYS:
            metric.summaries = self.event_summaries(rows, definition.scale)
        elif definition.per_source:
            for row in rows:
                value = _instant(row)
                if (
                    self.instance_id(row.get('metric') or {}) is not None
                    and value is not None
                ):
                    metric.summaries.append(
                        MetricSummary(
                            statistic='current',
                            dimensions=self.dimensions(row['metric'], source=True),
                            value=_finite(value * definition.scale),
                        )
                    )
        else:
            metric.summaries = [
                MetricSummary(
                    statistic=_statistics(definition)[0],
                    dimensions=MetricDimensions(role=role or None),
                    value=_finite(sum(values) * definition.scale),
                )
                for role, values in sorted(self.grouped_values(rows).items())
            ]
        self.numeric_status(metric)

    def event_summaries(self, rows, scale):
        """Retain per-role event rates and full-interval estimated event counts."""
        grouped = {}
        for row in rows:
            labels, value = row.get('metric') or {}, _instant(row)
            if self.instance_id(labels) is None or value is None or value < 0:
                continue
            dimensions = self.dimensions(labels, instances=self.view == 'instances')
            key = (dimensions.role, dimensions.model_instance_id)
            if key not in grouped:
                grouped[key] = [dimensions, 0.0]
            grouped[key][1] += value * scale
        return [
            MetricSummary(
                statistic='rate',
                dimensions=dimensions,
                value=_finite(rate),
                sample_count=_finite(rate * self.window),
            )
            for dimensions, rate in grouped.values()
        ]

    @staticmethod
    def numeric_status(metric):
        if any(
            summary.value is not None or summary.lower_bound is not None
            for summary in metric.summaries
        ):
            metric.status, metric.reason = 'available', None
        elif any(
            value is not None for line in metric.series for _, value in line.points
        ) or any(line.lower_bounds for line in metric.series):
            metric.status, metric.reason = 'stale', 'no_samples'

    def ratio_groups(self, numerator, denominator):
        populations = []
        for rows in (numerator, denominator):
            cohorts = {}
            for row in rows:
                labels, value = row.get('metric') or {}, _instant(row)
                if (
                    self.instance_id(labels) is not None
                    and value is not None
                    and value >= 0
                ):
                    cohorts[cohort_identity(labels)] = (labels, value, row)
            populations.append(cohorts)
        numerators, denominators = populations
        grouped, complete_rows, incomplete_ids = {}, [], set()
        for identity in numerators.keys() | denominators.keys():
            hit, query = numerators.get(identity), denominators.get(identity)
            labels = (hit or query)[0]
            group = grouped.setdefault(labels.get('role', ''), [0.0, 0.0, False])
            if query is not None:
                group[1] += query[1]
            if hit is None or query is None:
                group[2] = True
                incomplete_ids.add(self.instance_id(labels))
            else:
                group[0] += hit[1]
                complete_rows.append(query[2])
        return grouped, complete_rows, incomplete_ids

    async def ratio(self, definition, metric, scope):
        queries = [
            build_counter_observations(
                family,
                scope,
                self.window,
                summary=True,
                profile=definition.query_profile,
            )
            for family in (definition.family, definition.denominator)
        ]
        numerator, denominator = await asyncio.gather(*(self.read(q) for q in queries))
        grouped, complete_rows, incomplete_ids = self.ratio_groups(
            numerator, denominator
        )
        metric.coverage = self.coverage(
            [
                row
                for row in complete_rows
                if self.instance_id(row['metric']) not in incomplete_ids
            ],
            scope,
        )
        partial = any(group[2] for group in grouped.values())
        metric.coverage.partial |= partial
        for role, (hits, total, incomplete) in sorted(grouped.items()):
            value = (
                hits / total * definition.scale if total and not incomplete else None
            )
            metric.summaries.append(
                MetricSummary(
                    statistic='rate',
                    dimensions=MetricDimensions(role=role or None),
                    value=_finite(value),
                    sample_count=_finite(total),
                )
            )
        ranges = await self.read(
            build_ratio_query(
                definition.family,
                definition.denominator,
                scope,
                self.rate_window,
                profile=definition.query_profile,
            ),
            ranged=True,
        )
        metric.series = self.series(ranges, 'rate', scale=definition.scale)
        self.numeric_status(metric)
        if partial:
            metric.reason = 'partial_source_observations'
        if (
            denominator
            and not partial
            and all(s.sample_count == 0 for s in metric.summaries)
        ):
            metric.status, metric.reason = 'no_observations', 'zero_observations'

    async def histogram_components(self, definition, scope, seconds):
        queries = build_histogram_components(
            definition.family, scope, seconds, profile=definition.query_profile
        )
        sums, counts = await asyncio.gather(
            self.read(queries['sum']), self.read(queries['count'])
        )
        bucket_failed = False
        try:
            buckets = await self.read(queries['buckets'])
        except _QUERY_ERRORS:
            buckets, bucket_failed = [], True
        return sums, counts, buckets, bucket_failed

    async def histogram(self, definition, metric, scope):
        sums, counts, buckets, bucket_failed = await self.histogram_components(
            definition, scope, self.window
        )
        metric.coverage = self.coverage(counts, scope)
        groups = self.histogram_groups(sums, counts, buckets)
        incompatible = False
        if groups:
            metric.reason = None
            incompatible = self.histogram_summaries(
                metric, groups, exact_zero=definition.key == 'tpot'
            )
            metric.status = (
                'available'
                if any(group['count'] > 0 for group in groups.values())
                else 'no_observations'
            )
            if metric.status == 'no_observations':
                metric.reason = 'zero_observations'
        context = self.allow_trend_lookback and self.window < self.rate_window
        if context:
            # Trend cohorts describe the rolling engine context. Their samples
            # cannot supply the selected interval's summaries or coverage.
            sums, counts, buckets, bucket_failed = await self.histogram_components(
                definition, scope, self.rate_window
            )
            incompatible = any(
                group['count'] > 0 and not self.histogram_group_statistics(group)[1]
                for group in self.histogram_groups(sums, counts, buckets).values()
            )
        elif not groups:
            return
        await self.histogram_trends(
            definition,
            metric,
            scope,
            sums,
            counts,
            buckets,
            incompatible,
            bucket_failed,
        )
        if context:
            reason = metric.reason
            self.numeric_status(metric)
            if incompatible or reason in (
                'histogram_quantile_out_of_range',
                'histogram_first_bucket_unresolved',
                'histogram_bounds_unavailable',
                'query_failed',
            ):
                metric.reason = reason

    async def histogram_trends(
        self,
        definition,
        metric,
        scope,
        sums,
        counts,
        buckets,
        incompatible,
        bucket_failed,
    ):
        trend_scopes = self.histogram_trend_scopes(scope, sums, counts, buckets)
        if incompatible:
            if not trend_scopes and not any(
                summary.value is not None and summary.statistic != 'avg'
                for summary in metric.summaries
            ):
                metric.supported_statistics = ['avg']
            metric.reason = (
                'query_failed' if bucket_failed else 'incompatible_histogram_buckets'
            )
        statistics = [s for s in self.statistics if s in metric.supported_statistics]
        requests = [
            (stat, candidate)
            for stat in statistics
            for candidate in ([scope] if stat == 'avg' else trend_scopes)
        ]
        ranges = await asyncio.gather(
            *(
                self.read(
                    build_histogram_query(
                        definition.family,
                        candidate,
                        stat,
                        self.rate_window,
                        profile=definition.query_profile,
                    ),
                    ranged=True,
                )
                for stat, candidate in requests
            )
        )
        for (stat, _), entries in zip(requests, ranges):
            metric.series.extend(self.series(entries, stat))
        await self.histogram_trend_bounds(
            definition, metric, sums, counts, buckets, trend_scopes
        )

    def histogram_first_bucket_points(self, rows, *, exact_zero):
        points = {}
        for timestamp, parts in _histogram_range_samples(rows).items():
            groups = self.histogram_groups(
                parts['sum'],
                parts['count'],
                parts['buckets'],
                by_instance=self.view == 'instances',
                validate_buckets=True,
            )
            for key, group in groups.items():
                sample = key, timestamp
                if exact_zero and self.histogram_is_exact_zero(group):
                    points[sample] = 0.0
                elif self.histogram_group_statistics(group)[1]:
                    profiles = group['profiles']
                    points[sample] = (
                        first_bucket_ceiling(profiles)
                        if all_in_first_bucket(profiles, group['count'])
                        else None
                    )
        return points

    def apply_histogram_first_bucket_trends(
        self, metric, percentiles, rows, *, exact_zero, failed
    ):
        points = self.histogram_first_bucket_points(rows, exact_zero=exact_zero)
        censored, incomplete = False, False
        for line in percentiles:
            key = group_key(line.dimensions, self.view)
            quantile = QUANTILES[line.statistic]
            for index, (timestamp, value) in enumerate(line.points):
                sample = key, timestamp
                bound = points.get(sample)
                if bound == 0:
                    line.points[index] = timestamp, 0.0
                elif bound is not None and is_first_bucket_estimate(
                    value, bound, quantile
                ):
                    line.points[index] = timestamp, None
                    censored = True
                elif value is not None and sample not in points:
                    incomplete = True
        if incomplete:
            self.histogram_bounds_unavailable(metric, failed)
        elif censored:
            self.histogram_first_bucket_unresolved(metric)

    @staticmethod
    def histogram_bounds_unavailable(metric, failed):
        if failed:
            metric.reason = 'query_failed'
        elif metric.reason not in ('query_failed', 'incompatible_histogram_buckets'):
            metric.reason = 'histogram_bounds_unavailable'

    @staticmethod
    def histogram_first_bucket_unresolved(metric):
        if metric.reason in (
            None,
            'no_samples',
            'zero_observations',
            'histogram_quantile_out_of_range',
        ):
            metric.reason = 'histogram_first_bucket_unresolved'

    async def histogram_trend_bounds(
        self, definition, metric, sums, counts, buckets, candidates
    ):
        percentiles = [line for line in metric.series if line.statistic in QUANTILES]
        if not percentiles:
            return
        groups = self.histogram_groups(
            sums, counts, buckets, by_instance=self.view == 'instances'
        )
        ceilings = trend_ceilings(groups)
        check_tail = requires_bound_check(percentiles, ceilings, self.view)
        exact_zero = definition.key == 'tpot'
        rows, failed = [], False
        try:
            for candidate in candidates:
                query = build_histogram_trend_components(
                    definition.family,
                    candidate,
                    self.rate_window,
                    include_sum=exact_zero,
                    profile=definition.query_profile,
                )
                populations = (
                    ceiling_scopes(candidate, ceilings, buckets) if check_tail else []
                )
                try:
                    if populations:
                        query += ' or ' + build_histogram_cdf_query(
                            definition.family,
                            populations,
                            self.rate_window,
                            profile=definition.query_profile,
                        )
                    rows.extend(await self.read(query, ranged=True))
                except _QUERY_ERRORS:
                    failed = True
        finally:
            self.apply_histogram_first_bucket_trends(
                metric,
                percentiles,
                rows,
                exact_zero=exact_zero,
                failed=failed,
            )
            if check_tail:
                censored, incomplete = apply_lower_bounds(
                    percentiles, ceilings, rows, view=self.view
                )
                if incomplete:
                    self.histogram_bounds_unavailable(metric, failed)
                elif censored and metric.reason in (
                    None,
                    'no_samples',
                    'zero_observations',
                ):
                    metric.reason = 'histogram_quantile_out_of_range'

    def histogram_groups(
        self, sums, counts, buckets, *, by_instance=False, validate_buckets=False
    ):
        groups, cohort_roles = {}, {}
        for row in counts:
            labels, value = row.get('metric') or {}, _instant(row)
            if self.instance_id(labels) is None or value is None or value < 0:
                continue
            role, identity = labels.get('role', ''), cohort_identity(labels)
            key = (role, self.instance_id(labels)) if by_instance else role
            group = groups.setdefault(
                key, {'count': 0.0, 'sum': 0.0, 'profiles': {}, 'sum_profiles': {}}
            )
            group['count'] += value
            cohort_roles[identity] = key, value
            if value > 0:
                group['profiles'][identity] = {}
        for row in sums:
            identity = cohort_identity(row.get('metric') or {})
            value = _instant(row)
            if identity in cohort_roles and value is not None:
                key, _ = cohort_roles[identity]
                groups[key]['sum'] += value
                groups[key]['sum_profiles'][identity] = value
        self.histogram_bucket_profiles(
            groups, cohort_roles, buckets, by_instance, validate_buckets
        )
        return groups

    def histogram_bucket_profiles(
        self, groups, cohort_roles, buckets, by_instance, validate_buckets
    ):
        for row in buckets:
            labels, value = row.get('metric') or {}, _instant(row)
            identity = cohort_identity(labels, omit=('le',))
            if value is None:
                continue
            try:
                bound = float(labels['le'])
            except (KeyError, TypeError, ValueError):
                continue
            if identity not in cohort_roles:
                role = labels.get('role', '')
                key = (role, self.instance_id(labels)) if by_instance else role
                if validate_buckets and value != 0 and key in groups:
                    groups[key]['ambiguous_bounds'] = True
                continue
            key, count = cohort_roles[identity]
            group = groups[key]
            if validate_buckets and count == 0 and value != 0:
                group['ambiguous_bounds'] = True
            if count == 0:
                continue
            profile = group['profiles'][identity]
            if bound in profile or (
                validate_buckets
                and bound == math.inf
                and not math.isclose(value, count, rel_tol=1e-9)
            ):
                group['ambiguous_bounds'] = True
            else:
                profile[bound] = value

    @staticmethod
    def histogram_is_exact_zero(group):
        return (
            group['count'] > 0
            and group['sum'] == 0
            and not group.get('ambiguous_bounds')
            and all(
                group['sum_profiles'].get(source) == 0 for source in group['profiles']
            )
        )

    @staticmethod
    def histogram_group_statistics(group):
        total = (
            group['sum']
            if set(group['profiles']).issubset(group['sum_profiles'])
            else math.nan
        )
        stats, compatible = histogram_statistics(
            group['profiles'], count=group['count'], total=total
        )
        if group.get('ambiguous_bounds'):
            return {
                key: value if key == 'avg' else None for key, value in stats.items()
            }, False
        return stats, compatible

    def histogram_summaries(self, metric, groups, *, exact_zero=False):
        metric.summaries.clear()
        incompatible = False
        for role, group in sorted(groups.items()):
            stats, compatible = self.histogram_group_statistics(group)
            all_zero = exact_zero and self.histogram_is_exact_zero(group)
            incompatible |= group['count'] > 0 and not (compatible or all_zero)
            first_bucket_only = (
                compatible
                and not all_zero
                and all_in_first_bucket(group['profiles'], group['count'])
            )
            bounds = (
                histogram_lower_bounds(group['profiles'], count=group['count'])
                if compatible and not all_zero
                else {}
            )
            if first_bucket_only:
                self.histogram_first_bucket_unresolved(metric)
            elif bounds and metric.reason in (None, 'no_samples'):
                metric.reason = 'histogram_quantile_out_of_range'
            for stat in HISTOGRAM_STATISTICS:
                value = stats[stat]
                if stat != 'avg':
                    if all_zero:
                        value = 0.0
                    elif first_bucket_only or stat in bounds:
                        value = None
                metric.summaries.append(
                    MetricSummary(
                        statistic=stat,
                        dimensions=MetricDimensions(role=role or None),
                        value=_finite(value),
                        lower_bound=bounds.get(stat),
                        sample_count=group['count'],
                    )
                )
        return incompatible

    def histogram_trend_scopes(self, scope, sums, counts, buckets):
        groups = self.histogram_groups(
            sums, counts, buckets, by_instance=self.view == 'instances'
        )
        valid = [
            key
            for key, group in groups.items()
            if self.histogram_group_statistics(group)[1]
        ]
        if self.view == 'aggregate':
            return (
                [
                    MetricQueryScope(
                        scope.cluster_id,
                        scope.model_id,
                        scope.instance_ids,
                        tuple(sorted(valid)),
                        scope.view,
                        scope.launches,
                        scope.source_identities,
                        scope.sample_start,
                    )
                ]
                if valid
                else []
            )
        # Each role receives only instance groups whose native buckets can be merged.
        roles = defaultdict(list)
        for role, instance_id in valid:
            roles[role].append(instance_id)
        return [
            MetricQueryScope(
                scope.cluster_id,
                scope.model_id,
                tuple(ids),
                (role,),
                scope.view,
                scope.launches,
                scope.source_identities,
                scope.sample_start,
            )
            for role, ids in sorted(roles.items())
        ]

    def upstream_instance(self, worker):
        normalized = _upstream_address(worker)
        if normalized is None:
            return None
        matches = set()
        for instance in self.upstreams.values():
            if getattr(instance, 'role', None) == 'router':
                continue
            if normalized in self.instance_addresses(instance):
                matches.add(instance.id)
        return next(iter(matches)) if len(matches) == 1 else None

    @staticmethod
    def instance_addresses(instance):
        recorded = getattr(instance, 'source_addresses', None)
        if recorded is not None:
            return set(recorded)
        worker_id, ip = getattr(instance, 'worker_id', None), getattr(
            instance, 'worker_ip', None
        )
        ips = {worker_id: ip} if worker_id and ip else {}
        servers = getattr(instance, 'distributed_servers', None)
        for worker in getattr(servers, 'subordinate_workers', None) or []:
            if getattr(worker, 'worker_ip', None):
                ips[worker.worker_id] = worker.worker_ip
        ports = [(worker_id, getattr(instance, 'port', None))]
        try:
            ports.extend(
                (source.worker_id, source.api_port)
                for source in model_instance_metrics_source_inventory(instance)
            )
        except (AttributeError, TypeError, ValueError):
            pass
        return {
            (str(ips[owner]).lower(), port)
            for owner, port in ports
            if owner in ips and port
        }

    async def health(self, definition, metric, scope):
        grouping = ('role', 'model_instance_id', 'worker')
        query = build_gauge_query(
            definition.family,
            scope,
            aggregate='max',
            group_by=grouping,
            profile=definition.query_profile,
        )
        rows, ranges = await asyncio.gather(
            self.read(query), self.read(query, ranged=True)
        )
        unresolved = False
        for row in rows:
            labels = row.get('metric') or {}
            upstream_id = self.upstream_instance(labels.get('worker'))
            if upstream_id is None:
                unresolved = True
                continue
            dims = self.dimensions(labels, instances=True)
            dims.upstream_instance_id = upstream_id
            value = _instant(row)
            if value in (-1, 0, 1):
                metric.summaries.append(
                    MetricSummary(statistic='current', dimensions=dims, value=value)
                )
        for entry in ranges:
            upstream_id = self.upstream_instance(
                (entry.get('metric') or {}).get('worker')
            )
            if upstream_id is None:
                unresolved = True
                continue
            for line in self.series([entry], 'current'):
                producer_id = self.instance_id(entry.get('metric') or {})
                if producer_id is not None:
                    line.dimensions.model_instance_id = producer_id
                    line.dimensions.model_instance_name = self.instances[
                        producer_id
                    ].name
                line.dimensions.upstream_instance_id = upstream_id
                line.name = self.upstreams[upstream_id].name
                line.points = [
                    (t, v if v in (-1, 0, 1) else None) for t, v in line.points
                ]
                metric.series.append(line)
        self.numeric_status(metric)
        if unresolved:
            metric.reason = 'source_identity_unresolved'
        metric.coverage = self.coverage(rows, scope)

    async def fill(self, definition, metric, scope):
        try:
            if definition.measurement_scope == 'service_ingress':
                observed = await collect_ingress_metric(
                    definition,
                    model_id=self.model.id,
                    instance_ids=tuple(self.instances),
                    read=self.read,
                    window_seconds=self.window,
                    rate_window_seconds=self.rate_window,
                    statistics=self.statistics,
                )
                for field in ('status', 'reason', 'series', 'summaries'):
                    setattr(metric, field, getattr(observed, field))
            elif definition.key == 'router_health':
                await self.health(definition, metric, scope)
            elif definition.kind == 'histogram':
                await self.histogram(definition, metric, scope)
            else:
                await self.numeric(definition, metric, scope)
        except _QUERY_ERRORS:
            logger.warning(
                'Model %s metric %s query failed',
                self.model.id,
                definition.key,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
            metric.status, metric.reason = 'query_error', 'query_failed'
        self.completed.add(definition.key)


async def _wait_for_fills(fills):
    # Protected cleanup owns cancellation of fills and shared reads.
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _COLLECT_DEADLINE_SECONDS
    pending = set(fills)
    while pending:
        done, pending = await asyncio.wait(
            pending,
            timeout=max(0, deadline - loop.time()),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if not done:
            raise asyncio.TimeoutError
        for task in done:
            task.result()


async def _close_collection(collection, fills, client):
    async def close():
        try:
            if collection is not None:
                await collection.close(fills)
        finally:
            if client is not None:
                await client.close()

    # Keep ownership until all cleanup finishes, including after repeated request
    # cancellation. Shielding once would leave the cleanup running independently.
    cleanup = asyncio.create_task(close())
    cancelled = False
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            cancelled = True
    cleanup.result()
    if cancelled:
        raise asyncio.CancelledError


async def collect_model_metrics(
    model: Model,
    instances: List[ModelInstance],
    *,
    window_seconds: int,
    start: Optional[float] = None,
    end: Optional[float] = None,
    role: str = 'all',
    view: str = 'aggregate',
    statistics: Tuple[TrendStatistic, ...] = HISTOGRAM_STATISTICS,
    section: str = 'all',
    client: Optional[aiohttp.ClientSession] = None,
    upstream_instances: Optional[List[ModelInstance]] = None,
    interval: Optional[MonitoringInterval] = None,
    launches: tuple = (),
    source_identities: tuple = (),
    allow_trend_lookback: bool = False,
    query_step: Optional[int] = None,
    query_interval: Optional[MonitoringInterval] = None,
    query_rate_window_seconds: Optional[int] = None,
    summary_only: bool = False,
    source_not_before: Optional[float] = None,
    ingress_context: str = 'deployment',
) -> ModelMetrics:
    """Collect guarded runtime observations for an already authorized population.

    Args:
        model: Deployment whose ownership has been checked by the route.
        instances: Current visible instances after the user's filters.
        start: Optional fixed beginning in whole Unix seconds.
        end: Fixed end, supplied together with start; independent of refresh time.
        statistics: Latency trend statistics; interval summaries retain all four.
        upstream_instances: Full visible population for resolving Router peer IDs.
        allow_trend_lookback: Resolve short-interval histogram trends independently
            of summaries, retaining their normal rolling engine context.
        query_step: Internal evaluation step override; rate windows remain unchanged.
        query_interval: Internal range-query bounds; summaries retain the interval.
        ingress_context: Gateway/deployment observations, direct instance traffic,
            a role/instance-filtered query, or an unknown historical request path.

    Returns:
        Typed, bounded observations with local missing states for each metric.
    """
    interval = interval or monitoring_interval(
        window_seconds, start, end, now=time.time()
    )
    if ingress_context not in (
        'deployment',
        'gateway',
        'direct',
        'filtered',
        'unknown',
    ):
        raise ValueError('Invalid ingress observation context')
    start, end, window_seconds = interval.start, interval.end, interval.seconds
    time_policy = query_config.get_query_config().time_windows.llm
    step = time_policy.step(window_seconds)
    result = ModelMetrics(
        model_id=model.id,
        start=start,
        end=end,
        step=query_step if query_step is not None else step,
        rate_window_seconds=(
            query_rate_window_seconds
            if query_rate_window_seconds is not None
            else time_policy.rate_window(step)
        ),
        last_refreshed_at=end,
        history_scope=_HISTORY_SCOPE,
    )
    backend = str(getattr(model.backend, 'value', model.backend)).lower()
    url, owned = _prometheus_url(), client is None
    definitions = [
        definition
        for definition in model_metric_definitions(backend)
        if section == 'all' or definition.section == section
    ]
    fills, scheduled, fill_tasks = [], set(), []
    collection = None
    if owned and url:
        client = aiohttp.ClientSession()
    try:
        collection = _Collection(
            model,
            instances,
            client,
            url,
            result.start,
            end,
            result.step,
            result.rate_window_seconds,
            view,
            statistics,
            upstream_instances,
            allow_trend_lookback,
            query_interval,
            summary_only,
        )
        for definition in definitions:
            roles, status, reason = _observation_roles(
                definition, model, backend, role, ingress_context
            )
            gateway = definition.measurement_scope == 'service_ingress'
            metric = ModelMetric(
                kind=definition.kind,
                unit=definition.unit,
                measurement_scope=definition.measurement_scope,
                supported_statistics=(
                    []
                    if status in ('unsupported', 'not_applicable')
                    else _statistics(definition)
                ),
                status=status or 'no_data',
                reason=reason or 'no_samples',
            )
            if definition.kind == 'histogram' and not gateway:
                metric.summaries = [
                    MetricSummary(
                        statistic=stat,
                        dimensions=MetricDimensions(role=r or None),
                        value=None,
                    )
                    for r in roles
                    for stat in HISTOGRAM_STATISTICS
                ]
            result.metrics[definition.key] = metric
            selected = tuple(
                i.id
                for i in instances
                if gateway or (getattr(i, 'role', None) or '') in roles
            )
            if status or not selected or model.cluster_id is None:
                continue
            if not url:
                metric.reason = 'prometheus_unavailable'
                continue
            if gateway:
                fills.append((definition, metric, None))
                scheduled.add(definition.key)
                continue
            scope = MetricQueryScope(
                model.cluster_id,
                model.id,
                selected,
                roles,
                view,
                launches,
                source_identities,
                (
                    (
                        source_not_before
                        if source_not_before is not None
                        else interval.start
                    )
                    if source_identities
                    else None
                ),
            )
            fills.append((definition, metric, scope))
            scheduled.add(definition.key)
        try:
            if fills:
                fill_tasks = [
                    asyncio.create_task(collection.fill(*fill)) for fill in fills
                ]
                await _wait_for_fills(fill_tasks)
        except asyncio.TimeoutError:
            for key in scheduled - collection.completed:
                metric = result.metrics[key]
                metric.status, metric.reason = 'query_error', 'query_failed'
    finally:
        await _close_collection(collection, fill_tasks, client if owned else None)
    result.last_refreshed_at = time.time()
    result.available = bool(url) and any(
        m.status in ('available', 'no_data', 'no_observations', 'stale')
        for m in result.metrics.values()
    )
    if not url:
        result.reason = 'prometheus_unavailable'
    elif not result.available:
        result.reason = (
            'query_failed'
            if any(m.status == 'query_error' for m in result.metrics.values())
            else 'no_samples'
        )
    return result
