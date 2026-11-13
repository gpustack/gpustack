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


def _roles(definition, model, backend, role):
    pd = bool(getattr(model, 'disaggregation', None))
    if definition.key in ('pd_kv_transfer', 'router_health') and not pd:
        return (), 'not_applicable', 'role_not_applicable'
    selected = {'prefill', 'decode'} if pd else _engine_roles(model)
    if pd and definition.key in ('tpot', 'itl', 'output_tokens'):
        selected = {'decode'}
    if pd and definition.key == 'input_tokens':
        selected = {'prefill'}
    if pd and definition.key == 'ttft' and backend == 'sglang':
        selected = {'decode'}
    if definition.key == 'router_health':
        selected = {'router'}
    if definition.key == 'pd_kv_transfer':
        from gpustack.server.pd_mode_catalog import get_pd_mode

        mode_name = model.disaggregation.mode
        mode = get_pd_mode(getattr(mode_name, 'value', mode_name))
        transfer = mode.transfer_metrics if mode else None
        if transfer is None:
            return (), 'no_data', 'source_identity_unresolved'
        selected = {transfer.read_from_role}
        if getattr(mode_name, 'value', mode_name) == 'vllm-ascend-mooncake':
            return (), 'unsupported', 'metric_not_provided'
    if role != 'all':
        selected &= {role}
    if not selected:
        return (), 'not_applicable', 'role_not_applicable'
    unsupported = backend == 'sglang' and definition.key in (
        'prefill',
        'decode',
        'external_cache_hit_rate',
    )
    unsupported |= definition.key == 'router_health' and backend != 'sglang'
    if unsupported:
        return (), 'unsupported', 'metric_not_provided'
    return tuple(sorted(selected)), None, None


def _statistics(definition):
    if definition.kind == 'histogram':
        return list(HISTOGRAM_STATISTICS)
    return ['current' if definition.kind == 'gauge' else 'rate']


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
        self.slots = asyncio.Semaphore(8)
        self.completed = set()

    async def read(self, query, *, ranged=False):
        async with self.slots:
            if ranged:
                entries = await _query_range(
                    self.client, self.url, query, self.start, self.end, self.step
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
        if definition.per_source:
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

    @staticmethod
    def numeric_status(metric):
        if any(summary.value is not None for summary in metric.summaries):
            metric.status, metric.reason = 'available', None
        elif any(
            value is not None for line in metric.series for _, value in line.points
        ):
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

    async def histogram(self, definition, metric, scope):
        queries = build_histogram_components(
            definition.family, scope, self.window, profile=definition.query_profile
        )
        sums, counts = await asyncio.gather(
            self.read(queries['sum']), self.read(queries['count'])
        )
        bucket_failed = False
        try:
            buckets = await self.read(queries['buckets'])
        except _QUERY_ERRORS:
            buckets, bucket_failed = [], True
        metric.coverage = self.coverage(counts, scope)
        groups = self.histogram_groups(sums, counts, buckets)
        if not groups:
            return
        metric.reason = None
        incompatible = self.histogram_summaries(metric, groups)
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
        metric.status = (
            'available'
            if any(group['count'] > 0 for group in groups.values())
            else 'no_observations'
        )
        if metric.status == 'no_observations':
            metric.reason = 'zero_observations'
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

    def histogram_groups(self, sums, counts, buckets, *, by_instance=False):
        groups, cohort_roles = {}, {}
        for row in counts:
            labels, value = row.get('metric') or {}, _instant(row)
            if self.instance_id(labels) is None or value is None or value < 0:
                continue
            role, identity = labels.get('role', ''), cohort_identity(labels)
            key = (role, self.instance_id(labels)) if by_instance else role
            group = groups.setdefault(
                key, {'count': 0.0, 'sum': 0.0, 'profiles': {}, 'sum_profiles': set()}
            )
            group['count'] += value
            cohort_roles[identity] = key
            if value > 0:
                group['profiles'][identity] = {}
        for row in sums:
            identity = cohort_identity(row.get('metric') or {})
            value = _instant(row)
            if identity in cohort_roles and value is not None:
                groups[cohort_roles[identity]]['sum'] += value
                groups[cohort_roles[identity]]['sum_profiles'].add(identity)
        for row in buckets:
            labels, value = row.get('metric') or {}, _instant(row)
            identity = cohort_identity(labels, omit=('le',))
            if identity not in cohort_roles or value is None:
                continue
            try:
                bound = float(labels['le'])
            except (KeyError, TypeError, ValueError):
                continue
            profile = groups[cohort_roles[identity]]['profiles'].get(identity)
            if profile is not None:
                profile[bound] = value
        return groups

    @staticmethod
    def histogram_group_statistics(group):
        total = (
            group['sum']
            if set(group['profiles']).issubset(group['sum_profiles'])
            else math.nan
        )
        return histogram_statistics(
            group['profiles'], count=group['count'], total=total
        )

    def histogram_summaries(self, metric, groups):
        metric.summaries.clear()
        incompatible = False
        for role, group in sorted(groups.items()):
            stats, compatible = self.histogram_group_statistics(group)
            incompatible |= group['count'] > 0 and not compatible
            for stat in HISTOGRAM_STATISTICS:
                metric.summaries.append(
                    MetricSummary(
                        statistic=stat,
                        dimensions=MetricDimensions(role=role or None),
                        value=_finite(stats[stat]),
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
            if definition.key == 'router_health':
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
) -> ModelMetrics:
    """Collect guarded runtime observations for an already authorized population.

    Args:
        model: Deployment whose ownership has been checked by the route.
        instances: Current visible instances after the user's filters.
        start: Optional fixed beginning in whole Unix seconds.
        end: Fixed end, supplied together with start; independent of refresh time.
        statistics: Latency trend statistics; interval summaries retain all four.
        upstream_instances: Full visible population for resolving Router peer IDs.

    Returns:
        Typed, bounded observations with local missing states for each metric.
    """
    interval = interval or monitoring_interval(
        window_seconds, start, end, now=time.time()
    )
    start, end, window_seconds = interval.start, interval.end, interval.seconds
    time_policy = query_config.get_query_config().time_windows.llm
    step = time_policy.step(window_seconds)
    result = ModelMetrics(
        model_id=model.id,
        start=start,
        end=end,
        step=step,
        rate_window_seconds=time_policy.rate_window(step),
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
    fills, scheduled = [], set()
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
            step,
            result.rate_window_seconds,
            view,
            statistics,
            upstream_instances,
        )
        for definition in definitions:
            roles, status, reason = _roles(definition, model, backend, role)
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
            if definition.kind == 'histogram':
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
                i.id for i in instances if (getattr(i, 'role', None) or '') in roles
            )
            if status or not selected or model.cluster_id is None:
                continue
            if not url:
                metric.reason = 'prometheus_unavailable'
                continue
            scope = MetricQueryScope(
                model.cluster_id,
                model.id,
                selected,
                roles,
                view,
                launches,
                source_identities,
                interval.start if source_identities else None,
            )
            fills.append((definition, metric, scope))
            scheduled.add(definition.key)
        try:
            if fills:
                await asyncio.wait_for(
                    asyncio.gather(*(collection.fill(*fill) for fill in fills)),
                    _COLLECT_DEADLINE_SECONDS,
                )
        except asyncio.TimeoutError:
            for key in scheduled - collection.completed:
                metric = result.metrics[key]
                metric.status, metric.reason = 'query_error', 'query_failed'
    finally:
        if owned and client is not None:
            await client.close()
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
