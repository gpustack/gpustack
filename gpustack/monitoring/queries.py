"""Fixed model populations and classic-histogram statistics."""

from dataclasses import dataclass
import json
import math
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from . import config as query_config

SOURCE_LABELS = (
    'cluster_id',
    'model_id',
    'model_instance_id',
    'restart_count',
    'source_kind',
    'source_name',
    'worker_id',
    'endpoint_dp_rank',
    'component_attempt',
)
PROFILE_LABELS = SOURCE_LABELS + ('role', 'engine', 'dp_rank')
HISTOGRAM_STATISTICS = ('avg', 'p50', 'p95', 'p99')
PLUMBING_LABELS = (
    'job',
    'instance',
    'cluster_name',
    'model_name',
    'model_instance_name',
    'worker_name',
)


def cohort_identity(labels: Mapping[str, Any], *, omit: Sequence[str] = ()) -> tuple:
    return tuple(
        sorted(
            (key, str(value))
            for key, value in labels.items()
            if key not in PLUMBING_LABELS + ('__name__',) + tuple(omit)
        )
    )


def deduplicate_instant(entries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ignore duplicate scrape paths, retaining every native measurement cohort."""
    distinct = {}
    for entry in entries:
        try:
            value = float(entry['value'][1])
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        labels = entry.get('metric') or {}
        identity = cohort_identity(labels)
        previous = distinct.get(identity)
        if previous is None or value > float(previous['value'][1]):
            distinct[identity] = {
                **entry,
                'metric': {
                    key: val
                    for key, val in labels.items()
                    if key not in PLUMBING_LABELS
                },
            }
    return list(distinct.values())


def _metric_name(name: str) -> str:
    if not re.fullmatch(r'gpustack:[A-Za-z_][A-Za-z0-9_]*', name):
        raise ValueError('Invalid semantic metric')
    return name


@dataclass(frozen=True)
class MetricQueryScope:
    cluster_id: int
    model_id: int
    instance_ids: Tuple[int, ...]
    roles: Tuple[str, ...]
    view: str = 'aggregate'

    def __post_init__(self):
        ids = (self.cluster_id, self.model_id) + tuple(self.instance_ids)
        if not self.instance_ids or any(type(i) is not int or i <= 0 for i in ids):
            raise ValueError('Invalid model population')
        if not self.roles or any(
            not isinstance(r, str)
            or (r != '' and not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', r))
            for r in self.roles
        ):
            raise ValueError('Invalid model roles')
        if self.view not in ('aggregate', 'instances'):
            raise ValueError('Invalid metric view')

    @property
    def grouping(self) -> Tuple[str, ...]:
        return ('role', 'model_instance_id') if self.view == 'instances' else ('role',)

    def selector(self, extra: Optional[Mapping[str, str]] = None) -> str:
        labels = {
            'cluster_id': str(self.cluster_id),
            'model_id': str(self.model_id),
        }
        exact = [f'{key}={json.dumps(value)}' for key, value in labels.items()]
        instances = '|'.join(str(i) for i in sorted(set(self.instance_ids)))
        roles = '|'.join(self.roles)
        exact.extend(
            [
                f'model_instance_id=~"{instances}"',
                f'role=~"{roles}"',
                'source_kind!=""',
                'source_name!=""',
                'restart_count!=""',
                'worker_id!=""',
            ]
        )
        exact.extend(
            f'{key}={json.dumps(value)}' for key, value in (extra or {}).items()
        )
        return '{' + ','.join(exact) + '}'

    def guard(self, expression: str, history_seconds: Optional[int] = None) -> str:
        config = query_config.get_query_config()
        success = config.common(
            'engine_success', selector=self.selector({'status': 'success'})
        )
        inventory = config.common('engine_inventory', selector=self.selector())
        if history_seconds is not None:
            success = config.common(
                'history_presence', expression=success, window_seconds=history_seconds
            )
            inventory = config.common(
                'history_presence', expression=inventory, window_seconds=history_seconds
            )
        return config.common(
            'engine_guard',
            expression=expression,
            source_labels=', '.join(SOURCE_LABELS),
            plumbing_labels=', '.join(PLUMBING_LABELS),
            success=success,
            inventory=inventory,
        )


def _reduce(expression, scope, grouping=None, operation='sum') -> str:
    labels = scope.grouping if grouping is None else grouping
    if operation not in ('sum', 'avg', 'max', 'count') or any(
        not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', label) for label in labels
    ):
        raise ValueError('Invalid aggregation')
    return query_config.get_query_config().common(
        'aggregate',
        operation=operation,
        grouping=', '.join(labels),
        expression=expression,
    )


def build_gauge_query(
    name: str,
    scope: MetricQueryScope,
    *,
    aggregate: str = 'sum',
    group_by: Optional[Sequence[str]] = None,
    profile: str = 'gauge',
) -> str:
    if aggregate not in ('sum', 'avg', 'max'):
        raise ValueError('Invalid gauge aggregation')
    return _reduce(
        scope.guard(
            query_config.get_query_config().query(
                profile,
                'gauge',
                'sample',
                family=_metric_name(name),
                selector=scope.selector(),
            )
        ),
        scope,
        group_by,
        aggregate,
    )


def build_counter_query(
    name: str,
    scope: MetricQueryScope,
    seconds: int,
    *,
    summary: bool = False,
    group_by: Optional[Sequence[str]] = None,
    profile: str = 'counter',
) -> str:
    observations = build_counter_observations(
        name, scope, seconds, summary=summary, profile=profile
    )
    expression = _reduce(observations, scope, group_by)
    return (
        query_config.get_query_config().query(
            profile,
            'counter',
            'window_rate',
            expression=expression,
            window_seconds=seconds,
        )
        if summary
        else expression
    )


def build_counter_observations(
    name: str,
    scope: MetricQueryScope,
    seconds: int,
    *,
    summary: bool = False,
    profile: str = 'counter',
) -> str:
    """Retain source and native cohorts until numerator/denominator pairing."""
    operation = 'increase' if summary else 'rate'
    raw = query_config.get_query_config().query(
        profile,
        'counter',
        operation,
        family=_metric_name(name),
        selector=scope.selector(),
        window_seconds=seconds,
    )
    return scope.guard(raw, seconds if summary else None)


def build_ratio_query(
    numerator: str,
    denominator: str,
    scope: MetricQueryScope,
    seconds: int,
    *,
    profile: str = 'counter',
) -> str:
    config = query_config.get_query_config()
    numerator = build_counter_observations(numerator, scope, seconds, profile=profile)
    denominator = build_counter_observations(
        denominator, scope, seconds, profile=profile
    )
    paired_numerator = config.common('pair', left=numerator, right=denominator)
    paired_denominator = config.common('pair', left=denominator, right=numerator)
    ratio = config.common(
        'divide',
        numerator=_reduce(paired_numerator, scope),
        denominator=_reduce(paired_denominator, scope),
    )
    # Either unmatched population creates gaps instead of a biased ratio.
    grouping = ', '.join(scope.grouping)
    for paired, population in (
        (paired_numerator, numerator),
        (paired_denominator, denominator),
    ):
        complete = config.common(
            'compare_population',
            paired=_reduce(paired, scope, operation='count'),
            population=_reduce(population, scope, operation='count'),
        )
        ratio = config.common(
            'match_group', expression=ratio, grouping=grouping, condition=complete
        )
    return ratio


def build_histogram_query(
    name: str,
    scope: MetricQueryScope,
    statistic: str,
    seconds: int,
    *,
    summary: bool = False,
    group_by: Optional[Sequence[str]] = None,
    profile: str = 'histogram',
) -> str:
    name = _metric_name(name)
    if statistic not in HISTOGRAM_STATISTICS:
        raise ValueError('Invalid histogram statistic')
    labels = scope.grouping if group_by is None else tuple(group_by)
    operation = 'increase' if summary else 'rate'

    def part(suffix, grouping):
        raw = query_config.get_query_config().query(
            profile,
            'histogram',
            operation,
            family=name,
            part=suffix,
            selector=scope.selector(),
            window_seconds=seconds,
        )
        return _reduce(scope.guard(raw, seconds if summary else None), scope, grouping)

    if statistic == 'avg':
        return query_config.get_query_config().query(
            profile,
            'histogram',
            'avg',
            sum=part('sum', labels),
            count=part('count', labels),
        )
    return query_config.get_query_config().query(
        profile,
        'histogram',
        statistic,
        buckets=part('bucket', labels + ('le',)),
    )


def build_histogram_components(
    name: str, scope: MetricQueryScope, seconds: int, *, profile: str = 'histogram'
) -> Dict[str, str]:
    """Keep source and native rank identity until bucket compatibility is known."""
    name = _metric_name(name)
    queries = {}
    for key, suffix in (('buckets', 'bucket'), ('sum', 'sum'), ('count', 'count')):
        raw = query_config.get_query_config().query(
            profile,
            'histogram',
            'increase',
            family=name,
            part=suffix,
            selector=scope.selector(),
            window_seconds=seconds,
        )
        queries[key] = scope.guard(raw, seconds)
    return queries


def _quantile(quantile: float, buckets: Mapping[float, float]) -> Optional[float]:
    ordered = sorted(buckets.items())
    if len(ordered) < 2 or ordered[-1][0] != math.inf or ordered[-1][1] <= 0:
        return None
    rank = quantile * ordered[-1][1]
    lower_bound, lower_count = 0.0, 0.0
    for bound, count in ordered:
        if count >= rank:
            if bound == math.inf:
                return ordered[-2][0]
            if count == lower_count:
                return bound
            return lower_bound + (bound - lower_bound) * (
                (rank - lower_count) / (count - lower_count)
            )
        lower_bound, lower_count = bound, count
    return None


def histogram_statistics(
    buckets_by_source: Mapping[object, Mapping[float, float]],
    *,
    count: float,
    total: float,
) -> Tuple[Dict[str, Optional[float]], bool]:
    """Merge only equal, valid cumulative bucket profiles; Avg is independent."""
    stats = dict.fromkeys(HISTOGRAM_STATISTICS)
    if count <= 0 or not math.isfinite(count):
        return stats, False
    stats['avg'] = total / count if math.isfinite(total) else None
    profiles: Sequence[Mapping[float, float]] = list(buckets_by_source.values())
    if not profiles:
        return stats, False
    bounds = set(profiles[0])
    if len(bounds) < 2 or math.inf not in bounds:
        return stats, False
    merged = dict.fromkeys(bounds, 0.0)
    for profile in profiles:
        if set(profile) != bounds:
            return stats, False
        previous = 0.0
        for bound, value in sorted(profile.items()):
            if math.isnan(bound) or not math.isfinite(value) or value < previous:
                return stats, False
            previous = value
            merged[bound] += value
    if not math.isclose(merged[math.inf], count, rel_tol=1e-9, abs_tol=1e-9):
        return stats, False
    for statistic, quantile in (('p50', 0.5), ('p95', 0.95), ('p99', 0.99)):
        stats[statistic] = _quantile(quantile, merged)
    return stats, True
