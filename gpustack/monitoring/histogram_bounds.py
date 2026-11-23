"""Identify quantiles that a histogram cannot resolve."""

import math
from collections import defaultdict
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from gpustack.schemas.model_metrics import MetricDimensions, MetricSeries

from .queries import MetricQueryScope, histogram_statistics

QUANTILES = {'p50': 0.5, 'p95': 0.95, 'p99': 0.99}


def histogram_tail(
    profiles: Mapping[object, Mapping[float, float]], *, count: float
) -> Optional[Tuple[float, float]]:
    if not histogram_statistics(profiles, count=count, total=math.nan)[1]:
        return None
    bounds = next(iter(profiles.values()))
    finite = [bound for bound in bounds if math.isfinite(bound)]
    if not finite or max(finite) < 0:
        return None
    ceiling = max(finite)
    return ceiling, sum(profile[ceiling] for profile in profiles.values()) / count


def histogram_lower_bounds(
    profiles: Mapping[object, Mapping[float, float]], *, count: float
) -> Dict[str, float]:
    tail = histogram_tail(profiles, count=count)
    if tail is None:
        return {}
    ceiling, fraction = tail
    return {
        stat: ceiling
        for stat, quantile in QUANTILES.items()
        if quantile > fraction and not math.isclose(quantile, fraction, rel_tol=1e-12)
    }


def first_bucket_ceiling(
    profiles: Mapping[object, Mapping[float, float]],
) -> Optional[float]:
    """Find the first positive ceiling of a nonnegative histogram."""
    bounds = next(iter(profiles.values()), {})
    bound = min((bound for bound in bounds if bound != 0), default=math.inf)
    return bound if 0 < bound < math.inf else None


def all_in_first_bucket(
    profiles: Mapping[object, Mapping[float, float]], count: float
) -> bool:
    if count <= 0:
        return False
    bound = first_bucket_ceiling(profiles)
    return (
        bound is not None
        and all(profile.get(0, 0) == 0 for profile in profiles.values())
        and math.isclose(
            sum(profile[bound] for profile in profiles.values()), count, rel_tol=1e-9
        )
    )


def group_key(dimensions: MetricDimensions, view: str) -> object:
    role = dimensions.role or ''
    return (role, dimensions.model_instance_id) if view == 'instances' else role


def trend_ceilings(groups: Mapping[object, dict]) -> Dict[object, float]:
    return {
        key: tail[0]
        for key, group in groups.items()
        if not group.get('ambiguous_bounds')
        and (tail := histogram_tail(group['profiles'], count=group['count']))
        is not None
    }


def trend_first_bucket_ceilings(groups: Mapping[object, dict]) -> Dict[object, float]:
    return {
        key: bound
        for key in trend_ceilings(groups)
        if (bound := first_bucket_ceiling(groups[key]['profiles'])) is not None
    }


def ceiling_scopes(
    scope: MetricQueryScope, ceilings: Mapping[object, float], buckets: Sequence[dict]
) -> List[Tuple[MetricQueryScope, float, Tuple[str, ...]]]:
    populations = defaultdict(set)
    for key, bound in ceilings.items():
        role, identity = key if isinstance(key, tuple) else (key, None)
        if role in scope.roles:
            ids = (identity,) if identity is not None else scope.instance_ids
            populations[role, bound].update(i for i in ids if i in scope.instance_ids)
    aliases = defaultdict(set)
    for row in buckets:
        label = (row.get('metric') or {}).get('le', '')
        try:
            bound = float(label)
        except (TypeError, ValueError):
            continue
        if math.isfinite(bound):
            aliases[bound].add(label)
    return [
        (
            MetricQueryScope(
                scope.cluster_id,
                scope.model_id,
                tuple(sorted(ids)),
                (role,),
                scope.view,
                scope.launches,
                scope.source_identities,
                scope.sample_start,
            ),
            bound,
            tuple(sorted(aliases[bound])),
        )
        for (role, bound), ids in sorted(populations.items())
        if ids
    ]


def requires_bound_check(
    series: Sequence[MetricSeries], ceilings: Mapping[object, float], view: str
) -> bool:
    return any(
        line.statistic in QUANTILES
        and (bound := ceilings.get(group_key(line.dimensions, view))) is not None
        and any(value is not None and value >= bound for _, value in line.points)
        for line in series
    )


def _is_first_bucket_estimate(
    value: Optional[float], bound: float, quantile: float
) -> bool:
    return value is not None and math.isclose(value, bound * quantile, rel_tol=1e-9)


def requires_first_bucket_check(
    series: Sequence[MetricSeries], ceilings: Mapping[object, float], view: str
) -> bool:
    return any(
        (quantile := QUANTILES.get(line.statistic)) is not None
        and (bound := ceilings.get(group_key(line.dimensions, view))) is not None
        and any(
            _is_first_bucket_estimate(value, bound, quantile)
            for _, value in line.points
        )
        for line in series
    )


def _cdf_fractions(rows, ceilings, view):
    fractions = {}
    for row in rows:
        labels = row.get('metric') or {}
        role = labels.get('role', '')
        try:
            identity = int(labels['model_instance_id']) if view == 'instances' else None
            key = (role, identity) if view == 'instances' else role
            bound = float(labels['histogram_upper_bound'])
        except (KeyError, TypeError, ValueError):
            continue
        if key not in ceilings or bound != ceilings[key]:
            continue
        for timestamp, value in row.get('values') or []:
            try:
                timestamp, value = float(timestamp), float(value)
            except (TypeError, ValueError):
                continue
            if (
                not math.isfinite(timestamp)
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                continue
            sample = key, timestamp
            if sample in fractions and fractions[sample] != value:
                fractions[sample] = None
            else:
                fractions[sample] = value
    return fractions


def apply_lower_bounds(
    series: Sequence[MetricSeries],
    ceilings: Mapping[object, float],
    rows: Sequence[dict],
    *,
    view: str,
) -> Tuple[bool, bool]:
    fractions = _cdf_fractions(rows, ceilings, view)
    censored, incomplete = False, False
    for line in series:
        quantile, key = QUANTILES.get(line.statistic), group_key(line.dimensions, view)
        bound = ceilings.get(key)
        if quantile is None or bound is None:
            continue
        points, limits = [], []
        for timestamp, value in line.points:
            if value is None or value < bound:
                points.append((timestamp, value))
                continue
            fraction = fractions.get((key, timestamp))
            if fraction is None:
                incomplete = True
                points.append((timestamp, None))
            elif quantile > fraction and not math.isclose(
                quantile, fraction, rel_tol=1e-12
            ):
                censored = True
                points.append((timestamp, None))
                limits.append((timestamp, bound))
            else:
                points.append((timestamp, value))
        line.points = points
        line.lower_bounds = limits or None
    return censored, incomplete


def censor_first_bucket_estimates(
    series: Sequence[MetricSeries],
    ceilings: Mapping[object, float],
    rows: Sequence[dict],
    *,
    view: str,
) -> Tuple[bool, bool]:
    fractions = _cdf_fractions(rows, ceilings, view)
    censored, incomplete = False, False
    for line in series:
        quantile, key = QUANTILES.get(line.statistic), group_key(line.dimensions, view)
        bound = ceilings.get(key)
        if quantile is None or bound is None:
            continue
        for index, (timestamp, value) in enumerate(line.points):
            if not _is_first_bucket_estimate(value, bound, quantile):
                continue
            fraction = fractions.get((key, timestamp))
            if fraction is None:
                incomplete = True
            elif math.isclose(fraction, 1.0, rel_tol=1e-9):
                line.points[index] = timestamp, None
                censored = True
    return censored, incomplete
