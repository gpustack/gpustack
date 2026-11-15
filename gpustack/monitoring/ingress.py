"""Gateway-observed latency averages, independent of engine timing and buckets."""

import asyncio
import json
import math

from gpustack.schemas.model_metrics import (
    MetricDimensions,
    MetricSeries,
    MetricSummary,
    ModelMetric,
)

from .config import get_query_config
from .queries import cohort_identity


def gateway_selector(model_id, instance_ids):
    """Select globally unique deployment registries, never a model display name."""
    if (
        type(model_id) is not int
        or model_id <= 0
        or not instance_ids
        or any(type(i) is not int or i <= 0 for i in instance_ids)
    ):
        raise ValueError('Invalid gateway model population')
    ids = '|'.join(str(i) for i in sorted(set(instance_ids)))
    # These are model_instance_registry / ModelInstanceEndpoint.registry_name
    # names, including native-DP and hybrid-worker endpoint suffixes.
    cluster = (
        rf'outbound\|[0-9]+\|\|model-{model_id}-({ids})'
        r'(-dp[0-9]+|-w[0-9]+)?\.(static|dns)'
    )
    return '{ai_cluster=~' + json.dumps(cluster) + '}'


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _samples(entries, ranged=False):
    samples = {}
    for row in entries:
        identity = cohort_identity(row.get('metric') or {})
        points = row.get('values', []) if ranged else [(None, row['value'][1])]
        for timestamp, raw in points:
            value = _finite(raw)
            if value is None:
                continue
            if value < 0:
                raise ValueError('Negative gateway counter increment')
            cohort = samples.setdefault(timestamp, {})
            if identity in cohort and cohort[identity] != value:
                raise ValueError('Conflicting duplicate gateway metric samples')
            cohort[identity] = value
    return samples


def _average(sums, counts, scale):
    if not counts:
        return None, None
    active = {key: value for key, value in counts.items() if value > 0}
    if any(key not in sums for key in active):
        raise ValueError('Gateway duration and count populations do not match')
    if any(value > 0 and key not in counts for key, value in sums.items()):
        raise ValueError('Gateway duration and count populations do not match')
    count = sum(counts.values())
    return (
        sum(sums.get(key, 0) for key in active) / count * scale if count else None
    ), count


async def collect_ingress_metric(
    definition,
    *,
    model_id,
    instance_ids,
    read,
    window_seconds,
    rate_window_seconds,
    statistics,
):
    """Pair native sums and counts before calculating a weighted Avg in seconds.

    No consumer identities leave this collector. Instance view still returns one
    service-level series; the gateway does not measure P/D role-local timings.
    """
    selector = gateway_selector(model_id, instance_ids)
    config = get_query_config()

    def query(family, summary=False):
        return config.query(
            definition.query_profile,
            'histogram',
            'increase' if summary else 'rate',
            scope='gateway',
            family=family,
            selector=selector,
            window_seconds=window_seconds if summary else rate_window_seconds,
        )

    requests = [
        read(query(family, True))
        for family in (definition.family, definition.denominator)
    ]
    if 'avg' in statistics:
        requests.extend(
            read(query(family), ranged=True)
            for family in (definition.family, definition.denominator)
        )
    values = await asyncio.gather(*requests)
    sums, counts = (_samples(rows) for rows in values[:2])
    value, count = _average(sums.get(None, {}), counts.get(None, {}), definition.scale)
    dimensions = MetricDimensions(source_name='gateway')
    metric = ModelMetric(
        kind='histogram',
        unit='seconds',
        measurement_scope='service_ingress',
        supported_statistics=['avg'],
        status='no_data',
        reason='no_samples',
        summaries=[
            MetricSummary(
                statistic='avg', value=value, sample_count=count, dimensions=dimensions
            )
        ],
    )
    if len(values) > 2:
        trend_sums, trend_counts = (_samples(rows, True) for rows in values[2:])
        timestamps = sorted(trend_sums.keys() | trend_counts.keys())
        points = [
            (
                t,
                _average(
                    trend_sums.get(t, {}), trend_counts.get(t, {}), definition.scale
                )[0],
            )
            for t in timestamps
        ]
        if points:
            metric.series = [
                MetricSeries(
                    name='Service ingress',
                    statistic='avg',
                    dimensions=dimensions,
                    points=points,
                )
            ]
    if value is not None or any(
        v is not None for line in metric.series for _, v in line.points
    ):
        metric.status, metric.reason = 'available', None
    elif count is not None:
        metric.status, metric.reason = 'no_observations', 'zero_observations'
    return metric
