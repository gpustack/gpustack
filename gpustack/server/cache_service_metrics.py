"""Semantic metrics for cache services, proxied from Prometheus.

The provider catalog declares *what* each provider's metrics mean
(``metrics.mappings`` / ``metrics.throughput``: semantic keys mapped to
extraction rules over the provider's Prometheus exposition). This module
is the consumer of those declarations: it translates each rule into a
PromQL ``query_range`` against whichever Prometheus is configured —
the built-in one, or the external stack ``--prometheus-url`` names —
scoped to one
service through the ``cache_service_id`` label the exporter's scrape
discovery stamps on every target. The UI receives ready-to-chart
semantic series and never learns PromQL or per-provider metric names —
and because the label selector is injected server-side, a caller can
only ever read the series of a service it was authorized for.
"""

import asyncio
import logging
import time
from typing import List, Optional

import aiohttp

from gpustack.server.prometheus_query import (  # noqa: F401
    MAX_WINDOW_SECONDS,
    MIN_WINDOW_SECONDS,
    instant_value as _instant_value,
    parse_window,
    prometheus_url as _prometheus_url,
    promql_regex_literal as _promql_regex_literal,
    query_instant as _query_instant,
    query_range as _query_range,
)
from gpustack.schemas.cache_providers import (
    CacheProviderMetrics,
    CacheProviderMetricValue,
)
from gpustack.schemas.cache_services import (
    CacheServiceAttachedMetrics,
    CacheServiceMetricChart,
    CacheServiceMetricSeries,
    CacheServiceMetricsPublic,
    ModelCacheMetricsPublic,
)

logger = logging.getLogger(__name__)

# Labels worth returning per series: enough for the UI to draw one line
# per instance; scrape plumbing labels (job, instance, __*) stay out.
# l2_name distinguishes LMCache's per-adapter L2 usage series; on one
# instance every adapter shares the scrape labels, so dropping it would
# collapse the adapters into duplicate series.
_SERIES_LABEL_KEYS = ("worker_name", "cache_service_instance_id", "l2_name")

# Overall budget for one collection (charts + attached): with the
# per-query timeout and the fan-out this bounds a stuck Prometheus to
# one predictable failure instead of a near-minute hang.
_COLLECT_DEADLINE_SECONDS = 15.0


def build_metric_query(
    rule: CacheProviderMetricValue, selector: str, rate_window: str
) -> Optional[str]:
    """One extraction rule -> one PromQL expression.

    Counter ratios and histogram averages divide increases over the
    rate window (both operands come from the same target, so the label
    sets match and the division joins per series); gauges chart as-is.
    """
    if rule.gauge:
        return f"{rule.gauge}{selector}"
    if rule.rate:
        return f"rate({rule.rate}{selector}[{rate_window}])"
    if rule.ratio:
        numerator = rule.ratio.get("numerator")
        denominator = rule.ratio.get("denominator")
        if not numerator or not denominator:
            return None
        return (
            f"increase({numerator}{selector}[{rate_window}])"
            f" / increase({denominator}{selector}[{rate_window}])"
        )
    if rule.gauge_ratio:
        numerator = rule.gauge_ratio.get("numerator")
        denominator = rule.gauge_ratio.get("denominator")
        if not numerator or not denominator:
            return None
        return f"{numerator}{selector} / {denominator}{selector}"
    if rule.histogram_avg:
        base = rule.histogram_avg
        return (
            f"increase({base}_sum{selector}[{rate_window}])"
            f" / increase({base}_count{selector}[{rate_window}])"
        )
    return None


def build_aggregate_query(
    rule: CacheProviderMetricValue, selector: str, rate_window: str
) -> Optional[str]:
    """The service-level form of one extraction rule.

    Ratio and histogram forms sum their operands before dividing, so the
    aggregate is weighted by each instance's actual traffic (a mean of
    per-instance ratios would weight an idle instance like a busy one);
    gauges combine by their declared aggregate (sum by default, avg for
    ratio-shaped gauges)."""
    if rule.gauge:
        op = "avg" if rule.aggregate == "avg" else "sum"
        return f"{op}({rule.gauge}{selector})"
    if rule.rate:
        return f"sum(rate({rule.rate}{selector}[{rate_window}]))"
    if rule.ratio:
        numerator = rule.ratio.get("numerator")
        denominator = rule.ratio.get("denominator")
        if not numerator or not denominator:
            return None
        return (
            f"sum(increase({numerator}{selector}[{rate_window}]))"
            f" / sum(increase({denominator}{selector}[{rate_window}]))"
        )
    if rule.gauge_ratio:
        numerator = rule.gauge_ratio.get("numerator")
        denominator = rule.gauge_ratio.get("denominator")
        if not numerator or not denominator:
            return None
        return f"sum({numerator}{selector}) / sum({denominator}{selector})"
    if rule.histogram_avg:
        base = rule.histogram_avg
        return (
            f"sum(increase({base}_sum{selector}[{rate_window}]))"
            f" / sum(increase({base}_count{selector}[{rate_window}]))"
        )
    return None


# Engine-side external-cache hit accounting: the worker's metrics
# aggregator maps vLLM's external_prefix_cache_* counters to these
# unified names on the scraped /metrics exposition (the raw vllm:*
# names live on /metrics/raw, which Prometheus does not scrape) and
# relabels them with model/instance identity. Engine counters, not
# provider metrics — hence platform-known names here and not a catalog
# declaration. Only vLLM maps them today; other engines' instances
# keep None values.
# The names carry the _total suffix the Prometheus client appends to
# counters at exposition — the scraped series name, not the registered
# one.
_ENGINE_HIT_COUNTER = "gpustack:external_prefix_cache_hits_total"
_ENGINE_QUERY_COUNTER = "gpustack:external_prefix_cache_queries_total"
_ATTACHED_GROUP_LABELS = "(model_instance_name)"


def _to_series(result: List[dict]) -> List[CacheServiceMetricSeries]:
    """A Prometheus query_range matrix -> chartable series. Non-finite
    samples (an idle counter ratio evaluates 0/0 = NaN) become None so
    the chart shows a gap instead of a bogus value."""
    series = []
    for entry in result:
        metric = entry.get("metric") or {}
        labels = {key: metric[key] for key in _SERIES_LABEL_KEYS if metric.get(key)}
        points: List[List[Optional[float]]] = []
        for timestamp, value in entry.get("values") or []:
            try:
                parsed = float(value)
                if parsed != parsed or parsed in (float("inf"), float("-inf")):
                    parsed = None
            except (TypeError, ValueError):
                parsed = None
            points.append([float(timestamp), parsed])
        series.append(CacheServiceMetricSeries(labels=labels, points=points))
    return series


async def _collect_attached(
    client: aiohttp.ClientSession,
    prometheus_url: str,
    cluster_id: int,
    attached: List[CacheServiceAttachedMetrics],
    window_seconds: int,
    at: float,
) -> List[CacheServiceAttachedMetrics]:
    """Fill the DB-enumerated attached instances with the engines'
    external-cache hit accounting over the window.

    The database is the authority on which instances exist — metrics
    only carry the numbers. Driving the row set off Prometheus series
    would lag it in both directions: a fresh instance is invisible
    until its first scrape, a deleted one lingers for the window, and
    an engine without the counters would vanish instead of showing
    empty values."""
    if not attached:
        return []
    model_ids = sorted(
        {record.model_id for record in attached if record.model_id is not None}
    )
    if not model_ids:
        return attached
    ids = "|".join(str(model_id) for model_id in model_ids)
    selector = f'{{cluster_id="{cluster_id}",model_id=~"{ids}"}}'
    window = f"{window_seconds}s"
    by_instance = {record.model_instance_name: record for record in attached}
    # Both counters buffer before any record is touched: hits landing
    # while the second query fails would leave rows claiming hits out of
    # zero lookups. A failed query raises, leaving what to make of it to
    # the caller — a deployment's own view has nothing else to show, a
    # service's view has its own numbers already.
    buffered: dict = {}
    for field, counter in (
        ("hit_tokens", _ENGINE_HIT_COUNTER),
        ("queried_tokens", _ENGINE_QUERY_COUNTER),
    ):
        query = (
            f"sum by {_ATTACHED_GROUP_LABELS} "
            f"(increase({counter}{selector}[{window}]))"
        )
        result = await _query_instant(client, prometheus_url, query, at)
        buffered[field] = {
            (entry.get("metric") or {}).get("model_instance_name"): _instant_value(
                entry
            )
            for entry in result
        }
    for field, by_name in buffered.items():
        for name, value in by_name.items():
            record = by_instance.get(name)
            if record is not None:
                setattr(record, field, value)
    for record in attached:
        if record.queried_tokens:
            record.hit_rate = (record.hit_tokens or 0.0) / record.queried_tokens
    return attached


async def _collect_charts(
    client: aiohttp.ClientSession,
    prometheus_url: str,
    metrics: CacheProviderMetrics,
    result: CacheServiceMetricsPublic,
    cache_service_id: int,
    selector: str,
    rate_window: str,
    start: float,
    end: float,
    step: int,
) -> None:
    """One query per declared key and granularity: they run concurrently
    (a catalog like LMCache's declares ~18), capped below the built-in
    Prometheus's query-concurrency limit. A ValueError stays isolated to
    its own chart; connection-level errors surface after the batch and
    degrade the collection."""
    query_slots = asyncio.Semaphore(8)
    failures: List[str] = []

    async def _fill(chart: CacheServiceMetricChart, attr: str, key: str, query: str):
        try:
            async with query_slots:
                series = _to_series(
                    await _query_range(client, prometheus_url, query, start, end, step)
                )
            setattr(chart, attr, series)
        except (ValueError, asyncio.TimeoutError) as e:
            # a per-query timeout is one chart's failure like any other
            # (a heavy 7d query must not blank the page); the
            # collection-wide deadline stays with wait_for, whose
            # cancellation propagates as CancelledError and is not
            # caught here
            message = str(e) or "timed out"
            failures.append(f"{key}: {message}")
            logger.warning(
                f"Cache service {cache_service_id} metric "
                f"'{key}' query failed: {message}"
            )

    fills = []
    for target, rules in (
        (result.mappings, metrics.mappings),
        (result.throughput, metrics.throughput),
    ):
        for key, rule in rules.items():
            chart = CacheServiceMetricChart()
            for attr, query in (
                ("aggregate", build_aggregate_query(rule, selector, rate_window)),
                ("instances", build_metric_query(rule, selector, rate_window)),
            ):
                if query is not None:
                    fills.append(_fill(chart, attr, key, query))
            target[key] = chart
    if fills:
        outcomes = await asyncio.gather(*fills, return_exceptions=True)
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                raise outcome
    # Isolation is for partial failures; every query failing is one
    # systematic cause (a bad selector, a broken declaration) that must
    # not present as a wall of silently empty charts.
    if fills and len(failures) == len(fills):
        raise ValueError(
            f"all {len(fills)} metric queries failed; first: {failures[0]}"
        )


async def collect_model_cache_metrics(
    cluster_id: int,
    attached: List[CacheServiceAttachedMetrics],
    window_seconds: int,
    client: Optional[aiohttp.ClientSession] = None,
) -> ModelCacheMetricsPublic:
    """Engine-side external-cache hit accounting for one deployment's
    instances.

    The rows come in database-enumerated (the caller knows the
    deployment's instances); this fills their numbers from the same
    engine counters the cache service's own view reads, bounded to the
    handed-in rows.
    """
    prometheus_url = _prometheus_url()
    if not prometheus_url:
        return ModelCacheMetricsPublic(
            available=False,
            reason=(
                "No Prometheus is reachable: built-in observability is "
                "disabled and no external Prometheus is configured"
            ),
            instances=attached,
        )
    if not attached:
        return ModelCacheMetricsPublic(
            available=True, window=window_seconds, instances=[]
        )

    owned = client is None
    try:
        if owned:
            client = aiohttp.ClientSession()
        await asyncio.wait_for(
            _collect_attached(
                client,
                prometheus_url,
                cluster_id,
                attached,
                window_seconds,
                time.time(),
            ),
            timeout=_COLLECT_DEADLINE_SECONDS,
        )
    except asyncio.TimeoutError:
        return ModelCacheMetricsPublic(
            available=False,
            reason="Prometheus queries timed out",
            instances=attached,
        )
    except ValueError as e:
        return ModelCacheMetricsPublic(
            available=False, reason=str(e), instances=attached
        )
    except (aiohttp.ClientError, OSError) as e:
        return ModelCacheMetricsPublic(
            available=False,
            reason=f"Prometheus is unreachable: {str(e) or e.__class__.__name__}",
            instances=attached,
        )
    finally:
        if owned and client is not None:
            await client.close()
    return ModelCacheMetricsPublic(
        available=True, window=window_seconds, instances=attached
    )


async def collect_cache_service_metrics(
    metrics: Optional[CacheProviderMetrics],
    cache_service_id: int,
    window_seconds: int,
    cluster_id: Optional[int] = None,
    attached: Optional[List[CacheServiceAttachedMetrics]] = None,
    worker_names: Optional[List[str]] = None,
    client: Optional[aiohttp.ClientSession] = None,
) -> CacheServiceMetricsPublic:
    """Chartable semantic series for one cache service.

    Every declared key charts at two granularities (service aggregate +
    per-instance breakdown); ``attached`` (DB-enumerated instances of the
    attached deployments) comes back filled with the engines' own hit
    accounting where the counters exist. ``worker_names`` narrows the
    charts to the selected workers' instances — filtered inside the
    PromQL selector, so the aggregate over the subset stays weighted by
    the selected instances' actual traffic (a client-side filter could
    not re-aggregate ratios correctly).

    ``available=False`` carries the reason the charts cannot render at
    all (no declaration, observability disabled, Prometheus down); a
    single failing query only logs and yields an empty chart for its
    key, so one bad declaration entry cannot blank every chart.
    """
    if metrics is None or not (metrics.mappings or metrics.throughput):
        return CacheServiceMetricsPublic(
            available=False,
            reason="The provider declares no metrics",
        )

    prometheus_url = _prometheus_url()
    if not prometheus_url:
        return CacheServiceMetricsPublic(
            available=False,
            reason=(
                "No Prometheus is reachable: built-in observability is "
                "disabled and no external Prometheus is configured"
            ),
        )

    end = time.time()
    start = end - window_seconds
    step = max(window_seconds // 120, 15)
    # The increase() lookback covers one step plus a scrape-interval
    # cushion (Grafana's $__rate_interval convention): long enough that
    # activity between evaluation points is never missed, short enough
    # that a burst charts near when it happened — a lookback of several
    # steps would smear it up to that far past its actual time.
    rate_window = f"{max(step + 60, 60)}s"
    matchers = [f'cache_service_id="{cache_service_id}"']
    if worker_names:
        names = "|".join(
            _promql_regex_literal(name) for name in sorted(set(worker_names))
        )
        matchers.append(f'worker_name=~"{names}"')
    selector = "{" + ",".join(matchers) + "}"

    result = CacheServiceMetricsPublic(available=True, start=start, end=end, step=step)
    # The caller may hand in the app's shared session (queries carry
    # their own timeout); without one, a session is created and closed
    # here.
    owned = client is None
    try:
        if owned:
            client = aiohttp.ClientSession()

        async def _run():
            await _collect_charts(
                client,
                prometheus_url,
                metrics,
                result,
                cache_service_id,
                selector,
                rate_window,
                start,
                end,
                step,
            )
            if cluster_id is not None:
                try:
                    result.attached = await _collect_attached(
                        client,
                        prometheus_url,
                        cluster_id,
                        attached or [],
                        window_seconds,
                        end,
                    )
                except ValueError as e:
                    # The service's own numbers are already collected and
                    # real; losing the per-deployment breakdown leaves
                    # them worth serving, with its rows empty.
                    logger.warning(f"Attached cache metrics query failed: {e}")
                    result.attached = attached or []

        await asyncio.wait_for(_run(), timeout=_COLLECT_DEADLINE_SECONDS)
    # On 3.11+ asyncio.TimeoutError is TimeoutError, a subclass of
    # OSError: this arm must stay ahead of the OSError one below or the
    # deadline would report as "unreachable".
    except asyncio.TimeoutError:
        return CacheServiceMetricsPublic(
            available=False,
            reason="Prometheus queries timed out",
        )
    except ValueError as e:
        return CacheServiceMetricsPublic(available=False, reason=str(e))
    except (aiohttp.ClientError, OSError) as e:
        return CacheServiceMetricsPublic(
            available=False,
            reason=f"Prometheus is unreachable: {str(e) or e.__class__.__name__}",
        )
    finally:
        if owned and client is not None:
            await client.close()
    return result
