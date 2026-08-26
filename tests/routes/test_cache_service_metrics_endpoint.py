"""Cache-service semantic metrics endpoint.

The provider catalog declares what each provider's metrics mean; the
endpoint translates those declarations into PromQL scoped by a
server-injected service-label selector and returns chartable semantic
series. The router mounts Org-owner-only; the handler asserts the role
itself, so the gate holds independent of mount policy. Missing
observability degrades to available=False instead of erroring, so the
UI can hide the charts with a reason.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gpustack.api.exceptions import BadRequestException, ForbiddenException
from gpustack.api.tenant import TenantContext
from gpustack.routes import cache_services as cache_services_route
from gpustack.schemas.cache_providers import (
    CacheProviderMetrics,
    CacheProviderMetricValue,
)
from gpustack.schemas.cache_services import CacheServiceAttachedMetrics
from gpustack.schemas.principals import OrgRole, PrincipalType
from gpustack.server import cache_service_metrics as metrics_module
from gpustack.server.cache_service_metrics import (
    build_aggregate_query,
    build_metric_query,
    collect_cache_service_metrics,
    parse_window,
)

ORG_PRINCIPAL = 42


def _ctx(org_role=None, is_platform_admin=False) -> TenantContext:
    user = MagicMock()
    user.kind = PrincipalType.USER
    return TenantContext(
        user=user,
        is_platform_admin=is_platform_admin,
        current_principal_id=ORG_PRINCIPAL,
        org_role=org_role,
    )


def _request():
    # the route reads request.app.state.http_client_no_proxy with a
    # getattr fallback, so bare namespaces suffice
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))


def _service(**overrides):
    fields = dict(
        id=5,
        name="lmcache-svc",
        provider_name="LMCache",
        cluster_id=1,
        owner_principal_id=ORG_PRINCIPAL,
        deleted_at=None,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _declaration() -> CacheProviderMetrics:
    return CacheProviderMetrics(
        mappings={
            "hit_rate": CacheProviderMetricValue(
                ratio={
                    "numerator": "hit_total",
                    "denominator": "requested_total",
                }
            ),
            "l1_usage_bytes": CacheProviderMetricValue(gauge="l1_usage_bytes"),
        },
        throughput={
            "l2_store": CacheProviderMetricValue(histogram_avg="l2_store_tp"),
        },
    )


def test_parse_window():
    assert parse_window("30m") == 1800
    assert parse_window("6h") == 6 * 3600
    assert parse_window("7d") == 7 * 86400
    for invalid in ("", "1s", "10x", "1m", "8d", "0h"):
        with pytest.raises(ValueError):
            parse_window(invalid)


def test_build_metric_query_forms():
    selector = '{cache_service_id="5"}'
    assert (
        build_metric_query(
            CacheProviderMetricValue(gauge="l1_usage_bytes"), selector, "60s"
        )
        == 'l1_usage_bytes{cache_service_id="5"}'
    )
    assert build_metric_query(
        CacheProviderMetricValue(ratio={"numerator": "hit", "denominator": "req"}),
        selector,
        "60s",
    ) == (
        'increase(hit{cache_service_id="5"}[60s])'
        ' / increase(req{cache_service_id="5"}[60s])'
    )
    assert build_metric_query(
        CacheProviderMetricValue(
            gauge_ratio={"numerator": "allocated", "denominator": "capacity"}
        ),
        selector,
        "60s",
    ) == ('allocated{cache_service_id="5"} / capacity{cache_service_id="5"}')
    assert build_metric_query(
        CacheProviderMetricValue(histogram_avg="tp"), selector, "60s"
    ) == (
        'increase(tp_sum{cache_service_id="5"}[60s])'
        ' / increase(tp_count{cache_service_id="5"}[60s])'
    )
    assert build_metric_query(
        CacheProviderMetricValue(rate="lookup_total"), selector, "60s"
    ) == ('rate(lookup_total{cache_service_id="5"}[60s])')
    assert build_metric_query(CacheProviderMetricValue(), selector, "60s") is None


def test_metric_rule_declaration_validation():
    """A rule sets at most one extraction form (the builder would
    otherwise silently pick one of several), and aggregate only
    modifies the gauge form."""
    with pytest.raises(ValueError):
        CacheProviderMetricValue(gauge="a", rate="b")
    with pytest.raises(ValueError):
        CacheProviderMetricValue(
            ratio={"numerator": "a", "denominator": "b"}, aggregate="sum"
        )
    with pytest.raises(ValueError):
        CacheProviderMetricValue(gauge="a", aggregate="max")
    assert CacheProviderMetricValue(gauge="a", aggregate="avg").aggregate == "avg"


def test_build_aggregate_query_forms():
    """Service-level forms: ratio/histogram operands sum before dividing
    (traffic-weighted), gauges combine by their declared aggregate."""
    selector = '{cache_service_id="5"}'
    assert build_aggregate_query(
        CacheProviderMetricValue(gauge="l1_usage_bytes"), selector, "60s"
    ) == ('sum(l1_usage_bytes{cache_service_id="5"})')
    assert build_aggregate_query(
        CacheProviderMetricValue(gauge="l1_usage_ratio", aggregate="avg"),
        selector,
        "60s",
    ) == ('avg(l1_usage_ratio{cache_service_id="5"})')
    assert build_aggregate_query(
        CacheProviderMetricValue(ratio={"numerator": "hit", "denominator": "req"}),
        selector,
        "60s",
    ) == (
        'sum(increase(hit{cache_service_id="5"}[60s]))'
        ' / sum(increase(req{cache_service_id="5"}[60s]))'
    )
    assert build_aggregate_query(
        CacheProviderMetricValue(histogram_avg="tp"), selector, "60s"
    ) == (
        'sum(increase(tp_sum{cache_service_id="5"}[60s]))'
        ' / sum(increase(tp_count{cache_service_id="5"}[60s]))'
    )
    assert build_aggregate_query(
        CacheProviderMetricValue(rate="lookup_total"), selector, "60s"
    ) == ('sum(rate(lookup_total{cache_service_id="5"}[60s]))')


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status = status

    async def text(self):
        # a non-dict payload stands in for a non-JSON body (e.g. a
        # gateway error page)
        if isinstance(self._payload, str):
            return self._payload
        return json.dumps(self._payload)


class _FakeAsyncCtx:
    def __init__(self, value):
        self._value = value

    async def __aenter__(self):
        return self._value

    async def __aexit__(self, *args):
        return False


class _FakeHTTPClient:
    def __init__(self, payload, status=200):
        self._payload = payload
        self._status = status
        self.requests = []

    def get(self, url, params=None, timeout=None):
        self.requests.append((url, params))
        return _FakeAsyncCtx(_FakeResponse(self._payload, status=self._status))

    async def close(self):
        pass


def _patch_prometheus(monkeypatch, client, url="http://127.0.0.1:19090"):
    monkeypatch.setattr(
        metrics_module,
        "get_global_config",
        lambda: SimpleNamespace(get_builtin_prometheus_url=lambda: url),
    )
    monkeypatch.setattr(
        metrics_module.aiohttp,
        "ClientSession",
        lambda *args, **kwargs: client,
    )


@pytest.mark.asyncio
async def test_collect_returns_semantic_series(monkeypatch):
    payload = {
        "status": "success",
        "data": {
            "result": [
                {
                    "metric": {
                        "worker_name": "w1",
                        "cache_service_instance_id": "11",
                        "l2_name": "fs_native",
                        "job": "scrape-plumbing",
                        "instance": "10.0.0.5:9188",
                    },
                    # an idle counter ratio evaluates 0/0 = NaN: charted
                    # as a gap, never as a bogus number
                    "values": [[1000, "0.5"], [1030, "NaN"]],
                }
            ]
        },
    }
    client = _FakeHTTPClient(payload)
    _patch_prometheus(monkeypatch, client)

    result = await collect_cache_service_metrics(_declaration(), 5, 3600)

    assert result.available is True
    assert set(result.mappings) == {"hit_rate", "l1_usage_bytes"}
    assert set(result.throughput) == {"l2_store"}
    chart = result.mappings["hit_rate"]
    # both granularities chart: the aggregate default view and the
    # per-instance breakdown
    assert chart.aggregate and chart.instances
    series = chart.instances[0]
    # scrape plumbing labels are filtered; identity labels stay,
    # including the per-adapter l2_name that keeps multi-adapter
    # usage series apart
    assert series.labels == {
        "worker_name": "w1",
        "cache_service_instance_id": "11",
        "l2_name": "fs_native",
    }
    assert series.points == [[1000.0, 0.5], [1030.0, None]]
    # every query carries the server-injected service selector
    queries = [params["query"] for _, params in client.requests]
    assert all('cache_service_id="5"' in q for q in queries)
    assert any(q.startswith("sum(increase(hit_total") for q in queries)


@pytest.mark.asyncio
async def test_collect_filters_by_worker_names(monkeypatch):
    """A worker filter lands inside the PromQL selector (regex-escaped),
    so the aggregate over the subset stays weighted by the selected
    instances' traffic."""
    payload = {"status": "success", "data": {"result": []}}
    client = _FakeHTTPClient(payload)
    _patch_prometheus(monkeypatch, client)

    await collect_cache_service_metrics(
        _declaration(), 5, 3600, worker_names=["worker-1", "w2.node", 'x"y']
    )

    for _, params in client.requests:
        # two escaping layers: RE2 metacharacters, then the PromQL
        # string literal (whose lexer rejects unknown escapes like a
        # lone \-)
        assert r'worker_name=~"w2\\.node|worker\\-1|x\"y"' in params["query"]


@pytest.mark.asyncio
async def test_collect_without_observability(monkeypatch):
    monkeypatch.setattr(
        metrics_module,
        "get_global_config",
        lambda: SimpleNamespace(get_builtin_prometheus_url=lambda: None),
    )
    result = await collect_cache_service_metrics(_declaration(), 5, 3600)
    assert result.available is False
    assert "observability is disabled" in result.reason


@pytest.mark.asyncio
async def test_collect_without_declaration():
    result = await collect_cache_service_metrics(None, 5, 3600)
    assert result.available is False
    assert "declares no metrics" in result.reason

    result = await collect_cache_service_metrics(
        CacheProviderMetrics(path="/metrics"), 5, 3600
    )
    assert result.available is False


@pytest.mark.asyncio
async def test_collect_single_query_timeout_keeps_other_charts(monkeypatch):
    """A per-query timeout is one chart's failure like any other — a
    heavy query hitting its timeout must not blank the page."""
    import asyncio as _asyncio

    class _SlowOneClient(_FakeHTTPClient):
        def get(self, url, params=None, timeout=None):
            self.requests.append((url, params))
            if "l1_usage_bytes" in params["query"]:
                raise _asyncio.TimeoutError()
            return _FakeAsyncCtx(_FakeResponse(self._payload))

    payload = {
        "status": "success",
        "data": {"result": [{"metric": {}, "values": [[1000, "1"]]}]},
    }
    client = _SlowOneClient(payload)
    _patch_prometheus(monkeypatch, client)

    result = await collect_cache_service_metrics(_declaration(), 5, 3600)

    assert result.available is True
    assert result.mappings["l1_usage_bytes"].aggregate == []
    assert result.mappings["hit_rate"].instances[0].points == [[1000.0, 1.0]]


@pytest.mark.asyncio
async def test_collect_all_queries_failing_degrades_with_reason(monkeypatch):
    """Failure isolation is for partial failures; every query failing
    (e.g. a gateway error page on each) is one systematic cause and
    degrades the collection with a reason instead of presenting a wall
    of silently empty charts."""
    client = _FakeHTTPClient("<html>Bad Gateway</html>", status=502)
    _patch_prometheus(monkeypatch, client)

    result = await collect_cache_service_metrics(_declaration(), 5, 3600)

    assert result.available is False
    assert "502" in result.reason


@pytest.mark.asyncio
async def test_collect_prometheus_unreachable(monkeypatch):
    class _RefusingClient(_FakeHTTPClient):
        def get(self, url, params=None, timeout=None):
            raise metrics_module.aiohttp.ClientError("connection refused")

    _patch_prometheus(monkeypatch, _RefusingClient(None))
    result = await collect_cache_service_metrics(_declaration(), 5, 3600)
    assert result.available is False
    assert "unreachable" in result.reason


@pytest.mark.asyncio
async def test_collect_single_query_failure_keeps_other_charts(monkeypatch):
    """One failing query (e.g. a bad declaration entry) yields an empty
    list for its key without blanking the other charts."""

    class _MixedClient(_FakeHTTPClient):
        def get(self, url, params=None, timeout=None):
            self.requests.append((url, params))
            if "l1_usage_bytes" in params["query"]:
                return _FakeAsyncCtx(
                    _FakeResponse({"status": "error", "error": "bad query"}, status=400)
                )
            return _FakeAsyncCtx(_FakeResponse(self._payload))

    payload = {
        "status": "success",
        "data": {"result": [{"metric": {}, "values": [[1000, "1"]]}]},
    }
    client = _MixedClient(payload)
    _patch_prometheus(monkeypatch, client)

    result = await collect_cache_service_metrics(_declaration(), 5, 3600)
    assert result.available is True
    assert result.mappings["l1_usage_bytes"].aggregate == []
    assert result.mappings["l1_usage_bytes"].instances == []
    assert result.mappings["hit_rate"].instances[0].points == [[1000.0, 1.0]]


@pytest.mark.asyncio
async def test_metrics_endpoint_requires_org_owner(monkeypatch):
    """The handler itself rejects members and admits Org owners and
    platform admins, independent of the router's owner-only mount."""
    with (
        patch(
            "gpustack.routes.cache_services.CacheService.one_by_id",
            AsyncMock(return_value=_service()),
        ),
        patch(
            "gpustack.routes.cache_services.Model.all_by_fields",
            AsyncMock(return_value=[]),
        ),
    ):
        with pytest.raises(ForbiddenException):
            await cache_services_route.get_cache_service_metrics(
                request=_request(),
                session=MagicMock(),
                ctx=_ctx(org_role=OrgRole.MEMBER),
                id=5,
            )

        collected = AsyncMock(
            return_value=metrics_module.CacheServiceMetricsPublic(available=True)
        )
        monkeypatch.setattr(
            cache_services_route, "collect_cache_service_metrics", collected
        )
        result = await cache_services_route.get_cache_service_metrics(
            request=_request(),
            session=MagicMock(),
            ctx=_ctx(org_role=OrgRole.OWNER),
            id=5,
        )
        assert result.available is True

        result = await cache_services_route.get_cache_service_metrics(
            request=_request(),
            session=MagicMock(),
            ctx=_ctx(is_platform_admin=True),
            id=5,
        )
        assert result.available is True
        # the window parses server-side; the declaration drives queries
        assert collected.await_count == 2


@pytest.mark.asyncio
async def test_collect_attached_buffers_both_counters(monkeypatch):
    """The two counter queries land atomically: hits arriving while
    queries fail must not leave rows claiming hits out of zero
    lookups — the failed round leaves every row empty."""

    class _HalfFailingClient(_FakeHTTPClient):
        def get(self, url, params=None, timeout=None):
            self.requests.append((url, params))
            if "queries_total" in (params or {}).get("query", ""):
                return _FakeAsyncCtx(_FakeResponse("<html>boom</html>", status=502))
            return _FakeAsyncCtx(_FakeResponse(self._payload))

    payload = {
        "status": "success",
        "data": {
            "result": [
                {
                    "metric": {"model_instance_name": "qwen-abc12"},
                    "value": [1000, "80"],
                }
            ]
        },
    }
    client = _HalfFailingClient(payload)
    _patch_prometheus(monkeypatch, client)

    attached = [
        CacheServiceAttachedMetrics(model_id=3, model_instance_name="qwen-abc12")
    ]
    result = await collect_cache_service_metrics(
        _declaration(), 5, 3600, cluster_id=1, attached=attached
    )

    entry = result.attached[0]
    assert entry.hit_tokens is None
    assert entry.queried_tokens is None
    assert entry.hit_rate is None


@pytest.mark.asyncio
async def test_metrics_endpoint_caps_worker_filter():
    with (
        patch(
            "gpustack.routes.cache_services.CacheService.one_by_id",
            AsyncMock(return_value=_service()),
        ),
        patch(
            "gpustack.routes.cache_services.Model.all_by_fields",
            AsyncMock(return_value=[]),
        ),
    ):
        with pytest.raises(BadRequestException):
            await cache_services_route.get_cache_service_metrics(
                request=_request(),
                session=MagicMock(),
                ctx=_ctx(org_role=OrgRole.OWNER),
                id=5,
                workers=",".join(f"w{i}" for i in range(101)),
            )


@pytest.mark.asyncio
async def test_metrics_endpoint_rejects_bad_window():
    with patch(
        "gpustack.routes.cache_services.CacheService.one_by_id",
        AsyncMock(return_value=_service()),
    ):
        with pytest.raises(BadRequestException):
            await cache_services_route.get_cache_service_metrics(
                request=_request(),
                session=MagicMock(),
                ctx=_ctx(org_role=OrgRole.OWNER),
                id=5,
                window="99x",
            )


@pytest.mark.asyncio
async def test_collect_attached_engine_hit_accounting(monkeypatch):
    """The database enumerates the attached instances (the row set);
    metrics only fill their numbers — an instance whose engine exports
    no counters keeps None values instead of vanishing, and the queries
    use the unified metric names the worker exposes on the scraped
    /metrics (raw vllm:* names live on the unscraped /metrics/raw)."""

    class _AttachedClient(_FakeHTTPClient):
        def get(self, url, params=None, timeout=None):
            self.requests.append((url, params))
            if url.endswith("/api/v1/query"):
                counter_value = (
                    "80" if "external_prefix_cache_hits" in params["query"] else "100"
                )
                return _FakeAsyncCtx(
                    _FakeResponse(
                        {
                            "status": "success",
                            "data": {
                                "result": [
                                    {
                                        "metric": {
                                            "model_instance_name": "qwen-abc12",
                                        },
                                        "value": [1000, counter_value],
                                    }
                                ]
                            },
                        }
                    )
                )
            return _FakeAsyncCtx(_FakeResponse(self._payload))

    payload = {
        "status": "success",
        "data": {"result": [{"metric": {}, "values": [[1000, "1"]]}]},
    }
    client = _AttachedClient(payload)
    _patch_prometheus(monkeypatch, client)

    attached = [
        CacheServiceAttachedMetrics(
            model_id=3,
            model_name="qwen",
            model_instance_name="qwen-abc12",
            worker_name="worker-1",
        ),
        # a second instance with no exported counters (e.g. SGLang, or
        # not yet scraped): the row survives with empty values
        CacheServiceAttachedMetrics(
            model_id=4, model_name="glm", model_instance_name="glm-def34"
        ),
    ]
    result = await collect_cache_service_metrics(
        _declaration(), 5, 3600, cluster_id=1, attached=attached
    )

    assert result.available is True
    assert len(result.attached) == 2
    entry = result.attached[0]
    assert entry.model_instance_name == "qwen-abc12"
    assert entry.worker_name == "worker-1"
    assert entry.hit_tokens == 80.0
    assert entry.queried_tokens == 100.0
    assert entry.hit_rate == 0.8
    silent = result.attached[1]
    assert silent.model_instance_name == "glm-def34"
    assert silent.hit_tokens is None
    assert silent.hit_rate is None
    # unified names, scoped to the cluster and the attached model ids
    instant_queries = [
        params["query"]
        for url, params in client.requests
        if url.endswith("/api/v1/query")
    ]
    assert instant_queries
    for query in instant_queries:
        # the scraped series carry the client-appended _total suffix
        assert query.startswith(
            "sum by (model_instance_name) (increase(gpustack:external_prefix_cache_"
        )
        assert "_total{" in query
        assert 'cluster_id="1"' in query
        assert 'model_id=~"3|4"' in query
