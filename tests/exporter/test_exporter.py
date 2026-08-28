import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gpustack.exporter.exporter import MetricExporter, _metrics_targets
from gpustack.schemas.cache_providers import (
    CacheProvider,
    CacheProviderExternalField,
    CacheProviderL2Backend,
    CacheProviderL2Field,
)
from gpustack.schemas.cache_services import (
    CacheServiceConfig,
    CacheServiceEndpoint,
    CacheServiceL2Storage,
    CacheServiceModeEnum,
    CacheServiceStateEnum,
)
from gpustack.schemas.config import ModelInstanceProxyModeEnum
from gpustack.schemas.models import ModelInstanceStateEnum
from gpustack.schemas.workers import WorkerStateEnum


def _sample_value(metrics, metric_name):
    for metric in metrics:
        if metric.name == metric_name:
            assert len(metric.samples) == 1
            return metric.samples[0].value
    raise AssertionError(f"Metric {metric_name} not found")


def _sample_labels(metrics, metric_name):
    for metric in metrics:
        if metric.name == metric_name:
            assert len(metric.samples) == 1
            return metric.samples[0].labels
    raise AssertionError(f"Metric {metric_name} not found")


def _cluster_with_model(model):
    return SimpleNamespace(
        id=1,
        name="default",
        provider="docker",
        state="ready",
        cluster_workers=[],
        cluster_models=[model],
    )


@pytest.mark.asyncio
async def test_model_instance_restart_metrics_are_collected():
    exporter = MetricExporter(SimpleNamespace(metrics_port=10161))
    latest_restart_time = datetime(2026, 4, 17, 8, 30, tzinfo=timezone.utc)

    instance = SimpleNamespace(
        worker_id=2,
        worker_name="worker-2",
        name="qwen-1",
        state=ModelInstanceStateEnum.RUNNING,
        restart_count=3,
        last_restart_time=latest_restart_time,
    )
    model = SimpleNamespace(
        id=10,
        name="qwen",
        backend="vllm",
        backend_version="0.8.0",
        source="huggingface",
        model_source_key="Qwen/Qwen2.5-0.5B-Instruct",
        categories=[],
        replicas=1,
        ready_replicas=1,
        instances=[instance],
        extended_kv_cache=None,
    )
    cluster = SimpleNamespace(
        id=1,
        name="default",
        provider="docker",
        state="ready",
        cluster_workers=[],
        cluster_models=[model],
    )

    with (
        patch("gpustack.exporter.exporter.Cluster.all", return_value=[cluster]),
        patch(
            "gpustack.exporter.exporter.CacheService.all_by_fields",
            return_value=[],
        ),
    ):
        metrics = await exporter._collect_metrics(session=SimpleNamespace())

    assert _sample_value(metrics, "gpustack:model_instance_restart_count") == 3
    assert (
        _sample_value(metrics, "gpustack:model_instance_latest_restart_time")
        == latest_restart_time.timestamp()
    )
    assert _sample_labels(metrics, "gpustack:model_instance_restart_count") == {
        "cluster_id": "1",
        "cluster_name": "default",
        "worker_id": "2",
        "worker_name": "worker-2",
        "model_id": "10",
        "model_name": "qwen",
        "model_instance_name": "qwen-1",
    }


@pytest.mark.asyncio
async def test_model_info_metric_includes_category_label():
    exporter = MetricExporter(SimpleNamespace(metrics_port=10161))
    model = SimpleNamespace(
        id=10,
        name="embedding",
        backend="vllm",
        backend_version="0.8.0",
        source="huggingface",
        model_source_key="BAAI/bge-m3",
        categories=["embedding"],
        replicas=1,
        ready_replicas=1,
        instances=[],
        extended_kv_cache=None,
    )

    with (
        patch(
            "gpustack.exporter.exporter.Cluster.all",
            return_value=[_cluster_with_model(model)],
        ),
        patch(
            "gpustack.exporter.exporter.CacheService.all_by_fields",
            return_value=[],
        ),
    ):
        metrics = await exporter._collect_metrics(session=SimpleNamespace())

    assert _sample_labels(metrics, "gpustack:model") == {
        "cluster_id": "1",
        "cluster_name": "default",
        "model_id": "10",
        "model_name": "embedding",
        "runtime": "vllm",
        "runtime_version": "0.8.0",
        "source": "huggingface",
        "source_key": "BAAI/bge-m3",
        "category": "embedding",
    }


@pytest.mark.asyncio
async def test_model_info_metric_uses_unknown_for_uncategorized_models():
    exporter = MetricExporter(SimpleNamespace(metrics_port=10161))
    model = SimpleNamespace(
        id=10,
        name="qwen",
        backend="vllm",
        backend_version=None,
        source="huggingface",
        model_source_key="Qwen/Qwen2.5-0.5B-Instruct",
        categories=[],
        replicas=1,
        ready_replicas=1,
        instances=[],
        extended_kv_cache=None,
    )

    with (
        patch(
            "gpustack.exporter.exporter.Cluster.all",
            return_value=[_cluster_with_model(model)],
        ),
        patch(
            "gpustack.exporter.exporter.CacheService.all_by_fields",
            return_value=[],
        ),
    ):
        metrics = await exporter._collect_metrics(session=SimpleNamespace())

    labels = _sample_labels(metrics, "gpustack:model")
    assert labels["category"] == "unknown"
    assert labels["runtime_version"] == "unknown"


class _NoopSession:
    async def __aenter__(self):
        return SimpleNamespace()

    async def __aexit__(self, *exc):
        return False


@pytest.mark.asyncio
async def test_generate_metrics_cache_survives_transient_db_error(monkeypatch):
    """A transient DB error while refreshing the cache must not escape the
    loop. If it did, the exception would propagate through the server's
    asyncio.gather and take the whole process down (the #5839 restart). The
    loop should keep the last cache and retry on the next tick.
    """
    exporter = MetricExporter(SimpleNamespace(metrics_port=10162))
    exporter._cache_metrics = ["stale"]

    collect_calls = {"n": 0}

    async def _boom(session):
        collect_calls["n"] += 1
        raise TimeoutError(
            "QueuePool limit of size 30 overflow 20 reached, connection timed out"
        )

    async def _sleep_then_stop(_seconds):
        # Break the otherwise-infinite loop the way a real shutdown would.
        raise asyncio.CancelledError()

    monkeypatch.setattr(exporter, "_collect_metrics", _boom)
    monkeypatch.setattr(
        "gpustack.exporter.exporter.async_session", lambda: _NoopSession()
    )
    monkeypatch.setattr("gpustack.exporter.exporter.asyncio.sleep", _sleep_then_stop)

    # The transient error is swallowed; the loop proceeds to the sleep, where
    # our stand-in raises CancelledError to end the test. CancelledError itself
    # must propagate (clean shutdown), unlike the DB error.
    with pytest.raises(asyncio.CancelledError):
        await exporter.generate_metrics_cache()

    assert collect_calls["n"] == 1  # ran once, error swallowed, reached sleep
    assert exporter._cache_metrics == ["stale"]  # kept last cache, no crash


@pytest.mark.asyncio
async def test_cache_service_attached_model_relation_metric():
    """A model whose extended_kv_cache attaches to a shared cache service
    yields the relation series dashboards join on; local-mode and
    dangling-service models yield none."""
    from gpustack.schemas.models import ExtendedKVCacheConfig, KVCacheModeEnum

    exporter = MetricExporter(SimpleNamespace(metrics_port=10163))

    def _model(mid, name, kv_cache):
        return SimpleNamespace(
            id=mid,
            name=name,
            backend="vllm",
            backend_version="0.8.0",
            source="huggingface",
            model_source_key=name,
            categories=[],
            replicas=1,
            ready_replicas=1,
            instances=[],
            extended_kv_cache=kv_cache,
        )

    attached = _model(
        10,
        "qwen-shared",
        ExtendedKVCacheConfig(
            enabled=True, mode=KVCacheModeEnum.SHARED, cache_service_id=3
        ),
    )
    local = _model(
        11,
        "qwen-local",
        ExtendedKVCacheConfig(enabled=True, mode=KVCacheModeEnum.LOCAL),
    )
    dangling = _model(
        12,
        "qwen-dangling",
        ExtendedKVCacheConfig(
            enabled=True, mode=KVCacheModeEnum.SHARED, cache_service_id=99
        ),
    )
    cluster = SimpleNamespace(
        id=1,
        name="default",
        provider="docker",
        state="ready",
        cluster_workers=[],
        cluster_models=[attached, local, dangling],
    )
    service = SimpleNamespace(id=3, name="shared-lmcache")

    with (
        patch("gpustack.exporter.exporter.Cluster.all", return_value=[cluster]),
        patch(
            "gpustack.exporter.exporter.CacheService.all_by_fields",
            return_value=[service],
        ),
    ):
        metrics = await exporter._collect_metrics(session=SimpleNamespace())

    assert _sample_labels(metrics, "gpustack:cache_service_attached_model") == {
        "cluster_id": "1",
        "cluster_name": "default",
        "cache_service_id": "3",
        "cache_service_name": "shared-lmcache",
        "model_id": "10",
        "model_name": "qwen-shared",
    }


# ---- cache-service scrape targets ----


def _worker(**overrides):
    fields = dict(
        id=2,
        name="worker-2",
        cluster_id=1,
        cluster=SimpleNamespace(name="default"),
        state=WorkerStateEnum.READY,
        metrics_port=10151,
        advertise_address="10.0.0.5",
        ip="192.168.1.5",
        proxy_mode=ModelInstanceProxyModeEnum.DIRECT,
        deleted_at=None,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _cache_service(**overrides):
    fields = dict(
        id=3,
        name="shared-lmcache",
        provider_name="LMCache",
        provider_version=None,
        mode=CacheServiceModeEnum.MANAGED,
        state=CacheServiceStateEnum.RUNNING,
        cluster_id=1,
        worker_id=2,
        endpoint=None,
        config=None,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _instance(**overrides):
    fields = dict(
        id=31,
        cache_service_id=3,
        worker_id=2,
        state=CacheServiceStateEnum.RUNNING,
        metrics_port=40011,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _patch_target_sources(
    monkeypatch, workers, services, clusters=None, instances=None
):
    if clusters is None:
        clusters = [SimpleNamespace(id=1, name="default")]
    monkeypatch.setattr(
        "gpustack.exporter.exporter.Worker.all", AsyncMock(return_value=workers)
    )
    services_mock = AsyncMock(return_value=services)
    monkeypatch.setattr(
        "gpustack.exporter.exporter.CacheService.all_by_fields", services_mock
    )
    monkeypatch.setattr(
        "gpustack.exporter.exporter.CacheServiceInstance.all_by_fields",
        AsyncMock(return_value=instances or []),
    )
    monkeypatch.setattr(
        "gpustack.exporter.exporter.Cluster.all", AsyncMock(return_value=clusters)
    )
    return services_mock


def _cache_groups(targets):
    return [
        group
        for group in targets
        if group["labels"].get("gpustack_target_type") == "cache-service"
    ]


@pytest.mark.asyncio
async def test_managed_cache_service_yields_direct_target_per_instance(monkeypatch):
    services_mock = _patch_target_sources(
        monkeypatch,
        workers=[_worker()],
        services=[_cache_service()],
        instances=[_instance()],
    )

    targets = await _metrics_targets(session=SimpleNamespace(), is_proxy=False)

    cache_groups = _cache_groups(targets)
    assert cache_groups == [
        {
            "labels": {
                "cluster_id": "1",
                "cluster_name": "default",
                "cache_service_id": "3",
                "cache_service_name": "shared-lmcache",
                "provider": "LMCache",
                "gpustack_target_type": "cache-service",
                "worker_name": "worker-2",
                "cache_service_instance_id": "31",
            },
            "targets": ["10.0.0.5:40011"],
        }
    ]
    # Only running services are considered; the state filter lives in the
    # query, so pin it there.
    assert services_mock.await_args.kwargs["fields"] == {
        "state": CacheServiceStateEnum.RUNNING
    }

    # A direct-mode worker's cache service must not leak onto the proxy list.
    proxy_targets = await _metrics_targets(session=SimpleNamespace(), is_proxy=True)
    assert _cache_groups(proxy_targets) == []


@pytest.mark.asyncio
async def test_managed_cache_service_emits_group_per_instance(monkeypatch):
    """A per-node service is scraped on every worker that runs one of its
    instances, each group labeled with that instance's identity."""
    workers = [
        _worker(id=2, name="node-a", advertise_address="10.0.0.5"),
        _worker(id=3, name="node-b", advertise_address="10.0.0.6"),
    ]
    instances = [
        _instance(id=31, worker_id=2, metrics_port=40011),
        _instance(id=32, worker_id=3, metrics_port=40021),
    ]
    _patch_target_sources(
        monkeypatch, workers=workers, services=[_cache_service()], instances=instances
    )

    targets = await _metrics_targets(session=SimpleNamespace(), is_proxy=False)

    groups = _cache_groups(targets)
    assert [
        (
            g["labels"]["worker_name"],
            g["labels"]["cache_service_instance_id"],
            g["targets"],
        )
        for g in groups
    ] == [
        ("node-a", "31", ["10.0.0.5:40011"]),
        ("node-b", "32", ["10.0.0.6:40021"]),
    ]
    for group in groups:
        assert group["labels"]["cache_service_id"] == "3"


@pytest.mark.asyncio
async def test_managed_cache_service_instance_follows_its_workers_proxy_split(
    monkeypatch,
):
    """The direct/proxy split is decided per instance by the worker that
    runs it, so a service spanning mixed-mode workers lands on both lists."""
    workers = [
        _worker(id=2, name="direct-node"),
        _worker(
            id=3,
            name="tunnel-node",
            ip="192.168.1.6",
            proxy_mode=ModelInstanceProxyModeEnum.TUNNEL,
        ),
    ]
    instances = [
        _instance(id=31, worker_id=2, metrics_port=40011),
        _instance(id=32, worker_id=3, metrics_port=40021),
    ]
    _patch_target_sources(
        monkeypatch, workers=workers, services=[_cache_service()], instances=instances
    )

    direct = await _metrics_targets(session=SimpleNamespace(), is_proxy=False)
    proxied = await _metrics_targets(session=SimpleNamespace(), is_proxy=True)

    (direct_group,) = _cache_groups(direct)
    assert direct_group["targets"] == ["10.0.0.5:40011"]
    (proxy_group,) = _cache_groups(proxied)
    # Proxied scrapes go through the server tunnel, which addresses the
    # worker by its private IP.
    assert proxy_group["targets"] == ["192.168.1.6:40021"]
    assert proxy_group["labels"]["cache_service_instance_id"] == "32"


@pytest.mark.asyncio
async def test_external_cache_service_with_metrics_url(monkeypatch):
    service = _cache_service(
        mode=CacheServiceModeEnum.EXTERNAL,
        worker_id=None,
        metrics_port=None,
        endpoint=CacheServiceEndpoint(
            host="cache.example.com",
            port=8100,
            metrics_url="http://cache.example.com:9500/custom/metrics",
        ),
    )
    _patch_target_sources(monkeypatch, workers=[], services=[service])

    targets = await _metrics_targets(session=SimpleNamespace(), is_proxy=False)

    (group,) = _cache_groups(targets)
    assert group["targets"] == ["cache.example.com:9500"]
    assert group["labels"]["__metrics_path__"] == "/custom/metrics"

    # External services are reached from the server network directly.
    proxy_targets = await _metrics_targets(session=SimpleNamespace(), is_proxy=True)
    assert _cache_groups(proxy_targets) == []


@pytest.mark.asyncio
async def test_external_cache_service_with_host_and_metrics_port(monkeypatch):
    service = _cache_service(
        mode=CacheServiceModeEnum.EXTERNAL,
        worker_id=None,
        metrics_port=None,
        endpoint=CacheServiceEndpoint(
            host="cache.internal", port=8100, metrics_port=9500
        ),
    )
    _patch_target_sources(monkeypatch, workers=[], services=[service])

    targets = await _metrics_targets(session=SimpleNamespace(), is_proxy=False)

    (group,) = _cache_groups(targets)
    assert group["targets"] == ["cache.internal:9500"]
    # The provider's declared path is the Prometheus default, so no
    # __metrics_path__ override is emitted.
    assert "__metrics_path__" not in group["labels"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "service",
    [
        # Provider unknown to the catalog declares no metrics.
        _cache_service(provider_name="NoSuchProvider"),
        # External without any registered metrics endpoint.
        _cache_service(
            mode=CacheServiceModeEnum.EXTERNAL,
            worker_id=None,
            endpoint=CacheServiceEndpoint(host="cache.internal", port=8100),
        ),
    ],
)
async def test_uncollectable_cache_services_are_excluded(monkeypatch, service):
    _patch_target_sources(
        monkeypatch, workers=[_worker()], services=[service], instances=[_instance()]
    )

    for is_proxy in (False, True):
        targets = await _metrics_targets(session=SimpleNamespace(), is_proxy=is_proxy)
        assert _cache_groups(targets) == []


def _l2_metrics_provider() -> CacheProvider:
    """A managed provider whose L2 backend declares a metrics_target
    field — the generic mechanism a storage-partner declaration can use
    to scrape its cluster (no shipped provider uses it this version)."""
    return CacheProvider(
        name="StubCache",
        supported_modes=["managed"],
        l2_backends={
            "stub_store": CacheProviderL2Backend(
                fields=[
                    CacheProviderL2Field(name="base_path", required=True),
                    CacheProviderL2Field(name="metrics_endpoint", metrics_target=True),
                ]
            )
        },
    )


@pytest.mark.asyncio
async def test_l2_metrics_target_field_adds_direct_scrape_target(monkeypatch):
    """A metrics_target-flagged L2 backend field becomes its own scrape
    target group labeled with the backend, independent of the engine's
    per-instance targets."""
    monkeypatch.setattr(
        "gpustack.exporter.exporter.get_cache_provider",
        lambda name: _l2_metrics_provider(),
    )
    service = _cache_service(
        provider_name="StubCache",
        config=CacheServiceConfig(
            l2_storages=[
                CacheServiceL2Storage(
                    backend="stub_store",
                    params={
                        "base_path": "/mnt/cache",
                        "metrics_endpoint": "10.0.0.20:9100",
                    },
                )
            ]
        ),
    )
    _patch_target_sources(monkeypatch, workers=[], services=[service])

    targets = await _metrics_targets(session=SimpleNamespace(), is_proxy=False)

    (group,) = _cache_groups(targets)
    assert group["targets"] == ["10.0.0.20:9100"]
    assert group["labels"]["cache_l2_backend"] == "stub_store"
    assert group["labels"]["provider"] == "StubCache"
    assert "__metrics_path__" not in group["labels"]
    assert "__scheme__" not in group["labels"]

    # The storage endpoint lives outside the workers' networks: direct only.
    proxy_targets = await _metrics_targets(session=SimpleNamespace(), is_proxy=True)
    assert _cache_groups(proxy_targets) == []


@pytest.mark.asyncio
async def test_l2_metrics_target_accepts_full_url(monkeypatch):
    monkeypatch.setattr(
        "gpustack.exporter.exporter.get_cache_provider",
        lambda name: _l2_metrics_provider(),
    )
    service = _cache_service(
        provider_name="StubCache",
        config=CacheServiceConfig(
            l2_storages=[
                CacheServiceL2Storage(
                    backend="stub_store",
                    params={
                        "base_path": "/mnt/cache",
                        "metrics_endpoint": "https://store.example.com:9443/store/metrics",
                    },
                )
            ]
        ),
    )
    _patch_target_sources(monkeypatch, workers=[], services=[service])

    targets = await _metrics_targets(session=SimpleNamespace(), is_proxy=False)

    (group,) = _cache_groups(targets)
    assert group["targets"] == ["store.example.com:9443"]
    assert group["labels"]["__metrics_path__"] == "/store/metrics"
    assert group["labels"]["__scheme__"] == "https"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field_value", "expected_targets"),
    [
        ("10.1.1.5:9100", [["10.1.1.5:9100"]]),
        # A blank value yields no target rather than a broken one.
        ("   ", []),
        # Prometheus cannot scrape non-HTTP schemes; drop them.
        ("redis://10.1.1.5:6379", []),
    ],
)
async def test_external_fields_metrics_target_field(
    monkeypatch, field_value, expected_targets
):
    """An external_fields field flagged metrics_target adds a scrape
    target even when the provider declares no engine metrics of its own."""
    provider = CacheProvider(
        name="StubCache",
        supported_modes=["external"],
        external_fields=[
            CacheProviderExternalField(name="metrics_endpoint", metrics_target=True)
        ],
    )
    monkeypatch.setattr(
        "gpustack.exporter.exporter.get_cache_provider", lambda name: provider
    )
    service = _cache_service(
        provider_name="StubCache",
        mode=CacheServiceModeEnum.EXTERNAL,
        worker_id=None,
        endpoint=CacheServiceEndpoint(
            host="cache.internal",
            port=8100,
            params={"metrics_endpoint": field_value},
        ),
    )
    _patch_target_sources(monkeypatch, workers=[], services=[service])

    targets = await _metrics_targets(session=SimpleNamespace(), is_proxy=False)

    assert [group["targets"] for group in _cache_groups(targets)] == expected_targets


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "instances",
    [
        # No running instances, nothing to scrape.
        [],
        # Instance without an allocated metrics port cannot be scraped.
        [_instance(metrics_port=None)],
        # Instance whose worker is gone has no scrape address.
        [_instance(worker_id=99)],
    ],
)
async def test_uncollectable_managed_instances_are_excluded(monkeypatch, instances):
    _patch_target_sources(
        monkeypatch,
        workers=[_worker()],
        services=[_cache_service()],
        instances=instances,
    )

    for is_proxy in (False, True):
        targets = await _metrics_targets(session=SimpleNamespace(), is_proxy=is_proxy)
        assert _cache_groups(targets) == []
