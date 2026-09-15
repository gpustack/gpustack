from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gpustack.schemas.cache_providers import (
    CacheProvider,
    CacheProviderHealthCheck,
)
from gpustack.schemas.cache_services import (
    CacheService,
    CacheServiceConfig,
    CacheServiceEndpoint,
    CacheServiceInstance,
    CacheServiceStateEnum,
)
from gpustack.schemas.models import ExtendedKVCacheConfig, KVCacheModeEnum
from gpustack.server.cache_services import (
    resolve_instance_cache_config,
    resolve_instance_cache_config_safe,
)
from tests.utils.model import new_model


def shared_cache_model(cache_service_id=5, chunk_size=None):
    return new_model(
        1,
        "test-model",
        huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct",
        extended_kv_cache=ExtendedKVCacheConfig(
            enabled=True,
            mode=KVCacheModeEnum.SHARED,
            cache_service_id=cache_service_id,
            chunk_size=chunk_size,
        ),
    )


def managed_cache_service(**overrides):
    fields = dict(
        id=5,
        name="lmcache-svc",
        provider_name="LMCache",
        provider_version="v0.5.2",
        cluster_id=1,
        worker_id=2,
        state=CacheServiceStateEnum.RUNNING,
        config=CacheServiceConfig(fields={"ram_size": 8, "chunk_size": 256}),
    )
    fields.update(overrides)
    return CacheService(**fields)


def cache_service_instance(**overrides):
    fields = dict(
        id=11,
        name="lmcache-svc-a1b2c",
        cache_service_id=5,
        worker_id=2,
        cluster_id=1,
        port=9000,
        # LMCache's cache servers are its "server" component; engines
        # attach to that one.
        component="server",
        state=CacheServiceStateEnum.RUNNING,
    )
    fields.update(overrides)
    return CacheServiceInstance(**fields)


@contextmanager
def patch_lookups(service, worker=..., instances=...):
    """Back the service / cache-instance-worker / instance lookups the
    resolver performs. ``worker`` is the worker row returned for a cache
    instance's worker_id; ``instances`` are the managed service's
    CacheServiceInstance rows."""
    if worker is ...:
        worker = SimpleNamespace(id=2, ip="10.0.0.5", deleted_at=None)
    if instances is ...:
        instances = [cache_service_instance()]
    with (
        patch(
            "gpustack.server.cache_services.CacheService.one_by_id",
            AsyncMock(return_value=service),
        ),
        patch(
            "gpustack.server.cache_services.Worker.one_by_id",
            AsyncMock(return_value=worker),
        ),
        patch(
            "gpustack.server.cache_services.CacheServiceInstance.all_by_fields",
            AsyncMock(return_value=instances),
        ),
    ):
        yield


@pytest.mark.asyncio
async def test_resolve_returns_none_without_extended_kv_cache():
    model = new_model(1, "m", huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct")
    assert await resolve_instance_cache_config(MagicMock(), model) is None


@pytest.mark.asyncio
async def test_resolve_returns_none_for_local_mode():
    model = new_model(
        1,
        "m",
        huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct",
        extended_kv_cache=ExtendedKVCacheConfig(
            enabled=True, mode=KVCacheModeEnum.LOCAL
        ),
    )
    assert await resolve_instance_cache_config(MagicMock(), model) is None


@pytest.mark.asyncio
async def test_resolve_injects_for_running_managed_instance():
    model = shared_cache_model()
    with patch_lookups(managed_cache_service()):
        snapshot = await resolve_instance_cache_config(
            MagicMock(),
            model,
            worker=SimpleNamespace(id=2, ip="10.0.0.5", deleted_at=None),
        )

    assert snapshot is not None
    assert snapshot.injected is True
    assert snapshot.reason is None
    assert snapshot.cache_service_id == 5
    assert snapshot.cache_service_name == "lmcache-svc"
    assert snapshot.provider_name == "LMCache"
    assert snapshot.provider_version == "v0.5.2"
    assert snapshot.endpoint == CacheServiceEndpoint(
        host="10.0.0.5", port=9000, params={"locality": "node_local"}
    )
    # A pinned hash seed keeps chunk keys consistent across engine
    # processes on the builtin-hash fallback path — without it,
    # cross-instance sharing silently never hits.
    assert snapshot.env == {"PYTHONHASHSEED": "0"}
    assert snapshot.args[0] == "--kv-transfer-config"
    assert '"lmcache.mp.host":"tcp://10.0.0.5"' in snapshot.args[1]
    assert '"lmcache.mp.port":9000' in snapshot.args[1]


@pytest.mark.asyncio
async def test_resolve_per_node_pending_before_scheduling():
    """per_node attaches node-local only, so a pre-scheduling resolve
    (no worker yet) yields an explicit pending snapshot; the scheduler
    re-resolves once the instance has a worker."""
    model = shared_cache_model()
    with patch_lookups(managed_cache_service()):
        snapshot = await resolve_instance_cache_config(MagicMock(), model)

    assert snapshot.injected is False
    assert "resolves with the instance's worker" in snapshot.reason


@pytest.mark.asyncio
async def test_resolve_managed_ignores_lagging_aggregate_state():
    """The managed decision rides on instances, not the service-level
    aggregate: a RUNNING instance serves even while the aggregate lags."""
    model = shared_cache_model()
    service = managed_cache_service(state=CacheServiceStateEnum.PENDING)
    with patch_lookups(service):
        snapshot = await resolve_instance_cache_config(
            MagicMock(),
            model,
            worker=SimpleNamespace(id=2, ip="10.0.0.5", deleted_at=None),
        )

    assert snapshot.injected is True
    assert snapshot.endpoint == CacheServiceEndpoint(
        host="10.0.0.5", port=9000, params={"locality": "node_local"}
    )


@pytest.mark.asyncio
async def test_resolve_degrades_without_running_instance():
    model = shared_cache_model()
    instances = [
        cache_service_instance(state=CacheServiceStateEnum.PENDING),
        cache_service_instance(id=12, worker_id=3, state=CacheServiceStateEnum.ERROR),
    ]
    with patch_lookups(managed_cache_service(), instances=instances):
        snapshot = await resolve_instance_cache_config(MagicMock(), model)

    assert snapshot.injected is False
    assert "no running instance" in snapshot.reason
    assert snapshot.cache_service_name == "lmcache-svc"


@pytest.mark.asyncio
async def test_resolve_prefers_instance_on_model_worker():
    """A per-node deployment should keep the engine attached to the cache
    server on its own node when that one is RUNNING."""
    model = shared_cache_model()
    instances = [
        cache_service_instance(id=11, worker_id=2, port=9000),
        cache_service_instance(id=12, worker_id=3, port=9001),
    ]
    workers = {
        2: SimpleNamespace(id=2, ip="10.0.0.5", deleted_at=None),
        3: SimpleNamespace(id=3, ip="10.0.0.6", deleted_at=None),
    }
    model_worker = SimpleNamespace(id=3, ip="10.0.0.6", deleted_at=None)
    with (
        patch(
            "gpustack.server.cache_services.CacheService.one_by_id",
            AsyncMock(return_value=managed_cache_service()),
        ),
        patch(
            "gpustack.server.cache_services.Worker.one_by_id",
            AsyncMock(side_effect=lambda session, id: workers.get(id)),
        ),
        patch(
            "gpustack.server.cache_services.CacheServiceInstance.all_by_fields",
            AsyncMock(return_value=instances),
        ),
    ):
        snapshot = await resolve_instance_cache_config(
            MagicMock(), model, worker=model_worker
        )

    assert snapshot.injected is True
    assert snapshot.endpoint == CacheServiceEndpoint(
        host="10.0.0.6", port=9001, params={"locality": "node_local"}
    )
    # Same-node attach may negotiate the CUDA-IPC zero-copy path.
    assert any('"lmcache.mp.mp_transfer_mode":"auto"' in arg for arg in snapshot.args)


@pytest.mark.asyncio
async def test_resolve_passes_worker_framework_to_injection():
    """The engine worker's accelerator framework reaches the injection
    lookup, so a provider can scope an integration entry per framework
    (e.g. a cann-specific vLLM contract)."""
    model = shared_cache_model()
    model_worker = SimpleNamespace(
        id=2,
        ip="10.0.0.5",
        deleted_at=None,
        status=SimpleNamespace(gpu_devices=[SimpleNamespace(type="cann")]),
    )
    captured = {}

    def fake_render_injection(provider, backend, params, framework=None):
        captured["framework"] = framework
        return {}, [], {}

    with (
        patch_lookups(managed_cache_service()),
        patch(
            "gpustack.server.cache_services.render_injection",
            side_effect=fake_render_injection,
        ),
    ):
        snapshot = await resolve_instance_cache_config(
            MagicMock(), model, worker=model_worker
        )

    assert snapshot.injected is True
    assert captured["framework"] == "cann"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "backend_parameters",
    [
        # every legal backend_parameters spelling of the same override —
        # the contract is a semantically concatenated argv, so detection
        # must flatten exactly like the worker does
        ["--kv-transfer-config", '{"kv_connector":"MyConnector"}'],
        ['--kv-transfer-config={"kv_connector":"MyConnector"}'],
        ['--kv-transfer-config {"kv_connector":"MyConnector"}'],
        ['--max-model-len 8192 --kv-transfer-config {"kv_connector":"MyConnector"}'],
    ],
)
async def test_resolve_marks_user_kv_transfer_override_as_takeover(
    backend_parameters,
):
    """A user-supplied connector-slot parameter is a deliberate escape
    hatch (user args win the single-value flag), but never a silent one:
    the snapshot degrades with a takeover reason and none of the
    injection applies — the user owns the whole connector wiring."""
    model = shared_cache_model()
    model.backend_parameters = backend_parameters
    with patch_lookups(managed_cache_service()):
        snapshot = await resolve_instance_cache_config(
            MagicMock(),
            model,
            worker=SimpleNamespace(id=2, ip="10.0.0.5", deleted_at=None),
        )

    assert snapshot.injected is False
    assert "takes over the KV connector" in snapshot.reason
    assert not snapshot.args
    assert not snapshot.env


@pytest.mark.asyncio
async def test_resolve_degrades_when_engine_version_below_floor():
    """Existing models bypass creation-time validation, so the resolver
    re-checks the integration's version floor: an engine below it would
    crash on injected args (e.g. --shutdown-timeout) — degrading keeps
    it running, just without the cache."""
    model = shared_cache_model()
    model.backend_version = "0.24.1"
    with patch_lookups(managed_cache_service()):
        snapshot = await resolve_instance_cache_config(
            MagicMock(),
            model,
            worker=SimpleNamespace(id=2, ip="10.0.0.5", deleted_at=None),
        )

    assert snapshot.injected is False
    assert "outside the cache provider's supported" in snapshot.reason
    assert not snapshot.args


@pytest.mark.asyncio
async def test_resolve_degrades_when_instance_spans_workers():
    """The distributed permission flag rides on most vLLM/SGLang models;
    only an instance actually placed across workers conflicts with the
    per_node providers' node-local contract — decided here with the real
    placement, and degrading keeps the engine running without the
    cache."""
    model = shared_cache_model()
    with patch_lookups(managed_cache_service()):
        snapshot = await resolve_instance_cache_config(
            MagicMock(),
            model,
            worker=SimpleNamespace(id=2, ip="10.0.0.5", deleted_at=None),
            spans_workers=True,
        )

    assert snapshot.injected is False
    assert "spans multiple workers" in snapshot.reason
    assert not snapshot.args


@pytest.mark.asyncio
async def test_resolve_per_node_degrades_without_node_local_instance():
    """per_node never attaches across nodes: the engine-driven copy path
    measures slower than running without the cache, and a silent
    fallback would funnel every uncovered engine onto one instance —
    so a worker without a RUNNING cache instance degrades explicitly."""
    model = shared_cache_model()
    instances = [
        cache_service_instance(id=11, worker_id=2, state=CacheServiceStateEnum.ERROR),
        cache_service_instance(id=12, worker_id=3, port=9001),
    ]
    with patch_lookups(
        managed_cache_service(),
        worker=SimpleNamespace(id=3, ip="10.0.0.6", deleted_at=None),
        instances=instances,
    ):
        snapshot = await resolve_instance_cache_config(
            MagicMock(),
            model,
            worker=SimpleNamespace(id=2, ip="10.0.0.5", deleted_at=None),
        )

    assert snapshot.injected is False
    assert "No running cache instance on worker" in snapshot.reason


@pytest.mark.asyncio
async def test_resolve_degrades_when_service_missing():
    model = shared_cache_model()
    with patch_lookups(None):
        snapshot = await resolve_instance_cache_config(MagicMock(), model)

    assert snapshot.injected is False
    assert snapshot.cache_service_id == 5
    assert "not found" in snapshot.reason


@pytest.mark.asyncio
async def test_resolve_degrades_when_instance_worker_missing():
    model = shared_cache_model()
    with patch_lookups(managed_cache_service(), worker=None):
        snapshot = await resolve_instance_cache_config(MagicMock(), model)

    assert snapshot.injected is False
    assert "worker" in snapshot.reason


def cluster_attach_provider() -> CacheProvider:
    """A pool engines reach over the network: its endpoint is one address
    for the whole cluster, not one per node."""
    return CacheProvider(
        name="Pool",
        attach_locality="cluster",
        default_image="repo/pool:v1",
        versions={"v1.0": {}},
        inference_backend_integrations=[
            {
                "backend": "vLLM",
                "injection": {
                    "kv_transfer_config": {
                        "kv_connector": "PoolStoreConnector",
                        "kv_connector_extra_config": {
                            "master_server_address": "{{host}}:{{port}}"
                        },
                    }
                },
            }
        ],
    )


@pytest.mark.asyncio
async def test_resolve_cluster_attach_provider_serves_spanning_instances():
    """Node-locality is the per_node providers' contract, not a
    shared-cache property: a pool engines reach over the network serves
    multi-worker instances by design — every subordinate worker's engine
    reaches the same endpoint."""
    model = shared_cache_model()
    instance_worker = SimpleNamespace(id=7, ip="10.0.0.7", deleted_at=None)
    pool_worker = SimpleNamespace(id=9, ip="10.0.0.9", deleted_at=None)
    with (
        patch(
            "gpustack.server.cache_services.get_cache_provider",
            return_value=cluster_attach_provider(),
        ),
        patch_lookups(
            managed_cache_service(provider_name="Pool", provider_version="v1.0"),
            worker=pool_worker,
            instances=[cache_service_instance(worker_id=9, port=50051, component="")],
        ),
    ):
        snapshot = await resolve_instance_cache_config(
            MagicMock(),
            model,
            worker=instance_worker,
            spans_workers=True,
        )

    assert snapshot.injected is True
    assert '"kv_connector":"PoolStoreConnector"' in snapshot.args[1]
    assert '"master_server_address":"10.0.0.9:50051"' in snapshot.args[1]


@pytest.mark.asyncio
async def test_resolve_safe_degrades_on_unexpected_error():
    model = shared_cache_model()
    with patch(
        "gpustack.server.cache_services.CacheService.one_by_id",
        AsyncMock(side_effect=RuntimeError("db down")),
    ):
        snapshot = await resolve_instance_cache_config_safe(MagicMock(), model)

    assert snapshot is not None
    assert snapshot.injected is False
    assert snapshot.cache_service_id == 5
    assert "db down" in snapshot.reason


@pytest.mark.asyncio
async def test_resolve_safe_returns_none_without_shared_cache():
    model = new_model(1, "m", huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct")
    assert await resolve_instance_cache_config_safe(MagicMock(), model) is None


@pytest.mark.asyncio
async def test_resolve_degrades_for_unknown_provider():
    model = shared_cache_model()
    service = managed_cache_service(provider_name="no-such-provider")
    with patch_lookups(service):
        snapshot = await resolve_instance_cache_config(MagicMock(), model)

    assert snapshot.injected is False
    assert "no-such-provider" in snapshot.reason


@pytest.mark.asyncio
async def test_resolve_chunk_size_is_service_scoped():
    # extended_kv_cache.chunk_size is in-process vocabulary; in shared
    # mode the service value is the single source, so the engine always
    # chunks the way the cache server does — a deployment-side value
    # (e.g. residue from switching modes) must not leak in.
    model = shared_cache_model(chunk_size=512)
    with patch_lookups(managed_cache_service()):
        snapshot = await resolve_instance_cache_config(MagicMock(), model)

    assert snapshot.chunk_size == 256

    model = shared_cache_model(chunk_size=None)
    with patch_lookups(managed_cache_service()):
        snapshot = await resolve_instance_cache_config(MagicMock(), model)

    assert snapshot.chunk_size == 256


def tcp_provider():
    return CacheProvider(
        custom_version=True,
        name="tcp-provider",
        health_check=CacheProviderHealthCheck(scheme="tcp"),
    )


def http_provider(path=None, target="port"):
    return CacheProvider(
        custom_version=True,
        name="http-provider",
        health_check=CacheProviderHealthCheck(scheme="http", path=path, target=target),
    )
