"""Shared extended-KV-cache validation on model create/update.

``validate_shared_kv_cache`` gates a model's attachment to a cache
service: the service must exist in the model's Org (cross-tenant ids read
as missing), sit in the model's cluster, and have a provider compatible
with the model's backend; local mode must not carry a service id.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gpustack.api.exceptions import BadRequestException, NotFoundException
from gpustack.routes import models as models_route
from gpustack.schemas.cache_providers import (
    CacheProvider,
    CacheProviderIntegration,
)
from gpustack.schemas.models import ExtendedKVCacheConfig, KVCacheModeEnum

OWNER_PRINCIPAL = 42


def _model_in(ext, backend=None, distributed=False, backend_version=None):
    return SimpleNamespace(
        extended_kv_cache=ext,
        backend=backend,
        distributed_inference_across_workers=distributed,
        backend_version=backend_version,
    )


def _shared_ext(cache_service_id=9):
    return ExtendedKVCacheConfig(
        enabled=True, mode=KVCacheModeEnum.SHARED, cache_service_id=cache_service_id
    )


def _service(owner_principal_id=OWNER_PRINCIPAL, cluster_id=1):
    return SimpleNamespace(
        id=9,
        deleted_at=None,
        owner_principal_id=owner_principal_id,
        cluster_id=cluster_id,
        provider_name="LMCache",
    )


def _worker(framework=None):
    devices = [SimpleNamespace(type=framework)] if framework else []
    return SimpleNamespace(status=SimpleNamespace(gpu_devices=devices))


def _patch_lookups(
    monkeypatch,
    service,
    provider_backends=("vLLM",),
    frameworks=None,
    workers=(),
    versions=None,
):
    monkeypatch.setattr(
        models_route.CacheService, "one_by_id", AsyncMock(return_value=service)
    )
    provider = CacheProvider(
        name="LMCache",
        inference_backend_integrations=[
            CacheProviderIntegration(
                backend=b, frameworks=frameworks, versions=versions
            )
            for b in provider_backends
        ],
    )
    monkeypatch.setattr(models_route, "get_cache_provider", lambda name: provider)
    monkeypatch.setattr(
        models_route.Worker, "all_by_fields", AsyncMock(return_value=list(workers))
    )


@pytest.mark.asyncio
async def test_disabled_config_passes():
    await models_route.validate_shared_kv_cache(
        MagicMock(),
        _model_in(ExtendedKVCacheConfig(enabled=False)),
        OWNER_PRINCIPAL,
        1,
    )


@pytest.mark.asyncio
async def test_shared_requires_cache_service_id():
    ext = ExtendedKVCacheConfig(enabled=True, mode=KVCacheModeEnum.SHARED)
    with pytest.raises(BadRequestException) as exc_info:
        await models_route.validate_shared_kv_cache(
            MagicMock(), _model_in(ext), OWNER_PRINCIPAL, 1
        )

    assert "cache_service_id is required" in exc_info.value.message


@pytest.mark.asyncio
async def test_local_rejects_cache_service_id():
    ext = ExtendedKVCacheConfig(
        enabled=True, mode=KVCacheModeEnum.LOCAL, cache_service_id=9
    )
    with pytest.raises(BadRequestException) as exc_info:
        await models_route.validate_shared_kv_cache(
            MagicMock(), _model_in(ext), OWNER_PRINCIPAL, 1
        )

    assert "only valid when mode is 'shared'" in exc_info.value.message


@pytest.mark.asyncio
async def test_shared_denies_cross_org_service(monkeypatch):
    _patch_lookups(monkeypatch, _service(owner_principal_id=999))

    with pytest.raises(NotFoundException):
        await models_route.validate_shared_kv_cache(
            MagicMock(), _model_in(_shared_ext()), OWNER_PRINCIPAL, 1
        )


@pytest.mark.asyncio
async def test_shared_denies_missing_service(monkeypatch):
    _patch_lookups(monkeypatch, None)

    with pytest.raises(NotFoundException):
        await models_route.validate_shared_kv_cache(
            MagicMock(), _model_in(_shared_ext()), OWNER_PRINCIPAL, 1
        )


@pytest.mark.asyncio
async def test_shared_denies_cross_cluster_service(monkeypatch):
    _patch_lookups(monkeypatch, _service(cluster_id=2))

    with pytest.raises(BadRequestException) as exc_info:
        await models_route.validate_shared_kv_cache(
            MagicMock(), _model_in(_shared_ext()), OWNER_PRINCIPAL, 1
        )

    assert "same cluster" in exc_info.value.message


@pytest.mark.asyncio
async def test_shared_denies_incompatible_backend(monkeypatch):
    _patch_lookups(monkeypatch, _service(), provider_backends=("vLLM",))

    with pytest.raises(BadRequestException) as exc_info:
        await models_route.validate_shared_kv_cache(
            MagicMock(),
            _model_in(_shared_ext(), backend="SGLang"),
            OWNER_PRINCIPAL,
            1,
        )

    assert "not compatible" in exc_info.value.message


@pytest.mark.asyncio
async def test_shared_defaults_backend_to_vllm(monkeypatch):
    """A model without an explicit backend deploys on vLLM, so
    compatibility is checked against vLLM."""
    _patch_lookups(monkeypatch, _service(), provider_backends=("vLLM",))

    await models_route.validate_shared_kv_cache(
        MagicMock(), _model_in(_shared_ext()), OWNER_PRINCIPAL, 1
    )


@pytest.mark.asyncio
async def test_shared_happy_path_vllm(monkeypatch):
    _patch_lookups(monkeypatch, _service(), provider_backends=("vLLM",))

    await models_route.validate_shared_kv_cache(
        MagicMock(),
        _model_in(_shared_ext(), backend="vLLM"),
        OWNER_PRINCIPAL,
        1,
    )


@pytest.mark.asyncio
async def test_shared_skips_cluster_check_without_effective_cluster(monkeypatch):
    """No chosen cluster means default-cluster resolution runs later; the
    cluster match can't be validated here and must not false-positive."""
    _patch_lookups(monkeypatch, _service(cluster_id=2), provider_backends=("vLLM",))

    await models_route.validate_shared_kv_cache(
        MagicMock(), _model_in(_shared_ext()), OWNER_PRINCIPAL, None
    )


@pytest.mark.asyncio
async def test_shared_allows_distributed_permission_flag(monkeypatch):
    """distributed_inference_across_workers defaults to True for vLLM and
    SGLang as a permission, not a placement plan — creation must not
    reject it; the node-local incompatibility is decided at scheduling,
    where the real placement is known (see the injection resolver)."""
    _patch_lookups(monkeypatch, _service())
    await models_route.validate_shared_kv_cache(
        MagicMock(),
        _model_in(_shared_ext(), distributed=True),
        OWNER_PRINCIPAL,
        1,
    )


@pytest.mark.asyncio
async def test_shared_rejects_cluster_without_supported_accelerator(monkeypatch):
    """A cluster whose accelerators are all outside the provider's
    framework-scoped integrations would degrade on every instance —
    caught at creation instead."""
    _patch_lookups(
        monkeypatch,
        _service(),
        frameworks=["cuda"],
        workers=[_worker("cann"), _worker()],
    )
    with pytest.raises(BadRequestException) as exc_info:
        await models_route.validate_shared_kv_cache(
            MagicMock(), _model_in(_shared_ext()), OWNER_PRINCIPAL, 1
        )

    assert "cann" in exc_info.value.message


@pytest.mark.asyncio
async def test_shared_passes_with_one_supported_accelerator(monkeypatch):
    _patch_lookups(
        monkeypatch,
        _service(),
        frameworks=["cuda"],
        workers=[_worker("cann"), _worker("cuda")],
    )
    await models_route.validate_shared_kv_cache(
        MagicMock(), _model_in(_shared_ext()), OWNER_PRINCIPAL, 1
    )


@pytest.mark.asyncio
async def test_shared_skips_accelerator_precheck_without_gpu_workers(monkeypatch):
    """Accelerator-less clusters are left to scheduling — the pre-check
    only fires when the cluster has detectable accelerators."""
    _patch_lookups(
        monkeypatch,
        _service(),
        frameworks=["cuda"],
        workers=[_worker()],
    )
    await models_route.validate_shared_kv_cache(
        MagicMock(), _model_in(_shared_ext()), OWNER_PRINCIPAL, 1
    )


@pytest.mark.asyncio
async def test_shared_rejects_pinned_version_below_floor(monkeypatch):
    """A pinned engine version below the integration's floor would crash
    on injected args the engine does not accept — rejected at creation."""
    _patch_lookups(monkeypatch, _service(), versions=">=0.25.0")
    with pytest.raises(BadRequestException) as exc_info:
        await models_route.validate_shared_kv_cache(
            MagicMock(),
            _model_in(_shared_ext(), backend_version="0.24.1"),
            OWNER_PRINCIPAL,
            1,
        )

    assert ">=0.25.0" in exc_info.value.message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "backend_version",
    [
        "0.25.0",
        "0.26.1",
        # unparseable pins fail open — enforcement must never block an
        # exotic version string
        "custom-build",
        None,
    ],
)
async def test_shared_allows_version_in_range_or_unknown(monkeypatch, backend_version):
    _patch_lookups(monkeypatch, _service(), versions=">=0.25.0")
    await models_route.validate_shared_kv_cache(
        MagicMock(),
        _model_in(_shared_ext(), backend_version=backend_version),
        OWNER_PRINCIPAL,
        1,
    )
