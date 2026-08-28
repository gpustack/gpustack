from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import gpustack.routes.benchmarks as benchmarks_route
from gpustack.schemas.models import (
    ExtendedKVCacheConfig,
    KVCacheModeEnum,
    Model,
    ModelInstance,
    ModelInstanceStateEnum,
)


def _patch_worker_side(monkeypatch):
    """The worker/GPU half of the snapshot is not under test here."""
    monkeypatch.setattr(
        benchmarks_route,
        "WorkerService",
        lambda session: MagicMock(get_by_id=AsyncMock(return_value=None)),
    )
    monkeypatch.setattr(
        benchmarks_route, "create_worker_snapshot", lambda *args: (None, None)
    )


def _instance() -> ModelInstance:
    return ModelInstance(
        id=11,
        name="mi-1",
        worker_id=3,
        worker_name="w1",
        state=ModelInstanceStateEnum.RUNNING,
    )


def _model(extended_kv_cache=None) -> Model:
    return Model(name="m1", extended_kv_cache=extended_kv_cache)


@pytest.mark.asyncio
async def test_snapshot_names_the_attached_shared_cache_service(monkeypatch):
    _patch_worker_side(monkeypatch)
    one_by_id = AsyncMock(return_value=SimpleNamespace(name="svc-cache"))
    monkeypatch.setattr(benchmarks_route.CacheService, "one_by_id", one_by_id)

    snapshot = await benchmarks_route.get_benchmark_snapshot(
        session=MagicMock(),
        mi=_instance(),
        model=_model(
            ExtendedKVCacheConfig(
                enabled=True, mode=KVCacheModeEnum.SHARED, cache_service_id=7
            )
        ),
    )

    assert snapshot.instances["mi-1"].cache_service_name == "svc-cache"
    assert one_by_id.await_args.args[1] == 7


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extended_kv_cache",
    [
        None,
        ExtendedKVCacheConfig(enabled=True, mode=KVCacheModeEnum.LOCAL),
    ],
)
async def test_snapshot_skips_the_lookup_without_a_shared_cache(
    monkeypatch, extended_kv_cache
):
    _patch_worker_side(monkeypatch)
    one_by_id = AsyncMock()
    monkeypatch.setattr(benchmarks_route.CacheService, "one_by_id", one_by_id)

    snapshot = await benchmarks_route.get_benchmark_snapshot(
        session=MagicMock(),
        mi=_instance(),
        model=_model(extended_kv_cache),
    )

    assert snapshot.instances["mi-1"].cache_service_name is None
    one_by_id.assert_not_awaited()


@pytest.mark.asyncio
async def test_snapshot_tolerates_a_deleted_cache_service(monkeypatch):
    _patch_worker_side(monkeypatch)
    monkeypatch.setattr(
        benchmarks_route.CacheService, "one_by_id", AsyncMock(return_value=None)
    )

    snapshot = await benchmarks_route.get_benchmark_snapshot(
        session=MagicMock(),
        mi=_instance(),
        model=_model(
            ExtendedKVCacheConfig(
                enabled=True, mode=KVCacheModeEnum.SHARED, cache_service_id=7
            )
        ),
    )

    assert snapshot.instances["mi-1"].cache_service_name is None
