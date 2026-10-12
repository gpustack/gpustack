from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gpustack.policies.base import Allocated
from gpustack.server import cache_service_resources, worker_allocated_cache


@pytest.mark.asyncio
async def test_worker_display_reads_current_cache_claims_without_mutating_cached_models(
    monkeypatch,
):
    models = Allocated(ram=10, vram={0: 20})
    monkeypatch.setattr(
        worker_allocated_cache,
        "_get_model_worker_allocated",
        AsyncMock(return_value=models),
    )

    @asynccontextmanager
    async def session():
        yield object()

    monkeypatch.setattr(worker_allocated_cache, "async_session", session)
    rows = AsyncMock(
        side_effect=[
            [SimpleNamespace(worker_id=1, computed_resource_claim={"ram": 40})],
            [],
        ]
    )
    monkeypatch.setattr(
        cache_service_resources.CacheServiceInstance, "all_by_fields", rows
    )
    first = await worker_allocated_cache.get_worker_allocated(1)
    second = await worker_allocated_cache.get_worker_allocated(1)
    assert (first.ram, second.ram, models.ram) == (50, 10, 10)
    assert first.vram == second.vram == {0: 20}
    assert rows.call_args.args[1] == {"worker_id": 1}


@pytest.mark.asyncio
async def test_failed_cache_read_does_not_report_free_memory(monkeypatch):
    monkeypatch.setattr(
        cache_service_resources.CacheServiceInstance,
        "all_by_fields",
        AsyncMock(side_effect=RuntimeError("unavailable")),
    )
    with pytest.raises(RuntimeError, match="unavailable"):
        await cache_service_resources.get_cache_ram_by_worker(object(), cluster_id=2)


@pytest.mark.asyncio
async def test_batch_allocation_reads_cache_claims_once_and_preserves_model_cache(
    monkeypatch,
):
    models = Allocated(ram=10, vram={0: 20})

    async def model_allocated(worker_id):
        if worker_id == 3:
            raise RuntimeError("model allocation unavailable")
        return models

    monkeypatch.setattr(
        worker_allocated_cache, "_get_model_worker_allocated", model_allocated
    )

    @asynccontextmanager
    async def session():
        yield object()

    monkeypatch.setattr(worker_allocated_cache, "async_session", session)
    rows = AsyncMock(
        return_value=[
            SimpleNamespace(worker_id=1, computed_resource_claim={"ram": 40}),
            SimpleNamespace(worker_id=2, computed_resource_claim={"ram": 50}),
        ]
    )
    monkeypatch.setattr(
        cache_service_resources.CacheServiceInstance, "all_by_fields", rows
    )
    result = await worker_allocated_cache.get_workers_allocated([1, 2, 3])
    assert {key: value.ram for key, value in result.items()} == {1: 50, 2: 60}
    assert models.ram == 10
    assert result[1].vram == {0: 20}
    rows.assert_awaited_once()
    condition = rows.call_args.kwargs["extra_conditions"][0]
    assert set(next(iter(condition.compile().params.values()))) == {1, 2, 3}


@pytest.mark.asyncio
async def test_empty_allocation_batch_does_not_query_storage(monkeypatch):
    session = AsyncMock()
    monkeypatch.setattr(worker_allocated_cache, "async_session", session)
    assert await worker_allocated_cache.get_workers_allocated([]) == {}
    session.assert_not_called()
