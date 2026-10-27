import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio

from gpustack.policies.base import Allocated
from gpustack.schemas.cache_services import (
    CacheServiceInstance,
    CacheServiceInstanceUpdate,
    CacheServiceStateEnum,
)
from gpustack.server import cache as server_cache
from gpustack.server import cache_service_resources, services, worker_allocated_cache
from gpustack.server.services import CacheServiceInstanceService


@pytest_asyncio.fixture(autouse=True)
async def isolated_reservations(monkeypatch):
    monkeypatch.setattr(worker_allocated_cache, "_cache_ram_lock", asyncio.Lock())
    await worker_allocated_cache.invalidate_cache_workers_ram()
    yield
    await worker_allocated_cache.invalidate_cache_workers_ram()


@pytest.mark.asyncio
async def test_worker_display_refreshes_invalidated_claims_without_mutating_cached_models(
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
    await worker_allocated_cache.invalidate_cache_workers_ram()
    second = await worker_allocated_cache.get_worker_allocated(1)
    assert (first.ram, second.ram, models.ram) == (50, 10, 10)
    assert first.vram == second.vram == {0: 20}
    assert rows.call_args.args[1] == {}


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
    assert rows.call_args.args[1] == {}


@pytest.mark.asyncio
async def test_empty_allocation_batch_does_not_query_storage(monkeypatch):
    session = AsyncMock()
    monkeypatch.setattr(worker_allocated_cache, "async_session", session)
    assert await worker_allocated_cache.get_workers_allocated([]) == {}
    session.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("ram", [0, 40])
async def test_watch_and_list_share_one_reservation_read(monkeypatch, ram):
    monkeypatch.setattr(
        worker_allocated_cache,
        "_get_model_worker_allocated",
        AsyncMock(return_value=Allocated(ram=10, vram={0: 20})),
    )

    @asynccontextmanager
    async def session():
        yield object()

    monkeypatch.setattr(worker_allocated_cache, "async_session", session)
    rows = AsyncMock(
        return_value=(
            [SimpleNamespace(worker_id=1, computed_resource_claim={"ram": ram})]
            if ram
            else []
        )
    )
    monkeypatch.setattr(CacheServiceInstance, "all_by_fields", rows)
    results = await asyncio.gather(
        *(worker_allocated_cache.get_worker_allocated(i) for i in range(1, 201))
    )
    assert results[0].ram == 10 + ram
    assert all(result.ram == 10 for result in results[1:])
    batch = await worker_allocated_cache.get_workers_allocated(range(1, 201))
    assert batch[1].ram == 10 + ram
    rows.assert_awaited_once()


@pytest.mark.asyncio
async def test_gpu_allocation_never_reads_cache_reservations(monkeypatch):
    models = Allocated(ram=10, vram={0: 20})
    monkeypatch.setattr(
        worker_allocated_cache,
        "_get_model_worker_allocated",
        AsyncMock(return_value=models),
    )
    rows = AsyncMock(side_effect=AssertionError("GPU lookup must not read RAM"))
    monkeypatch.setattr(CacheServiceInstance, "all_by_fields", rows)
    monkeypatch.setattr(worker_allocated_cache, "async_session", rows)
    for _ in range(8):
        vram = await worker_allocated_cache.get_worker_vram_allocated(1)
        assert vram == {0: 20}
        vram[0] = 99
    assert models.vram == {0: 20}
    rows.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation, before, after",
    [
        ("create", [], [(1, 40)]),
        ("update", [(1, 40)], [(1, 80)]),
        ("update", [(1, 40)], [(2, 40)]),
        ("delete", [(1, 40)], []),
    ],
)
@pytest.mark.parametrize("fail", [False, True])
async def test_instance_service_invalidates_reservations_only_after_successful_write(
    monkeypatch, operation, before, after, fail
):
    monkeypatch.setattr(
        worker_allocated_cache,
        "_get_model_worker_allocated",
        AsyncMock(return_value=Allocated(ram=10, vram={})),
    )

    @asynccontextmanager
    async def session():
        yield object()

    monkeypatch.setattr(worker_allocated_cache, "async_session", session)
    committed = before

    async def read(*args, **kwargs):
        return [
            SimpleNamespace(worker_id=worker_id, computed_resource_claim={"ram": ram})
            for worker_id, ram in committed
        ]

    rows = AsyncMock(side_effect=read)
    monkeypatch.setattr(CacheServiceInstance, "all_by_fields", rows)
    initial = await worker_allocated_cache.get_workers_allocated([1, 2])
    instance = CacheServiceInstance(
        id=3, name="cache", cache_service_id=9, worker_id=1, cluster_id=1
    )

    async def write(*args, **kwargs):
        nonlocal committed
        # Uncommitted writes must leave the shared snapshot available.
        assert await worker_allocated_cache.get_workers_allocated([1, 2]) == initial
        rows.assert_awaited_once()
        if fail:
            raise RuntimeError("commit failed")
        committed = after
        return instance

    mutation = AsyncMock(side_effect=write)
    monkeypatch.setattr(CacheServiceInstance, operation, mutation)
    transaction = object()
    service = CacheServiceInstanceService(transaction)
    if fail:
        with pytest.raises(RuntimeError, match="commit failed"):
            await getattr(service, operation)(instance)
    else:
        result = await getattr(service, operation)(instance)
        if operation == "create":
            assert result is instance
    mutation.assert_awaited_once()
    assert mutation.call_args.args[0] is transaction

    observed = await worker_allocated_cache.get_workers_allocated([1, 2])
    expected = dict(before if fail else after)
    assert {key: value.ram for key, value in observed.items()} == {
        worker_id: 10 + expected.get(worker_id, 0) for worker_id in (1, 2)
    }
    assert rows.await_count == (1 if fail else 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("remote", [False, True])
async def test_invalidation_during_read_cannot_repopulate_current_snapshot(
    monkeypatch, remote
):
    @asynccontextmanager
    async def session():
        yield object()

    monkeypatch.setattr(worker_allocated_cache, "async_session", session)
    started, release = asyncio.Event(), asyncio.Event()

    async def read(*args, **kwargs):
        if not started.is_set():
            started.set()
            await release.wait()
            return [SimpleNamespace(worker_id=1, computed_resource_claim={"ram": 40})]
        return []

    rows = AsyncMock(side_effect=read)
    monkeypatch.setattr(CacheServiceInstance, "all_by_fields", rows)
    first = asyncio.create_task(worker_allocated_cache._get_cache_workers_ram())
    await asyncio.wait_for(started.wait(), timeout=2)
    if remote:
        await server_cache._local_delete_cache(worker_allocated_cache.CACHE_RAM_KEY)
    else:
        await worker_allocated_cache.invalidate_cache_workers_ram()
    release.set()
    assert await first == {1: 40}
    assert await worker_allocated_cache._get_cache_workers_ram() == {}
    assert await worker_allocated_cache._get_cache_workers_ram() == {}
    assert rows.await_count == 2


@pytest.mark.asyncio
async def test_failed_snapshot_read_is_retried_instead_of_reporting_zero(monkeypatch):
    @asynccontextmanager
    async def session():
        yield object()

    monkeypatch.setattr(worker_allocated_cache, "async_session", session)
    rows = AsyncMock(
        side_effect=[
            RuntimeError("unavailable"),
            [SimpleNamespace(worker_id=1, computed_resource_claim={"ram": 40})],
        ]
    )
    monkeypatch.setattr(CacheServiceInstance, "all_by_fields", rows)
    with pytest.raises(RuntimeError, match="unavailable"):
        await worker_allocated_cache._get_cache_workers_ram()
    assert await worker_allocated_cache._get_cache_workers_ram() == {1: 40}
    assert rows.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "initial, patch, changes_reservation",
    [
        ({}, {"healthy": True}, False),
        ({}, {"state": CacheServiceStateEnum.UNREACHABLE}, False),
        ({}, {"ports": {"service": 8080}, "port": 8080}, False),
        ({}, {"restart_count": 2, "state_message": "restarting"}, False),
        ({}, CacheServiceInstanceUpdate(healthy=False), False),
        ({}, {"computed_resource_claim": {"ram": 40}}, False),
        ({}, {"computed_resource_claim": {"ram": 40, "other": 2}}, False),
        ({}, {"computed_resource_claim": {"ram": 80}}, True),
        ({}, {"computed_resource_claim": None}, True),
        (
            {"computed_resource_claim": None},
            {"computed_resource_claim": {"ram": 40}},
            True,
        ),
        ({}, {"worker_id": 2}, True),
        ({}, {"deleted_at": datetime(2026, 1, 1, tzinfo=timezone.utc)}, True),
        (
            {"deleted_at": datetime(2026, 1, 1, tzinfo=timezone.utc)},
            {"deleted_at": None},
            True,
        ),
        ({}, None, True),
    ],
)
@pytest.mark.parametrize("fail", [False, True])
async def test_instance_update_invalidates_only_changed_reservations(
    monkeypatch, initial, patch, changes_reservation, fail
):
    fields = dict(
        id=3,
        name="cache",
        cache_service_id=9,
        worker_id=1,
        cluster_id=1,
        computed_resource_claim={"ram": 40},
    )
    fields.update(initial)
    instance = CacheServiceInstance(**fields)
    invalidate = AsyncMock()
    monkeypatch.setattr(services, "invalidate_cache_workers_ram", invalidate)
    monkeypatch.setattr(CacheServiceInstance, "_publish_event_after_commit", Mock())

    async def commit(*args, **kwargs):
        invalidate.assert_not_awaited()
        if fail:
            raise RuntimeError("commit failed")

    save = AsyncMock(side_effect=commit)
    monkeypatch.setattr(CacheServiceInstance, "save", save)
    session = object()
    if fail:
        with pytest.raises(RuntimeError, match="commit failed"):
            await CacheServiceInstanceService(session).update(instance, patch)
    else:
        await CacheServiceInstanceService(session).update(instance, patch)
    save.assert_awaited_once_with(session, auto_commit=True)
    assert invalidate.await_count == int(changes_reservation and not fail)
