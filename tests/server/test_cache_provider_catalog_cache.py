"""Catalog polling shares committed snapshots without hiding catalog rewrites."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio

from gpustack.schemas import cache_provider_source as source
from gpustack.server import cache as server_cache
from gpustack.server import cache_provider_catalog as catalog


def _entry(name="Demo", version="v1", position=0):
    return source.CacheProviderEntry(
        name=name,
        position=position,
        payload={
            "name": name,
            "display_name": name,
            "topology": "per_node",
            "default_version": version,
            "default_image": f"demo:{version}",
            "versions": {version: {}},
            "default_run_command": "demo --port {{port}}",
        },
    )


@pytest_asyncio.fixture
async def reader(monkeypatch):
    await server_cache._local_delete_cache(catalog.CATALOG_CACHE_KEY)
    monkeypatch.setattr(catalog, "_catalog_lock", asyncio.Lock())
    monkeypatch.setattr(catalog, "_catalog_revision", None)
    monkeypatch.setattr(catalog, "_catalog", [])
    monkeypatch.setattr(server_cache, "_coordinator", None)
    session = SimpleNamespace(in_transaction=lambda: False)

    @asynccontextmanager
    async def open_session():
        yield session

    monkeypatch.setattr(catalog, "async_session", open_session)
    query = AsyncMock(return_value=[_entry()])
    monkeypatch.setattr(source.CacheProviderEntry, "all", query)
    yield SimpleNamespace(query=query, session=session)
    await server_cache._local_delete_cache(catalog.CATALOG_CACHE_KEY)


@pytest.mark.asyncio
async def test_concurrent_worker_polls_and_name_lookups_share_one_query(reader):
    reader.query.return_value = [_entry("Second", position=1), _entry("First")]
    results = await asyncio.gather(
        *(
            catalog.get_cache_providers(SimpleNamespace(in_transaction=lambda: False))
            for _ in range(200)
        )
    )
    assert all([p.name for p in result] == ["First", "Second"] for result in results)
    assert (await catalog.get_cache_provider(reader.session, "FIRST")).name == "First"
    assert await catalog.get_cache_provider(reader.session, "missing") is None
    reader.query.assert_awaited_once_with(reader.session)


@pytest.mark.asyncio
async def test_callers_cannot_mutate_the_shared_catalog(reader):
    providers = await catalog.get_cache_providers(reader.session)
    providers[0].versions.clear()
    providers.clear()
    provider = await catalog.get_cache_provider(reader.session, "demo")
    assert "v1" in provider.versions
    provider.versions.clear()
    assert "v1" in (await catalog.get_cache_providers(reader.session))[0].versions
    reader.query.assert_awaited_once()


@pytest.mark.asyncio
async def test_empty_catalog_is_cached(reader):
    reader.query.return_value = []
    for _ in range(2):
        assert await catalog.get_cache_providers(reader.session) == []
        assert await catalog.get_cache_provider(reader.session, "demo") is None
    reader.query.assert_awaited_once()


@pytest.mark.asyncio
async def test_failed_read_is_retried(reader):
    reader.query.side_effect = [RuntimeError("unavailable"), [_entry()]]
    with pytest.raises(RuntimeError, match="unavailable"):
        await catalog.get_cache_providers(reader.session)
    assert (await catalog.get_cache_provider(reader.session, "demo")).name == "Demo"
    assert reader.query.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("remote", [False, True])
async def test_invalidation_during_read_forces_the_next_read_to_reload(reader, remote):
    started, release = asyncio.Event(), asyncio.Event()

    async def query(session):
        if reader.query.await_count == 1:
            started.set()
            await release.wait()
            return [_entry(version="v1")]
        return [_entry(version="v2")]

    reader.query.side_effect = query
    task = asyncio.create_task(catalog.get_cache_providers(reader.session))
    await started.wait()
    try:
        if remote:
            await server_cache._local_delete_cache(catalog.CATALOG_CACHE_KEY)
        else:
            await catalog.invalidate_cache_provider_catalog()
    finally:
        release.set()
        await task
    assert (
        await catalog.get_cache_provider(reader.session, "demo")
    ).default_version == "v2"
    assert reader.query.await_count == 2


@pytest.mark.asyncio
async def test_expired_catalog_reloads(reader):
    await catalog.get_cache_providers(reader.session)
    await server_cache.cache.expire(catalog.CATALOG_CACHE_KEY, 0.001)
    await asyncio.sleep(0.01)
    reader.query.return_value = [_entry(version="v2")]
    assert (
        await catalog.get_cache_provider(reader.session, "demo")
    ).default_version == "v2"
    assert reader.query.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("commit_fails", [False, True])
async def test_reconcile_invalidates_only_after_commit(
    reader, monkeypatch, empty, commit_fails
):
    await catalog.get_cache_providers(reader.session)
    writer = SimpleNamespace(commit=AsyncMock())
    row = SimpleNamespace(name="Demo", update=AsyncMock(), delete=AsyncMock())
    desired = [] if empty else [_entry(version="v2"), _entry("Added", position=1)]
    monkeypatch.setattr(
        source, "build_cache_provider_entries", Mock(return_value=desired)
    )
    monkeypatch.setattr(source.RunnerOverrideEntry, "all", AsyncMock(return_value=[]))
    create = AsyncMock()
    monkeypatch.setattr(source.CacheProviderEntry, "create", create)
    reader.query.side_effect = lambda session: [row] if session is writer else desired
    coordinator = SimpleNamespace(publish=AsyncMock())
    monkeypatch.setattr(server_cache, "_coordinator", coordinator)

    async def commit():
        assert (
            await catalog.get_cache_provider(reader.session, "demo")
        ).default_version == "v1"
        coordinator.publish.assert_not_awaited()
        if commit_fails:
            raise RuntimeError("commit failed")

    writer.commit.side_effect = commit
    if commit_fails:
        with pytest.raises(RuntimeError, match="commit failed"):
            await source.reconcile_cache_providers(writer, [])
        assert (
            await catalog.get_cache_provider(reader.session, "demo")
        ).default_version == "v1"
        coordinator.publish.assert_not_awaited()
    else:
        await source.reconcile_cache_providers(writer, [])
        providers = await catalog.get_cache_providers(reader.session)
        if empty:
            assert providers == []
            row.delete.assert_awaited_once_with(writer, auto_commit=False)
        else:
            assert [p.name for p in providers] == ["Demo", "Added"]
            assert providers[0].default_version == "v2"
            row.update.assert_awaited_once()
            create.assert_awaited_once_with(writer, desired[1], auto_commit=False)
        coordinator.publish.assert_awaited_once()
        topic, event = coordinator.publish.call_args.args
        assert topic == "cache"
        assert event.data == {"key": catalog.CATALOG_CACHE_KEY}


@pytest.mark.asyncio
@pytest.mark.parametrize("pool_size", [1, 2])
@pytest.mark.parametrize("by_name", [False, True])
async def test_pool_holders_do_not_wait_for_a_catalog_loader(
    reader, monkeypatch, pool_size, by_name
):
    pool = asyncio.Semaphore(pool_size)
    loading = asyncio.Event()
    for _ in range(pool_size):
        await pool.acquire()

    @asynccontextmanager
    async def open_session():
        loading.set()
        async with pool:
            yield reader.session

    monkeypatch.setattr(catalog, "async_session", open_session)
    reading_holders = 0
    all_holders_reading = asyncio.Event()

    async def query(session):
        nonlocal reading_holders
        if session is reader.session:
            return [_entry(version="v2")]
        reading_holders += 1
        if reading_holders == pool_size:
            all_holders_reading.set()
        await all_holders_reading.wait()
        return [_entry(version="transaction-only")]

    reader.query.side_effect = query
    loader = asyncio.create_task(catalog.get_cache_providers(reader.session))
    await loading.wait()

    async def read_and_release():
        session = SimpleNamespace(in_transaction=lambda: True)
        try:
            if by_name:
                result = await catalog.get_cache_provider(session, "Demo")
            else:
                result = (await catalog.get_cache_providers(session))[0]
            assert result.default_version == "transaction-only"
        finally:
            pool.release()

    try:
        await asyncio.wait_for(
            asyncio.gather(*(read_and_release() for _ in range(pool_size))), 1
        )
    finally:
        await asyncio.wait_for(loader, 1)
    assert (
        await catalog.get_cache_provider(reader.session, "Demo")
    ).default_version == "v2"
    assert reader.query.await_count == pool_size + 1


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["older-snapshot", "uncommitted", None])
@pytest.mark.parametrize("by_name", [False, True])
async def test_warm_cache_preserves_the_callers_transaction_view(
    reader, monkeypatch, version, by_name
):
    assert (await catalog.get_cache_providers(reader.session))[
        0
    ].default_version == "v1"
    transaction = SimpleNamespace(in_transaction=lambda: True)
    reader.query.return_value = [_entry(version=version)] if version else []
    fresh_session = Mock(side_effect=AssertionError("must reuse the caller's session"))
    monkeypatch.setattr(catalog, "async_session", fresh_session)

    if by_name:
        provider = await catalog.get_cache_provider(transaction, "Demo")
        assert (provider.default_version if provider else None) == version
    else:
        providers = await catalog.get_cache_providers(transaction)
        assert [p.default_version for p in providers] == ([version] if version else [])
    reader.query.assert_awaited_with(transaction)
    assert reader.query.await_count == 2
    fresh_session.assert_not_called()

    assert (await catalog.get_cache_providers(reader.session))[
        0
    ].default_version == "v1"
    assert reader.query.await_count == 2
