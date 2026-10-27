"""Cache deletion uses the shared transactional cascade and instance events."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.orm.attributes import set_committed_value

from gpustack.mixins.active_record import send_post_commit_events
from gpustack.schemas.cache_services import CacheService, CacheServiceInstance
from gpustack.server.bus import EventType, event_bus
from gpustack.server import services


@pytest.mark.asyncio
async def test_service_cascade_emits_instance_deletions_only_after_commit(monkeypatch):
    instances = [
        CacheServiceInstance(
            id=index,
            name=f"cache-{index}",
            cache_service_id=9,
            worker_id=index,
            cluster_id=1,
        )
        for index in (1, 2)
    ]
    service = CacheService(
        id=9,
        name="cache",
        provider_name="mooncake",
        cluster_id=1,
    )
    # selectinload hydrates the collection without loading each instance's
    # reverse relationship, which is configured as noload.
    set_committed_value(service, "instances", instances)
    published = AsyncMock()
    delivered = asyncio.Event()

    async def on_publish(name, event):
        if published.await_count == 3:
            delivered.set()

    published.side_effect = on_publish
    monkeypatch.setattr(event_bus, "publish", published)
    monkeypatch.setattr(CacheService, "save", AsyncMock())
    monkeypatch.setattr(CacheService, "_invalidate_cached_all", AsyncMock())
    session = SimpleNamespace(info={}, refresh=AsyncMock(), delete=AsyncMock())
    invalidate = AsyncMock()
    monkeypatch.setattr(services, "invalidate_cache_workers_ram", invalidate)

    async def commit():
        published.assert_not_awaited()
        invalidate.assert_not_awaited()
        assert [call.args[0] for call in session.delete.await_args_list] == [
            *instances,
            service,
        ]
        send_post_commit_events(session)

    session.commit = AsyncMock(side_effect=commit)
    await services.CacheServiceService(session).delete(service)
    invalidate.assert_awaited_once()
    await asyncio.wait_for(delivered.wait(), timeout=2)
    session.commit.assert_awaited_once()
    deleted = [
        call.args[1]
        for call in published.await_args_list
        if call.args[0] == "cacheserviceinstance"
    ]
    assert {event.data.id for event in deleted} == {1, 2}
    assert all(event.type == EventType.DELETED for event in deleted)
    assert {event.data.worker_id for event in deleted} == {1, 2}
    assert all(event.data.cache_service_id == 9 for event in deleted)


@pytest.mark.asyncio
async def test_failed_service_delete_preserves_reservation_cache(monkeypatch):
    service = SimpleNamespace(
        delete=AsyncMock(side_effect=RuntimeError("commit failed"))
    )
    invalidate = AsyncMock()
    monkeypatch.setattr(services, "invalidate_cache_workers_ram", invalidate)
    with pytest.raises(RuntimeError, match="commit failed"):
        await services.CacheServiceService(object()).delete(service)
    invalidate.assert_not_awaited()
