from unittest.mock import AsyncMock, MagicMock

import pytest

from gpustack.schemas.workers import Worker
from gpustack.server import services


@pytest.mark.asyncio
async def test_batch_save_commits_once_and_invalidates_each_worker(monkeypatch):
    session = AsyncMock()
    session.add = MagicMock()
    publish = MagicMock()
    invalidate_all = AsyncMock()
    invalidate_worker = AsyncMock()
    monkeypatch.setattr(Worker, "_publish_event_after_commit", publish)
    monkeypatch.setattr(Worker, "_invalidate_cached_all", invalidate_all)
    monkeypatch.setattr(services, "delete_cache_by_key", invalidate_worker)
    workers = [Worker(id=1, name="w1"), Worker(id=2, name="w2")]

    assert await services.WorkerService(session).batch_update(workers) == 2
    session.commit.assert_awaited_once()
    session.rollback.assert_not_awaited()
    assert session.add.call_count == 2
    assert publish.call_count == 2
    invalidate_all.assert_awaited_once()
    assert invalidate_worker.await_count == 4


@pytest.mark.asyncio
async def test_failed_batch_save_rolls_back_and_does_not_invalidate(monkeypatch):
    session = AsyncMock()
    session.add = MagicMock()
    session.commit.side_effect = RuntimeError("commit failed")
    invalidate_all = AsyncMock()
    invalidate_worker = AsyncMock()
    monkeypatch.setattr(Worker, "_publish_event_after_commit", MagicMock())
    monkeypatch.setattr(Worker, "_invalidate_cached_all", invalidate_all)
    monkeypatch.setattr(services, "delete_cache_by_key", invalidate_worker)

    with pytest.raises(RuntimeError, match="commit failed"):
        await services.WorkerService(session).batch_update(
            [Worker(id=1, name="w1"), Worker(id=2, name="w2")]
        )
    session.commit.assert_awaited_once()
    session.rollback.assert_awaited_once()
    invalidate_all.assert_not_awaited()
    invalidate_worker.assert_not_awaited()
