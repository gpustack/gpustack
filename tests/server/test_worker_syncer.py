import copy
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import update as sa_update
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession as SQLModelAsyncSession

from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.server.cache import delete_cache_by_key
from gpustack.server.services import WorkerService
from gpustack.server.worker_syncer import WorkerSyncer
from tests.fixtures.workers.fixtures import linux_nvidia_1_4090_24gx1
from tests.utils.mock import mock_async_session


def _syncer():
    return WorkerSyncer(
        http_client_getter=lambda: MagicMock(),
        http_client_no_proxy_getter=lambda: MagicMock(),
    )


@pytest.mark.asyncio
async def test_sync_workers_states_does_not_clobber_a_fresher_status_flush():
    """
    Regression test for #6090.

    The worker read at the top of _sync_workers_states() can be up to
    worker_unreachable_timeout seconds old by the time the write phase runs.
    If flush_worker_status() commits a fresh heartbeat and a corrected READY
    state inside that window, the write phase must not overwrite it with a
    decision computed from the stale read.
    """
    now = datetime.now(timezone.utc)

    stale_read = linux_nvidia_1_4090_24gx1()
    stale_read.id = 1
    stale_read.name = "stale-race-worker"
    stale_read.state = WorkerStateEnum.READY
    stale_read.state_message = None
    stale_read.maintenance = None
    stale_read.unreachable = False
    # Already past the heartbeat grace period at the moment it was read.
    stale_read.heartbeat_time = now - timedelta(seconds=200)

    # What the per-worker fetch sees at write time: flush_worker_status()
    # already committed a fresh heartbeat and confirmed READY in between.
    fresh_row = copy.deepcopy(stale_read)
    fresh_row.heartbeat_time = now

    one_by_id = AsyncMock(return_value=fresh_row)
    update = AsyncMock()
    worker_service = MagicMock()
    worker_service.update = update

    with (
        patch(
            "gpustack.server.worker_syncer.async_session",
            return_value=mock_async_session(),
        ),
        patch(
            "gpustack.server.worker_syncer.Worker.all",
            AsyncMock(return_value=[stale_read]),
        ),
        patch(
            "gpustack.server.worker_syncer.Worker.one_by_id",
            one_by_id,
        ),
        patch(
            "gpustack.server.worker_syncer.envs.WORKER_UNREACHABLE_CHECK_MODE",
            "disabled",
        ),
        patch(
            "gpustack.server.worker_syncer.WorkerService",
            return_value=worker_service,
        ),
    ):
        await _syncer()._sync_workers_states()

    # The fresh row is already correctly READY; recomputing from its own
    # heartbeat_time must agree, so nothing should be written back.
    update.assert_not_awaited()
    assert fresh_row.state == WorkerStateEnum.READY


@pytest.mark.asyncio
async def test_sync_workers_states_still_persists_a_genuine_state_change():
    """A real transition (heartbeat genuinely lost) must still be written and
    grouped for the summary log; the fix must not turn every sync into a no-op."""
    now = datetime.now(timezone.utc)

    stale_read = linux_nvidia_1_4090_24gx1()
    stale_read.id = 2
    stale_read.name = "genuinely-offline-worker"
    stale_read.state = WorkerStateEnum.READY
    stale_read.state_message = None
    stale_read.maintenance = None
    stale_read.unreachable = False
    stale_read.heartbeat_time = now - timedelta(seconds=200)

    # The per-worker fetch sees a row just as offline as the initial read: no
    # fresher heartbeat landed in between, so the transition to NOT_READY is real.
    fresh_row = copy.deepcopy(stale_read)

    one_by_id = AsyncMock(return_value=fresh_row)
    update = AsyncMock()
    worker_service = MagicMock()
    worker_service.update = update

    with (
        patch(
            "gpustack.server.worker_syncer.async_session",
            return_value=mock_async_session(),
        ),
        patch(
            "gpustack.server.worker_syncer.Worker.all",
            AsyncMock(return_value=[stale_read]),
        ),
        patch(
            "gpustack.server.worker_syncer.Worker.one_by_id",
            one_by_id,
        ),
        patch(
            "gpustack.server.worker_syncer.envs.WORKER_UNREACHABLE_CHECK_MODE",
            "disabled",
        ),
        patch(
            "gpustack.server.worker_syncer.WorkerService",
            return_value=worker_service,
        ),
    ):
        await _syncer()._sync_workers_states()

    update.assert_awaited_once_with(fresh_row)
    assert fresh_row.state == WorkerStateEnum.NOT_READY


@pytest.mark.asyncio
async def test_sync_workers_states_skips_a_worker_deleted_mid_sync():
    """The per-worker fetch returning None (deleted between the initial read
    and the fetch) must be skipped, not raise."""
    stale_read = linux_nvidia_1_4090_24gx1()
    stale_read.id = 3
    stale_read.name = "deleted-mid-sync-worker"
    stale_read.state = WorkerStateEnum.READY
    stale_read.state_message = None
    stale_read.maintenance = None
    stale_read.unreachable = False

    one_by_id = AsyncMock(return_value=None)
    update = AsyncMock()
    worker_service = MagicMock()
    worker_service.update = update

    with (
        patch(
            "gpustack.server.worker_syncer.async_session",
            return_value=mock_async_session(),
        ),
        patch(
            "gpustack.server.worker_syncer.Worker.all",
            AsyncMock(return_value=[stale_read]),
        ),
        patch(
            "gpustack.server.worker_syncer.Worker.one_by_id",
            one_by_id,
        ),
        patch(
            "gpustack.server.worker_syncer.envs.WORKER_UNREACHABLE_CHECK_MODE",
            "disabled",
        ),
        patch(
            "gpustack.server.worker_syncer.WorkerService",
            return_value=worker_service,
        ),
    ):
        await _syncer()._sync_workers_states()

    update.assert_not_awaited()


@pytest.mark.asyncio
async def test_sync_workers_states_ignores_get_by_ids_stale_cache():
    """End-to-end against a real DB and the real WorkerService cache.

    get_by_id is @locked_cached with a 10-minute TTL that flush_heartbeats()'s
    raw bulk UPDATE never invalidates (confirmed by reading flush_heartbeats(),
    which updates via a bare `update(Worker)...` statement with no
    delete_cache_by_key call). If the syncer's per-worker fetch went through
    that cache, a heartbeat recorded after the cache was warmed would be
    invisible to it. Warm the cache first via a real get_by_id call, then
    apply the same kind of bulk update, and confirm the sync still sees it.
    """
    worker_id = 9_460_001  # unlikely to collide with another test's cache key
    now = datetime.now(timezone.utc)

    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(Worker.metadata.create_all)

    async with SQLModelAsyncSession(engine, expire_on_commit=False) as session:
        worker = linux_nvidia_1_4090_24gx1()
        worker.id = worker_id
        worker.worker_uuid = f"probe-{worker_id}"
        worker.ifname = "eth0"
        worker.port = worker.port or 10150
        worker.worker_version = worker.worker_version or "0.0.0"
        worker.cluster_id = 1
        worker.owner_principal_id = 1
        worker.created_at = now
        worker.updated_at = now
        worker.state = WorkerStateEnum.READY
        worker.state_message = None
        worker.maintenance = None
        worker.unreachable = False
        # Already past the heartbeat grace period at insert time.
        worker.heartbeat_time = now - timedelta(seconds=200)
        session.add(worker)
        await session.commit()

        # Warm WorkerService's real get_by_id cache, as an unrelated route
        # handler reading this worker's detail page would.
        await WorkerService(session).get_by_id(worker_id)

        # flush_heartbeats()'s shape: a bare bulk UPDATE, no cache invalidation.
        await session.execute(
            sa_update(Worker).where(Worker.id == worker_id).values(heartbeat_time=now)
        )
        await session.commit()

    try:
        with (
            patch(
                "gpustack.server.worker_syncer.async_session",
                lambda: SQLModelAsyncSession(engine, expire_on_commit=False),
            ),
            patch(
                "gpustack.server.worker_syncer.Worker.all",
                AsyncMock(return_value=[worker]),
            ),
            patch(
                "gpustack.server.worker_syncer.envs.WORKER_UNREACHABLE_CHECK_MODE",
                "disabled",
            ),
        ):
            await _syncer()._sync_workers_states()

        async with SQLModelAsyncSession(engine, expire_on_commit=False) as session:
            reloaded = await Worker.one_by_id(session, worker_id)
            # If the sync had used the stale cached row instead of the fresh
            # heartbeat, it would have (wrongly) written NOT_READY here.
            assert reloaded.state == WorkerStateEnum.READY
    finally:
        await delete_cache_by_key(WorkerService.get_by_id, worker_id)
        await engine.dispose()
