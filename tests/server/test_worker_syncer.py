from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from gpustack import envs
from gpustack.schemas.workers import Maintenance, Worker, WorkerStateEnum
from gpustack.server.cache import cache
from gpustack.server.services import WorkerService
from gpustack.server.worker_syncer import WorkerSyncer
from tests.fixtures.workers.fixtures import linux_nvidia_1_4090_24gx1
from tests.utils.mock import mock_async_session


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fresh_state, state_message, maintenance, reachable, expected_state",
    [
        pytest.param(
            WorkerStateEnum.READY,
            None,
            None,
            True,
            WorkerStateEnum.READY,
            id="heartbeat-recovered",
        ),
        pytest.param(
            WorkerStateEnum.NOT_READY,
            "Worker reported a status error",
            None,
            True,
            WorkerStateEnum.NOT_READY,
            id="status-error",
        ),
        pytest.param(
            WorkerStateEnum.READY,
            None,
            Maintenance(enabled=True, message="Planned maintenance"),
            True,
            WorkerStateEnum.MAINTENANCE,
            id="maintenance-enabled",
        ),
        pytest.param(
            WorkerStateEnum.PROVISIONING,
            "Worker is being reprovisioned",
            None,
            True,
            WorkerStateEnum.PROVISIONING,
            id="provisioning-started",
        ),
        pytest.param(
            WorkerStateEnum.READY,
            None,
            None,
            False,
            WorkerStateEnum.UNREACHABLE,
            id="reachability-result-applied",
        ),
        pytest.param(None, None, None, True, None, id="worker-deleted"),
    ],
)
async def test_sync_recomputes_state_on_the_fresh_row_before_writing(
    caplog,
    monkeypatch,
    fresh_state,
    state_message,
    maintenance,
    reachable,
    expected_state,
):
    now = datetime.now(timezone.utc)
    stale_snapshot = linux_nvidia_1_4090_24gx1()
    stale_snapshot.state = WorkerStateEnum.READY
    stale_snapshot.state_message = None
    stale_snapshot.maintenance = None
    stale_snapshot.heartbeat_time = now - timedelta(seconds=600)
    stale_snapshot.unreachable = False

    cached_row = stale_snapshot.model_copy(deep=True)
    fresh_row = None
    if fresh_state is not None:
        fresh_row = stale_snapshot.model_copy(deep=True)
        fresh_row.name = "renamed-worker"
        fresh_row.heartbeat_time = now
        fresh_row.state = fresh_state
        fresh_row.state_message = state_message
        fresh_row.maintenance = maintenance
        fresh_row.unreachable = reachable

    monkeypatch.setattr(envs, "WORKER_UNREACHABLE_CHECK_MODE", "enabled")
    monkeypatch.setattr(envs, "WORKER_HEARTBEAT_GRACE_PERIOD", 60)
    with (
        patch("gpustack.server.worker_syncer.async_session") as session_factory,
        patch(
            "gpustack.server.worker_syncer.Worker.all",
            AsyncMock(return_value=[stale_snapshot]),
        ),
        patch(
            "gpustack.server.worker_syncer.Worker.one_by_id",
            AsyncMock(side_effect=[cached_row, fresh_row]),
        ) as load_row,
        patch.object(Worker, "update", autospec=True) as update_row,
        patch(
            "gpustack.server.worker_syncer.is_worker_reachable",
            AsyncMock(return_value=reachable),
        ) as reachability_check,
    ):
        session_context = mock_async_session()
        session_factory.return_value = session_context
        session = session_context.__aenter__.return_value
        await cache.clear()
        try:
            assert (
                await WorkerService(session).get_by_id(stale_snapshot.id) is cached_row
            )
            assert (
                await WorkerService(session).get_by_id(stale_snapshot.id) is cached_row
            )
            load_row.assert_awaited_once_with(session, stale_snapshot.id)
            caplog.set_level("INFO", logger="gpustack.server.worker_syncer")
            await WorkerSyncer(lambda: None, lambda: None)._sync_workers_states()
        finally:
            await cache.clear()

    reachability_check.assert_awaited_once()
    assert load_row.await_count == 2
    assert stale_snapshot.state == WorkerStateEnum.NOT_READY
    if fresh_row is None:
        update_row.assert_not_awaited()
        assert "Marked worker" not in caplog.text
        return

    update_row.assert_awaited_once_with(fresh_row, session, None)
    assert fresh_row.state == expected_state
    assert fresh_row.heartbeat_time == now
    assert fresh_row.unreachable is not reachable
    if expected_state == WorkerStateEnum.MAINTENANCE:
        assert fresh_row.state_message == maintenance.message
    elif expected_state == WorkerStateEnum.UNREACHABLE:
        assert "/healthz" in fresh_row.state_message
    else:
        assert fresh_row.state_message == state_message

    if expected_state.is_provisioning:
        assert "Marked worker" not in caplog.text
    else:
        assert f"Marked worker {fresh_row.name} as {expected_state}" in caplog.text
    assert f"Marked worker {stale_snapshot.name} as" not in caplog.text
