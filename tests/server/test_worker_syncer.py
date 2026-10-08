from unittest.mock import AsyncMock, patch

import pytest

from gpustack.schemas.workers import WorkerStateEnum
from gpustack.server.worker_syncer import WorkerSyncer


class _FakeWorker:
    def __init__(self, worker_id, state, state_message):
        self.id = worker_id
        self.name = f"worker-{worker_id}"
        self.state = state
        self.state_message = state_message
        self.unreachable = False
        self.compute_calls = 0

    def compute_state(self):
        self.compute_calls += 1
        if self.state == WorkerStateEnum.NOT_READY and self.state_message:
            return
        self.state = WorkerStateEnum.READY
        self.state_message = None


class _SessionContext:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


@pytest.mark.asyncio
async def test_sync_recomputes_state_on_the_fresh_row_before_writing(caplog):
    stale_snapshot = _FakeWorker(1, WorkerStateEnum.READY, None)
    fresh_row = _FakeWorker(
        1,
        WorkerStateEnum.NOT_READY,
        "Heartbeat lost while status flush was writing",
    )
    updates = []

    class _WorkerService:
        def __init__(self, session):
            pass

        async def get_by_id(self, worker_id):
            return fresh_row

        async def update(self, worker):
            updates.append(worker)

    with (
        patch(
            "gpustack.server.worker_syncer.async_session",
            return_value=_SessionContext(),
        ),
        patch(
            "gpustack.server.worker_syncer.Worker.all",
            AsyncMock(return_value=[stale_snapshot]),
        ),
        patch(
            "gpustack.server.worker_syncer.WorkerService",
            _WorkerService,
        ),
        patch.object(
            WorkerSyncer,
            "_should_check_unreachable",
            return_value=False,
        ),
        patch.object(
            WorkerSyncer,
            "filter_state_change_workers",
            return_value=[stale_snapshot],
        ),
    ):
        caplog.set_level("INFO", logger="gpustack.server.worker_syncer")
        await WorkerSyncer(None, None)._sync_workers_states()

    assert updates == [fresh_row]
    assert fresh_row.compute_calls == 1
    assert fresh_row.state == WorkerStateEnum.NOT_READY
    assert fresh_row.state_message == "Heartbeat lost while status flush was writing"
    assert "Marked worker worker-1 as WorkerStateEnum.NOT_READY" in caplog.text
    assert "Marked worker worker-1 as WorkerStateEnum.READY" not in caplog.text
