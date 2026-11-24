"""Worker state must not read server downtime as worker downtime.

The stored ``heartbeat_time`` is absolute, so a server that was stopped for
longer than the grace period would otherwise mark every worker ``NOT_READY``
as soon as it starts again, even though the workers never stopped.
"""

from datetime import datetime, timedelta, timezone

import pytest

from gpustack.schemas.workers import WorkerStateEnum
from gpustack.utils import observation
from tests.fixtures.workers.fixtures import linux_nvidia_1_4090_24gx1


def _worker(heartbeat_age_seconds: int):
    worker = linux_nvidia_1_4090_24gx1()
    worker.maintenance = None
    worker.state = WorkerStateEnum.READY
    worker.state_message = None
    worker.unreachable = False
    worker.heartbeat_time = datetime.now(timezone.utc) - timedelta(
        seconds=heartbeat_age_seconds
    )
    return worker


@pytest.mark.parametrize("heartbeat_age_seconds", [200, 600])
def test_recently_started_server_keeps_workers_ready(
    monkeypatch, heartbeat_age_seconds
):
    worker = _worker(heartbeat_age_seconds)
    monkeypatch.setattr(
        observation,
        "_observation_started_at",
        datetime.now(timezone.utc) - timedelta(seconds=30),
    )

    worker.compute_state()

    assert worker.state == WorkerStateEnum.READY
    assert worker.state_message is None


def test_a_long_observation_window_still_detects_a_lost_heartbeat(monkeypatch):
    worker = _worker(600)
    monkeypatch.setattr(
        observation,
        "_observation_started_at",
        datetime.now(timezone.utc) - timedelta(seconds=600),
    )

    worker.compute_state()

    assert worker.state == WorkerStateEnum.NOT_READY
    assert "Heartbeat lost" in worker.state_message
