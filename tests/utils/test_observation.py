from datetime import datetime, timedelta, timezone

from gpustack.utils import observation
from gpustack.utils.network import is_heartbeat_offline, is_offline
from gpustack.utils.observation import mark_observation_start


def test_offline_without_an_observation_window_ages_from_the_record():
    now = datetime.now(timezone.utc)

    offline, last_update = is_offline(now - timedelta(seconds=600), 300, now)

    assert offline is True
    assert last_update == (now - timedelta(seconds=600)).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )


def test_observation_window_does_not_count_its_own_downtime():
    now = datetime.now(timezone.utc)
    heartbeat = now - timedelta(seconds=3600)

    offline, last_update = is_offline(
        heartbeat, 300, now, since=now - timedelta(seconds=60)
    )

    assert offline is False
    # The message still points at the real heartbeat, not at the window start.
    assert last_update == heartbeat.strftime("%Y-%m-%d %H:%M:%S UTC")


def test_observation_window_still_expires_after_the_grace_period():
    now = datetime.now(timezone.utc)

    offline, _ = is_offline(
        now - timedelta(seconds=3600), 300, now, since=now - timedelta(seconds=400)
    )

    assert offline is True


def test_observation_window_keeps_an_unknown_heartbeat_offline():
    now = datetime.now(timezone.utc)

    offline, last_update = is_offline(None, 300, now, since=now)

    assert offline is True
    assert last_update == "unknown"


def test_mark_observation_start_resets_what_counts_as_silence(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(observation, "_observation_started_at", None)
    worker = SimpleNamespace(
        heartbeat_time=datetime.now(timezone.utc) - timedelta(seconds=600)
    )
    assert is_heartbeat_offline(worker, 300)[0] is True

    mark_observation_start()

    assert is_heartbeat_offline(worker, 300)[0] is False

    mark_observation_start(datetime.now(timezone.utc) - timedelta(seconds=600))

    assert is_heartbeat_offline(worker, 300)[0] is True
