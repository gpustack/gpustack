"""Recognising a crash loop instead of calling it "still launching".

The sync loop skips a workload in PENDING or INITIALIZING as slow to start. A
container that binds, fails and is restarted never leaves those states, so the
instance sits at `starting` for as long as anyone leaves it — and that is the
shape every port-level failure in a disaggregated deployment takes. Nothing
marked it failed, so nothing surfaced it and nothing stopped it.

These pin the two halves: that the loop is recognised at all, and that the
reason attached to it comes from the log's first error rather than from the
workload (which has none — a restarted container reports no message and no
useful exit code).
"""

import os
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gpustack.schemas.models import (
    ModelInstanceStateEnum,
    PortBand,
)
from gpustack.worker.serve_manager import ServeManager
from tests.utils.model import new_model_instance


def _manager(tmp_path, worker_id: int = 1) -> ServeManager:
    clientset = MagicMock()
    cfg = SimpleNamespace(
        log_dir=str(tmp_path),
        service_port_range="40000-40063",
        system_default_container_registry=None,
    )
    os.makedirs(f"{tmp_path}/serve", exist_ok=True)
    manager = ServeManager(lambda: worker_id, lambda: clientset, cfg)
    manager._inference_backend_manager = MagicMock()
    manager._update_model_instance = MagicMock()
    return manager


def _instance(state=ModelInstanceStateEnum.STARTING, named_ports=None):
    mi = new_model_instance(1, "mi", 1, worker_id=1, state=state)
    mi.named_ports = named_ports
    return mi


def _workload(restart_count: int):
    return SimpleNamespace(
        exits=[SimpleNamespace(restart_count=restart_count, exit_code=1)]
    )


def _write_log(tmp_path, text: str, restart: int = 0):
    path = f"{tmp_path}/serve/1.container.{restart}.log"
    with open(path, "w") as f:
        f.write(text)
    return path


# --- recognising the loop -------------------------------------------------- #


def test_a_few_restarts_are_still_launching(tmp_path):
    manager = _manager(tmp_path)
    mi = _instance()

    for count in (0, 1, 2):
        assert manager._mark_crash_loop(mi, _workload(count), True) is False
    manager._update_model_instance.assert_not_called()


def test_repeated_restarts_become_an_error(tmp_path):
    manager = _manager(tmp_path)
    mi = _instance()

    for count in (0, 1, 2):
        manager._mark_crash_loop(mi, _workload(count), True)
    assert manager._mark_crash_loop(mi, _workload(3), True) is True

    _, kwargs = manager._update_model_instance.call_args
    assert kwargs["state"] == ModelInstanceStateEnum.ERROR
    assert "Restarted 3 times without serving" in kwargs["state_message"]


def test_a_member_that_served_is_left_alone(tmp_path):
    """It ran, so its configuration is not the problem — a later loop is a
    different failure and the existing restart-on-error path owns it."""
    manager = _manager(tmp_path)
    mi = _instance()
    manager._restart_tracker.observe_restart_count(mi.id, 0, datetime.now(timezone.utc))
    manager._restart_tracker.observe_running(mi.id)

    for count in (1, 2, 3, 9):
        assert manager._mark_crash_loop(mi, _workload(count), True) is False


def test_a_subordinate_worker_is_not_written_here(tmp_path):
    """Its state lives inside `distributed_servers`, and the existing failure
    path already owns that shape. Two writers for one field is worse than a
    missed loop."""
    manager = _manager(tmp_path)
    mi = _instance()

    for count in range(6):
        assert manager._mark_crash_loop(mi, _workload(count), False) is False


def test_an_instance_already_in_error_is_not_rewritten(tmp_path):
    manager = _manager(tmp_path)
    mi = _instance(state=ModelInstanceStateEnum.ERROR)

    for count in range(6):
        assert manager._mark_crash_loop(mi, _workload(count), True) is False


# --- the reason ------------------------------------------------------------ #


def test_the_reason_is_the_logs_first_error_not_its_last(tmp_path):
    """The measured case: `IndexError` is what ends the process, and it says
    nothing a user can act on."""
    manager = _manager(tmp_path)
    mi = _instance()
    _write_log(
        tmp_path,
        "INFO loading\n"
        "ERROR NIXL_ERR_BACKEND: could not load remote metadata\n"
        "ERROR IndexError: list index out of range\n",
    )

    for count in range(4):
        manager._mark_crash_loop(mi, _workload(count), True)

    _, kwargs = manager._update_model_instance.call_args
    assert "kv_ifname" in kwargs["state_message"]
    assert "IndexError" not in kwargs["state_message"]


def test_a_taken_port_is_named_by_its_band(tmp_path):
    manager = _manager(tmp_path)
    mi = _instance(named_ports={"kv_side_channel": PortBand(base=40031, count=1)})
    _write_log(tmp_path, "ERROR Address already in use: 0.0.0.0:40031\n")

    for count in range(4):
        manager._mark_crash_loop(mi, _workload(count), True)

    _, kwargs = manager._update_model_instance.call_args
    assert "kv_side_channel" in kwargs["state_message"]


def test_an_unreadable_log_still_marks_the_failure(tmp_path):
    """A missing log makes the diagnosis less specific. It must not stop the
    instance being marked failed — "forever starting" is the thing being
    fixed."""
    manager = _manager(tmp_path)
    mi = _instance()

    for count in range(4):
        manager._mark_crash_loop(mi, _workload(count), True)

    _, kwargs = manager._update_model_instance.call_args
    assert kwargs["state"] == ModelInstanceStateEnum.ERROR
    assert "first error, not the last" in kwargs["state_message"]


def test_the_newest_container_log_is_read(tmp_path):
    """Restart N's log is where the current failure is; restart 0's is where
    the first one was."""
    manager = _manager(tmp_path)
    mi = _instance()
    _write_log(tmp_path, "ERROR rdma_create_event_channel failed\n", restart=0)
    time.sleep(0.01)
    _write_log(tmp_path, "ERROR NIXL_ERR_BACKEND: unreachable\n", restart=3)

    text = manager._read_container_log(mi)
    assert "NIXL_ERR_BACKEND" in text
    assert "rdma_create_event_channel" not in text


def test_a_large_log_is_tailed(tmp_path):
    """An engine's startup log carries a full config dump; the signatures being
    looked for are startup-time and near the end of what matters."""
    manager = _manager(tmp_path)
    mi = _instance()
    _write_log(tmp_path, ("x" * 1000 + "\n") * 500 + "ERROR NIXL_ERR_BACKEND\n")

    text = manager._read_container_log(mi, limit=2000)

    assert "NIXL_ERR_BACKEND" in text
    assert len(text) <= 2000


# --- lifecycle ------------------------------------------------------------- #


@pytest.mark.parametrize("clear", [True, False])
def test_stopping_clears_the_verdict_only_with_the_backoff(tmp_path, clear):
    """The two answer the same question — is this one still worth retrying —
    so a stop that keeps the backoff has to keep the loop history, or a member
    being restarted for the fourth time looks like one starting for the
    first."""
    manager = _manager(tmp_path)
    mi = _instance()
    for count in range(4):
        manager._mark_crash_loop(mi, _workload(count), True)

    if clear:
        manager._restart_tracker.forget(mi.id)
        assert (
            manager._restart_tracker.observe_restart_count(
                mi.id, 0, datetime.now(timezone.utc)
            )
            is False
        )
    else:
        # History retained: the next observation still reports the loop.
        assert manager._mark_crash_loop(_instance(), _workload(5), True) is True
