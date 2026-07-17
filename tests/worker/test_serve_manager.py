from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, patch

from gpustack.schemas.models import (
    BackendEnum,
    DistributedServerCoordinateModeEnum,
    DistributedServers,
    ModelInstanceSubordinateWorker,
    ModelInstanceStateEnum,
    SourceEnum,
)
from gpustack.server.bus import Event, EventType
from gpustack.worker.serve_manager import ServeManager
from gpustack_runtime.deployer import WorkloadStatusStateEnum
from tests.utils.model import new_model, new_model_instance


def _fake_stop_event(max_waits: int = 100):
    """A stop event whose wait() returns instantly (tests aren't driven by real
    time) and stays unset, so the log persistence loop is driven purely by the
    get_workload state sequence. It auto-sets after max_waits waits so a
    mis-sized mock or a runaway loop fails the test fast instead of hanging CI."""
    state = {"waits": 0, "stopped": False}

    def is_set():
        return state["stopped"]

    def wait(timeout=None):
        state["waits"] += 1
        if state["waits"] >= max_waits:
            state["stopped"] = True
        return state["stopped"]

    stop_event = MagicMock()
    stop_event.is_set.side_effect = is_set
    stop_event.wait.side_effect = wait
    return stop_event


def _get_workload_sequence(states):
    """side_effect for a patched get_workload. The recovery grace-poll queries
    get_workload several times per stream EOF, so once the sequence reaches its
    terminal state it must keep returning it: a list that runs dry would raise
    IndexError, which _container_still_running treats as "still alive", spinning
    the reconnect loop forever."""
    remaining = list(states)

    def next_state(name):
        return remaining.pop(0) if len(remaining) > 1 else remaining[0]

    return next_state


def _build_serve_manager(worker_id: int = 1):
    clientset = MagicMock()
    clientset.model_instances.list.return_value = SimpleNamespace(items=[])
    cfg = SimpleNamespace(log_dir="/tmp")
    manager = ServeManager(lambda: worker_id, lambda: clientset, cfg)
    manager._inference_backend_manager = MagicMock()
    return manager, clientset


def test_sync_model_instances_state_marks_main_unreachable_when_subordinate_unreachable():
    manager, clientset = _build_serve_manager()

    model_instance = new_model_instance(
        1,
        "distributed-instance",
        1,
        worker_id=1,
        state=ModelInstanceStateEnum.RUNNING,
    )
    model_instance.worker_ip = "127.0.0.1"
    model_instance.port = 8000
    model_instance.distributed_servers = DistributedServers(
        mode=DistributedServerCoordinateModeEnum.RUN_FIRST,
        subordinate_workers=[
            ModelInstanceSubordinateWorker(
                worker_id=2,
                worker_name="worker-2",
                worker_ip="10.0.0.2",
                state=ModelInstanceStateEnum.UNREACHABLE,
                state_message="Worker is unreachable from the server",
            )
        ],
    )
    clientset.model_instances.list.return_value = SimpleNamespace(
        items=[model_instance]
    )

    model = new_model(1, "test", 1, huggingface_repo_id="Qwen/Qwen2.5-0.5B-Instruct")
    model.backend = BackendEnum.VLLM
    model.backend_version = "0.8.0"

    with (
        patch(
            "gpustack.worker.serve_manager.get_workload",
            return_value=SimpleNamespace(state="running"),
        ),
        patch.object(manager, "_is_provisioning", return_value=False),
        patch.object(manager, "_get_model", return_value=model),
        patch.object(manager, "_update_model_instance") as update_model_instance,
    ):
        manager.sync_model_instances_state()

    update_model_instance.assert_called_once_with(
        model_instance.id,
        state=ModelInstanceStateEnum.UNREACHABLE,
        state_message=(
            "Distributed serving unreachable in subordinate worker "
            "10.0.0.2: Worker is unreachable from the server."
        ),
    )


def test_restart_error_model_instance_uses_transient_backoff_count():
    manager, _ = _build_serve_manager()
    model_instance = new_model_instance(
        1,
        "restarted-instance",
        1,
        worker_id=1,
        state=ModelInstanceStateEnum.ERROR,
    )
    model_instance.restart_count = 20
    model_instance.last_restart_time = datetime.now(timezone.utc)

    with (
        patch.object(manager, "_is_provisioning", return_value=False),
        patch.object(manager, "_update_model_instance") as update_model_instance,
        patch("gpustack.worker.serve_manager.logger"),
    ):
        manager._restart_error_model_instance(model_instance)

    update_model_instance.assert_called_once_with(
        model_instance.id,
        restart_count=21,
        last_restart_time=ANY,
        state=ModelInstanceStateEnum.SCHEDULED,
        state_message="",
    )


def test_restart_model_instance_preserves_transient_backoff_count():
    manager, _ = _build_serve_manager()
    model_instance = new_model_instance(
        1,
        "restarted-instance",
        1,
        worker_id=1,
        state=ModelInstanceStateEnum.SCHEDULED,
    )
    manager._restart_backoff_counts[model_instance.id] = 1

    with (
        patch.object(manager, "_is_provisioning", return_value=False),
        patch.object(manager, "_start_model_instance"),
        # _stop_model_instance runs for real to exercise clear_restart_backoff=
        # False, but its delete_workload side effect would hit the runtime socket.
        patch("gpustack.worker.serve_manager.delete_workload"),
    ):
        manager._restart_model_instance(model_instance)

    assert manager._restart_backoff_counts[model_instance.id] == 1


def test_cleanup_old_logs_keeps_only_current_and_previous_restart(tmp_path: Path):
    """Keep main/container logs for R and R-1; delete older restart_count files."""
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir(parents=True)
    mid = 42
    for name in (
        f"{mid}.0.log",
        f"{mid}.1.log",
        f"{mid}.2.log",
        f"{mid}.container.0.log",
        f"{mid}.container.1.log",
        f"{mid}.container.2.log",
    ):
        (serve_dir / name).write_text("x", encoding="utf-8")

    manager, _clients = _build_serve_manager()
    manager._serve_log_dir = str(serve_dir)

    manager._cleanup_old_logs(mid, 2)

    remaining = sorted(p.name for p in serve_dir.iterdir())
    assert remaining == [
        f"{mid}.1.log",
        f"{mid}.2.log",
        f"{mid}.container.1.log",
        f"{mid}.container.2.log",
    ]


def test_cleanup_old_logs_restart_zero_purges_all(tmp_path: Path):
    """Fresh start (restart_count 0) removes every log for the id, incl. sidecar,
    but leaves other instances' logs and model-file download logs."""
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir(parents=True)
    mid, other = 7, 8
    for name in (
        f"{mid}.0.log",
        f"{mid}.container.1.log",
        f"{mid}.container.ray-head.0.log",
        f"{other}.0.log",
        f"model_file_{mid}.download.log",
    ):
        (serve_dir / name).write_text("x", encoding="utf-8")

    manager, _clients = _build_serve_manager()
    manager._serve_log_dir = str(serve_dir)

    manager._cleanup_old_logs(mid, 0)

    remaining = sorted(p.name for p in serve_dir.iterdir())
    assert remaining == [f"{other}.0.log", f"model_file_{mid}.download.log"]


def test_delete_event_purges_logs_but_restart_keeps_them(tmp_path: Path):
    """DELETED removes the instance's serve logs so a reused id can't inherit them;
    a restart (default stop) keeps them for the log viewer."""
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir(parents=True)
    log = serve_dir / "1.container.ray-head.0.log"
    log.write_text("x", encoding="utf-8")

    manager, _clients = _build_serve_manager(worker_id=1)
    manager._serve_log_dir = str(serve_dir)
    model_instance = new_model_instance(
        1, "qwen3-0.6b", 1, worker_id=1, state=ModelInstanceStateEnum.RUNNING
    )
    # _handle_model_instance_event re-validates the payload; source/repo required.
    model_instance.source = SourceEnum.HUGGING_FACE
    model_instance.huggingface_repo_id = "Qwen/Qwen3-0.6B"

    with (
        patch("gpustack.worker.serve_manager.logger"),
        patch("gpustack.worker.serve_manager.delete_workload"),
        patch.object(manager, "_stop_container_log_persistence"),
        patch.object(manager, "_is_provisioning", return_value=False),
    ):
        manager._stop_model_instance(model_instance)
        assert log.exists()

        manager._handle_model_instance_event(
            Event(type=EventType.DELETED, data=model_instance)
        )
        assert not log.exists()


def test_reap_stale_instance_purges_logs(tmp_path: Path):
    """Reaping an instance the server no longer reports (a dropped DELETED) must
    also remove its serve logs, mirroring the DELETED handler."""
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir(parents=True)
    log = serve_dir / "1.container.ray-head.0.log"
    log.write_text("x", encoding="utf-8")

    manager, clientset = _build_serve_manager(worker_id=1)
    manager._serve_log_dir = str(serve_dir)
    stale = new_model_instance(
        1, "qwen3-0.6b", 1, worker_id=1, state=ModelInstanceStateEnum.RUNNING
    )
    manager._model_instance_by_instance_id[stale.id] = stale
    clientset.model_instances.list.return_value = SimpleNamespace(items=[])

    with (
        patch("gpustack.worker.serve_manager.logger"),
        patch("gpustack.worker.serve_manager.delete_workload"),
        patch.object(manager, "_stop_container_log_persistence"),
        patch.object(manager, "_is_provisioning", return_value=False),
    ):
        manager.sync_model_instances_state()

    assert not log.exists()


def test_persist_container_logs_reconnects_and_dedupes(tmp_path: Path):
    """On stream EOF while the workload is still running, reconnect and resume
    by skipping already-written history (anchor), appending only new lines."""
    manager, _clients = _build_serve_manager()
    log_path = str(tmp_path / "1.container.0.log")

    # First stream: initial history. Reconnect: full history replay + new line.
    streams = [iter(["a\n", "b\n"]), iter(["a\n", "b\n", "c\n"])]
    tails = []

    def fake_logs_workload(**kwargs):
        tails.append(kwargs["tail"])
        return streams.pop(0)

    # First EOF -> still RUNNING (reconnect); second EOF -> FAILED (exit).
    states = [
        SimpleNamespace(state=WorkloadStatusStateEnum.RUNNING),
        SimpleNamespace(state=WorkloadStatusStateEnum.FAILED),
    ]

    with (
        patch(
            "gpustack.worker.serve_manager.logs_workload",
            side_effect=fake_logs_workload,
        ),
        patch(
            "gpustack.worker.serve_manager.get_workload",
            side_effect=_get_workload_sequence(states),
        ),
    ):
        manager._persist_container_logs("wl", log_path, _fake_stop_event())

    assert tails == [-1, -1]
    assert Path(log_path).read_text(encoding="utf-8") == "a\nb\nc\n"


def test_persist_container_logs_exits_when_workload_gone(tmp_path: Path):
    """EOF while the workload no longer exists -> exit immediately, no reconnect."""
    manager, _clients = _build_serve_manager()
    log_path = str(tmp_path / "1.container.0.log")
    tails = []

    def fake_logs_workload(**kwargs):
        tails.append(kwargs["tail"])
        return iter(["a\n"])

    with (
        patch(
            "gpustack.worker.serve_manager.logs_workload",
            side_effect=fake_logs_workload,
        ),
        patch("gpustack.worker.serve_manager.get_workload", return_value=None),
    ):
        manager._persist_container_logs("wl", log_path, _fake_stop_event())

    assert tails == [-1]  # only one connection, no reconnect
    assert Path(log_path).read_text(encoding="utf-8") == "a\n"


def test_persist_container_logs_resets_when_anchor_rotated(tmp_path: Path):
    """If the anchor line was rotated out of the reconnect logs, restart from
    scratch (full rewrite) instead of skipping new lines forever."""
    manager, _clients = _build_serve_manager()
    log_path = str(tmp_path / "1.container.0.log")

    streams = [
        iter(["a\n", "b\n"]),  # round1: write a,b (anchor=b)
        iter(["x\n", "c\n"]),  # round2: anchor 'b' rotated out -> skip all, reset
        iter(["x\n", "c\n", "d\n"]),  # round3: fresh rewrite recovers
    ]
    states = [
        SimpleNamespace(state=WorkloadStatusStateEnum.RUNNING),
        SimpleNamespace(state=WorkloadStatusStateEnum.RUNNING),
        SimpleNamespace(state=WorkloadStatusStateEnum.FAILED),
    ]

    with (
        patch(
            "gpustack.worker.serve_manager.logs_workload",
            side_effect=lambda **kwargs: streams.pop(0),
        ),
        patch(
            "gpustack.worker.serve_manager.get_workload",
            side_effect=_get_workload_sequence(states),
        ),
    ):
        manager._persist_container_logs("wl", log_path, _fake_stop_event())

    assert Path(log_path).read_text(encoding="utf-8") == "x\nc\nd\n"


def test_persist_container_logs_empty_reconnect_keeps_history(tmp_path: Path):
    """An empty reconnect (0 lines) must not reset first_connect; otherwise the
    next reconnect reopens in 'w' and truncates already-persisted logs."""
    manager, _clients = _build_serve_manager()
    log_path = str(tmp_path / "1.container.0.log")

    streams = [
        iter(["a\n", "b\n"]),  # round1: write a,b
        iter([]),  # round2: empty reconnect (0 lines) -> must NOT reset
        iter(["b\n"]),  # round3: suffix replay; a,b already persisted survive
    ]
    states = [
        SimpleNamespace(state=WorkloadStatusStateEnum.RUNNING),
        SimpleNamespace(state=WorkloadStatusStateEnum.RUNNING),
        SimpleNamespace(state=WorkloadStatusStateEnum.FAILED),
    ]

    with (
        patch(
            "gpustack.worker.serve_manager.logs_workload",
            side_effect=lambda **kwargs: streams.pop(0),
        ),
        patch(
            "gpustack.worker.serve_manager.get_workload",
            side_effect=_get_workload_sequence(states),
        ),
    ):
        manager._persist_container_logs("wl", log_path, _fake_stop_event())

    # Had the empty round2 reset first_connect, round3 would reopen in 'w' and
    # truncate 'a'; a,b surviving proves it did not.
    assert Path(log_path).read_text(encoding="utf-8") == "a\nb\n"


def test_persist_container_logs_window_anchor_ignores_repeated_line(
    tmp_path: Path,
):
    """The multi-line anchor window only matches the true tail: a single-line
    anchor would false-match an earlier identical line and duplicate history."""
    manager, _clients = _build_serve_manager()
    log_path = str(tmp_path / "1.container.0.log")

    streams = [
        iter(["A\n", "B\n", "A\n", "B\n"]),  # round1: last line B repeats earlier
        iter(["A\n", "B\n", "A\n", "B\n", "C\n"]),  # round2: full replay + new C
    ]
    states = [
        SimpleNamespace(state=WorkloadStatusStateEnum.RUNNING),
        SimpleNamespace(state=WorkloadStatusStateEnum.FAILED),
    ]

    with (
        patch(
            "gpustack.worker.serve_manager.logs_workload",
            side_effect=lambda **kwargs: streams.pop(0),
        ),
        patch(
            "gpustack.worker.serve_manager.get_workload",
            side_effect=_get_workload_sequence(states),
        ),
    ):
        manager._persist_container_logs("wl", log_path, _fake_stop_event())

    # Window [A,B,A,B] matches only at the end; single-line 'B' would match
    # index 1 and duplicate A,B.
    assert Path(log_path).read_text(encoding="utf-8") == "A\nB\nA\nB\nC\n"
