from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, patch

import pytest

from gpustack.routes.worker.logs import (
    get_all_log_files,
    get_serve_log_options,
    resolve_restart_count,
)
from gpustack.schemas.models import (
    BackendEnum,
    DistributedServerCoordinateModeEnum,
    DistributedServers,
    ModelInstanceSubordinateWorker,
    ModelInstanceStateEnum,
    SourceEnum,
)
from gpustack.server.bus import Event, EventType
from gpustack.worker.serve_manager import (
    _LOG_TAIL_CHUNK_SIZE,
    ServeManager,
    _describe_workload_failure,
    _LogPersistence,
    _tail_lines,
)
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


def _fake_thread(alive: bool):
    thread = MagicMock()
    thread.is_alive.return_value = alive
    return thread


def _log_persistence(main_alive: bool):
    """A generation with the given main-thread liveness, always beside a live
    (forever-polling) sidecar discovery thread."""
    persistence = _LogPersistence(MagicMock(), _fake_thread(main_alive))
    persistence.add_aux_thread(_fake_thread(True))
    return persistence


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


def test_restart_error_model_instance_accepts_naive_database_timestamp():
    manager, _ = _build_serve_manager()
    model_instance = new_model_instance(
        1,
        "restarted-instance",
        1,
        worker_id=1,
        state=ModelInstanceStateEnum.ERROR,
    )
    model_instance.last_restart_time = datetime.now(timezone.utc).replace(
        microsecond=0, tzinfo=None
    )
    manager._restart_backoff_counts[model_instance.id] = 1

    with (
        patch.object(manager, "_is_provisioning", return_value=False),
        patch.object(manager, "_update_model_instance") as update_model_instance,
        patch("gpustack.worker.serve_manager.logger"),
    ):
        manager._restart_error_model_instance(model_instance)

    update_model_instance.assert_not_called()


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


# --- serve log discovery across the pre-v2.2.0 {id}.log naming ---


def _write_serve_logs(tmp_path: Path, *names: str) -> Path:
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir(parents=True)
    for name in names:
        (serve_dir / name).write_text("x", encoding="utf-8")
    return serve_dir


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "on_disk, expected_main_logs, expected_restart_count",
    [
        # Only a pre-v2.2.0 main log: discovered, and restart 0 by convention.
        (["1.log"], ["1.log"], 0),
        # Legacy and numbered coexist: both kept as restart 0, legacy first.
        (["1.log", "1.0.log"], ["1.log", "1.0.log"], 0),
        # Numbered only: unchanged behavior.
        (["1.5.log", "1.3.log"], ["1.3.log", "1.5.log"], 5),
        # Container and sidecar logs never leak into the main log branch.
        (["1.log", "1.container.0.log", "1.container.ray-head.0.log"], ["1.log"], 0),
        # No main log at all: restart count stays unresolvable.
        (["1.container.0.log"], [], None),
    ],
)
async def test_main_log_discovery_includes_legacy_file(
    tmp_path: Path, on_disk, expected_main_logs, expected_restart_count
):
    """The {id}.*.log glob cannot match the pre-v2.2.0 {id}.log name, so
    discovery has to add it back."""
    serve_dir = _write_serve_logs(tmp_path, *on_disk)

    files = await get_all_log_files(serve_dir, 1, container=False)

    assert [f.name for f in files] == expected_main_logs
    assert (
        await resolve_restart_count(serve_dir, 1, previous=False)
        == expected_restart_count
    )


@pytest.mark.asyncio
async def test_serve_log_options_after_upgrade_from_legacy_naming(tmp_path: Path):
    """The upgrade case from #5988: a legacy-only main log next to container logs
    the new worker wrote still yields one restart entry, and the container and
    sidecar branches keep returning only their own files."""
    serve_dir = _write_serve_logs(
        tmp_path,
        "1.log",
        "1.container.0.log",
        "1.container.ray-head.0.log",
    )
    config = SimpleNamespace(log_dir=str(tmp_path))
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(config=config)))

    response = await get_serve_log_options(request, 1)
    container_logs = await get_all_log_files(serve_dir, 1, container=True)
    sidecar_logs = await get_all_log_files(
        serve_dir, 1, container=True, container_name="ray-head"
    )

    assert len(response.restarts) == 1
    assert response.restarts[0].previous is False
    assert response.restarts[0].containers == ["default", "ray-head"]
    assert [f.name for f in container_logs] == ["1.container.0.log"]
    assert [f.name for f in sidecar_logs] == ["1.container.ray-head.0.log"]


def test_cleanup_old_logs_keeps_only_current_and_previous_restart(tmp_path: Path):
    """Keep main/container logs for R and R-1; delete older restart_count files.
    A pre-v2.2.0 {id}.log counts as restart 0, so it goes with the rest."""
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir(parents=True)
    mid = 42
    for name in (
        f"{mid}.log",
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
    """Fresh start (restart_count 0) removes every log for the id, incl. sidecar
    and {id}.log, but leaves other instances' and model-file download logs."""
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir(parents=True)
    mid, other = 7, 8
    for name in (
        f"{mid}.log",
        f"{mid}.0.log",
        f"{mid}.container.1.log",
        f"{mid}.container.ray-head.0.log",
        f"{other}.log",
        f"{other}.0.log",
        f"model_file_{mid}.download.log",
    ):
        (serve_dir / name).write_text("x", encoding="utf-8")

    manager, _clients = _build_serve_manager()
    manager._serve_log_dir = str(serve_dir)

    manager._cleanup_old_logs(mid, 0)

    remaining = sorted(p.name for p in serve_dir.iterdir())
    assert remaining == [
        f"{other}.0.log",
        f"{other}.log",
        f"model_file_{mid}.download.log",
    ]


def test_cleanup_old_logs_keeps_legacy_main_log_as_previous_restart(tmp_path: Path):
    """At R==1 the kept window is {1, 0}, so a legacy {id}.log survives as the
    previous restart alongside the numbered logs."""
    serve_dir = _write_serve_logs(tmp_path, "1.log", "1.1.log", "1.container.1.log")

    manager, _clients = _build_serve_manager()
    manager._serve_log_dir = str(serve_dir)

    manager._cleanup_old_logs(1, 1)

    remaining = sorted(p.name for p in serve_dir.iterdir())
    assert remaining == ["1.1.log", "1.container.1.log", "1.log"]


def test_permanent_teardown_purges_logs_but_restart_keeps_them(tmp_path: Path):
    """A permanent teardown (delete_logs=True, used by the reap) removes the
    instance's serve logs so a reused id can't inherit them; a restart (default
    stop) keeps them for the log viewer."""
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir(parents=True)
    log = serve_dir / "1.container.ray-head.0.log"
    log.write_text("x", encoding="utf-8")

    manager, _clients = _build_serve_manager(worker_id=1)
    manager._serve_log_dir = str(serve_dir)
    model_instance = new_model_instance(
        1, "qwen3-0.6b", 1, worker_id=1, state=ModelInstanceStateEnum.RUNNING
    )

    with (
        patch("gpustack.worker.serve_manager.logger"),
        patch("gpustack.worker.serve_manager.delete_workload"),
        patch.object(manager, "_stop_container_log_persistence"),
        patch.object(manager, "_is_provisioning", return_value=False),
    ):
        # Restart-style stop keeps the logs.
        manager._stop_model_instance(model_instance)
        assert log.exists()

        # Permanent teardown removes them.
        manager._stop_model_instance(model_instance, delete_logs=True)
        assert not log.exists()


def test_reap_stale_instance_purges_logs(tmp_path: Path):
    """Reaping an instance the server no longer reports (a dropped DELETED) must
    also remove its serve logs, {id}.log included, mirroring the DELETED
    handler."""
    serve_dir = _write_serve_logs(tmp_path, "1.container.ray-head.0.log", "1.log")

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

    assert sorted(p.name for p in serve_dir.iterdir()) == []


def test_reap_confirmation_skips_when_authoritative_fetch_still_has_instance():
    """A cache read momentarily missing an instance (e.g. mid-reconnect, before
    the watch replay repopulates) must not reap it: the authoritative
    confirmation fetch still lists it, so it is a false positive and the live
    workload is left alone."""
    manager, clientset = _build_serve_manager(worker_id=1)
    mi = new_model_instance(
        1, "qwen3-0.6b", 1, worker_id=1, state=ModelInstanceStateEnum.RUNNING
    )
    manager._model_instance_by_instance_id[mi.id] = mi
    # First (cache) read misses it -> reap candidate; the authoritative
    # confirmation fetch still has it -> not stale.
    clientset.model_instances.list.side_effect = [
        SimpleNamespace(items=[]),
        SimpleNamespace(items=[mi]),
    ]

    with (
        patch("gpustack.worker.serve_manager.logger"),
        patch.object(manager, "_stop_model_instance") as stop_model_instance,
        patch.object(manager, "_is_provisioning", return_value=False),
        patch(
            "gpustack.worker.serve_manager.get_workload",
            return_value=SimpleNamespace(state=WorkloadStatusStateEnum.INITIALIZING),
        ),
    ):
        manager.sync_model_instances_state()

    stop_model_instance.assert_not_called()
    # Cache read plus exactly one authoritative confirmation fetch.
    assert clientset.model_instances.list.call_count == 2
    confirm_call = clientset.model_instances.list.call_args_list[1]
    assert confirm_call.kwargs.get("params") == {"page": -1}
    assert confirm_call.kwargs.get("use_cache") is False


def test_reap_confirmation_reaps_when_missing_from_authoritative_fetch():
    """When an instance is absent from both the cache read and the authoritative
    confirmation fetch, it is genuinely gone and gets reaped."""
    manager, clientset = _build_serve_manager(worker_id=1)
    stale = new_model_instance(
        1, "qwen3-0.6b", 1, worker_id=1, state=ModelInstanceStateEnum.RUNNING
    )
    manager._model_instance_by_instance_id[stale.id] = stale
    clientset.model_instances.list.side_effect = [
        SimpleNamespace(items=[]),
        SimpleNamespace(items=[]),
    ]

    with (
        patch("gpustack.worker.serve_manager.logger"),
        patch.object(manager, "_stop_model_instance") as stop_model_instance,
        patch.object(manager, "_is_provisioning", return_value=False),
    ):
        manager.sync_model_instances_state()

    stop_model_instance.assert_called_once_with(stale, delete_logs=True)
    # Cache read plus the authoritative confirmation fetch before reaping.
    assert clientset.model_instances.list.call_count == 2


def test_ghost_in_cache_not_reconciled_when_backstop_disabled():
    """With the periodic reconciliation disabled (default), an instance present
    in both local state and the watch cache but gone from DB is not a
    `local - cache` reap candidate, so the sync trusts the cache and takes no
    DB round trip."""
    manager, clientset = _build_serve_manager(worker_id=1)
    ghost = new_model_instance(
        1, "qwen3-0.6b", 1, worker_id=1, state=ModelInstanceStateEnum.RUNNING
    )
    manager._model_instance_by_instance_id[ghost.id] = ghost
    clientset.model_instances.list.return_value = SimpleNamespace(items=[ghost])

    with (
        patch("gpustack.worker.serve_manager.logger"),
        patch(
            "gpustack.worker.serve_manager.envs.MODEL_INSTANCE_STATE_RECONCILE_INTERVAL",
            0,
        ),
        patch.object(manager, "_stop_model_instance") as stop_model_instance,
        patch.object(manager, "_is_provisioning", return_value=False),
        patch(
            "gpustack.worker.serve_manager.get_workload",
            return_value=SimpleNamespace(state=WorkloadStatusStateEnum.INITIALIZING),
        ),
    ):
        manager.sync_model_instances_state()

    stop_model_instance.assert_not_called()
    # Cache read only; no authoritative DB fetch when the backstop is off.
    assert clientset.model_instances.list.call_count == 1


def test_ghost_in_cache_reaped_when_backstop_interval_elapsed():
    """When the reconciliation backstop is enabled and its interval has elapsed,
    the forced authoritative DB read reaps a ghost living in both local state
    and the cache — the case `local - cache` can never flag on its own."""
    manager, clientset = _build_serve_manager(worker_id=1)
    ghost = new_model_instance(
        1, "qwen3-0.6b", 1, worker_id=1, state=ModelInstanceStateEnum.RUNNING
    )
    manager._model_instance_by_instance_id[ghost.id] = ghost
    # Last reconciliation is far enough in the past to be due.
    manager._last_state_reconcile_time = 0
    # Cache still lists the ghost; the authoritative DB fetch does not.
    clientset.model_instances.list.side_effect = [
        SimpleNamespace(items=[ghost]),
        SimpleNamespace(items=[]),
    ]

    with (
        patch("gpustack.worker.serve_manager.logger"),
        patch(
            "gpustack.worker.serve_manager.envs.MODEL_INSTANCE_STATE_RECONCILE_INTERVAL",
            60,
        ),
        patch.object(manager, "_stop_model_instance") as stop_model_instance,
        patch.object(manager, "_is_provisioning", return_value=False),
    ):
        manager.sync_model_instances_state()

    stop_model_instance.assert_called_once_with(ghost, delete_logs=True)
    # Cache read plus the forced authoritative fetch.
    assert clientset.model_instances.list.call_count == 2
    confirm_call = clientset.model_instances.list.call_args_list[1]
    assert confirm_call.kwargs.get("params") == {"page": -1}
    assert confirm_call.kwargs.get("use_cache") is False


def test_delete_event_defers_teardown_to_reap():
    """The DELETED handler no longer tears down directly; teardown is left to the
    periodic reap so there is a single teardown path (no reap-vs-event race)."""
    manager, _ = _build_serve_manager(worker_id=1)
    mi = new_model_instance(
        4, "qwen3-0.6b-kqkco", 1, worker_id=1, state=ModelInstanceStateEnum.RUNNING
    )
    mi.source = SourceEnum.HUGGING_FACE
    mi.huggingface_repo_id = "Qwen/Qwen3-0.6B"
    manager._model_instance_by_instance_id[mi.id] = mi

    with (
        patch("gpustack.worker.serve_manager.logger"),
        patch.object(manager, "_stop_model_instance") as stop_model_instance,
    ):
        manager._handle_model_instance_event(Event(type=EventType.DELETED, data=mi))

    stop_model_instance.assert_not_called()


def test_updated_event_error_does_not_crash_watch():
    """A failure on any event path (not just DELETED) must be swallowed so it
    never escapes the awatch callback. Here an UPDATED->restart raises."""
    manager, _ = _build_serve_manager(worker_id=1)
    mi = new_model_instance(
        5, "qwen3-0.6b", 1, worker_id=1, state=ModelInstanceStateEnum.SCHEDULED
    )
    mi.source = SourceEnum.HUGGING_FACE
    mi.huggingface_repo_id = "Qwen/Qwen3-0.6B"

    with (
        patch("gpustack.worker.serve_manager.logger"),
        patch.object(
            manager,
            "_restart_model_instance",
            side_effect=RuntimeError("start failed"),
        ),
    ):
        # Must not raise.
        manager._handle_model_instance_event(Event(type=EventType.UPDATED, data=mi))


def test_scheduled_update_does_not_restart_while_provisioning():
    manager, _ = _build_serve_manager(worker_id=1)
    mi = new_model_instance(
        5,
        "distributed-instance",
        1,
        worker_id=1,
        state=ModelInstanceStateEnum.SCHEDULED,
    )
    mi.source = SourceEnum.HUGGING_FACE
    mi.huggingface_repo_id = "Qwen/Qwen3-0.6B"

    with (
        patch.object(manager, "_is_provisioning", return_value=True),
        patch.object(manager, "_restart_model_instance") as restart_model_instance,
    ):
        manager._dispatch_model_instance_event(Event(type=EventType.UPDATED, data=mi))

    restart_model_instance.assert_not_called()


def test_subordinate_does_not_schedule_global_error_restart():
    manager, _ = _build_serve_manager(worker_id=2)
    mi = new_model_instance(
        5, "distributed-instance", 1, worker_id=1, state=ModelInstanceStateEnum.ERROR
    )
    mi.source = SourceEnum.HUGGING_FACE
    mi.huggingface_repo_id = "Qwen/Qwen3-0.6B"
    mi.distributed_servers = DistributedServers(
        mode=DistributedServerCoordinateModeEnum.INITIALIZE_LATER,
        subordinate_workers=[
            ModelInstanceSubordinateWorker(
                worker_id=2,
                worker_name="worker-2",
                worker_ip="10.0.0.2",
                state=ModelInstanceStateEnum.ERROR,
            )
        ],
    )

    with patch.object(manager, "_get_model") as get_model:
        manager._dispatch_model_instance_event(Event(type=EventType.UPDATED, data=mi))

    get_model.assert_not_called()
    assert mi.id not in manager._error_model_instances


def test_initialize_later_subordinate_wait_is_not_treated_as_failure():
    manager, clientset = _build_serve_manager(worker_id=2)
    mi = new_model_instance(
        5,
        "distributed-instance",
        1,
        worker_id=1,
        state=ModelInstanceStateEnum.INITIALIZING,
    )
    mi.distributed_servers = DistributedServers(
        mode=DistributedServerCoordinateModeEnum.INITIALIZE_LATER,
        subordinate_workers=[
            ModelInstanceSubordinateWorker(
                worker_id=2,
                worker_name="worker-2",
                worker_ip="10.0.0.2",
                state=ModelInstanceStateEnum.PENDING,
            )
        ],
    )
    clientset.model_instances.list.return_value = SimpleNamespace(items=[mi])

    with (
        patch.object(manager, "_is_provisioning", return_value=False),
        patch("gpustack.worker.serve_manager.get_workload") as get_workload,
        patch.object(manager, "_update_model_instance") as update_model_instance,
    ):
        manager.sync_model_instances_state()

    get_workload.assert_not_called()
    update_model_instance.assert_not_called()


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


def test_persist_container_logs_resume_appends_to_adopted_file(tmp_path: Path):
    """Re-attaching must append: the runtime replays from the start, so a
    rewrite would drop whatever it has already rotated away."""
    manager, _clients = _build_serve_manager()
    adopted = tmp_path / "1.container.0.log"
    adopted.write_text("".join(f"l{i}\n" for i in range(1, 8)), encoding="utf-8")
    fresh = tmp_path / "2.container.0.log"

    # l1 and l2 rotated away, so the replay starts at l3 and still carries the
    # anchor (the file's last five lines).
    replay = [f"l{i}\n" for i in range(3, 9)]
    states = [SimpleNamespace(state=WorkloadStatusStateEnum.FAILED)]

    for log_path in (adopted, fresh):
        with (
            patch(
                "gpustack.worker.serve_manager.logs_workload",
                return_value=iter(replay),
            ),
            patch(
                "gpustack.worker.serve_manager.get_workload",
                side_effect=_get_workload_sequence(states),
            ),
        ):
            manager._persist_container_logs(
                "wl", str(log_path), _fake_stop_event(), resume=True
            )

    # l1 and l2 survive even though the runtime no longer has them, and the
    # replayed l3..l7 are not written a second time.
    assert adopted.read_text(encoding="utf-8") == "".join(
        f"l{i}\n" for i in range(1, 9)
    )
    # Nothing to resume from: behaves exactly like a first connect.
    assert fresh.read_text(encoding="utf-8") == "".join(replay)


@pytest.mark.parametrize(
    "adopted_tail, replayed_tail",
    [
        # A progress bar is one streamed line carrying bare '\r'; splitlines()
        # would break it into pieces that can never equal one streamed line.
        (
            "shards:  0%\rshards: 50%\rshards: 100%\n",
            "shards:  0%\rshards: 50%\rshards: 100%\n",
        ),
        # A worker killed mid-write leaves a fragment; the runtime replays that
        # line whole, so the fragment has to go or the two would be joined.
        ("INFO star", "INFO starting engine\n"),
    ],
)
def test_persist_container_logs_resume_matches_the_runtime_line_framing(
    tmp_path: Path, adopted_tail, replayed_tail
):
    """The anchor is compared against streamed items, so it has to be rebuilt on
    the same '\\n' framing the runtime uses."""
    manager, _clients = _build_serve_manager()
    log_path = tmp_path / "1.container.0.log"
    head = "".join(f"l{i}\n" for i in range(1, 8))
    log_path.write_text(head + adopted_tail, encoding="utf-8")

    # l1 and l2 rotated away; the replay still carries the anchor window.
    replay = [f"l{i}\n" for i in range(3, 8)] + [replayed_tail, "l8-NEW\n"]
    states = [SimpleNamespace(state=WorkloadStatusStateEnum.FAILED)]

    with (
        patch(
            "gpustack.worker.serve_manager.logs_workload",
            return_value=iter(replay),
        ),
        patch(
            "gpustack.worker.serve_manager.get_workload",
            side_effect=_get_workload_sequence(states),
        ),
    ):
        manager._persist_container_logs(
            "wl", str(log_path), _fake_stop_event(), resume=True
        )

    # Read as bytes: universal newlines would rewrite the bare '\r' under test.
    written = log_path.read_bytes().decode("utf-8")
    assert written == head + replayed_tail + "l8-NEW\n"


def test_persist_container_logs_resume_gives_up_on_an_unmatchable_anchor(
    tmp_path: Path,
):
    """An unreplayable anchor must not hold the live stream back, and giving up
    on it must not disable the ordinary reconnect dedupe: otherwise every later
    reconnect rewrites the file and loses what the runtime rotated away."""
    manager, _clients = _build_serve_manager()
    log_path = tmp_path / "1.container.0.log"
    log_path.write_text("".join(f"gone-{i}\n" for i in range(1, 8)), encoding="utf-8")

    rewritten = [f"kept-{i}\n" for i in range(7)]
    streams = [
        # The runtime replays a different container generation entirely.
        iter(["other-a\n", "other-b\n"]),
        # Given up on, so this connection rewrites.
        iter(rewritten),
        # An ordinary reconnect, with kept-0 rotated away: the anchor built from
        # the rewrite above still has to dedupe the replay.
        iter(rewritten[1:] + ["kept-7\n"]),
    ]
    states = [
        SimpleNamespace(state=WorkloadStatusStateEnum.RUNNING),
        SimpleNamespace(state=WorkloadStatusStateEnum.RUNNING),
        SimpleNamespace(state=WorkloadStatusStateEnum.FAILED),
    ]

    with (
        patch("gpustack.worker.serve_manager._LOG_RESUME_SKIP_TIMEOUT", -1),
        patch(
            "gpustack.worker.serve_manager.logs_workload",
            side_effect=lambda **kwargs: streams.pop(0),
        ),
        patch(
            "gpustack.worker.serve_manager.get_workload",
            side_effect=_get_workload_sequence(states),
        ),
    ):
        manager._persist_container_logs(
            "wl", str(log_path), _fake_stop_event(), resume=True
        )

    # kept-0 survives and kept-7 is appended: the anchor deduped the reconnect.
    assert log_path.read_text(encoding="utf-8") == "".join(rewritten + ["kept-7\n"])


def test_tail_lines_widens_past_a_record_longer_than_the_first_read(tmp_path: Path):
    """A final record longer than the first read leaves no whole line behind.
    Reporting "no anchor" there would reopen the adopted log in 'w'."""
    log_path = tmp_path / "1.container.0.log"
    head = [f"l{i}\n" for i in range(1, 5)]
    oversized = "CONFIG " + "x" * (_LOG_TAIL_CHUNK_SIZE * 2) + "\n"
    log_path.write_text("".join(head) + oversized)

    assert log_path.stat().st_size > _LOG_TAIL_CHUNK_SIZE
    assert _tail_lines(str(log_path), 5) == head + [oversized]


def test_tail_lines_reads_only_the_end_of_a_large_file(tmp_path: Path):
    """The chunked read must not hand back the line the chunk boundary cut."""
    log_path = tmp_path / "big.log"
    lines = [f"line-{i:06d}" + "x" * 80 + "\n" for i in range(1000)]
    log_path.write_text("".join(lines), encoding="utf-8")

    assert log_path.stat().st_size > _LOG_TAIL_CHUNK_SIZE
    assert _tail_lines(str(log_path), 5) == lines[-5:]


def test_adoption_reattaches_container_log_persistence(tmp_path: Path):
    """A worker restart kills the log persistence threads while the workload
    keeps running, and the replayed CREATED event returns early for an
    already-running instance. The periodic sync has to re-attach them, without
    restarting the instance and without disturbing a thread still alive."""
    manager, clientset = _build_serve_manager()
    manager._serve_log_dir = str(_write_serve_logs(tmp_path))

    model_instance = new_model_instance(
        1, "qwen3-0.6b", 1, worker_id=1, state=ModelInstanceStateEnum.RUNNING
    )
    clientset.model_instances.list.return_value = SimpleNamespace(
        items=[model_instance]
    )
    model = new_model(1, "test", 1, huggingface_repo_id="Qwen/Qwen2.5-0.5B-Instruct")
    model.backend = BackendEnum.VLLM
    model.backend_version = "0.8.0"

    def sync():
        with (
            patch(
                "gpustack.worker.serve_manager.get_workload",
                return_value=SimpleNamespace(state="running"),
            ),
            patch.object(manager, "_is_provisioning", return_value=False),
            patch.object(manager, "_get_model", return_value=model),
            patch.object(manager, "_start_container_log_persistence") as start_logs,
            patch.object(manager, "_start_model_instance") as start_instance,
        ):
            manager.sync_model_instances_state()
        return start_logs, start_instance

    # A fresh worker process has no threads registered: re-attach, don't restart.
    start_logs, start_instance = sync()
    start_logs.assert_called_once_with(model_instance, resume=True)
    start_instance.assert_not_called()

    # A live main log thread is left alone: two would interleave their appends.
    manager._log_persistence[1] = _log_persistence(main_alive=True)
    start_logs, _ = sync()
    start_logs.assert_not_called()

    # A dead main thread beside the forever-polling discovery thread re-attaches;
    # folding the two together would never fire here.
    manager._log_persistence[1] = _log_persistence(main_alive=False)
    start_logs, _ = sync()
    start_logs.assert_called_once_with(model_instance, resume=True)


def test_stop_container_log_persistence_tears_down_the_whole_generation():
    """The main thread is tracked apart from the sidecar ones, but stopping still
    has to signal and join every one of them."""
    manager, _clients = _build_serve_manager()
    main_stop_event, sidecar_stop_event = MagicMock(), MagicMock()
    main_log_thread, sidecar_thread = _fake_thread(True), _fake_thread(True)
    persistence = _LogPersistence(main_stop_event, main_log_thread)
    persistence.add_aux_thread(sidecar_thread, sidecar_stop_event)
    manager._log_persistence[1] = persistence

    manager._stop_container_log_persistence(1)

    main_stop_event.set.assert_called_once()
    sidecar_stop_event.set.assert_called_once()
    main_log_thread.join.assert_called_once_with(timeout=2.0)
    sidecar_thread.join.assert_called_once_with(timeout=2.0)
    assert manager._log_persistence == {}

    # A sidecar discovered after teardown began is signalled, not orphaned.
    late_stop_event = MagicMock()
    persistence.add_aux_thread(_fake_thread(True), late_stop_event)
    late_stop_event.set.assert_called_once()


def test_starting_log_persistence_retires_the_previous_generation(tmp_path: Path):
    """Starts arrive from both the watch thread and the periodic sync thread. Were
    the registry swap and the teardown it replaces not one critical section, the
    loser's threads would keep running with nothing able to signal them."""
    manager, _clients = _build_serve_manager()
    manager._serve_log_dir = str(_write_serve_logs(tmp_path))
    model_instance = new_model_instance(
        1, "qwen3-0.6b", 1, worker_id=1, state=ModelInstanceStateEnum.RUNNING
    )

    retire = manager._retire_log_persistence
    held_while_retiring = []

    def probe_lock(model_instance_id, timeout=2.0):
        # A non-reentrant lock refuses its own holder, so failing to take it here
        # proves the caller is already inside the critical section.
        acquired = manager._log_persistence_lock.acquire(blocking=False)
        if acquired:
            manager._log_persistence_lock.release()
        held_while_retiring.append(not acquired)
        return retire(model_instance_id, timeout)

    with (
        patch("gpustack.worker.serve_manager.logs_workload", return_value=iter([])),
        patch("gpustack.worker.serve_manager.get_workload", return_value=None),
        patch.object(manager, "_retire_log_persistence", side_effect=probe_lock),
    ):
        manager._start_container_log_persistence(model_instance)
        first = manager._log_persistence[1]
        manager._start_container_log_persistence(model_instance)
        second = manager._log_persistence[1]

        assert held_while_retiring == [True, True]
        assert first is not second
        assert first.stop_event.is_set()
        assert not second.stop_event.is_set()

        manager._stop_container_log_persistence(1)

    assert second.stop_event.is_set()
    assert manager._log_persistence == {}


def test_sync_retires_log_persistence_the_server_no_longer_assigns(tmp_path: Path):
    """A DELETED event landing mid-pass leaves a generation nothing will ever
    stop, and the persistence loop retries a missing workload forever rather than
    winding down, so the reconciler has to retire it."""
    manager, clientset = _build_serve_manager()
    manager._serve_log_dir = str(_write_serve_logs(tmp_path))

    kept = new_model_instance(
        1, "qwen3-0.6b", 1, worker_id=1, state=ModelInstanceStateEnum.RUNNING
    )
    clientset.model_instances.list.return_value = SimpleNamespace(items=[kept])
    manager._log_persistence[1] = _log_persistence(main_alive=True)
    manager._log_persistence[99] = _log_persistence(main_alive=True)
    orphan_stop_event = manager._log_persistence[99].stop_event

    model = new_model(1, "test", 1, huggingface_repo_id="Qwen/Qwen2.5-0.5B-Instruct")
    model.backend = BackendEnum.VLLM
    model.backend_version = "0.8.0"

    with (
        patch(
            "gpustack.worker.serve_manager.get_workload",
            return_value=SimpleNamespace(state="running"),
        ),
        # Provisioning: still assigned, so its generation must be left alone.
        patch.object(manager, "_is_provisioning", return_value=True),
        patch.object(manager, "_get_model", return_value=model),
    ):
        manager.sync_model_instances_state()

    assert set(manager._log_persistence) == {1}
    orphan_stop_event.set.assert_called_once()


def test_adoption_aligns_legacy_main_log_with_restart_count(tmp_path: Path):
    """A legacy main log counts as restart 0, so without the rename the viewer
    would file it under a different restart than the container log."""
    serve_dir = _write_serve_logs(tmp_path, "1.log")
    manager, _clients = _build_serve_manager()
    manager._serve_log_dir = str(serve_dir)

    model_instance = new_model_instance(
        1, "qwen3-0.6b", 1, worker_id=1, state=ModelInstanceStateEnum.RUNNING
    )
    model_instance.restart_count = 5

    with patch.object(manager, "_start_container_log_persistence") as start_logs:
        manager._ensure_container_log_persistence(model_instance)
        # A later adoption with the target already in place must not overwrite it.
        (serve_dir / "1.log").write_text("newer owner", encoding="utf-8")
        manager._ensure_container_log_persistence(model_instance)

    assert start_logs.call_count == 2
    start_logs.assert_called_with(model_instance, resume=True)
    assert sorted(p.name for p in serve_dir.iterdir()) == ["1.5.log", "1.log"]
    assert (serve_dir / "1.5.log").read_text(encoding="utf-8") == "x"


# --- vGPU allocation read-back (gpu_type_selector) ---

from gpustack.schemas.models import (  # noqa: E402
    ComputedResourceClaim,
    GPUTypeSelector,
)
from gpustack.worker.serve_manager import (  # noqa: E402
    _ALLOCATED_ACCELERATORS_ANNOTATION,
    _parse_allocated_accelerators,
)

_VGPU_ANNOTATION_VALUE = (
    '{"run-0": {"devices": {"groups": [{"id": "a100", "manufacturer": "nvidia",'
    ' "accelerators": [{"id": "GPU-uuid-1", "index": 1, "mode": 3,'
    ' "allocated": 640000}]}]}, "deviceIDs": ["a100:GPU-uuid-1:0124"]}}'
)


def test_parse_allocated_accelerators():
    accelerators = _parse_allocated_accelerators(
        {_ALLOCATED_ACCELERATORS_ANNOTATION: _VGPU_ANNOTATION_VALUE}
    )
    assert [a["id"] for a in accelerators] == ["GPU-uuid-1"]
    assert accelerators[0]["index"] == 1


def test_parse_allocated_accelerators_tolerates_skew():
    assert _parse_allocated_accelerators(None) == []
    assert _parse_allocated_accelerators({}) == []
    assert (
        _parse_allocated_accelerators({_ALLOCATED_ACCELERATORS_ANNOTATION: "not json"})
        == []
    )
    assert (
        _parse_allocated_accelerators({_ALLOCATED_ACCELERATORS_ANNOTATION: '["x"]'})
        == []
    )
    assert (
        _parse_allocated_accelerators(
            {_ALLOCATED_ACCELERATORS_ANNOTATION: '{"c": "unexpected"}'}
        )
        == []
    )


def _build_vgpu_manager(worker_id=1, device_index=1):
    clientset = MagicMock()
    clientset.model_instances.list.return_value = SimpleNamespace(items=[])
    clientset.workers.get.return_value = SimpleNamespace(
        status=SimpleNamespace(
            gpu_devices=[SimpleNamespace(uuid="GPU-uuid-1", index=device_index)]
        )
    )
    cfg = SimpleNamespace(log_dir="/tmp")
    manager = ServeManager(lambda: worker_id, lambda: clientset, cfg)
    manager._inference_backend_manager = MagicMock()
    return manager, clientset


def _vgpu_model():
    model = new_model(1, "test", 1, huggingface_repo_id="Qwen/Qwen2.5-0.5B-Instruct")
    model.backend = BackendEnum.VLLM
    model.backend_version = "0.8.0"
    model.gpu_type_selector = GPUTypeSelector(
        type="pool-a100",
        accelerator_sliced_memory_percentage=50,
        accelerator_sliced_cores_percentage=50,
    )
    return model


def test_sync_vgpu_allocation_patches_main_addresses_and_rekeys_claim():
    manager, clientset = _build_vgpu_manager(worker_id=1, device_index=1)

    model_instance = new_model_instance(
        1,
        "vgpu-instance",
        1,
        worker_id=1,
        state=ModelInstanceStateEnum.RUNNING,
        computed_resource_claim=ComputedResourceClaim(vram={0: 42949672960}),
    )
    clientset.model_instances.list.return_value = SimpleNamespace(
        items=[model_instance]
    )

    workload = SimpleNamespace(
        state="Running",
        annotations={_ALLOCATED_ACCELERATORS_ANNOTATION: _VGPU_ANNOTATION_VALUE},
    )

    with (
        patch("gpustack.worker.serve_manager.get_workload", return_value=workload),
        patch.object(manager, "_is_provisioning", return_value=False),
        patch.object(manager, "_get_model", return_value=_vgpu_model()),
        patch.object(manager, "_update_model_instance") as update_model_instance,
    ):
        manager.sync_model_instances_state()

    update_model_instance.assert_called_once()
    _, kwargs = update_model_instance.call_args
    assert kwargs["gpu_addresses"] == ["GPU-uuid-1"]
    # The allocated card's index is backfilled for display and accounting.
    assert kwargs["gpu_indexes"] == [1]
    # Placeholder key 0 re-keyed to the allocated card index 1.
    assert kwargs["computed_resource_claim"].vram == {1: 42949672960}


def test_sync_vgpu_allocation_patches_subordinate_worker():
    manager, clientset = _build_vgpu_manager(worker_id=2, device_index=3)

    sw = ModelInstanceSubordinateWorker(
        worker_id=2,
        worker_name="worker-2",
        worker_ip="10.0.0.2",
        state=ModelInstanceStateEnum.RUNNING,
        computed_resource_claim=ComputedResourceClaim(vram={0: 42949672960}),
    )
    model_instance = new_model_instance(
        1,
        "vgpu-distributed",
        1,
        worker_id=1,
        state=ModelInstanceStateEnum.RUNNING,
    )
    model_instance.distributed_servers = DistributedServers(
        mode=DistributedServerCoordinateModeEnum.RUN_FIRST,
        subordinate_workers=[sw],
    )
    clientset.model_instances.list.return_value = SimpleNamespace(
        items=[model_instance]
    )

    workload = SimpleNamespace(
        state="Running",
        annotations={_ALLOCATED_ACCELERATORS_ANNOTATION: _VGPU_ANNOTATION_VALUE},
    )

    with (
        patch("gpustack.worker.serve_manager.get_workload", return_value=workload),
        patch.object(manager, "_is_provisioning", return_value=False),
        patch.object(manager, "_get_model", return_value=_vgpu_model()),
        patch.object(manager, "_update_model_instance") as update_model_instance,
    ):
        manager.sync_model_instances_state()

    update_model_instance.assert_called_once()
    args, _ = update_model_instance.call_args
    assert args[0] == model_instance.id
    patched_sw = update_model_instance.call_args.kwargs[
        "distributed_servers.subordinate_workers.0"
    ]
    assert patched_sw.gpu_addresses == ["GPU-uuid-1"]
    assert patched_sw.gpu_indexes == [3]
    assert patched_sw.computed_resource_claim.vram == {3: 42949672960}


def test_sync_vgpu_allocation_noop_without_annotation():
    manager, clientset = _build_vgpu_manager(worker_id=1)

    model_instance = new_model_instance(
        1,
        "vgpu-instance",
        1,
        worker_id=1,
        state=ModelInstanceStateEnum.RUNNING,
        computed_resource_claim=ComputedResourceClaim(vram={0: 42949672960}),
    )
    clientset.model_instances.list.return_value = SimpleNamespace(
        items=[model_instance]
    )

    workload = SimpleNamespace(state="Running", annotations={})

    with (
        patch("gpustack.worker.serve_manager.get_workload", return_value=workload),
        patch.object(manager, "_is_provisioning", return_value=False),
        patch.object(manager, "_get_model", return_value=_vgpu_model()),
        patch.object(manager, "_update_model_instance") as update_model_instance,
    ):
        manager.sync_model_instances_state()

    update_model_instance.assert_not_called()


def test_error_state_surfaces_workload_state_message():
    manager, clientset = _build_vgpu_manager(worker_id=1)

    model_instance = new_model_instance(
        1,
        "vgpu-instance",
        1,
        worker_id=1,
        state=ModelInstanceStateEnum.INITIALIZING,
    )
    clientset.model_instances.list.return_value = SimpleNamespace(
        items=[model_instance]
    )

    workload = SimpleNamespace(
        state="Failed",
        state_message="Allocate failed due to no enough sliced units",
    )

    with (
        patch("gpustack.worker.serve_manager.get_workload", return_value=workload),
        patch.object(manager, "_is_provisioning", return_value=False),
        patch.object(manager, "_update_model_instance") as update_model_instance,
    ):
        manager.sync_model_instances_state()

    update_model_instance.assert_called_once_with(
        model_instance.id,
        state=ModelInstanceStateEnum.ERROR,
        state_message="Allocate failed due to no enough sliced units",
    )


def test_error_state_falls_back_to_generic_message():
    manager, clientset = _build_vgpu_manager(worker_id=1)

    model_instance = new_model_instance(
        1,
        "vgpu-instance",
        1,
        worker_id=1,
        state=ModelInstanceStateEnum.INITIALIZING,
    )
    clientset.model_instances.list.return_value = SimpleNamespace(
        items=[model_instance]
    )

    with (
        patch(
            "gpustack.worker.serve_manager.get_workload",
            return_value=SimpleNamespace(state="Failed"),
        ),
        patch.object(manager, "_is_provisioning", return_value=False),
        patch.object(manager, "_update_model_instance") as update_model_instance,
    ):
        manager.sync_model_instances_state()

    update_model_instance.assert_called_once_with(
        model_instance.id,
        state=ModelInstanceStateEnum.ERROR,
        state_message="Inference server exited or unhealthy.",
    )


def _workload_exit(name="default", exit_code=None, reason=""):
    return SimpleNamespace(name=name, exit_code=exit_code, reason=reason)


def test_error_state_surfaces_the_container_exit_code():
    """gpustack/gpustack#4217: the exit code must reach the instance, not just
    the runtime's workload status."""
    manager, clientset = _build_vgpu_manager(worker_id=1)

    model_instance = new_model_instance(
        1,
        "vgpu-instance",
        1,
        worker_id=1,
        state=ModelInstanceStateEnum.INITIALIZING,
    )
    clientset.model_instances.list.return_value = SimpleNamespace(
        items=[model_instance]
    )

    workload = SimpleNamespace(
        state="Failed",
        state_message="Error",
        exits=[_workload_exit(exit_code=7, reason="Error")],
    )

    with (
        patch("gpustack.worker.serve_manager.get_workload", return_value=workload),
        patch.object(manager, "_is_provisioning", return_value=False),
        patch.object(manager, "_update_model_instance") as update_model_instance,
    ):
        manager.sync_model_instances_state()

    update_model_instance.assert_called_once_with(
        model_instance.id,
        state=ModelInstanceStateEnum.ERROR,
        state_message="Error (exit code 7)",
    )


def test_workload_failure_appends_the_exit_code():
    """Both backends must answer gpustack/gpustack#4217, and they arrive at it
    differently: Docker names the reason on state_message, Kubernetes leaves it
    empty and the exits are the only source."""
    assert (
        _describe_workload_failure(
            SimpleNamespace(
                state_message="OOMKilled",
                exits=[_workload_exit(exit_code=137, reason="OOMKilled")],
            )
        )
        == "OOMKilled (exit code 137)"
    )
    assert (
        _describe_workload_failure(
            SimpleNamespace(
                state_message="",
                exits=[_workload_exit(exit_code=1, reason="Error")],
            )
        )
        == "Error (exit code 1)"
    )


def test_workload_failure_keeps_the_image_pull_diagnosis():
    """gpustack/gpustack#5869: a container blocked on its image pull never
    terminated, so it has no exit code, and its state message already carries
    the registry error the Pod's Events explain it with. Nothing may displace
    or pad it."""
    message = (
        'ImagePullBackOff: Back-off pulling image "registry.invalid/nope:latest"; '
        "Failed to pull image: not found"
    )

    assert (
        _describe_workload_failure(
            SimpleNamespace(
                state_message=message,
                exits=[_workload_exit(reason="ImagePullBackOff")],
            )
        )
        == message
    )


def test_workload_failure_names_each_container_when_several_exit():
    assert (
        _describe_workload_failure(
            SimpleNamespace(
                state_message="",
                exits=[
                    _workload_exit(exit_code=1, reason="Error"),
                    _workload_exit(name="ray-head", exit_code=137, reason="OOMKilled"),
                ],
            )
        )
        == "Error, OOMKilled (default exit code 1, ray-head exit code 137)"
    )


def test_workload_failure_falls_back_when_the_workload_explains_nothing():
    # A workload reaped out from under the sync, and one that failed without a
    # message or any exit entry (the pre-0.2.3 shape, which has no `exits` at
    # all) both land on the generic message.
    assert _describe_workload_failure(None) == "Inference server exited or unhealthy."
    assert (
        _describe_workload_failure(SimpleNamespace(state="Failed"))
        == "Inference server exited or unhealthy."
    )
    assert (
        _describe_workload_failure(
            SimpleNamespace(state_message="", exits=[_workload_exit(exit_code=2)])
        )
        == "Inference server exited or unhealthy. (exit code 2)"
    )


def test_sync_vgpu_allocation_defers_when_card_not_in_reported_devices():
    """M6: an allocated UUID missing from the worker's reported devices must
    not leave the claim charged to a wrong placeholder index — defer instead
    of patching."""
    manager, clientset = _build_vgpu_manager(worker_id=1, device_index=1)
    # The reported device carries a different UUID than the annotation's.
    clientset.workers.get.return_value = SimpleNamespace(
        status=SimpleNamespace(gpu_devices=[SimpleNamespace(uuid="GPU-other", index=1)])
    )

    model_instance = new_model_instance(
        1,
        "vgpu-instance",
        1,
        worker_id=1,
        state=ModelInstanceStateEnum.RUNNING,
        computed_resource_claim=ComputedResourceClaim(vram={0: 42949672960}),
    )
    clientset.model_instances.list.return_value = SimpleNamespace(
        items=[model_instance]
    )

    workload = SimpleNamespace(
        state="Running",
        annotations={_ALLOCATED_ACCELERATORS_ANNOTATION: _VGPU_ANNOTATION_VALUE},
    )

    with (
        patch("gpustack.worker.serve_manager.get_workload", return_value=workload),
        patch.object(manager, "_is_provisioning", return_value=False),
        patch.object(manager, "_get_model", return_value=_vgpu_model()),
        patch.object(manager, "_update_model_instance") as update_model_instance,
    ):
        manager.sync_model_instances_state()

    for call in update_model_instance.call_args_list:
        _, kwargs = call
        assert "gpu_addresses" not in kwargs
        assert "gpu_indexes" not in kwargs
        assert "computed_resource_claim" not in kwargs


def test_sync_vgpu_allocation_steady_state_skips_worker_fetch():
    """M7: once addresses, indexes and the re-keyed claim agree with the
    annotation, the sync must not call the workers API again."""
    manager, clientset = _build_vgpu_manager(worker_id=1, device_index=1)

    model_instance = new_model_instance(
        1,
        "vgpu-instance",
        1,
        worker_id=1,
        state=ModelInstanceStateEnum.RUNNING,
        computed_resource_claim=ComputedResourceClaim(vram={1: 42949672960}),
    )
    model_instance.gpu_addresses = ["GPU-uuid-1"]
    model_instance.gpu_indexes = [1]
    clientset.model_instances.list.return_value = SimpleNamespace(
        items=[model_instance]
    )

    workload = SimpleNamespace(
        state="Running",
        annotations={_ALLOCATED_ACCELERATORS_ANNOTATION: _VGPU_ANNOTATION_VALUE},
    )

    with (
        patch("gpustack.worker.serve_manager.get_workload", return_value=workload),
        patch.object(manager, "_is_provisioning", return_value=False),
        patch.object(manager, "_get_model", return_value=_vgpu_model()),
        patch.object(manager, "_update_model_instance"),
    ):
        manager.sync_model_instances_state()

    clientset.workers.get.assert_not_called()
