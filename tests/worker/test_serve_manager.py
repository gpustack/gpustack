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
from tests.utils.model import new_model, new_model_instance


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
