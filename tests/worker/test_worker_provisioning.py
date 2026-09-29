import threading
from unittest.mock import MagicMock, patch

from gpustack.worker.provisioning import _provision, run_provisioning


def test_cancelled_launch_does_not_spawn(tmp_path):
    cancelled = threading.Event()
    cancelled.set()
    with patch("gpustack.worker.provisioning.multiprocessing.get_context") as context:
        assert (
            run_provisioning(MagicMock(), tmp_path / "startup.log", cancelled) is None
        )
    context.assert_not_called()


def test_cancellation_terminates_and_reaps_the_child(tmp_path):
    cancelled = threading.Event()
    process = MagicMock()
    process.pid = 12345
    process.start.side_effect = cancelled.set
    process.is_alive.side_effect = [True, False]
    with (
        patch("gpustack.worker.provisioning.multiprocessing.get_context") as context,
        patch("gpustack.worker.provisioning.terminate_process_tree") as terminate,
    ):
        context.return_value.Process.return_value = process
        assert (
            run_provisioning(MagicMock(), tmp_path / "startup.log", cancelled) is None
        )
    terminate.assert_called_once_with(12345)
    process.join.assert_called_once()
    process.close.assert_called_once()


def test_provisioning_failure_is_returned_to_the_status_writer(tmp_path):
    process = MagicMock()
    process.is_alive.return_value = False
    process.exitcode = 1
    with patch("gpustack.worker.provisioning.multiprocessing.get_context") as context:
        context.return_value.Process.return_value = process
        assert (
            run_provisioning(MagicMock(), tmp_path / "startup.log", threading.Event())
            == 1
        )
    process.join.assert_called_once()
    process.close.assert_called_once()


def test_runtime_stdout_is_written_to_the_instance_startup_log(tmp_path):
    path = tmp_path / "startup.log"
    with (
        patch("gpustack.worker.provisioning.add_signal_handlers"),
        patch("gpustack.worker.provisioning.setup_logging"),
        patch("gpustack.worker.provisioning.setup_runtime_logging"),
        patch(
            "gpustack.worker.provisioning.create_workload",
            side_effect=lambda plan: print("Pulling image: 42%", flush=True),
        ),
    ):
        _provision(MagicMock(), str(path), False)
    assert "Pulling image: 42%" in path.read_text()
