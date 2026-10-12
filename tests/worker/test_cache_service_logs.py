import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gpustack.worker.cache_service_logs import CacheServiceLogManager


def instance():
    return SimpleNamespace(
        id=11,
        name="cache-test",
        get_deployment_metadata=lambda: SimpleNamespace(name="cache-svc-5-i11"),
    )


def test_keep_two_launches_and_remove_instance_archives(tmp_path):
    manager = CacheServiceLogManager(str(tmp_path))
    paths = []
    for _ in range(3):
        path = manager.prepare(instance())
        path.write_text("startup\n")
        paths.append(path)
    assert not paths[0].exists()
    assert all(path.exists() for path in paths[1:])
    assert manager.tracked_ids() == {11}
    manager.remove(11)
    assert not paths[-1].parent.parent.exists()
    assert manager.tracked_ids() == set()


def test_live_writer_fences_restart_and_cleanup(tmp_path):
    manager = CacheServiceLogManager(str(tmp_path))
    path = manager.prepare(instance())
    path.write_text("startup\n")
    thread = MagicMock()
    thread.is_alive.return_value = True
    stop = threading.Event()
    manager._writers[11] = (thread, stop, path)
    with pytest.raises(RuntimeError, match="still stopping"):
        manager.prepare(instance())
    manager.remove(11)
    assert stop.is_set()
    assert path.exists()
    thread.is_alive.return_value = False
    manager.remove(11)
    assert not path.exists()


def test_reattaching_writer_resumes_and_is_not_duplicated(tmp_path):
    manager = CacheServiceLogManager(str(tmp_path))
    path = manager.prepare(instance())
    entered = threading.Event()

    def persist(_name, _path, stop, resume):
        assert resume
        entered.set()
        stop.wait(2)

    with patch.object(
        manager, "_persist_container_logs", side_effect=persist
    ) as persist_log:
        manager.ensure(instance(), path)
        assert entered.wait(1)
        manager.ensure(instance(), path)
        assert persist_log.call_count == 1
        assert manager.stop(11)


@pytest.mark.parametrize("output_type", ["bytes", "text", "stream"])
def test_archive_drains_exit_output_and_resumes_without_duplicates(
    tmp_path, output_type
):
    manager = CacheServiceLogManager(str(tmp_path))
    path = manager.prepare(instance())
    lines = [
        b"2026-09-30T00:00:00Z booting\n",
        b"2026-09-30T00:00:01Z fatal: invalid config\n",
    ]

    def runtime_logs(**kwargs):
        assert kwargs["follow"] is False
        if output_type == "stream":
            return iter(lines)
        output = b"".join(lines)
        return output if output_type == "bytes" else output.decode()

    with patch(
        "gpustack.worker.container_logs.logs_workload", side_effect=runtime_logs
    ):
        assert manager.archive(instance(), path)
        assert manager.archive(instance(), path)
    assert (
        path.with_name("container.log").read_text()
        == "booting\nfatal: invalid config\n"
    )


def test_failed_archive_keeps_retry_possible(tmp_path):
    manager = CacheServiceLogManager(str(tmp_path))
    path = manager.prepare(instance())
    with patch(
        "gpustack.worker.container_logs.logs_workload",
        side_effect=RuntimeError("runtime unavailable"),
    ):
        assert not manager.archive(instance(), path)
