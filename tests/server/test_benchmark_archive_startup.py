from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from gpustack.server import benchmark_monitoring_archiver
from gpustack.server.server import Server


def test_archive_scheduler_starts_without_storage_configuration(monkeypatch):
    server = object.__new__(Server)
    server._config = SimpleNamespace()
    server._benchmark_monitoring_archiver_task = None
    archiver = SimpleNamespace(run=AsyncMock())
    constructor = Mock(return_value=archiver)
    monkeypatch.setattr(
        benchmark_monitoring_archiver, 'BenchmarkMonitoringArchiver', constructor
    )

    def start(coroutine):
        coroutine.close()
        return SimpleNamespace(done=lambda: False)

    server._create_async_task = Mock(side_effect=start)
    server._start_benchmark_monitoring_archiver()
    server._start_benchmark_monitoring_archiver()
    constructor.assert_called_once_with(server._config)
    server._create_async_task.assert_called_once()
