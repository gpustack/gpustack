import multiprocessing
import tracemalloc
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from gpustack.cmd import start
from gpustack.api.exceptions import BadRequestException
from gpustack.routes.debug import get_memory_profile


@pytest.mark.parametrize("debug", [False, True])
@pytest.mark.parametrize("profiling", [False, True])
@pytest.mark.parametrize("worker", [False, True])
def test_memory_profiling_is_independent_of_debug(
    monkeypatch, debug, profiling, worker
):
    cfg = SimpleNamespace(debug=debug, server_url="http://server" if worker else None)
    monkeypatch.setattr(start, "parse_args", lambda args: cfg)
    for name in (
        "export_insecure_tls_env",
        "setup_logging",
        "debug_env_info",
        "set_third_party_env",
        "initialize_gateway",
    ):
        monkeypatch.setattr(start, name, Mock())
    monkeypatch.setattr(start, "resolve_version_info", lambda: ("test", "test"))
    monkeypatch.setattr(start.multiprocessing, "set_start_method", Mock())
    monkeypatch.setattr(start.envs, "DEBUG_ENABLE_TRACEMALLOC", profiling)
    trace = Mock()
    monkeypatch.setattr("tracemalloc.start", trace)
    monkeypatch.setattr("tracemalloc.is_tracing", lambda: False)

    def run(cfg):
        # Profiling must be configured before either runtime starts.
        assert trace.call_count == int(profiling)

    server_run = Mock(side_effect=run)
    worker_run = Mock(side_effect=run)
    monkeypatch.setattr(start, "run_server", server_run)
    monkeypatch.setattr(start, "run_worker", worker_run)
    start.run(SimpleNamespace())

    assert server_run.call_count == int(not worker)
    assert worker_run.call_count == int(worker)


def test_repeated_initialization_preserves_existing_tracing(monkeypatch):
    monkeypatch.setattr(start.envs, "DEBUG_ENABLE_TRACEMALLOC", True)
    tracing = False

    def begin_tracing():
        nonlocal tracing
        tracing = True

    trace = Mock(side_effect=begin_tracing)
    log = Mock()
    monkeypatch.setattr(tracemalloc, "start", trace)
    monkeypatch.setattr(tracemalloc, "is_tracing", lambda: tracing)
    monkeypatch.setattr(start.logger, "info", log)
    start.start_tracemalloc_if_enabled()
    start.start_tracemalloc_if_enabled()
    trace.assert_called_once_with()
    log.assert_called_once()


def _probe_spawned_worker(connection, debug):
    before = tracemalloc.is_tracing()

    class ProbeWorker:
        def __init__(self, cfg):
            self.tracing_at_init = tracemalloc.is_tracing()

        def start(self):
            try:
                profile_available = "top_memory_lines" in get_memory_profile()
            except BadRequestException:
                profile_available = False
            connection.send(
                {
                    "before": before,
                    "enabled": start.envs.DEBUG_ENABLE_TRACEMALLOC,
                    "tracing_at_init": self.tracing_at_init,
                    "profile_available": profile_available,
                }
            )

    try:
        # Exercise the worker entry point without starting hardware or network services.
        with patch.object(start, "Worker", ProbeWorker):
            start.run_worker(SimpleNamespace(debug=debug, server_url=None))
    finally:
        connection.close()


@pytest.mark.parametrize("debug", [False, True])
@pytest.mark.parametrize("profiling", [False, True])
def test_spawned_worker_initializes_its_own_memory_tracing(
    monkeypatch, debug, profiling
):
    monkeypatch.setenv("GPUSTACK_DEBUG_ENABLE_TRACEMALLOC", str(profiling).lower())
    monkeypatch.delenv("PYTHONTRACEMALLOC", raising=False)
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_probe_spawned_worker, args=(send, debug))
    try:
        process.start()
        send.close()
        assert receive.poll(30), "Worker did not report its profiling state"
        result = receive.recv()
        process.join(timeout=10)
        assert process.exitcode == 0
        assert result == {
            "before": False,
            "enabled": profiling,
            "tracing_at_init": profiling,
            "profile_available": profiling,
        }
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        receive.close()
        send.close()
