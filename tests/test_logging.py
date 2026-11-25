"""Logging configuration and structured application output.

Both Kubernetes clients log full request and response bodies at DEBUG for
every API call. During workload state sync those dumps crowd out GPUStack's
own --debug output, so both clients must stay disabled whatever the level.
"""

import json
import logging
from types import SimpleNamespace

import pytest
import uvicorn

from gpustack import logging as app_logging


KUBERNETES_REST_LOGGERS = (
    "kubernetes.client.rest",
    "kubernetes_asyncio.client.rest",
)


@pytest.fixture
def restore_logging_state():
    root = logging.getLogger()
    saved_level = root.level
    saved_handlers = list(root.handlers)
    saved_disabled = {
        name: logger.disabled
        for name, logger in logging.root.manager.loggerDict.items()
        if isinstance(logger, logging.Logger)
    }
    try:
        yield
    finally:
        root.setLevel(saved_level)
        for handler in list(root.handlers):
            if handler not in saved_handlers:
                root.removeHandler(handler)
        for name, logger in logging.root.manager.loggerDict.items():
            if isinstance(logger, logging.Logger):
                logger.disabled = saved_disabled.get(name, False)


@pytest.mark.usefixtures("restore_logging_state")
@pytest.mark.parametrize("debug", [True, False])
@pytest.mark.parametrize("logger_name", KUBERNETES_REST_LOGGERS)
def test_setup_logging_keeps_kubernetes_wire_dumps_disabled(debug, logger_name):
    logger = logging.getLogger(logger_name)
    logger.disabled = False

    app_logging.setup_logging(debug=debug)

    assert logger.disabled is True


@pytest.fixture
def configure_logging(monkeypatch):
    root = logging.getLogger()
    level = root.level
    configured = False

    def configure(
        tmp_path,
        role="server",
        *,
        endpoint="http://collector/v1/logs",
        pod_release=None,
    ):
        nonlocal configured
        if not configured:
            monkeypatch.setattr(root, "handlers", [])
            configured = True
        cfg = SimpleNamespace(
            debug=True,
            log_dir=str(tmp_path),
            log_otlp_endpoint=endpoint,
            log_pod_release=pod_release,
        )
        app_logging.setup_application_logging(cfg, role)
        return root.handlers[0]

    yield configure
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()
    root.setLevel(level)


def test_logging_without_endpoint_keeps_text(configure_logging, tmp_path, capsys):
    configure_logging(tmp_path, endpoint=None)

    logging.getLogger("test.application").info("hello")

    assert " - test.application - INFO - hello" in capsys.readouterr().err
    assert not (tmp_path / "application").exists()


def test_uvicorn_startup_keeps_structured_console_output(
    configure_logging, tmp_path, capsys
):
    handler = configure_logging(tmp_path)
    uvicorn.Config(
        app="gpustack.server.app:create_app",
        log_config=app_logging.uvicorn_log_config(),
    )

    logging.getLogger("test.application").info("API started")

    assert handler in logging.getLogger().handlers
    assert json.loads(capsys.readouterr().err)["message"] == "API started"


@pytest.mark.parametrize("role", ["server", "worker"])
@pytest.mark.parametrize("pod_release", [None, "inference"])
def test_application_log_preserves_context_on_console(
    configure_logging, tmp_path, capsys, role, pod_release
):
    handler = configure_logging(tmp_path, role, pod_release=pod_release)
    assert configure_logging(tmp_path, role, pod_release=pod_release) is handler
    logger = logging.getLogger("test.application")
    logger.info("starting")
    if role == "worker":
        app_logging.set_worker_log_identity(42, 7, "qa-worker")

    try:
        raise ValueError("test failure")
    except ValueError:
        logger.exception(
            "failed\n继续",
            extra={
                "request_id": "request-1",
                "model_route_id": 11,
                "trace_id": "a" * 32,
                "span_id": "b" * 16,
                "trace_flags": 1,
                "authorization": "never copy this field",
            },
        )

    console = capsys.readouterr().err
    assert not (tmp_path / "application").exists()
    start, failure = [json.loads(line) for line in console.splitlines()]
    assert start["service"] == f"gpustack.{role}"
    assert "worker_id" not in start
    assert failure["message"] == "failed\n继续"
    assert failure["severity"] == "ERROR"
    assert "ValueError: test failure" in failure["exception"]
    assert failure["request_id"] == "request-1"
    assert failure["trace_id"] == "a" * 32
    assert failure["span_id"] == "b" * 16
    assert failure["trace_flags"] == 1
    assert failure["model_route_id"] == 11
    assert "authorization" not in failure
    if role == "worker":
        assert failure["worker_id"] == 42
        assert failure["worker_name"] == "qa-worker"
        assert failure["cluster_id"] == 7
