from datetime import datetime, timezone
import json
import logging
import os
import sys
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from gpustack.config.config import Config

# Suppress warnings from transformers
# https://github.com/huggingface/transformers/issues/27214
# Note: This should be set before importing transformers
if "TRANSFORMERS_NO_ADVISORY_WARNINGS" not in os.environ:
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

TRACE_LEVEL = 5
_TEXT_HANDLER_NAME = "gpustack-console"


class _ApplicationLogFormatter(logging.Formatter):
    _fields = (
        "request_id",
        "pd_coordinator_id",
        "trace_id",
        "span_id",
        "trace_flags",
        "cluster_id",
        "worker_id",
        "worker_name",
        "model_id",
        "model_route_id",
        "model_instance_id",
        "owner_principal_id",
        "restart_count",
        "method",
        "path",
        "response_code",
        "duration",
    )

    def __init__(self, role: str):
        super().__init__()
        self.role = role
        self.identity: dict = {}

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(
                record.created, timezone.utc
            ).isoformat(),
            "severity": record.levelname,
            "service": f"gpustack.{self.role}",
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for field in self._fields:
            value = getattr(record, field, None)
            if isinstance(value, (str, int)):
                payload[field] = value
        payload.update(self.identity)
        return json.dumps(payload, ensure_ascii=False)


class _ApplicationLogHandler(logging.StreamHandler):
    def __init__(self, role: str):
        super().__init__()
        self.formatter = _ApplicationLogFormatter(role)


def setup_application_logging(cfg: "Config", role: str) -> None:
    """Configure structured application output on the container console."""
    setup_logging(cfg.debug)
    if not cfg.log_otlp_endpoint:
        return
    root = logging.getLogger()
    if any(isinstance(handler, _ApplicationLogHandler) for handler in root.handlers):
        return
    for handler in root.handlers[:]:
        if handler.name == _TEXT_HANDLER_NAME:
            root.removeHandler(handler)
            handler.close()
    root.addHandler(_ApplicationLogHandler(role))
    root.setLevel(logging.DEBUG if cfg.debug else logging.INFO)


def set_worker_log_identity(
    worker_id: int, cluster_id: Optional[int], name: str
) -> None:
    """Set Worker identity once its registration has been confirmed."""
    identity = {"worker_id": worker_id, "worker_name": name}
    if cluster_id is not None:
        identity["cluster_id"] = cluster_id
    for handler in logging.getLogger().handlers:
        if (
            isinstance(handler, _ApplicationLogHandler)
            and handler.formatter.role == "worker"
        ):
            handler.formatter.identity = identity


def uvicorn_log_config() -> Optional[dict]:
    """Preserve structured console output when an API listener starts."""
    # Uvicorn's default dictConfig closes previously installed handlers.
    if any(
        isinstance(handler, _ApplicationLogHandler)
        for handler in logging.getLogger().handlers
    ):
        return None
    from uvicorn.config import LOGGING_CONFIG

    return LOGGING_CONFIG


def setup_logging(debug: bool = False):
    level = logging.DEBUG if debug else logging.INFO

    if not logging.getLogger().handlers:
        handler = logging.StreamHandler()
        handler.name = _TEXT_HANDLER_NAME
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[handler],
        )
    logging.addLevelName(TRACE_LEVEL, "TRACE")
    logging.Logger.trace = trace

    logging.Formatter.formatTime = (
        lambda self, record, datefmt=None: datetime.fromtimestamp(
            record.created, timezone.utc
        ).astimezone()
    )

    # Third-party loggers to disable
    disable_logger_names = [
        "httpcore.connection",
        "httpcore.http11",
        "httpcore.proxy",
        "httpx",
        "httpcore2.connection",
        "httpcore2.http11",
        "httpcore2.proxy",
        "httpx2",
        "asyncio",
        "aiocache.base",
        "urllib3.connectionpool",
        "multipart.multipart",
        "apscheduler.scheduler",
        "apscheduler.executors.default",
        "tzlocal",
        "alembic.runtime.migration",
        "python_multipart.multipart",
        "filelock",
        "fastapi-cdn-host",
        "huggingface_hub.file_download",
        "docker.auth",
        "kubernetes_asyncio.client.rest",
        "kubernetes.client.rest",
        "azure.core.pipeline.policies.http_logging_policy",
        "websockets.client",
    ]

    for logger_name in disable_logger_names:
        logger = logging.getLogger(logger_name)
        logger.disabled = True

    # Third-party loggers to print on debug
    debug_logger_names = [
        "alembic.runtime.migration",
    ]

    for logger_name in debug_logger_names:
        logger = logging.getLogger(logger_name)
        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.disabled = True


def trace(self, message, *args, **kwargs):
    if self.isEnabledFor(TRACE_LEVEL):
        self._log(TRACE_LEVEL, message, args, **kwargs)


class RedirectStdoutStderr:
    def __init__(self, target):
        self.target = target

    def __enter__(self):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = self.target
        sys.stderr = self.target

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
