from datetime import datetime, timezone
import logging
import os
import sys


# Suppress warnings from transformers
# https://github.com/huggingface/transformers/issues/27214
# Note: This should be set before importing transformers
if "TRANSFORMERS_NO_ADVISORY_WARNINGS" not in os.environ:
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

TRACE_LEVEL = 5

KUBERNETES_REST_LOGGER_NAME = "kubernetes.client.rest"


class WireDumpTraceFilter(logging.Filter):
    """Gate sync Kubernetes client wire dumps behind the TRACE level.

    The sync Kubernetes client logs full request and response bodies at DEBUG
    for every API call. During worker workload state sync this floods --debug
    output with multi-kilobyte pod list dumps, so DEBUG records from this
    logger are dropped unless the root logger is at TRACE, and are relabeled
    from DEBUG to TRACE when it is. Non-DEBUG records pass unchanged.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno != logging.DEBUG:
            return True
        if not logging.getLogger().isEnabledFor(TRACE_LEVEL):
            return False
        record.levelno = TRACE_LEVEL
        record.levelname = logging.getLevelName(TRACE_LEVEL)
        return True


_wire_dump_trace_filter = WireDumpTraceFilter()


def setup_logging(debug: bool = False):
    level = logging.DEBUG if debug else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
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
        "azure.core.pipeline.policies.http_logging_policy",
        "websockets.client",
    ]

    for logger_name in disable_logger_names:
        logger = logging.getLogger(logger_name)
        logger.disabled = True

    # The async Kubernetes client logger above is disabled entirely, while the
    # sync one must keep non-DEBUG records visible, so its wire dumps are
    # gated by a filter instead. setup_logging runs multiple times per
    # process, so the filter is attached only once.
    kubernetes_rest_logger = logging.getLogger(KUBERNETES_REST_LOGGER_NAME)
    if _wire_dump_trace_filter not in kubernetes_rest_logger.filters:
        kubernetes_rest_logger.addFilter(_wire_dump_trace_filter)

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
