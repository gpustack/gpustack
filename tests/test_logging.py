"""Third-party loggers silenced by ``setup_logging``.

Both Kubernetes clients log full request and response bodies at DEBUG for
every API call. During workload state sync those dumps crowd out GPUStack's
own --debug output, so both clients must stay disabled whatever the level.
"""

import logging

import pytest

from gpustack.logging import setup_logging

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

    setup_logging(debug=debug)

    assert logger.disabled is True
