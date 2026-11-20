"""How the sync Kubernetes client's wire dumps are logged.

``kubernetes.client.rest`` dumps full request and response bodies at DEBUG for
every API call; the filter registered by ``setup_logging`` keeps those records
silent until trace is enabled and relabels them to TRACE when it is.
"""

import logging

import pytest

from gpustack.logging import (
    KUBERNETES_REST_LOGGER_NAME,
    TRACE_LEVEL,
    WireDumpTraceFilter,
    setup_logging,
)


def wire_dump_record(level=logging.DEBUG):
    return logging.LogRecord(
        name=KUBERNETES_REST_LOGGER_NAME,
        level=level,
        pathname=__file__,
        lineno=1,
        msg="response body: %s",
        args=('{"kind":"PodList"}',),
        exc_info=None,
    )


@pytest.mark.parametrize(
    "root_level, visible",
    [
        (logging.INFO, False),
        (logging.DEBUG, False),
        (TRACE_LEVEL, True),
    ],
)
def test_wire_dump_gated_by_trace_level(root_level, visible):
    root = logging.getLogger()
    saved_level = root.level
    record = wire_dump_record()
    try:
        root.setLevel(root_level)
        passed = WireDumpTraceFilter().filter(record)
    finally:
        root.setLevel(saved_level)
    assert passed is visible
    if visible:
        assert record.levelno == TRACE_LEVEL
        assert record.levelname == "TRACE"


@pytest.mark.parametrize("level", [logging.WARNING, logging.ERROR])
def test_wire_dump_filter_passes_non_debug_records(level):
    root = logging.getLogger()
    saved_level = root.level
    record = wire_dump_record(level)
    try:
        root.setLevel(logging.INFO)
        passed = WireDumpTraceFilter().filter(record)
    finally:
        root.setLevel(saved_level)
    assert passed is True
    assert record.levelno == level


def test_kubernetes_rest_logger_gates_wire_dumps(caplog):
    kubernetes_logger = logging.getLogger(KUBERNETES_REST_LOGGER_NAME)
    wire_dump_filter = WireDumpTraceFilter()
    kubernetes_logger.addFilter(wire_dump_filter)
    try:
        with caplog.at_level(logging.DEBUG):
            kubernetes_logger.debug("response body: %s", '{"kind":"PodList"}')
            with caplog.at_level(TRACE_LEVEL):
                kubernetes_logger.debug("response body: %s", '{"kind":"PodList"}')
        records = [
            record
            for record in caplog.records
            if record.name == KUBERNETES_REST_LOGGER_NAME
        ]
        assert [record.levelname for record in records] == ["TRACE"]
    finally:
        kubernetes_logger.removeFilter(wire_dump_filter)


def test_setup_logging_gates_kubernetes_wire_dumps(caplog):
    root = logging.getLogger()
    saved_root_level = root.level
    saved_root_handlers = list(root.handlers)
    kubernetes_logger = logging.getLogger(KUBERNETES_REST_LOGGER_NAME)
    saved_filters = list(kubernetes_logger.filters)
    saved_disabled = {
        name: existing.disabled
        for name, existing in logging.root.manager.loggerDict.items()
        if isinstance(existing, logging.Logger)
    }
    try:
        setup_logging(debug=True)
        assert any(
            isinstance(attached, WireDumpTraceFilter)
            for attached in kubernetes_logger.filters
        )
        with caplog.at_level(TRACE_LEVEL):
            kubernetes_logger.debug("response body: %s", '{"kind":"PodList"}')
        records = [
            record
            for record in caplog.records
            if record.name == KUBERNETES_REST_LOGGER_NAME
        ]
        assert [record.levelname for record in records] == ["TRACE"]
    finally:
        root.setLevel(saved_root_level)
        for handler in list(root.handlers):
            if handler not in saved_root_handlers:
                root.removeHandler(handler)
        kubernetes_logger.filters[:] = saved_filters
        for name, existing in logging.root.manager.loggerDict.items():
            if isinstance(existing, logging.Logger):
                existing.disabled = saved_disabled.get(name, False)
