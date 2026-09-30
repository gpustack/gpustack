"""Lifecycle and failure-isolation tests for opt-in OpenTelemetry tracing.

The server app (``gpustack.server.app.create_app``) is built with a custom
``lifespan``. Starlette runs a router's startup/shutdown handler lists only
through its default lifespan, so a flush registered on
``app.router.on_shutdown`` never executes for this app and spans still buffered
by the ``BatchSpanProcessor`` are dropped on exit. These tests pin the
behaviour the server relies on: batched spans are flushed when the app's own
lifespan ends, and a failed tracing setup never aborts startup.
"""

from contextlib import asynccontextmanager

import pytest

pytest.importorskip("opentelemetry.sdk.trace")
pytest.importorskip("opentelemetry.instrumentation.fastapi")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from opentelemetry.sdk.trace.export import BatchSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)

import gpustack.observability.tracing as tracing_mod  # noqa: E402


def _server_spans(exporter) -> list:
    """Server spans only: the ASGI instrumentation also emits INTERNAL spans
    for each ``send`` event, which are unrelated to this contract."""
    return [s for s in exporter.get_finished_spans() if s.kind.name == "SERVER"]


def _server_like_app() -> FastAPI:
    """A FastAPI app shaped like ``gpustack.server.app.create_app``'s.

    It owns a custom ``lifespan`` — Starlette uses it instead of its default
    lifespan — and releases tracing from that lifespan, the same way the server
    releases its HTTP clients.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        tracing_mod.shutdown_tracing(app)

    app = FastAPI(lifespan=lifespan)

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    return app


def test_batched_spans_flushed_when_app_lifespan_exits():
    """Spans buffered by the BatchSpanProcessor must be exported on shutdown."""
    exporter = InMemorySpanExporter()
    app = _server_like_app()
    provider = tracing_mod.setup_tracing(
        app,
        enabled=True,
        service_name="gpustack-test",
        exporter=exporter,
        processor_cls=BatchSpanProcessor,
        set_global=False,
    )
    assert provider is not None

    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        # The batch processor buffers by design: nothing is exported while the
        # request is being served.
        assert _server_spans(exporter) == []

    # The app's lifespan has exited, so the flush must have run.
    assert len(_server_spans(exporter)) == 1


def test_shutdown_tracing_is_a_noop_without_a_provider():
    """Tearing down tracing that never started must not raise."""
    app = FastAPI()
    tracing_mod.shutdown_tracing(app)
    assert getattr(app.state, "tracer_provider", None) is None


def test_setup_failure_does_not_abort_startup(monkeypatch, caplog):
    """A failure while wiring tracing leaves the server serving instead of
    aborting startup."""
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    def boom(*args, **kwargs):
        raise RuntimeError("simulated instrumentation failure")

    monkeypatch.setattr(FastAPIInstrumentor, "instrument_app", boom)
    app = FastAPI()

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    with caplog.at_level("WARNING"):
        provider = tracing_mod.setup_tracing(app, enabled=True)
    assert provider is None
    assert getattr(app.state, "tracer_provider", None) is None
    assert any("setup failed" in r.message for r in caplog.records)
    assert TestClient(app).get("/healthz").status_code == 200
