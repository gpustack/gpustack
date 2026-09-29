"""Tests for opt-in OpenTelemetry request tracing (issue #5609, stage 1).

These avoid importing the full gpustack package (which pulls heavy server deps)
and exercise ``setup_tracing`` against the real OTel SDK + FastAPI
instrumentation with an in-memory exporter.
"""

import pytest
pytest.importorskip("opentelemetry.sdk.trace")
pytest.importorskip("opentelemetry.instrumentation.fastapi")



import importlib
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

import gpustack.observability.tracing as tracing_mod


@pytest.fixture
def exporter():
    """Fresh in-memory exporter per test.

    setup_tracing is called with set_global=False and the resulting provider
    is passed straight to FastAPIInstrumentor, so tests never touch OTel's
    one-shot global provider and never collide with each other.
    """
    tracing_mod._tracer_provider = None
    exp = InMemorySpanExporter()
    yield exp
    tracing_mod._tracer_provider = None


def _instrument(app, exporter):
    return tracing_mod.setup_tracing(
        app,
        enabled=True,
        service_name="gpustack-test",
        exporter=exporter,
        processor_cls=SimpleSpanProcessor,
        set_global=False,
    )


def test_disabled_is_a_noop(exporter):
    """When tracing is off the app handles requests and installs no provider."""
    app = FastAPI()

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    provider = tracing_mod.setup_tracing(app, enabled=False)
    assert provider is None
    assert tracing_mod._tracer_provider is None
    client = TestClient(app)
    assert client.get("/healthz").status_code == 200


def test_enabled_emits_one_server_span_joined_to_inbound_trace(exporter):
    """A traceparent from the gateway makes the server span the same trace."""
    app = FastAPI()

    @app.get("/v1/models")
    def models():
        return {"data": []}

    exporter = InMemorySpanExporter()
    _instrument(app, exporter)
    client = TestClient(app)
    resp = client.get(
        "/v1/models",
        headers={
            "traceparent": (
                "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
            )
        },
    )
    assert resp.status_code == 200

    server_spans = [s for s in exporter.get_finished_spans() if s.kind.name == "SERVER"]
    assert len(server_spans) == 1
    span = server_spans[0]
    # Same trace id as the inbound W3C traceparent: end-to-end correlation.
    assert format(span.context.trace_id, "032x") == (
        "0af7651916cd43dd8448eb211c80319c"
    )
    assert span.attributes.get("http.method") == "GET"
    assert span.attributes.get("http.route") == "/v1/models"
    assert span.attributes.get("http.status_code") == 200


def test_request_id_is_attributed_when_present(exporter):
    """Envoy's x-request-id lands on the server span for request lookup."""
    app = FastAPI()

    @app.get("/v1/chat/completions")
    def chat():
        return {"ok": True}

    exporter = InMemorySpanExporter()
    _instrument(app, exporter)
    client = TestClient(app)
    client.get(
        "/v1/chat/completions",
        headers={"x-request-id": "req-abc-123"},
    )

    span = next(s for s in exporter.get_finished_spans() if s.kind.name == "SERVER")
    assert span.attributes.get("gpustack.request_id") == "req-abc-123"


def test_request_id_absent_is_fine(exporter):
    """A request without x-request-id still traces and simply lacks the attr."""
    app = FastAPI()

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    exporter = InMemorySpanExporter()
    _instrument(app, exporter)
    client = TestClient(app)
    assert client.get("/healthz").status_code == 200

    span = next(s for s in exporter.get_finished_spans() if s.kind.name == "SERVER")
    assert "gpustack.request_id" not in (span.attributes or {})


def test_missing_dependency_starts_without_tracing(exporter, monkeypatch, caplog):
    """Enabled + uninstalled OTel packages must not break server startup."""
    import builtins

    real_import = builtins.__import__
    blocked = {
        "opentelemetry.instrumentation.fastapi",
        "opentelemetry.sdk.trace.export",
    }

    def guarded(name, *args, **kwargs):
        if name in blocked:
            raise ImportError(f"simulated missing {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    app = FastAPI()
    with caplog.at_level("WARNING"):
        provider = tracing_mod.setup_tracing(app, enabled=True)
    assert provider is None
    assert any("not installed" in r.message for r in caplog.records)
    # Server still serves.
    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    assert TestClient(app).get("/healthz").status_code == 200
