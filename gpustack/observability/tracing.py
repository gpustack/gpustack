"""Opt-in OpenTelemetry request tracing for the GPUStack server.

Emit one server span per inbound HTTP request, joined to the trace the
gateway (Envoy/Higress) already starts via W3C ``traceparent``, and export
over standard OTLP.

Hard rules:
* Default off: ``setup_tracing`` returns before importing OpenTelemetry when
  disabled, so users who leave it off pay nothing on the hot path.
* Optional dependency: the OpenTelemetry packages live in the
  ``gpustack[tracing]`` extra; enabled-but-missing logs one warning and keeps
  serving instead of failing startup.
* Async export only: production uses a BatchSpanProcessor, flushed from the
  app's own lifespan on shutdown so spans near termination are not lost.
  GPUStack builds the server app with a custom lifespan, which Starlette uses
  instead of its default one — and the default lifespan is the only thing that
  runs a router's startup/shutdown handler lists. The provider is therefore
  kept on ``app.state`` and flushed through :func:`shutdown_tracing` rather
  than registered on ``app.router.on_shutdown``.
* The exporter is driven entirely by standard ``OTEL_EXPORTER_OTLP_*`` env
  vars; no exporter is constructed before the OTel packages are known to be
  importable, and any failure while wiring tracing is caught so a failed setup
  leaves the app serving instead of aborting startup.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

_tracer_provider: Any = None


def _build_provider(
    service_name: str,
    exporter: Any,
    processor_cls: Any,
) -> Any:
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if exporter is not None and processor_cls is not None:
        provider.add_span_processor(processor_cls(exporter))
    return provider


def _server_request_hook(span: Any, scope: Any) -> None:
    """Annotate the active server span with the gateway's x-request-id."""
    request_id = None
    for key, value in scope.get("headers") or ():
        if key == b"x-request-id":
            request_id = value.decode("utf-8", "replace")
            break
    if request_id is not None and span is not None and span.is_recording():
        span.set_attribute("gpustack.request_id", request_id)


def setup_tracing(
    app: "FastAPI",
    *,
    enabled: bool,
    service_name: str = "gpustack",
    exporter: Any = None,
    processor_cls: Any = None,
    set_global: bool = True,
) -> Optional[Any]:
    """Install request tracing on ``app`` when ``enabled``.

    Returns the TracerProvider, or None when tracing is off/unavailable.
    Any failure while wiring tracing is logged and swallowed: the server keeps
    serving, and nothing is instrumented or left behind on the app.
    """
    global _tracer_provider

    if not enabled:
        return None

    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.warning(
            "Tracing is enabled (enable_tracing=true) but the OpenTelemetry "
            "packages are not installed. Install the 'tracing' extra "
            "(pip install 'gpustack[tracing]'); starting without tracing."
        )
        return None

    if exporter is None:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            exporter = OTLPSpanExporter()
        except ImportError:
            logger.warning(
                "Tracing is enabled but the OTLP HTTP span exporter is not "
                "available. Install the 'tracing' extra; starting without "
                "tracing."
            )
            return None

    if processor_cls is None:
        processor_cls = BatchSpanProcessor

    provider = None
    try:
        provider = _build_provider(service_name, exporter, processor_cls)
        # Pass the provider explicitly: it must be used even in environments
        # where a global provider is already installed (and OTel forbids
        # overriding it).
        FastAPIInstrumentor.instrument_app(
            app,
            tracer_provider=provider,
            server_request_hook=_server_request_hook,
        )
    except Exception:
        # A half-built provider owns a background export thread; stop it so the
        # failure leaves no processor or thread behind.
        if provider is not None:
            provider.shutdown()
        logger.warning(
            "Tracing is enabled but the OpenTelemetry setup failed; starting "
            "without tracing.",
            exc_info=True,
        )
        return None

    if set_global:
        from opentelemetry import trace

        trace.set_tracer_provider(provider)

    # The server app owns a custom lifespan, so it never runs
    # ``app.router.on_shutdown``; keep the provider for the app's lifespan to
    # shut down through ``shutdown_tracing``.
    app.state.tracer_provider = provider
    _tracer_provider = provider
    logger.info(
        "OpenTelemetry request tracing enabled (service.name=%s).", service_name
    )
    return provider


def shutdown_tracing(app: "FastAPI") -> None:
    """Flush and stop the tracer provider installed by :func:`setup_tracing`.

    Call this from the application's own lifespan teardown. It is a no-op when
    tracing is disabled or its setup failed.
    """
    global _tracer_provider
    provider = getattr(app.state, "tracer_provider", None)
    if provider is None:
        return
    provider.shutdown()
    app.state.tracer_provider = None
    _tracer_provider = None
