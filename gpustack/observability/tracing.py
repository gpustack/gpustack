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
* Async export only: production uses a BatchSpanProcessor, flushed when the
  app's lifespan exits so spans near termination are not lost.
* The exporter is driven entirely by standard ``OTEL_EXPORTER_OTLP_*`` env
  vars; no exporter is constructed before the OTel packages are known to be
  importable, and any failure while wiring tracing is caught so a failed setup
  leaves the app serving instead of aborting startup.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

_tracer_provider: Any = None


def _build_provider(
    service_name: str,
    exporter: Any,
    processor_cls: Any,
    *,
    set_global: bool = True,
) -> Any:
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if exporter is not None and processor_cls is not None:
        provider.add_span_processor(processor_cls(exporter))
    if set_global:
        from opentelemetry import trace

        trace.set_tracer_provider(provider)
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


def _chain_shutdown_flush(app: "FastAPI", provider: Any) -> None:
    """Flush queued batch spans when the app's lifespan exits.

    Registering the flush on ``app.router.on_shutdown`` does not reach it: a
    router's startup/shutdown handler lists are run by Starlette's *default*
    lifespan, and the server app is built with an explicit one, so those lists
    stay inert. Chaining the flush onto whichever lifespan the app actually
    uses covers both the default and the explicit case.
    """
    inner_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan_with_flush(app_: Any) -> Any:
        async with inner_lifespan(app_) as maybe_state:
            try:
                yield maybe_state
            finally:
                provider.shutdown()

    app.router.lifespan_context = lifespan_with_flush


def _resolve_exporter(exporter: Any) -> Any:
    """Return the exporter to use, or None when none can be built.

    A caller-supplied exporter is used as-is. Otherwise the OTLP HTTP exporter
    is imported and constructed -- the only place the standard
    ``OTEL_EXPORTER_OTLP_*`` env vars are read -- and any failure is logged and
    reported to the caller as None so the app is never touched.
    """
    if exporter is not None:
        return exporter

    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
    except ImportError:
        logger.warning(
            "Tracing is enabled but the OTLP HTTP span exporter is not "
            "available. Install the 'tracing' extra; starting without tracing."
        )
        return None

    try:
        return OTLPSpanExporter()
    except Exception:
        logger.warning(
            "Tracing is enabled but the OTLP exporter could not be configured "
            "(check the OTEL_EXPORTER_OTLP_* env vars); starting without "
            "tracing.",
            exc_info=True,
        )
        return None


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
    The app is left untouched on any failure path: the OTel packages and the
    exporter are resolved first, and only then are the instrumentation, the
    request-id hook and the shutdown flush attached.
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

    exporter = _resolve_exporter(exporter)
    if exporter is None:
        return None

    if processor_cls is None:
        processor_cls = BatchSpanProcessor

    try:
        provider = _build_provider(
            service_name, exporter, processor_cls, set_global=set_global
        )
        # Pass the provider explicitly: it must be used even in environments
        # where a global provider is already installed (and OTel forbids
        # overriding it).
        FastAPIInstrumentor.instrument_app(
            app,
            tracer_provider=provider,
            server_request_hook=_server_request_hook,
        )
        _chain_shutdown_flush(app, provider)
    except Exception:
        logger.warning(
            "Tracing is enabled but could not be initialised; starting "
            "without tracing.",
            exc_info=True,
        )
        return None

    _tracer_provider = provider
    logger.info(
        "OpenTelemetry request tracing enabled (service.name=%s).", service_name
    )
    return provider
