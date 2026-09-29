"""Opt-in OpenTelemetry request tracing for the GPUStack server.

Stage-1 seam for https://github.com/gpustack/gpustack/issues/5609: one server
span per inbound HTTP request, joined to the trace the gateway (Envoy/Higress)
already starts via W3C ``traceparent``, exported over standard OTLP.

Hard rules:
* Default off: ``setup_tracing`` returns before importing OpenTelemetry when
  disabled, so users who leave it off pay nothing on the hot path.
* Optional dependency: the OpenTelemetry packages live in the
  ``gpustack[tracing]`` extra; enabled-but-missing logs one warning and keeps
  serving instead of failing startup.
* Async export only: production uses a BatchSpanProcessor.

The exporter is driven entirely by standard ``OTEL_EXPORTER_OTLP_*`` env vars.
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
    *,
    set_global: bool = True,
) -> Any:
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider

    provider = TracerProvider(
        resource=Resource.create({"service.name": service_name})
    )
    if exporter is not None and processor_cls is not None:
        provider.add_span_processor(processor_cls(exporter))
    if set_global:
        from opentelemetry import trace

        trace.set_tracer_provider(provider)
    return provider


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

    _add_request_context_middleware(app)

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

    provider = _build_provider(
        service_name, exporter, processor_cls, set_global=set_global
    )
    # Pass the provider explicitly: it must be used even in environments where
    # a global provider is already installed (and OTel forbids overriding it).
    FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)
    _tracer_provider = provider
    logger.info(
        "OpenTelemetry request tracing enabled (service.name=%s).", service_name
    )
    return provider


def _add_request_context_middleware(app: "FastAPI") -> None:
    """Annotate the active server span with Envoy's x-request-id."""
    from opentelemetry import trace

    @app.middleware("http")
    async def request_context_middleware(request, call_next):
        span = trace.get_current_span()
        request_id = request.headers.get("x-request-id")
        if request_id is not None and span is not None and span.is_recording():
            span.set_attribute("gpustack.request_id", request_id)
        return await call_next(request)
