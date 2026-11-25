import asyncio
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from typing import AsyncIterator, Iterator, Optional
from uuid import UUID

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ALWAYS_ON
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from starlette.datastructures import Headers
from starlette.requests import ClientDisconnect
from starlette.types import ASGIApp, Message, Receive, Scope, Send

_TRACE_CONTEXT = TraceContextTextMapPropagator()
PUBLIC_INFERENCE_PATHS = frozenset(
    f"/v1/{operation}"
    for operation in (
        "chat/completions",
        "completions",
        "embeddings",
        "rerank",
        "images/generations",
        "images/edits",
        "audio/speech",
        "audio/transcriptions",
        "audio/translations",
        "responses",
        "messages",
        "videos",
        "videos/sync",
    )
)
_INFERENCE_TRACER: ContextVar[Optional[trace.Tracer]] = ContextVar(
    "inference_tracer", default=None
)


@asynccontextmanager
async def inference_tracing(
    endpoint: Optional[str], role: str
) -> AsyncIterator[Optional[trace.Tracer]]:
    """Own the exporter for one API lifespan, including an embedded Worker's API."""
    if not endpoint:
        yield None
        return

    provider = TracerProvider(
        resource=Resource.create({"service.name": f"gpustack.{role}"}),
        sampler=ALWAYS_ON,
        shutdown_on_exit=False,
    )
    provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(endpoint=endpoint, insecure=True, timeout=3),
            max_queue_size=2048,
            max_export_batch_size=512,
            schedule_delay_millis=500,
        )
    )
    try:
        yield provider.get_tracer(__name__)
    finally:
        await asyncio.to_thread(provider.shutdown)


def current_inference_span() -> trace.Span:
    if _INFERENCE_TRACER.get() is None:
        return trace.INVALID_SPAN
    return trace.get_current_span()


def trace_log_fields() -> dict:
    context = current_inference_span().get_span_context()
    if not context.is_valid:
        return {}
    return {
        "trace_id": f"{context.trace_id:032x}",
        "span_id": f"{context.span_id:016x}",
        "trace_flags": int(context.trace_flags),
    }


def inference_request_id(headers: Headers) -> Optional[str]:
    value = headers.get("x-request-id", "")
    if len(value) == 36:
        try:
            if str(UUID(value)) == value.lower():
                return value
        except ValueError:
            pass
    return None


def inject_trace_context(headers: dict, span: trace.Span) -> None:
    if not span.is_recording():
        return
    for key in list(headers):
        if key.lower() in ("traceparent", "tracestate", "connection"):
            del headers[key]
    _TRACE_CONTEXT.inject(headers, context=trace.set_span_in_context(span))


def set_span_error(span: trace.Span, error_type: str) -> None:
    span.set_attribute("error.type", error_type)
    span.set_status(trace.StatusCode.ERROR)


def set_http_status(
    span: Optional[trace.Span], status: int, *, client: bool = False
) -> None:
    if span is None:
        return
    span.set_attribute("http.response.status_code", status)
    if status == 499:
        span.set_attribute("gpustack.cancelled", True)
    elif status >= 500 or (client and status >= 400):
        set_span_error(span, str(status))


async def trace_inference_http(
    app: ASGIApp,
    scope: Scope,
    receive: Receive,
    send: Send,
    tracer: trace.Tracer,
    name: str,
    attributes: dict,
) -> None:
    """Keep a server span active through the final ASGI body or disconnect."""
    # Continue the remote trace and parent ID without accepting opaque
    # caller-supplied tracestate or baggage into exported spans.
    parents = Headers(scope=scope).getlist("traceparent")
    parent = _TRACE_CONTEXT.extract(
        {"traceparent": parents[0]} if len(parents) == 1 else {}
    )
    disconnected = False
    complete = False

    async def traced_receive() -> Message:
        nonlocal disconnected
        message = await receive()
        if message["type"] == "http.disconnect" and not complete:
            disconnected = True
        return message

    async def traced_send(message: Message) -> None:
        nonlocal complete, disconnected
        if message["type"] == "http.response.start":
            set_http_status(span, message["status"])
            media_type = Headers(raw=list(message.get("headers", []))).get(
                "content-type"
            )
            if media_type:
                span.set_attribute(
                    "gpustack.stream",
                    media_type.partition(";")[0].strip().lower() == "text/event-stream",
                )
        if message["type"] == "http.response.body" and not message.get("more_body"):
            complete = True
        try:
            await send(message)
        except OSError:
            disconnected = True
            raise

    token = _INFERENCE_TRACER.set(tracer)
    try:
        with tracer.start_as_current_span(
            name,
            context=parent,
            kind=trace.SpanKind.SERVER,
            attributes=attributes,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                await app(scope, traced_receive, traced_send)
            except (asyncio.CancelledError, ClientDisconnect):
                span.set_attribute("gpustack.cancelled", True)
                raise
            except Exception as error:
                if not disconnected:
                    set_span_error(span, type(error).__name__)
                raise
            finally:
                if disconnected:
                    span.set_attribute("gpustack.cancelled", True)
    finally:
        _INFERENCE_TRACER.reset(token)


@contextmanager
def worker_request_span(
    headers: dict, method: str, worker_id: int, tunneled: bool
) -> Iterator[Optional[trace.Span]]:
    """Trace one Worker HTTP exchange without holding a context across yields."""
    tracer = _INFERENCE_TRACER.get()
    if tracer is None:
        yield None
        return

    # Use an explicit span for propagation. A response generator can be closed
    # in a different task from the one that opened the HTTP response.
    with tracer.start_span(
        "HTTP request to Worker",
        kind=trace.SpanKind.CLIENT,
        attributes={
            "http.request.method": method,
            "gpustack.worker_id": worker_id,
            "gpustack.tunneled": tunneled,
        },
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        inject_trace_context(headers, span)
        try:
            yield span
        except (asyncio.CancelledError, GeneratorExit):
            span.set_attribute("gpustack.cancelled", True)
            raise
        except Exception as error:
            set_span_error(span, type(error).__name__)
            raise
