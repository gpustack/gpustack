import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode, get_current_span

from gpustack.api.responses import StreamingResponseWithStatusCode
from gpustack.observability import tracing
from gpustack.routes.openai import _stream_response


@pytest.mark.asyncio
async def test_exporter_lifespan_batches_and_flushes_without_global_provider(
    monkeypatch,
):
    exporters = []

    def make_exporter(**kwargs):
        assert kwargs == {"endpoint": "receiver:4317", "insecure": True, "timeout": 3}
        exporter = InMemorySpanExporter()
        exporters.append(exporter)
        return exporter

    monkeypatch.setattr(tracing, "OTLPSpanExporter", make_exporter)
    async with tracing.inference_tracing(None, "server") as disabled:
        assert disabled is None
    assert exporters == []

    async with (
        tracing.inference_tracing("receiver:4317", "server") as server,
        tracing.inference_tracing("receiver:4317", "worker") as worker,
    ):
        with server.start_as_current_span("entry"):
            with worker.start_as_current_span("proxy"):
                pass

    (entry,) = exporters[0].get_finished_spans()
    (proxy,) = exporters[1].get_finished_spans()
    assert entry.resource.attributes["service.name"] == "gpustack.server"
    assert proxy.resource.attributes["service.name"] == "gpustack.worker"
    assert proxy.parent.span_id == entry.context.span_id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome", ["complete", "cancel", "fail_before_body", "fail_mid_body"]
)
async def test_inference_spans_and_worker_response_lifetime(outcome):
    provider = TracerProvider(shutdown_on_exit=False)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer(__name__)
    started_stream = asyncio.Event()
    finished_sending = asyncio.Event()
    disconnect = asyncio.Event()
    chunks = []

    async def upstream_chunks(_chunk_size):
        assert not exporter.get_finished_spans()
        if outcome == "fail_before_body":
            raise TimeoutError("private URL before body")
        yield b"data: first\n\n"
        if outcome == "fail_mid_body":
            raise TimeoutError("private URL after body")
        yield b"data: [DONE]\n\n"

    def close():
        finished_sending.set()

    session = SimpleNamespace(
        request=AsyncMock(
            return_value=SimpleNamespace(
                status=200,
                headers={"content-type": "text/event-stream"},
                content=SimpleNamespace(iter_chunked=upstream_chunks),
                close=close,
            )
        ),
    )
    worker = SimpleNamespace(
        id=7,
        ip="worker",
        advertise_address=None,
        port=80,
        token="private token",
        get_proxy_address=lambda: None,
    )

    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def proxy():
        return StreamingResponseWithStatusCode(
            _stream_response(
                worker, "POST", "v1/chat/completions", {}, None, session, session
            )
        )

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "method": "POST",
        "path": "/v1/chat/completions",
        "scheme": "http",
        "headers": [(b"traceparent", b"invalid")],
        "query_string": b"private=query",
    }

    async def receive():
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        chunks.append(message)
        if message["type"] == "http.response.body" and message.get("more_body"):
            assert not any(
                span.kind == SpanKind.SERVER for span in exporter.get_finished_spans()
            )
            started_stream.set()
            if outcome == "cancel":
                # A disconnect while the client is backpressuring the send
                # happens with the Worker iterator suspended at a yield.
                await asyncio.Future()

    task = asyncio.create_task(
        tracing.trace_inference_http(
            app,
            scope,
            receive,
            send,
            tracer,
            "inference",
            {"gpustack.request_id": "test"},
        )
    )
    if outcome == "cancel":
        await asyncio.wait_for(started_stream.wait(), 2)
        disconnect.set()
    await asyncio.wait_for(task, 2)

    assert finished_sending.is_set()
    spans = {span.kind: span for span in exporter.get_finished_spans()}
    assert set(spans) == {SpanKind.SERVER, SpanKind.CLIENT}
    entry = spans[SpanKind.SERVER]
    upstream = spans[SpanKind.CLIENT]
    assert upstream.context.trace_id == entry.context.trace_id
    assert upstream.parent.span_id == entry.context.span_id
    sent_headers = session.request.await_args.kwargs["headers"]
    trace_id, span_id = sent_headers["traceparent"].split("-")[1:3]
    assert trace_id == f"{entry.context.trace_id:032x}"
    assert span_id == f"{upstream.context.span_id:016x}"
    assert entry.end_time >= upstream.end_time
    assert not get_current_span().get_span_context().is_valid
    if outcome == "cancel":
        assert entry.attributes["gpustack.cancelled"] is True
        assert upstream.attributes["gpustack.cancelled"] is True
        assert not any(
            msg["type"] == "http.response.body" and not msg.get("more_body")
            for msg in chunks
        )
    else:
        expected_status = 500 if outcome == "fail_before_body" else 200
        assert chunks[0]["status"] == expected_status
        assert entry.attributes["http.response.status_code"] == expected_status
        if outcome == "complete":
            assert (
                entry.status.status_code
                == upstream.status.status_code
                == StatusCode.UNSET
            )
            assert b"data: [DONE]" in b"".join(msg.get("body", b"") for msg in chunks)
        else:
            errors = [
                "500" if outcome == "fail_before_body" else "TimeoutError",
                "TimeoutError",
            ]
            for span, expected_error in zip((entry, upstream), errors):
                assert span.status.status_code == StatusCode.ERROR
                assert span.attributes["error.type"] == expected_error
                assert not span.events
                assert "private" not in str(dict(span.attributes))
    provider.shutdown()
