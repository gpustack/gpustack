"""Tests for the authenticated Worker-to-Server OTLP trace proxy."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import aiohttp
from fastapi import FastAPI
from fastapi.testclient import TestClient
import httpx
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
import pytest

from gpustack.api import exceptions
from gpustack.observability.trace_proxy import LocalOTLPTraceProxy
from gpustack.routes import request_traces as trace_routes


def _make_ingest_app(ingest_url, tempo_response):
    app = FastAPI()
    app.add_api_route(
        "/v2/traces/otlp", trace_routes.ingest_worker_traces, methods=["POST"]
    )
    exceptions.register_handlers(app)
    session = MagicMock()
    session.post.return_value.__aenter__.return_value = tempo_response
    app.state.http_client_no_proxy = session
    return app, session


def _tempo_response(status=200, body=b"", content_type="application/x-protobuf"):
    async def read():
        return body

    return SimpleNamespace(
        status=status,
        read=read,
        headers={"Content-Type": content_type},
    )


def test_local_otlp_trace_proxy_forwards_grpc_spans_to_authenticated_server(
    monkeypatch,
):
    forwarded = []
    app, session = _make_ingest_app(
        "http://127.0.0.1:4318/v1/traces", _tempo_response(status=200, body=b"")
    )
    monkeypatch.setattr(
        trace_routes,
        "get_global_config",
        lambda: SimpleNamespace(
            get_trace_ingest_url=lambda: "http://127.0.0.1:4318/v1/traces"
        ),
    )

    with TestClient(app) as server_client:
        real_post = server_client.post

        def capture_post(url, **kwargs):
            forwarded.append((url, kwargs.get("headers", {})))
            return real_post(url, **kwargs)

        transport_client = SimpleNamespace(post=capture_post)
        proxy = LocalOTLPTraceProxy(
            server_url="http://testserver",
            headers={"Authorization": "Bearer worker-secret"},
            client=transport_client,
        )
        endpoint = proxy.start()
        try:
            provider = TracerProvider(
                resource=Resource.create({"service.name": "vllm"})
            )
            provider.add_span_processor(
                SimpleSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True))
            )
            tracer = provider.get_tracer(__name__)
            with tracer.start_as_current_span("llm_request") as span:
                span.set_attribute("gpustack.model_id", 42)
            provider.force_flush()
            provider.shutdown()
        finally:
            proxy.stop()

    assert len(forwarded) == 1
    assert forwarded[0][0] == "http://testserver/v2/traces/otlp"
    assert forwarded[0][1]["Authorization"] == "Bearer worker-secret"
    assert forwarded[0][1]["Content-Type"] == "application/x-protobuf"
    session.post.assert_called_once()
    call_args, call_kwargs = session.post.call_args
    assert call_args == ("http://127.0.0.1:4318/v1/traces",)
    assert call_kwargs["headers"] == {"Content-Type": "application/x-protobuf"}
    assert b"llm_request" in call_kwargs["data"]


@pytest.mark.parametrize(
    "content_type, encoding, body, expected_status",
    [
        ("application/json", "", b"{}", 415),
        ("application/x-protobuf", "br", b"abc", 415),
        ("application/x-protobuf", "", b"x" * (4 * 1024 * 1024 + 1), 413),
    ],
    ids=["non_protobuf", "unsupported_encoding", "too_large"],
)
def test_ingest_worker_traces_validates_headers_and_size(
    monkeypatch, content_type, encoding, body, expected_status
):
    app, session = _make_ingest_app(
        "http://127.0.0.1:4318/v1/traces", _tempo_response()
    )
    monkeypatch.setattr(
        trace_routes,
        "get_global_config",
        lambda: SimpleNamespace(
            get_trace_ingest_url=lambda: "http://127.0.0.1:4318/v1/traces"
        ),
    )
    headers = {"Content-Type": content_type}
    if encoding:
        headers["Content-Encoding"] = encoding

    with TestClient(app) as client:
        response = client.post("/v2/traces/otlp", content=body, headers=headers)

    assert response.status_code == expected_status
    session.post.assert_not_called()


def test_ingest_worker_traces_returns_503_when_disabled_or_tempo_unreachable(
    monkeypatch,
):
    app, session = _make_ingest_app(None, _tempo_response())
    monkeypatch.setattr(
        trace_routes,
        "get_global_config",
        lambda: SimpleNamespace(get_trace_ingest_url=lambda: None),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v2/traces/otlp",
            content=b"proto",
            headers={"Content-Type": "application/x-protobuf"},
        )
    assert response.status_code == 503

    monkeypatch.setattr(
        trace_routes,
        "get_global_config",
        lambda: SimpleNamespace(
            get_trace_ingest_url=lambda: "http://127.0.0.1:4318/v1/traces"
        ),
    )
    session.post.return_value.__aenter__.side_effect = aiohttp.ClientError("down")
    with TestClient(app) as client:
        response = client.post(
            "/v2/traces/otlp",
            content=b"proto",
            headers={"Content-Type": "application/x-protobuf"},
        )
    assert response.status_code == 503


def test_local_otlp_trace_proxy_propagates_server_errors_without_crashing():
    def failing_post(_url, **_kwargs):
        raise httpx.ConnectError("server down")

    proxy = LocalOTLPTraceProxy(
        server_url="http://testserver",
        headers={"Authorization": "Bearer worker-secret"},
        client=SimpleNamespace(post=failing_post),
    )
    endpoint = proxy.start()
    try:
        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        with provider.get_tracer(__name__).start_as_current_span("failed_span"):
            pass
        provider.shutdown()
    finally:
        proxy.stop()


def test_trace_ingest_route_is_restricted_to_worker_principal():
    from gpustack.routes import routes

    assert ("POST", trace_routes.ingest_worker_traces) in routes.SYSTEM_ROUTES
    worker_routes = {
        (tuple(sorted(route.methods)), route.path, route.endpoint)
        for route in routes.worker_client_router.routes
    }
    assert (
        ("POST",),
        "/traces/otlp",
        trace_routes.ingest_worker_traces,
    ) in worker_routes


@pytest.mark.parametrize(
    "http_status, expected_grpc_status",
    [
        (400, "INVALID_ARGUMENT"),
        (401, "PERMISSION_DENIED"),
        (403, "PERMISSION_DENIED"),
        (413, "RESOURCE_EXHAUSTED"),
        (415, "INVALID_ARGUMENT"),
        (503, "UNAVAILABLE"),
    ],
)
def test_local_otlp_trace_proxy_maps_http_errors_to_grpc_status(
    http_status, expected_grpc_status
):
    import grpc

    aborted = []

    class _Context:
        def abort(self, code, details):
            aborted.append((code, details))
            raise grpc.RpcError()

    proxy = LocalOTLPTraceProxy(
        server_url="http://testserver",
        headers={"Authorization": "Bearer worker-secret"},
        client=SimpleNamespace(
            post=lambda _url, **_kwargs: SimpleNamespace(
                status_code=http_status, content=b""
            )
        ),
    )
    with pytest.raises(grpc.RpcError):
        proxy._export(b"proto", _Context())

    assert len(aborted) == 1
    assert aborted[0][0] == getattr(grpc.StatusCode, expected_grpc_status)


def test_local_otlp_trace_proxy_drains_exports_before_closing_client(monkeypatch):
    events = []
    fake_event = SimpleNamespace(wait=lambda: events.append("server_waited"))
    fake_server = SimpleNamespace(
        add_generic_rpc_handlers=lambda _handlers: None,
        add_insecure_port=lambda _addr: 49152,
        start=lambda: events.append("server_started"),
        stop=lambda grace=1: (events.append("server_stopping"), fake_event)[1],
    )
    fake_client = SimpleNamespace(close=lambda: events.append("client_closed"))
    executor = SimpleNamespace(
        shutdown=lambda *, wait, cancel_futures: events.append(
            ("exports_drained", wait, cancel_futures)
        )
    )
    monkeypatch.setattr(
        "gpustack.observability.trace_proxy.futures.ThreadPoolExecutor",
        lambda *a, **kw: executor,
    )
    monkeypatch.setattr(
        "gpustack.observability.trace_proxy.grpc.server", lambda *a, **kw: fake_server
    )
    monkeypatch.setattr(
        "gpustack.observability.trace_proxy.httpx.Client", lambda *a, **kw: fake_client
    )

    proxy = LocalOTLPTraceProxy(
        server_url="http://testserver",
        headers={"Authorization": "Bearer worker-secret"},
    )
    assert proxy.start() == "127.0.0.1:49152"
    proxy.stop()

    assert events == [
        "server_started",
        "server_stopping",
        "server_waited",
        ("exports_drained", True, True),
        "client_closed",
    ]


def test_local_otlp_trace_proxy_does_not_create_client_when_bind_fails(monkeypatch):
    created = []
    fake_server = SimpleNamespace(
        add_generic_rpc_handlers=lambda _handlers: None,
        add_insecure_port=lambda _addr: 0,
        stop=MagicMock(),
    )
    monkeypatch.setattr(
        "gpustack.observability.trace_proxy.grpc.server", lambda *a, **kw: fake_server
    )
    monkeypatch.setattr(
        "gpustack.observability.trace_proxy.httpx.Client",
        lambda *a, **kw: created.append(True),
    )

    proxy = LocalOTLPTraceProxy(
        server_url="http://testserver",
        headers={"Authorization": "Bearer worker-secret"},
    )
    with pytest.raises(RuntimeError, match="Failed to bind"):
        proxy.start()
    assert not created


@pytest.mark.parametrize("owns_client", [True, False])
def test_local_otlp_trace_proxy_cleans_up_after_start_failure(monkeypatch, owns_client):
    server = MagicMock()
    server.add_insecure_port.return_value = 49152
    server.start.side_effect = RuntimeError("gRPC startup failed")
    client = MagicMock()
    monkeypatch.setattr(
        "gpustack.observability.trace_proxy.grpc.server", lambda *a, **kw: server
    )
    monkeypatch.setattr(
        "gpustack.observability.trace_proxy.httpx.Client", lambda *a, **kw: client
    )
    proxy = LocalOTLPTraceProxy(
        server_url="http://testserver",
        headers={"Authorization": "Bearer worker-secret"},
        client=None if owns_client else client,
    )

    with pytest.raises(RuntimeError, match="gRPC startup failed"):
        proxy.start()

    assert proxy.endpoint is None
    server.stop.assert_called_once()
    assert client.close.call_count == int(owns_client)


def test_worker_reuses_trace_proxy_port_after_restart(monkeypatch, tmp_path):
    from gpustack.worker.worker import Worker

    server = MagicMock()
    server.add_insecure_port.return_value = 49152
    monkeypatch.setattr(
        "gpustack.observability.trace_proxy.grpc.server", lambda *a, **kw: server
    )
    monkeypatch.setattr("gpustack.observability.trace_proxy.httpx.Client", MagicMock())

    for port in (0, 49152):
        worker = Worker.__new__(Worker)
        worker._config = SimpleNamespace(
            data_dir=str(tmp_path),
            disable_tracing=False,
            trace_otlp_endpoint=None,
            use_builtin_tempo=lambda: False,
            get_server_url=lambda: "http://testserver",
            _local_trace_proxy_endpoint=None,
        )
        worker._clientset = SimpleNamespace(headers={})
        worker._worker_manager = SimpleNamespace(trace_proxy_enabled=True)
        worker._trace_proxy = None

        try:
            worker._start_trace_proxy_if_needed()
            server.add_insecure_port.assert_called_with(f"127.0.0.1:{port}")
            assert worker._config._local_trace_proxy_endpoint == "127.0.0.1:49152"
        finally:
            worker._stop_trace_proxy()
