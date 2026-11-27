from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import pytest_asyncio

from gpustack.worker import worker as worker_module


@pytest_asyncio.fixture
async def worker_app(monkeypatch, tmp_path, request):
    worker = worker_module.Worker.__new__(worker_module.Worker)
    worker._config = SimpleNamespace(
        data_dir=tmp_path,
        disable_openapi_docs=False,
        effective_trace_otlp_endpoint=getattr(request, "param", None),
        token="registration-token",
        get_server_url=lambda: "http://server",
        get_api_port=lambda embedded: 10150,
    )
    worker._address = "127.0.0.1"
    worker._is_embedded = True
    worker._serve_manager = MagicMock()
    server = MagicMock()
    server.return_value.serve = AsyncMock()
    monkeypatch.setattr(worker_module.uvicorn, "Server", server)
    monkeypatch.setattr(worker_module, "read_worker_token", lambda _: "worker-token")
    monkeypatch.setattr(worker_module, "setup_logging", lambda: None)

    await worker._serve_apis()
    app = server.call_args.args[0].app
    http_client = MagicMock()
    http_client.get.return_value.__aenter__ = AsyncMock(
        return_value=SimpleNamespace(status=200)
    )
    app.state.http_client_no_proxy = http_client
    return app


@pytest.mark.asyncio
@pytest.mark.parametrize("header", ["Authorization", "X-API-Key"])
async def test_inference_key_cannot_read_worker_files(worker_app, tmp_path, header):
    target = tmp_path / "private-file"
    target.write_text("private")
    credential = "inference-only-key"
    headers = {
        header: f"Bearer {credential}" if header == "Authorization" else credential,
        "X-Higress-Llm-Model": "allowed-model",
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=worker_app), base_url="http://worker"
    ) as client:
        response = await client.get(
            "/files/file-exists", params={"path": str(target)}, headers=headers
        )

    assert response.status_code == 401
    worker_app.state.http_client_no_proxy.get.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("header", ["Authorization", "X-API-Key"])
@pytest.mark.parametrize(
    "method, path",
    [
        ("GET", "/v2/config"),
        ("PUT", "/v2/config"),
        ("GET", "/debug/log_level"),
        ("PUT", "/debug/log_level"),
        ("GET", "/debug/memory"),
        ("GET", "/serveLogs/8"),
        ("GET", "/serveLogOptions/8"),
        ("GET", "/cacheServiceInstanceLogs/8"),
        ("GET", "/benchmark_logs/8"),
        ("GET", "/files/model-config?path=/private/config.json"),
        ("GET", "/files/model-weight-size?path=/private"),
        ("POST", "/files/parse-gguf"),
        ("GET", "/cluster-proxy/api/v1/secrets"),
        ("POST", "/cluster-proxy/api/v1/namespaces"),
        ("PUT", "/cluster-proxy/api/v1/namespaces/example"),
        ("PATCH", "/cluster-proxy/api/v1/namespaces/example"),
        ("DELETE", "/cluster-proxy/api/v1/namespaces/example"),
    ],
)
async def test_management_routes_reject_inference_key(worker_app, method, path, header):
    credential = "inference-only-key"
    headers = {
        header: f"Bearer {credential}" if header == "Authorization" else credential,
        "X-Higress-Llm-Model": "allowed-model",
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=worker_app), base_url="http://worker"
    ) as client:
        response = await client.request(method, path, headers=headers)

    assert response.status_code == 401
    worker_app.state.http_client_no_proxy.get.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("credential", ["worker-token", "registration-token"])
@pytest.mark.parametrize("header", ["Authorization", "X-API-Key"])
@pytest.mark.parametrize("path", ["/debug/log_level", "/proxy/v1/models"])
async def test_control_plane_tokens_remain_authorized(
    worker_app, credential, header, path
):
    headers = {
        header: f"Bearer {credential}" if header == "Authorization" else credential,
        "X-Higress-Llm-Model": "allowed-model",
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=worker_app), base_url="http://worker"
    ) as client:
        response = await client.get(path, headers=headers)

    if path.startswith("/proxy/"):
        # An authorized request reaches the proxy's target-port validation.
        assert response.status_code == 400
        assert "Missing target port" in response.text
    else:
        assert response.status_code == 200
    worker_app.state.http_client_no_proxy.get.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("header", ["Authorization", "X-API-Key"])
@pytest.mark.parametrize("server_status", [200, 401, 403, 500])
async def test_proxy_rejects_user_credentials_without_server_auth(
    worker_app, header, server_status
):
    http_client = worker_app.state.http_client_no_proxy
    http_client.get.return_value.__aenter__.return_value.status = server_status
    credential = "inference-only-key"
    headers = {
        header: f"Bearer {credential}" if header == "Authorization" else credential,
        "X-Higress-Llm-Model": "allowed-model",
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=worker_app), base_url="http://worker"
    ) as client:
        response = await client.get("/proxy/v1/models", headers=headers)

    assert response.status_code == 401
    http_client.get.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/debug/log_level", "/proxy/v1/models"])
@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"X-Higress-Llm-Model": "allowed-model"},
        {"Authorization": "Bearer inference-only-key"},
        {"X-API-Key": "inference-only-key"},
    ],
)
async def test_missing_credentials_or_model_header_are_rejected(
    worker_app, path, headers
):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=worker_app), base_url="http://worker"
    ) as client:
        response = await client.get(path, headers=headers)

    assert response.status_code == 401
    worker_app.state.http_client_no_proxy.get.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("worker_app", ["collector:4317"], indirect=True)
async def test_failed_worker_routes_cannot_poison_trace_batch(worker_app, monkeypatch):
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
    from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
    from gpustack.api.exceptions import ServiceUnavailableException
    from gpustack.routes import request_traces
    from gpustack.utils.model_instance_workers import ModelInstanceEndpoint

    exporter = InMemorySpanExporter()
    provider = TracerProvider(
        resource=Resource.create({"service.name": "gpustack.worker"})
    )
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    worker_app.state.inference_tracer = provider.get_tracer(__name__)
    endpoint = ModelInstanceEndpoint(42, 7, 1, 9900)
    lookup = worker_app.state.get_instance_endpoint_by_model_instance_id
    lookup.side_effect = [None, ServiceUnavailableException(), endpoint, endpoint]

    async def chunks(_size):
        yield b'{"ok":true}'

    upstream = worker_app.state.http_client_no_proxy
    upstream.request = AsyncMock(
        return_value=SimpleNamespace(
            status=200,
            headers={"content-type": "application/json"},
            content=SimpleNamespace(iter_chunked=chunks),
            close=MagicMock(),
        )
    )
    worker_app.state.http_client = upstream
    worker_app.state.worker_ip_getter = lambda: "127.0.0.1"
    worker_app.state.record_successful_inference = MagicMock()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=worker_app), base_url="http://worker"
    ) as client:
        for target, expected in [
            (None, 404),
            ("invalid", 404),
            ("model-42-7.static", 404),
            ("model-42-7.static", 503),
            ("model-42-7.static", 401),
        ]:
            headers = {"X-GPUStack-Model-Instance": target} if target else {}
            response = await client.post("/v1/chat/completions", headers=headers)
            assert response.status_code == expected
        response = await client.post(
            "/v1/chat/completions",
            headers={
                "X-GPUStack-Model-Instance": "model-42-7.static",
                "Authorization": "Bearer worker-token",
            },
        )
        assert response.status_code == 200
    spans = exporter.get_finished_spans()
    assert [s.attributes["http.response.status_code"] for s in spans] == [401, 200]
    batch = encode_spans(spans)
    lookup_instance = AsyncMock(
        return_value=[
            SimpleNamespace(
                id=7, model_id=42, cluster_id=3, worker_id=1, deleted_at=None
            )
        ]
    )
    monkeypatch.setattr(request_traces.ModelInstance, "all_by_ids", lookup_instance)
    await request_traces._authorize_trace_batch(
        batch.SerializeToString(), "", None, SimpleNamespace(id=1, cluster_id=3)
    )
    lookup_instance.assert_awaited_once_with(None, [7])
    provider.shutdown()
