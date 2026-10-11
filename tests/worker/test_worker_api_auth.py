from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import httpx
import pytest
import pytest_asyncio

from gpustack.worker import worker as worker_module


@pytest_asyncio.fixture
async def worker_app(monkeypatch, tmp_path):
    worker = worker_module.Worker.__new__(worker_module.Worker)
    worker._config = SimpleNamespace(
        data_dir=tmp_path,
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
async def test_proxy_delegates_model_authorization(worker_app, header, server_status):
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

    if server_status == 200:
        assert response.status_code == 400
        assert "Missing target port" in response.text
    else:
        assert response.status_code == 401
    http_client.get.assert_called_once_with(
        "http://server/token-auth",
        headers={
            "Authorization": f"Bearer {credential}",
            "X-Higress-Llm-Model": "allowed-model",
        },
    )


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
async def test_proxy_rejects_unavailable_auth_server(worker_app):
    worker_app.state.http_client_no_proxy.get.side_effect = aiohttp.ClientError()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=worker_app), base_url="http://worker"
    ) as client:
        response = await client.get(
            "/proxy/v1/models",
            headers={
                "Authorization": "Bearer inference-only-key",
                "X-Higress-Llm-Model": "allowed-model",
            },
        )

    assert response.status_code == 401
