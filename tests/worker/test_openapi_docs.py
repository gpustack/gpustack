from unittest.mock import AsyncMock, MagicMock

import httpx
from prometheus_client import CollectorRegistry
from prometheus_client.core import GaugeMetricFamily
import pytest
import pytest_asyncio

from gpustack.worker import exporter as exporter_module
from gpustack.worker import worker as worker_module

DOC_PATHS = [
    "/docs",
    "/docs/",
    "/redoc",
    "/redoc/",
    "/openapi.json",
    "/docs/oauth2-redirect",
]


@pytest.fixture(params=[False, True], ids=["docs-enabled", "docs-disabled"])
def disable_openapi_docs(request):
    return request.param


@pytest_asyncio.fixture(params=[False, True], ids=["standalone", "embedded"])
async def worker_app(request, config, monkeypatch, disable_openapi_docs):
    monkeypatch.setattr(config, "disable_openapi_docs", disable_openapi_docs)
    worker = worker_module.Worker.__new__(worker_module.Worker)
    worker._config = config
    worker._address = "127.0.0.1"
    worker._is_embedded = request.param
    worker._serve_manager = MagicMock()
    server = MagicMock()
    server.return_value.serve = AsyncMock()
    monkeypatch.setattr(worker_module.uvicorn, "Server", server)
    monkeypatch.setattr(worker_module, "read_worker_token", lambda _: "worker-token")
    monkeypatch.setattr(worker_module, "setup_logging", lambda: None)

    await worker._serve_apis()

    server_config = server.call_args.args[0]
    assert server_config.port == config.get_api_port(request.param)
    return server_config.app


@pytest.fixture
def exporter_app(config, monkeypatch, disable_openapi_docs):
    monkeypatch.setattr(config, "disable_openapi_docs", disable_openapi_docs)
    monkeypatch.setattr(exporter_module, "raw_registry", CollectorRegistry())
    monkeypatch.setattr(exporter_module, "unified_registry", CollectorRegistry())
    server = MagicMock()
    monkeypatch.setattr(exporter_module.uvicorn, "Server", server)
    monkeypatch.setattr(exporter_module, "setup_logging", lambda: None)
    exporter = exporter_module.MetricExporter(
        cfg=config,
        collector=None,
        worker_name_getter=lambda: "worker",
        worker_ip_getter=lambda: "127.0.0.1",
        worker_id_getter=lambda: 1,
        cache={
            "unified": {
                "sample": GaugeMetricFamily("test_worker_metric", "sample", value=1)
            },
            "raw": {"sample": GaugeMetricFamily("test_raw_metric", "sample", value=2)},
        },
    )
    monkeypatch.setattr(exporter, "collect_worker_metrics", lambda: iter(()))

    exporter.start()

    server.assert_called_once()
    server.return_value.run.assert_called_once()
    server_config = server.call_args.args[0]
    assert server_config.port == config.worker_metrics_port
    return server_config.app


@pytest.mark.asyncio
@pytest.mark.parametrize("path", DOC_PATHS)
async def test_worker_documentation_respects_config(
    worker_app, disable_openapi_docs, path
):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=worker_app),
        base_url="http://worker",
        follow_redirects=True,
    ) as client:
        response = await client.get(path)

    assert response.status_code == (404 if disable_openapi_docs else 200)
    if path == "/openapi.json" and not disable_openapi_docs:
        assert "/healthz" in response.json()["paths"]


@pytest.mark.asyncio
@pytest.mark.parametrize("path", DOC_PATHS)
async def test_exporter_documentation_respects_config(
    exporter_app, disable_openapi_docs, path
):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=exporter_app),
        base_url="http://exporter",
        follow_redirects=True,
    ) as client:
        response = await client.get(path)

    assert response.status_code == (404 if disable_openapi_docs else 200)
    if path == "/openapi.json" and not disable_openapi_docs:
        assert "/metrics" in response.json()["paths"]


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/healthz", "/readyz"])
async def test_worker_probes_remain_available(worker_app, path):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=worker_app), base_url="http://worker"
    ) as client:
        response = await client.get(path)

    assert response.status_code == 200
    assert response.json() == "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize("credential", [None, "worker-token", "test"])
async def test_worker_management_auth_remains_enforced(worker_app, credential):
    headers = {"Authorization": "Bearer " + credential} if credential else {}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=worker_app), base_url="http://worker"
    ) as client:
        response = await client.get("/debug/log_level", headers=headers)

    assert response.status_code == (200 if credential else 401)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path, sample",
    [("/metrics", "test_worker_metric 1.0"), ("/metrics/raw", "test_raw_metric 2.0")],
)
async def test_exporter_metrics_remain_available(exporter_app, path, sample):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=exporter_app), base_url="http://exporter"
    ) as client:
        response = await client.get(path)

    assert response.status_code == 200
    assert sample in response.text
