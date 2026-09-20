from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from gpustack.worker import worker_manager


@pytest.mark.asyncio
async def test_version_check_bootstraps_tls_before_requesting_server_version(
    monkeypatch,
):
    manager = worker_manager.WorkerManager.__new__(worker_manager.WorkerManager)
    manager._cfg = SimpleNamespace(get_server_url=lambda: "https://server.example")
    calls = []

    async def bootstrap(server_url):
        calls.append(("bootstrap", server_url))

    async def fetch_version():
        calls.append(("version", None))
        return None

    monkeypatch.setattr(worker_manager, "ensure_server_tls_trust", bootstrap)
    monkeypatch.setattr(manager, "_fetch_server_version", fetch_version)

    await manager.check_server_version()

    assert calls == [
        ("bootstrap", "https://server.example"),
        ("version", None),
    ]


@pytest.mark.asyncio
async def test_version_check_skips_request_when_tls_bootstrap_fails(
    monkeypatch, caplog
):
    manager = worker_manager.WorkerManager.__new__(worker_manager.WorkerManager)
    manager._cfg = SimpleNamespace(get_server_url=lambda: "https://server.example")

    async def bootstrap(server_url):
        raise RuntimeError("temporary connection failure")

    async def fetch_version():
        pytest.fail("must not request the version before TLS bootstrap succeeds")

    monkeypatch.setattr(worker_manager, "ensure_server_tls_trust", bootstrap)
    monkeypatch.setattr(manager, "_fetch_server_version", fetch_version)

    await manager.check_server_version()

    assert "Failed to bootstrap server TLS" in caplog.text


@pytest.mark.asyncio
async def test_version_check_retries_request_after_transport_error(monkeypatch, caplog):
    manager = worker_manager.WorkerManager.__new__(worker_manager.WorkerManager)
    manager._cfg = SimpleNamespace(get_server_url=lambda: "https://server.example")
    monkeypatch.setattr(
        worker_manager,
        "ensure_server_tls_trust",
        AsyncMock(side_effect=httpx.ConnectError("connection refused")),
    )
    fetch = AsyncMock(return_value=None)
    monkeypatch.setattr(manager, "_fetch_server_version", fetch)

    await manager.check_server_version()

    fetch.assert_awaited_once()
    assert "Server connection failed" in caplog.text
    assert "Failed to bootstrap server TLS" not in caplog.text


@pytest.mark.asyncio
async def test_fetch_version_uses_worker_trust_context(monkeypatch):
    manager = worker_manager.WorkerManager.__new__(worker_manager.WorkerManager)
    manager._cfg = SimpleNamespace(get_server_url=lambda: "https://server.example")
    context = object()
    monkeypatch.setattr(worker_manager, "make_ssl_context", lambda: context)
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.get.return_value = httpx.Response(
        200, json={"version": "2.3.0"}, request=httpx.Request("GET", "https://server")
    )
    factory = Mock(return_value=client)
    monkeypatch.setattr(worker_manager.httpx, "AsyncClient", factory)

    assert await manager._fetch_server_version() == {"version": "2.3.0"}
    assert factory.call_args.kwargs["verify"] is context
