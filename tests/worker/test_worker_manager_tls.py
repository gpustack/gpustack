from types import SimpleNamespace

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
