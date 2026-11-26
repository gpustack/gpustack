from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gpustack.schemas.workers import WorkerRegistrationPublic, WorkerStatus
from gpustack.worker import worker_manager


def _registration_public():
    now = datetime.now(timezone.utc)
    return WorkerRegistrationPublic(
        id=1,
        name="worker-1",
        hostname="worker-host",
        ip="127.0.0.1",
        ifname="eth0",
        port=10100,
        worker_uuid="uuid-1",
        token="worker-token",
        status=WorkerStatus.get_default_status(),
        created_at=now,
        updated_at=now,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("registration_fails", [False, True])
async def test_register_worker_closes_registration_client(
    monkeypatch, config, tmp_path, registration_fails
):
    """The registration ClientSet is single-use; it must be closed whether
    registration succeeds or fails, otherwise its pooled keep-alive
    connection is stranded until the process exits."""
    manager = worker_manager.WorkerManager.__new__(worker_manager.WorkerManager)
    manager._cfg = config.model_copy(update={"data_dir": str(tmp_path)})
    manager._is_embedded = False
    manager._collector = Mock()
    manager._collector.timed_collect.return_value = SimpleNamespace(
        model_dump=lambda: {
            "hostname": "worker-host",
            "ip": "127.0.0.1",
            "ifname": "eth0",
            "port": 10100,
            "worker_uuid": "uuid-1",
            "status": WorkerStatus.get_default_status().model_dump(),
        }
    )

    registration_client = Mock(spec=worker_manager.WorkerRegistrationClient)
    registration_client.create_async = AsyncMock(
        side_effect=RuntimeError("server unavailable") if registration_fails else None
    )
    if not registration_fails:
        registration_client.create_async.return_value = _registration_public()
    registration_client.aclose = AsyncMock()
    monkeypatch.setattr(
        worker_manager,
        "registration_client",
        Mock(return_value=registration_client),
    )
    monkeypatch.setattr(worker_manager, "ensure_server_tls_trust", AsyncMock())

    if registration_fails:
        with pytest.raises(RuntimeError):
            await manager._register_worker()
    else:
        created = await manager._register_worker()
        assert created.id == 1

    registration_client.create_async.assert_awaited_once()
    registration_client.aclose.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize('worker_version', ['2.3.1', '2.3.0'])
async def test_version_mismatch_warns_and_continues_registration(
    monkeypatch, caplog, worker_version
):
    manager = worker_manager.WorkerManager.__new__(worker_manager.WorkerManager)
    manager._cfg = SimpleNamespace(get_server_url=lambda: 'https://server.example')
    monkeypatch.setattr(worker_manager, 'ensure_server_tls_trust', AsyncMock())
    monkeypatch.setattr(
        worker_manager, 'resolve_version_info', lambda: (worker_version, 'worker')
    )
    monkeypatch.setattr(
        manager,
        '_fetch_server_version',
        AsyncMock(return_value={'version': '2.3.1-ent1', 'git_commit': 'server'}),
    )
    registered = _registration_public()
    register = AsyncMock(return_value=registered)
    monkeypatch.setattr(manager, '_register_worker', register)
    monkeypatch.setattr(manager, '_prepare_clients', Mock())
    monkeypatch.setattr(worker_manager, 'write_worker_token', Mock())
    manager._cfg.data_dir = '/unused'
    manager._clientset = object()

    await manager.check_server_version()
    client, config = await manager.register_with_server()

    assert 'Version mismatch detected' in caplog.text
    assert worker_version in caplog.text and '2.3.1-ent1' in caplog.text
    register.assert_awaited_once()
    assert client is manager._clientset
    assert config == registered.worker_config
