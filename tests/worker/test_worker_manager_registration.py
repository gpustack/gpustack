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
