from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gpustack.schemas.clusters import CloudCredential, Cluster, ClusterProvider
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.server import controllers


@pytest.fixture
def provisioning(monkeypatch):
    session = AsyncMock()
    session_context = MagicMock()
    session_context.__aenter__.return_value = session
    monkeypatch.setattr(controllers, "async_session", lambda: session_context)
    cluster = Cluster(
        id=10,
        name="cloud",
        owner_principal_id=101,
        provider=ClusterProvider.DigitalOcean,
        credential_id=20,
        server_url="https://example.invalid",
    )
    worker = SimpleNamespace(id=30, name="worker", cluster=cluster)
    credential = CloudCredential(
        id=20,
        name="cloud-key",
        owner_principal_id=101,
        provider=ClusterProvider.DigitalOcean,
        secret="private-test-secret",
    )
    lookup = AsyncMock(return_value=credential)
    monkeypatch.setattr(CloudCredential, "one_by_id", lookup)
    monkeypatch.setattr(Worker, "one_by_id", AsyncMock(return_value=worker))
    client_factory = MagicMock()
    monkeypatch.setattr(controllers, "get_client_from_provider", client_factory)
    service = SimpleNamespace(update=AsyncMock())
    monkeypatch.setattr(controllers, "WorkerService", lambda session: service)
    controller = controllers.WorkerProvisioningController(MagicMock())
    controller.check_server_external_url = AsyncMock()
    controller._provisioning_instance = AsyncMock()
    controller._deleting_instance = AsyncMock()
    return SimpleNamespace(
        session=session,
        cluster=cluster,
        worker=worker,
        credential=credential,
        lookup=lookup,
        client_factory=client_factory,
        service=service,
        controller=controller,
        event=SimpleNamespace(data=worker, type="updated"),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state",
    [WorkerStateEnum.PENDING, WorkerStateEnum.PROVISIONING, WorkerStateEnum.DELETING],
)
@pytest.mark.parametrize(
    "invalid",
    ["missing", "deleted", "foreign", "ownerless", "provider", "ownerless_cluster"],
)
async def test_invalid_binding_cannot_reach_cloud_provider(
    provisioning, state, invalid, caplog
):
    p = provisioning
    p.worker.state = state
    if invalid == "missing":
        p.lookup.return_value = None
    elif invalid == "deleted":
        p.credential.deleted_at = datetime.now(timezone.utc)
    elif invalid == "foreign":
        p.credential.owner_principal_id = 202
    elif invalid == "ownerless":
        p.credential.owner_principal_id = None
    elif invalid == "ownerless_cluster":
        p.cluster.owner_principal_id = None
        p.credential.owner_principal_id = None
    else:
        p.credential.provider = ClusterProvider.Shuihua

    await p.controller._reconcile(p.event)

    p.client_factory.assert_not_called()
    p.controller.check_server_external_url.assert_not_awaited()
    p.controller._provisioning_instance.assert_not_awaited()
    p.controller._deleting_instance.assert_not_awaited()
    p.session.commit.assert_not_awaited()
    p.session.rollback.assert_awaited_once()
    assert p.worker.state == WorkerStateEnum.ERROR
    p.service.update.assert_awaited_once_with(
        worker=p.worker, source=None, auto_commit=True
    )
    assert "private-test-secret" not in p.worker.state_message
    assert "private-test-secret" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider", [ClusterProvider.DigitalOcean, ClusterProvider.Shuihua]
)
@pytest.mark.parametrize(
    "state",
    [WorkerStateEnum.PENDING, WorkerStateEnum.PROVISIONING, WorkerStateEnum.DELETING],
)
async def test_valid_binding_can_provision_and_delete(provisioning, state, provider):
    p = provisioning
    p.worker.state = state
    p.cluster.provider = p.credential.provider = provider

    await p.controller._reconcile(p.event)

    p.client_factory.assert_called_once_with(provider, credential=p.credential)
    if state == WorkerStateEnum.DELETING:
        p.controller._deleting_instance.assert_awaited_once()
        p.controller._provisioning_instance.assert_not_awaited()
    else:
        p.controller._provisioning_instance.assert_awaited_once()
        p.controller._deleting_instance.assert_not_awaited()
    p.session.commit.assert_awaited_once()
    p.session.rollback.assert_not_awaited()
