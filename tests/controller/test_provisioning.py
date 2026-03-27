import pytest
from unittest.mock import AsyncMock, MagicMock
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.schemas.clusters import (
    Cluster,
    WorkerPool,
    CloudCredential,
    ClusterProvider,
    ClusterStateEnum,
    CloudOptions,
)
from gpustack.server.controllers import WorkerProvisioningController
from gpustack.cloud_providers.abstract import InstanceState, Volume


@pytest.mark.asyncio
async def test_provisioning_flow(monkeypatch):
    session = AsyncMock()
    session.info = {}
    client = AsyncMock()
    cluster = Cluster(
        id=1, provider=ClusterProvider.DigitalOcean, region="nyc3", credential_id=1
    )
    cluster.state = ClusterStateEnum.PROVISIONED
    pool = WorkerPool(
        id=1,
        cluster=cluster,
        cloud_options=CloudOptions(
            volumes=[
                Volume(size_gb=10, format="ext4"),
                Volume(size_gb=20, format="ext4"),
            ]
        ),
    )
    worker = Worker(
        id=1,
        name="test-worker",
        cluster=cluster,
        worker_pool=pool,
        state=WorkerStateEnum.PENDING,
        provider_config={},
        cluster_id=1,
    )
    credential = CloudCredential(id=1, token="dummy")
    cfg = MagicMock()
    cfg.server_external_url = "http://dummy-server"
    cfg.image_name_override = "dummy-image"
    monkeypatch.setattr("gpustack.config.config.get_global_config", lambda: cfg)
    mock_sshkey = MagicMock()
    mock_sshkey.id = "ssh-key-id"
    monkeypatch.setattr(
        "gpustack.schemas.clusters.Credential.create",
        AsyncMock(return_value=mock_sshkey),
    )
    monkeypatch.setattr(
        "gpustack.cloud_providers.common.get_client_from_provider",
        lambda provider, credential: client,
    )
    monkeypatch.setattr(
        "gpustack.schemas.clusters.Credential.one_by_id",
        AsyncMock(return_value=MagicMock(id=1, external_id="ssh-key-id")),
    )
    monkeypatch.setattr(
        "gpustack.schemas.workers.Worker.one_by_id", AsyncMock(return_value=worker)
    )
    monkeypatch.setattr(
        "gpustack.schemas.clusters.CloudCredential.one_by_id",
        AsyncMock(return_value=credential),
    )
    monkeypatch.setattr("gpustack.server.services.WorkerService.update", AsyncMock())

    mock_instance = MagicMock()
    mock_instance.id = "instance-id"

    client.get_instance = AsyncMock(return_value=mock_instance)
    client.create_ssh_key = AsyncMock(return_value="ssh-key-id")
    mock_user_data = MagicMock()
    mock_user_data.format.return_value = "#!/bin/bash\necho hello"
    client.construct_user_data = AsyncMock(return_value=mock_user_data)
    client.create_instance = AsyncMock(return_value="instance-id")
    client.wait_for_started = AsyncMock(return_value={"id": "instance-id"})
    client.wait_for_public_ip = AsyncMock(
        return_value={"id": "instance-id", "ip_address": "1.2.3.4"}
    )
    client.determine_linux_distribution = AsyncMock(return_value=("ubuntu", True))
    client.create_volumes_and_attach = AsyncMock(return_value=["vol-1", "vol-2"])

    # First call, should enter the SSH key creation process
    await WorkerProvisioningController._provisioning_instance(
        session, client, worker, cfg
    )
    assert worker.state == WorkerStateEnum.PROVISIONING
    assert worker.state_message == "Creating SSH key"
    # Second call, should create SSH key and assign to worker.ssh_key_id
    # Here, simulate SSH key not yet created, worker.ssh_key_id should be assigned
    await WorkerProvisioningController._provisioning_instance(
        session, client, worker, cfg
    )
    assert worker.ssh_key_id == "ssh-key-id"
    assert worker.state_message == "Creating cloud instance"
    # Third call, should enter the cloud instance creation process
    await WorkerProvisioningController._provisioning_instance(
        session, client, worker, cfg
    )
    assert worker.external_id == "instance-id"
    assert worker.state_message == "Waiting for cloud instance started"
    # Fourth call, should wait for cloud instance to start
    client.wait_for_started.return_value = {"id": "instance-id"}
    await WorkerProvisioningController._provisioning_instance(
        session, client, worker, cfg
    )
    assert worker.state_message == "Waiting for instance's public ip"
    # Fifth call, the instance should have public ip
    mock_instance = MagicMock()
    mock_instance.id = "instance-id"
    mock_instance.ip_address = "1.2.3.4"
    mock_instance.status = InstanceState.RUNNING
    client.get_instance.return_value = mock_instance
    client.wait_for_public_ip.return_value = mock_instance
    await WorkerProvisioningController._provisioning_instance(
        session, client, worker, cfg
    )
    assert worker.state_message == "Waiting for volumes to attach"

    # Sixth call, should create and attach volumes
    client.create_volumes_and_attach.return_value = ["vol-1", "vol-2"]
    await WorkerProvisioningController._provisioning_instance(
        session, client, worker, cfg
    )
    assert worker.provider_config is not None
    assert worker.provider_config.get("volume_ids") == ["vol-1", "vol-2"]

    # final call, worker provisioning state should have provisioned
    await WorkerProvisioningController._provisioning_instance(
        session, client, worker, cfg
    )
    assert worker.state == WorkerStateEnum.INITIALIZING


@pytest.mark.asyncio
async def test_deleting_flow(monkeypatch):
    session = AsyncMock()
    client = AsyncMock()
    cluster = Cluster(id=1, provider="DigitalOcean", region="nyc3", credential_id=1)
    pool = WorkerPool(id=1, cluster=cluster)
    worker = Worker(
        id=1,
        name="test-worker",
        cluster=cluster,
        worker_pool=pool,
        state=WorkerStateEnum.DELETING,
        external_id="instance-id",
        deleted_at="2025-08-29",
    )
    credential = CloudCredential(id=1, token="dummy")

    monkeypatch.setattr(
        "gpustack.cloud_providers.common.get_client_from_provider",
        lambda provider, credential: client,
    )
    monkeypatch.setattr(
        "gpustack.schemas.workers.Worker.one_by_id", AsyncMock(return_value=worker)
    )
    monkeypatch.setattr(
        "gpustack.schemas.clusters.CloudCredential.one_by_id",
        AsyncMock(return_value=credential),
    )
    monkeypatch.setattr("gpustack.server.services.WorkerService.delete", AsyncMock())

    client.delete_instance = AsyncMock()

    await WorkerProvisioningController._deleting_instance(session, client, worker)
    client.delete_instance.assert_awaited_with("instance-id")
