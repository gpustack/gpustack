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
from gpustack.cloud_providers.abstract import CloudInstance, InstanceState, Volume


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
    # A provider that serves SSH on the instance itself reports no mapping.
    mock_instance.ssh_endpoint = None
    client.get_instance.return_value = mock_instance
    client.wait_for_public_ip.return_value = mock_instance
    await WorkerProvisioningController._provisioning_instance(
        session, client, worker, cfg
    )
    assert worker.state_message == "Waiting for volumes to attach"
    assert "ssh_endpoint" not in (worker.provider_config or {})

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
async def test_provisioning_records_mapped_ssh_endpoint():
    """A provider that maps SSH elsewhere gets that recorded for the UI.

    Shuihua publishes every instance behind one shared IP on an
    instance-specific port, so advertise_address:22 answers nothing and the
    mapping is the only way to reach the host.
    """
    session = AsyncMock()
    client = AsyncMock()
    worker = Worker(
        id=1,
        name="w",
        hostname="h",
        ip="",
        advertise_address="",
        external_id="33",
        state=WorkerStateEnum.PROVISIONING,
        provider_config={"volume_ids": []},
    )
    worker.worker_pool = WorkerPool(id=1, cloud_options=CloudOptions())
    instance = CloudInstance(
        name="w",
        image="",
        type="",
        region="",
        ip_address="172.16.46.22",
        status=InstanceState.RUNNING,
        ssh_endpoint=("125.67.215.17", 33004),
        ssh_user="ubuntu",
    )
    client.wait_for_public_ip = AsyncMock(return_value=instance)

    changed = await WorkerProvisioningController._provisioning_started(
        session, client, worker, instance
    )

    assert changed
    # The private per-instance address stays the identity, the mapping is only
    # a connection hint.
    assert worker.advertise_address == "172.16.46.22"
    assert worker.provider_config["ssh_endpoint"] == {
        "host": "125.67.215.17",
        "port": 33004,
        "user": "ubuntu",
    }
    # Pre-existing keys survive the write.
    assert worker.provider_config["volume_ids"] == []

    # An unreported user is left out rather than stored as "".
    worker.provider_config = {"volume_ids": []}
    worker.advertise_address = ""
    instance.ssh_user = None
    await WorkerProvisioningController._provisioning_started(
        session, client, worker, instance
    )
    assert worker.provider_config["ssh_endpoint"] == {
        "host": "125.67.215.17",
        "port": 33004,
    }


@pytest.mark.asyncio
async def test_provisioning_backfills_late_ssh_endpoint():
    """A mapping published after the address appears still gets recorded.

    wait_for_public_ip returns as soon as an address exists, which can be
    before the provider has published the port mapping. That branch only runs
    while advertise_address is empty, so the endpoint has to be picked up on a
    later pass or it would stay missing for good.
    """
    session = AsyncMock()
    client = AsyncMock()
    worker = Worker(
        id=1,
        name="w",
        hostname="h",
        ip="",
        advertise_address="",
        external_id="33",
        state=WorkerStateEnum.PROVISIONING,
        provider_config={},
    )
    worker.worker_pool = WorkerPool(id=1, cloud_options=CloudOptions())
    worker.cluster = Cluster(id=1, name="c", state=ClusterStateEnum.PROVISIONED)

    def instance(ssh_endpoint):
        return CloudInstance(
            name="w",
            image="",
            type="",
            region="",
            ip_address="172.16.46.22",
            status=InstanceState.RUNNING,
            ssh_endpoint=ssh_endpoint,
            ssh_user="ubuntu",
        )

    # First pass: an address but no mapping yet.
    client.wait_for_public_ip = AsyncMock(return_value=instance(None))
    await WorkerProvisioningController._provisioning_started(
        session, client, worker, instance(None)
    )
    assert worker.advertise_address == "172.16.46.22"
    assert "ssh_endpoint" not in worker.provider_config

    # Later pass: the mapping is published, and advertise_address is set by now.
    worker.state = WorkerStateEnum.PROVISIONING
    changed = await WorkerProvisioningController._provisioning_started(
        session, client, worker, instance(("125.67.215.17", 33004))
    )
    assert changed
    assert worker.provider_config["ssh_endpoint"] == {
        "host": "125.67.215.17",
        "port": 33004,
        "user": "ubuntu",
    }

    # A later poll without the mapping must not erase what was recorded.
    worker.state = WorkerStateEnum.PROVISIONING
    await WorkerProvisioningController._provisioning_started(
        session, client, worker, instance(None)
    )
    assert worker.provider_config["ssh_endpoint"]["port"] == 33004


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
