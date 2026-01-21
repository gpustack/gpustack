from gpustack.routes.workers import update_worker_data
from gpustack.schemas.workers import (
    Worker,
    WorkerCreate,
    WorkerStateEnum,
    WorkerStatus,
    Maintenance,
    SystemReserved,
)
from gpustack.schemas.clusters import Cluster, ClusterProvider


def test_update_worker_data_preserves_maintenance_mode():
    """
    Test that maintenance mode is preserved when a worker re-registers.
    This verifies the fix for the issue where workers automatically exit
    maintenance mode after restart.
    """
    # Create an existing worker with maintenance mode enabled
    existing_worker = Worker(
        id=1,
        name="test-worker",
        labels={"env": "test"},
        maintenance=Maintenance(enabled=True, message="Scheduled maintenance"),
        state=WorkerStateEnum.MAINTENANCE,
        cluster_id=1,
        hostname="test-host",
        ip="192.168.1.100",
        ifname="eth0",
        port=8080,
        worker_uuid="test-uuid-123",
        status=WorkerStatus.get_default_status(),
    )

    # Create a worker registration request without maintenance field
    # (simulating a worker restart/re-registration)
    worker_in = WorkerCreate(
        name="test-worker",
        labels={"env": "test", "new": "label"},
        maintenance=None,  # Not set during re-registration
        hostname="test-host",
        ip="192.168.1.100",
        ifname="eth0",
        port=8080,
        worker_uuid="test-uuid-123",
        cluster_id=1,
        status=WorkerStatus.get_default_status(),
        system_reserved=SystemReserved(ram=0, vram=0),
    )

    # Update the worker data
    updated_worker = update_worker_data(worker_in, existing=existing_worker)

    # Verify that maintenance mode is preserved
    assert updated_worker.maintenance is not None
    assert updated_worker.maintenance.enabled is True
    assert updated_worker.maintenance.message == "Scheduled maintenance"
    # State will be computed as MAINTENANCE because of compute_state()
    assert updated_worker.state == WorkerStateEnum.MAINTENANCE


def test_update_worker_data_can_disable_maintenance_mode():
    """
    Test that maintenance mode can be explicitly disabled when provided.
    """
    # Create an existing worker with maintenance mode enabled
    existing_worker = Worker(
        id=1,
        name="test-worker",
        labels={"env": "test"},
        maintenance=Maintenance(enabled=True, message="Scheduled maintenance"),
        state=WorkerStateEnum.MAINTENANCE,
        cluster_id=1,
        hostname="test-host",
        ip="192.168.1.100",
        ifname="eth0",
        port=8080,
        worker_uuid="test-uuid-123",
        status=WorkerStatus.get_default_status(),
    )

    # Create a worker update request with maintenance explicitly disabled
    worker_in = WorkerCreate(
        name="test-worker",
        labels={"env": "test"},
        maintenance=Maintenance(enabled=False, message=None),
        hostname="test-host",
        ip="192.168.1.100",
        ifname="eth0",
        port=8080,
        worker_uuid="test-uuid-123",
        cluster_id=1,
        status=WorkerStatus.get_default_status(),
        system_reserved=SystemReserved(ram=0, vram=0),
    )

    # Update the worker data
    updated_worker = update_worker_data(worker_in, existing=existing_worker)

    # Verify that maintenance mode is disabled
    assert updated_worker.maintenance is not None
    assert updated_worker.maintenance.enabled is False
    assert updated_worker.maintenance.message is None
    # State will be computed based on heartbeat, but maintenance is disabled
    # Since maintenance is disabled, the state won't be MAINTENANCE
    assert updated_worker.state != WorkerStateEnum.MAINTENANCE


def test_update_worker_data_new_worker_without_maintenance():
    """
    Test that a new worker can be created without maintenance mode.
    """
    # Create a new worker registration request
    worker_in = WorkerCreate(
        name="new-worker",
        labels={"env": "prod"},
        maintenance=None,
        hostname="new-host",
        ip="192.168.1.101",
        ifname="eth0",
        port=8080,
        worker_uuid="new-uuid-456",
        cluster_id=1,
        status=WorkerStatus.get_default_status(),
        system_reserved=SystemReserved(ram=0, vram=0),
    )

    # Create cluster for new worker
    cluster = Cluster(
        id=1,
        name="test-cluster",
        provider=ClusterProvider.Docker,
    )

    # Create a new worker (no existing worker)
    new_worker = update_worker_data(worker_in, existing=None, cluster=cluster)

    # Verify that the new worker is created without maintenance mode
    assert new_worker.maintenance is None
    # State may be NOT_READY due to missing heartbeat, but not MAINTENANCE
    assert new_worker.state != WorkerStateEnum.MAINTENANCE


def test_update_worker_data_preserves_labels_merge():
    """
    Test that labels are properly merged when updating a worker.
    """
    # Create an existing worker with some labels
    existing_worker = Worker(
        id=1,
        name="test-worker",
        labels={"env": "test", "region": "us-west"},
        maintenance=None,
        state=WorkerStateEnum.READY,
        cluster_id=1,
        hostname="test-host",
        ip="192.168.1.100",
        ifname="eth0",
        port=8080,
        worker_uuid="test-uuid-123",
        status=WorkerStatus.get_default_status(),
    )

    # Create a worker update with new labels
    worker_in = WorkerCreate(
        name="test-worker",
        labels={"env": "prod", "zone": "a"},  # env changes, zone is new
        maintenance=None,
        hostname="test-host",
        ip="192.168.1.100",
        ifname="eth0",
        port=8080,
        worker_uuid="test-uuid-123",
        cluster_id=1,
        status=WorkerStatus.get_default_status(),
        system_reserved=SystemReserved(ram=0, vram=0),
    )

    # Update the worker data
    updated_worker = update_worker_data(worker_in, existing=existing_worker)

    # Verify that labels are properly merged
    assert updated_worker.labels["env"] == "prod"  # Updated
    assert updated_worker.labels["region"] == "us-west"  # Preserved
    assert updated_worker.labels["zone"] == "a"  # New
