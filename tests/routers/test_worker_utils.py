import pytest
from gpustack.routes.workers import (
    filter_workers_by_fields,
    find_available_worker_name,
    retry_create_unique_worker_uuid,
    check_worker_name_conflict,
    get_existing_worker,
)
from gpustack.api.exceptions import AlreadyExistsException
from gpustack.schemas.workers import WorkerCreate, WorkerStatus


class DummyWorker:
    def __init__(
        self,
        name,
        worker_uuid,
        cluster_id,
        deleted_at=None,
        labels=None,
        id=None,
        external_id=None,
    ):
        self.name = name
        self.worker_uuid = worker_uuid
        self.cluster_id = cluster_id
        self.deleted_at = deleted_at
        self.labels = labels or {}
        self.id = id
        self.external_id = external_id


def test_filter_workers_by_fields_exact():
    workers = [
        DummyWorker("foo", "uuid1", 1),
        DummyWorker("bar", "uuid2", 1),
        DummyWorker("baz", "uuid3", 2),
    ]
    result = filter_workers_by_fields(workers, {"name": "foo", "cluster_id": 1})
    assert len(result) == 1
    assert result[0].name == "foo"


def test_filter_workers_by_fields_fuzzy():
    workers = [
        DummyWorker("foo-worker", "uuid1", 1),
        DummyWorker("bar-worker", "uuid2", 1),
        DummyWorker("baz", "uuid3", 2),
    ]
    result = filter_workers_by_fields(
        workers, {"cluster_id": 1}, fuzzy_fields={"name": "foo"}
    )
    assert len(result) == 1
    assert result[0].name == "foo-worker"


def test_filter_workers_by_fields_with_cluster_id():
    """Test that cluster_id parameter filters workers by cluster."""
    workers = [
        DummyWorker("worker-1", "uuid1", 1, id=1),
        DummyWorker("worker-2", "uuid2", 1, id=2),
        DummyWorker("worker-1", "uuid3", 2, id=3),  # Same name in different cluster
    ]
    # Filter by cluster_id parameter (new behavior)
    result = filter_workers_by_fields(
        workers, fields={"deleted_at": None}, cluster_id=1
    )
    assert len(result) == 2
    assert all(w.cluster_id == 1 for w in result)

    # Fuzzy search with cluster_id filter
    result = filter_workers_by_fields(
        workers,
        fields={"deleted_at": None},
        fuzzy_fields={"name": "worker"},
        cluster_id=2,
    )
    assert len(result) == 1
    assert result[0].name == "worker-1"
    assert result[0].cluster_id == 2


def test_find_available_worker_name_basic():
    related_names = {"foo", "foo-1", "foo-2"}
    assert find_available_worker_name("foo", "foo", related_names) == "foo-3"
    assert find_available_worker_name("foo", "foo-2", related_names) == "foo-3"
    assert find_available_worker_name("foo", "foo-1", related_names) == "foo-3"
    assert find_available_worker_name("foo", "foo-10", related_names) == "foo-11"
    assert find_available_worker_name("foo", "foo", set()) == "foo"


def test_retry_create_unique_worker_uuid():
    # Simulate existing uuids
    class Dummy:
        def __init__(self, worker_uuid, cluster_id):
            self.worker_uuid = worker_uuid
            self.cluster_id = cluster_id

    existing = [Dummy("uuid1", 1), Dummy("uuid2", 1), Dummy("uuid3", 2)]
    # Patch uuid4 to control output
    import uuid

    orig_uuid4 = uuid.uuid4
    uuids = iter(["uuid1", "uuid2", "unique-uuid"])
    uuid.uuid4 = lambda: next(uuids)
    result = retry_create_unique_worker_uuid(existing)
    assert result == "unique-uuid"
    uuid.uuid4 = orig_uuid4


class TestCheckWorkerNameConflict:
    """Tests for check_worker_name_conflict function."""

    def test_no_conflict_in_different_clusters(self):
        """Same name in different clusters should not conflict.

        When creating a new worker, the workers list should not include the
        worker being created. This test verifies that the function works
        correctly when workers don't have overlapping names across clusters.
        """
        # Simulate a clean workers list without the worker being created
        workers = [
            DummyWorker("existing-worker", "uuid1", 1, id=1),
            DummyWorker(
                "existing-worker", "uuid2", 2, id=2
            ),  # Same name, different cluster
        ]
        # No exception when cluster 1 has a different worker name
        check_worker_name_conflict(
            "new-worker", workers, cluster_id=1, existing_id=None
        )
        # No exception when cluster 2 has a different worker name
        check_worker_name_conflict(
            "new-worker", workers, cluster_id=2, existing_id=None
        )

    def test_conflict_in_same_cluster(self):
        """Same name in same cluster should conflict."""
        workers = [
            DummyWorker("worker-1", "uuid1", 1, id=1),
            DummyWorker("worker-1", "uuid2", 1, id=2),
        ]
        with pytest.raises(AlreadyExistsException):
            check_worker_name_conflict(
                "worker-1", workers, cluster_id=1, existing_id=None
            )

    def test_exclude_existing_worker_from_conflict_check(self):
        """When updating an existing worker, it should be excluded from conflict check."""
        workers = [
            DummyWorker("worker-1", "uuid1", 1, id=1),
            DummyWorker("worker-2", "uuid2", 1, id=2),
        ]
        # No exception when updating worker-1 (it should be excluded)
        check_worker_name_conflict("worker-1", workers, cluster_id=1, existing_id=1)

    def test_empty_name_allowed_when_no_existing_id(self):
        """Empty name should be allowed when creating a new worker."""
        workers = [
            DummyWorker("worker-1", "uuid1", 1, id=1),
        ]
        # Should not raise any exception
        check_worker_name_conflict("", workers, cluster_id=1, existing_id=None)

    def test_empty_name_invalid_when_updating(self):
        """Empty name should be invalid when updating an existing worker."""
        workers = [
            DummyWorker("worker-1", "uuid1", 1, id=1),
        ]
        from gpustack.api.exceptions import InvalidException

        with pytest.raises(InvalidException):
            check_worker_name_conflict("", workers, cluster_id=1, existing_id=1)


class TestGetExistingWorker:
    """Tests for get_existing_worker function."""

    def test_find_by_external_id_in_same_cluster(self):
        """Should find worker by external_id in the same cluster."""
        workers = [
            DummyWorker("worker-1", "uuid1", 1, id=1, external_id="ext-1"),
            DummyWorker("worker-2", "uuid2", 1, id=2),
        ]
        worker_in = WorkerCreate(
            name="new-worker",
            hostname="host",
            ip="192.168.1.1",
            ifname="eth0",
            port=8080,
            external_id="ext-1",
            worker_uuid="new-uuid",
            status=WorkerStatus.get_default_status(),
        )
        result = get_existing_worker(1, worker_in, workers)
        assert result is not None
        assert result.name == "worker-1"

    def test_find_by_name_with_existence_check_in_same_cluster(self):
        """Should find worker by name when existence check label is set, within same cluster."""
        workers = [
            DummyWorker("worker-1", "uuid1", 1, id=1),
            DummyWorker("worker-1", "uuid2", 2, id=2),  # Same name, different cluster
        ]
        worker_in = WorkerCreate(
            name="worker-1",
            hostname="host",
            ip="192.168.1.1",
            ifname="eth0",
            port=8080,
            worker_uuid="new-uuid",
            status=WorkerStatus.get_default_status(),
            labels={"gpustack.existence-check": "true"},
        )
        # Should find worker in cluster 1, not in cluster 2
        result = get_existing_worker(1, worker_in, workers)
        assert result is not None
        assert result.cluster_id == 1
        assert result.name == "worker-1"

    def test_no_cross_cluster_conflict(self):
        """Same worker name in different clusters should not be treated as existing worker issue."""
        workers = [
            DummyWorker("worker-1", "uuid1", 1, id=1),
            DummyWorker("worker-1", "uuid2", 2, id=2),  # Same name, different cluster
        ]
        worker_in = WorkerCreate(
            name="worker-1",
            hostname="host",
            ip="192.168.1.1",
            ifname="eth0",
            port=8080,
            worker_uuid="new-uuid",
            status=WorkerStatus.get_default_status(),
            labels={"gpustack.existence-check": "true"},
        )
        # Should return the worker in cluster 1, not raise an exception
        result = get_existing_worker(1, worker_in, workers)
        assert result is not None
        assert result.cluster_id == 1

    def test_empty_name_returns_none(self):
        """Empty name should return None (no existing worker check)."""
        workers = [
            DummyWorker("worker-1", "uuid1", 1, id=1),
        ]
        worker_in = WorkerCreate(
            name="",
            hostname="host",
            ip="192.168.1.1",
            ifname="eth0",
            port=8080,
            worker_uuid="new-uuid",
            status=WorkerStatus.get_default_status(),
        )
        result = get_existing_worker(1, worker_in, workers)
        assert result is None

    def test_no_existence_check_label_returns_none(self):
        """Without existence check label, should not search by name."""
        workers = [
            DummyWorker("worker-1", "uuid1", 1, id=1),
        ]
        worker_in = WorkerCreate(
            name="worker-1",
            hostname="host",
            ip="192.168.1.1",
            ifname="eth0",
            port=8080,
            worker_uuid="new-uuid",
            status=WorkerStatus.get_default_status(),
            labels={},  # No existence check label
        )
        result = get_existing_worker(1, worker_in, workers)
        assert result is None


class TestFilterWorkersByFieldsClusterId:
    """Tests for filter_workers_by_fields with cluster_id parameter."""

    def test_cluster_id_filter_only(self):
        """When only cluster_id is specified, return all workers in that cluster."""
        workers = [
            DummyWorker("worker-1", "uuid1", 1, id=1),
            DummyWorker("worker-2", "uuid2", 1, id=2),
            DummyWorker("worker-3", "uuid3", 2, id=3),
        ]
        # Only cluster_id, no fields or fuzzy_fields
        result = filter_workers_by_fields(workers, fields=None, cluster_id=1)
        assert len(result) == 2
        assert all(w.cluster_id == 1 for w in result)

    def test_cluster_id_with_fields(self):
        """cluster_id should be applied even when fields are specified."""
        workers = [
            DummyWorker("worker-1", "uuid1", 1, id=1),
            DummyWorker("worker-1", "uuid2", 2, id=2),  # Same name in different cluster
        ]
        result = filter_workers_by_fields(
            workers, fields={"name": "worker-1"}, cluster_id=1
        )
        assert len(result) == 1
        assert result[0].cluster_id == 1

    def test_cluster_id_with_fuzzy_fields(self):
        """cluster_id should be applied even when fuzzy_fields are specified."""
        workers = [
            DummyWorker("foo-bar", "uuid1", 1, id=1),
            DummyWorker("foo-bar", "uuid2", 2, id=2),  # Same name in different cluster
            DummyWorker("baz-qux", "uuid3", 1, id=3),
        ]
        result = filter_workers_by_fields(
            workers, fields={}, fuzzy_fields={"name": "foo"}, cluster_id=1
        )
        assert len(result) == 1
        assert result[0].name == "foo-bar"
        assert result[0].cluster_id == 1
