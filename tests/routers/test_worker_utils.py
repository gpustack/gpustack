from gpustack.routes.workers import (
    filter_workers_by_fields,
    find_available_worker_name,
    retry_create_unique_worker_uuid,
)


class DummyWorker:
    def __init__(self, name, worker_uuid, cluster_id, deleted_at=None, labels=None):
        self.name = name
        self.worker_uuid = worker_uuid
        self.cluster_id = cluster_id
        self.deleted_at = deleted_at
        self.labels = labels or {}


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
