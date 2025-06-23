import pytest

from gpustack.schemas import ModelInstance
from gpustack.schemas.models import (
    DistributedServers,
    ModelInstanceSubordinateWorker,
    ModelInstanceStateEnum,
)
from gpustack.utils.attrs import get_attr, set_attr


@pytest.mark.parametrize(
    "o, path, expected",
    [
        # Dict access
        (
            {"a": {"b": {"c": 42}}},
            "a.b.c",
            42,
        ),
        # Dict access with list index
        (
            {"a": [{"b": {"c": 42}}]},
            "a.0.b.c",
            42,
        ),
        # None access
        (
            None,
            "a.b.c",
            None,
        ),
        # Dict access with on-existent path
        (
            {"a": {"b": {"c": 42}}},
            "a.b.d",
            None,
        ),
        # List access
        (
            [1, 2, 3],
            "0",
            1,
        ),
        # List of dicts access
        (
            [{"a": 1}, {"b": 2}],
            "0.a",
            1,
        ),
        # Complex object access
        (
            ModelInstance(
                distributed_servers=DistributedServers(
                    subordinate_workers=[
                        ModelInstanceSubordinateWorker(
                            worker_ip="192.168.50.3",
                        ),
                    ],
                ),
            ),
            "distributed_servers.subordinate_workers.0.worker_ip",
            "192.168.50.3",
        ),
        # Complex object access with non-existent path
        (
            ModelInstance(
                distributed_servers=DistributedServers(
                    subordinate_workers=[
                        ModelInstanceSubordinateWorker(
                            worker_ip="192.168.50.3",
                        ),
                    ],
                ),
            ),
            "distributed_servers.subordinate_workers.0.name",
            None,
        ),
    ],
)
@pytest.mark.unit
def test_get_attr(o, path, expected):
    actual = get_attr(o, path)
    assert (
        actual == expected
    ), f"Expected {expected} but got {actual} for path '{path}' in object {o}"


@pytest.mark.parametrize(
    "o, path, value, expected",
    [
        # Dict access
        (
            {"a": {"b": {"c": 42}}},
            "a.b.c",
            100,
            {"a": {"b": {"c": 100}}},
        ),
        # Dict access with list index
        (
            {"a": [{"b": {"c": 42}}]},
            "a.0.b.c",
            100,
            {"a": [{"b": {"c": 100}}]},
        ),
        # None access
        (
            None,
            "a.b.c",
            100,
            None,
        ),
        # Dict access with non-existent path: insert new item
        (
            {"a": {"b": {"c": 42}}},
            "a.b.d",
            100,
            {"a": {"b": {"c": 42, "d": 100}}},
        ),
        # Dict access with non-existent path: nothing to do
        (
            {"a": {"b": {"c": 42}}},
            "a.d.c",
            100,
            {"a": {"b": {"c": 42}}},
        ),
        # List access
        (
            [1, 2, 3],
            "0",
            100,
            [100, 2, 3],
        ),
        # List of dicts access
        (
            [{"a": 1}, {"b": 2}],
            "0.a",
            100,
            [{"a": 100}, {"b": 2}],
        ),
        # Complex object access
        (
            ModelInstance(
                distributed_servers=DistributedServers(
                    subordinate_workers=[
                        ModelInstanceSubordinateWorker(
                            worker_ip="192.168.50.3",
                            state=ModelInstanceStateEnum.RUNNING,
                        ),
                        ModelInstanceSubordinateWorker(
                            worker_ip="192.168.50.5",
                            state=ModelInstanceStateEnum.ERROR,
                        ),
                    ],
                ),
            ),
            "distributed_servers.subordinate_workers.0.worker_ip",
            "192.168.50.4",
            ModelInstance(
                distributed_servers=DistributedServers(
                    subordinate_workers=[
                        ModelInstanceSubordinateWorker(
                            worker_ip="192.168.50.4",
                            state=ModelInstanceStateEnum.RUNNING,
                        ),
                        ModelInstanceSubordinateWorker(
                            worker_ip="192.168.50.5",
                            state=ModelInstanceStateEnum.ERROR,
                        ),
                    ],
                ),
            ),
        ),
        # Complex object access: replace an item
        (
            ModelInstance(
                distributed_servers=DistributedServers(
                    subordinate_workers=[
                        ModelInstanceSubordinateWorker(
                            worker_ip="192.168.50.3",
                            state=ModelInstanceStateEnum.RUNNING,
                        ),
                        ModelInstanceSubordinateWorker(
                            worker_ip="192.168.50.5",
                            state=ModelInstanceStateEnum.ERROR,
                        ),
                    ],
                ),
            ),
            "distributed_servers.subordinate_workers.-1",
            ModelInstanceSubordinateWorker(
                worker_ip="192.168.50.4",
            ),
            ModelInstance(
                distributed_servers=DistributedServers(
                    subordinate_workers=[
                        ModelInstanceSubordinateWorker(
                            worker_ip="192.168.50.3",
                            state=ModelInstanceStateEnum.RUNNING,
                        ),
                        ModelInstanceSubordinateWorker(
                            worker_ip="192.168.50.4",
                        ),
                    ],
                ),
            ),
        ),
        # Complex object access with non-existent path: insert new item
        (
            ModelInstance(
                distributed_servers=DistributedServers(
                    subordinate_workers=[
                        ModelInstanceSubordinateWorker(
                            worker_ip="192.168.50.3",
                            state=ModelInstanceStateEnum.RUNNING,
                        ),
                        ModelInstanceSubordinateWorker(
                            worker_ip="192.168.50.5",
                            state=ModelInstanceStateEnum.ERROR,
                        ),
                    ],
                ),
            ),
            "distributed_servers.subordinate_workers.2",
            ModelInstanceSubordinateWorker(
                worker_ip="192.168.50.4",
            ),
            ModelInstance(
                distributed_servers=DistributedServers(
                    subordinate_workers=[
                        ModelInstanceSubordinateWorker(
                            worker_ip="192.168.50.3",
                            state=ModelInstanceStateEnum.RUNNING,
                        ),
                        ModelInstanceSubordinateWorker(
                            worker_ip="192.168.50.5",
                            state=ModelInstanceStateEnum.ERROR,
                        ),
                        ModelInstanceSubordinateWorker(
                            worker_ip="192.168.50.4",
                        ),
                    ],
                ),
            ),
        ),
        # Complex object access with non-existent path: nothing to do
        (
            ModelInstance(
                distributed_servers=DistributedServers(
                    subordinate_workers=[
                        ModelInstanceSubordinateWorker(
                            worker_ip="192.168.50.3",
                            state=ModelInstanceStateEnum.RUNNING,
                        ),
                        ModelInstanceSubordinateWorker(
                            worker_ip="192.168.50.5",
                            state=ModelInstanceStateEnum.ERROR,
                        ),
                    ],
                ),
            ),
            "distributed_servers.subordinate_workers.0.name",
            "test",
            ModelInstance(
                distributed_servers=DistributedServers(
                    subordinate_workers=[
                        ModelInstanceSubordinateWorker(
                            worker_ip="192.168.50.3",
                            state=ModelInstanceStateEnum.RUNNING,
                        ),
                        ModelInstanceSubordinateWorker(
                            worker_ip="192.168.50.5",
                            state=ModelInstanceStateEnum.ERROR,
                        ),
                    ],
                ),
            ),
        ),
    ],
)
@pytest.mark.unit
def test_set_attr(o, path, value, expected):
    set_attr(o, path, value)
    actual = o
    assert (
        actual == expected
    ), f"Expected {expected} but got {actual} for path '{path}' in object {o}"
