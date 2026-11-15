import pytest

from gpustack.schemas.models import (
    DistributedServers,
    ModelInstance,
    ModelInstanceSubordinateWorker,
)


@pytest.mark.parametrize("index", [0, 9, 99, 100])
def test_deployment_metadata_bounds_follower_names(index):
    instance = ModelInstance(
        name="m" * 53 + "-abc12",
        worker_id=1,
        distributed_servers=DistributedServers(
            subordinate_workers=[
                ModelInstanceSubordinateWorker(worker_id=i + 2)
                for i in range(index + 1)
            ]
        ),
    )
    metadata = instance.get_deployment_metadata(index + 2)
    assert len(metadata.name) <= 63
    assert metadata.name.endswith(f"-f{index}")
    assert metadata.distributed_follower_index == index
    assert instance.get_deployment_metadata(index + 2).name == metadata.name
    assert instance.get_deployment_metadata(1).name == instance.name
    assert instance.get_deployment_metadata(index + 3) is None
