import pytest

from gpustack.schemas.cache_services import CacheServiceInstance


@pytest.mark.parametrize("name", ["test-server-ddo69", "shared-kv-master-a1b2c"])
def test_deployment_uses_instance_name_and_identity_labels(name):
    instance = CacheServiceInstance(
        id=11,
        name=name,
        cache_service_id=5,
        worker_id=1,
        cluster_id=1,
    )

    metadata = instance.get_deployment_metadata()

    assert metadata.name == name
    assert metadata.labels == {
        "type": "cache-service",
        "cache-service-id": "5",
        "cache-service-instance-id": "11",
    }
