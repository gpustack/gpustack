from unittest.mock import MagicMock

import pytest

from gpustack.schemas.cache_services import CacheServiceInstance
from gpustack.schemas.common import ItemList
from gpustack.schemas.models import ModelInstance
from gpustack.worker.port_reservations import WorkerPortReservations
from tests.utils.model import new_model_instance


def _clientset():
    clientset = MagicMock()
    clientset.model_instances.list.return_value = ItemList[ModelInstance](items=[])
    clientset.cache_service_instances.list.return_value = ItemList[
        CacheServiceInstance
    ](items=[])
    return clientset


def _recorded_instance(resource):
    if resource == "model":
        instance = new_model_instance(7, "model-7", 1, worker_id=1)
        instance.port = 40000
        instance.ports = [40000]
        return ItemList[ModelInstance](items=[instance])
    instance = CacheServiceInstance(
        id=7,
        name="cache-7",
        cache_service_id=1,
        cluster_id=1,
        worker_id=1,
        port=40000,
        ports={"address": 40000},
    )
    return ItemList[CacheServiceInstance](items=[instance])


def _clientset_with_concurrent_release(reservations, resource, retry_failure=None):
    clientset = _clientset()

    def read_stale_caches(**kwargs):
        # Another owner can complete its lifecycle while an API read is pending.
        assert not reservations.lock.locked()
        with reservations.allocation(
            _clientset(), 1, **{f"{resource}_instance_id": 7}
        ) as unavailable:
            assert 40000 not in unavailable
            getattr(reservations, f"{resource}_ports")[7] = (
                {40000} if resource == "model" else (40000,)
            )
        client = (
            clientset.model_instances
            if resource == "model"
            else clientset.cache_service_instances
        )
        # Separate response objects keep already-read snapshots immutable.
        client.list.return_value = _recorded_instance(resource)
        getattr(reservations, f"release_{resource}")(7)
        clientset.cache_service_instances.list.side_effect = None
        if retry_failure:
            getattr(clientset, retry_failure).list.side_effect = RuntimeError(
                "API unavailable"
            )
        return ItemList[CacheServiceInstance](items=[])

    clientset.cache_service_instances.list.side_effect = read_stale_caches
    return clientset


@pytest.mark.parametrize("resource", ["model", "cache"])
def test_persisted_ports_are_reloaded_after_their_local_reservation_is_released(
    resource,
):
    reservations = WorkerPortReservations()
    clientset = _clientset_with_concurrent_release(reservations, resource)
    allocating_resource = "cache" if resource == "model" else "model"

    with reservations.allocation(
        clientset, 1, **{f"{allocating_resource}_instance_id": 7}
    ) as unavailable:
        port = next(port for port in range(40000, 40002) if port not in unavailable)
        getattr(reservations, f"{allocating_resource}_ports")[7] = (
            {port} if allocating_resource == "model" else (port,)
        )

    assert port == 40001
    assert not getattr(reservations, f"{resource}_ports")
    assert set(getattr(reservations, f"{allocating_resource}_ports")[7]) == {40001}


@pytest.mark.parametrize("resource", ["model", "cache"])
@pytest.mark.parametrize(
    "retry_failure", ["model_instances", "cache_service_instances"]
)
def test_unreadable_refreshed_snapshot_prevents_allocation(resource, retry_failure):
    reservations = WorkerPortReservations()
    clientset = _clientset_with_concurrent_release(
        reservations, resource, retry_failure=retry_failure
    )
    allocating_resource = "cache" if resource == "model" else "model"

    with pytest.raises(RuntimeError, match="API unavailable"):
        with reservations.allocation(
            clientset, 1, **{f"{allocating_resource}_instance_id": 7}
        ):
            pytest.fail("An unreadable snapshot must not allow port allocation")

    assert not reservations.model_ports
    assert not reservations.cache_ports
