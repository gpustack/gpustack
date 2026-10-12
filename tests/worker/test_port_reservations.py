from unittest.mock import MagicMock

import pytest

from gpustack.schemas.cache_services import (
    CacheServiceInstance,
    CacheServiceStateEnum,
)
from gpustack.schemas.common import ItemList
from gpustack.schemas.models import (
    DistributedServers,
    ModelInstance,
    ModelInstanceStateEnum,
    ModelInstanceSubordinateWorker,
    PortBand,
)
from gpustack.worker.port_reservations import WorkerPortReservations
from tests.utils.model import new_model_instance


def _clientset(models=(), caches=()):
    clientset = MagicMock()
    clientset.model_instances.list.return_value = ItemList[ModelInstance](
        items=list(models)
    )
    clientset.cache_service_instances.list.return_value = ItemList[
        CacheServiceInstance
    ](items=list(caches))
    return clientset


def _cache_instance(instance_id=1, worker_id=1, **kwargs):
    return CacheServiceInstance(
        id=instance_id,
        name=f"cache-{instance_id}",
        cache_service_id=1,
        cluster_id=1,
        worker_id=worker_id,
        **kwargs,
    )


def test_local_and_persisted_reservations_both_block_allocation():
    reservations = WorkerPortReservations()
    reservations.model_ports[1] = {40000, 40001}
    reservations.cache_ports[1] = (40001, 40002)
    model = new_model_instance(2, "model-2", 1, worker_id=1)
    model.port = 40003
    model.ports = [40003, 40004]
    cache = _cache_instance(2, port=40005, ports={"metrics": 40006})
    clientset = _clientset(models=[model], caches=[cache])

    with reservations.allocation(clientset, 1) as unavailable:
        assert unavailable == {40000, 40001, 40002, 40003, 40004, 40005, 40006}


@pytest.mark.parametrize(
    "owner, expected",
    [
        ("model_instance_id", {40010, 40011, 40012}),
        ("cache_instance_id", {40000, 40001, 40002}),
    ],
)
def test_reusing_own_ports_keeps_same_id_other_resource_reserved(owner, expected):
    reservations = WorkerPortReservations()
    reservations.model_ports[7] = {40000, 40001}
    reservations.cache_ports[7] = (40010, 40011)
    model = new_model_instance(7, "model-7", 1, worker_id=1)
    model.port = 40001
    model.ports = [40001, 40002]
    cache = _cache_instance(7, port=40011, ports={"address": 40011, "metrics": 40012})
    clientset = _clientset(models=[model], caches=[cache])

    with reservations.allocation(clientset, 1, **{owner: 7}) as unavailable:
        assert unavailable == expected


@pytest.mark.parametrize(
    "worker_id, expected",
    [
        (1, {40000, 40001, 40010, 40011, 40012, 40050}),
        (2, {40000, 40001, 40010, 40011, 40012, 40020, 40021, 40051}),
        (3, {40000, 40001, 40010, 40011, 40012, 40030, 40052}),
        (4, {40040, 40053}),
    ],
)
def test_worker_placement_reserves_complete_bands_and_local_subordinate_ports(
    worker_id, expected
):
    reservations = WorkerPortReservations()
    spanning = new_model_instance(1, "spanning", 1, worker_id=1)
    spanning.port = 40000
    spanning.ports = [40001]
    spanning.named_ports = {"kv": PortBand(base=40010, count=3)}
    spanning.distributed_servers = DistributedServers(
        subordinate_workers=[
            ModelInstanceSubordinateWorker(worker_id=2, ports=[40020, 40021]),
            ModelInstanceSubordinateWorker(worker_id=3, ports=[40030]),
        ]
    )
    other = new_model_instance(2, "other", 1, worker_id=4)
    other.port = 40040
    clientset = _clientset(
        models=[spanning, other],
        caches=[
            _cache_instance(1, worker_id=1, port=40050),
            _cache_instance(2, worker_id=2, port=40051),
            _cache_instance(3, worker_id=3, port=40052),
            _cache_instance(4, worker_id=4, port=40053),
        ],
    )

    with reservations.allocation(clientset, worker_id) as unavailable:
        assert unavailable == expected


@pytest.mark.parametrize("state", list(ModelInstanceStateEnum))
def test_persisted_model_ports_remain_reserved_in_every_state(state):
    reservations = WorkerPortReservations()
    model = new_model_instance(1, "model", 1, worker_id=1, state=state)
    model.port = 40000
    model.named_ports = {"kv": PortBand(base=40010, count=2)}
    clientset = _clientset(models=[model])

    with reservations.allocation(clientset, 1) as unavailable:
        assert unavailable == {40000, 40010, 40011}


@pytest.mark.parametrize("state", list(CacheServiceStateEnum))
def test_persisted_cache_ports_remain_reserved_in_every_state(state):
    reservations = WorkerPortReservations()
    clientset = _clientset(
        caches=[
            _cache_instance(
                state=state,
                port=40000,
                ports={"address": 40000, "metrics": 40001},
            )
        ]
    )

    with reservations.allocation(clientset, 1) as unavailable:
        assert unavailable == {40000, 40001}


@pytest.mark.parametrize("resource", ["model_instances", "cache_service_instances"])
def test_unreadable_persisted_reservations_abort_allocation(resource):
    reservations = WorkerPortReservations()
    reservations.model_ports[1] = {40000}
    reservations.cache_ports[1] = (40001,)
    clientset = _clientset()
    getattr(clientset, resource).list.side_effect = RuntimeError("API unavailable")

    with pytest.raises(RuntimeError, match="API unavailable"):
        with reservations.allocation(clientset, 1):
            pytest.fail("Allocation must reject unreadable persisted reservations")


def test_reservations_use_all_pages_without_the_watch_cache():
    reservations = WorkerPortReservations()
    clientset = _clientset()

    with reservations.allocation(clientset, 3) as unavailable:
        assert unavailable == set()

    clientset.model_instances.list.assert_called_once_with(
        params={"page": -1}, use_cache=False
    )
    clientset.cache_service_instances.list.assert_called_once_with(
        params={"worker_id": 3, "page": -1}, use_cache=False
    )
