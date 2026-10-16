from unittest.mock import MagicMock
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4
import threading

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


@pytest.mark.parametrize("source", ["model", "claim", "cache"])
@pytest.mark.parametrize(
    "placement", ["assigned", "removed", "moved", "current", "empty"]
)
def test_snapshot_identifies_ports_waiting_for_authoritative_cleanup(source, placement):
    reservations = WorkerPortReservations()
    ports = {40020} if placement != "empty" else set()
    if source == "model":
        reservations.model_ports[7] = ports
    elif source == "claim":
        reservations.model_claims[7] = ("attempt", ports)
    else:
        reservations.cache_ports[7] = tuple(ports)
    worker_id = 2 if placement == "moved" else 1
    models, caches = [], []
    if placement in {"assigned", "moved"}:
        if source == "cache":
            caches = [_cache_instance(7, worker_id=worker_id)]
        else:
            models = [new_model_instance(7, "model", 1, worker_id=worker_id)]
    owner = {}
    if placement == "current":
        owner["cache_instance_id" if source == "cache" else "model_instance_id"] = 7
    clientset = _clientset(models, caches)

    with reservations.allocation(clientset, 1, **owner) as unavailable:
        assert unavailable.pending_cleanup is (placement in {"removed", "moved"})
        assert unavailable == (set() if placement == "current" else ports)
    clientset.model_instances.list.assert_called_once_with(
        params={"page": -1}, use_cache=False
    )
    clientset.cache_service_instances.list.assert_called_once_with(
        params={"worker_id": 1, "page": -1}, use_cache=False
    )


def test_follower_assignment_does_not_report_pending_cleanup():
    model = new_model_instance(7, "model", 1, worker_id=2)
    model.distributed_servers = DistributedServers(
        subordinate_workers=[ModelInstanceSubordinateWorker(worker_id=1)]
    )
    reservations = WorkerPortReservations()
    reservations.model_claims[7] = ("attempt", {40020})
    with reservations.allocation(_clientset(models=[model]), 1) as taken:
        assert not taken.pending_cleanup
    reservations.release_model(7)
    with reservations.allocation(_clientset(), 1) as taken:
        assert not taken.pending_cleanup


def _claim(
    reservations,
    token="attempt-a",
    ports=(40020, 40021),
    generation=None,
    replace_version=None,
):
    return reservations.reserve_model(
        _clientset(),
        1,
        7,
        token,
        set(ports),
        "40000-40063",
        "127.0.0.1",
        generation=generation or reservations.claim_generation(7),
        replace_version=replace_version,
    )


def test_concurrent_recovery_has_one_owner_and_fences_old_requests(monkeypatch):
    monkeypatch.setattr("gpustack.utils.network.is_port_available", lambda *args: True)
    reservations = WorkerPortReservations()
    generation = reservations.claim_generation(7)
    assert _claim(reservations, generation=generation).accepted
    _, version = reservations.claim_state(7)

    def recover(token):
        return _claim(
            reservations, token=token, generation=generation, replace_version=version
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(recover, ["attempt-b", "attempt-c"]))
    assert sum(result.accepted for result in results) == 1
    winner = "attempt-b" if results[0].accepted else "attempt-c"
    current = reservations.model_claims[7]
    assert current[0] == winner
    assert not _claim(reservations, generation=generation).accepted
    reservations.release_claim(7, "attempt-a")
    assert reservations.model_claims[7] == current
    assert _claim(reservations, token=winner, generation=generation).accepted


def test_stop_waits_for_claim_registration_before_releasing(monkeypatch):
    reservations = WorkerPortReservations()
    generation = reservations.claim_generation(7)
    probing, finish_probe, releasing = (threading.Event() for _ in range(3))

    def probe(*args):
        assert reservations.lock.locked()
        probing.set()
        assert finish_probe.wait(timeout=3)
        return True

    def stop():
        releasing.set()
        reservations.release_model(7)

    monkeypatch.setattr("gpustack.utils.network.is_port_available", probe)
    with ThreadPoolExecutor(max_workers=2) as executor:
        claim = executor.submit(_claim, reservations, generation=generation)
        try:
            assert probing.wait(timeout=2)
            release = executor.submit(stop)
            assert releasing.wait(timeout=2)
        finally:
            finish_probe.set()
        assert claim.result(timeout=2).accepted
        release.result(timeout=2)

    assert 7 not in reservations.model_claims
    assert 7 not in reservations._claim_versions
    assert reservations.claim_generation(7) != generation


@pytest.mark.parametrize("blocker", ["missing_version", "wrong_version", "committed"])
def test_recovery_requires_the_revision_of_an_uncommitted_claim(monkeypatch, blocker):
    monkeypatch.setattr("gpustack.utils.network.is_port_available", lambda *args: True)
    reservations = WorkerPortReservations()
    assert _claim(reservations).accepted
    generation, version = reservations.claim_state(7)
    if blocker == "committed":
        reservations.model_ports[7] = {40020, 40021}
    elif blocker == "missing_version":
        version = None
    else:
        version = uuid4()

    assert not _claim(
        reservations,
        token="attempt-b",
        generation=generation,
        replace_version=version,
    ).accepted
    assert reservations.model_claims[7][0] == "attempt-a"


def test_tentative_claim_blocks_cache_and_other_models(monkeypatch):
    monkeypatch.setattr("gpustack.utils.network.is_port_available", lambda *args: True)
    reservations = WorkerPortReservations()
    assert _claim(reservations).accepted
    with reservations.allocation(_clientset(), 1, cache_instance_id=7) as taken:
        assert {40020, 40021} <= taken
    with reservations.allocation(_clientset(), 1, model_instance_id=7) as taken:
        assert not taken


@pytest.mark.parametrize("source", ["cache", "model", "socket"])
def test_tentative_claim_rechecks_occupancy_atomically(monkeypatch, source):
    reservations = WorkerPortReservations()
    if source == "cache":
        reservations.cache_ports[2] = (40021,)
    if source == "model":
        reservations.model_ports[2] = {40021}
    monkeypatch.setattr(
        "gpustack.utils.network.is_port_available",
        lambda port, host: source != "socket" or port != 40021,
    )
    result = _claim(reservations)
    assert not result.accepted
    assert result.unavailable_ports == [40021]
    with reservations.allocation(_clientset(), 1) as taken:
        assert 40020 not in taken


def test_rollback_cannot_release_a_different_attempt(monkeypatch):
    monkeypatch.setattr("gpustack.utils.network.is_port_available", lambda *args: True)
    reservations = WorkerPortReservations()
    generation = reservations.claim_generation(7)
    assert _claim(reservations).accepted
    reservations.release_claim(7, "attempt-b")
    with reservations.allocation(_clientset(), 1) as taken:
        assert {40020, 40021} <= taken
    reservations.release_claim(7, "attempt-a")
    with reservations.allocation(_clientset(), 1) as taken:
        assert not taken
    assert not _claim(reservations, generation=generation).accepted


def test_cleanup_releases_claim_only_after_workload_cleanup(monkeypatch):
    monkeypatch.setattr("gpustack.utils.network.is_port_available", lambda *args: True)
    reservations = WorkerPortReservations()
    generation = reservations.claim_generation(7)
    assert _claim(reservations).accepted
    reservations.model_ports[7] = {40020, 40021}
    reservations.release_claim(7, "attempt-a")
    with reservations.allocation(_clientset(), 1) as taken:
        assert {40020, 40021} <= taken
    reservations.release_model(7)
    with reservations.allocation(_clientset(), 1) as taken:
        assert not taken
    assert not _claim(reservations, generation=generation).accepted


def test_repeated_claim_cancellation_uses_constant_per_instance_state(monkeypatch):
    monkeypatch.setattr("gpustack.utils.network.is_port_available", lambda *args: True)
    reservations = WorkerPortReservations()
    original = reservations.claim_generation(7)
    for attempt in range(1000):
        token = str(attempt)
        assert _claim(reservations, token=token).accepted
        reservations.release_claim(7, token)
    assert len(reservations._claim_generations) == 1
    assert not _claim(reservations, token="delayed", generation=original).accepted
    assert _claim(reservations, token="new-attempt").accepted
    generation = reservations.claim_generation(7)
    reservations.release_claim(7, "old-attempt")
    assert reservations.claim_generation(7) == generation


def test_concurrent_claims_have_exactly_one_owner(monkeypatch):
    monkeypatch.setattr("gpustack.utils.network.is_port_available", lambda *args: True)
    reservations = WorkerPortReservations()

    def claim(instance_id):
        return reservations.reserve_model(
            _clientset(),
            1,
            instance_id,
            str(instance_id),
            {40020, 40021},
            "40000-40063",
            "127.0.0.1",
            generation=reservations.claim_generation(instance_id),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(claim, [7, 8]))
    assert sum(result.accepted for result in results) == 1


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
