from concurrent.futures import ThreadPoolExecutor
from functools import partial
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gpustack.schemas.cache_services import CacheServiceInstance, CacheServiceStateEnum
from gpustack.schemas.models import BackendEnum
from gpustack.worker.cache_service_manager import CacheServiceManager
from gpustack.worker.port_reservations import WorkerPortReservations
from gpustack.worker.serve_manager import ServeManager
from tests.utils.model import new_model, new_model_instance


def _pick_port(port_range, unavailable_ports, **kwargs):
    start, end = map(int, port_range.split("-"))
    return next(port for port in range(start, end + 1) if port not in unavailable_ports)


@pytest.fixture
def managers(tmp_path, monkeypatch):
    clientset = MagicMock()
    clientset.model_instances.list.return_value = SimpleNamespace(items=[])
    clientset.cache_service_instances.list.return_value = SimpleNamespace(items=[])
    config = SimpleNamespace(log_dir=str(tmp_path), service_port_range="40032-40063")
    reservations = WorkerPortReservations()
    serve = ServeManager(lambda: 1, lambda: clientset, config, reservations)
    cache = CacheServiceManager(
        lambda: 1,
        lambda: "127.0.0.1",
        lambda: clientset,
        config,
        MagicMock(),
        reservations,
    )
    monkeypatch.setattr("gpustack.utils.network.get_free_port", _pick_port)
    monkeypatch.setattr("gpustack.utils.network.is_port_available", lambda *args: True)
    return serve, cache, clientset


def _cache_instance(**kwargs):
    return CacheServiceInstance(
        id=11, name="cache", cache_service_id=5, worker_id=1, cluster_id=1, **kwargs
    )


def _router(instance_id=11):
    instance = new_model_instance(instance_id, "router", 1, worker_id=1)
    instance.worker_ip = "127.0.0.1"
    instance.role = "router"
    return instance


def _allocate_router(manager, instance):
    model = new_model(1, "model", huggingface_repo_id="test/model")
    manager._assign_ports(instance, model, BackendEnum.CUSTOM)
    return set(instance.ports)


@pytest.mark.parametrize("persisted", [False, True])
@pytest.mark.parametrize(
    "state",
    [
        CacheServiceStateEnum.PENDING,
        CacheServiceStateEnum.STARTING,
        CacheServiceStateEnum.ERROR,
    ],
)
def test_unbound_cache_http_port_is_not_assigned_to_router(managers, persisted, state):
    serve, cache, clientset = managers
    instance = _cache_instance(
        port=40054, ports={"port": 40054, "metrics": 40032}, state=state
    )
    if persisted:
        # A fresh worker sees persisted reservations before watch events arrive.
        clientset.cache_service_instances.list.return_value.items = [instance]
    else:
        cache._allocate_ports(instance, ["port", "metrics"])

    router = _router()
    _allocate_router(serve, router)

    assert router.port == 40033
    assert not set(router.ports) & set(instance.ports.values())


@pytest.mark.parametrize("persisted", [False, True])
def test_unbound_router_port_is_not_assigned_to_cache(managers, persisted):
    serve, cache, clientset = managers
    router = _router()
    if persisted:
        router.port = 40032
        router.ports = [40032]
        clientset.model_instances.list.return_value.items = [router]
    else:
        _allocate_router(serve, router)

    ports = cache._allocate_ports(_cache_instance(), ["port", "metrics"])

    assert ports == {"port": 40033, "metrics": 40034}
    assert router.id == _cache_instance().id


@pytest.mark.parametrize("owner", ["cache", "model"])
def test_restart_keeps_own_ports_while_fencing_the_other_kind(managers, owner):
    serve, cache, _ = managers
    cache_instance = _cache_instance(ports={"port": 40054, "metrics": 40032})
    router = _router()
    if owner == "cache":
        expected = cache._allocate_ports(cache_instance, ["port", "metrics"])
        _allocate_router(serve, router)
        assert cache._allocate_ports(cache_instance, ["port", "metrics"]) == expected
    else:
        expected = _allocate_router(serve, router)
        cache._allocate_ports(_cache_instance(), ["port", "metrics"])
        assert _allocate_router(serve, router) == expected


def test_existing_model_port_conflict_is_reported_before_launch(managers):
    serve, _, clientset = managers
    clientset.cache_service_instances.list.return_value.items = [
        _cache_instance(ports={"metrics": 40032})
    ]
    router = _router()
    router.port = 40032
    router.ports = [40032]

    with pytest.raises(RuntimeError, match="40032.*reserved by another instance"):
        _allocate_router(serve, router)

    assert router.id not in serve._assigned_ports


class _ObservedLock:
    def __init__(self):
        self.lock = threading.Lock()
        self.waiter = threading.Event()

    def __enter__(self):
        if self.lock.locked():
            self.waiter.set()
        self.lock.acquire()

    def __exit__(self, *args):
        self.lock.release()


@pytest.mark.parametrize("first_owner", ["model", "cache"])
def test_concurrent_starts_reserve_before_the_other_allocator_selects(
    managers, monkeypatch, first_owner
):
    serve, cache, _ = managers
    lock = _ObservedLock()
    serve._port_reservations.lock = lock
    picked = threading.Event()
    finish = threading.Event()

    def pick(port_range, unavailable_ports, **kwargs):
        port = _pick_port(port_range, unavailable_ports)
        if not picked.is_set():
            picked.set()
            assert finish.wait(5)
        return port

    monkeypatch.setattr("gpustack.utils.network.get_free_port", pick)
    allocations = {
        "model": lambda: _allocate_router(serve, _router()),
        "cache": lambda: set(
            cache._allocate_ports(_cache_instance(), ["port"]).values()
        ),
    }
    second_owner = "cache" if first_owner == "model" else "model"
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(allocations[first_owner])
        try:
            assert picked.wait(5)
            second = executor.submit(allocations[second_owner])
            assert lock.waiter.wait(5)
        finally:
            finish.set()
        assert not first.result(timeout=5) & second.result(timeout=5)


@pytest.mark.parametrize("owner", ["cache", "model"])
def test_successful_workload_deletion_releases_ports_to_the_other_kind(managers, owner):
    serve, cache, _ = managers
    if owner == "cache":
        instance = _cache_instance()
        ports = cache._allocate_ports(instance, ["port"])
        with patch("gpustack.worker.cache_service_manager.delete_workload"):
            cache._stop_cache_service_instance(instance)
        assert _allocate_router(serve, _router()) == set(ports.values())
    else:
        router = _router()
        ports = _allocate_router(serve, router)
        with (
            patch.object(serve, "_stop_container_log_persistence"),
            patch("gpustack.worker.serve_manager.delete_workload"),
        ):
            serve._stop_model_instance(router)
        assert set(cache._allocate_ports(_cache_instance(), ["port"]).values()) == ports


def test_deleted_cache_keeps_ports_until_in_flight_provisioning_finishes(managers):
    serve, cache, _ = managers
    instance = _cache_instance()
    cache_ports = cache._allocate_ports(instance, ["port", "metrics"])
    assert cache._claim_start(instance.id)
    cancelled = cache._start_tokens[instance.id]

    with patch("gpustack.worker.cache_service_manager.delete_workload"):
        cache._stop_cache_service_instance(instance)
        router_ports = _allocate_router(serve, _router())
        assert not router_ports & set(cache_ports.values())
        cache._finish_start(instance, cancelled)

    assert _allocate_router(serve, _router(instance_id=12)) == {40032}


@pytest.mark.parametrize("resource", ["model_instances", "cache_service_instances"])
@pytest.mark.parametrize("owner", ["model", "cache"])
def test_slow_api_read_does_not_block_workload_cleanup(managers, resource, owner):
    serve, cache, clientset = managers
    if owner == "model":
        instance = _router()
        _allocate_router(serve, instance)
        manager = serve
        stop = partial(serve._stop_model_instance, instance)
        allocate = partial(cache._allocate_ports, _cache_instance(), ["port"])
    else:
        instance = _cache_instance()
        cache._allocate_ports(instance, ["port"])
        manager = cache
        stop = partial(cache._stop_cache_service_instance, instance)
        allocate = partial(_allocate_router, serve, _router())

    reading = threading.Event()
    finish_read = threading.Event()

    def read(**kwargs):
        reading.set()
        assert finish_read.wait(5)
        return SimpleNamespace(items=[])

    getattr(clientset, resource).list.side_effect = read
    with (
        patch.object(serve, "_stop_container_log_persistence"),
        patch("gpustack.worker.serve_manager.delete_workload"),
        patch("gpustack.worker.cache_service_manager.delete_workload"),
        ThreadPoolExecutor(max_workers=2) as executor,
    ):
        pending = executor.submit(allocate)
        try:
            assert reading.wait(5)
            executor.submit(stop).result(timeout=2)
            assert instance.id not in manager._assigned_ports
        finally:
            finish_read.set()
        pending.result(timeout=5)
