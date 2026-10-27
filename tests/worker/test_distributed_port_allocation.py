import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import aiohttp
import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gpustack.api.exceptions import ServiceUnavailableException, register_handlers
from gpustack.client.distributed_port_client import DistributedPortClient
from gpustack.routes import model_instance_ports as server_routes
from gpustack.routes.worker import port_reservations as worker_routes
from gpustack.schemas.models import (
    BackendEnum,
    DistributedServerCoordinateModeEnum,
    DistributedServers,
    ModelInstanceSubordinateWorker,
)
from gpustack.schemas.port_reservations import ModelPortClaim
from gpustack.server.deps import get_session, get_tenant_context
from gpustack.utils import network
from gpustack.worker.cache_service_manager import CacheServiceManager
from gpustack.worker.port_reservations import (
    WorkerPortReservations,
    model_instance_ports,
)
from gpustack.worker.serve_manager import ServeManager
from tests.utils.model import new_model, new_model_instance
from tests.worker.test_port_reservations import _cache_instance


@pytest.fixture
def cluster(tmp_path, monkeypatch, request):
    primary_id = getattr(request, "param", 1)
    instance = new_model_instance(
        7, "distributed", 1, worker_id=primary_id, gpu_indexes=[0]
    )
    instance.cluster_id = 1
    instance.worker_ip = "127.0.0.1"
    instance.distributed_servers = DistributedServers(
        mode=DistributedServerCoordinateModeEnum.INITIALIZE_LATER,
        subordinate_workers=[
            ModelInstanceSubordinateWorker(worker_id=3 - primary_id, gpu_indexes=[0])
        ],
    )
    model = new_model(
        1,
        "model",
        huggingface_repo_id="Qwen/Qwen3-0.6B",
        backend_parameters=["--distributed-executor-backend=mp"],
    )
    workers = {}
    apps = {}
    managers = {}
    for worker_id in (1, 2):
        clientset = MagicMock()
        clientset.model_instances.get.return_value = instance
        clientset.model_instances.list.return_value = SimpleNamespace(items=[instance])
        clientset.cache_service_instances.list.return_value = SimpleNamespace(items=[])
        config = SimpleNamespace(
            token="registration-token",
            log_dir=str(tmp_path / str(worker_id)),
            service_port_range="40000-40063",
        )
        manager = ServeManager(lambda w=worker_id: w, lambda c=clientset: c, config)
        app = FastAPI()
        app.state.config = config
        app.state.token = f"worker-{worker_id}"
        app.state.worker_ip_getter = lambda: "127.0.0.1"
        app.state.serve_manager = manager
        app.include_router(worker_routes.router)
        register_handlers(app)
        managers[worker_id] = manager
        apps[worker_id] = app
        workers[worker_id] = SimpleNamespace(
            id=worker_id, cluster_id=1, token=app.state.token
        )

    server = FastAPI()
    server.state.http_client = MagicMock()
    server.state.http_client_no_proxy = MagicMock()
    server.include_router(server_routes.router, prefix="/model-instances")
    server.dependency_overrides[get_session] = lambda: AsyncMock()
    server.dependency_overrides[get_tenant_context] = lambda: SimpleNamespace(
        user=SimpleNamespace(worker=workers[primary_id])
    )
    register_handlers(server)
    monkeypatch.setattr(server_routes, "assert_resource_visible", MagicMock())
    monkeypatch.setattr(
        server_routes.ModelInstance, "one_by_id", AsyncMock(return_value=instance)
    )
    monkeypatch.setattr(
        server_routes.Worker,
        "one_by_id",
        AsyncMock(side_effect=lambda session, worker_id: workers[worker_id]),
    )
    calls = []
    hook = MagicMock()

    async def bridge(**kwargs):
        worker_id, method = kwargs["worker"].id, kwargs["method"]
        calls.append((worker_id, method))
        payload = json.loads(kwargs["data"]) if kwargs.get("data") else None
        hook(worker_id, method, payload)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(apps[worker_id]), base_url="http://worker"
        ) as client:
            response = await client.request(
                method,
                "/" + kwargs["path"],
                params=kwargs.get("params"),
                content=kwargs["data"],
                headers={
                    **kwargs["headers"],
                    "Authorization": f"Bearer {workers[worker_id].token}",
                },
            )
        return SimpleNamespace(status=response.status_code), response.content

    monkeypatch.setattr(server_routes, "request_to_worker", bridge)
    monkeypatch.setattr(network, "is_port_available", lambda *args: True)

    with TestClient(server) as http:
        managers[primary_id]._distributed_port_client = DistributedPortClient(
            SimpleNamespace(http_client=SimpleNamespace(get_httpx_client=lambda: http))
        )
        yield SimpleNamespace(
            managers=managers, instance=instance, model=model, calls=calls, hook=hook
        )


def _allocate(cluster):
    primary_id = cluster.instance.worker_id
    cluster.managers[primary_id]._assign_ports(
        cluster.instance, cluster.model, BackendEnum.VLLM
    )
    return model_instance_ports(cluster.instance, primary_id)


def _reserve_tentative_ports(cluster):
    coordinator = cluster.managers[cluster.instance.worker_id]._port_coordinator()
    snapshots = coordinator.availability(7)
    old = ModelPortClaim(
        token=uuid4(),
        generations={item.worker_id: item.generation for item in snapshots},
        ports=list(range(40000, 40014)),
    )
    assert coordinator.reserve(7, old).accepted
    return coordinator, old


@pytest.mark.parametrize("cluster", [1, 2], indirect=True)
def test_primary_restart_recovers_unpublished_claims_and_fences_old_requests(cluster):
    primary_id = cluster.instance.worker_id
    main, follower = cluster.managers[primary_id], cluster.managers[3 - primary_id]
    coordinator, old = _reserve_tentative_ports(cluster)
    assert cluster.instance.port is None

    # Restarting the primary loses its tentative registry and coordinator token.
    main._port_reservations = WorkerPortReservations()
    main._assigned_ports = main._port_reservations.model_ports
    assert not main._distributed_port_claims
    ports = _allocate(cluster)

    current = follower._port_reservations.model_claims[7]
    assert current[0] != str(old.token)
    assert current[1] == ports
    assert not coordinator.reserve(7, old).accepted
    coordinator.release(7, old.token)
    assert follower._port_reservations.model_claims[7] == current
    assert main._port_reservations.model_claims[7] == current


@pytest.mark.parametrize("cluster", [1, 2], indirect=True)
def test_recovery_checks_primary_before_mutating_followers(cluster):
    coordinator, old = _reserve_tentative_ports(cluster)
    snapshots = coordinator.availability(7)
    replacement = old.model_copy(
        update={
            "token": uuid4(),
            "replaces": {
                item.worker_id: item.claim_version
                for item in snapshots
                if item.worker_id != cluster.instance.worker_id
            },
        }
    )

    assert not coordinator.reserve(7, replacement).accepted
    for manager in cluster.managers.values():
        assert manager._port_reservations.model_claims[7][0] == str(old.token)


def test_superseded_claim_cannot_publish_after_all_acknowledgements(
    cluster, monkeypatch
):
    main = cluster.managers[1]
    coordinator = main._port_coordinator()
    reserve = coordinator.reserve
    recovered = []

    def recover_before_acknowledgement(instance_id, claim):
        result = reserve(instance_id, claim)
        assert result.accepted
        main._port_reservations.release_claim(instance_id, str(claim.token))
        snapshots = coordinator.availability(instance_id)
        replacement = ModelPortClaim(
            token=uuid4(),
            generations={item.worker_id: item.generation for item in snapshots},
            replaces={
                item.worker_id: item.claim_version
                for item in snapshots
                if item.claim_version is not None
            },
            ports=claim.ports,
        )
        assert reserve(instance_id, replacement).accepted
        recovered.append(str(replacement.token))
        return result

    monkeypatch.setattr(coordinator, "reserve", recover_before_acknowledgement)
    with pytest.raises(RuntimeError, match="claim changed before publication"):
        _allocate(cluster)

    assert cluster.instance.port is None
    assert not main._assigned_ports
    for manager in cluster.managers.values():
        assert manager._port_reservations.model_claims[7][0] == recovered[0]


def test_live_mp_only_protocol_reserves_five_fragmented_ports_on_both_nodes(cluster):
    cluster.model.backend_parameters.extend(["--tp=2", "--dp=1"])
    free = {40000, 40002, 40004, 40006, 40008}
    for manager in cluster.managers.values():
        manager._config.service_port_range = "40000-40008"
        manager._port_reservations.cache_ports[99] = (40001, 40003, 40005, 40007)

    ports = _allocate(cluster)

    assert ports == free
    assert len(cluster.instance.ports) == 5
    assert not cluster.instance.named_ports
    for manager in cluster.managers.values():
        assert manager._port_reservations.model_claims[7][1] == free
    follower = cluster.managers[2]
    follower._assign_ports(cluster.instance, cluster.model, BackendEnum.VLLM)
    assert follower._assigned_ports[7] == free


def test_live_protocol_reserves_complete_band_on_both_nodes(cluster):
    main, follower = cluster.managers[1], cluster.managers[2]
    main._port_reservations.cache_ports[90] = tuple(range(40000, 40019))
    follower._port_reservations.cache_ports[102] = (40021, 40029)
    follower._port_reservations.cache_ports[114] = (40047, 40022)
    ports = _allocate(cluster)
    assert len(ports) == 14
    assert not ports & {40021, 40022, 40029, 40047}
    for manager in (main, follower):
        assert manager._port_reservations.model_claims[7][1] == ports

    cache = CacheServiceManager(
        lambda: 2,
        lambda: "127.0.0.1",
        lambda: follower._clientset,
        follower._config,
        MagicMock(),
        follower._port_reservations,
    )
    assert not ports & set(
        cache._allocate_ports(_cache_instance(99, 2), ["port", "metrics"]).values()
    )


def test_cache_racing_the_snapshot_forces_a_new_attempt(cluster):
    contested = []

    def take_port(worker_id, method, payload):
        if worker_id == 2 and method == "POST" and not contested:
            contested.append(payload["ports"][0])
            cluster.managers[2]._port_reservations.cache_ports[99] = tuple(contested)

    cluster.hook.side_effect = take_port
    ports = _allocate(cluster)
    assert contested
    assert not ports & set(contested)
    assert cluster.calls.count((1, "POST")) == 2
    assert cluster.calls.count((1, "DELETE")) == 1
    for manager in cluster.managers.values():
        assert manager._port_reservations.model_claims[7][1] == ports


def test_stale_generation_retries_without_excluding_still_free_ports(cluster):
    free_ports = set(range(40002, 40012)) | {40015, 40018, 40021, 40024}
    busy_ports = set(range(40000, 40064)) - free_ports
    for worker_id, manager in cluster.managers.items():
        cache = _cache_instance(
            90 + worker_id,
            worker_id,
            ports={f"port-{port}": port for port in busy_ports},
        )
        manager._clientset.cache_service_instances.list.return_value = SimpleNamespace(
            items=[cache]
        )
    rotated = []

    def cancel_before_follower_claim(worker_id, method, payload):
        if worker_id == 2 and method == "POST" and not rotated:
            registry = cluster.managers[2]._port_reservations
            rotated.append(registry.claim_generation(7))
            registry.release_claim(7, "old-delayed")

    cluster.hook.side_effect = cancel_before_follower_claim
    ports = _allocate(cluster)

    assert ports == free_ports
    for worker_id, manager in cluster.managers.items():
        assert cluster.calls.count((worker_id, "POST")) == 2
        assert manager._port_reservations.model_claims[7][1] == free_ports


def test_local_conflict_after_all_worker_claims_rolls_back_and_retries(cluster):
    contested = []
    first_token = []

    def take_port_after_primary_claim(worker_id, method, payload):
        if worker_id == 2 and method == "POST" and not contested:
            contested.append(payload["ports"][0])
            first_token.append(payload["token"])
            cluster.managers[1]._port_reservations.cache_ports[99] = tuple(contested)

    cluster.hook.side_effect = take_port_after_primary_claim
    ports = _allocate(cluster)

    assert contested
    assert not ports & set(contested)
    for worker_id, manager in cluster.managers.items():
        assert cluster.calls.count((worker_id, "POST")) == 2
        assert cluster.calls.count((worker_id, "DELETE")) == 1
        token, claimed_ports = manager._port_reservations.model_claims[7]
        assert token != first_token[0]
        assert claimed_ports == ports
    assert not cluster.managers[1]._pending_distributed_port_releases


@pytest.mark.parametrize("range_on_follower", ["40016-40047", "40100-40163"])
def test_allocation_uses_only_the_common_range(cluster, range_on_follower):
    cluster.managers[2]._config.service_port_range = range_on_follower
    if range_on_follower == "40100-40163":
        with pytest.raises(network.PortRangeExhaustedError, match="no common"):
            _allocate(cluster)
        assert not any(method == "POST" for _, method in cluster.calls)
        assert cluster.instance.port is None
    else:
        assert all(40016 <= port <= 40047 for port in _allocate(cluster))


def test_failed_worker_rpc_rolls_back_the_primary_claim(cluster):
    def disconnect(worker_id, method, payload):
        if worker_id == 2 and method == "POST":
            raise aiohttp.ClientConnectionError("worker disconnected")

    cluster.hook.side_effect = disconnect
    with pytest.raises(ServiceUnavailableException, match="worker 2"):
        _allocate(cluster)
    assert cluster.instance.port is None
    assert not cluster.managers[1]._assigned_ports
    for manager in cluster.managers.values():
        assert not manager._port_reservations.model_claims


def test_failed_rollback_keeps_cleanup_metadata_for_retry(cluster):
    def disconnect(worker_id, method, payload):
        if (worker_id, method) in [(2, "POST"), (1, "DELETE")]:
            raise aiohttp.ClientConnectionError("worker disconnected")

    cluster.hook.side_effect = disconnect
    with pytest.raises(
        ServiceUnavailableException, match="Cannot coordinate model ports with worker 2"
    ):
        _allocate(cluster)
    main = cluster.managers[1]
    assert 7 in main._distributed_port_claims
    assert 7 in main._port_reservations.model_claims
    cluster.hook.side_effect = None
    assert len(_allocate(cluster)) == 14


def test_rejected_claim_with_failed_rollback_keeps_its_token(cluster):
    def reject_and_disconnect(worker_id, method, payload):
        if worker_id == 2 and method == "POST":
            cluster.managers[2]._port_reservations.cache_ports[99] = (
                payload["ports"][0],
            )
        if worker_id == 1 and method == "DELETE":
            raise aiohttp.ClientConnectionError("rollback disconnected")

    cluster.hook.side_effect = reject_and_disconnect
    with pytest.raises(ServiceUnavailableException):
        _allocate(cluster)
    main = cluster.managers[1]
    assert cluster.instance.port is None
    assert not main._assigned_ports
    assert main._pending_distributed_port_releases
    assert cluster.calls.count((1, "POST")) == 1
    assert str(main._distributed_port_claims[7]) == (
        main._port_reservations.model_claims[7][0]
    )
    cluster.hook.side_effect = None
    assert len(_allocate(cluster)) == 14


def test_cleanup_preserves_ports_of_a_follower_until_its_workload_stops(cluster):
    main, follower = cluster.managers[1], cluster.managers[2]
    ports = _allocate(cluster)
    follower._assigned_ports[7] = set(ports)
    with patch(
        "gpustack.worker.serve_manager.delete_workload",
        side_effect=RuntimeError("runtime unavailable"),
    ):
        with pytest.raises(RuntimeError, match="runtime unavailable"):
            main._stop_model_instance(cluster.instance)
    assert main._port_reservations.model_claims[7][1] == ports
    assert follower._port_reservations.model_claims[7][1] == ports

    with patch("gpustack.worker.serve_manager.delete_workload"):
        main._stop_model_instance(cluster.instance)
        assert not main._port_reservations.model_claims
        assert not follower._port_reservations.model_claims
        assert follower._assigned_ports[7] == ports
        follower._stop_model_instance(cluster.instance)
        assert not follower._assigned_ports


def _fragmented_cache_layout(cluster):
    cache_ports = {
        1: {
            40048,
            40050,
            40051,
            40057,
            40059,
            40001,
            40012,
            40029,
            40031,
            40062,
            40056,
            40060,
        },
        2: {40021, 40047, 40036, 40050, 40029, 40022, 40030, 40031},
    }
    for worker_id, ports in cache_ports.items():
        manager = cluster.managers[worker_id]
        cache = _cache_instance(
            90 + worker_id,
            worker_id,
            ports={f"port-{index}": port for index, port in enumerate(sorted(ports))},
        )
        manager._clientset.cache_service_instances.list.return_value = SimpleNamespace(
            items=[cache]
        )
    return {40009, 40010, 40011, 40020} | set(range(40037, 40047))


def _hold_deleted_owner_ports(manager, source, ports):
    if source == "model":
        manager._port_reservations.model_ports[99] = ports
    elif source == "claim":
        manager._port_reservations.model_claims[99] = ("old-attempt", ports)
    else:
        manager._port_reservations.cache_ports[99] = tuple(ports)


@pytest.mark.parametrize("source", ["model", "claim", "cache"])
@pytest.mark.parametrize("pending_worker", [1, 2])
def test_deleted_owner_cleanup_allows_a_waiting_distributed_allocation(
    cluster, source, pending_worker
):
    old_ports = _fragmented_cache_layout(cluster)
    manager = cluster.managers[pending_worker]
    _hold_deleted_owner_ports(manager, source, old_ports)

    def finish_cleanup(delay):
        assert delay == 2
        assert cluster.instance.port is None
        with manager._port_reservations.allocation(
            manager._clientset, pending_worker, model_instance_id=7
        ) as taken:
            assert old_ports <= taken
        if source == "cache":
            manager._port_reservations.release_cache(99)
        else:
            manager._port_reservations.release_model(99)

    with patch(
        "gpustack.worker.serve_manager.time.sleep", side_effect=finish_cleanup
    ) as sleep:
        ports = _allocate(cluster)

    sleep.assert_called_once_with(2)
    assert len(ports) == 14
    assert cluster.instance.port is not None
    for worker in cluster.managers.values():
        assert worker._port_reservations.model_claims[7][1] == ports


def test_exhaustion_without_pending_cleanup_fails_without_waiting(cluster):
    old_ports = _fragmented_cache_layout(cluster)
    old = new_model_instance(99, "other-deployment", 1, worker_id=1)
    old.ports = sorted(old_ports)
    old.distributed_servers = cluster.instance.distributed_servers.model_copy(deep=True)
    for manager in cluster.managers.values():
        manager._clientset.model_instances.list.return_value = SimpleNamespace(
            items=[cluster.instance, old]
        )

    with patch("gpustack.worker.serve_manager.time.sleep") as sleep:
        with pytest.raises(network.PortRangeExhaustedError, match="No band of 10"):
            _allocate(cluster)

    sleep.assert_not_called()
    assert cluster.instance.port is None
    assert not any(method == "POST" for _, method in cluster.calls)


def test_pending_cleanup_wait_is_bounded_and_does_not_publish_partial_ports(cluster):
    old_ports = _fragmented_cache_layout(cluster)
    main = cluster.managers[1]
    _hold_deleted_owner_ports(main, "model", old_ports)

    with patch("gpustack.worker.serve_manager.time.sleep") as sleep:
        with pytest.raises(network.PortRangeExhaustedError, match="No band of 10"):
            _allocate(cluster)

    assert sleep.call_count == 15
    assert all(call.args == (2,) for call in sleep.call_args_list)
    assert cluster.instance.port is None
    assert not cluster.instance.ports
    assert not cluster.instance.named_ports
    assert cluster.instance.id not in main._assigned_ports
    assert main._assigned_ports[99] == old_ports
    assert not main._distributed_port_claims
    assert main._port_reservations.model_ports[99] == old_ports
    assert not any(method == "POST" for _, method in cluster.calls)


@pytest.mark.parametrize("status_code", [204, 404])
def test_client_cleanup_keeps_the_claim_token_out_of_the_url(status_code):
    from uuid import uuid4

    token = uuid4()

    def respond(request):
        assert not request.url.query
        assert request.headers["X-Claim-Token"] == str(token)
        return httpx.Response(status_code)

    with httpx.Client(
        transport=httpx.MockTransport(respond), base_url="http://server"
    ) as http:
        client = DistributedPortClient(
            SimpleNamespace(http_client=SimpleNamespace(get_httpx_client=lambda: http))
        )
        client.release(7, token)


def test_repeated_local_conflicts_stop_after_five_claim_attempts(cluster):
    contested = []

    def take_a_port_from_every_attempt(worker_id, method, payload):
        if worker_id == 2 and method == "POST":
            contested.append(payload["ports"][0])
            cluster.managers[1]._port_reservations.cache_ports[99] = tuple(contested)

    cluster.hook.side_effect = take_a_port_from_every_attempt
    with pytest.raises(RuntimeError, match="changed during cross-worker allocation"):
        _allocate(cluster)

    assert len(contested) == 5
    for worker_id, manager in cluster.managers.items():
        assert cluster.calls.count((worker_id, "POST")) == 5
        assert cluster.calls.count((worker_id, "DELETE")) == 5
        assert not manager._port_reservations.model_claims
    assert cluster.instance.port is None
    assert not cluster.managers[1]._assigned_ports
