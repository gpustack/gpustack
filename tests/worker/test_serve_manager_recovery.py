from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from gpustack.api.exceptions import NotFoundException
from gpustack.schemas.models import (
    BackendEnum,
    DistributedServerCoordinateModeEnum,
    DistributedServers,
    ModelInstance,
    ModelInstanceStateEnum,
    ModelInstanceSubordinateWorker,
)
from gpustack.schemas.workers import WorkerStateEnum
from gpustack.worker import serve_manager
from gpustack_runtime.deployer import WorkloadStatusStateEnum
from tests.utils.model import new_model, new_model_instance


@pytest.fixture
def recovery(monkeypatch, config):
    model = new_model(
        1,
        "test-model",
        huggingface_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        backend=BackendEnum.VLLM,
        backend_version="0.8.0",
    )
    instance = new_model_instance(
        1,
        "distributed-instance",
        model.id,
        worker_id=1,
        state=ModelInstanceStateEnum.UNREACHABLE,
    )
    instance.source = model.source
    instance.huggingface_repo_id = model.huggingface_repo_id
    instance.worker_ip = "10.0.0.1"
    instance.port = 8000
    instance.state_message = "Distributed serving unreachable"
    instance.distributed_servers = DistributedServers(
        mode=DistributedServerCoordinateModeEnum.INITIALIZE_LATER,
        subordinate_workers=[
            ModelInstanceSubordinateWorker(
                worker_id=2,
                worker_name="worker-2",
                worker_ip="10.0.0.2",
                state=ModelInstanceStateEnum.UNREACHABLE,
                state_message="Worker is unreachable from the server",
            ),
        ],
    )
    state = SimpleNamespace(
        instance=ModelInstance.model_validate(instance),
        worker=SimpleNamespace(state=WorkerStateEnum.READY, unreachable=False),
        workload=SimpleNamespace(state=WorkloadStatusStateEnum.RUNNING),
        managers=[],
        clients=[],
    )

    def update_instance(*, id, model_update):
        assert id == state.instance.id
        state.instance = ModelInstance.model_validate(
            state.instance.model_dump() | model_update.model_dump()
        )
        return state.instance.model_copy(deep=True)

    for worker_id in (1, 2):
        client = MagicMock()
        # Each API read is a separate snapshot, not a shared mutable record.
        client.model_instances.list.side_effect = lambda **kwargs: SimpleNamespace(
            items=[state.instance.model_copy(deep=True)]
        )
        client.model_instances.get.side_effect = (
            lambda **kwargs: state.instance.model_copy(deep=True)
        )
        client.model_instances.update.side_effect = update_instance
        client.models.get.return_value = model
        client.workers.get.side_effect = lambda *args, **kwargs: state.worker
        manager = serve_manager.ServeManager(
            lambda worker_id=worker_id: worker_id, lambda client=client: client, config
        )
        manager._inference_backend_manager = MagicMock()
        manager._inference_backend_manager.get_backend_by_name.return_value = (
            SimpleNamespace(health_check_path="/health")
        )
        monkeypatch.setattr(manager, "_ensure_container_log_persistence", MagicMock())
        state.managers.append(manager)
        state.clients.append(client)

    state.get_workload = MagicMock(side_effect=lambda *args: state.workload)
    state.health = MagicMock(return_value=SimpleNamespace(status_code=200))
    monkeypatch.setattr(serve_manager, "get_workload", state.get_workload)
    monkeypatch.setattr(serve_manager.requests, "get", state.health)
    monkeypatch.setattr(
        serve_manager, "get_meta_from_running_instance", lambda *args: {}
    )
    return state


@pytest.mark.parametrize(
    "mode",
    [
        DistributedServerCoordinateModeEnum.INITIALIZE_LATER,
        DistributedServerCoordinateModeEnum.RUN_FIRST,
    ],
)
def test_recovered_subordinate_unblocks_main_health_check(recovery, mode):
    recovery.instance.distributed_servers.mode = mode
    main, subordinate = recovery.managers
    main.sync_model_instances_state()
    recovery.health.assert_not_called()
    assert recovery.instance.state == ModelInstanceStateEnum.UNREACHABLE

    subordinate.sync_model_instances_state()
    sw = recovery.instance.distributed_servers.subordinate_workers[0]
    assert sw.state == ModelInstanceStateEnum.RUNNING
    assert sw.state_message == ""
    assert recovery.instance.state == ModelInstanceStateEnum.UNREACHABLE
    recovery.health.assert_not_called()

    main.sync_model_instances_state()
    recovery.health.assert_called_once()
    assert recovery.instance.state == ModelInstanceStateEnum.RUNNING
    assert recovery.instance.state_message == ""

    if mode == DistributedServerCoordinateModeEnum.INITIALIZE_LATER:
        recovery.clients[1].workers.get.assert_called_once_with(2, use_cache=False)
    for client in recovery.clients:
        client.reset_mock()
    subordinate.sync_model_instances_state()
    main.sync_model_instances_state()
    for client in recovery.clients:
        client.model_instances.update.assert_not_called()
        client.workers.get.assert_not_called()


@pytest.mark.parametrize("initially_ready", [True, False])
def test_recovery_shares_worker_read_only_within_one_sync(recovery, initially_ready):
    client = recovery.clients[1]
    instances = {}
    client.model_instances.list.side_effect = lambda **kwargs: SimpleNamespace(
        items=[instance.model_copy(deep=True) for instance in instances.values()]
    )
    client.model_instances.get.side_effect = lambda id: instances[id].model_copy(
        deep=True
    )

    def update_instance(*, id, model_update):
        instances[id] = ModelInstance.model_validate(
            instances[id].model_dump() | model_update.model_dump()
        )
        return instances[id].model_copy(deep=True)

    client.model_instances.update.side_effect = update_instance
    for ready in (initially_ready, not initially_ready):
        instances = {
            id: recovery.instance.model_copy(
                deep=True, update={"id": id, "name": f"distributed-instance-{id}"}
            )
            for id in (1, 2)
        }
        # Replace the API snapshot so a previous pass cannot observe the change.
        recovery.worker = SimpleNamespace(
            state=WorkerStateEnum.READY if ready else WorkerStateEnum.NOT_READY,
            unreachable=False,
        )
        client.reset_mock()

        recovery.managers[1].sync_model_instances_state()

        client.workers.get.assert_called_once_with(2, use_cache=False)
        assert client.model_instances.update.call_count == (2 if ready else 0)
        expected = (
            ModelInstanceStateEnum.RUNNING
            if ready
            else ModelInstanceStateEnum.UNREACHABLE
        )
        for instance in instances.values():
            assert instance.distributed_servers.subordinate_workers[0].state == expected
            assert instance.state == ModelInstanceStateEnum.UNREACHABLE
        recovery.health.assert_not_called()


@pytest.mark.parametrize(
    "worker_state, unreachable",
    [
        (WorkerStateEnum.UNREACHABLE, True),
        (WorkerStateEnum.NOT_READY, False),
        (WorkerStateEnum.MAINTENANCE, False),
        (WorkerStateEnum.READY, True),
    ],
)
def test_subordinate_waits_for_server_confirmed_recovery(
    recovery, worker_state, unreachable
):
    recovery.worker.state = worker_state
    recovery.worker.unreachable = unreachable
    recovery.managers[1].sync_model_instances_state()
    recovery.clients[1].model_instances.update.assert_not_called()

    recovery.worker.state = WorkerStateEnum.READY
    recovery.worker.unreachable = False
    recovery.managers[1].sync_model_instances_state()
    assert (
        recovery.instance.distributed_servers.subordinate_workers[0].state
        == ModelInstanceStateEnum.RUNNING
    )


@pytest.mark.parametrize(
    "subordinate_state",
    [
        ModelInstanceStateEnum.PENDING,
        ModelInstanceStateEnum.INITIALIZING,
        ModelInstanceStateEnum.RUNNING,
        ModelInstanceStateEnum.ERROR,
    ],
)
def test_initialize_later_recovery_leaves_other_states_unchanged(
    recovery, subordinate_state
):
    sw = recovery.instance.distributed_servers.subordinate_workers[0]
    sw.state = subordinate_state
    recovery.managers[1].sync_model_instances_state()
    assert recovery.instance.distributed_servers.subordinate_workers[0] == sw
    recovery.clients[1].model_instances.update.assert_not_called()
    recovery.clients[1].workers.get.assert_not_called()


@pytest.mark.parametrize(
    "workload_state, expected",
    [
        (WorkloadStatusStateEnum.PENDING, ModelInstanceStateEnum.UNREACHABLE),
        (WorkloadStatusStateEnum.INITIALIZING, ModelInstanceStateEnum.UNREACHABLE),
        (WorkloadStatusStateEnum.UNHEALTHY, ModelInstanceStateEnum.ERROR),
        (WorkloadStatusStateEnum.FAILED, ModelInstanceStateEnum.ERROR),
        (None, ModelInstanceStateEnum.ERROR),
    ],
)
def test_subordinate_does_not_recover_without_running_workload(
    recovery, workload_state, expected
):
    recovery.workload = (
        SimpleNamespace(state=workload_state, state_message="", exits=[])
        if workload_state is not None
        else None
    )
    recovery.managers[1].sync_model_instances_state()
    assert (
        recovery.instance.distributed_servers.subordinate_workers[0].state == expected
    )
    recovery.clients[1].workers.get.assert_not_called()


def test_recovered_subordinate_does_not_bypass_main_health_check(recovery):
    recovery.health.return_value.status_code = 503
    recovery.managers[1].sync_model_instances_state()
    assert (
        recovery.instance.distributed_servers.subordinate_workers[0].state
        == ModelInstanceStateEnum.RUNNING
    )
    recovery.managers[0].sync_model_instances_state()
    recovery.health.assert_called_once()
    assert recovery.instance.state == ModelInstanceStateEnum.UNREACHABLE


@pytest.mark.parametrize(
    "peer_state",
    [
        ModelInstanceStateEnum.UNREACHABLE,
        ModelInstanceStateEnum.ERROR,
    ],
)
def test_recovery_does_not_override_another_subordinate(recovery, peer_state):
    peer = ModelInstanceSubordinateWorker(
        worker_id=3,
        worker_name="worker-3",
        worker_ip="10.0.0.3",
        state=peer_state,
        state_message="peer unavailable",
    )
    recovery.instance.distributed_servers.subordinate_workers.insert(0, peer)
    recovery.managers[1].sync_model_instances_state()
    subordinates = recovery.instance.distributed_servers.subordinate_workers
    assert subordinates[0] == peer
    assert subordinates[1].state == ModelInstanceStateEnum.RUNNING
    recovery.managers[0].sync_model_instances_state()
    assert recovery.instance.state == peer_state
    recovery.health.assert_not_called()


def test_recovery_does_not_clear_main_error(recovery):
    recovery.instance.state = ModelInstanceStateEnum.ERROR
    recovery.instance.state_message = "Inference health check failed."
    recovery.managers[1].sync_model_instances_state()
    recovery.managers[0].sync_model_instances_state()
    assert recovery.instance.state == ModelInstanceStateEnum.ERROR
    assert recovery.instance.state_message == "Inference health check failed."
    recovery.health.assert_not_called()


@pytest.mark.parametrize("deleted", [False, True])
def test_recovery_requires_successful_worker_read(recovery, deleted):
    client = recovery.clients[1]
    client.workers.get.side_effect = (
        NotFoundException() if deleted else httpx.ConnectError("API unavailable")
    )
    if deleted:
        recovery.managers[1].sync_model_instances_state()
    else:
        with pytest.raises(httpx.ConnectError, match="API unavailable"):
            recovery.managers[1].sync_model_instances_state()
    client.model_instances.update.assert_not_called()
    client.workers.get.side_effect = lambda *args, **kwargs: recovery.worker
    recovery.managers[1].sync_model_instances_state()
    assert (
        recovery.instance.distributed_servers.subordinate_workers[0].state
        == ModelInstanceStateEnum.RUNNING
    )


def test_failed_recovery_write_does_not_mutate_watch_cache(recovery):
    client = recovery.clients[1]
    cached = recovery.instance.model_copy(deep=True)
    client.model_instances.list.side_effect = lambda **kwargs: SimpleNamespace(
        items=[cached]
    )
    persist_update = client.model_instances.update.side_effect
    client.model_instances.update.side_effect = httpx.ConnectError("API unavailable")
    with pytest.raises(httpx.ConnectError, match="API unavailable"):
        recovery.managers[1].sync_model_instances_state()
    assert (
        cached.distributed_servers.subordinate_workers[0].state
        == ModelInstanceStateEnum.UNREACHABLE
    )
    assert (
        recovery.instance.distributed_servers.subordinate_workers[0].state
        == ModelInstanceStateEnum.UNREACHABLE
    )

    client.model_instances.update.side_effect = persist_update
    recovery.managers[1].sync_model_instances_state()
    assert (
        recovery.instance.distributed_servers.subordinate_workers[0].state
        == ModelInstanceStateEnum.RUNNING
    )
