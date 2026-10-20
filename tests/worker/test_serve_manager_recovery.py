from types import SimpleNamespace
from datetime import datetime, timedelta, timezone
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
from gpustack.server.bus import Event, EventType
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
    "subordinate_state",
    [
        ModelInstanceStateEnum.UNREACHABLE,
        ModelInstanceStateEnum.PENDING,
        ModelInstanceStateEnum.ERROR,
    ],
)
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
    recovery, worker_state, unreachable, subordinate_state
):
    sw = recovery.instance.distributed_servers.subordinate_workers[0]
    sw.state = subordinate_state
    sw.state_message = "Inference server exited or unhealthy."
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
    "subordinate_state, message",
    [
        (ModelInstanceStateEnum.RUNNING, ""),
        (ModelInstanceStateEnum.ERROR, "Failed to run vLLM: invalid configuration"),
        (ModelInstanceStateEnum.ERROR, "Inference health check failed."),
    ],
)
def test_initialize_later_recovery_preserves_running_and_explicit_errors(
    recovery, subordinate_state, message
):
    sw = recovery.instance.distributed_servers.subordinate_workers[0]
    sw.state = subordinate_state
    sw.state_message = message
    recovery.managers[1].sync_model_instances_state()
    assert recovery.instance.distributed_servers.subordinate_workers[0] == sw
    recovery.clients[1].model_instances.update.assert_not_called()
    recovery.clients[1].workers.get.assert_not_called()


@pytest.mark.parametrize("role", [None, "prefill", "decode"])
@pytest.mark.parametrize("restart_on_error", [False, True])
@pytest.mark.parametrize(
    "subordinate_state",
    [
        ModelInstanceStateEnum.PENDING,
        ModelInstanceStateEnum.SCHEDULED,
        ModelInstanceStateEnum.INITIALIZING,
        ModelInstanceStateEnum.DOWNLOADING,
        ModelInstanceStateEnum.STARTING,
        ModelInstanceStateEnum.ERROR,
    ],
)
def test_running_follower_converges_without_restarting(
    recovery, monkeypatch, role, restart_on_error, subordinate_state
):
    instance = recovery.instance
    instance.role = role
    instance.state = ModelInstanceStateEnum.STARTING
    instance.restart_count = 0
    sw = instance.distributed_servers.subordinate_workers[0]
    sw.state = subordinate_state
    sw.state_message = "Inference server exited or unhealthy."
    if subordinate_state == ModelInstanceStateEnum.ERROR:
        instance.state = ModelInstanceStateEnum.ERROR
        instance.state_message = (
            "Distributed serving error in subordinate worker 10.0.0.2: "
            "Inference server exited or unhealthy.."
        )
    for client in recovery.clients:
        client.models.get.return_value.restart_on_error = restart_on_error
    main, follower = recovery.managers
    for manager in (main, follower):
        monkeypatch.setattr(manager, "_start_model_instance", MagicMock())
        monkeypatch.setattr(manager, "_stop_model_instance", MagicMock())
        if instance.state == ModelInstanceStateEnum.ERROR:
            manager._error_model_instances[instance.id] = instance.model_copy(deep=True)

    follower.sync_model_instances_state()
    recovery.get_workload.assert_called_with(instance.name + "-f0")
    assert recovery.instance.distributed_servers.subordinate_workers[0].state == (
        ModelInstanceStateEnum.RUNNING
    )
    recovery.health.assert_not_called()
    main.sync_model_instances_state()
    recovery.health.assert_called_once()
    assert recovery.instance.state == ModelInstanceStateEnum.RUNNING
    assert recovery.instance.state_message == ""
    assert recovery.instance.restart_count == 0
    for manager in (main, follower):
        manager._start_model_instance.assert_not_called()
        manager._stop_model_instance.assert_not_called()
        assert instance.id not in manager._error_model_instances


@pytest.mark.parametrize("status", [503, None])
def test_distributed_error_recovery_still_requires_main_health(recovery, status):
    instance = recovery.instance
    instance.state = ModelInstanceStateEnum.ERROR
    instance.state_message = "Distributed serving error in subordinate worker 10.0.0.2"
    if status is None:
        recovery.health.side_effect = httpx.ConnectError("engine is starting")
    else:
        recovery.health.return_value.status_code = status
    recovery.managers[1].sync_model_instances_state()
    recovery.managers[0].sync_model_instances_state()
    recovery.health.assert_called_once()
    assert recovery.instance.state == ModelInstanceStateEnum.ERROR


@pytest.mark.parametrize(
    "message, reason, exit_code, diagnosis",
    [
        ("CrashLoopBackOff", "Error", 1, ""),
        ("OOMKilled", "OOMKilled", 137, "CUDA out of memory"),
        ("", "OOMKilled", 137, ""),
        ("ImagePullBackOff", "", None, ""),
        ("", "", None, ""),
    ],
)
def test_subordinate_recovers_after_diagnosed_workload_failure(
    recovery, monkeypatch, message, reason, exit_code, diagnosis
):
    main, follower = recovery.managers
    recovery.workload = SimpleNamespace(
        state=WorkloadStatusStateEnum.FAILED,
        state_message=message,
        exits=[SimpleNamespace(name="default", reason=reason, exit_code=exit_code)],
    )
    failure_message = serve_manager._describe_workload_failure(recovery.workload)
    if diagnosis:
        failure_message += f"\n{diagnosis}"
    monkeypatch.setattr(
        follower, "_append_log_diagnosis", lambda *args: failure_message
    )

    follower.sync_model_instances_state()
    sw = recovery.instance.distributed_servers.subordinate_workers[0]
    assert sw.state == ModelInstanceStateEnum.ERROR
    assert failure_message in sw.state_message

    # The main workload survives while the follower's failure is propagated.
    recovery.workload = SimpleNamespace(state=WorkloadStatusStateEnum.RUNNING)
    main.sync_model_instances_state()
    assert recovery.instance.state == ModelInstanceStateEnum.ERROR
    recovery.health.assert_not_called()

    follower.sync_model_instances_state()
    sw = recovery.instance.distributed_servers.subordinate_workers[0]
    assert sw.state == ModelInstanceStateEnum.RUNNING
    assert sw.state_message == ""
    main.sync_model_instances_state()
    recovery.health.assert_called_once()
    assert recovery.instance.state == ModelInstanceStateEnum.RUNNING
    assert recovery.instance.state_message == ""


@pytest.mark.parametrize(
    "subordinate_state",
    [
        ModelInstanceStateEnum.UNREACHABLE,
        ModelInstanceStateEnum.PENDING,
        ModelInstanceStateEnum.ERROR,
    ],
)
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
    recovery, workload_state, expected, subordinate_state
):
    sw = recovery.instance.distributed_servers.subordinate_workers[0]
    sw.state = subordinate_state
    sw.state_message = "Inference server exited or unhealthy."
    recovery.workload = (
        SimpleNamespace(state=workload_state, state_message="", exits=[])
        if workload_state is not None
        else None
    )
    recovery.managers[1].sync_model_instances_state()
    if expected == ModelInstanceStateEnum.UNREACHABLE:
        expected = subordinate_state
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


@pytest.mark.parametrize("recovery_source", ["event", "sync", "restart"])
def test_recovery_watch_events_do_not_restart_healthy_follower(
    recovery, recovery_source
):
    main, follower = recovery.managers
    recovery.clients[1].models.get.return_value.restart_on_error = True
    recovery.instance.state = ModelInstanceStateEnum.ERROR
    recovery.instance.state_message = (
        "Distributed serving error in subordinate worker 10.0.0.2"
    )
    follower.sync_model_instances_state()
    follower._dispatch_model_instance_event(
        Event(type=EventType.UPDATED, data=recovery.instance.model_copy(deep=True))
    )
    queued_error = follower._error_model_instances[recovery.instance.id]

    main.sync_model_instances_state()
    assert recovery.instance.state == ModelInstanceStateEnum.RUNNING
    recovery.clients[1].reset_mock()
    if recovery_source == "event":
        follower._dispatch_model_instance_event(
            Event(type=EventType.UPDATED, data=recovery.instance.model_copy(deep=True))
        )
    elif recovery_source == "sync":
        follower.sync_model_instances_state()
    if recovery_source != "restart":
        assert recovery.instance.id not in follower._error_model_instances

    # The restart loop may already have copied the error queue before recovery.
    follower._restart_error_model_instance(queued_error)

    assert recovery.instance.state == ModelInstanceStateEnum.RUNNING
    recovery.clients[1].model_instances.update.assert_not_called()
    recovery.clients[1].model_instances.get.assert_called_with(
        id=recovery.instance.id, use_cache=False
    )
    assert recovery.instance.id not in follower._error_model_instances


@pytest.mark.parametrize("worker_index", [0, 1])
@pytest.mark.parametrize("deleted", [False, True])
def test_restart_requires_successful_live_instance_read(
    recovery, worker_index, deleted
):
    manager = recovery.managers[worker_index]
    client = recovery.clients[worker_index]
    recovery.instance.state = ModelInstanceStateEnum.ERROR
    client.models.get.return_value.restart_on_error = True
    queued_error = recovery.instance.model_copy(deep=True)
    manager._error_model_instances[queued_error.id] = queued_error
    client.model_instances.get.side_effect = (
        NotFoundException() if deleted else httpx.ConnectError("API unavailable")
    )

    if deleted:
        manager._restart_error_model_instance(queued_error)
        assert queued_error.id not in manager._error_model_instances
    else:
        with pytest.raises(httpx.ConnectError, match="API unavailable"):
            manager._restart_error_model_instance(queued_error)
        assert queued_error.id in manager._error_model_instances
    client.model_instances.update.assert_not_called()
    assert queued_error.id not in manager._restart_backoff_counts


@pytest.mark.parametrize("membership", ["absent", "unknown", "empty", "other-worker"])
def test_removed_follower_discards_queued_recovery(recovery, membership):
    follower = recovery.managers[1]
    client = recovery.clients[1]
    recovery.instance.state = ModelInstanceStateEnum.ERROR
    recovery.instance.group_id = "1-old"
    queued = recovery.instance.model_copy(deep=True)
    follower._error_model_instances[queued.id] = queued
    client.models.get.return_value.restart_on_error = True
    if membership == "absent":
        recovery.instance.distributed_servers = None
    elif membership == "unknown":
        recovery.instance.distributed_servers.subordinate_workers = None
    elif membership == "empty":
        recovery.instance.distributed_servers.subordinate_workers = []
    else:
        recovery.instance.distributed_servers.subordinate_workers[0].worker_id = 3

    follower._restart_error_model_instance(queued)

    assert queued.id not in follower._error_model_instances
    assert queued.id not in follower._pending_model_instance_errors
    assert queued.id not in follower._restart_backoff_counts
    client.models.get.assert_not_called()
    client.model_instances.update.assert_not_called()


@pytest.mark.parametrize("worker_index", [0, 1])
def test_restart_uses_live_error_and_restart_count(recovery, worker_index):
    manager = recovery.managers[worker_index]
    client = recovery.clients[worker_index]
    client.models.get.return_value.restart_on_error = True
    recovery.instance.state = ModelInstanceStateEnum.ERROR
    recovery.instance.restart_count = 1
    queued_error = recovery.instance.model_copy(deep=True)
    manager._error_model_instances[queued_error.id] = queued_error
    recovery.instance.restart_count = 3
    if worker_index == 1:
        recovery.instance.state = ModelInstanceStateEnum.STARTING
        recovery.instance.distributed_servers.subordinate_workers[0].state = (
            ModelInstanceStateEnum.ERROR
        )

    manager._restart_error_model_instance(queued_error)

    assert recovery.instance.state == ModelInstanceStateEnum.SCHEDULED
    assert recovery.instance.restart_count == 4
    assert queued_error.id not in manager._error_model_instances


@pytest.mark.parametrize("worker_index", [0, 1])
def test_restart_rechecks_backoff_against_live_instance(recovery, worker_index):
    manager = recovery.managers[worker_index]
    client = recovery.clients[worker_index]
    recovery.instance.state = ModelInstanceStateEnum.ERROR
    queued_error = recovery.instance.model_copy(deep=True)
    queued_error.last_restart_time = datetime.now(timezone.utc) - timedelta(minutes=6)
    manager._error_model_instances[queued_error.id] = queued_error
    manager._restart_backoff_counts[queued_error.id] = 6
    recovery.instance.last_restart_time = datetime.now(timezone.utc)

    manager._restart_error_model_instance(queued_error)

    client.model_instances.get.assert_called_once_with(
        id=queued_error.id, use_cache=False
    )
    client.models.get.assert_not_called()
    client.model_instances.update.assert_not_called()
    assert manager._restart_backoff_counts[queued_error.id] == 6
    assert queued_error.id in manager._error_model_instances


@pytest.mark.parametrize("worker_index", [0, 1])
@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_pd_blocked_recovery_preserves_error_without_status_write(
    recovery, worker_index, role
):
    manager = recovery.managers[worker_index]
    client = recovery.clients[worker_index]
    recovery.instance.state = ModelInstanceStateEnum.ERROR
    recovery.instance.role = role
    recovery.instance.group_id = "1-old"
    local = (
        recovery.instance
        if worker_index == 0
        else recovery.instance.distributed_servers.subordinate_workers[0]
    )
    local.state = ModelInstanceStateEnum.ERROR
    local.state_message = "Inference server exited or unhealthy. Exit code 137."
    client.models.get.return_value.stale = True
    client.models.get.return_value.restart_on_error = True
    queued = recovery.instance.model_copy(deep=True)
    manager._error_model_instances[queued.id] = queued
    original = recovery.instance.model_dump()

    manager._restart_error_model_instance(queued)

    assert recovery.instance.model_dump() == original
    assert queued.id not in manager._error_model_instances
    assert queued.id not in manager._restart_backoff_counts
    client.model_instances.get.assert_called_once_with(id=queued.id, use_cache=False)
    client.model_instances.update.assert_not_called()


@pytest.mark.parametrize("worker_index", [0, 1])
def test_pd_recovery_polling_uses_watch_cache(recovery, worker_index):
    manager = recovery.managers[worker_index]
    client = recovery.clients[worker_index]
    recovery.instance.group_id = "1-old"
    client.models.get.return_value.stale = True
    client.models.get.return_value.restart_on_error = True

    assert not manager._auto_restart_enabled(recovery.instance)

    client.models.get.assert_called_with(recovery.instance.model_id, use_cache=True)
    client.model_instances.get.assert_not_called()
    client.model_instances.update.assert_not_called()


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


@pytest.mark.parametrize(
    "subordinate_state",
    [
        ModelInstanceStateEnum.UNREACHABLE,
        ModelInstanceStateEnum.PENDING,
        ModelInstanceStateEnum.ERROR,
    ],
)
def test_failed_recovery_write_does_not_mutate_watch_cache(recovery, subordinate_state):
    client = recovery.clients[1]
    sw = recovery.instance.distributed_servers.subordinate_workers[0]
    sw.state = subordinate_state
    sw.state_message = "Inference server exited or unhealthy."
    cached = recovery.instance.model_copy(deep=True)
    client.model_instances.list.side_effect = lambda **kwargs: SimpleNamespace(
        items=[cached]
    )
    persist_update = client.model_instances.update.side_effect
    client.model_instances.update.side_effect = httpx.ConnectError("API unavailable")
    with pytest.raises(httpx.ConnectError, match="API unavailable"):
        recovery.managers[1].sync_model_instances_state()
    assert cached.distributed_servers.subordinate_workers[0].state == subordinate_state
    assert (
        recovery.instance.distributed_servers.subordinate_workers[0].state
        == subordinate_state
    )

    client.model_instances.update.side_effect = persist_update
    recovery.managers[1].sync_model_instances_state()
    assert (
        recovery.instance.distributed_servers.subordinate_workers[0].state
        == ModelInstanceStateEnum.RUNNING
    )
