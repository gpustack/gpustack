from types import SimpleNamespace
from datetime import datetime, timedelta, timezone
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ReadTimeout

from gpustack.api.exceptions import NotFoundException
from gpustack.schemas.models import (
    BackendEnum,
    DistributedServerCoordinateModeEnum,
    DistributedServers,
    ModelInstanceStateEnum,
    ModelInstanceSubordinateWorker,
    PortBand,
    RoleSpec,
    SourceEnum,
)
from gpustack.server.bus import Event, EventType
from gpustack.server.controllers import model_members_stale, model_spec_digest
from gpustack.utils import network
from gpustack.worker.serve_manager import ServeManager
from gpustack.worker.backends.base import InferenceServer
from gpustack_runtime.deployer import WorkloadStatusStateEnum
from tests.utils.model import new_model, new_model_instance
from tests.worker.test_port_reservations import _cache_instance
from tests.worker.test_serve_manager_ports import (
    _instance as _pd_instance,
    _pd_model,
    _wide_band_mode,
)


def _instance(instance_id=1):
    instance = new_model_instance(
        instance_id,
        f"distributed-{instance_id}",
        1,
        worker_id=1,
        state=ModelInstanceStateEnum.RUNNING,
    )
    instance.source = SourceEnum.HUGGING_FACE
    instance.huggingface_repo_id = "Qwen/Qwen3-0.6B"
    instance.worker_ip = "127.0.0.1"
    instance.port = 40000
    instance.ports = [40000, 40011, 40012, 40013, 40001]
    instance.named_ports = {"_vllm_mp_connecting": PortBand(base=40001, count=10)}
    instance.distributed_servers = DistributedServers(
        mode=DistributedServerCoordinateModeEnum.INITIALIZE_LATER,
        subordinate_workers=[ModelInstanceSubordinateWorker(worker_id=2)],
    )
    return instance


@pytest.fixture(params=[1, 2])
def manager(request, tmp_path, monkeypatch):
    clientset = MagicMock()
    clientset.model_instances.list.return_value = SimpleNamespace(items=[])
    clientset.cache_service_instances.list.return_value = SimpleNamespace(items=[])
    config = SimpleNamespace(
        log_dir=str(tmp_path),
        service_port_range="40000-40063",
        system_default_container_registry=None,
    )
    manager = ServeManager(lambda: request.param, lambda: clientset, config)
    manager._inference_backend_manager = MagicMock()
    model = new_model(
        1,
        "model",
        huggingface_repo_id="Qwen/Qwen3-0.6B",
        backend_parameters=["--distributed-executor-backend", "mp"],
    )
    clientset.models.get.return_value = model
    monkeypatch.setattr(manager, "_get_model", MagicMock(return_value=model))
    for name in (
        "_align_instance_log_dir",
        "_cleanup_old_logs",
        "_purge_instance_logs",
        "_start_container_log_persistence",
        "_stop_container_log_persistence",
        "_update_model_instance",
    ):
        monkeypatch.setattr(manager, name, MagicMock())
    process = MagicMock()
    process.is_alive.return_value = False
    process.pid = 1234
    monkeypatch.setattr(
        "gpustack.worker.serve_manager.multiprocessing.Process",
        MagicMock(return_value=process),
    )
    monkeypatch.setattr(
        "gpustack.worker.serve_manager.terminate_process_tree", MagicMock()
    )
    monkeypatch.setattr(
        "gpustack.worker.serve_manager.registration.determine_default_registry",
        lambda *args: None,
    )
    monkeypatch.setattr(
        "gpustack.worker.serve_manager.get_workload",
        MagicMock(return_value=SimpleNamespace(state=WorkloadStatusStateEnum.RUNNING)),
    )
    monkeypatch.setattr("gpustack.utils.network.is_port_available", lambda *args: True)
    return manager, clientset


@pytest.mark.parametrize("delete_event", [False, True])
def test_repeated_recreation_releases_all_local_ports(manager, delete_event):
    serve, _ = manager
    with patch("gpustack.worker.serve_manager.delete_workload") as delete:
        for instance_id in range(1, 21):
            instance = _instance(instance_id)
            serve._update_model_instance.reset_mock()
            serve._start_model_instance(instance)
            assert len(serve._assigned_ports[instance_id]) == 14
            update = serve._update_model_instance.call_args.kwargs
            if serve._worker_id == 1:
                assert update["state"] == ModelInstanceStateEnum.INITIALIZING
            else:
                assert update["distributed_servers.subordinate_workers.0"].state == (
                    ModelInstanceStateEnum.RUNNING
                )
            if delete_event:
                serve._dispatch_model_instance_event(
                    Event(type=EventType.DELETED, data=instance)
                )
            serve.sync_model_instances_state()
            delete.assert_called_with(
                instance.get_deployment_metadata(serve._worker_id).name
            )
            assert not serve._assigned_ports
            assert not serve._model_instance_by_instance_id
        assert delete.call_count == 20


def test_failed_cleanup_keeps_reservations_until_retry_succeeds(manager):
    serve, _ = manager
    instance = _instance()
    serve._start_model_instance(instance)
    ports = set(serve._assigned_ports[instance.id])

    def delete_workload(name):
        assert serve._assigned_ports[instance.id] == ports

    with patch(
        "gpustack.worker.serve_manager.delete_workload",
        side_effect=RuntimeError("runtime unavailable"),
    ):
        serve.sync_model_instances_state()
    assert serve._assigned_ports[instance.id] == ports
    assert instance.id in serve._model_instance_by_instance_id

    with patch(
        "gpustack.worker.serve_manager.delete_workload", side_effect=delete_workload
    ) as delete:
        serve.sync_model_instances_state()
    delete.assert_called_once()
    assert instance.id not in serve._assigned_ports


def test_cleanup_waits_for_in_flight_start(manager):
    serve, _ = manager
    instance = _instance()
    observed = []
    with (
        patch("gpustack.worker.serve_manager.delete_workload") as delete,
        patch("gpustack.worker.serve_manager.multiprocessing.Process") as process,
    ):
        process.return_value.pid = 1234
        process.return_value.is_alive.return_value = False

        def start():
            serve.sync_model_instances_state()
            observed.append(
                (delete.call_count, len(serve._assigned_ports.get(instance.id, set())))
            )

        process.return_value.start.side_effect = start
        serve._start_model_instance(instance)
        assert observed == [(0, 14)]
        serve.sync_model_instances_state()
    delete.assert_called_once_with(
        instance.get_deployment_metadata(serve._worker_id).name
    )
    assert instance.id not in serve._assigned_ports


def test_concurrent_start_returns_while_the_same_instance_is_starting(manager):
    serve, _ = manager
    instance = _instance()
    entered, release, second_returned = (threading.Event() for _ in range(3))

    def launch(mi):
        entered.set()
        assert release.wait(timeout=3)

    def duplicate_start():
        try:
            serve._start_model_instance(instance)
        finally:
            second_returned.set()

    with patch.object(serve, "_launch_model_instance", side_effect=launch) as start:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(serve._start_model_instance, instance)
            try:
                assert entered.wait(timeout=2)
                second = executor.submit(duplicate_start)
                assert second_returned.wait(timeout=1)
            finally:
                release.set()
            first.result(timeout=2)
            second.result(timeout=2)

    start.assert_called_once_with(instance)
    assert instance.id not in serve._starting_model_instances


@pytest.mark.parametrize("restart_on_error", [False, True])
def test_delayed_events_cannot_start_a_deleted_instance(manager, restart_on_error):
    serve, clientset = manager
    instance = _instance()
    serve._get_model.return_value.restart_on_error = restart_on_error
    instance.state = ModelInstanceStateEnum.STARTING
    instance.distributed_servers.subordinate_workers[0].state = (
        ModelInstanceStateEnum.PENDING
    )
    clientset.model_instances.get.side_effect = NotFoundException("instance deleted")
    serve._model_instance_by_instance_id[instance.id] = instance
    serve._assigned_ports[instance.id] = {40000, 40001}
    serve._error_model_instances[instance.id] = instance
    serve._pending_model_instance_errors[instance.id] = {"state_message": "unreported"}
    event_type = EventType.CREATED if serve._worker_id == 1 else EventType.UPDATED

    with (
        patch("gpustack.worker.serve_manager.delete_workload") as delete,
        patch("gpustack.worker.serve_manager.multiprocessing.Process") as process,
        patch("gpustack.worker.serve_manager.get_workload", return_value=None),
        patch.object(serve, "_assign_ports", wraps=serve._assign_ports) as assign,
    ):
        process.return_value.is_alive.return_value = False
        for _ in range(3):
            serve._dispatch_model_instance_event(Event(type=event_type, data=instance))

    assign.assert_not_called()
    process.assert_not_called()
    serve._update_model_instance.assert_not_called()
    clientset.model_instances.get.assert_called_with(id=instance.id, use_cache=False)
    delete.assert_called_with(instance.get_deployment_metadata(serve._worker_id).name)
    assert instance.id not in serve._assigned_ports
    assert instance.id not in serve._model_instance_by_instance_id
    assert instance.id not in serve._error_model_instances
    assert instance.id not in serve._pending_model_instance_errors


@pytest.mark.parametrize("restart_on_error", [False, True])
@pytest.mark.parametrize("report_fails", [False, True])
def test_failed_existence_lookup_reports_error_and_retains_recovery(
    manager, restart_on_error, report_fails
):
    serve, clientset = manager
    instance = _instance()
    instance.state = (
        ModelInstanceStateEnum.SCHEDULED
        if serve._worker_id == instance.worker_id
        else ModelInstanceStateEnum.STARTING
    )
    clientset.models.get.return_value.restart_on_error = restart_on_error
    clientset.model_instances.get.side_effect = ReadTimeout("server unavailable")
    if report_fails:
        serve._update_model_instance.side_effect = ReadTimeout("update unavailable")
    event_type = EventType.CREATED if serve._worker_id == 1 else EventType.UPDATED
    with (
        patch("gpustack.worker.serve_manager.get_workload", return_value=None),
        patch("gpustack.worker.serve_manager.delete_workload") as delete,
        patch("gpustack.worker.serve_manager.multiprocessing.Process") as process,
    ):
        serve._handle_model_instance_event(Event(type=event_type, data=instance))
    delete.assert_not_called()
    process.assert_not_called()
    assert not serve._assigned_ports
    assert instance.id not in serve._starting_model_instances
    assert instance.id in serve._error_model_instances
    update = serve._update_model_instance.call_args.kwargs
    if serve._worker_id == instance.worker_id:
        assert update["state"] == ModelInstanceStateEnum.ERROR
        assert "server unavailable" in update["state_message"]
    else:
        worker = update["distributed_servers.subordinate_workers.0"]
        assert worker.state == ModelInstanceStateEnum.ERROR
        assert "server unavailable" in worker.state_message

    serve._update_model_instance.reset_mock()
    if report_fails:
        with pytest.raises(ReadTimeout, match="update unavailable"):
            serve._restart_error_model_instance(
                serve._error_model_instances[instance.id]
            )
        assert instance.id in serve._error_model_instances
    serve._update_model_instance.side_effect = None
    serve._update_model_instance.reset_mock()
    clientset.model_instances.get.side_effect = None
    live_instance = instance.model_copy(deep=True)
    if serve._worker_id == instance.worker_id:
        live_instance.state = ModelInstanceStateEnum.ERROR
    else:
        live_instance.distributed_servers.subordinate_workers[0] = update[
            "distributed_servers.subordinate_workers.0"
        ]
    clientset.model_instances.get.return_value = live_instance
    serve._restart_error_model_instance(serve._error_model_instances[instance.id])
    updates = [call.kwargs for call in serve._update_model_instance.call_args_list]
    assert len(updates) == int(report_fails) + int(restart_on_error)
    if restart_on_error:
        assert updates[-1]["state"] == ModelInstanceStateEnum.SCHEDULED
        assert updates[-1]["restart_count"] == 1
    assert instance.id not in serve._error_model_instances


@pytest.mark.parametrize("ports", [None, []])
def test_running_instance_without_ports_has_no_inference_endpoint(manager, ports):
    serve, _ = manager
    instance = _instance()
    instance.ports = ports
    serve._model_instance_by_instance_id[instance.id] = instance
    assert serve.get_instance_port_by_model_instance_id(instance.id) is None


def test_successful_launch_cancels_pending_failure_recovery(manager):
    serve, clientset = manager
    instance = _instance()
    clientset.model_instances.get.side_effect = ReadTimeout("server unavailable")
    serve._update_model_instance.side_effect = ReadTimeout("update unavailable")
    with pytest.raises(ReadTimeout, match="update unavailable"):
        serve._start_model_instance(instance)
    assert instance.id in serve._error_model_instances

    clientset.model_instances.get.side_effect = None
    serve._update_model_instance.side_effect = None
    serve._start_model_instance(instance)
    assert instance.id in serve._provisioning_processes
    assert instance.id not in serve._error_model_instances
    assert instance.id not in serve._pending_model_instance_errors


def test_releasing_an_old_claim_preserves_a_new_claim(manager):
    serve, _ = manager
    old, new = uuid4(), uuid4()
    serve._distributed_port_claims[1] = old
    coordinator = MagicMock()
    serve._distributed_port_client = coordinator

    def new_claim(instance_id, token):
        assert token == old
        serve._distributed_port_claims[instance_id] = new

    coordinator.release.side_effect = new_claim
    serve._release_distributed_ports(1)
    assert serve._distributed_port_claims[1] == new


def test_remote_release_failure_does_not_block_local_cleanup(manager):
    serve, _ = manager
    instance = _instance()
    token = uuid4()
    serve._distributed_port_claims[instance.id] = token
    serve._assigned_ports[instance.id] = {40000, 40001}
    serve._model_instance_by_instance_id[instance.id] = instance
    coordinator = MagicMock()
    coordinator.release.side_effect = RuntimeError("coordinator unavailable")
    serve._distributed_port_client = coordinator
    with patch("gpustack.worker.serve_manager.delete_workload"):
        serve._stop_model_instance(instance)
    assert instance.id not in serve._assigned_ports
    assert instance.id not in serve._model_instance_by_instance_id
    assert (instance.id, token) in serve._pending_distributed_port_releases
    coordinator.release.side_effect = None
    serve._retry_distributed_port_releases()
    assert not serve._pending_distributed_port_releases


def _prepare_local_pd_allocation(manager):
    serve, clientset = manager
    serve._config.service_port_range = "40000-40004"
    instance = _pd_instance(7)
    instance.worker_id = serve._worker_id
    model = _pd_model()
    clientset.model_instances.get.return_value = instance
    clientset.model_instances.list.return_value = SimpleNamespace(items=[instance])
    serve._get_model.return_value = model
    return instance, model


@pytest.mark.parametrize("source", ["model", "cache"])
def test_local_pd_allocation_waits_for_deleted_owner_cleanup(manager, source):
    serve, _ = manager
    instance, model = _prepare_local_pd_allocation(manager)
    if source == "model":
        serve._port_reservations.model_ports[99] = {40002}
    else:
        serve._port_reservations.cache_ports[99] = (40002,)

    def finish_cleanup(delay):
        assert delay == 2
        assert instance.port is None
        assert instance.id not in serve._assigned_ports
        with serve._port_reservations.allocation(
            serve._clientset, serve._worker_id, model_instance_id=instance.id
        ) as taken:
            assert 40002 in taken
        if source == "model":
            serve._port_reservations.release_model(99)
        else:
            serve._port_reservations.release_cache(99)

    with (
        patch(
            "gpustack.worker.pd_injection.get_pd_mode", return_value=_wide_band_mode(4)
        ),
        patch(
            "gpustack.worker.serve_manager.time.sleep", side_effect=finish_cleanup
        ) as sleep,
    ):
        serve._assign_ports(instance, model, BackendEnum.VLLM)

    sleep.assert_called_once_with(2)
    assert instance.named_ports["kv_port"].count == 4
    assert len(serve._assigned_ports[instance.id]) == 5


def test_local_pd_exhaustion_without_cleanup_does_not_wait(manager):
    serve, clientset = manager
    instance, model = _prepare_local_pd_allocation(manager)
    cache = _cache_instance(99, serve._worker_id, port=40002)
    clientset.cache_service_instances.list.return_value = SimpleNamespace(items=[cache])
    with (
        patch(
            "gpustack.worker.pd_injection.get_pd_mode", return_value=_wide_band_mode(4)
        ),
        patch("gpustack.worker.serve_manager.time.sleep") as sleep,
    ):
        with pytest.raises(network.PortRangeExhaustedError, match="No band of 4"):
            serve._assign_ports(instance, model, BackendEnum.VLLM)

    sleep.assert_not_called()
    assert instance.port is None
    assert instance.id not in serve._assigned_ports


def test_instance_deleted_during_port_cleanup_wait_cannot_start(manager):
    serve, clientset = manager
    instance, _ = _prepare_local_pd_allocation(manager)
    serve._port_reservations.cache_ports[99] = (40002,)

    def delete_instance(delay):
        assert delay == 2
        serve._port_reservations.release_cache(99)
        clientset.model_instances.get.side_effect = NotFoundException(
            "instance deleted"
        )

    with (
        patch(
            "gpustack.worker.pd_injection.get_pd_mode", return_value=_wide_band_mode(4)
        ),
        patch(
            "gpustack.worker.serve_manager.time.sleep", side_effect=delete_instance
        ) as sleep,
        patch("gpustack.worker.serve_manager.multiprocessing.Process") as process,
    ):
        serve._start_model_instance(instance)

    sleep.assert_called_once_with(2)
    process.assert_not_called()
    assert instance.port is None
    assert not instance.ports
    assert instance.id not in serve._assigned_ports
    assert instance.id not in serve._starting_model_instances


@pytest.mark.asyncio
async def test_launch_callback_keeps_worker_api_event_loop_available(manager):
    serve, _ = manager
    loop = asyncio.get_running_loop()
    calls = []

    async def worker_api():
        calls.append("worker API responded")

    def launch(event):
        asyncio.run_coroutine_threadsafe(worker_api(), loop).result(timeout=2)

    with patch.object(serve, "_handle_model_instance_event", side_effect=launch):
        await asyncio.wait_for(
            serve._handle_model_instance_event_async(
                Event(type=EventType.CREATED, data=_instance())
            ),
            timeout=3,
        )
    assert calls == ["worker API responded"]


def test_failed_launch_releases_ports_before_provisioning(manager):
    serve, _ = manager
    instance = _instance()
    serve._inference_backend_manager.get_backend_by_name.side_effect = RuntimeError(
        "backend unavailable"
    )
    with patch("gpustack.worker.serve_manager.delete_workload"):
        serve._start_model_instance(instance)
    assert instance.id not in serve._assigned_ports


def test_failed_launch_reports_error_even_when_cleanup_fails(manager):
    serve, _ = manager
    instance = _instance()
    serve._inference_backend_manager.get_backend_by_name.side_effect = RuntimeError(
        "backend unavailable"
    )
    with patch(
        "gpustack.worker.serve_manager.delete_workload",
        side_effect=RuntimeError("runtime unavailable"),
    ):
        serve._start_model_instance(instance)
    assert len(serve._assigned_ports[instance.id]) == 14
    update = serve._update_model_instance.call_args.kwargs
    if serve._worker_id == 1:
        assert update["state"] == ModelInstanceStateEnum.ERROR
        assert "backend unavailable" in update["state_message"]
    else:
        worker = update["distributed_servers.subordinate_workers.0"]
        assert worker.state == ModelInstanceStateEnum.ERROR
        assert "backend unavailable" in worker.state_message


def test_cleanup_does_not_release_ports_on_authoritative_read_failure(manager):
    serve, clientset = manager
    instance = _instance()
    serve._start_model_instance(instance)
    clientset.model_instances.list.side_effect = [
        SimpleNamespace(items=[]),
        RuntimeError("server unavailable"),
    ]
    with patch("gpustack.worker.serve_manager.delete_workload") as delete:
        with pytest.raises(RuntimeError, match="server unavailable"):
            serve.sync_model_instances_state()
    delete.assert_not_called()
    assert len(serve._assigned_ports[instance.id]) == 14


def test_reconnect_snapshot_cannot_reap_an_instance_still_assigned(manager):
    serve, clientset = manager
    instance = _instance()
    serve._start_model_instance(instance)
    clientset.model_instances.list.side_effect = [
        SimpleNamespace(items=[]),
        SimpleNamespace(items=[instance]),
    ]
    with (
        patch("gpustack.worker.serve_manager.delete_workload") as delete,
        patch.object(serve, "_is_provisioning", return_value=True),
    ):
        serve.sync_model_instances_state()
    delete.assert_not_called()
    assert len(serve._assigned_ports[instance.id]) == 14


def test_recreation_can_retry_after_old_instance_finishes_cleanup(manager):
    serve, clientset = manager
    old, replacement = _instance(1), _instance(2)
    serve._start_model_instance(old)
    serve._start_model_instance(replacement)
    assert replacement.id not in serve._assigned_ports
    clientset.model_instances.list.return_value = SimpleNamespace(items=[replacement])

    with (
        patch("gpustack.worker.serve_manager.delete_workload") as delete,
        patch.object(serve, "_is_provisioning", return_value=True),
    ):
        serve.sync_model_instances_state()
    delete.assert_called_once_with(old.get_deployment_metadata(serve._worker_id).name)
    assert old.id not in serve._assigned_ports

    serve._start_model_instance(replacement)
    assert len(serve._assigned_ports[replacement.id]) == 14


@pytest.mark.parametrize(
    "state", [ModelInstanceStateEnum.DOWNLOADING, ModelInstanceStateEnum.RUNNING]
)
def test_observed_local_instance_is_tracked_before_start_readiness(manager, state):
    serve, _ = manager
    instance = _instance()
    instance.state = state
    with patch.object(serve, "_start_model_instance") as start:
        serve._dispatch_model_instance_event(
            Event(type=EventType.UPDATED, data=instance)
        )
    start.assert_not_called()
    assert instance.id in serve._model_instance_by_instance_id


def test_followers_are_not_used_as_inference_entrypoints(manager):
    serve, _ = manager
    instance = _instance()
    serve._model_instance_by_instance_id[instance.id] = instance
    serve._get_model.return_value.env = {
        "GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_ENABLED": "true"
    }
    with patch.object(serve, "_probe_running_instance", return_value=True) as probe:
        serve.sync_model_instances_inference_health()
    if serve._worker_id == instance.worker_id:
        probe.assert_called_once()
        assert (
            serve.get_instance_port_by_model_instance_id(instance.id) == instance.port
        )
    else:
        probe.assert_not_called()
        assert serve.get_instance_port_by_model_instance_id(instance.id) is None


@pytest.mark.parametrize("placement", ["unrelated", "delegated"])
@pytest.mark.parametrize("manager", [2], indirect=True)
def test_workers_without_local_workload_do_not_track_instance(manager, placement):
    serve, _ = manager
    instance = _instance()
    if placement == "unrelated":
        instance.distributed_servers.subordinate_workers[0].worker_id = 3
    else:
        instance.distributed_servers.mode = (
            DistributedServerCoordinateModeEnum.DELEGATED
        )
    serve._dispatch_model_instance_event(Event(type=EventType.UPDATED, data=instance))
    assert not serve._model_instance_by_instance_id


@pytest.mark.parametrize("manager", [2], indirect=True)
@pytest.mark.parametrize("restart_on_error", [False, True])
@pytest.mark.parametrize(
    "local_state",
    [ModelInstanceStateEnum.ERROR, ModelInstanceStateEnum.RUNNING],
)
@pytest.mark.parametrize(
    "workload_state",
    [
        None,
        WorkloadStatusStateEnum.FAILED,
        WorkloadStatusStateEnum.UNHEALTHY,
        WorkloadStatusStateEnum.INACTIVE,
        WorkloadStatusStateEnum.UNKNOWN,
    ],
)
def test_follower_retries_respect_restart_on_error(
    manager, restart_on_error, local_state, workload_state
):
    serve, _ = manager
    instance = _instance()
    instance.state = ModelInstanceStateEnum.STARTING
    instance.distributed_servers.subordinate_workers[0].state = local_state
    serve._get_model.return_value.restart_on_error = restart_on_error
    workload = (
        SimpleNamespace(state=workload_state) if workload_state is not None else None
    )
    with (
        patch("gpustack.worker.serve_manager.get_workload", return_value=workload),
        patch.object(serve, "_start_model_instance") as start,
        patch.object(serve, "_stop_model_instance") as stop,
    ):
        for _ in range(3):
            serve._dispatch_model_instance_event(
                Event(type=EventType.UPDATED, data=instance)
            )
    assert start.call_count == (3 if restart_on_error else 0)
    assert stop.call_count == (3 if restart_on_error and workload is not None else 0)


@pytest.mark.parametrize("manager", [2], indirect=True)
@pytest.mark.parametrize(
    "local_state",
    [
        ModelInstanceStateEnum.PENDING,
        ModelInstanceStateEnum.SCHEDULED,
        ModelInstanceStateEnum.DOWNLOADING,
    ],
)
def test_follower_initial_start_is_allowed_without_auto_restart(manager, local_state):
    serve, _ = manager
    instance = _instance()
    instance.state = ModelInstanceStateEnum.STARTING
    instance.distributed_servers.subordinate_workers[0].state = local_state
    serve._get_model.return_value.restart_on_error = False
    with (
        patch("gpustack.worker.serve_manager.get_workload", return_value=None),
        patch.object(serve, "_start_model_instance") as start,
    ):
        serve._dispatch_model_instance_event(
            Event(type=EventType.UPDATED, data=instance)
        )
    start.assert_called_once()


@pytest.mark.parametrize("manager", [2], indirect=True)
def test_port_allocation_failure_is_not_retried_when_auto_restart_is_disabled(manager):
    serve, _ = manager
    instance = _instance()
    instance.state = ModelInstanceStateEnum.STARTING
    serve._get_model.return_value.restart_on_error = False
    serve._port_reservations.model_ports[99] = {instance.port}

    def publish(instance_id, **kwargs):
        instance.distributed_servers.subordinate_workers[0] = kwargs[
            "distributed_servers.subordinate_workers.0"
        ]

    serve._update_model_instance.side_effect = publish
    with (
        patch("gpustack.worker.serve_manager.get_workload", return_value=None),
        patch.object(serve, "_assign_ports", wraps=serve._assign_ports) as assign,
    ):
        for _ in range(3):
            serve._dispatch_model_instance_event(
                Event(type=EventType.UPDATED, data=instance)
            )
    assert assign.call_count == 1
    worker = instance.distributed_servers.subordinate_workers[0]
    assert worker.state == ModelInstanceStateEnum.ERROR
    assert "reserved by another instance" in worker.state_message


@pytest.mark.parametrize("restart_on_error", [False, True])
def test_policy_read_failure_keeps_error_instance_until_policy_is_known(
    manager, restart_on_error
):
    serve, clientset = manager
    instance = _instance()
    instance.state = ModelInstanceStateEnum.ERROR
    clientset.model_instances.get.return_value = instance
    clientset.models.get.side_effect = ReadTimeout("policy unavailable")
    serve._handle_model_instance_event(Event(type=EventType.UPDATED, data=instance))
    assert instance.id in serve._error_model_instances
    with pytest.raises(ReadTimeout, match="policy unavailable"):
        serve._restart_error_model_instance(serve._error_model_instances[instance.id])
    assert instance.id in serve._error_model_instances
    serve._update_model_instance.assert_not_called()

    clientset.models.get.side_effect = None
    clientset.models.get.return_value.restart_on_error = restart_on_error
    serve._restart_error_model_instance(serve._error_model_instances[instance.id])
    assert serve._update_model_instance.call_count == int(restart_on_error)
    if restart_on_error:
        assert serve._update_model_instance.call_args.kwargs["state"] == (
            ModelInstanceStateEnum.SCHEDULED
        )
    assert instance.id not in serve._error_model_instances


def test_disabling_auto_restart_cancels_queued_recovery(manager):
    serve, clientset = manager
    instance = _instance()
    instance.state = ModelInstanceStateEnum.ERROR
    clientset.model_instances.get.return_value = instance
    serve._error_model_instances[instance.id] = instance
    clientset.models.get.return_value = serve._get_model.return_value.model_copy(
        update={"restart_on_error": False}
    )

    serve._restart_error_model_instance(instance)

    serve._update_model_instance.assert_not_called()
    assert instance.id not in serve._error_model_instances


def test_missing_parent_model_is_reported_as_launch_failure(manager):
    serve, clientset = manager
    instance = _instance()
    instance.group_id = "1-old"
    clientset.model_instances.get.return_value = instance
    clientset.models.get.side_effect = NotFoundException("Parent model not found")

    with (
        patch.object(serve, "_stop_model_instance") as stop,
        patch("gpustack.worker.serve_manager.multiprocessing.Process") as process,
    ):
        serve._start_model_instance(instance)

    stop.assert_not_called()
    process.assert_not_called()
    assert instance.id in serve._error_model_instances
    assert serve._update_model_instance.call_count == 1


@pytest.mark.parametrize("role", ["prefill", "decode", "router"])
def test_pending_configuration_blocks_member_recovery(manager, role):
    serve, clientset = manager
    instance = _instance()
    instance.role = role
    instance.group_id = "1-old"
    instance.spec_digest = "old"
    instance.state = ModelInstanceStateEnum.ERROR
    clientset.model_instances.get.return_value = instance
    clientset.models.get.return_value = serve._get_model.return_value.model_copy(
        update={"stale": True, "restart_on_error": True}
    )
    serve._dispatch_model_instance_event(Event(type=EventType.UPDATED, data=instance))

    serve._restart_error_model_instance(serve._error_model_instances[instance.id])

    serve._update_model_instance.assert_not_called()
    clientset.model_instances.update.assert_not_called()
    assert instance.id not in serve._restart_backoff_counts
    clientset.models.get.assert_called_with(instance.model_id, use_cache=False)


@pytest.mark.parametrize("manager", [2], indirect=True)
@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_pending_configuration_blocks_follower_recovery(manager, role):
    serve, clientset = manager
    instance = _instance()
    instance.role = role
    instance.group_id = "1-old"
    instance.distributed_servers.subordinate_workers[0].state = (
        ModelInstanceStateEnum.ERROR
    )
    clientset.model_instances.get.return_value = instance
    clientset.models.get.return_value.stale = True
    with (
        patch("gpustack.worker.serve_manager.get_workload", return_value=None),
        patch.object(serve, "_start_model_instance") as start,
    ):
        serve._start_or_restart_subordinate(instance)

    start.assert_not_called()


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_edit_during_backoff_cancels_recovery(manager, role):
    serve, clientset = manager
    instance = _instance()
    instance.role = role
    instance.group_id = "1-old"
    instance.state = ModelInstanceStateEnum.ERROR
    instance.last_restart_time = datetime.now(timezone.utc)
    clientset.model_instances.get.return_value = instance
    serve._restart_backoff_counts[instance.id] = 1
    serve._error_model_instances[instance.id] = instance

    serve._restart_error_model_instance(instance)
    serve._update_model_instance.assert_not_called()
    assert instance.id in serve._error_model_instances

    clientset.models.get.return_value.stale = True
    instance.last_restart_time -= timedelta(minutes=1)
    serve._restart_error_model_instance(instance)

    serve._update_model_instance.assert_not_called()
    clientset.model_instances.update.assert_not_called()
    assert serve._restart_backoff_counts[instance.id] == 1


@pytest.mark.parametrize("group_id", [None, "1-old"])
@pytest.mark.parametrize("backoff_count", [1, 6])
def test_backoff_does_not_read_recovery_policy(manager, group_id, backoff_count):
    serve, clientset = manager
    instance = _instance()
    instance.group_id = group_id
    instance.state = ModelInstanceStateEnum.ERROR
    instance.last_restart_time = datetime.now(timezone.utc)
    clientset.model_instances.get.return_value = instance
    serve._restart_backoff_counts[instance.id] = backoff_count
    serve._error_model_instances[instance.id] = instance

    for _ in range(3):
        serve._restart_error_model_instance(instance)

    clientset.models.get.assert_not_called()
    clientset.model_instances.get.assert_not_called()
    serve._update_model_instance.assert_not_called()
    assert instance.id in serve._error_model_instances


def test_due_pd_recovery_checks_server_even_when_cached_config_is_current(manager):
    serve, clientset = manager
    instance = _instance()
    instance.group_id = "1-old"
    instance.state = ModelInstanceStateEnum.ERROR
    clientset.model_instances.get.return_value = instance
    cached = clientset.models.get.return_value.model_copy(update={"stale": False})
    current = cached.model_copy(update={"stale": True})
    clientset.models.get.side_effect = lambda id, use_cache=True: (
        cached if use_cache else current
    )

    serve._restart_error_model_instance(instance)

    clientset.models.get.assert_called_once_with(instance.model_id, use_cache=False)
    serve._update_model_instance.assert_not_called()
    clientset.model_instances.update.assert_not_called()


@pytest.mark.parametrize("role", ["prefill", "decode", "router"])
def test_edit_after_recovery_decision_blocks_actual_launch(manager, role):
    serve, clientset = manager
    instance = _instance()
    instance.role = role
    instance.group_id = "1-old"
    instance.state = ModelInstanceStateEnum.ERROR
    assert serve._auto_restart_enabled(instance)
    instance.state = ModelInstanceStateEnum.SCHEDULED
    clientset.models.get.return_value = serve._get_model.return_value.model_copy(
        update={"stale": True}
    )

    with (
        patch("gpustack.worker.serve_manager.multiprocessing.Process") as process,
        patch.object(serve, "_assign_ports") as assign,
    ):
        serve._start_model_instance(instance)

    process.assert_not_called()
    assign.assert_not_called()
    update = serve._update_model_instance.call_args.kwargs
    if serve._worker_id == instance.worker_id:
        assert update["state"] == ModelInstanceStateEnum.ERROR
    else:
        subordinate = update["distributed_servers.subordinate_workers.0"]
        assert subordinate.state == ModelInstanceStateEnum.ERROR
    clientset.models.get.assert_called_with(instance.model_id, use_cache=False)


def test_group_launch_waits_when_current_configuration_is_unknown(manager):
    serve, clientset = manager
    instance = _instance()
    instance.group_id = "1-old"
    clientset.models.get.side_effect = ReadTimeout("configuration unavailable")

    with patch("gpustack.worker.serve_manager.multiprocessing.Process") as process:
        serve._start_model_instance(instance)

    process.assert_not_called()
    assert instance.id in serve._error_model_instances


def test_rebuilt_group_launch_uses_fresh_staleness(manager):
    serve, clientset = manager
    instance = _instance()
    instance.role = "prefill"
    instance.group_id = "1-current"
    instance.state = ModelInstanceStateEnum.SCHEDULED
    serve._get_model.return_value.stale = True
    clientset.models.get.return_value = serve._get_model.return_value.model_copy(
        update={"stale": False}
    )

    with patch("gpustack.worker.serve_manager.multiprocessing.Process") as process:
        serve._start_model_instance(instance)

    process.return_value.start.assert_called_once()
    clientset.models.get.assert_called_with(instance.model_id, use_cache=False)


def test_pending_configuration_keeps_healthy_members_running(manager):
    serve, clientset = manager
    instance = _instance()
    instance.role = "prefill"
    instance.group_id = "1-old"
    clientset.models.get.return_value.stale = True

    with patch.object(serve, "_stop_model_instance") as stop:
        serve._dispatch_model_instance_event(
            Event(type=EventType.UPDATED, data=instance)
        )

    stop.assert_not_called()
    serve._update_model_instance.assert_not_called()


@pytest.mark.parametrize(
    "group_id, stale", [(None, True), ("1-current", False), ("1-legacy", None)]
)
def test_plain_and_current_members_still_recover(manager, group_id, stale):
    serve, clientset = manager
    instance = _instance()
    instance.role = "prefill" if group_id else None
    instance.group_id = group_id
    instance.state = ModelInstanceStateEnum.ERROR
    clientset.model_instances.get.return_value = instance
    clientset.models.get.return_value.stale = stale
    serve._error_model_instances[instance.id] = instance

    serve._restart_error_model_instance(instance)

    assert serve._update_model_instance.call_args.kwargs["state"] == (
        ModelInstanceStateEnum.SCHEDULED
    )
    clientset.models.get.assert_called_with(
        instance.model_id, use_cache=group_id is None
    )


@pytest.mark.parametrize("manager", [2], indirect=True)
def test_follower_recovery_uses_current_restart_policy(manager):
    serve, clientset = manager
    instance = _instance()
    instance.state = ModelInstanceStateEnum.STARTING
    instance.distributed_servers.subordinate_workers[0].state = (
        ModelInstanceStateEnum.ERROR
    )
    clientset.models.get.return_value = serve._get_model.return_value.model_copy(
        update={"restart_on_error": False}
    )
    with (
        patch("gpustack.worker.serve_manager.get_workload", return_value=None),
        patch.object(serve, "_start_model_instance") as start,
    ):
        serve._dispatch_model_instance_event(
            Event(type=EventType.UPDATED, data=instance)
        )
    start.assert_not_called()


@pytest.mark.parametrize("manager", [1], indirect=True)
@pytest.mark.parametrize(
    "local_state, restart_on_error, should_update",
    [
        (ModelInstanceStateEnum.ERROR, False, True),
        (ModelInstanceStateEnum.ERROR, True, False),
        (ModelInstanceStateEnum.PENDING, False, False),
    ],
)
def test_starting_main_observes_follower_recovery_policy(
    manager, local_state, restart_on_error, should_update
):
    serve, clientset = manager
    instance = _instance()
    instance.state = ModelInstanceStateEnum.STARTING
    worker = instance.distributed_servers.subordinate_workers[0]
    worker.state = local_state
    worker.state_message = (
        "Model instance ports [40000] are reserved by another instance"
    )
    serve._get_model.return_value.restart_on_error = restart_on_error
    clientset.model_instances.list.return_value = SimpleNamespace(items=[instance])
    with patch(
        "gpustack.worker.serve_manager.get_workload",
        return_value=SimpleNamespace(state=WorkloadStatusStateEnum.INITIALIZING),
    ):
        serve.sync_model_instances_state()
    if local_state == ModelInstanceStateEnum.ERROR:
        clientset.models.get.assert_called_with(instance.model_id, use_cache=True)
    if not should_update:
        serve._update_model_instance.assert_not_called()
        return
    serve._update_model_instance.assert_called_once()
    update = serve._update_model_instance.call_args.kwargs
    assert update["state"] == ModelInstanceStateEnum.ERROR
    assert "40000" in update["state_message"]


def test_deleted_model_cancels_queued_recovery(manager):
    serve, clientset = manager
    instance = _instance()
    instance.state = ModelInstanceStateEnum.ERROR
    clientset.model_instances.get.return_value = instance
    serve._error_model_instances[instance.id] = instance
    clientset.models.get.side_effect = NotFoundException("Model deleted")

    serve._restart_error_model_instance(instance)

    serve._update_model_instance.assert_not_called()
    assert instance.id not in serve._error_model_instances


@pytest.mark.asyncio
async def test_watch_waits_for_launch_callback_before_dispatching_next_event(manager):
    import json

    import httpx

    from gpustack.client.generated_model_instance_client import ModelInstanceClient

    serve, _ = manager
    instance = _instance()
    events = [EventType.CREATED, EventType.UPDATED]
    body = "\n".join(
        json.dumps({"type": kind.value, "data": instance.model_dump(mode="json")})
        for kind in events
    )
    started, release = asyncio.Event(), threading.Event()
    observed = []
    loop = asyncio.get_running_loop()

    def handle(event):
        observed.append(event.type)
        if event.type == EventType.CREATED:
            loop.call_soon_threadsafe(started.set)
            assert release.wait(timeout=2)

    with patch.object(serve, "_handle_model_instance_event", side_effect=handle):
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, text=body)
            ),
            base_url="http://server",
        ) as http:
            client = ModelInstanceClient(
                SimpleNamespace(get_async_httpx_client=lambda: http), enable_cache=False
            )
            task = asyncio.create_task(
                client.awatch(
                    callback=serve._handle_model_instance_event_async,
                    stop_condition=lambda event: event.type == EventType.UPDATED,
                )
            )
            try:
                await asyncio.wait_for(started.wait(), timeout=1)
                await asyncio.sleep(0)
                assert observed == [EventType.CREATED]
            finally:
                release.set()
                await asyncio.wait_for(task, timeout=3)
    assert observed == events


@pytest.mark.parametrize("edit_stage", [None, "manager", "backend"])
def test_staggered_group_startup_allows_version_writeback_but_blocks_edits(
    manager, monkeypatch, edit_stage
):
    serve, clientset = manager
    model = serve._get_model.return_value
    model.backend_version = None
    model.roles = [RoleSpec(name="prefill"), RoleSpec(name="decode")]
    monkeypatch.setattr(
        "gpustack.server.controllers.GPUInstanceType.all_by_fields",
        AsyncMock(return_value=[]),
    )
    session = MagicMock()
    session.info = {}
    digest = asyncio.run(model_spec_digest(session, model))
    waiting = _instance()
    waiting.group_id = "1-group"
    waiting.spec_digest = digest
    waiting.role = "decode"
    waiting.state = ModelInstanceStateEnum.SCHEDULED
    first = waiting.model_copy(
        update={
            "id": 2,
            "role": "prefill",
            "state": ModelInstanceStateEnum.RUNNING,
            "backend_version": "0.31.0",
        }
    )
    model.backend_version = "0.31.0"
    clientset.model_instances.get.return_value = waiting

    def read_model(*args, **kwargs):
        session.info.clear()
        model.stale = asyncio.run(model_members_stale(session, model, [first, waiting]))
        return model.model_copy(deep=True)

    clientset.models.get.side_effect = read_model
    if edit_stage == "manager":
        model.roles[0].backend_parameters = ["--max-model-len=8192"]

    with patch("gpustack.worker.serve_manager.multiprocessing.Process") as process:
        serve._start_model_instance(waiting)
        assert process.return_value.start.call_count == int(edit_stage != "manager")

    if edit_stage == "backend":
        model.roles[0].backend_parameters = ["--max-model-len=8192"]
    server = SimpleNamespace(
        _clientset=clientset,
        _model_instance=waiting,
        _config=SimpleNamespace(data_dir="/tmp"),
    )
    if edit_stage is None:
        InferenceServer.get_model(server)
        assert server._model.backend_version == "0.31.0"
        assert waiting.backend_version is None
    else:
        with pytest.raises(RuntimeError):
            InferenceServer.get_model(server)


@pytest.mark.parametrize("manager", [1], indirect=True)
@pytest.mark.parametrize(
    "waiting_state",
    [
        ModelInstanceStateEnum.SCHEDULED,
        ModelInstanceStateEnum.ERROR,
    ],
)
def test_group_can_launch_and_recover_between_actual_version_writeback_requests(
    manager, monkeypatch, waiting_state
):
    serve, clientset = manager
    model = serve._get_model.return_value
    model.backend_version = None
    model.restart_on_error = True
    model.roles = [RoleSpec(name="prefill"), RoleSpec(name="decode")]
    monkeypatch.setattr(
        "gpustack.server.controllers.GPUInstanceType.all_by_fields",
        AsyncMock(return_value=[]),
    )
    session = MagicMock()
    session.info = {}
    digest = asyncio.run(model_spec_digest(session, model))
    waiting = _instance()
    waiting.group_id = "1-group"
    waiting.spec_digest = digest
    waiting.role = "decode"
    waiting.state = waiting_state
    first = waiting.model_copy(
        deep=True,
        update={"id": 2, "role": "prefill", "state": ModelInstanceStateEnum.STARTING},
    )
    instances = {first.id: first, waiting.id: waiting}
    clientset.model_instances.get.side_effect = lambda id, **kwargs: instances[
        id
    ].model_copy(deep=True)

    def read_model(*args, **kwargs):
        session.info.clear()
        model.stale = asyncio.run(model_members_stale(session, model, [first, waiting]))
        return model.model_copy(deep=True)

    clientset.models.get.side_effect = read_model
    observations = []
    writes = []

    def persist_instance(*, id, model_update):
        writes.append(("instance", id))
        target = instances[id]
        for field in model_update.model_fields_set:
            setattr(target, field, getattr(model_update, field))

    clientset.model_instances.update.side_effect = persist_instance

    def persist_manager_update(id, **fields):
        for field, value in fields.items():
            setattr(instances[id], field, value)

    serve._update_model_instance.side_effect = persist_manager_update
    backend = SimpleNamespace(
        _clientset=clientset,
        _model_instance=waiting,
        _config=SimpleNamespace(data_dir="/tmp"),
    )

    def persist_model(id, model_update):
        writes.append(("model", id))
        model.backend_version = model_update.backend_version
        observations.append((first.backend_version, read_model().stale))
        if waiting_state == ModelInstanceStateEnum.ERROR:
            serve._error_model_instances[waiting.id] = waiting.model_copy(deep=True)
            serve._restart_error_model_instance(waiting.model_copy(deep=True))
        serve._start_model_instance(waiting.model_copy(deep=True))
        try:
            InferenceServer.get_model(backend)
            observations.append("backend-started")
        except RuntimeError:
            observations.append("backend-blocked")

    clientset.models.update.side_effect = persist_model
    writer = SimpleNamespace(
        _clientset=clientset,
        _model=model.model_copy(deep=True),
        _model_spec=model.model_copy(deep=True),
        _model_instance=first.model_copy(deep=True),
    )
    writer._update_model_instance = lambda id, **fields: (
        InferenceServer._update_model_instance(writer, id, **fields)
    )
    with patch.object(serve, "_launch_model_instance") as launch:
        InferenceServer._update_model_backend_service_version(writer, "0.31.0")

    assert writes == [("model", model.id), ("instance", first.id)]
    assert observations == [(None, False), "backend-started"]
    assert first.backend_version == "0.31.0"
    assert read_model().stale is False
    launch.assert_called_once()
    if waiting_state == ModelInstanceStateEnum.ERROR:
        assert waiting.state == ModelInstanceStateEnum.SCHEDULED
        assert waiting.restart_count == 1
        assert waiting.id not in serve._error_model_instances
