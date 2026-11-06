from types import SimpleNamespace
from datetime import datetime, timedelta, timezone
import asyncio
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ReadTimeout

from gpustack.api.exceptions import (
    ConflictException,
    ForbiddenException,
    HTTPException,
    NotFoundException,
)
from gpustack.schemas.models import (
    BackendEnum,
    DistributedServerCoordinateModeEnum,
    DistributedServers,
    ModelInstanceStateEnum,
    ModelInstanceSubordinateWorker,
    ModelInstanceWorkloadUpdate,
    ModelInstanceWorkload,
    ModelInstanceWorkloadStateEnum,
    PortBand,
    RoleSpec,
    SourceEnum,
)
from gpustack.server.bus import Event, EventType
from gpustack.schemas.workers import WorkerStateEnum
from gpustack.server.controllers import model_members_stale, model_spec_digest
from gpustack.utils import network
from gpustack.worker.serve_manager import ServeManager
from gpustack.worker.backends.base import InferenceServer
from gpustack_runtime.deployer import Container, WorkloadPlan, WorkloadStatusStateEnum
from tests.utils.model import new_model, new_model_instance, new_model_instance_workload
from tests.worker.test_port_reservations import _cache_instance
from tests.worker.test_serve_manager_ports import (
    _instance as _pd_instance,
    _pd_model,
    _wide_band_mode,
)


@pytest.fixture
def managed(manager, monkeypatch):
    serve, client = manager
    instance = _instance()
    instance.cluster_id = 1
    instance.restart_count = 3
    instance.state = ModelInstanceStateEnum.STARTING
    instance.gpu_indexes = [0, 1]
    instance.distributed_servers.subordinate_workers[0].gpu_indexes = [0, 1]
    items = []
    for worker_id in (1, 2):
        for rank in range(2):
            item = new_model_instance_workload(
                worker_id=worker_id,
                dp_rank=rank,
                node_rank=worker_id - 1,
                name=f"distributed-1-r3-w{worker_id}-{rank}",
            )
            item["dp_rank_placements"][0]["gpu_indexes"] = [rank]
            if worker_id == 1:
                item["dp_rank_placements"][0]["api_port"] = 40020 + rank
            items.append(ModelInstanceWorkload.model_validate(item))
    instance.distributed_servers.workloads = items
    instance.distributed_servers.orchestration_mode = (
        "managed" if instance.distributed_servers.workloads is not None else "legacy"
    )
    state = SimpleNamespace(
        serve=serve, client=client, instance=instance, runtime=[], events=[]
    )
    client.model_instances.get.side_effect = lambda **kwargs: state.instance.model_copy(
        deep=True
    )
    client.model_instances.list.side_effect = lambda **kwargs: SimpleNamespace(
        items=[state.instance.model_copy(deep=True)]
    )
    client.workers.get.return_value = SimpleNamespace(
        state=WorkerStateEnum.READY, unreachable=False
    )

    def report(*, id, report):
        model_update = report
        assert isinstance(model_update, ModelInstanceWorkloadUpdate)
        assert model_update.restart_count == state.instance.restart_count
        from gpustack.routes.model_instances import _apply_workload_observations
        from gpustack.utils.model_instance_workers import get_model_instance_workloads

        local = get_model_instance_workloads(state.instance, serve._worker_id)
        if model_update.workloads is not None:
            local = [
                ModelInstanceWorkload.model_validate(w.model_dump())
                for w in model_update.workloads
            ]
        elif model_update.observations is not None:
            local = _apply_workload_observations(local, model_update.observations)
        assert all(item.worker_id == serve._worker_id for item in local)
        state.events.append("report")
        state.instance.distributed_servers.workloads = [
            item
            for item in state.instance.distributed_servers.workloads
            if item.worker_id != serve._worker_id
        ] + [item.model_copy(deep=True) for item in local]
        state.instance.distributed_servers.orchestration_mode = (
            "managed"
            if state.instance.distributed_servers.workloads is not None
            else "legacy"
        )
        return state.instance.model_copy(deep=True)

    client.model_instances.report_workloads.side_effect = report

    def create(plan):
        state.events.append(f"create:{plan.name}")
        state.runtime.append(
            SimpleNamespace(
                name=plan.name,
                labels=plan.labels,
                state=WorkloadStatusStateEnum.RUNNING,
            )
        )

    def delete(name):
        state.events.append(f"delete:{name}")
        state.runtime[:] = [item for item in state.runtime if item.name != name]

    state.create = MagicMock(side_effect=create)
    state.delete = MagicMock(side_effect=delete)
    state.list = MagicMock(side_effect=lambda: list(state.runtime))
    state.get = MagicMock(
        side_effect=lambda name: next(
            (item for item in state.runtime if item.name == name), None
        )
    )
    monkeypatch.setattr("gpustack.worker.serve_manager.create_workload", state.create)
    monkeypatch.setattr("gpustack.worker.serve_manager.delete_workload", state.delete)
    monkeypatch.setattr("gpustack.worker.serve_manager.list_workloads", state.list)
    monkeypatch.setattr("gpustack.worker.serve_manager.get_workload", state.get)
    return state


def _local_manifest(state):
    return [
        item.model_copy(deep=True)
        for item in state.instance.distributed_servers.workloads
        if item.worker_id == state.serve._worker_id
    ]


def _managed_plans(items):
    return [
        WorkloadPlan(
            name=item.name, containers=[Container(name="default", image="test:v1")]
        )
        for item in items
    ]


def _finish_managed_tasks(state):
    with state.serve._model_instance_start_lock:
        jobs = list(state.serve._managed_tasks.values()) + list(
            state.serve._managed_stop_tasks.values()
        )
    for job in jobs:
        job.result(timeout=2)


def _running_managed(state):
    for item in _local_manifest(state):
        state.runtime.append(
            SimpleNamespace(
                name=item.name,
                labels={
                    "gpustack.ai/model-instance-id": str(state.instance.id),
                    "gpustack.ai/restart-count": str(state.instance.restart_count),
                    "gpustack.ai/worker-id": str(state.serve._worker_id),
                },
                state=WorkloadStatusStateEnum.RUNNING,
            )
        )
    state.serve._model_instance_by_instance_id[state.instance.id] = (
        state.instance.model_copy(deep=True)
    )
    state.serve._assigned_ports[state.instance.id] = {40020, 40021}
    return state.instance.model_copy(deep=True)


@pytest.mark.parametrize('hybrid', [False, True])
def test_managed_registration_precedes_every_runtime_creation(managed, hybrid):
    state = managed
    if hybrid:
        from tests.utils.model import new_hybrid_workload

        state.instance.distributed_servers.workloads = [
            ModelInstanceWorkload.model_validate(new_hybrid_workload(worker_id))
            for worker_id in (1, 2)
        ]
        state.instance.distributed_servers.orchestration_mode = "managed"
    local = _local_manifest(state)
    state.instance.distributed_servers.workloads = []
    state.instance.distributed_servers.orchestration_mode = "managed"
    assert state.serve._start_managed_workloads(
        state.instance.model_copy(deep=True), local, _managed_plans(local)
    )
    assert state.events[0] == "report"
    assert state.create.call_count == (1 if hybrid else 2)
    for call in state.create.call_args_list:
        assert call.args[0].labels["gpustack.ai/restart-count"] == "3"
        assert call.args[0].labels["gpustack.ai/worker-id"] == str(
            state.serve._worker_id
        )


@pytest.mark.parametrize('check', ['state', 'health'])
def test_slow_managed_creation_does_not_block_state_reconciliation(managed, check):
    state = managed
    local = _local_manifest(state)
    state.instance.distributed_servers.workloads = []
    state.instance.distributed_servers.orchestration_mode = 'managed'
    connection = MagicMock()
    connection.poll.return_value = True
    connection.recv.return_value = (local, _managed_plans(local))
    state.serve._managed_launches[(state.instance.id, state.instance.restart_count)] = (
        connection
    )
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    errors = []
    create = state.create.side_effect

    def slow_create(plan):
        entered.set()
        assert release.wait(10)
        create(plan)

    def reconcile():
        try:
            state.serve._sync_managed_model_instance(state.instance)
        except BaseException as exc:
            errors.append(exc)
        finally:
            returned.set()

    def check_health():
        try:
            state.serve._sync_managed_rank_health(state.instance)
        except BaseException as exc:
            errors.append(exc)
        finally:
            returned.set()

    state.create.side_effect = slow_create
    caller = threading.Thread(target=reconcile, daemon=True)
    try:
        caller.start()
        assert entered.wait(2)
        assert returned.wait(2), 'A slow image pull must not hold the sync caller'
        caller.join(2)
        returned.clear()
        caller = threading.Thread(
            target=reconcile if check == 'state' else check_health, daemon=True
        )
        caller.start()
        assert returned.wait(2), 'A busy instance must not hold the next sync pass'
        assert state.create.call_count == 1
    finally:
        release.set()
        caller.join(2)
        executor = getattr(state.serve, '_managed_task_executor', None)
        if executor is not None:
            executor.shutdown(wait=True)
    assert not errors
    assert state.create.call_count == len(local)
    connection.close.assert_called_once()


@pytest.mark.parametrize('teardown', [False, True])
def test_managed_tasks_have_bounded_queues_and_reuse_finished_slots(manager, teardown):
    serve, _ = manager
    jobs = []

    def submit(*args):
        job = Future()
        jobs.append(job)
        return job

    executor = MagicMock()
    executor.submit.side_effect = submit
    if teardown:
        serve._managed_stop_executor = executor
    else:
        serve._managed_task_executor = executor
    instances = [_instance().model_copy(update={'id': i}) for i in range(10)]
    tasks = serve._managed_stop_tasks if teardown else serve._managed_tasks
    other = serve._managed_tasks if teardown else serve._managed_stop_tasks

    for instance in instances:
        admitted = serve._schedule_managed_task(
            instance, serve._prepare_managed_model_instance, teardown=teardown
        )
        assert admitted == (instance.id < 4)
        if instance.id in tasks:
            assert not serve._schedule_managed_task(
                instance, serve._stop_model_instance, teardown=not teardown
            )
    assert executor.submit.call_count == 4
    assert len(tasks) == 4
    assert not other

    jobs[0].set_result(None)
    serve._schedule_managed_task(
        instances[-1], serve._prepare_managed_model_instance, teardown=teardown
    )
    assert executor.submit.call_count == 5
    assert len(tasks) == 4


def test_new_generation_stop_cleans_tracked_runtime_before_releasing_ports(managed):
    state = managed
    previous = _running_managed(state)
    state.instance.restart_count += 1
    state.instance.state = ModelInstanceStateEnum.ERROR
    state.instance.distributed_servers.workloads = []
    delete = state.delete.side_effect
    state.delete.side_effect = lambda name: None

    state.serve._sync_managed_model_instance(state.instance)
    _finish_managed_tasks(state)
    assert state.delete.call_count == 2
    assert previous.id in state.serve._assigned_ports
    assert state.serve._model_instance_by_instance_id[previous.id].restart_count == 3

    state.delete.side_effect = delete
    state.serve._sync_managed_model_instance(state.instance)
    _finish_managed_tasks(state)
    assert not state.runtime
    assert previous.id not in state.serve._assigned_ports
    state.create.assert_not_called()


def test_saturated_preparation_does_not_starve_another_instances_stop(managed):
    state = managed
    instance = _running_managed(state)
    entered = [threading.Event() for _ in range(4)]
    release = threading.Event()

    def prepare(mi):
        entered[mi.id - 10].set()
        assert release.wait(10)

    try:
        for identity in range(10, 14):
            state.serve._schedule_managed_task(
                instance.model_copy(update={'id': identity}), prepare
            )
        assert all(event.wait(2) for event in entered)
        stopped = threading.Event()
        delete = state.delete.side_effect

        def observe_delete(name):
            delete(name)
            stopped.set()

        state.delete.side_effect = observe_delete
        state.instance.state = ModelInstanceStateEnum.ERROR
        state.serve._sync_managed_model_instance(state.instance)
        state.serve._retry_managed_stops()
        assert stopped.wait(2), 'Unrelated creation must not consume teardown capacity'
    finally:
        release.set()
        _finish_managed_tasks(state)
    assert not state.runtime
    assert instance.id not in state.serve._assigned_ports


@pytest.mark.parametrize('registry', ['managed', 'smg'])
def test_stop_retry_snapshots_exclude_concurrent_lifecycle_writes(
    smg, monkeypatch, registry
):
    state = smg
    entered, release, writing, written = [threading.Event() for _ in range(4)]
    errors = []

    class PausedSnapshot(dict):
        def snapshot(self, iterator):
            yield next(iterator)
            entered.set()
            assert release.wait(5)
            yield from iterator

        def items(self):
            return self.snapshot(iter(super().items()))

        def values(self):
            return self.snapshot(iter(super().values()))

    instance = state.instance.model_copy(deep=True)
    item = instance.distributed_servers.workloads[-1]
    if registry == 'managed':
        state.serve._managed_stopping = PausedSnapshot({(1, 3): False, (2, 3): False})
    else:
        state.serve._smg_stopping = PausedSnapshot(
            {
                (1, 3, item.name, 0): (instance, item),
                (2, 3, item.name, 0): (instance, item),
            }
        )
    monkeypatch.setattr(state.serve, '_schedule_managed_task', MagicMock())
    state.list.side_effect = RuntimeError('Runtime unavailable')

    def retry():
        try:
            state.serve._retry_managed_stops()
        except BaseException as exc:
            errors.append(exc)

    def write():
        writing.set()
        try:
            other = instance.model_copy(update={'id': 10})
            if registry == 'managed':
                state.serve._stop_model_instance(other)
            else:
                state.serve._stop_smg(other, item)
        except BaseException as exc:
            errors.append(exc)
        finally:
            written.set()

    reader = threading.Thread(target=retry, daemon=True)
    writer = threading.Thread(target=write, daemon=True)
    try:
        reader.start()
        assert entered.wait(2)
        writer.start()
        assert writing.wait(2)
        assert not written.wait(0.1), 'Retry snapshots must exclude concurrent writes'
    finally:
        release.set()
        reader.join(2)
        if writer.ident is not None:
            writer.join(2)
    assert written.is_set()
    assert not errors


@pytest.mark.parametrize('operation', ['stop', 'retire', 'create', 'engine_stop'])
@pytest.mark.parametrize('manager', [1], indirect=True)
def test_slow_smg_lifecycle_does_not_block_reconciliation(smg, monkeypatch, operation):
    state = smg
    if operation == 'create':
        _pending_cpu(state)
        target, attribute = state.create, 'side_effect'
    else:
        if operation == 'stop':
            state.instance.distributed_servers.workloads[
                -1
            ].component.stop_requested = True
        elif operation == 'engine_stop':
            state.instance.distributed_servers.workloads[0].state = 'failed'
        else:
            item = state.instance.distributed_servers.workloads[-1]
            item.name = f'{state.instance.name}-r3-smg-a1'
            item.component.attempt = 1
            item.state = ModelInstanceWorkloadStateEnum.PENDING
        target, attribute = state.delete, 'side_effect'
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    original = getattr(target, attribute)
    errors = []

    def slow(*args):
        entered.set()
        assert release.wait(10)
        return original(*args)

    def reconcile():
        try:
            state.serve._sync_managed_model_instance(state.instance)
        except BaseException as exc:
            errors.append(exc)
        finally:
            returned.set()

    monkeypatch.setattr(target, attribute, slow)
    caller = threading.Thread(target=reconcile, daemon=True)
    try:
        caller.start()
        assert entered.wait(2)
        assert returned.wait(
            2
        ), 'CPU lifecycle work must not hold shared reconciliation'
        assert state.instance.id in state.serve._assigned_ports
    finally:
        release.set()
        caller.join(2)
        _finish_managed_tasks(state)
    assert not errors
    assert any(w.name == 'engine-1' for w in state.runtime) == (
        operation != 'engine_stop'
    )


@pytest.mark.parametrize('operation', ['observe', 'stop'])
@pytest.mark.parametrize('manager', [1], indirect=True)
def test_saturated_preparation_preserves_smg_observation_and_teardown(smg, operation):
    state = smg
    executor = MagicMock()
    executor.submit.side_effect = lambda *args: Future()
    state.serve._managed_task_executor = executor
    for identity in range(10, 14):
        state.serve._schedule_managed_task(
            state.instance.model_copy(update={'id': identity}),
            state.serve._prepare_managed_model_instance,
        )
    engine = state.instance.distributed_servers.workloads[0]
    engine.state = ModelInstanceWorkloadStateEnum.PENDING
    engine.component.ready = False
    if operation == 'stop':
        state.instance.distributed_servers.workloads[-1].component.stop_requested = True
    state.serve._sync_managed_model_instance(state.instance)
    with state.serve._model_instance_start_lock:
        stopping = list(state.serve._managed_stop_tasks.values())
    for job in stopping:
        job.result(timeout=2)
    engine = state.instance.distributed_servers.workloads[0]
    assert engine.state == ModelInstanceWorkloadStateEnum.RUNNING
    assert any(w.name == 'engine-1' for w in state.runtime)
    assert len(state.runtime) == (1 if operation == 'stop' else 2)
    assert state.instance.id in state.serve._assigned_ports


@pytest.mark.parametrize('manager', [1], indirect=True)
def test_pending_smg_at_capacity_still_detects_and_stops_failed_engine(smg):
    state = smg
    _pending_cpu(state)
    executor = MagicMock()
    executor.submit.side_effect = lambda *args: Future()
    state.serve._managed_task_executor = executor
    for identity in range(10, 14):
        state.serve._schedule_managed_task(
            state.instance.model_copy(update={'id': identity}),
            state.serve._prepare_managed_model_instance,
        )
    state.runtime[0].state = WorkloadStatusStateEnum.FAILED

    state.serve._sync_managed_model_instance(state.instance)
    for job in list(state.serve._managed_stop_tasks.values()):
        job.result(timeout=2)

    assert state.client.model_instances.report_workloads.called
    assert not state.runtime
    assert state.instance.id not in state.serve._assigned_ports
    state.create.assert_not_called()


@pytest.mark.parametrize('manager', [1], indirect=True)
def test_unconfirmed_smg_cleanup_still_detects_and_stops_failed_engine(smg):
    state = smg
    state.instance.state = ModelInstanceStateEnum.STARTING
    cpu = state.instance.distributed_servers.workloads[-1]
    cpu.state = ModelInstanceWorkloadStateEnum.FAILED
    cpu.component.ready = False
    cpu.component.stop_requested = True
    state.serve._hold_managed_instance(state.instance)
    delete = state.delete.side_effect
    state.delete.side_effect = lambda name: (
        delete(name) if name.startswith('engine') else None
    )
    assert not state.serve._stop_smg(state.instance, cpu)
    state.runtime[0].state = WorkloadStatusStateEnum.FAILED

    state.serve._sync_managed_model_instance(state.instance)
    _finish_managed_tasks(state)
    state.serve._retry_managed_stops()
    _finish_managed_tasks(state)

    engine = next(
        w for w in state.instance.distributed_servers.workloads if w.name == 'engine-1'
    )
    assert engine.state == ModelInstanceWorkloadStateEnum.FAILED
    assert not any(w.name.startswith('engine') for w in state.runtime)
    assert state.serve._managed_stopping
    assert state.instance.id in state.serve._assigned_ports
    state.create.assert_not_called()


@pytest.mark.parametrize('manager', [1], indirect=True)
@pytest.mark.parametrize('retired', [False, True])
def test_whole_instance_stop_takes_priority_over_unconfirmed_smg_retry(smg, retired):
    state = smg
    cpu = state.instance.distributed_servers.workloads[-1]
    state.instance.state = ModelInstanceStateEnum.ERROR
    runtime_list = state.list.side_effect
    state.list.side_effect = OSError('Runtime unavailable')
    assert not state.serve._stop_smg(state.instance, cpu)
    if retired:
        current = cpu.model_copy(deep=True)
        current.name = f'{state.instance.name}-r3-smg-a1'
        current.component.attempt = 1
        current.component.ready = False
        current.state = ModelInstanceWorkloadStateEnum.PENDING
        state.instance.distributed_servers.workloads[-1] = current
        state.serve._hold_managed_instance(state.instance)
    assert not state.serve._stop_model_instance(state.instance)
    state.list.side_effect = runtime_list
    entered, release = threading.Event(), threading.Event()
    delete = state.delete.side_effect

    def unconfirmed_cpu_delete(name):
        if name.startswith('engine'):
            return delete(name)
        entered.set()
        assert release.wait(5)

    state.delete.side_effect = unconfirmed_cpu_delete
    try:
        state.serve._retry_managed_stops()
        assert entered.wait(2)
        assert not any(w.name.startswith('engine') for w in state.runtime)
        assert state.instance.id in state.serve._assigned_ports
    finally:
        release.set()
        _finish_managed_tasks(state)
    assert state.serve._managed_stopping
    assert state.serve._smg_stopping
    state.delete.side_effect = delete
    state.serve._retry_managed_stops()
    _finish_managed_tasks(state)
    assert not state.runtime
    assert not state.serve._managed_stopping
    assert not state.serve._smg_stopping
    assert state.instance.id not in state.serve._assigned_ports


@pytest.mark.parametrize('manager', [1], indirect=True)
@pytest.mark.parametrize(
    'stopping, pause_at',
    [
        ('component', 'runtime'),
        ('instance', 'runtime'),
        ('instance', 'logs'),
        ('instance', 'ports'),
        ('instance', 'forgotten'),
    ],
)
def test_deleted_instance_reaping_does_not_wait_for_background_stop(
    smg, monkeypatch, stopping, pause_at
):
    state = smg
    cpu = state.instance.distributed_servers.workloads[-1]
    entered, release, returned = [threading.Event() for _ in range(3)]
    errors = []
    delete = state.delete.side_effect
    state.serve._purge_instance_logs = MagicMock()

    def pause(*args, **kwargs):
        entered.set()
        assert release.wait(5)

    def slow_delete(name):
        if stopping == 'instance' or not name.startswith('engine'):
            pause()
        delete(name)

    if pause_at == 'runtime':
        state.delete.side_effect = slow_delete
    elif pause_at == 'logs':
        monkeypatch.setattr(state.serve, '_stop_container_log_persistence', pause)
    elif pause_at == 'ports':
        monkeypatch.setattr(state.serve, '_release_distributed_ports', pause)
    else:
        forget = state.serve._forget_stopped_model_instance

        def paused_forget(*args, **kwargs):
            forget(*args, **kwargs)
            pause()

        monkeypatch.setattr(
            state.serve, '_forget_stopped_model_instance', paused_forget
        )
    if stopping == 'instance':
        state.serve._schedule_managed_stop(state.instance)
    else:
        state.serve._schedule_managed_task(
            state.instance,
            lambda mi: state.serve._retry_smg_stop(mi, cpu),
            teardown=True,
        )
    assert entered.wait(2)
    state.client.model_instances.list.side_effect = lambda **kwargs: SimpleNamespace(
        items=[]
    )
    state.client.model_instances.get.side_effect = NotFoundException('Instance deleted')

    def reconcile():
        try:
            state.serve.sync_model_instances_state()
        except BaseException as exc:
            errors.append(exc)
        finally:
            returned.set()

    caller = threading.Thread(target=reconcile, daemon=True)
    try:
        caller.start()
        assert returned.wait(2), 'Reaping must not wait for an in-flight stop'
        if pause_at in ('runtime', 'logs', 'ports'):
            assert state.instance.id in state.serve._assigned_ports
        assert state.serve._managed_stopping[(state.instance.id, 3)] is True
    finally:
        release.set()
        caller.join(2)
        _finish_managed_tasks(state)
    assert not errors
    state.serve._retry_managed_stops()
    _finish_managed_tasks(state)
    assert not state.runtime
    state.serve._purge_instance_logs.assert_called_once_with(state.instance.id)
    assert state.instance.id not in state.serve._assigned_ports
    assert state.instance.id not in state.serve._port_reservations._claim_generations


@pytest.mark.parametrize('manager', [1], indirect=True)
def test_deleted_instance_stop_retries_after_cleanup_until_its_future_finishes(smg):
    state = smg
    entered, release = threading.Event(), threading.Event()

    def stop(mi):
        assert state.serve._stop_model_instance(mi)
        entered.set()
        assert release.wait(5)

    state.serve._schedule_managed_task(state.instance, stop, teardown=True)
    try:
        assert entered.wait(2)
        assert state.instance.id not in state.serve._model_instance_by_instance_id
        state.client.model_instances.get.side_effect = NotFoundException('Deleted')
        state.serve._sync_managed_model_instance(state.instance)
        assert state.serve._managed_stopping[(state.instance.id, 3)] is True
    finally:
        release.set()
        _finish_managed_tasks(state)

    state.serve._retry_managed_stops()
    _finish_managed_tasks(state)
    state.serve._purge_instance_logs.assert_called_once_with(state.instance.id)
    assert not state.serve._managed_stopping
    assert state.instance.id not in state.serve._model_instance_by_instance_id
    assert state.instance.id not in state.serve._port_reservations._claim_generations


@pytest.mark.parametrize('manager', [1], indirect=True)
def test_managed_stop_retries_give_other_instances_smg_cleanup_a_turn(smg):
    state = smg
    # Give the waiting CPU cleanup a later identity than the blocked stops.
    state.serve._model_instance_by_instance_id.pop(state.instance.id)
    state.serve._port_reservations.release_model(state.instance.id)
    state.instance.id = 100
    for item in state.runtime:
        item.labels['gpustack.ai/model-instance-id'] = str(state.instance.id)
    cpu = state.instance.distributed_servers.workloads[-1]
    cpu.state = ModelInstanceWorkloadStateEnum.FAILED
    cpu.component.ready = False
    cpu.component.stop_requested = True
    state.instance.state = ModelInstanceStateEnum.STARTING
    state.serve._hold_managed_instance(state.instance)
    runtime_list = state.list.side_effect
    state.list.side_effect = OSError('Runtime unavailable')
    assert not state.serve._stop_smg(state.instance, cpu)
    instances = {state.instance.id: state.instance}
    entered = {i: threading.Event() for i in range(10, 14)}
    release, cpu_deleted = threading.Event(), threading.Event()
    engine = state.instance.distributed_servers.workloads[0]
    for i in entered:
        item = engine.model_copy(deep=True, update={'name': f'blocked-{i}'})
        other = state.instance.model_copy(
            deep=True,
            update={
                'id': i,
                'name': f'blocked-instance-{i}',
                'state': ModelInstanceStateEnum.ERROR,
                'distributed_servers': DistributedServers(workloads=[item]),
            },
        )
        instances[i] = other
        state.runtime.append(
            SimpleNamespace(
                name=item.name,
                labels={
                    **state.serve._managed_workload_labels(other),
                    **state.serve._component_workload_labels(item),
                },
                state=WorkloadStatusStateEnum.RUNNING,
            )
        )
        state.serve._hold_managed_instance(other)
        assert not state.serve._stop_model_instance(other)
    state.list.side_effect = runtime_list
    state.client.model_instances.get.side_effect = lambda **kwargs: instances[
        kwargs['id']
    ].model_copy(deep=True)
    delete = state.delete.side_effect

    def unconfirmed_delete(name):
        if name.startswith('blocked-'):
            entered[int(name.split('-')[-1])].set()
            assert release.wait(5)
        else:
            delete(name)
            cpu_deleted.set()

    state.delete.side_effect = unconfirmed_delete
    try:
        state.serve._retry_managed_stops()
        assert all(event.wait(2) for event in entered.values())
        assert not cpu_deleted.is_set()
        release.set()
        _finish_managed_tasks(state)
        release.clear()

        state.serve._retry_managed_stops()
        assert cpu_deleted.wait(2), 'Other instances must not monopolize stop retries'
    finally:
        release.set()
        _finish_managed_tasks(state)
    assert state.serve._managed_stopping
    assert not state.serve._smg_stopping
    assert any(item.name == 'engine-1' for item in state.runtime)
    assert state.instance.id in state.serve._assigned_ports
    assert all(call.args[0] != 'engine-1' for call in state.delete.call_args_list)


@pytest.mark.parametrize('manager', [1], indirect=True)
@pytest.mark.parametrize('entrypoint', ['state', 'health'])
def test_instance_snapshot_survives_background_teardown(smg, monkeypatch, entrypoint):
    state = smg
    other = state.instance.model_copy(
        deep=True,
        update={
            'id': 10,
            'name': 'other',
            'state': ModelInstanceStateEnum.SCHEDULED,
            'distributed_servers': DistributedServers(workloads=[]),
            'port': None,
            'ports': None,
            'named_ports': None,
        },
    )
    state.serve._hold_managed_instance(other)
    state.client.model_instances.get.side_effect = lambda **kwargs: (
        other if kwargs['id'] == other.id else state.instance
    ).model_copy(deep=True)
    entered, release = threading.Event(), threading.Event()
    errors = []

    def pause():
        entered.set()
        assert release.wait(5)

    if entrypoint == 'state':
        is_local = state.serve._is_local_model_instance

        def paused_local(mi):
            if mi.id == state.instance.id:
                pause()
            return is_local(mi)

        monkeypatch.setattr(state.serve, '_is_local_model_instance', paused_local)
    else:
        # Pause the legacy-health filter while its instance iterator is active.
        state.instance.state = ModelInstanceStateEnum.STARTING

        class PausedWorkerId(int):
            def __eq__(self, other):
                pause()
                return super().__eq__(other)

        state.serve._model_instance_by_instance_id[state.instance.id].worker_id = (
            PausedWorkerId(1)
        )

    def reconcile():
        try:
            if entrypoint == 'state':
                state.serve.sync_model_instances_state()
            else:
                state.serve.sync_model_instances_inference_health()
        except BaseException as exc:
            errors.append(exc)

    caller = threading.Thread(target=reconcile, daemon=True)
    try:
        caller.start()
        assert entered.wait(2)
        state.serve._schedule_managed_task(
            other, state.serve._stop_model_instance, teardown=True
        )
        _finish_managed_tasks(state)
        assert other.id not in state.serve._model_instance_by_instance_id
    finally:
        release.set()
        caller.join(2)
    assert not caller.is_alive()
    assert not errors


@pytest.mark.parametrize(
    'entrypoint', ['sync_model_instances_state', '_retry_managed_stops']
)
def test_slow_managed_stop_keeps_reconciliation_free_and_ports_reserved(
    managed, entrypoint
):
    state = managed
    instance = _running_managed(state)
    state.instance.state = ModelInstanceStateEnum.ERROR
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    delete = state.delete.side_effect
    errors = []

    def slow_delete(name):
        entered.set()
        assert release.wait(10)
        delete(name)

    def reconcile():
        try:
            state.serve._sync_managed_model_instance(state.instance)
            getattr(state.serve, entrypoint)()
        except BaseException as exc:
            errors.append(exc)
        finally:
            returned.set()

    state.delete.side_effect = slow_delete
    caller = threading.Thread(target=reconcile, daemon=True)
    try:
        caller.start()
        assert entered.wait(2)
        assert returned.wait(2), 'Pending teardown must not hold shared reconciliation'
        assert instance.id in state.serve._assigned_ports
        state.client.model_instances.report_workloads.assert_not_called()
    finally:
        release.set()
        caller.join(2)
        state.serve._managed_stop_executor.shutdown(wait=True)
    assert not errors
    assert not state.runtime
    assert instance.id not in state.serve._assigned_ports


@pytest.mark.parametrize(
    "error",
    [
        ForbiddenException("unbound worker"),
        ConflictException("stale generation"),
        ReadTimeout("report response lost"),
    ],
)
def test_managed_registration_failure_never_creates_workloads(managed, error):
    state = managed
    local = _local_manifest(state)
    state.instance.distributed_servers.workloads = []
    state.instance.distributed_servers.orchestration_mode = "managed"
    state.client.model_instances.report_workloads.side_effect = error
    assert not state.serve._start_managed_workloads(
        state.instance, local, _managed_plans(local)
    )
    state.create.assert_not_called()


@pytest.fixture
def preparing_external(managed, monkeypatch):
    state = managed
    mi = state.instance
    mi.worker_id = state.serve._worker_id
    mi.distributed_servers = DistributedServers(workloads=[])
    mi.gpu_indexes = [0, 1]
    model = state.client.models.get.return_value
    model.backend = BackendEnum.VLLM
    model.backend_parameters = ["--data-parallel-external-lb", "--data-parallel-size=2"]
    mi.port = None
    mi.ports = None
    mi.named_ports = None

    def persist(id, **patch):
        for key, value in patch.items():
            setattr(state.instance, key, value)

    state.serve._update_model_instance.side_effect = persist
    receive, send = MagicMock(), MagicMock()
    receive.poll.return_value = False
    monkeypatch.setattr(
        "gpustack.worker.serve_manager.multiprocessing.Pipe",
        MagicMock(return_value=(receive, send)),
    )
    state.serve._sync_managed_model_instance(mi)
    assert (mi.id, mi.restart_count) in state.serve._managed_launches
    state.create.assert_not_called()
    assert state.client.model_instances.report_workloads.call_count == 0
    process = state.serve._provisioning_processes[mi.id]
    process.is_alive.return_value = True
    state.receiver = receive
    state.preparer = process
    assert len(state.serve._assigned_ports[mi.id]) == 13
    return state


def _prepared_external_workload(state):
    item = ModelInstanceWorkload.model_validate(
        new_model_instance_workload(
            worker_id=state.serve._worker_id,
            name=state.instance.name,
        )
    )
    item.dp_rank_placements[0].gpu_indexes = [0]
    item.dp_rank_placements[0].api_port = state.instance.ports[0]
    second = item.dp_rank_placements[0].model_copy(deep=True)
    second.dp_rank = 1
    second.gpu_indexes = [1]
    second.api_port = state.instance.ports[1]
    item.dp_rank_placements.append(second)
    return item


def test_managed_launch_payloads_precede_local_registration(managed, monkeypatch):
    state = managed
    serve, mi = state.serve, state.instance
    local = _local_manifest(state)
    mi.distributed_servers.workloads = [
        item
        for item in mi.distributed_servers.workloads
        if item.worker_id != serve._worker_id
    ]
    mi.distributed_servers.orchestration_mode = "managed"
    model = state.client.models.get.return_value
    model.backend = BackendEnum.VLLM
    model.backend_parameters = ["--data-parallel-external-lb", "--dp=2", "--tp=2"]
    monkeypatch.setattr(serve, "_assign_external_ports", lambda instance, model: None)
    monkeypatch.setattr(
        serve,
        "_update_model_instance",
        ServeManager._update_model_instance.__get__(serve),
    )
    receiver, sender = MagicMock(), MagicMock()
    receiver.poll.return_value = False
    monkeypatch.setattr(
        "gpustack.worker.serve_manager.multiprocessing.Pipe",
        MagicMock(return_value=(receiver, sender)),
    )
    normal_report = state.client.model_instances.report_workloads.side_effect
    phases = []

    def report(*, id, report):
        model_update = report
        update = ModelInstanceWorkloadUpdate.model_validate(model_update.model_dump())
        if update.preparation is None:
            phases.append("registration")
            return normal_report(id=id, report=update)
        assert update.workloads is None
        assert update.observations is None
        assert not any(
            item.worker_id == serve._worker_id
            for item in mi.distributed_servers.workloads
        )
        phases.append("preparation")
        return mi.model_copy(deep=True)

    state.client.model_instances.report_workloads.side_effect = report
    serve._start_model_instance(mi.model_copy(deep=True))
    assert phases == ["preparation", "preparation"]
    assert (
        state.client.model_instances.report_workloads.call_args.kwargs[
            "report"
        ].preparation.pid
        == 1234
    )
    state.create.assert_not_called()
    receiver.poll.return_value = True
    receiver.recv.return_value = (local, _managed_plans(local))
    serve._sync_managed_model_instance(mi.model_copy(deep=True))
    _finish_managed_tasks(state)
    assert phases == ["preparation", "preparation", "registration"]
    assert state.create.call_count == len(local)


@pytest.mark.parametrize("delivered", [False, True])
def test_external_preparation_returns_to_parent_before_registration(
    preparing_external, delivered
):
    state = preparing_external
    mi = state.instance
    state.serve._sync_managed_model_instance(mi)
    state.preparer.start.assert_called_once()
    if delivered:
        before_registration = mi.model_copy(deep=True)
        local = [_prepared_external_workload(state)]
        state.receiver.poll.return_value = True
        state.receiver.recv.return_value = (local, _managed_plans(local))
        state.serve._sync_managed_model_instance(mi)
        _finish_managed_tasks(state)
        assert state.events[0] == "report"
        assert state.create.call_count == len(local)
        state.receiver.close.assert_called_once()
        state.serve._prepare_managed_model_instance(before_registration)
        state.preparer.start.assert_called_once()
        state.serve._sync_managed_model_instance(state.instance)
        assert len(state.serve._assigned_ports[mi.id]) == 13


@pytest.mark.parametrize("stopped", [False, True])
def test_external_stop_during_preparation_closes_channel_before_any_create(
    preparing_external, stopped
):
    state = preparing_external
    state.instance.state = ModelInstanceStateEnum.ERROR
    state.preparer.is_alive.return_value = not stopped
    state.receiver.poll.return_value = True
    state.serve._sync_managed_model_instance(state.instance)
    _finish_managed_tasks(state)
    state.receiver.recv.assert_not_called()
    state.receiver.close.assert_called_once()
    state.create.assert_not_called()
    assert (state.instance.id in state.serve._assigned_ports) is not stopped


def test_ordinary_preparation_can_launch_while_other_replicas_are_stale(
    preparing_external,
):
    state = preparing_external
    state.client.models.get.return_value.stale = True
    local = [_prepared_external_workload(state)]
    state.receiver.poll.return_value = True
    state.receiver.recv.return_value = (local, _managed_plans(local))

    state.serve._sync_managed_model_instance(state.instance)
    _finish_managed_tasks(state)

    assert state.events[0] == 'report'
    state.create.assert_called_once()
    state.receiver.close.assert_called_once()


def test_managed_stop_intent_without_local_snapshot_resumes_from_server(managed):
    state = managed
    state.instance.state = ModelInstanceStateEnum.ERROR
    key = (state.instance.id, state.instance.restart_count)
    state.serve._managed_stopping[key] = False
    state.serve._model_instance_by_instance_id.clear()

    state.serve.sync_model_instances_state()
    _finish_managed_tasks(state)

    state.create.assert_not_called()
    assert all(
        item.state == ModelInstanceWorkloadStateEnum.STOPPED
        for item in _local_manifest(state)
    )
    assert key not in state.serve._managed_stopping


@pytest.mark.parametrize(
    "failure", ["eof", "stale", "registration", "connect", "timeout"]
)
def test_external_failed_preparation_never_creates_a_container(
    preparing_external, failure
):
    state = preparing_external
    state.receiver.poll.return_value = True
    item = ModelInstanceWorkload.model_validate(
        new_model_instance_workload(worker_id=state.serve._worker_id, name="prepared")
    )
    item.dp_rank_placements[0].gpu_indexes = [0]
    state.receiver.recv.return_value = ([item], _managed_plans([item]))
    if failure == "eof":
        state.receiver.recv.side_effect = EOFError()
    elif failure in ("connect", "timeout"):
        import httpx

        error = httpx.ConnectError if failure == "connect" else httpx.ReadTimeout
        state.client.models.get.side_effect = error("control plane unavailable")
    elif failure == "stale":
        state.instance.group_id = 'pd-generation'
        state.client.models.get.return_value.stale = True
    else:
        state.client.model_instances.report_workloads.side_effect = ConflictException(
            "stale generation"
        )
    state.serve._sync_managed_model_instance(state.instance)
    _finish_managed_tasks(state)
    state.create.assert_not_called()
    state.receiver.close.assert_called_once()
    assert (
        state.instance.id,
        state.instance.restart_count,
    ) not in state.serve._managed_launches


def test_managed_replayed_start_adopts_existing_workloads(managed):
    state = managed
    local = _local_manifest(state)
    assert state.serve._start_managed_workloads(
        state.instance, local, _managed_plans(local)
    )
    assert state.serve._start_managed_workloads(
        state.instance, local, _managed_plans(local)
    )
    assert state.create.call_count == 2
    assert state.instance.restart_count == 3


def test_managed_partial_create_failure_cleans_up_even_a_lost_create_response(managed):
    state = managed
    local = _local_manifest(state)
    normal_create = state.create.side_effect

    def create(plan):
        normal_create(plan)
        if state.create.call_count == 2:
            raise OSError("create response lost")

    state.create.side_effect = create
    assert not state.serve._start_managed_workloads(
        state.instance, local, _managed_plans(local)
    )
    assert not state.runtime
    assert {item.state for item in _local_manifest(state)} == {
        ModelInstanceWorkloadStateEnum.STOPPED
    }
    assert state.instance.id not in state.serve._assigned_ports


@pytest.mark.parametrize(
    "runtime_state",
    [WorkloadStatusStateEnum.FAILED, WorkloadStatusStateEnum.UNHEALTHY],
)
def test_managed_failed_start_reports_runtime_failure_while_deletion_is_pending(
    managed, runtime_state
):
    state = managed
    instance = _running_managed(state)
    for item in state.instance.distributed_servers.workloads:
        item.state = ModelInstanceWorkloadStateEnum.RUNNING
        for placement in item.dp_rank_placements:
            if placement.node_rank == 0:
                placement.ready = True
    for runtime in state.runtime:
        runtime.state = runtime_state
    state.delete.side_effect = lambda name: None
    local = _local_manifest(state)
    assert not state.serve._start_managed_workloads(
        instance, local, _managed_plans(local)
    )
    assert {item.state for item in _local_manifest(state)} == {
        ModelInstanceWorkloadStateEnum.FAILED
    }
    assert all(
        not placement.ready
        for item in _local_manifest(state)
        for placement in item.dp_rank_placements
    )
    assert state.instance.id in state.serve._assigned_ports


def test_managed_unhealthy_runtime_stops_group_and_retains_claims(managed):
    state = managed
    _running_managed(state)
    state.instance.state = ModelInstanceStateEnum.RUNNING
    for item in state.instance.distributed_servers.workloads:
        item.state = ModelInstanceWorkloadStateEnum.RUNNING
        for placement in item.dp_rank_placements:
            if placement.node_rank == 0:
                placement.ready = True
    failed_name = state.runtime[0].name
    state.runtime[0].state = WorkloadStatusStateEnum.UNHEALTHY
    state.delete.side_effect = lambda name: None

    state.serve._sync_managed_model_instance(state.instance)
    _finish_managed_tasks(state)

    failed = next(item for item in _local_manifest(state) if item.name == failed_name)
    assert failed.state == ModelInstanceWorkloadStateEnum.FAILED
    assert not any(placement.ready for placement in failed.dp_rank_placements)
    assert {call.args[0] for call in state.delete.call_args_list} == {
        item.name for item in _local_manifest(state)
    }
    assert state.instance.id in state.serve._assigned_ports
    assert state.instance.restart_count == 3
    state.create.assert_not_called()


def test_managed_stop_during_creation_cancels_the_remaining_plans(managed):
    state = managed
    local = _local_manifest(state)
    normal_create = state.create.side_effect

    def create(plan):
        normal_create(plan)
        assert not state.serve._stop_model_instance(state.instance)
        assert state.instance.id in state.serve._assigned_ports
        assert all(
            item.state != ModelInstanceWorkloadStateEnum.STOPPED
            for item in _local_manifest(state)
        )

    state.create.side_effect = create
    assert not state.serve._start_managed_workloads(
        state.instance, local, _managed_plans(local)
    )
    assert state.create.call_count == 1
    assert not state.runtime
    assert state.instance.id not in state.serve._assigned_ports


def test_managed_worker_restart_adopts_all_local_containers_without_a_new_generation(
    managed,
):
    state = managed
    instance = _running_managed(state)
    state.serve._model_instance_by_instance_id.clear()
    state.serve._assigned_ports.clear()
    state.serve._dispatch_model_instance_event(
        Event(type=EventType.CREATED, data=instance)
    )
    assert instance.id in state.serve._model_instance_by_instance_id
    assert instance.id in state.serve._assigned_ports
    assert {item.state for item in _local_manifest(state)} == {
        ModelInstanceWorkloadStateEnum.RUNNING
    }
    assert state.instance.restart_count == 3
    state.create.assert_not_called()
    state.delete.assert_not_called()


def test_empty_managed_manifest_does_not_fall_back_to_legacy_launch_or_ports(managed):
    state = managed
    state.instance.distributed_servers.workloads = []
    state.instance.distributed_servers.orchestration_mode = "managed"
    with patch.object(state.serve, "_launch_model_instance") as legacy_launch:
        state.serve._dispatch_model_instance_event(
            Event(type=EventType.CREATED, data=state.instance)
        )
    legacy_launch.assert_not_called()
    state.create.assert_not_called()
    state.client.model_instances.report_workloads.assert_not_called()
    assert not state.serve._assigned_ports.get(state.instance.id)
    assert state.serve.get_instance_port_by_model_instance_id(state.instance.id) is None


def test_managed_running_instance_does_not_use_legacy_health_or_proxy_port(managed):
    state = managed
    state.instance.state = ModelInstanceStateEnum.RUNNING
    state.instance.distributed_servers.workloads = []
    state.instance.distributed_servers.orchestration_mode = "managed"
    state.serve._model_instance_by_instance_id[state.instance.id] = state.instance
    state.serve._get_model.return_value.env = {
        "GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_ENABLED": "true"
    }
    with patch.object(state.serve, "_probe_running_instance") as probe:
        state.serve.sync_model_instances_inference_health()
    probe.assert_not_called()
    assert state.serve.get_instance_port_by_model_instance_id(state.instance.id) is None


def test_managed_stop_from_another_thread_fences_in_flight_creation(
    managed, monkeypatch
):
    state = managed
    local = _local_manifest(state)
    entered, release, stop_requested = (threading.Event() for _ in range(3))
    normal_create = state.create.side_effect
    normal_stop = state.serve._stop_assigned_model_instance

    def create(plan):
        normal_create(plan)
        entered.set()
        assert release.wait(timeout=3)

    def stop(*args):
        stop_requested.set()
        return normal_stop(*args)

    state.create.side_effect = create
    monkeypatch.setattr(state.serve, "_stop_assigned_model_instance", stop)
    with ThreadPoolExecutor(max_workers=2) as executor:
        launch = executor.submit(
            state.serve._start_managed_workloads,
            state.instance,
            local,
            _managed_plans(local),
        )
        try:
            assert entered.wait(timeout=2)
            cleanup = executor.submit(state.serve._stop_model_instance, state.instance)
            assert stop_requested.wait(timeout=2)
            assert state.instance.id in state.serve._assigned_ports
            assert not cleanup.done()
        finally:
            release.set()
        assert not launch.result(timeout=3)
        assert cleanup.result(timeout=3)
    assert state.create.call_count == 1
    assert not state.runtime
    assert state.instance.id not in state.serve._assigned_ports


def test_managed_generation_change_during_creation_cannot_launch_more_old_tasks(
    managed,
):
    state = managed
    local = _local_manifest(state)
    normal_create = state.create.side_effect

    def create(plan):
        normal_create(plan)
        state.instance.restart_count = 4
        state.instance.distributed_servers.workloads = []
        state.instance.distributed_servers.orchestration_mode = "managed"

    state.create.side_effect = create
    assert not state.serve._start_managed_workloads(
        state.instance.model_copy(deep=True), local, _managed_plans(local)
    )
    assert state.create.call_count == 1
    assert not state.runtime
    assert state.instance.restart_count == 4
    assert state.instance.distributed_servers.workloads == []
    assert state.client.model_instances.report_workloads.call_count == 1


def test_late_managed_stop_never_releases_the_new_generations_ports(managed):
    state = managed
    old = state.instance.model_copy(deep=True)
    old_key = (old.id, old.restart_count)
    old_connection = MagicMock()
    state.serve._managed_launches[old_key] = old_connection
    state.serve._managed_stopping[old_key] = False
    state.instance.restart_count = 4
    for item in state.instance.distributed_servers.workloads:
        item.name = item.name.replace("-r3-", "-r4-")
    current = _running_managed(state)
    current_key = (current.id, current.restart_count)
    current_connection = MagicMock()
    state.serve._managed_launches[current_key] = current_connection
    assert state.serve._stop_model_instance(old)
    old_connection.close.assert_called_once_with()
    assert old_key not in state.serve._managed_launches
    assert old_key not in state.serve._managed_stopping
    assert state.serve._managed_launches[current_key] is current_connection
    current_connection.close.assert_not_called()
    state.delete.assert_not_called()
    assert current.id in state.serve._assigned_ports
    assert state.serve._model_instance_by_instance_id[current.id].restart_count == 4


@pytest.mark.parametrize("trigger", ["error", "draining"])
def test_managed_watch_stops_only_local_tasks_and_keeps_other_workers_reports(
    managed, trigger
):
    state = managed
    instance = _running_managed(state)
    other = [
        item.model_dump()
        for item in instance.distributed_servers.workloads
        if item.worker_id != state.serve._worker_id
    ]
    state.runtime.append(
        SimpleNamespace(
            name="unrelated-model", labels={}, state=WorkloadStatusStateEnum.RUNNING
        )
    )
    if trigger == "error":
        state.instance.state = ModelInstanceStateEnum.ERROR
    else:
        state.instance.draining_since = datetime.now(timezone.utc)
        state.instance.distributed_servers.stop_requested = True
    state.serve._dispatch_model_instance_event(
        Event(type=EventType.UPDATED, data=state.instance)
    )
    _finish_managed_tasks(state)
    assert [item.name for item in state.runtime] == ["unrelated-model"]
    assert state.delete.call_count == 2
    assert [
        item.model_dump()
        for item in state.instance.distributed_servers.workloads
        if item.worker_id != state.serve._worker_id
    ] == other


def test_managed_periodic_sync_adopts_all_local_tasks_on_a_headless_worker(managed):
    state = managed
    _running_managed(state)
    state.serve._model_instance_by_instance_id.clear()
    state.serve._assigned_ports.clear()
    state.serve.sync_model_instances_state()
    assert state.instance.id in state.serve._model_instance_by_instance_id
    state.create.assert_not_called()
    state.delete.assert_not_called()
    state.serve._update_model_instance.assert_not_called()


@pytest.mark.parametrize("runtime_state", list(WorkloadStatusStateEnum))
def test_managed_delete_acceptance_or_terminal_status_does_not_confirm_stop(
    managed, runtime_state
):
    state = managed
    instance = _running_managed(state)
    for item in state.runtime:
        item.state = runtime_state
    state.delete.side_effect = lambda name: None
    assert not state.serve._stop_model_instance(instance)
    assert state.delete.call_count == 2
    assert instance.id in state.serve._assigned_ports
    assert all(
        item.state != ModelInstanceWorkloadStateEnum.STOPPED
        for item in _local_manifest(state)
    )
    state.runtime.clear()
    assert state.serve._stop_model_instance(instance)
    assert {item.state for item in _local_manifest(state)} == {
        ModelInstanceWorkloadStateEnum.STOPPED
    }
    assert instance.id not in state.serve._assigned_ports


def test_managed_stop_checks_all_same_name_runtime_objects(managed):
    state = managed
    instance = _running_managed(state)
    state.runtime.insert(
        0,
        SimpleNamespace(
            name=state.runtime[0].name,
            labels=state.runtime[0].labels,
            state=WorkloadStatusStateEnum.INACTIVE,
        ),
    )
    state.delete.side_effect = lambda name: None
    assert not state.serve._stop_model_instance(instance)
    assert instance.id in state.serve._assigned_ports


def test_managed_unlisted_pending_object_cannot_be_confirmed_stopped(managed):
    state = managed
    instance = _running_managed(state)
    pending = list(state.runtime)
    for item in pending:
        item.state = WorkloadStatusStateEnum.PENDING
    state.runtime.clear()
    state.get.side_effect = lambda name: next(
        (item for item in pending if item.name == name), None
    )
    state.delete.side_effect = lambda name: None
    assert not state.serve._stop_model_instance(instance)
    assert instance.id in state.serve._assigned_ports
    state.client.model_instances.report_workloads.assert_not_called()


def test_managed_unlisted_pending_object_is_not_created_twice(managed):
    state = managed
    instance = _running_managed(state)
    pending = list(state.runtime)
    for item in pending:
        item.state = WorkloadStatusStateEnum.PENDING
    state.runtime.clear()
    state.get.side_effect = lambda name: next(
        (item for item in pending if item.name == name), None
    )
    local = _local_manifest(state)
    assert state.serve._start_managed_workloads(instance, local, _managed_plans(local))
    state.create.assert_not_called()


@pytest.mark.parametrize(
    "state_name",
    [ModelInstanceWorkloadStateEnum.STOPPED, ModelInstanceWorkloadStateEnum.RUNNING],
)
def test_managed_missing_started_or_stopped_task_is_not_recreated_in_the_same_generation(
    managed, state_name
):
    state = managed
    for item in state.instance.distributed_servers.workloads:
        item.state = state_name
    local = _local_manifest(state)
    assert not state.serve._start_managed_workloads(
        state.instance, local, _managed_plans(local)
    )
    state.create.assert_not_called()


@pytest.mark.parametrize("outcome", ["error", "none"])
def test_managed_unknown_runtime_snapshot_retains_resources(managed, outcome):
    state = managed
    instance = _running_managed(state)
    if outcome == "error":
        state.list.side_effect = OSError("runtime unavailable")
    else:
        state.list.side_effect = None
        state.list.return_value = None
    assert not state.serve._stop_model_instance(instance)
    assert instance.id in state.serve._assigned_ports
    state.client.model_instances.report_workloads.assert_not_called()


@pytest.mark.parametrize("label", ["model-instance-id", "restart-count", "worker-id"])
def test_managed_runtime_identity_mismatch_never_deletes_a_foreign_workload(
    managed, label
):
    state = managed
    instance = _running_managed(state)
    state.runtime[0].labels = dict(state.runtime[0].labels)
    state.runtime[0].labels[f"gpustack.ai/{label}"] = "999"
    assert not state.serve._stop_model_instance(instance)
    state.delete.assert_not_called()
    assert instance.id in state.serve._assigned_ports


@pytest.mark.parametrize(
    "worker_state,unreachable",
    [
        (WorkerStateEnum.NOT_READY, False),
        (WorkerStateEnum.UNREACHABLE, False),
        (WorkerStateEnum.READY, True),
        (WorkerStateEnum.MAINTENANCE, True),
    ],
)
def test_managed_unavailable_worker_cannot_confirm_stop(
    managed, worker_state, unreachable
):
    state = managed
    instance = _running_managed(state)
    state.client.workers.get.return_value = SimpleNamespace(
        state=worker_state, unreachable=unreachable
    )
    assert not state.serve._stop_model_instance(instance)
    assert not state.runtime
    assert instance.id in state.serve._assigned_ports
    state.client.model_instances.report_workloads.assert_not_called()


def test_managed_failed_stop_report_retains_ports_until_the_server_acknowledges(
    managed,
):
    state = managed
    instance = _running_managed(state)
    normal_report = state.client.model_instances.report_workloads.side_effect
    state.client.model_instances.report_workloads.side_effect = ReadTimeout(
        "report unavailable"
    )
    assert not state.serve._stop_model_instance(instance)
    assert not state.runtime
    assert instance.id in state.serve._assigned_ports
    state.client.model_instances.report_workloads.side_effect = normal_report
    assert state.serve._stop_model_instance(instance)
    assert instance.id not in state.serve._assigned_ports


def test_managed_worker_never_increments_restart_count_or_schedules_recovery(managed):
    state = managed
    state.instance.state = ModelInstanceStateEnum.ERROR
    instance = _running_managed(state)
    state.serve._restart_error_model_instance(instance)
    assert state.instance.restart_count == 3
    assert state.instance.state == ModelInstanceStateEnum.ERROR
    state.serve._update_model_instance.assert_not_called()


def test_managed_plans_cannot_escape_the_workers_runtime_namespace(managed):
    state = managed
    local = _local_manifest(state)
    plans = _managed_plans(local)
    plans[0].namespace = "another-namespace"
    assert not state.serve._start_managed_workloads(state.instance, local, plans)
    state.create.assert_not_called()
    state.client.model_instances.report_workloads.assert_not_called()


@pytest.mark.parametrize(
    'worker_state', [WorkerStateEnum.READY, WorkerStateEnum.MAINTENANCE]
)
def test_managed_confirmed_stop_does_not_generate_more_watch_updates(
    managed, worker_state
):
    state = managed
    state.instance.state = ModelInstanceStateEnum.ERROR
    instance = _running_managed(state)
    state.client.workers.get.return_value = SimpleNamespace(
        state=worker_state, unreachable=False
    )
    assert state.serve._stop_model_instance(instance)
    assert state.client.model_instances.report_workloads.call_count == 1
    for _ in range(3):
        state.serve._dispatch_model_instance_event(
            Event(type=EventType.UPDATED, data=state.instance)
        )
    assert state.client.model_instances.report_workloads.call_count == 1


def test_managed_lost_stop_reply_can_be_confirmed_without_another_update(managed):
    state = managed
    instance = _running_managed(state)
    normal_report = state.client.model_instances.report_workloads.side_effect

    def report(**kwargs):
        normal_report(**kwargs)
        raise ReadTimeout("stop reply lost after commit")

    state.client.model_instances.report_workloads.side_effect = report
    assert not state.serve._stop_model_instance(instance)
    assert state.serve._stop_model_instance(instance)
    assert state.client.model_instances.report_workloads.call_count == 1
    assert instance.id not in state.serve._assigned_ports


@pytest.mark.parametrize("cached", [False, True])
def test_pending_managed_cleanup_retries_locally_while_server_reads_fail(
    managed, cached
):
    state = managed
    instance = _running_managed(state)
    normal_delete = state.delete.side_effect
    state.delete.side_effect = lambda name: None
    assert not state.serve._stop_model_instance(instance)
    state.delete.side_effect = normal_delete
    state.client.model_instances.get.side_effect = ReadTimeout("server unavailable")
    state.client.workers.get.side_effect = ReadTimeout("server unavailable")
    if cached:
        state.serve.sync_model_instances_state()
        _finish_managed_tasks(state)
    else:
        state.client.model_instances.list.side_effect = ReadTimeout(
            "server unavailable"
        )
        with pytest.raises(ReadTimeout):
            state.serve.sync_model_instances_state()
        _finish_managed_tasks(state)
    assert not state.runtime
    assert instance.id in state.serve._assigned_ports
    state.client.model_instances.report_workloads.assert_not_called()


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
    clientset.model_instances.get.return_value = _instance()
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
    yield manager, clientset
    if manager._managed_task_executor is not None:
        manager._managed_task_executor.shutdown(wait=True)
    if manager._managed_stop_executor is not None:
        manager._managed_stop_executor.shutdown(wait=True)


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
    clientset.model_instances.report_workloads.assert_not_called()
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
    clientset.model_instances.report_workloads.assert_not_called()
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
    clientset.model_instances.report_workloads.assert_not_called()


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


@pytest.mark.parametrize("clock_offset", [-120, 0, 120])
@pytest.mark.parametrize("stop_requested", [False, True])
def test_managed_stop_follows_server_intent_without_worker_clock(
    managed, clock_offset, stop_requested
):
    state = managed
    _running_managed(state)
    state.instance.draining_since = datetime.now(timezone.utc) + timedelta(
        seconds=clock_offset
    )
    state.instance.distributed_servers.stop_requested = stop_requested
    state.serve._sync_managed_model_instance(state.instance)
    _finish_managed_tasks(state)
    if stop_requested:
        assert not state.runtime
        assert all(
            item.state == ModelInstanceWorkloadStateEnum.STOPPED
            for item in _local_manifest(state)
        )
    else:
        state.delete.assert_not_called()
        assert len(state.runtime) == 2
        assert state.instance.id in state.serve._assigned_ports


def test_managed_stop_resumes_when_the_same_worker_reconnects(managed):
    state = managed
    _running_managed(state)
    state.instance.state = ModelInstanceStateEnum.UNREACHABLE
    state.instance.draining_since = datetime.now(timezone.utc)
    state.instance.distributed_servers.stop_requested = True
    for item in state.instance.distributed_servers.workloads:
        if item.worker_id != state.serve._worker_id:
            continue
        item.state = ModelInstanceWorkloadStateEnum.UNKNOWN
        for placement in item.dp_rank_placements:
            if placement.node_rank == 0:
                placement.ready = False
    state.client.workers.get.return_value = SimpleNamespace(
        state=WorkerStateEnum.UNREACHABLE, unreachable=True
    )
    state.serve._sync_managed_model_instance(state.instance)
    _finish_managed_tasks(state)
    assert state.instance.id in state.serve._assigned_ports
    assert all(
        item.state == ModelInstanceWorkloadStateEnum.UNKNOWN
        for item in _local_manifest(state)
    )
    state.client.workers.get.return_value = SimpleNamespace(
        state=WorkerStateEnum.READY, unreachable=False
    )
    state.serve._sync_managed_model_instance(state.instance)
    _finish_managed_tasks(state)
    assert state.instance.id not in state.serve._assigned_ports
    assert all(
        item.state == ModelInstanceWorkloadStateEnum.STOPPED
        for item in _local_manifest(state)
    )
    assert state.instance.restart_count == 3
    state.create.assert_not_called()


@pytest.mark.parametrize(
    'blocked', ['terminating', 'runtime_unknown', 'preparer', 'server']
)
@pytest.mark.parametrize(
    'worker_state', [WorkerStateEnum.READY, WorkerStateEnum.MAINTENANCE]
)
def test_guarded_legacy_stop_holds_claims_until_proved_and_reported(
    managed, monkeypatch, blocked, worker_state
):
    monkeypatch.setattr(
        'gpustack.worker.serve_manager.psutil.process_iter', lambda *args, **kwargs: []
    )
    from gpustack.schemas.models import ModelInstanceLegacyStop, ModelInstanceStopReport

    s = managed
    mi = s.instance
    mi.distributed_servers.workloads = None
    mi.distributed_servers.orchestration_mode = "legacy"
    mi.draining_since = datetime.now(timezone.utc)
    mi.distributed_servers.legacy_stop = ModelInstanceLegacyStop(
        token=uuid4(),
        restart_count=mi.restart_count,
        worker_ids=[1, 2],
        worker_principal_ids={1: 10, 2: 20},
        confirmed_worker_ids=[
            worker_id for worker_id in [1, 2] if worker_id != s.serve._worker_id
        ],
    )
    name = mi.get_deployment_metadata(s.serve._worker_id).name
    s.serve._model_instance_by_instance_id[mi.id] = mi
    s.serve._assigned_ports[mi.id] = {40000}
    s.client.workers.get.return_value = SimpleNamespace(
        state=worker_state, unreachable=False
    )
    s.client.model_instances.confirm_stop.reset_mock()

    def report(*, id, report):
        model_update = report
        assert isinstance(model_update, ModelInstanceStopReport)
        mi.distributed_servers.legacy_stop.confirmed_worker_ids = [1, 2]
        return mi.model_copy(deep=True)

    s.client.model_instances.confirm_stop.side_effect = report
    process = MagicMock()
    process.is_alive.return_value = blocked == 'preparer'
    s.serve._provisioning_processes[mi.id] = process
    monkeypatch.setattr(
        'gpustack.worker.serve_manager.terminate_process_tree', MagicMock()
    )
    if blocked == 'terminating':
        s.runtime.append(SimpleNamespace(name=name))
        s.delete.side_effect = lambda name: None
    elif blocked == 'runtime_unknown':
        s.get.side_effect = RuntimeError('runtime unavailable')
    elif blocked == 'server':
        s.client.model_instances.confirm_stop.side_effect = ReadTimeout('reply lost')
    assert s.serve._stop_legacy_guarded_model_instance(mi, False) is False
    assert mi.id in s.serve._assigned_ports
    if blocked != 'server':
        s.client.model_instances.confirm_stop.assert_not_called()
    s.runtime.clear()
    process.is_alive.return_value = False
    s.get.side_effect = lambda name: None
    s.client.model_instances.confirm_stop.side_effect = report
    assert s.serve._stop_legacy_guarded_model_instance(mi, False) is True
    assert mi.id not in s.serve._assigned_ports


@pytest.mark.parametrize('kind', ['unknown', 'own_label', 'tracked_old'])
def test_legacy_stop_proves_all_known_names_and_refuses_ambiguous_objects(
    managed, kind, monkeypatch
):
    monkeypatch.setattr(
        'gpustack.worker.serve_manager.psutil.process_iter', lambda *args, **kwargs: []
    )
    from gpustack.schemas.models import ModelInstanceLegacyStop

    s = managed
    mi = s.instance
    mi.distributed_servers.workloads = None
    mi.distributed_servers.orchestration_mode = "legacy"
    mi.distributed_servers.legacy_stop = ModelInstanceLegacyStop(
        token=uuid4(),
        restart_count=mi.restart_count,
        worker_ids=[1, 2],
        worker_principal_ids={1: 10, 2: 20},
        confirmed_worker_ids=[1, 2],
    )
    s.serve._assigned_ports[mi.id] = {40000}
    s.client.workers.get.return_value = SimpleNamespace(
        state=WorkerStateEnum.READY, unreachable=False
    )
    s.client.model_instances.list.return_value = SimpleNamespace(items=[mi])
    old_record = mi.model_copy(update={'name': 'old-task'})
    name = (
        old_record.get_deployment_metadata(s.serve._worker_id).name
        if kind == 'tracked_old'
        else 'unmapped-task'
    )
    labels = (
        {'gpustack.ai/model-instance-id': str(mi.id)} if kind == 'own_label' else {}
    )
    s.runtime.append(SimpleNamespace(name=name, labels=labels))
    if kind == 'tracked_old':
        s.serve._model_instance_by_instance_id[mi.id] = old_record
        s.delete.side_effect = lambda name: None
    result = s.serve._stop_legacy_guarded_model_instance(mi, False)
    assert result is (kind == 'own_label')
    assert (mi.id in s.serve._assigned_ports) is (kind != 'own_label')
    if kind in ('tracked_old', 'own_label'):
        assert name in [call.args[0] for call in s.delete.call_args_list]
    else:
        s.delete.assert_not_called()


@pytest.mark.parametrize(
    'kind',
    [
        'owned',
        'reused',
        'missing',
        'unknown',
        'still_alive',
        'infra_manager',
        'infra_download',
        'unknown_fork',
    ],
)
def test_legacy_stop_after_worker_restart_checks_preparer_identity(
    managed, monkeypatch, kind
):
    s = managed
    mi = s.instance
    s.serve._provisioning_processes.clear()
    if mi.worker_id == s.serve._worker_id:
        mi.pid = 81
    else:
        next(
            w
            for w in mi.distributed_servers.subordinate_workers
            if w.worker_id == s.serve._worker_id
        ).pid = 81
    monkeypatch.setattr(
        'gpustack.worker.serve_manager.psutil.pid_exists',
        lambda pid: kind
        not in ('missing', 'infra_manager', 'infra_download', 'unknown_fork'),
    )
    title = f'gpustack_model_instance_{mi.id}'
    process = MagicMock()
    process.pid = 81
    process.info = {'pid': 81, 'name': title, 'cmdline': [title]}
    process.cmdline.return_value = [title]
    process.children.return_value = []
    alive = [True]
    process.is_running.side_effect = lambda: alive[0]
    if kind == 'reused':
        process.info.update(name='unrelated', cmdline=['unrelated'])
        process.cmdline.return_value = ['unrelated']
    elif kind == 'unknown_fork':
        process.pid = 82
        process.info.update(
            pid=82, name='python', cmdline=['python', '--multiprocessing-fork']
        )
    elif kind.startswith('infra_'):
        infra_title = (
            'gpustack_model_file_manager'
            if kind == 'infra_manager'
            else 'gpustack_model_file_download'
        )
        process.pid = 82
        process.info.update(pid=82, name=infra_title, cmdline=[infra_title])
    elif kind == 'unknown':
        process.info.update(name=title[:15], cmdline=None)
    monkeypatch.setattr(
        'gpustack.worker.serve_manager.psutil.process_iter',
        lambda *args, **kwargs: [] if kind == 'missing' else [process],
    )
    terminate = MagicMock(
        side_effect=lambda proc: alive.__setitem__(0, kind == 'still_alive')
    )
    monkeypatch.setattr('gpustack.worker.serve_manager.terminate_process', terminate)
    assert s.serve._legacy_preparers_absent(mi) is (
        kind in ('owned', 'missing', 'infra_manager', 'infra_download')
    )
    if kind in (
        'reused',
        'missing',
        'unknown',
        'infra_manager',
        'infra_download',
        'unknown_fork',
    ):
        terminate.assert_not_called()


@pytest.mark.parametrize(
    'kind',
    [
        'named',
        'recycled_pid',
        'tracked_fork',
        'dead_tracked_fork',
        'child_fork',
        'infra',
    ],
)
def test_legacy_stop_excludes_proven_other_preparers(managed, monkeypatch, kind):
    s = managed
    mi = s.instance
    if mi.worker_id == s.serve._worker_id:
        mi.pid = 81
    else:
        next(
            worker
            for worker in mi.distributed_servers.subordinate_workers
            if worker.worker_id == s.serve._worker_id
        ).pid = 81
    other_id = mi.id + 100
    other = MagicMock()
    other.pid = 81 if kind == 'recycled_pid' else 82
    other.info = {
        'pid': other.pid,
        'name': f'gpustack_model_instance_{other_id}'[:15],
        'cmdline': [f'gpustack_model_instance_{other_id}'],
        'ppid': 1,
    }
    processes = [other]
    if kind in ('tracked_fork', 'dead_tracked_fork'):
        other.info.update(name='python', cmdline=['python', '--multiprocessing-fork'])
        handle = MagicMock(pid=other.pid)
        handle.is_alive.return_value = kind == 'tracked_fork'
        s.serve._provisioning_processes[other_id] = handle
    elif kind == 'child_fork':
        child = MagicMock(pid=83)
        child.info = {
            'pid': 83,
            'name': 'python',
            'ppid': other.pid,
            'cmdline': ['python', '--multiprocessing-fork'],
        }
        processes.insert(0, child)
    elif kind == 'infra':
        other.info['cmdline'] = ['gpustack_model_file_download']
    monkeypatch.setattr(
        'gpustack.worker.serve_manager.psutil.process_iter',
        lambda *args, **kwargs: processes,
    )
    monkeypatch.setattr(
        'gpustack.worker.serve_manager.psutil.pid_exists',
        lambda pid: any(process.pid == pid for process in processes),
    )
    terminate = MagicMock()
    monkeypatch.setattr('gpustack.worker.serve_manager.terminate_process', terminate)
    assert s.serve._legacy_preparers_absent(mi) is (kind != 'dead_tracked_fork')
    terminate.assert_not_called()


def test_stale_start_snapshot_cannot_launch_after_legacy_stop_intent(
    manager, monkeypatch
):
    from gpustack.schemas.models import ModelInstanceLegacyStop

    serve, client = manager
    stale = _instance()
    current = stale.model_copy(deep=True)
    current.distributed_servers.legacy_stop = ModelInstanceLegacyStop(
        token=uuid4(),
        restart_count=0,
        worker_ids=[1, 2],
        worker_principal_ids={1: 10, 2: 20},
    )
    client.model_instances.get.return_value = current
    stop = MagicMock()
    monkeypatch.setattr(serve, '_stop_model_instance', stop)
    serve._start_model_instance(stale)
    assert stale.id not in serve._provisioning_processes
    assert stale.id not in serve._assigned_ports
    stop.assert_called_once()


def _sglang_local_components(state):
    worker_id = state.serve._worker_id
    primary = worker_id == state.instance.worker_id
    items = [
        ModelInstanceWorkload(
            name=state.instance.name if primary else state.instance.name + "-f0",
            worker_id=worker_id,
            state="pending",
            component={
                "kind": "sglang_engine",
                "api_port": 40020 if primary else None,
                "dp_size": 4 if primary else None,
                "image": "selected-engine-image",
                "router_version": "0.3.2",
            },
        )
    ]
    if primary:
        items.append(
            ModelInstanceWorkload(
                name=f"{state.instance.name}-r{state.instance.restart_count}-smg-a0",
                worker_id=worker_id,
                state="pending",
                component={
                    "kind": "sglang_smg",
                    "api_port": 40022,
                    "metrics_port": 40023,
                },
            )
        )
    return items


@pytest.fixture
def older_engine(managed):
    state = managed
    local = _sglang_local_components(state)
    state.instance.distributed_servers.workloads = local
    engine = local[0]
    labels = {
        **state.serve._managed_workload_labels(state.instance),
        **state.serve._component_workload_labels(engine),
        'gpustack.ai/restart-count': str(state.instance.restart_count - 1),
    }
    state.runtime.append(
        SimpleNamespace(
            name=engine.name, labels=labels, state=WorkloadStatusStateEnum.RUNNING
        )
    )
    return state


@pytest.mark.parametrize('listed', [False, True])
def test_owned_older_engine_cleanup_allows_the_next_generation_to_start(
    older_engine, listed
):
    state = older_engine
    if not listed:
        state.list.side_effect = lambda: []
    local = _local_manifest(state)
    assert not state.serve._start_managed_workloads(
        state.instance, local, _managed_plans(local)
    )
    state.create.assert_not_called()
    assert not state.runtime
    assert (state.instance.id, state.instance.restart_count) not in (
        state.serve._managed_stopping
    )
    assert state.instance.id not in state.serve._assigned_ports
    assert all(
        item.state == ModelInstanceWorkloadStateEnum.STOPPED
        for item in _local_manifest(state)
    )

    state.instance.restart_count += 1
    state.instance.distributed_servers.workloads = []
    local = _sglang_local_components(state)
    assert state.serve._start_managed_workloads(
        state.instance, local, _managed_plans(local)
    )
    assert state.create.call_count == len(local)
    assert all(
        status.labels['gpustack.ai/restart-count'] == str(state.instance.restart_count)
        for status in state.runtime
    )


def test_owned_older_engine_retains_resources_until_runtime_deletion(older_engine):
    state = older_engine
    delete = state.delete.side_effect
    state.delete.side_effect = lambda name: None
    assert not state.serve._stop_model_instance(state.instance)
    state.delete.assert_called_once_with(state.runtime[0].name)
    assert state.instance.id in state.serve._assigned_ports
    assert all(
        item.state != ModelInstanceWorkloadStateEnum.STOPPED
        for item in _local_manifest(state)
    )
    state.delete.side_effect = delete
    state.serve._retry_managed_stops()
    _finish_managed_tasks(state)
    assert not state.runtime
    assert state.instance.id not in state.serve._assigned_ports


def test_owned_older_engine_is_never_adopted_as_current(older_engine):
    state = older_engine
    with pytest.raises(RuntimeError, match='does not match this instance generation'):
        state.serve._managed_runtime_workloads(state.instance)
    state.delete.assert_not_called()


@pytest.mark.parametrize(
    'labels',
    [
        {'model-instance-id': '999'},
        {'worker-id': '999'},
        {'restart-count': '4'},
        {'restart-count': '-1'},
        {'restart-count': 'invalid'},
        {'restart-count': None},
        {'component-kind': 'sglang_smg'},
        {'component-kind': None},
        {'restart-count': '3', 'component-attempt': '1'},
    ],
)
def test_older_engine_cleanup_preserves_runtime_identity_fences(older_engine, labels):
    state = older_engine
    for key, value in labels.items():
        name = f'gpustack.ai/{key}'
        if value is None:
            state.runtime[0].labels.pop(name)
        else:
            state.runtime[0].labels[name] = value
    assert not state.serve._stop_model_instance(state.instance)
    state.delete.assert_not_called()
    assert state.instance.id in state.serve._assigned_ports


@pytest.mark.parametrize(
    'change',
    [
        'unavailable',
        'missing',
        'identity',
        'generation',
        'cluster',
        'worker',
        'unregistered',
        'attempt',
    ],
)
def test_older_engine_cleanup_requires_current_server_authority(older_engine, change):
    state = older_engine
    current = state.instance.model_copy(deep=True)
    if change in ('unavailable', 'missing'):
        error = (
            ReadTimeout('unavailable')
            if change == 'unavailable'
            else NotFoundException()
        )
        state.client.model_instances.get.side_effect = error
    else:
        if change == 'identity':
            current.id += 1
        elif change == 'generation':
            current.restart_count += 1
        elif change == 'cluster':
            current.cluster_id += 1
        elif change == 'worker':
            current.worker_id = 999
            current.distributed_servers.subordinate_workers = []
        elif change == 'unregistered':
            current.distributed_servers.workloads = []
        else:
            current.distributed_servers.workloads[0].component.attempt += 1
        state.client.model_instances.get.side_effect = lambda **kwargs: current
    assert not state.serve._stop_model_instance(state.instance)
    state.delete.assert_not_called()
    assert state.instance.id in state.serve._assigned_ports


def test_sglang_component_registration_precedes_both_creates(managed):
    local = _sglang_local_components(managed)
    managed.instance.distributed_servers.workloads = []
    managed.instance.distributed_servers.orchestration_mode = "managed"
    assert managed.serve._start_managed_workloads(
        managed.instance, local, _managed_plans(local)
    )
    assert managed.events[0] == 'report'
    assert managed.create.call_count == len(local)
    for call in managed.create.call_args_list:
        labels = call.args[0].labels
        assert labels['gpustack.ai/component-kind'] in ('sglang_engine', 'sglang_smg')
        assert labels['gpustack.ai/component-attempt'] == '0'


@pytest.mark.parametrize('change', ['generation', 'attempt', 'stop', 'cluster'])
def test_sglang_create_return_rechecks_authoritative_identity(managed, change):
    state = managed
    if change == 'attempt' and state.serve._worker_id != state.instance.worker_id:
        pytest.skip('SMG is owned only by the primary Worker')
    local = _sglang_local_components(state)
    state.instance.distributed_servers.workloads = []
    state.instance.distributed_servers.orchestration_mode = "managed"
    state.serve._stop_model_instance = MagicMock(return_value=False)
    original = state.create.side_effect

    def create(plan):
        original(plan)
        if change == 'generation':
            state.instance.restart_count += 1
        elif change == 'cluster':
            state.instance.cluster_id += 1
        elif change == 'attempt':
            state.instance.distributed_servers.workloads[0].component = (
                state.instance.distributed_servers.workloads[0].component.model_copy()
            )
            state.instance.distributed_servers.workloads[1].component.attempt = 1
            state.instance.distributed_servers.workloads[1].name = (
                f'{state.instance.name}-r3-smg-a1'
            )
        else:
            state.instance.draining_since = datetime.now(timezone.utc)

    state.create.side_effect = create
    accepted = state.serve._start_managed_workloads(
        state.instance.model_copy(deep=True), local, _managed_plans(local)
    )
    assert accepted == (change == 'attempt')
    assert state.create.call_count == 1
    if change == 'attempt':
        state.serve._stop_model_instance.assert_not_called()
        assert not state.serve._managed_stopping
    else:
        state.serve._stop_model_instance.assert_called()


@pytest.mark.parametrize('reason', ['registration', 'runtime_unknown'])
def test_sglang_failed_or_unknown_registration_never_creates(managed, reason):
    local = _sglang_local_components(managed)
    managed.instance.distributed_servers.workloads = []
    managed.instance.distributed_servers.orchestration_mode = "managed"
    if reason == 'registration':
        managed.client.model_instances.report_workloads.side_effect = ConflictException(
            'stale'
        )
    else:
        managed.list.return_value = None
        managed.list.side_effect = None
    assert not managed.serve._start_managed_workloads(
        managed.instance, local, _managed_plans(local)
    )
    managed.create.assert_not_called()


def test_managed_sglang_starts_preparer_without_waiting_for_peer_health(
    managed, monkeypatch
):
    state = managed
    mi = state.instance
    mi.distributed_servers.workloads = []
    mi.distributed_servers.orchestration_mode = "managed"
    model = new_model(
        1,
        'sg',
        huggingface_repo_id='org/sg',
        backend='SGLang',
        backend_parameters=['--dp=4'],
    )
    state.serve._get_model = lambda current: model
    coordinator = MagicMock()
    coordinator.availability.return_value = [
        SimpleNamespace(
            worker_id=worker_id,
            generation=uuid4(),
            claim_version=None,
            pending_cleanup=False,
            port_range="40000-40063",
            unavailable_ports=[],
        )
        for worker_id in (1, 2)
    ]

    def reserve(instance_id, claim):
        state.serve._port_reservations.model_claims[instance_id] = (
            str(claim.token),
            set(claim.ports),
        )
        return SimpleNamespace(accepted=True)

    coordinator.reserve.side_effect = reserve
    state.serve._distributed_port_client = coordinator
    if state.serve._worker_id == mi.worker_id:
        mi.port, mi.ports, mi.named_ports = None, None, None
    else:
        mi.port = 40020
        mi.ports = [40020, 40021, 40022, 40023]
        mi.named_ports = {
            name: PortBand(base=port)
            for name, port in zip(
                (
                    '_sglang_engine_api',
                    '_sglang_connecting',
                    '_sglang_smg_api',
                    '_sglang_smg_metrics',
                ),
                mi.ports,
            )
        }

    def persist(id, **fields):
        for name, value in fields.items():
            setattr(state.instance, name, value)

    state.serve._update_model_instance.side_effect = persist
    receiver, sender = MagicMock(), MagicMock()
    receiver.poll.return_value = False
    monkeypatch.setattr(
        'gpustack.worker.serve_manager.multiprocessing.Pipe',
        MagicMock(return_value=(receiver, sender)),
    )
    state.serve._sync_managed_model_instance(mi)
    assert (mi.id, mi.restart_count) in state.serve._managed_launches
    assert len(state.serve._assigned_ports[mi.id]) == 4
    assert state.serve._provisioning_processes[mi.id].start.called
    state.create.assert_not_called()
    assert state.client.model_instances.report_workloads.call_count == 0


def test_sglang_component_runtime_labels_fail_closed(managed):
    state = managed
    local = _sglang_local_components(state)
    state.instance.distributed_servers.workloads = local
    state.instance.distributed_servers.orchestration_mode = (
        "managed"
        if state.instance.distributed_servers.workloads is not None
        else "legacy"
    )
    item = local[0]
    item.state = 'running'
    if item.component.api_port is not None:
        item.component.ready = True
    state.runtime = [
        SimpleNamespace(
            name=item.name,
            state=WorkloadStatusStateEnum.RUNNING,
            labels={
                **state.serve._managed_workload_labels(state.instance),
                'gpustack.ai/component-kind': 'wrong',
                'gpustack.ai/component-attempt': '0',
            },
        )
    ]
    observed = state.serve._observe_managed_workloads(state.instance)
    assert observed[0].state == ModelInstanceWorkloadStateEnum.UNKNOWN
    assert observed[0].component.image == 'selected-engine-image'
    assert observed[0].component.router_version == '0.3.2'
    assert not observed[0].component.ready
    state.create.assert_not_called()


@pytest.fixture
def smg(managed, monkeypatch):
    s = managed
    s.instance.state = ModelInstanceStateEnum.RUNNING
    s.instance.port = 40020
    s.instance.ports = [40020, 40021, 40022, 40023]
    s.instance.distributed_servers.workloads = [
        ModelInstanceWorkload(
            name=f'engine-{worker}',
            worker_id=worker,
            state='running',
            component=dict(
                kind='sglang_engine',
                image='mirror/engine:v1',
                router_version='0.3.2',
                **(dict(api_port=40020, dp_size=2, ready=True) if worker == 1 else {}),
            ),
        )
        for worker in (1, 2)
    ] + [
        ModelInstanceWorkload(
            name=f'{s.instance.name}-r3-smg-a0',
            worker_id=1,
            state='running',
            component=dict(
                kind='sglang_smg', api_port=40022, metrics_port=40023, ready=True
            ),
        )
    ]
    s.instance.distributed_servers.orchestration_mode = (
        "managed" if s.instance.distributed_servers.workloads is not None else "legacy"
    )
    s.model = new_model(
        1,
        'sg',
        huggingface_repo_id='org/sg',
        backend='SGLang',
        backend_parameters=['--dp=2'],
    )
    s.client.models.get.return_value = s.model
    s.serve._get_model = lambda mi: s.model
    s.client.workers.get.return_value.ip = '10.0.0.1'
    s.client.workers.get.return_value.me = True
    s.serve._sync_managed_rank_health = MagicMock()
    s.serve._launch_model_instance = MagicMock()
    s.serve._prepare_managed_model_instance = MagicMock()
    s.serve._hold_managed_instance(s.instance)
    for item in _local_manifest(s):
        s.runtime.append(
            SimpleNamespace(
                name=item.name,
                labels={
                    **s.serve._managed_workload_labels(s.instance),
                    **s.serve._component_workload_labels(item),
                },
                state=WorkloadStatusStateEnum.RUNNING,
            )
        )
    monkeypatch.setattr(
        'gpustack.worker.serve_manager.transform_workload_plan',
        lambda config, plan, fallback=None: plan,
        raising=False,
    )
    return s


@pytest.mark.parametrize(
    'worker_state', [WorkerStateEnum.READY, WorkerStateEnum.MAINTENANCE]
)
def test_smg_stop_keeps_engine_and_port_claims(smg, worker_state):
    s = smg
    s.client.workers.get.return_value.state = worker_state
    engine = s.instance.distributed_servers.workloads[0].model_dump()
    item = s.instance.distributed_servers.workloads[-1]
    item.state = ModelInstanceWorkloadStateEnum.FAILED
    item.component.ready = False
    item.component.stop_requested = True
    s.serve._sync_managed_model_instance(s.instance)
    _finish_managed_tasks(s)
    assert s.instance.restart_count == 3
    assert (
        next(
            w
            for w in s.instance.distributed_servers.workloads
            if w.name == engine['name']
        ).model_dump()
        == engine
    )
    assert s.instance.id in s.serve._assigned_ports
    assert s.create.call_count == 0
    assert all(not c.args[0].startswith('engine') for c in s.delete.call_args_list)
    if s.serve._worker_id == 1:
        assert (
            s.instance.distributed_servers.workloads[-1].state
            == ModelInstanceWorkloadStateEnum.STOPPED
        )
        assert len(s.runtime) == 1
    else:
        s.delete.assert_not_called()


@pytest.mark.parametrize('public_model', [False, True])
def test_smg_server_pending_attempt_creates_only_cpu_and_is_adopted(smg, public_model):
    s = smg
    if public_model:
        from gpustack.schemas.models import ModelPublic

        s.model = ModelPublic.model_validate(
            {
                **s.model.model_dump(),
                'created_at': datetime.now(timezone.utc),
                'updated_at': datetime.now(timezone.utc),
            }
        )
        s.client.models.get.return_value = s.model
    item = s.instance.distributed_servers.workloads[-1]
    s.runtime[:] = [w for w in s.runtime if w.name.startswith('engine')]
    item.name = f'{s.instance.name}-r3-smg-a1'
    item.state = ModelInstanceWorkloadStateEnum.PENDING
    item.component.attempt = 1
    item.component.ready = False
    s.instance.state = ModelInstanceStateEnum.STARTING
    engine = s.instance.distributed_servers.workloads[0].model_dump()
    s.serve._sync_managed_model_instance(s.instance)
    _finish_managed_tasks(s)
    s.serve._sync_managed_model_instance(s.instance)
    _finish_managed_tasks(s)
    assert s.create.call_count == int(s.serve._worker_id == 1)
    s.delete.assert_not_called()
    s.serve._launch_model_instance.assert_not_called()
    s.serve._prepare_managed_model_instance.assert_not_called()
    assert (
        next(
            w
            for w in s.instance.distributed_servers.workloads
            if w.name == engine['name']
        ).model_dump()
        == engine
    )
    if s.create.called:
        plan = s.create.call_args.args[0]
        assert plan.name == item.name
        assert plan.containers[0].image == 'mirror/engine:v1'
        assert plan.labels['gpustack.ai/component-attempt'] == '1'


@pytest.mark.parametrize('unknown', ['list', 'lookup', 'worker', 'delete_pending'])
def test_smg_stop_unknown_cannot_confirm_stopped(smg, unknown):
    s = smg
    if s.serve._worker_id != 1:
        return
    item = s.instance.distributed_servers.workloads[-1]
    item.component.stop_requested = True
    if unknown == 'list':
        s.list.return_value = None
        s.list.side_effect = None
    elif unknown == 'lookup':
        s.runtime[:] = [w for w in s.runtime if w.name.startswith('engine')]
        s.get.side_effect = RuntimeError('lookup unavailable')
    elif unknown == 'worker':
        s.client.workers.get.return_value.state = WorkerStateEnum.UNREACHABLE
    else:
        s.delete.side_effect = lambda name: None
    s.serve._sync_managed_model_instance(s.instance)
    _finish_managed_tasks(s)
    assert (
        s.instance.distributed_servers.workloads[-1].state
        != ModelInstanceWorkloadStateEnum.STOPPED
    )
    assert s.instance.id in s.serve._assigned_ports
    assert any(w.name.startswith('engine') for w in s.runtime)


def _pending_cpu(s):
    item = next(
        w
        for w in s.instance.distributed_servers.workloads
        if w.component.kind == 'sglang_smg'
    )
    s.runtime[:] = [w for w in s.runtime if w.name.startswith('engine')]
    item.name = f'{s.instance.name}-r3-smg-a1'
    item.state = ModelInstanceWorkloadStateEnum.PENDING
    item.component.attempt = 1
    item.component.ready = False
    s.instance.state = ModelInstanceStateEnum.STARTING
    s.serve._hold_managed_instance(s.instance)
    return item


@pytest.mark.parametrize(
    'fence',
    [
        'metadata',
        'router_version',
        'policy',
        'stale',
        'public_stale',
        'model_restart',
        'model_stop',
        'model_deleted',
        'drain',
        'stop',
        'engine_unknown',
        'runtime_unknown',
        'engine_labels',
    ],
)
def test_pending_cpu_create_obeys_fresh_model_engine_and_identity(smg, fence):
    s = smg
    if s.serve._worker_id != 1:
        return
    item = _pending_cpu(s)
    engine = next(
        w
        for w in s.instance.distributed_servers.workloads
        if w.component.kind == 'sglang_engine' and w.worker_id == 1
    )
    if fence == 'metadata':
        engine.component.image = None
        engine.component.router_version = None
    elif fence == 'router_version':
        engine.component.router_version = '0.3.1'
    elif fence == 'policy':
        s.model.restart_on_error = False
    elif fence == 'stale':
        s.model.stale = True
    elif fence == 'public_stale':
        from gpustack.schemas.models import ModelPublic

        s.model = ModelPublic.model_validate(
            {
                **s.model.model_dump(),
                'created_at': datetime.now(timezone.utc),
                'updated_at': datetime.now(timezone.utc),
                'stale': True,
            }
        )
        s.client.models.get.return_value = s.model
    elif fence == 'model_restart':
        s.model.restarting_since = datetime.now(timezone.utc)
    elif fence == 'model_stop':
        s.model.replicas = 0
    elif fence == 'model_deleted':
        s.model.deleted_at = datetime.now(timezone.utc)
    elif fence == 'drain':
        s.instance.draining_since = datetime.now(timezone.utc)
    elif fence == 'stop':
        item.component.stop_requested = True
    elif fence == 'engine_unknown':
        engine.state = ModelInstanceWorkloadStateEnum.UNKNOWN
    elif fence == 'runtime_unknown':
        s.list.side_effect = RuntimeError('unavailable')
    else:
        s.runtime[0].labels['gpustack.ai/component-kind'] = 'foreign'
    s.serve._sync_managed_model_instance(s.instance)
    _finish_managed_tasks(s)
    s.create.assert_not_called()
    if fence in ('metadata', 'router_version'):
        assert any(
            (call.kwargs['report'].state_message or '').startswith(
                'SMG recovery is blocked:'
            )
            for call in s.client.model_instances.report_workloads.call_args_list
        )
    assert not any(c.args[0].startswith('engine') for c in s.delete.call_args_list)


@pytest.mark.parametrize(
    'fence', ['attempt', 'stop', 'generation', 'drain', 'response_lost']
)
def test_late_cpu_create_is_cleaned_without_deleting_engine(smg, fence):
    s = smg
    if s.serve._worker_id != 1:
        return
    item = _pending_cpu(s)
    old_name = item.name
    create = s.create.side_effect

    def late(plan):
        create(plan)
        current = next(
            w
            for w in s.instance.distributed_servers.workloads
            if w.component.kind == 'sglang_smg'
        )
        if fence == 'attempt':
            current.name = f'{s.instance.name}-r3-smg-a2'
            current.component.attempt = 2
        elif fence == 'stop':
            current.component.stop_requested = True
        elif fence == 'generation':
            s.instance.restart_count = 4
        elif fence == 'drain':
            s.instance.draining_since = datetime.now(timezone.utc)
        else:
            raise ReadTimeout('create response lost')

    s.create.side_effect = late
    s.serve._sync_managed_model_instance(s.instance)
    _finish_managed_tasks(s)
    assert s.create.call_count == 1
    assert old_name in [c.args[0] for c in s.delete.call_args_list]
    assert all(not c.args[0].startswith('engine') for c in s.delete.call_args_list)
    assert any(w.name == 'engine-1' for w in s.runtime)
    assert s.instance.id in s.serve._assigned_ports


def test_initial_cpu_create_failure_keeps_already_created_engine(smg):
    s = smg
    if s.serve._worker_id != 1:
        return
    local = _local_manifest(s)
    for w in local:
        w.state = ModelInstanceWorkloadStateEnum.PENDING
        w.component.ready = False
    s.instance.state = ModelInstanceStateEnum.STARTING
    s.instance.distributed_servers.workloads = []
    s.instance.distributed_servers.orchestration_mode = "managed"
    s.runtime.clear()
    create = s.create.side_effect

    def fail_cpu(plan):
        if not plan.name.startswith('engine'):
            raise RuntimeError('CPU admission failed')
        create(plan)

    s.create.side_effect = fail_cpu
    assert s.serve._start_managed_workloads(s.instance, local, _managed_plans(local))
    assert any(w.name == 'engine-1' for w in s.runtime)
    assert not s.serve._managed_stopping
    assert all(not c.args[0].startswith('engine') for c in s.delete.call_args_list)
    assert (
        next(
            w
            for w in s.instance.distributed_servers.workloads
            if w.component.kind == 'sglang_smg'
        ).state
        == ModelInstanceWorkloadStateEnum.STOPPED
    )


def test_cpu_stop_waits_for_inflight_create_lifecycle_lock(smg):
    s = smg
    if s.serve._worker_id != 1:
        return
    item = _pending_cpu(s)
    entered, release, stopped = threading.Event(), threading.Event(), threading.Event()
    create = s.create.side_effect

    def blocked(plan):
        entered.set()
        assert release.wait(5)
        create(plan)

    s.create.side_effect = blocked

    def stop():
        snapshot = s.instance.model_copy(deep=True)
        snapshot.distributed_servers.workloads[-1].component.stop_requested = True
        s.serve._retry_smg_stop(snapshot, item)
        stopped.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        launching = executor.submit(s.serve._sync_managed_model_instance, s.instance)
        assert entered.wait(5)
        stopping = executor.submit(stop)
        assert not stopped.is_set()
        release.set()
        launching.result(5)
        stopping.result(5)
    assert stopped.is_set()
    assert any(w.name == 'engine-1' for w in s.runtime)
    assert not any(w.name == item.name for w in s.runtime)


def test_cpu_stop_does_not_block_engine_startup_observation(smg):
    s = smg
    if s.serve._worker_id != 1:
        return
    engine = next(
        w for w in s.instance.distributed_servers.workloads if w.name == 'engine-1'
    )
    engine.state = ModelInstanceWorkloadStateEnum.PENDING
    engine.component.ready = False
    item = s.instance.distributed_servers.workloads[-1]
    item.state = ModelInstanceWorkloadStateEnum.FAILED
    item.component.ready = False
    item.component.stop_requested = True
    s.serve._sync_managed_model_instance(s.instance)
    _finish_managed_tasks(s)
    engine = next(
        w for w in s.instance.distributed_servers.workloads if w.name == 'engine-1'
    )
    assert engine.state == ModelInstanceWorkloadStateEnum.RUNNING
    assert any(w.name == 'engine-1' for w in s.runtime)


def test_worker_restart_adopts_current_engine_and_smg_without_create(smg):
    s = smg
    s.serve._model_instance_by_instance_id.clear()
    s.serve._sync_managed_model_instance(s.instance)
    _finish_managed_tasks(s)
    s.create.assert_not_called()
    s.delete.assert_not_called()
    assert s.instance.id in s.serve._assigned_ports
    s.serve._launch_model_instance.assert_not_called()


def test_engine_failure_stops_all_local_components(smg):
    s = smg
    engine = next(
        w
        for w in s.instance.distributed_servers.workloads
        if w.worker_id == s.serve._worker_id and w.component.kind == 'sglang_engine'
    )
    engine.state = ModelInstanceWorkloadStateEnum.FAILED
    engine.component.ready = False if engine.component.api_port else None
    s.serve._sync_managed_model_instance(s.instance)
    _finish_managed_tasks(s)
    assert not s.runtime
    assert s.instance.id not in s.serve._assigned_ports
    s.create.assert_not_called()


def test_instance_global_error_stops_engine_and_smg(smg):
    s = smg
    s.instance.state = ModelInstanceStateEnum.ERROR
    s.serve._sync_managed_model_instance(s.instance)
    _finish_managed_tasks(s)
    assert not s.runtime
    assert s.instance.id not in s.serve._assigned_ports


def test_cpu_server_read_failure_after_create_preserves_engine_and_retries_cleanup(smg):
    s = smg
    if s.serve._worker_id != 1:
        return
    item = _pending_cpu(s)
    create = s.create.side_effect
    get = s.client.model_instances.get.side_effect

    def lost(plan):
        create(plan)
        s.client.model_instances.get.side_effect = ReadTimeout('authority unavailable')

    s.create.side_effect = lost
    s.serve._sync_managed_model_instance(s.instance)
    _finish_managed_tasks(s)
    assert any(w.name == 'engine-1' for w in s.runtime)
    assert s.serve._smg_stopping
    assert all(
        not call.args[0].startswith('engine') for call in s.delete.call_args_list
    )
    s.client.model_instances.get.side_effect = get
    s.serve._retry_managed_stops()
    _finish_managed_tasks(s)
    assert not s.serve._smg_stopping
    assert (
        next(
            w for w in s.instance.distributed_servers.workloads if w.name == item.name
        ).state
        == ModelInstanceWorkloadStateEnum.STOPPED
    )
    assert s.instance.id in s.serve._assigned_ports


def test_new_cpu_attempt_does_not_adopt_old_runtime_identity(smg):
    s = smg
    if s.serve._worker_id != 1:
        return
    old = s.instance.distributed_servers.workloads[-1].name
    item = s.instance.distributed_servers.workloads[-1]
    item.name = f'{s.instance.name}-r3-smg-a1'
    item.component.attempt = 1
    item.component.ready = False
    item.state = ModelInstanceWorkloadStateEnum.PENDING
    s.instance.state = ModelInstanceStateEnum.STARTING
    s.serve._sync_managed_model_instance(s.instance)
    _finish_managed_tasks(s)
    assert old in [call.args[0] for call in s.delete.call_args_list]
    s.serve._sync_managed_model_instance(s.instance)
    _finish_managed_tasks(s)
    assert s.create.call_count == 1
    assert s.create.call_args.args[0].name == item.name
    assert any(w.name == 'engine-1' for w in s.runtime)


def test_initial_cpu_authority_read_failure_does_not_stop_engine(smg):
    s = smg
    if s.serve._worker_id != 1:
        return
    local = _local_manifest(s)
    for item in local:
        item.state = ModelInstanceWorkloadStateEnum.PENDING
        item.component.ready = False
    s.instance.state = ModelInstanceStateEnum.STARTING
    s.instance.distributed_servers.workloads = []
    s.instance.distributed_servers.orchestration_mode = "managed"
    s.runtime.clear()
    get = s.client.model_instances.get.side_effect
    reads_after_engine = 0

    def read(**kwargs):
        nonlocal reads_after_engine
        if s.create.call_count:
            reads_after_engine += 1
            if reads_after_engine >= 2:
                raise ReadTimeout('authority unavailable before CPU create')
        return get(**kwargs)

    s.client.model_instances.get.side_effect = read
    assert s.serve._start_managed_workloads(s.instance, local, _managed_plans(local))
    assert s.create.call_count == 1
    assert any(w.name == 'engine-1' for w in s.runtime)
    assert not s.serve._managed_stopping
    assert s.serve._smg_stopping


@pytest.mark.parametrize(
    'change',
    [
        'me',
        'me_unknown',
        'worker_id',
        'worker_cluster',
        'worker_ip',
        'policy_argv',
        'backend_version',
    ],
)
def test_cpu_second_authority_fences_binding_and_prepared_inputs(smg, change):
    s = smg
    if s.serve._worker_id != 1:
        return
    _pending_cpu(s)
    worker = s.client.workers.get.return_value
    worker.id, worker.cluster_id, worker.me = 1, s.instance.cluster_id, True
    reads = 0

    def read(*args, **kwargs):
        nonlocal reads
        reads += 1
        if reads == 2:
            if change == 'me':
                worker.me = False
            elif change == 'me_unknown':
                worker.me = None
            elif change == 'worker_id':
                worker.id = 99
            elif change == 'worker_cluster':
                worker.cluster_id = 99
            elif change == 'worker_ip':
                worker.ip = '10.0.0.99'
            elif change == 'policy_argv':
                s.model.backend_parameters.append('--policy=random')
            else:
                s.model.backend_version = '0.5.22'
        return worker

    s.client.workers.get.side_effect = read
    s.serve._sync_managed_model_instance(s.instance)
    _finish_managed_tasks(s)
    s.create.assert_not_called()
    assert any(w.name == 'engine-1' for w in s.runtime)


@pytest.mark.parametrize('me', [False, None, 'missing'])
def test_cpu_requires_self_worker_for_first_create_authority(smg, me):
    s = smg
    if s.serve._worker_id != 1:
        return
    _pending_cpu(s)
    if me == 'missing':
        del s.client.workers.get.return_value.me
    else:
        s.client.workers.get.return_value.me = me
    s.serve._sync_managed_model_instance(s.instance)
    _finish_managed_tasks(s)
    s.create.assert_not_called()
    assert any(w.name == 'engine-1' for w in s.runtime)


def test_parent_late_preparer_failure_cannot_invalidate_registered_engine(smg):
    s = smg
    s.serve._handle_model_instance_launch_failure(
        s.instance, BrokenPipeError('CPU stop closed pipe')
    )
    s.serve._update_model_instance.assert_not_called()
    assert s.instance.id not in s.serve._error_model_instances
    assert s.instance.id not in s.serve._pending_model_instance_errors
    assert all(
        w.state == ModelInstanceWorkloadStateEnum.RUNNING
        for w in s.instance.distributed_servers.workloads
    )


def test_parent_uncertain_late_failure_uses_existing_handoff_fence(smg):
    s = smg
    initial = s.instance.model_copy(deep=True)
    initial.distributed_servers.workloads = []
    initial.distributed_servers.orchestration_mode = "managed"
    s.client.model_instances.get.side_effect = ReadTimeout('authority unavailable')
    s.serve._handle_model_instance_launch_failure(initial, BrokenPipeError('closed'))
    s.serve._update_model_instance.assert_not_called()
    assert s.instance.id not in s.serve._error_model_instances


def test_legacy_stop_reaps_owned_zombie_before_orphan_identity_scan(
    managed, monkeypatch
):
    s = managed
    mi = s.instance
    mi.worker_id = s.serve._worker_id
    mi.pid = 81
    mi.distributed_servers.workloads = None
    mi.distributed_servers.orchestration_mode = "legacy"
    reaped = []
    owned = MagicMock()
    owned.join.side_effect = lambda timeout: reaped.append(True)
    owned.is_alive.return_value = False
    s.serve._provisioning_processes[mi.id] = owned
    zombie = SimpleNamespace(pid=81, info={'pid': 81, 'name': 'python', 'cmdline': []})
    monkeypatch.setattr(
        'gpustack.worker.serve_manager.psutil.process_iter',
        lambda *args, **kwargs: [] if reaped else [zombie],
    )
    monkeypatch.setattr(
        'gpustack.worker.serve_manager.psutil.pid_exists', lambda pid: not reaped
    )
    terminate = MagicMock()
    monkeypatch.setattr('gpustack.worker.serve_manager.terminate_process', terminate)
    s.serve._legacy_stop_other_workloads_known = MagicMock(return_value=True)

    assert s.serve._legacy_stop_runtime_absent(mi, {'legacy-router'}) is True
    owned.join.assert_called_once_with(timeout=0)
    terminate.assert_not_called()


@pytest.mark.parametrize('event_type', [EventType.DELETED, EventType.UPDATED])
def test_ordinary_deleted_manifest_cleans_all_local_rank_tasks(managed, event_type):
    state = managed
    instance = _running_managed(state)
    instance.group_id = None
    expected = {w.name for w in _local_manifest(state)}
    state.client.model_instances.get.side_effect = NotFoundException('instance deleted')
    state.serve._dispatch_model_instance_event(Event(type=event_type, data=instance))
    _finish_managed_tasks(state)
    assert {call.args[0] for call in state.delete.call_args_list} == expected
    assert not state.runtime
    assert instance.id not in state.serve._assigned_ports
    assert instance.id not in state.serve._model_instance_by_instance_id
    state.create.assert_not_called()
    state.client.model_instances.report_workloads.assert_not_called()


def test_hybrid_components_do_not_enter_sglang_reconciliation(managed):
    from tests.utils.model import new_hybrid_workload

    managed.instance.distributed_servers.workloads = [
        ModelInstanceWorkload.model_validate(new_hybrid_workload())
    ]
    managed.instance.distributed_servers.orchestration_mode = "managed"
    assert not ServeManager._has_sglang_components(managed.instance)


@pytest.mark.parametrize('startup', [False, True])
def test_hybrid_health_uses_shared_api_and_optional_inference_probe(managed, startup):
    from tests.utils.model import new_hybrid_workload

    state = managed
    instance = state.instance
    instance.state = ModelInstanceStateEnum.RUNNING
    instance.distributed_servers.workloads = [
        ModelInstanceWorkload.model_validate(
            new_hybrid_workload(worker_id, headless=worker_id == 2, state='running')
        )
        for worker_id in (1, 2)
    ]
    instance.distributed_servers.orchestration_mode = "managed"
    instance.distributed_servers.workloads[0].component.ready = not startup
    model = new_model(
        1,
        'hybrid',
        huggingface_repo_id='org/moe',
        backend=BackendEnum.VLLM,
        backend_parameters=['--data-parallel-hybrid-lb', '--dp=2'],
        env={'GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_ENABLED': 'true'},
    )
    state.serve._get_model = MagicMock(return_value=model)
    state.client.workers.get.return_value.ip = '10.0.0.1'
    state.serve._rank_health_executor = MagicMock()
    state.serve._sync_managed_rank_health(instance)
    submissions = state.serve._rank_health_executor.submit.call_args_list
    assert len(submissions) == (
        (1 if startup else 2) if state.serve._worker_id == 1 else 0
    )
    if not submissions:
        return
    from gpustack.worker.serve_manager import _probe_rank_health

    assert all(call.args[0] is _probe_rank_health for call in submissions)
    assert {call.args[4] for call in submissions} == (
        {'get'} if startup else {'get', 'inference'}
    )
    assert all(call.args[1].port == 40000 for call in submissions)
    assert all(call.args[1].worker_ip == '10.0.0.1' for call in submissions)


@pytest.mark.parametrize('epoch_changed', [False, True])
def test_hybrid_completed_startup_probe_is_fenced_by_current_generation(
    managed, epoch_changed
):
    from concurrent.futures import Future
    from gpustack.worker.serve_manager import _RankHealthCheck
    from tests.utils.model import new_hybrid_workload

    state = managed
    instance = state.instance
    local = ModelInstanceWorkload.model_validate(
        new_hybrid_workload(state.serve._worker_id, state='running')
    )
    instance.distributed_servers.workloads = [local]
    instance.distributed_servers.orchestration_mode = "managed"
    snapshot = instance.model_copy(deep=True)
    key = (
        instance.id,
        instance.restart_count,
        local.name,
        'vllm_hybrid',
        0,
        'component',
    )
    future = Future()
    future.set_result((True, ''))
    state.serve._rank_health_checks[key] = _RankHealthCheck(startup=True, future=future)
    state.serve._rank_health_executor = MagicMock()
    state.serve._get_model = MagicMock(
        return_value=new_model(
            1,
            'hybrid',
            huggingface_repo_id='org/moe',
            backend=BackendEnum.VLLM,
            backend_parameters=['--data-parallel-hybrid-lb', '--dp=2'],
        )
    )
    state.client.workers.get.return_value.ip = '10.0.0.1'
    if epoch_changed:
        instance.restart_count += 1
    state.serve._sync_managed_rank_health(snapshot)
    assert bool(state.client.model_instances.report_workloads.called) != epoch_changed
    assert instance.distributed_servers.workloads[0].component.ready == (
        not epoch_changed
    )
    if epoch_changed:
        assert key not in state.serve._rank_health_checks
        state.serve._rank_health_executor.submit.assert_not_called()


@pytest.mark.parametrize('failing_kind', ['component', 'inference'])
@pytest.mark.parametrize('failure_first', [False, True])
def test_hybrid_failed_health_cannot_be_revived_by_same_batch_success(
    managed, failing_kind, failure_first
):
    from concurrent.futures import Future
    from gpustack.worker.serve_manager import _RankHealthCheck
    from tests.utils.model import new_hybrid_workload

    state = managed
    item = ModelInstanceWorkload.model_validate(
        new_hybrid_workload(state.serve._worker_id, state='running')
    )
    item.component.ready = True
    keys, targets = [], {}
    for kind in ('component', 'inference'):
        key = (
            state.instance.id,
            state.instance.restart_count,
            item.name,
            'vllm_hybrid',
            0,
            kind,
        )
        keys.append(key)
        future = Future()
        future.set_result(
            (kind != failing_kind, 'failed probe' if kind == failing_kind else '')
        )
        state.serve._rank_health_checks[key] = _RankHealthCheck(
            startup=False, failures=2, future=future
        )
        targets[key] = (item.component, 10, 15, 3, True)
    keys.sort(key=lambda key: (key[5] == failing_kind) != failure_first)
    diagnostic, processed = state.serve._collect_component_health([item], targets, keys)
    assert item.state == ModelInstanceWorkloadStateEnum.FAILED
    assert item.component.ready is False
    assert len(processed) == 2
    assert 'failed probe' in diagnostic['state_message']


@pytest.mark.parametrize('hybrid', [False, True])
@pytest.mark.parametrize('custom_path', [None, '/custom-health'])
def test_component_health_path_preserves_sglang_contract(managed, hybrid, custom_path):
    from tests.utils.model import new_hybrid_workload

    state = managed
    item = ModelInstanceWorkload.model_validate(
        new_hybrid_workload(state.serve._worker_id, state='running')
    )
    if not hybrid:
        item.component.kind = 'sglang_engine'
    state.instance.backend = BackendEnum.VLLM if hybrid else BackendEnum.SGLANG
    state.instance.distributed_servers.workloads = [item]
    state.instance.distributed_servers.orchestration_mode = "managed"
    model = new_model(
        1,
        'model',
        huggingface_repo_id='org/model',
        backend='Modular MAX',
        env={'GPUSTACK_MODEL_HEALTH_CHECK_PATH': custom_path} if custom_path else {},
    )
    state.client.workers.get.return_value.ip = '10.0.0.1'
    state.serve._schedule_rank_health = MagicMock()
    state.serve._get_health_check_path = MagicMock(return_value='/backend-health')
    state.serve._sync_component_health(state.instance, model, [])
    assert state.serve._schedule_rank_health.call_args.args[-1] == (
        (custom_path or '/backend-health') if hybrid else '/health'
    )
    assert bool(state.serve._get_health_check_path.called) == hybrid
    if hybrid:
        state.serve._get_health_check_path.assert_called_once_with(
            BackendEnum.VLLM, model.owner_principal_id
        )


@pytest.fixture
def hybrid_managed(managed):
    from tests.utils.model import new_hybrid_workload

    state = managed
    state.instance.state = ModelInstanceStateEnum.RUNNING
    state.instance.worker_ip = '10.0.0.1'
    state.instance.distributed_servers.subordinate_workers = [
        ModelInstanceSubordinateWorker(
            worker_id=w, worker_ip=f'10.0.0.{w}', gpu_indexes=[0, 1]
        )
        for w in (2, 3, 4)
    ]
    state.instance.distributed_servers.workloads = [
        ModelInstanceWorkload.model_validate(
            new_hybrid_workload(w, headless=w in (2, 4), state='running')
        )
        for w in (1, 2, 3, 4)
    ]
    state.instance.distributed_servers.orchestration_mode = "managed"
    for item in state.instance.distributed_servers.workloads:
        if item.component.api_port is not None:
            item.component.dp_size = 1
            item.component.ready = True
    state.instance.named_ports['_vllm_external_api'] = PortBand(base=40000)
    state.instance.named_ports['_vllm_external_rpc'] = PortBand(base=40011)
    state.instance.named_ports['_vllm_external_store'] = PortBand(base=40012, count=2)
    _running_managed(state)
    for runtime in state.runtime:
        runtime.labels.update(
            {
                'gpustack.ai/component-kind': 'vllm_hybrid',
                'gpustack.ai/component-attempt': '0',
            }
        )
    state.serve._hold_managed_instance(state.instance)
    state.serve._sync_managed_rank_health = MagicMock()
    return state


def test_hybrid_runtime_failure_withdraws_whole_group_and_retains_pending_claims(
    hybrid_managed,
):
    from gpustack.utils.model_instance_workers import model_instance_endpoints

    state = hybrid_managed
    assert len(model_instance_endpoints(state.instance)) == 2
    state.runtime[0].state = WorkloadStatusStateEnum.UNHEALTHY
    state.delete.side_effect = lambda name: None
    state.serve._sync_managed_model_instance(state.instance)
    _finish_managed_tasks(state)
    failed = _local_manifest(state)[0]
    assert failed.state == ModelInstanceWorkloadStateEnum.FAILED
    assert failed.component.ready is (False if failed.component.api_port else None)
    assert model_instance_endpoints(state.instance) == []
    state.delete.assert_called_once_with(failed.name)
    assert state.instance.id in state.serve._assigned_ports
    assert state.instance.restart_count == 3
    state.create.assert_not_called()


@pytest.mark.parametrize('pd', [False, True])
@pytest.mark.parametrize('unknown', ['runtime', 'worker', 'accepted'])
def test_hybrid_stop_waits_for_local_absence_and_available_worker(
    hybrid_managed, pd, unknown
):
    state = hybrid_managed
    if pd:
        state.instance.role, state.instance.group_id = 'prefill', 'group'
    other = [
        w.model_dump()
        for w in state.instance.distributed_servers.workloads
        if w.worker_id != state.serve._worker_id
    ]
    normal_list, normal_delete = state.list.side_effect, state.delete.side_effect
    if unknown == 'runtime':
        state.list.side_effect = OSError('runtime unavailable')
    elif unknown == 'worker':
        state.client.workers.get.return_value.unreachable = True
    else:
        state.delete.side_effect = lambda name: None
    assert not state.serve._stop_model_instance(state.instance.model_copy(deep=True))
    assert state.instance.id in state.serve._assigned_ports
    assert _local_manifest(state)[0].state != ModelInstanceWorkloadStateEnum.STOPPED
    state.client.model_instances.report_workloads.assert_not_called()
    state.list.side_effect, state.delete.side_effect = normal_list, normal_delete
    state.client.workers.get.return_value.unreachable = False
    state.serve._retry_managed_stops()
    _finish_managed_tasks(state)
    stopped = _local_manifest(state)[0]
    assert stopped.state == ModelInstanceWorkloadStateEnum.STOPPED
    assert stopped.component.ready is (False if stopped.component.api_port else None)
    assert state.instance.id not in state.serve._assigned_ports
    assert state.instance.restart_count == 3
    assert [
        w.model_dump()
        for w in state.instance.distributed_servers.workloads
        if w.worker_id != state.serve._worker_id
    ] == other


def test_hybrid_worker_claim_adopts_without_recreating_or_advancing_generation(
    hybrid_managed,
):
    state = hybrid_managed
    state.serve._model_instance_by_instance_id.clear()
    state.serve._assigned_ports.clear()
    state.serve._dispatch_model_instance_event(
        Event(type=EventType.CREATED, data=state.instance.model_copy(deep=True))
    )
    assert state.instance.id in state.serve._assigned_ports
    assert state.instance.restart_count == 3
    assert _local_manifest(state)[0].state == ModelInstanceWorkloadStateEnum.RUNNING
    state.create.assert_not_called()
    state.delete.assert_not_called()
    state.serve._update_model_instance.assert_not_called()


@pytest.mark.parametrize('stamped', [False, True])
@pytest.mark.parametrize('grouped', [False, True])
def test_smg_pending_attempt_is_not_blocked_by_a_stale_sibling(smg, stamped, grouped):
    s = smg
    if s.serve._worker_id != 1:
        return
    _pending_cpu(s)
    s.model.stale = True
    s.instance.spec_digest = 'server-validated-spec' if stamped else None
    s.instance.group_id = 'pd-group' if grouped else None
    s.serve._sync_managed_model_instance(s.instance)
    _finish_managed_tasks(s)
    assert s.create.call_count == int(stamped and not grouped)


@pytest.mark.parametrize('changed_at', [None, 'pending', 'prepared', 'created'])
def test_smg_create_rechecks_current_instance_configuration(smg, changed_at):
    s = smg
    if s.serve._worker_id != 1:
        return
    _pending_cpu(s)
    s.instance.spec_digest = 'authorized-spec'
    checks = 0

    def get(*, id, use_cache, check_configuration=False):
        nonlocal checks
        if check_configuration:
            checks += 1
            if (
                checks
                == {None: 99, 'pending': 1, 'prepared': 2, 'created': 3}[changed_at]
            ):
                raise ConflictException(message='The deployment configuration changed')
        return s.instance.model_copy(deep=True)

    s.client.model_instances.get.side_effect = get
    s.serve._sync_managed_model_instance(s.instance)
    _finish_managed_tasks(s)
    assert checks >= (
        3 if changed_at in (None, 'created') else 1 if changed_at == 'pending' else 2
    )
    assert s.create.call_count == int(changed_at in (None, 'created'))
    assert any(w.name == 'engine-1' for w in s.runtime)
    cpu = [w for w in s.runtime if not w.name.startswith('engine')]
    assert bool(cpu) == (changed_at is None)


@pytest.mark.parametrize('manager', [1], indirect=True)
@pytest.mark.parametrize('status', [409, 503, None])
@pytest.mark.parametrize('engine_failed', [False, True])
@pytest.mark.asyncio
async def test_blocked_smg_recovery_preserves_engine_observation(
    smg, monkeypatch, status, engine_failed
):
    s = smg
    session = SimpleNamespace(info={})
    s.instance.spec_digest = await model_spec_digest(session, s.model)
    _pending_cpu(s)
    s.model.backend_parameters.append('--policy=random')
    session.info.clear()
    s.model.stale = await model_members_stale(session, s.model, [s.instance])
    assert s.model.stale is True

    def get(**kwargs):
        if kwargs.get('check_configuration'):
            if status is None:
                raise ReadTimeout('Configuration authority unavailable')
            raise HTTPException(
                status,
                'Conflict' if status == 409 else 'Unavailable',
                'Cannot authorize SMG creation',
            )
        return s.instance.model_copy(deep=True)

    s.client.model_instances.get.side_effect = get
    if engine_failed:
        next(w for w in s.runtime if w.name == 'engine-1').state = (
            WorkloadStatusStateEnum.FAILED
        )
    stop = MagicMock()
    monkeypatch.setattr(s.serve, '_stop_model_instance', stop)

    s.serve._sync_managed_model_instance(s.instance)
    _finish_managed_tasks(s)

    s.create.assert_not_called()
    observed_failure = engine_failed
    assert stop.call_count == int(observed_failure)
    assert s.client.model_instances.report_workloads.call_count == int(observed_failure)
    assert s.serve._sync_managed_rank_health.call_count == 1
    engine = next(
        w for w in s.instance.distributed_servers.workloads if w.name == 'engine-1'
    )
    assert engine.state == ('failed' if observed_failure else 'running')
    assert s.instance.restart_count == 3
