from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from sqlalchemy.dialects import mysql, postgresql

from gpustack.policies.resource_view import ResourceView
from gpustack.policies.worker_filters.backend_framework_filter import (
    BackendFrameworkFilter,
)
from gpustack.scheduler import evaluator, port_budget, scheduler
from gpustack.scheduler.group_capacity import GroupCapacity
from gpustack.schemas.models import (
    BackendEnum,
    ComputedResourceClaim,
    DistributedServers,
    Model,
    ModelInstance,
    ModelInstanceStateEnum,
    RoleResources,
    RoleSpec,
    RouterCPURejection,
    SourceEnum,
)
from gpustack.schemas.port_reservations import WorkerPortAvailability
from gpustack.schemas.workers import (
    CPUInfo,
    MemoryInfo,
    SystemReserved,
    Worker,
    WorkerStateEnum,
    WorkerStatus,
)


def _worker(number=1, cores=2):
    return Worker(
        id=number,
        name=f"worker-{number}",
        cluster_id=1,
        labels={"worker-name": f"worker-{number}"},
        state=WorkerStateEnum.READY,
        system_reserved=SystemReserved(ram=0, vram=0),
        status=WorkerStatus(
            memory=MemoryInfo(total=8 * 1024**3),
            cpu=CPUInfo(total=cores) if cores is not None else None,
        ),
    )


def _model(number=1, *, cpu=None, allowed=None, replicas=1):
    return Model(
        id=number,
        name=f"pd-{number}",
        cluster_id=1,
        backend=BackendEnum.VLLM.value,
        source=SourceEnum.LOCAL_PATH,
        local_path="/models/pd",
        roles=[
            RoleSpec(
                name="router",
                replicas=replicas,
                resources=RoleResources(cpu=cpu) if cpu is not None else None,
                worker_selector={"worker-name": allowed} if allowed else None,
            )
        ],
    )


def _instance(number, model_id, worker_id=None, cpu_millicores=None):
    return ModelInstance(
        id=number,
        name=f"router-{number}",
        model_id=model_id,
        model_name=f"pd-{model_id}",
        source=SourceEnum.LOCAL_PATH,
        local_path="/models/pd",
        cluster_id=1,
        role="router",
        state=(
            ModelInstanceStateEnum.SCHEDULED
            if worker_id
            else ModelInstanceStateEnum.PENDING
        ),
        worker_id=worker_id,
        computed_resource_claim=(
            ComputedResourceClaim(cpu_millicores=cpu_millicores) if worker_id else None
        ),
    )


@pytest.fixture
def scheduling(monkeypatch, config):
    async def supported(_filter, workers):
        return workers, []

    async def port_snapshots(workers, exclude_instance_id=None):
        return {
            worker.id: WorkerPortAvailability(
                worker_id=worker.id, generation=uuid4(), port_range="40000-40063"
            )
            for worker in workers
        }

    monkeypatch.setattr(BackendFrameworkFilter, "filter", supported)
    monkeypatch.setattr(port_budget, "worker_port_snapshots", port_snapshots)
    monkeypatch.setattr(scheduler, "_group_topology", AsyncMock(return_value=None))
    monkeypatch.setattr('gpustack.envs.SCHEDULER_SCALE_UP_LOCALITY_MAX_SCORE', 0)
    return config


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('cores', 'existing_cpu', 'cpu_request', 'fits', 'diagnostic'),
    [
        (2, None, None, True, None),
        (2, 575, None, False, '1425m'),
        (2, 500, 1.5, True, None),
        (2, 500, 1.5001, False, '1501m'),
        (None, None, None, False, 'unavailable'),
    ],
)
async def test_router_preflight_and_placement_share_cpu_accounting(
    scheduling, cores, existing_cpu, cpu_request, fits, diagnostic
):
    worker = _worker(cores=cores)
    model = _model(cpu=cpu_request)
    instances = [_instance(2, 2, worker.id, existing_cpu)]
    view = ResourceView(instances)
    group = GroupCapacity(scheduling, model, [worker], instances, resource_view=view)

    slots = await group('router', [worker.id], [])
    candidate, messages = await scheduler.find_candidate(
        None, scheduling, model, [worker], instances, role='router', resource_view=view
    )

    assert slots == ({} if cores is None else {1: int(fits)})
    assert (candidate is not None) is fits
    if fits:
        assert candidate.computed_resource_claim.cpu_millicores == int(
            (cpu_request or 2) * 1000
        )
    else:
        assert diagnostic in ''.join(messages)
        assert diagnostic in ''.join(group.notes_for('router'))


@pytest.mark.asyncio
async def test_group_slot_simulation_subtracts_its_first_router(scheduling):
    model = _model(cpu=1.5, replicas=2)
    group = GroupCapacity(
        scheduling, model, [_worker()], [], resource_view=ResourceView([])
    )
    assert await group('router', [1], []) == {1: 1}
    assert await group.commit('router', [1, 1], []) == []


@pytest.mark.asyncio
async def test_commit_rechecks_other_models_under_the_worker_lock(
    scheduling, monkeypatch
):
    worker = _worker()
    first, second = _instance(10, 1), _instance(20, 2)
    models = [_model(), _model(2)]
    events = []
    session = SimpleNamespace(rollback=AsyncMock())

    async def lock_worker(db, ids, *, for_update, refresh):
        assert db is session and ids == {worker.id} and for_update and refresh
        events.append('worker-lock')
        return [worker]

    async def discovery(db, field, value, *, refresh):
        assert field == 'model_id' and refresh
        return [row for row in (first, second) if row.model_id == value]

    async def current(statement):
        assert events[-1] == 'worker-lock'
        for dialect in (mysql.dialect(), postgresql.dialect()):
            sql = str(
                statement.compile(
                    dialect=dialect, compile_kwargs={'literal_binds': True}
                )
            )
            assert 'FOR UPDATE' in sql and 'worker_id IN (1)' in sql
        assert statement.get_execution_options()['populate_existing']
        events.append('current-claims')
        return SimpleNamespace(all=lambda: [first, second])

    session.exec = AsyncMock(side_effect=current)
    monkeypatch.setattr(Worker, 'all_by_ids', lock_worker)
    monkeypatch.setattr(ModelInstance, 'all_by_field', discovery)
    service = SimpleNamespace(update=AsyncMock())
    monkeypatch.setattr(scheduler, 'ModelInstanceService', Mock(return_value=service))

    # Both solves see the same empty ledger. The first commit has no Pod yet.
    candidates = []
    for model in models:
        candidate, _ = await scheduler.find_candidate(
            None,
            scheduling,
            model,
            [worker],
            [],
            role='router',
            resource_view=ResourceView([]),
        )
        assert candidate is not None
        candidates.append(candidate)
    assert await scheduler.apply_candidate_to_instance(
        session, models[0], first, candidates[0]
    )
    assert first.computed_resource_claim.cpu_millicores == 2000
    assert not await scheduler.apply_candidate_to_instance(
        session, models[1], second, candidates[1]
    )
    session.rollback.assert_awaited_once()
    assert second.worker_id is None
    service.update.assert_awaited_once_with(first)

    first.worker_id = None
    first.computed_resource_claim = None
    assert await scheduler.apply_candidate_to_instance(
        session, models[1], second, candidates[1]
    )
    assert events == ['worker-lock', 'current-claims'] * 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('allowed', 'tried', 'eligible'),
    [
        (None, [1, 2], {3}),
        (None, [1, 2, 3], {1, 2, 3}),
        ('worker-1', [1, 2], {1}),
    ],
)
async def test_recovery_prefers_an_alternative_but_respects_selector(
    scheduling, allowed, tried, eligible
):
    model = _model(allowed=allowed)
    instance = _instance(1, 1)
    instance.distributed_servers = DistributedServers(
        router_cpu_rejection=RouterCPURejection(worker_ids=tried, message='OutOfcpu')
    )
    candidate, _ = await scheduler.find_candidate(
        None,
        scheduling,
        model,
        [_worker(1), _worker(2), _worker(3)],
        [instance],
        role='router',
        exclude_instance_id=instance.id,
        resource_view=ResourceView([instance]),
    )
    assert candidate.worker.id in eligible
    instance.state = ModelInstanceStateEnum.ANALYZING
    assert await scheduler.apply_candidate_to_instance(
        None,
        model,
        instance,
        candidate,
        commit=False,
        admission={instance.id: instance},
    )
    history = instance.distributed_servers.router_cpu_rejection
    if candidate.worker.id in tried:
        assert history is None
    else:
        assert history.worker_ids == tried


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'cores, replicas, fits', [(1, 1, False), (2, 1, True), (2, 2, False), (4, 2, True)]
)
async def test_router_only_evaluation_checks_all_cpu_slots(
    scheduling, monkeypatch, cores, replicas, fits
):
    monkeypatch.setattr(evaluator.Cluster, 'one_by_id', AsyncMock(return_value=None))
    result = await evaluator.evaluate_group(
        scheduling,
        None,
        _model(replicas=replicas),
        [_worker(cores=cores)],
        [],
        cluster_id=1,
        cache_instances=[],
    )
    assert (result.total is not None) is fits
    if fits:
        assert result.claims[0].replicas == replicas
    else:
        assert any('CPU' in message for message in result.messages)
        assert result.demands[0].placeable == cores // 2


@pytest.mark.asyncio
async def test_router_preflight_cache_tracks_claims_and_capacity_not_utilization(
    monkeypatch,
):
    model = _model()
    worker = _worker()
    existing = _instance(2, 2, worker.id, 1500)
    evaluate = AsyncMock(
        side_effect=lambda *args, **kwargs: evaluator.ModelEvaluationResult()
    )
    monkeypatch.setattr(evaluator, 'evaluate_cache', {})
    monkeypatch.setattr(
        evaluator, 'visible_cache_service', AsyncMock(return_value=None)
    )
    monkeypatch.setattr(evaluator, 'cache_instances_in', AsyncMock(return_value=[]))
    monkeypatch.setattr(evaluator, 'evaluate_model', evaluate)
    monkeypatch.setattr(evaluator, 'evaluate_model_limiter', evaluator.AsyncLimiter(10))
    args = (None, None, model, [worker])

    empty = await evaluator.evaluate_model_with_cache(*args, [])
    occupied = await evaluator.evaluate_model_with_cache(*args, [existing])
    assert occupied is not empty
    worker.status.cpu.utilization_rate = 78
    assert await evaluator.evaluate_model_with_cache(*args, [existing]) is occupied
    assert await evaluator.evaluate_model_with_cache(*args, []) is empty
    worker.status.cpu.total = 4
    assert await evaluator.evaluate_model_with_cache(*args, [existing]) is not occupied
    assert evaluate.await_count == 3
