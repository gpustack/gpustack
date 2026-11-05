"""A group's members land together or not at all.

`solve_then_commit` exists so a group is never half-placed: the search proves
every member fits before anything is written. That guarantee ends at the write
if the rows go in one commit each — a failure partway through leaves some
members on workers and the rest PENDING, which the next reconcile reads as a
scale-out against a stale picture rather than as a group still forming.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gpustack.schemas.models import (
    ComputedResourceClaim,
    DistributedServers,
    Model,
    ModelInstance,
    ModelInstanceWorkload,
    SourceEnum,
)
from gpustack.scheduler.scheduler import apply_candidate_to_instance
from tests.utils.model import new_model_instance_workload


@pytest.fixture(autouse=True)
def server_version(monkeypatch):
    monkeypatch.setattr(
        'gpustack.extension.resolve_version_info', lambda: ('2.3.1', 'test')
    )


def _model():
    model = Model(
        name="llm", source=SourceEnum.HUGGING_FACE, huggingface_repo_id="org/repo"
    )
    model.id = 1
    model.cluster_id = 1
    return model


def _instance():
    return ModelInstance(
        id=1,
        cluster_id=1,
        name="llm-0",
        model_id=1,
        model_name="llm",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/repo",
    )


def _candidate():
    return SimpleNamespace(
        worker=_registered_worker(),
        gpu_indexes=[0],
        gpu_addresses=None,
        gpu_type="cuda",
        computed_resource_claim=ComputedResourceClaim(vram={0: 1024}, ram=0),
        subordinate_workers=None,
    )


@pytest.mark.asyncio
async def test_the_single_instance_path_still_commits_its_own_row():
    """Unchanged: one instance is its own transaction, and nothing waits on it."""
    instance = _instance()
    with (
        patch("gpustack.scheduler.scheduler.ModelInstanceService") as service,
        patch.object(
            ModelInstance,
            "all_by_field",
            AsyncMock(return_value=[instance]),
        ),
        patch(
            'gpustack.scheduler.scheduler.Worker.all_by_ids',
            AsyncMock(return_value=[_registered_worker()]),
        ),
    ):
        service.return_value.update = AsyncMock()
        await apply_candidate_to_instance(None, _model(), instance, _candidate())

    service.return_value.update.assert_awaited_once()
    assert instance.worker_id == 7


@pytest.mark.asyncio
async def test_a_group_member_is_written_but_not_committed_on_its_own():
    """The row is filled in exactly as before — only the commit is deferred, so
    the caller can put the whole group in one transaction."""
    instance = _instance()
    with patch("gpustack.scheduler.scheduler.ModelInstanceService") as service:
        service.return_value.update = AsyncMock()
        await apply_candidate_to_instance(
            None,
            _model(),
            instance,
            _candidate(),
            commit=False,
            admission={instance.id: instance},
        )

    service.return_value.update.assert_not_awaited()
    # Everything the committing caller needs is already on the row.
    assert instance.worker_id == 7
    assert instance.gpu_indexes == [0]
    assert instance.computed_resource_claim.vram == {0: 1024}


@pytest.mark.asyncio
@pytest.mark.parametrize("workloads", [None, []])
async def test_candidate_writeback_preserves_manifest(workloads):
    instance = _instance()
    instance.distributed_servers = DistributedServers(workloads=workloads)
    expected = instance.distributed_servers.workloads
    await apply_candidate_to_instance(
        None,
        _model(),
        instance,
        _candidate(),
        commit=False,
        admission={instance.id: instance},
    )
    assert instance.distributed_servers.workloads == expected


@pytest.mark.asyncio
@pytest.mark.parametrize('state', ['pending', 'running', 'unknown', 'stopped'])
async def test_candidate_cannot_replace_a_registered_workload_allocation(state):
    instance = _instance()
    instance.worker_id = 99
    instance.gpu_indexes = [1]
    instance.distributed_servers = DistributedServers(
        workloads=[new_model_instance_workload(worker_id=99, state=state)]
    )
    before = instance.model_dump()
    accepted = await apply_candidate_to_instance(
        None,
        _model(),
        instance,
        _candidate(),
        commit=False,
        admission={instance.id: instance},
    )
    assert not accepted
    assert instance.model_dump() == before


@pytest.mark.asyncio
async def test_scheduling_does_not_enable_managed_workloads():
    instance = _instance()
    await apply_candidate_to_instance(
        None,
        _model(),
        instance,
        _candidate(),
        commit=False,
        admission={instance.id: instance},
    )
    assert instance.distributed_servers.workloads is None


@pytest.mark.asyncio
@pytest.mark.parametrize('changed', ['deleted', 'draining', 'generation'])
async def test_rejected_instance_fence_rolls_back_without_committing(changed):
    from datetime import datetime, timezone
    from unittest.mock import MagicMock
    from gpustack.scheduler.scheduler import _lock_scheduling_instances

    expected = _instance()
    current = expected.model_copy(deep=True)
    if changed == 'deleted':
        current.deleted_at = datetime.now(timezone.utc)
    elif changed == 'draining':
        current.draining_since = datetime.now(timezone.utc)
    else:
        current.restart_count = 1
    session = MagicMock(commit=AsyncMock(), rollback=AsyncMock())
    with patch.object(ModelInstance, 'one_by_id', AsyncMock(return_value=current)):
        assert await _lock_scheduling_instances(session, [expected]) is None
    session.rollback.assert_awaited_once()
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['guard', 'epoch', 'draining'])
async def test_candidate_commit_rechecks_stop_and_epoch_under_lock(change):
    from uuid import uuid4
    from datetime import datetime, timezone
    from unittest.mock import MagicMock
    from gpustack.schemas.models import ModelInstanceLegacyStop

    stale = _instance()
    current = stale.model_copy(deep=True)
    if change == 'guard':
        current.distributed_servers = DistributedServers(
            legacy_stop=ModelInstanceLegacyStop(
                token=uuid4(), restart_count=0, worker_ids=[], worker_principal_ids={}
            )
        )
    elif change == 'epoch':
        current.restart_count = 1
    else:
        current.draining_since = datetime.now(timezone.utc)
    session = MagicMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    with (
        patch.object(
            ModelInstance,
            'all_by_field',
            AsyncMock(
                side_effect=lambda *a, **kw: (
                    [current] if kw.get('for_update') else [stale]
                )
            ),
        ) as lookup,
        patch(
            'gpustack.scheduler.scheduler.Worker.all_by_ids',
            AsyncMock(return_value=[_registered_worker()]),
        ),
        patch('gpustack.scheduler.scheduler.ModelInstanceService') as service,
    ):
        service.return_value.update = AsyncMock()
        await apply_candidate_to_instance(session, _model(), stale, _candidate())
    lookup.assert_awaited_with(session, 'model_id', 1, for_update=True, refresh=True)
    assert current.worker_id is None
    service.return_value.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_group_rechecks_every_member_before_assigning_any_candidate():
    from uuid import uuid4
    from unittest.mock import MagicMock
    from gpustack.scheduler.scheduler import Scheduler
    from gpustack.schemas.models import ModelInstanceLegacyStop

    first = _instance()
    second = first.model_copy(update={'id': 2})
    first.group_id = second.group_id = 'g1'
    current_second = second.model_copy(deep=True)
    current_second.distributed_servers = DistributedServers(
        legacy_stop=ModelInstanceLegacyStop(
            token=uuid4(), restart_count=0, worker_ids=[], worker_principal_ids={}
        )
    )
    rows = {1: first, 2: current_second}
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._config = SimpleNamespace()
    session = MagicMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    with (
        patch('gpustack.scheduler.scheduler.is_group_forming', return_value=True),
        patch(
            'gpustack.scheduler.scheduler.schedule_group',
            AsyncMock(return_value=({1: _candidate(), 2: _candidate()}, [])),
        ),
        patch.object(
            ModelInstance,
            'all_by_field',
            AsyncMock(return_value=list(rows.values())),
        ),
        patch(
            'gpustack.scheduler.scheduler.Worker.all_by_ids',
            AsyncMock(return_value=[_registered_worker()]),
        ),
        patch('gpustack.scheduler.scheduler.ModelInstanceService') as service,
    ):
        service.return_value.batch_update = AsyncMock()
        assert (
            await scheduler._try_schedule_group(
                session, _model(), first, [], [first, second]
            )
            is True
        )
    assert first.worker_id is None
    assert second.worker_id is None
    service.return_value.batch_update.assert_not_awaited()


def _registered_worker(identity=7, supported=True):
    return SimpleNamespace(
        id=identity,
        name=f'w{identity}',
        ip=f'10.0.0.{identity}',
        advertise_address=None,
        ifname='eth0',
        cluster_id=1,
        deleted_at=None,
        state='ready',
        system_principal_id=identity + 100 if supported else None,
        worker_version='2.3.0',
    )


@pytest.fixture
def managed_admission(monkeypatch):
    from unittest.mock import MagicMock
    from gpustack.scheduler import scheduler
    from gpustack.schemas.workers import Worker

    target = _instance()
    target.cluster_id = 1
    target.distributed_servers = DistributedServers(workloads=[])
    model = _model()
    model.cluster_id = 1
    candidate = _candidate()
    candidate.worker = _registered_worker()
    session = MagicMock()
    session.rollback = AsyncMock()
    session.commit = AsyncMock()
    state = SimpleNamespace(
        target=target,
        model=model,
        candidate=candidate,
        session=session,
        discovery=[target],
        current=[target],
        workers={7: candidate.worker},
        events=[],
    )

    async def lock_workers(session, ids, **kwargs):
        state.events.append(('workers', sorted(ids), kwargs))
        return [state.workers[i] for i in sorted(ids) if i in state.workers]

    async def lock_instances(session, field, value, **kwargs):
        if not kwargs.get('for_update'):
            return state.discovery
        state.events.append(('instances', [r.id for r in state.current], kwargs))
        return state.current

    monkeypatch.setattr(Worker, 'all_by_ids', AsyncMock(side_effect=lock_workers))
    monkeypatch.setattr(
        ModelInstance,
        'all_by_field',
        AsyncMock(side_effect=lock_instances),
    )
    state.write = AsyncMock()
    state.batch = AsyncMock()
    monkeypatch.setattr(
        scheduler,
        'ModelInstanceService',
        lambda session: SimpleNamespace(update=state.write, batch_update=state.batch),
    )
    return state


@pytest.mark.asyncio
async def test_registration_after_solve_prevents_candidate_writeback(
    managed_admission,
):
    s = managed_admission
    current = s.target.model_copy(deep=True)
    current.distributed_servers.workloads = [
        ModelInstanceWorkload.model_validate(
            new_model_instance_workload(worker_id=7, state='pending')
        )
    ]
    current.distributed_servers.orchestration_mode = "managed"
    s.current = [current]
    before = current.model_dump()
    assert not await apply_candidate_to_instance(
        s.session, s.model, s.target, s.candidate
    )
    assert current.model_dump() == before
    s.session.rollback.assert_awaited_once()
    s.write.assert_not_awaited()
    s.session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_managed_admission_locks_workers_before_current_instance_cohort(
    managed_admission,
):
    s = managed_admission
    assert await apply_candidate_to_instance(s.session, s.model, s.target, s.candidate)
    assert [e[0] for e in s.events] == ['workers', 'instances']
    assert all(e[2] == {'for_update': True, 'refresh': True} for e in s.events)
    assert s.target.worker_id == 7
    s.write.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'fault',
    [
        'unbound',
        'deleted',
        'cluster',
        'not_ready',
        'missing',
    ],
)
async def test_managed_candidate_changed_after_solve_is_refused(
    managed_admission, fault
):
    s = managed_admission
    worker = s.workers[7]
    if fault == 'unbound':
        worker.system_principal_id = None
    elif fault == 'deleted':
        worker.deleted_at = object()
    elif fault == 'cluster':
        worker.cluster_id = 2
    elif fault == 'not_ready':
        worker.state = 'not_ready'
    else:
        s.workers.clear()
    assert not await apply_candidate_to_instance(
        s.session, s.model, s.target, s.candidate
    )
    assert s.target.worker_id is None
    s.write.assert_not_awaited()


@pytest.mark.asyncio
async def test_managed_subordinate_and_existing_legacy_router_require_identity(
    managed_admission,
):
    from gpustack.schemas.models import ModelInstanceSubordinateWorker

    s = managed_admission
    s.candidate.subordinate_workers = [ModelInstanceSubordinateWorker(worker_id=8)]
    s.workers[8] = _registered_worker(8, False)
    assert not await apply_candidate_to_instance(
        s.session, s.model, s.target, s.candidate
    )
    assert s.target.worker_id is None
    s.candidate.subordinate_workers = None
    router = _instance().model_copy(update={'id': 2, 'worker_id': 9, 'role': 'router'})
    router.cluster_id = 1
    s.discovery.append(router)
    s.current.append(router)
    s.workers[9] = _registered_worker(9, False)
    assert not await apply_candidate_to_instance(
        s.session, s.model, s.target, s.candidate
    )
    assert s.target.worker_id is None
    s.write.assert_not_awaited()


@pytest.mark.asyncio
async def test_new_cohort_worker_after_discovery_aborts_without_reverse_worker_lock(
    managed_admission,
):
    s = managed_admission
    newcomer = _instance().model_copy(update={'id': 2, 'worker_id': 9})
    newcomer.cluster_id = 1
    s.current = s.current + [newcomer]
    assert not await apply_candidate_to_instance(
        s.session, s.model, s.target, s.candidate
    )
    assert [event[0] for event in s.events] == ['workers', 'instances']
    s.session.rollback.assert_awaited()
    s.write.assert_not_awaited()
    assert s.target.worker_id is None


@pytest.mark.asyncio
async def test_managed_commit_false_acquires_admission_before_write(managed_admission):
    s = managed_admission
    assert await apply_candidate_to_instance(
        s.session, s.model, s.target, s.candidate, commit=False
    )
    assert [event[0] for event in s.events] == ['workers', 'instances']
    assert s.target.worker_id == 7
    s.write.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['spec', 'marker', 'owner', 'epoch', 'drain'])
async def test_managed_writeback_compares_preparation_snapshot(
    managed_admission, change
):
    from datetime import datetime, timezone

    s = managed_admission
    current = s.target.model_copy(deep=True)
    if change == 'spec':
        current.spec_digest = 'changed'
    elif change == 'marker':
        current.distributed_servers = DistributedServers(workloads=None)
    elif change == 'owner':
        current.worker_id = 7
    elif change == 'epoch':
        current.restart_count = 1
    else:
        current.draining_since = datetime.now(timezone.utc)
    s.current = [current]
    assert not await apply_candidate_to_instance(
        s.session, s.model, s.target, s.candidate
    )
    assert s.target.worker_id is None
    s.write.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['identity', 'drain'])
async def test_group_last_member_failure_has_zero_assignment_writes(
    managed_admission, monkeypatch, caplog, failure
):
    from datetime import datetime, timezone
    from gpustack.scheduler import scheduler
    from gpustack.schemas.workers import Worker

    s = managed_admission
    second = s.target.model_copy(deep=True, update={'id': 2})
    s.target.group_id = second.group_id = 'g1'
    current_second = second.model_copy(deep=True)
    if failure == 'drain':
        current_second.draining_since = datetime.now(timezone.utc)
    s.discovery = [s.target, second]
    s.current = [s.target, current_second]
    candidate2 = _candidate()
    candidate2.worker = _registered_worker(8)
    s.workers[8] = candidate2.worker
    original = Worker.all_by_ids.side_effect

    async def lock(*args, **kwargs):
        if failure == 'identity':
            s.workers[8].system_principal_id = None
        return await original(*args, **kwargs)

    monkeypatch.setattr(Worker, 'all_by_ids', AsyncMock(side_effect=lock))
    monkeypatch.setattr(scheduler, 'is_group_forming', lambda *a: True)
    monkeypatch.setattr(
        scheduler,
        'schedule_group',
        AsyncMock(return_value=({1: s.candidate, 2: candidate2}, [])),
    )
    scheduler_instance = scheduler.Scheduler.__new__(scheduler.Scheduler)
    scheduler_instance._config = SimpleNamespace()
    placement_identity = (s.model.name, s.target.group_id)

    def discard_identity():
        s.model.name = s.target.group_id = None

    s.session.rollback.side_effect = discard_identity
    with caplog.at_level('DEBUG', logger=scheduler.__name__):
        assert await scheduler_instance._try_schedule_group(
            s.session, s.model, s.target, list(s.workers.values()), s.discovery
        )
    assert s.target.worker_id is None and second.worker_id is None
    s.write.assert_not_awaited()
    s.batch.assert_not_awaited()
    s.session.rollback.assert_awaited()
    assert (
        'Group placement was fenced by a concurrent lifecycle or Worker change'
        in caplog.text
    )
    assert caplog.records[-1].args == placement_identity


@pytest.mark.asyncio
async def test_complete_group_admission_locks_once_then_writes_one_batch(
    managed_admission, monkeypatch
):
    from gpustack.scheduler import scheduler

    s = managed_admission
    second = s.target.model_copy(deep=True, update={'id': 2})
    s.target.group_id = second.group_id = 'g1'
    s.discovery = [s.target, second]
    s.current = [s.target, second]
    candidate2 = _candidate()
    candidate2.worker = _registered_worker(8)
    s.workers[8] = candidate2.worker
    monkeypatch.setattr(scheduler, 'is_group_forming', lambda *a: True)
    monkeypatch.setattr(
        scheduler,
        'schedule_group',
        AsyncMock(return_value=({1: s.candidate, 2: candidate2}, [])),
    )
    scheduler_instance = scheduler.Scheduler.__new__(scheduler.Scheduler)
    scheduler_instance._config = SimpleNamespace()
    assert await scheduler_instance._try_schedule_group(
        s.session, s.model, s.target, list(s.workers.values()), s.discovery
    )
    assert s.events == [
        ('workers', [7, 8], {'for_update': True, 'refresh': True}),
        ('instances', [1, 2], {'for_update': True, 'refresh': True}),
    ]
    assert [s.target.worker_id, second.worker_id] == [7, 8]
    s.write.assert_not_awaited()
    s.batch.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize('version', ['2.3.1', '2.3.0', '2.3.1-ent1', None, 'unknown'])
@pytest.mark.parametrize('managed', [False, True])
async def test_candidate_version_does_not_block_assignment(
    managed_admission, version, managed
):
    s = managed_admission
    if not managed:
        s.target.distributed_servers = DistributedServers(workloads=None)
        s.workers[7].system_principal_id = None
    s.workers[7].worker_version = version
    assert await apply_candidate_to_instance(s.session, s.model, s.target, s.candidate)
    assert s.target.worker_id == 7
    s.write.assert_awaited_once()


@pytest.mark.asyncio
async def test_legacy_offline_sibling_does_not_block_new_legacy_candidate(
    managed_admission,
):
    s = managed_admission
    s.target.distributed_servers = DistributedServers(workloads=None)
    offline = _instance().model_copy(update={'id': 2, 'worker_id': 9})
    s.discovery = [s.target, offline]
    s.current = [s.target, offline]
    s.workers[9] = _registered_worker(9, False)
    s.workers[9].state = 'unreachable'
    s.workers[7].system_principal_id = None
    s.workers[7].worker_version = '2.3.1'
    assert await apply_candidate_to_instance(s.session, s.model, s.target, s.candidate)
    assert s.events[0][1] == [7]
    assert s.target.worker_id == 7
    assert offline.worker_id == 9
    s.write.assert_awaited_once()


@pytest.mark.asyncio
async def test_legacy_becoming_managed_cohort_requires_retry_before_assignment(
    managed_admission,
):
    s = managed_admission
    s.target.distributed_servers = DistributedServers(workloads=None)
    old_member = _instance().model_copy(update={'id': 2, 'worker_id': 9})
    pending = _instance().model_copy(update={'id': 3})
    pending.distributed_servers = DistributedServers(workloads=[])
    s.discovery = [s.target, old_member]
    s.current = [s.target, old_member, pending]
    assert not await apply_candidate_to_instance(
        s.session, s.model, s.target, s.candidate
    )
    assert [event[0] for event in s.events] == ['workers', 'instances']
    assert s.events[0][1] == [7]
    s.session.rollback.assert_awaited_once()
    assert s.target.worker_id is None
    s.write.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('state', ['maintenance', 'unreachable'])
async def test_bound_existing_sibling_does_not_require_candidate_ready_state(
    managed_admission, state
):
    s = managed_admission
    sibling = _instance().model_copy(update={'id': 2, 'worker_id': 9})
    s.discovery.append(sibling)
    s.current = s.discovery
    s.workers[9] = _registered_worker(9)
    s.workers[9].state = state
    assert await apply_candidate_to_instance(s.session, s.model, s.target, s.candidate)
    assert s.events[0][1] == [7, 9]
    assert s.target.worker_id == 7
    assert sibling.worker_id == 9
    s.write.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize('has_resource_view', [False, True])
async def test_unbound_cohort_returns_before_resource_and_filter_work(
    managed_admission, monkeypatch, has_resource_view
):
    from unittest.mock import MagicMock
    from gpustack.scheduler import scheduler

    s = managed_admission
    s.target.worker_id = 7
    s.workers[7].system_principal_id = None
    load = AsyncMock(side_effect=AssertionError('resource loading is unnecessary'))
    topology = AsyncMock(side_effect=AssertionError('topology is unnecessary'))
    filters = MagicMock(side_effect=AssertionError('filters are unnecessary'))
    selector = MagicMock(side_effect=AssertionError('selector is unnecessary'))
    monkeypatch.setattr(scheduler, 'load_resource_view', load)
    monkeypatch.setattr(scheduler, '_group_topology', topology)
    monkeypatch.setattr(scheduler, 'WorkerFilterChain', filters)
    monkeypatch.setattr(scheduler, 'build_candidate_selector', selector)
    view = MagicMock() if has_resource_view else None
    candidate, messages = await scheduler.find_candidate(
        s.session,
        MagicMock(),
        s.model,
        [s.workers[7]],
        [s.target],
        resource_view=view,
    )
    assert candidate is None
    assert messages == [scheduler._MANAGED_WORKER_MESSAGE]
    load.assert_not_awaited()
    topology.assert_not_awaited()
    filters.assert_not_called()
    selector.assert_not_called()
    if view is not None:
        view.with_model_instances.assert_not_called()
