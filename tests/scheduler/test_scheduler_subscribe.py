"""Regression tests for scheduler bus-subscription parameters (issue #4794)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gpustack.scheduler.scheduler import Scheduler
from gpustack.server.bus import EventType


@pytest.mark.parametrize('state', ['pending', 'scheduled'])
@pytest.mark.parametrize('workload_state', ['pending', 'running', 'unknown', 'stopped'])
def test_periodic_scheduling_waits_until_the_manifest_is_retired(state, workload_state):
    from datetime import datetime, timedelta, timezone
    from gpustack.schemas.models import DistributedServers
    from tests.utils.model import new_model_instance, new_model_instance_workload

    scheduler = Scheduler(SimpleNamespace(cache_dir=None), check_interval=180)
    instance = new_model_instance(7, 'managed', 1, state=state)
    instance.worker_id = 1 if state == 'scheduled' else None
    instance.created_at = datetime.now(timezone.utc) - timedelta(minutes=10)
    instance.updated_at = datetime.now(timezone.utc) - timedelta(minutes=5)
    instance.distributed_servers = DistributedServers(
        workloads=[new_model_instance_workload(state=workload_state)]
    )
    assert not scheduler._should_schedule(instance)
    instance.distributed_servers.workloads = []
    instance.distributed_servers.orchestration_mode = "managed"
    assert scheduler._should_schedule(instance)


async def _stop_task(task: asyncio.Task) -> None:
    if not task.done():
        task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'state,bound,registration_needed',
    [
        ('ready', 7, False),
        ('ready', None, True),
        ('not_ready', 7, False),
        ('unreachable', 7, False),
    ],
)
async def test_managed_admission_message_distinguishes_offline_workers(
    state, bound, registration_needed, monkeypatch
):
    monkeypatch.setattr(
        'gpustack.extension.resolve_version_info', lambda: ('2.3.1', 'test')
    )
    from gpustack.policies.worker_filters.status_filter import StatusFilter

    worker = SimpleNamespace(
        state=state,
        system_principal_id=bound,
        worker_version='2.3.0',
    )
    candidates, messages = await StatusFilter(None, True).filter([worker])
    assert bool(candidates) == (state == 'ready' and bound is not None)
    assert (
        any('re-register' in message.lower() for message in messages)
        == registration_needed
    )


@pytest.mark.asyncio
async def test_scheduler_subscribes_only_to_created_and_skips_replay():
    cfg = SimpleNamespace(cache_dir=None)
    scheduler = Scheduler(cfg, check_interval=180)

    captured_kwargs = {}
    forever = asyncio.Event()

    async def fake_subscribe(*args, **kwargs):
        captured_kwargs.update(kwargs)
        await forever.wait()
        if False:
            yield  # pragma: no cover

    with (
        patch(
            "gpustack.scheduler.scheduler.ModelInstance.subscribe",
            side_effect=fake_subscribe,
        ),
        patch.object(scheduler, "_schedule_cycle", AsyncMock()),
        patch.object(
            scheduler, "_enqueue_pending_instances", AsyncMock()
        ) as mock_enqueue,
        patch("gpustack.scheduler.scheduler.AsyncIOScheduler"),
    ):
        task = asyncio.create_task(scheduler.start())
        # Give start() a chance to set up the subscription.
        for _ in range(50):
            await asyncio.sleep(0)
            if captured_kwargs:
                break
        try:
            assert captured_kwargs.get("source") == "scheduler"
            assert captured_kwargs.get("event_types") == {EventType.CREATED}
            assert captured_kwargs.get("replay_existing") is False
            # Bootstrap scan must have run at least once before the live loop.
            assert mock_enqueue.await_count >= 1
        finally:
            forever.set()
            await _stop_task(task)


@pytest.mark.asyncio
async def test_enqueue_event_instance_evaluates_only_the_passed_instance():
    cfg = SimpleNamespace(cache_dir=None)
    scheduler = Scheduler(cfg, check_interval=180)

    instance = SimpleNamespace(id=7)

    with (
        patch.object(scheduler, "_evaluate", AsyncMock()) as mock_evaluate,
        patch.object(scheduler, "_should_schedule", return_value=True) as mock_should,
    ):
        await scheduler._enqueue_event_instance(instance)
        mock_should.assert_called_once_with(instance)
        mock_evaluate.assert_awaited_once_with(instance)

    # When _should_schedule returns False, no evaluation happens.
    with (
        patch.object(scheduler, "_evaluate", AsyncMock()) as mock_evaluate,
        patch.object(scheduler, "_should_schedule", return_value=False),
    ):
        await scheduler._enqueue_event_instance(instance)
        mock_evaluate.assert_not_awaited()

    # None and id-less payloads are no-ops.
    with (
        patch.object(scheduler, "_evaluate", AsyncMock()) as mock_evaluate,
        patch.object(scheduler, "_should_schedule", return_value=True) as mock_should,
    ):
        await scheduler._enqueue_event_instance(None)
        await scheduler._enqueue_event_instance(SimpleNamespace(id=None))
        mock_should.assert_not_called()
        mock_evaluate.assert_not_awaited()


@pytest.mark.asyncio
async def test_scheduler_handles_live_created_event_with_single_instance_path():
    cfg = SimpleNamespace(cache_dir=None)
    scheduler = Scheduler(cfg, check_interval=180)

    instance_payload = SimpleNamespace(id=42)
    created_event = SimpleNamespace(
        type=EventType.CREATED, data=instance_payload, id=42
    )
    forever = asyncio.Event()

    async def fake_subscribe(*args, **kwargs):
        yield created_event
        await forever.wait()

    with (
        patch(
            "gpustack.scheduler.scheduler.ModelInstance.subscribe",
            side_effect=fake_subscribe,
        ),
        patch.object(scheduler, "_schedule_cycle", AsyncMock()),
        patch.object(
            scheduler, "_enqueue_pending_instances", AsyncMock()
        ) as mock_full_scan,
        patch.object(
            scheduler, "_enqueue_event_instance", AsyncMock()
        ) as mock_event_path,
    ):
        with patch("gpustack.scheduler.scheduler.AsyncIOScheduler"):
            task = asyncio.create_task(scheduler.start())
            for _ in range(100):
                await asyncio.sleep(0)
                if mock_event_path.await_count >= 1:
                    break
            try:
                # Bootstrap scan ran exactly once at startup.
                assert mock_full_scan.await_count == 1
                # Event path was invoked once with the instance from the event.
                assert mock_event_path.await_count == 1
                mock_event_path.assert_awaited_with(instance_payload)
            finally:
                forever.set()
                await _stop_task(task)


@pytest.mark.asyncio
@pytest.mark.parametrize('transition', ['identity_binding', 'not_ready_to_ready'])
async def test_periodic_retry_observes_current_worker_admission_without_generation_change(
    transition,
    monkeypatch,
):
    monkeypatch.setattr(
        'gpustack.extension.resolve_version_info', lambda: ('2.3.1', 'test')
    )
    from contextlib import asynccontextmanager
    from datetime import datetime, timedelta, timezone
    from gpustack.schemas.models import (
        Model,
        ModelInstance,
        ModelInstanceStateEnum,
        DistributedServers,
    )
    from gpustack.policies.worker_filters.status_filter import StatusFilter
    from tests.utils.model import new_model_instance

    scheduler = Scheduler(SimpleNamespace(cache_dir=None), check_interval=180)
    instance = new_model_instance(7, 'pending', 1, state=ModelInstanceStateEnum.PENDING)
    instance.worker_id = None
    instance.created_at = datetime.now(timezone.utc) - timedelta(minutes=10)
    instance.updated_at = datetime.now(timezone.utc) - timedelta(minutes=5)
    instance.distributed_servers = DistributedServers(workloads=[])
    initial_epoch = instance.restart_count
    worker = SimpleNamespace(
        id=1,
        state='ready',
        system_principal_id=101,
        worker_version='2.3.0',
    )
    if transition == 'not_ready_to_ready':
        worker.state = 'not_ready'
    else:
        worker.system_principal_id = None
    model = Model(id=1, name='m', source='huggingface', huggingface_repo_id='a/b')
    eligibility = []

    async def evaluate(row):
        kept, _ = await StatusFilter(model, True).filter([worker])
        eligibility.append([w.id for w in kept])

    @asynccontextmanager
    async def session():
        yield SimpleNamespace()

    with (
        patch('gpustack.scheduler.scheduler.async_session', session),
        patch.object(ModelInstance, 'all', AsyncMock(return_value=[instance])),
        patch.object(
            scheduler, '_evaluate', AsyncMock(side_effect=evaluate)
        ) as evaluate_mock,
    ):
        await scheduler._enqueue_pending_instances()
        worker.system_principal_id = 101
        worker.state = 'ready'
        await scheduler._enqueue_pending_instances()
    assert eligibility == [[], [1]]
    assert evaluate_mock.await_count == 2
    assert instance.restart_count == initial_epoch
    assert instance.state == ModelInstanceStateEnum.PENDING
