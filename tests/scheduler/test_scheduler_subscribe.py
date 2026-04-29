"""Regression tests for scheduler bus-subscription parameters (issue #4794)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gpustack.scheduler.scheduler import Scheduler
from gpustack.server.bus import EventType


async def _stop_task(task: asyncio.Task) -> None:
    if not task.done():
        task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass


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
