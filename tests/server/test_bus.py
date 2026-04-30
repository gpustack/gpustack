import asyncio
import logging

import pytest

from gpustack.server.bus import Event, EventType, Subscriber


@pytest.mark.asyncio
async def test_updated_event_overflow_does_not_leave_unreceivable_latest_event():
    """Regression for #4794: queue-full UPDATED ids must remain deliverable."""
    queue_size = 4
    subscriber = Subscriber(topic="modelinstance", source="test", queue_size=queue_size)

    total = queue_size + 5
    enqueue_tasks = [
        asyncio.create_task(
            subscriber.enqueue(
                Event(
                    type=EventType.UPDATED,
                    data={"id": event_id, "value": event_id},
                    id=event_id,
                )
            )
        )
        for event_id in range(total)
    ]

    received_ids = []
    for _ in range(total):
        event = await asyncio.wait_for(subscriber.receive(), timeout=2)
        received_ids.append(event.id)

    await asyncio.gather(*enqueue_tasks)

    assert sorted(received_ids) == list(range(total))
    assert subscriber.latest_by_key == {}
    assert subscriber.queue.empty()


@pytest.mark.asyncio
async def test_updated_events_for_same_id_are_coalesced_to_latest():
    subscriber = Subscriber(topic="modelinstance", source="test")

    await subscriber.enqueue(
        Event(type=EventType.UPDATED, data={"id": 1, "value": "old"}, id=1)
    )
    await subscriber.enqueue(
        Event(type=EventType.UPDATED, data={"id": 1, "value": "mid"}, id=1)
    )
    await subscriber.enqueue(
        Event(type=EventType.UPDATED, data={"id": 1, "value": "new"}, id=1)
    )

    event = await asyncio.wait_for(subscriber.receive(), timeout=1)
    assert event.id == 1
    assert event.data["value"] == "new"
    assert subscriber.latest_by_key == {}
    assert subscriber.queue.empty()


@pytest.mark.asyncio
async def test_subscriber_filters_event_types_before_enqueue():
    subscriber = Subscriber(
        topic="modelinstance",
        source="scheduler",
        event_types={EventType.CREATED},
    )

    await subscriber.enqueue(Event(type=EventType.UPDATED, data={"id": 1}, id=1))
    await subscriber.enqueue(Event(type=EventType.DELETED, data={"id": 2}, id=2))
    assert subscriber.queue.empty()
    assert subscriber.latest_by_key == {}

    await subscriber.enqueue(Event(type=EventType.CREATED, data={"id": 3}, id=3))
    event = await asyncio.wait_for(subscriber.receive(), timeout=1)
    assert event.type == EventType.CREATED
    assert event.id == 3


@pytest.mark.asyncio
async def test_queue_full_log_includes_metadata(caplog):
    """The warning must identify which subscriber backpressured."""
    subscriber = Subscriber(topic="modelinstance", source="scheduler", queue_size=1)
    await subscriber.enqueue(Event(type=EventType.CREATED, data={"id": 1}, id=1))

    caplog.set_level(logging.WARNING, logger="gpustack.server.bus")
    pending = asyncio.create_task(
        subscriber.enqueue(Event(type=EventType.CREATED, data={"id": 2}, id=2))
    )
    # Yield so the enqueue task hits the full-queue branch.
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    await asyncio.wait_for(subscriber.receive(), timeout=1)
    await asyncio.wait_for(subscriber.receive(), timeout=1)
    await pending

    matching = [
        rec
        for rec in caplog.records
        if "queue full, applying backpressure" in rec.getMessage()
    ]
    assert matching, "expected queue-full backpressure log entry"
    msg = matching[0].getMessage()
    assert "source=scheduler" in msg
    assert "topic=modelinstance" in msg
    assert "event_type=CREATED" in msg
    assert "id=2" in msg
    assert "queue_size=1" in msg


@pytest.mark.asyncio
async def test_publish_does_not_let_slow_subscriber_block_peers():
    """A full-queue subscriber must not head-of-line block its peers."""
    from gpustack.server.bus import EventBus

    bus = EventBus()
    topic = "_test_publish_fanout"
    slow = bus.subscribe(topic, source="slow")
    fast = bus.subscribe(topic, source="fast")
    slow.queue = asyncio.Queue(maxsize=1)
    await slow.enqueue(Event(type=EventType.CREATED, data={"id": 0}, id=0))

    try:
        await bus.publish(topic, Event(type=EventType.CREATED, data={"id": 1}, id=1))
        delivered = await asyncio.wait_for(fast.receive(), timeout=1)
        assert delivered.id == 1
        assert slow.queue.qsize() == 1  # still backpressured
    finally:
        bus.unsubscribe(topic, slow)
        bus.unsubscribe(topic, fast)


@pytest.mark.asyncio
async def test_cancelled_updated_put_rolls_back_latest_by_key():
    """If the producer task is cancelled while awaiting backpressure,
    ``latest_by_key`` must be rolled back so the next UPDATED for the same
    id can re-enter the queue. Without rollback this reproduces the
    #4794 stranded-id bug, just triggered by cancel rather than QueueFull.
    """
    subscriber = Subscriber(topic="modelinstance", source="test", queue_size=1)

    # Fill the queue with an unrelated event so the next put will block.
    await subscriber.enqueue(Event(type=EventType.CREATED, data={"id": 0}, id=0))

    # Start an UPDATED enqueue for id=42 — it writes latest_by_key[42]
    # then awaits put on the full queue.
    cancelled = asyncio.create_task(
        subscriber.enqueue(Event(type=EventType.UPDATED, data={"id": 42}, id=42))
    )
    for _ in range(5):
        await asyncio.sleep(0)
        if 42 in subscriber.latest_by_key:
            break
    assert 42 in subscriber.latest_by_key

    cancelled.cancel()
    try:
        await cancelled
    except asyncio.CancelledError:
        pass
    # Rollback should clear the orphan entry.
    assert 42 not in subscriber.latest_by_key

    # A fresh UPDATED for id=42 must be deliverable. Drain the prefill
    # first to avoid a second blocking put.
    drained = await asyncio.wait_for(subscriber.receive(), timeout=1)
    assert drained.id == 0
    await subscriber.enqueue(
        Event(type=EventType.UPDATED, data={"id": 42, "v": "fresh"}, id=42)
    )
    delivered = await asyncio.wait_for(subscriber.receive(), timeout=1)
    assert delivered.id == 42
    assert delivered.data["v"] == "fresh"


@pytest.mark.asyncio
async def test_non_updated_events_block_under_backpressure_not_drop():
    subscriber = Subscriber(topic="modelinstance", source="test", queue_size=2)

    await subscriber.enqueue(Event(type=EventType.CREATED, data={"id": 1}, id=1))
    await subscriber.enqueue(Event(type=EventType.CREATED, data={"id": 2}, id=2))
    pending = asyncio.create_task(
        subscriber.enqueue(Event(type=EventType.CREATED, data={"id": 3}, id=3))
    )
    await asyncio.sleep(0)
    assert not pending.done()

    first = await asyncio.wait_for(subscriber.receive(), timeout=1)
    assert first.id == 1
    await asyncio.wait_for(pending, timeout=1)

    second = await asyncio.wait_for(subscriber.receive(), timeout=1)
    third = await asyncio.wait_for(subscriber.receive(), timeout=1)
    assert {second.id, third.id} == {2, 3}


@pytest.mark.asyncio
async def test_unsubscribe_drains_pending_puts_and_releases_pending_tasks():
    """An SSE consumer that disconnects while its queue is full leaves
    enqueue tasks stuck on ``queue.put``. Without the close-on-unsubscribe
    drain, those tasks (held by the bus's ``_pending_tasks`` retain set)
    pin the subscriber + 1024 events alive forever — that's the
    ghost-subscriber half of the issue #5073 leak.
    """
    from gpustack.server.bus import EventBus

    bus = EventBus()
    topic = "_test_unsubscribe_drain"
    subscriber = bus.subscribe(topic, source="ghost")
    subscriber.queue = asyncio.Queue(maxsize=1)
    # Saturate the queue so the next route call will block on put.
    await subscriber.enqueue(Event(type=EventType.CREATED, data={"id": 0}, id=0))

    # Route a non-UPDATED event — fan-out spawns a task that blocks on put.
    bus._route_event(Event(type=EventType.CREATED, data={"id": 1}, id=1), topic)
    await asyncio.sleep(0)
    blocked = next(iter(bus._pending_tasks))
    assert not blocked.done()

    # Unsubscribe must close the subscriber, draining its queue so the
    # blocked enqueue task can finish and the retain-set discards it.
    qsize_before = subscriber.queue.qsize()
    bus.unsubscribe(topic, subscriber)
    await asyncio.wait_for(blocked, timeout=1)
    # The done callback on _spawn fires before ``await blocked`` returns,
    # so by here the retain set must have released the task.
    assert blocked not in bus._pending_tasks
    assert subscriber._closed is True
    # The drained event is gone; the previously-blocked put resolved and
    # placed its event back, so the net queue depth is at most what we
    # started with — no leak amplification.
    assert subscriber.queue.qsize() <= qsize_before

    # Post-close enqueues are silently dropped: no new entries reach the
    # queue, no new entries land in latest_by_key.
    pre_post_qsize = subscriber.queue.qsize()
    await subscriber.enqueue(Event(type=EventType.CREATED, data={"id": 2}, id=2))
    await subscriber.enqueue(Event(type=EventType.UPDATED, data={"id": 3}, id=3))
    assert subscriber.queue.qsize() == pre_post_qsize
    assert subscriber.latest_by_key == {}


@pytest.mark.asyncio
async def test_unsubscribe_unwinds_putters_beyond_queue_capacity():
    """Stalled consumer + high event rate parks more putters than the
    queue can hold. ``close`` must unwind every one of them — the surplus
    beyond ``maxsize`` cannot be reached by drain alone (each get_nowait
    wakes only one putter), so we also cancel residual ``_putters``.
    """
    from gpustack.server.bus import EventBus

    bus = EventBus()
    topic = "_test_unsubscribe_deep_backlog"
    subscriber = bus.subscribe(topic, source="stalled")
    subscriber.queue = asyncio.Queue(maxsize=2)

    # Fill the queue, then route enough additional events that the parked
    # putter count exceeds the queue's capacity by a wide margin.
    await subscriber.enqueue(Event(type=EventType.CREATED, data={"id": 0}, id=0))
    await subscriber.enqueue(Event(type=EventType.CREATED, data={"id": 1}, id=1))
    extra = 8
    for i in range(extra):
        bus._route_event(
            Event(type=EventType.CREATED, data={"id": 100 + i}, id=100 + i),
            topic,
        )
    # Yield so each enqueue task reaches its blocking put.
    for _ in range(extra + 2):
        await asyncio.sleep(0)
    blocked_tasks = list(bus._pending_tasks)
    assert len(blocked_tasks) == extra
    assert all(not t.done() for t in blocked_tasks)

    bus.unsubscribe(topic, subscriber)
    # All blocked enqueue tasks must finish — either via the woken
    # put-and-exit path (up to maxsize) or via the cancellation path.
    await asyncio.wait_for(
        asyncio.gather(*blocked_tasks, return_exceptions=True),
        timeout=1,
    )
    for t in blocked_tasks:
        assert t.done()
    assert all(t not in bus._pending_tasks for t in blocked_tasks)
    putters = getattr(subscriber.queue, "_putters", None)
    assert putters is None or len(putters) == 0
