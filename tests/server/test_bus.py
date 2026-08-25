import asyncio
import logging
from types import SimpleNamespace
from typing import Optional
from unittest.mock import patch

import pytest
from sqlmodel import SQLModel

from gpustack.mixins.active_record import CommitEvent, send_post_commit_events

from gpustack.server.bus import (
    Event,
    EventBus,
    EventType,
    Subscriber,
    event_field,
    resolve_event_id,
)
from gpustack.server.coordinator.cache import clear_all_caches, get_change_detector


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


class _PersistedRow(SQLModel):
    """A real model, for the paths that ``model_copy`` the payload."""

    id: Optional[int] = None
    name: Optional[str] = None


class _Row:
    """Stand-in for a SQLModel row: attribute access, and an ``id``."""

    def __init__(self, id, name=None, cluster_id=None):
        self.id = id
        self.name = name
        self.cluster_id = cluster_id


def _crossed_instances(event: Event) -> Event:
    """The event as another instance receives it, id-only payload and all.

    Mirrors what a coordinator does: ``to_dict`` on the publishing side,
    ``from_dict`` on the receiving one.
    """
    return Event.from_dict(event.to_dict())


def test_cross_instance_delete_payload_is_id_only():
    """The shape every consumer has to tolerate, pinned.

    A delete crossing instances carries ``{"id": N}``: to_dict strips the row
    down to its id, and by the time it lands the row is gone, so nothing can
    re-hydrate it. Attribute access is what breaks, and it breaks only on the
    replica that did not serve the write -- which is why it survives testing.
    """
    received = _crossed_instances(Event(type=EventType.DELETED, data=_Row(42, "r")))

    assert received.data == {"id": 42}
    with pytest.raises(AttributeError):
        received.data.id


def test_event_id_survives_the_serialization_round_trip():
    """``resolve_event_id`` can just read ``Event.id`` because both paths fill
    it: ``__post_init__`` derives it from a hydrated row, and ``to_dict`` /
    ``from_dict`` carry it explicitly."""
    hydrated = Event(type=EventType.DELETED, data=_Row(42, "r"))

    assert resolve_event_id(hydrated) == 42
    assert resolve_event_id(_crossed_instances(hydrated)) == 42


def test_a_created_event_has_no_id_until_the_insert_assigns_one():
    """Why :meth:`Event.refresh_id` exists.

    ``ActiveRecordMixin.create`` queues the event and *then* calls ``save``,
    so the Event is built while the row's primary key is still None. Deriving
    at construction is all ``__post_init__`` can do, and for a create it
    derives nothing.
    """
    pre_insert = Event(type=EventType.CREATED, data=_Row(None, "r"))
    assert resolve_event_id(pre_insert) is None

    pre_insert.data.id = 42  # the INSERT
    pre_insert.refresh_id()

    assert resolve_event_id(pre_insert) == 42


def test_refresh_id_does_not_overwrite_an_id_that_arrived_explicitly():
    """Over the wire the id is all that survives and ``data`` is stripped, so
    an explicit id outranks whatever can be read back off the payload."""
    crossed = Event(type=EventType.DELETED, data={"id": 42}, id=42)
    crossed.data = {}
    crossed.refresh_id()

    assert resolve_event_id(crossed) == 42


@pytest.mark.asyncio
async def test_the_commit_hook_publishes_a_created_event_with_its_id():
    """The gap this closes: every consumer reading ``Event.id`` rather than
    ``data.id`` saw None on every create, so anything that returned early on a
    missing id silently did nothing until a restart replayed the row."""
    row = _PersistedRow(id=None, name="r")
    commit_event = CommitEvent(name="modelroute", type=EventType.CREATED, data=row)
    assert commit_event.event.id is None, "built before the flush"

    row.id = 42  # the INSERT, during save()

    published = []

    class _RecordingBus:
        async def publish(self, name, event):
            published.append((name, event))

    session = SimpleNamespace(info={"pending_events": [commit_event]})
    with patch("gpustack.mixins.active_record.event_bus", _RecordingBus()):
        send_post_commit_events(session)
        await asyncio.sleep(0)

    assert len(published) == 1
    name, event = published[0]
    assert name == "modelroute"
    assert resolve_event_id(event) == 42
    # Derived after the detaching copy, so the two can never disagree.
    assert event.id == event.data.id


def test_resolve_event_id_survives_a_dataless_event():
    assert resolve_event_id(Event(type=EventType.DELETED, data=None, id=7)) == 7
    assert resolve_event_id(Event(type=EventType.HEARTBEAT, data=None)) is None


def test_event_field_reads_both_shapes_and_falls_back():
    hydrated = Event(type=EventType.DELETED, data=_Row(42, "r", cluster_id=9))
    crossed = _crossed_instances(hydrated)

    assert event_field(hydrated.data, "cluster_id") == 9
    # Gone from the id-only payload: must read as "unknown", not as a value.
    assert event_field(crossed.data, "cluster_id") is None
    assert event_field(crossed.data, "cluster_id", "?") == "?"
    assert event_field(None, "cluster_id", "?") == "?"


def test_event_field_treats_an_explicit_none_as_absent():
    """A null column and a missing key both mean "nothing to act on" -- callers
    branch on one check, not two."""
    assert event_field(_Row(1, name=None), "name", "fallback") == "fallback"
    assert event_field({"id": 1, "name": None}, "name", "fallback") == "fallback"


def test_principal_topic_is_registered_for_cross_instance_enrichment():
    """Identity consolidation renamed the topic out from under the registry.

    ``User is Principal``, and subscribe() keys the topic off ``__name__``, so
    events publish to 'principal'. While only the stale 'user' key was
    registered, _process_coordinator_event dropped every cross-instance
    principal event before it could reach a subscriber.
    """
    from gpustack.schemas.users import User
    from gpustack.server.coordinator.models import get_model_for_topic

    assert User.__name__.lower() == "principal"
    assert get_model_for_topic("principal") is not None


async def _deliver(event_type, warm_cache):
    """Push a wire-shaped event through the bus and return what a subscriber got.

    Mirrors the receiving side of a cross-instance hop: the coordinator hands
    _process_coordinator_event an id-only payload, and whatever comes out the
    other side is what every consumer must be written against.
    """
    clear_all_caches()
    bus = EventBus()
    subscriber = bus.subscribe("modelroute", source="test")

    with (
        patch("gpustack.server.bus.get_model_for_topic", return_value=_FakeRow),
        patch("gpustack.server.db.async_session", _FakeSession),
    ):
        if warm_cache:
            # Only a prior cross-instance non-DELETE warms the detector.
            await bus._process_coordinator_event(
                Event(type=EventType.UPDATED, data={"id": 5}, id=5), "modelroute"
            )
            await asyncio.sleep(0.05)
            while not subscriber.queue.empty():
                subscriber.queue.get_nowait()

        await bus._process_coordinator_event(
            Event(type=event_type, data={"id": 5}, id=5), "modelroute"
        )
        await asyncio.sleep(0.05)

    if subscriber.queue.empty():
        return None
    return subscriber.queue.get_nowait().data


class _FakeRow:
    def __init__(self, id):
        self.id = id

    @classmethod
    async def one_by_id(cls, session, id, **kwargs):
        return cls(id)


class _FakeSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeDistributedCoordinator:
    """Stands in for an out-of-tree coordinator: events reach us from peers."""

    is_distributed = True

    def subscribe(self, channel, callback):
        pass


class _FakeLocalCoordinator:
    """Stands in for LocalCoordinator: nothing reaches us from anywhere else."""

    is_distributed = False


@pytest.mark.asyncio
@pytest.mark.parametrize("event_type", [EventType.CREATED, EventType.UPDATED])
async def test_non_delete_events_are_always_delivered_hydrated(event_type):
    """The invariant the consumer guards rest on.

    A CREATED/UPDATED that crosses instances is re-read from the database
    before delivery, so it never reaches a subscriber id-only. That is why a
    handler may guard on the payload shape without skipping any update -- the
    guard can only ever fire for a delete.
    """
    delivered = await _deliver(event_type, warm_cache=False)

    assert isinstance(delivered, _FakeRow)
    assert delivered.id == 5


@pytest.mark.asyncio
async def test_only_a_delete_with_a_cold_cache_is_delivered_id_only():
    """The one path that produces the dict -- and the one that hides it.

    With the detector warm the delete is enriched from cache and arrives
    hydrated, which is why this reproduces on some instances and not others.
    """
    assert await _deliver(EventType.DELETED, warm_cache=False) == {"id": 5}
    assert isinstance(await _deliver(EventType.DELETED, warm_cache=True), _FakeRow)


@pytest.mark.asyncio
async def test_locally_published_row_warms_the_change_detector():
    """A row this instance created must be enrichable when another deletes it.

    The detector used to be warmed only by cross-instance events, so a row
    created here was never in our own cache -- create on A, delete on B, and
    A got a payload it could not act on. Publishing locally now warms it, so
    the later cross-instance DELETE is enriched from cache instead.
    """
    clear_all_caches()
    bus = EventBus()
    bus.set_coordinator(_FakeDistributedCoordinator())
    subscriber = bus.subscribe("modelroute", source="test")
    row = _FakeRow(5)

    with (
        patch("gpustack.server.bus.get_model_for_topic", return_value=_FakeRow),
        patch("gpustack.server.db.async_session", _FakeSession),
    ):
        # The local publish path: data is the row itself, not an id-only dict.
        await bus._process_coordinator_event(
            Event(type=EventType.CREATED, data=row, id=5), "modelroute"
        )
        await asyncio.sleep(0.05)
        while not subscriber.queue.empty():
            subscriber.queue.get_nowait()

        assert get_change_detector("modelroute").get(5) is row

        # Now the delete arrives from the instance that served it.
        await bus._process_coordinator_event(
            Event(type=EventType.DELETED, data={"id": 5}, id=5), "modelroute"
        )
        await asyncio.sleep(0.05)

    delivered = subscriber.queue.get_nowait().data
    assert delivered is row, "delete should have been enriched from the local write"


@pytest.mark.asyncio
async def test_local_delete_forgets_the_row():
    """The counterpart: a row deleted here must not linger in the cache."""
    clear_all_caches()
    bus = EventBus()
    bus.set_coordinator(_FakeDistributedCoordinator())
    bus.subscribe("modelroute", source="test")
    row = _FakeRow(5)

    with (
        patch("gpustack.server.bus.get_model_for_topic", return_value=_FakeRow),
        patch("gpustack.server.db.async_session", _FakeSession),
    ):
        await bus._process_coordinator_event(
            Event(type=EventType.CREATED, data=row, id=5), "modelroute"
        )
        await asyncio.sleep(0.05)
        assert get_change_detector("modelroute").get(5) is row

        await bus._process_coordinator_event(
            Event(type=EventType.DELETED, data=row, id=5), "modelroute"
        )
        await asyncio.sleep(0.05)

    assert get_change_detector("modelroute").get(5) is None


def test_event_field_rejects_being_handed_the_event():
    """The asymmetry with resolve_event_id is easy to trip over, and reading a
    field off an Event would otherwise silently yield the default."""
    event = Event(type=EventType.DELETED, data=_Row(42, "r"), id=42)

    with pytest.raises(TypeError, match="takes Event.data"):
        event_field(event, "name")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "coordinator, topic, reason",
    [
        (_FakeLocalCoordinator(), "modelroute", "single-node"),
        (_FakeDistributedCoordinator(), "_unregistered_topic", "unregistered topic"),
        (None, "modelroute", "no coordinator"),
    ],
)
async def test_local_row_is_not_cached_where_nothing_could_read_it(
    coordinator, topic, reason
):
    """The cache is only ever read from the id-only branch, so writing to it
    outside the cases that reach that branch is pure cost -- up to maxsize
    detached rows per topic, and on an append-only table with monotonic ids
    the LRU never even hits.

    Single-node is the default deployment and has no cross-instance path at
    all; an unregistered topic returns before the lookup in every mode.
    """
    clear_all_caches()
    bus = EventBus()
    if coordinator is not None:
        bus.set_coordinator(coordinator)
    bus.subscribe(topic, source="test")

    with (
        patch("gpustack.server.bus.get_model_for_topic", return_value=None),
        patch("gpustack.server.db.async_session", _FakeSession),
    ):
        await bus._process_coordinator_event(
            Event(type=EventType.CREATED, data=_FakeRow(5), id=5), topic
        )
        await asyncio.sleep(0.05)

    assert get_change_detector(topic).get(5) is None, f"cached despite {reason}"


@pytest.mark.asyncio
async def test_local_write_sharpens_changed_fields_of_a_later_remote_event():
    """Warming the detector locally also moves the diff baseline forward.

    ``changed_fields`` is computed against the last state this instance knew.
    Before, that was only the startup snapshot or the last cross-instance
    event, so a locally-served write was invisible to it and the next remote
    event diffed against a stale baseline. Several handlers gate on this
    (``"state"``, ``"labels"``, ``"deleted_at"``), so the sharper baseline
    means fewer missed transitions.
    """
    clear_all_caches()
    bus = EventBus()
    bus.set_coordinator(_FakeDistributedCoordinator())
    subscriber = bus.subscribe("worker", source="test")

    served_here = _FakeRow(5)
    served_here.state = "READY"
    remote = _FakeRow(5)
    remote.state = "NOT_READY"

    async def _returns_remote(cls, session, id, **kwargs):
        return remote

    with (
        patch("gpustack.server.bus.get_model_for_topic", return_value=_FakeRow),
        patch("gpustack.server.db.async_session", _FakeSession),
        patch.object(_FakeRow, "one_by_id", classmethod(_returns_remote)),
    ):
        # A write this instance served: state becomes READY.
        await bus._process_coordinator_event(
            Event(type=EventType.UPDATED, data=served_here, id=5), "worker"
        )
        await asyncio.sleep(0.05)
        # Drain via receive(), not get_nowait(): UPDATED events coalesce on
        # latest_by_key, and only receive() pops that.
        await asyncio.wait_for(subscriber.receive(), timeout=1)

        # Then a peer flips it back, and we learn about it id-only.
        await bus._process_coordinator_event(
            Event(type=EventType.UPDATED, data={"id": 5}, id=5), "worker"
        )
        await asyncio.sleep(0.05)

    delivered = subscriber.queue.get_nowait()
    assert delivered.changed_fields.get("state") == ("READY", "NOT_READY")
