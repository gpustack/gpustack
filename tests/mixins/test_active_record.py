"""Regression tests for the watch heartbeat in ActiveRecordMixin.subscribe().

The heartbeat used to fire only when the subscriber queue went quiet, so a
topic busier than one event per interval starved it entirely. Combined with
streaming() dropping events that don't belong to the connection, the response
body went byte-silent until the client's read timeout fired.
"""

import asyncio
import json
from contextlib import suppress
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gpustack.mixins import active_record
from gpustack.mixins.active_record import ActiveRecordMixin
from gpustack.server.bus import Event, EventType, event_bus

HEARTBEAT_BYTES = "\n\n"
TEST_INTERVAL = timedelta(seconds=0.05)


class HeartbeatProbe(ActiveRecordMixin):
    """Stands in for a real model: no table, no DB, and its own bus topic."""

    @classmethod
    async def cached_all(cls, options=None):
        return []


TOPIC = HeartbeatProbe.__name__.lower()


async def _publish(event_id: int, event_type: EventType):
    await event_bus.publish(
        TOPIC,
        Event(
            type=event_type,
            data=SimpleNamespace(id=event_id, worker_id=1),
            id=event_id,
        ),
    )


async def _flood(stop: asyncio.Event, every: float):
    """Publish faster than the heartbeat interval until told to stop."""
    event_id = 0
    while not stop.is_set():
        event_id += 1
        await _publish(event_id, EventType.CREATED)
        await asyncio.sleep(every)


async def _publish_updates(total: int, every: float):
    for event_id in range(1, total + 1):
        await _publish(event_id, EventType.UPDATED)
        await asyncio.sleep(every)


async def _take(stream, count: int) -> list:
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
        if len(chunks) == count:
            break
    return chunks


async def _drain_events(stream, total: int) -> tuple:
    """Collect ``total`` real events, counting the heartbeats in between."""
    received_ids = []
    heartbeats = 0
    async for event in stream:
        if event.type == EventType.HEARTBEAT:
            heartbeats += 1
            continue
        received_ids.append(event.data.id)
        if len(received_ids) == total:
            break
    return received_ids, heartbeats


@pytest.mark.asyncio
async def test_heartbeat_survives_a_flood_of_filtered_events():
    """A busy topic whose events this connection filters out still gets bytes."""
    interval = TEST_INTERVAL.total_seconds()
    stop = asyncio.Event()
    flood = asyncio.create_task(_flood(stop, interval / 5))

    with patch.object(active_record, "HEARTBEAT_INTERVAL", TEST_INTERVAL):
        # worker_id never matches, so every real event is dropped by streaming().
        stream = HeartbeatProbe.streaming(fields={"worker_id": 999})
        started = asyncio.get_running_loop().time()
        try:
            chunks = await asyncio.wait_for(_take(stream, 3), timeout=interval * 30)
        finally:
            stop.set()
            await flood
            with suppress(Exception):
                await stream.aclose()
        elapsed = asyncio.get_running_loop().time() - started

    # Only heartbeats reach the wire: the filter still drops every real event.
    assert chunks == [HEARTBEAT_BYTES] * 3
    # Peak rate is unchanged, so three heartbeats span at least two intervals.
    assert elapsed >= interval * 2


@pytest.mark.asyncio
async def test_heartbeat_survives_a_queue_that_is_never_empty():
    """A backlog makes every receive() return at once; the deadline still wins."""
    interval = TEST_INTERVAL.total_seconds()

    with patch.object(active_record, "HEARTBEAT_INTERVAL", TEST_INTERVAL):
        stream = HeartbeatProbe.streaming(fields={"worker_id": 999})
        take = asyncio.create_task(_take(stream, 3))
        for _ in range(100):
            await asyncio.sleep(0)
            if event_bus.subscribers.get(TOPIC):
                break
        # Enqueue straight into the subscriber so the backlog is deterministic:
        # publishing goes through spawned tasks that may drain as they arrive.
        subscriber = event_bus.subscribers[TOPIC][0]
        for event_id in range(1, 501):
            await subscriber.enqueue(
                Event(
                    type=EventType.CREATED,
                    data=SimpleNamespace(id=event_id, worker_id=1),
                    id=event_id,
                )
            )
        try:
            chunks = await asyncio.wait_for(take, timeout=interval * 30)
        finally:
            with suppress(Exception):
                await stream.aclose()

    assert chunks == [HEARTBEAT_BYTES] * 3


@pytest.mark.asyncio
async def test_matching_events_still_reach_the_wire_as_json():
    """The filter's positive half: a matching event is still serialized out.

    Left on the real interval so no heartbeat can race the event to the wire.
    """
    stream = HeartbeatProbe.streaming(fields={"worker_id": 1})
    take = asyncio.create_task(_take(stream, 1))
    for _ in range(100):
        await asyncio.sleep(0)
        if event_bus.subscribers.get(TOPIC):
            break
    await _publish(7, EventType.CREATED)
    try:
        chunks = await asyncio.wait_for(take, timeout=5)
    finally:
        with suppress(Exception):
            await stream.aclose()

    assert chunks != [HEARTBEAT_BYTES]
    payload = json.loads(chunks[0])
    assert payload["data"]["id"] == 7


@pytest.mark.asyncio
async def test_heartbeats_do_not_drop_in_flight_events():
    """Firing a heartbeat must not cancel a receive() that already dequeued."""
    interval = TEST_INTERVAL.total_seconds()
    total = 20

    with patch.object(active_record, "HEARTBEAT_INTERVAL", TEST_INTERVAL):
        stream = HeartbeatProbe.subscribe(source="test", replay_existing=False)
        drain = asyncio.create_task(_drain_events(stream, total))
        # The generator body — and with it the bus registration — only runs on
        # the first __anext__, so publishing before that would go nowhere.
        for _ in range(100):
            await asyncio.sleep(0)
            if event_bus.subscribers.get(TOPIC):
                break
        publisher = asyncio.create_task(_publish_updates(total, interval / 4))
        try:
            received_ids, heartbeats = await asyncio.wait_for(
                drain, timeout=interval * 60
            )
        finally:
            await publisher
            with suppress(Exception):
                await stream.aclose()

    assert received_ids == list(range(1, total + 1))
    # Heartbeats really did interleave, so the delivery guarantee was exercised.
    assert heartbeats >= 1
