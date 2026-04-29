"""Tests for the pull-based bus metrics collector."""

import asyncio

import pytest

from gpustack.exporter.bus_metrics import BusMetricsCollector
from gpustack.server.bus import (
    Event,
    EventCountKind,
    EventType,
    Subscriber,
    event_bus,
)


def _sample_for(metrics, metric_name, label_match):
    for metric in metrics:
        if metric.name != metric_name:
            continue
        for sample in metric.samples:
            if all(sample.labels.get(k) == v for k, v in label_match.items()):
                return sample
    raise AssertionError(
        f"Sample for metric={metric_name} labels={label_match} not found"
    )


def _all_samples(metrics, metric_name):
    for metric in metrics:
        if metric.name == metric_name:
            return list(metric.samples)
    return []


@pytest.mark.asyncio
async def test_bus_metrics_collector_reflects_subscriber_state():
    topic = "_test_bus_metrics_topic"
    sub_a = event_bus.subscribe(topic, source="test_a")
    sub_b = event_bus.subscribe(topic, source="test_b", event_types={EventType.CREATED})

    try:
        # sub_a takes everything (second UPDATED for id=2 → COALESCED);
        # sub_b filters non-CREATED.
        await sub_a.enqueue(Event(type=EventType.CREATED, data={"id": 1}, id=1))
        await sub_a.enqueue(Event(type=EventType.UPDATED, data={"id": 2}, id=2))
        await sub_a.enqueue(Event(type=EventType.UPDATED, data={"id": 2}, id=2))
        await sub_b.enqueue(Event(type=EventType.UPDATED, data={"id": 3}, id=3))
        await sub_b.enqueue(Event(type=EventType.CREATED, data={"id": 4}, id=4))

        metrics = list(BusMetricsCollector().collect())

        # 2 subscribers under our topic.
        sample = _sample_for(metrics, "gpustack:bus_subscribers", {"topic": topic})
        assert sample.value == 2

        # sub_a — RECEIVED total = 3.
        sample = _sample_for(
            metrics,
            "gpustack:bus_events",
            {
                "topic": topic,
                "source": "test_a",
                "kind": EventCountKind.RECEIVED.value,
                "event_type": EventType.UPDATED.name,
            },
        )
        assert sample.value == 2  # two UPDATED events received

        # sub_a — second UPDATED merged into latest_by_key.
        sample = _sample_for(
            metrics,
            "gpustack:bus_events",
            {
                "topic": topic,
                "source": "test_a",
                "kind": EventCountKind.COALESCED.value,
                "event_type": EventType.UPDATED.name,
            },
        )
        assert sample.value == 1

        # sub_a — first UPDATED actually entered the queue.
        sample = _sample_for(
            metrics,
            "gpustack:bus_events",
            {
                "topic": topic,
                "source": "test_a",
                "kind": EventCountKind.ENQUEUED.value,
                "event_type": EventType.UPDATED.name,
            },
        )
        assert sample.value == 1

        # sub_a — CREATED enqueued.
        sample = _sample_for(
            metrics,
            "gpustack:bus_events",
            {
                "topic": topic,
                "source": "test_a",
                "kind": EventCountKind.ENQUEUED.value,
                "event_type": EventType.CREATED.name,
            },
        )
        assert sample.value == 1

        # sub_b — UPDATED received then filtered.
        sample = _sample_for(
            metrics,
            "gpustack:bus_events",
            {
                "topic": topic,
                "source": "test_b",
                "kind": EventCountKind.RECEIVED.value,
                "event_type": EventType.UPDATED.name,
            },
        )
        assert sample.value == 1
        sample = _sample_for(
            metrics,
            "gpustack:bus_events",
            {
                "topic": topic,
                "source": "test_b",
                "kind": EventCountKind.FILTERED.value,
                "event_type": EventType.UPDATED.name,
            },
        )
        assert sample.value == 1
        # Filter rejected before enqueue → no ENQUEUED for that event_type.
        for sample in _all_samples(metrics, "gpustack:bus_events"):
            if (
                sample.labels.get("topic") == topic
                and sample.labels.get("source") == "test_b"
                and sample.labels.get("event_type") == EventType.UPDATED.name
            ):
                assert sample.labels.get("kind") != EventCountKind.ENQUEUED.value

        # sub_b — CREATED received and enqueued.
        sample = _sample_for(
            metrics,
            "gpustack:bus_events",
            {
                "topic": topic,
                "source": "test_b",
                "kind": EventCountKind.ENQUEUED.value,
                "event_type": EventType.CREATED.name,
            },
        )
        assert sample.value == 1

        # No backpressure samples expected — queues weren't full.
        for sample in _all_samples(metrics, "gpustack:bus_events"):
            if sample.labels.get("topic") == topic:
                assert sample.labels.get("kind") != EventCountKind.BACKPRESSURED.value

        # queue_full gauge = 0 for both subs (queues have items but aren't full).
        for source in ("test_a", "test_b"):
            sample = _sample_for(
                metrics,
                "gpustack:bus_queue_full",
                {
                    "topic": topic,
                    "source": source,
                },
            )
            assert sample.value == 0
    finally:
        event_bus.unsubscribe(topic, sub_a)
        event_bus.unsubscribe(topic, sub_b)


@pytest.mark.asyncio
async def test_bus_metrics_collector_reports_backpressure_and_queue_full():
    topic = "_test_bus_metrics_qfull"
    sub = Subscriber(topic=topic, source="slow", queue_size=1)
    event_bus.subscribers.setdefault(topic, []).append(sub)

    try:
        await sub.enqueue(Event(type=EventType.CREATED, data={"id": 1}, id=1))

        pending = asyncio.create_task(
            sub.enqueue(Event(type=EventType.CREATED, data={"id": 2}, id=2))
        )
        # Yield so the enqueue task hits the full-queue branch.
        for _ in range(5):
            await asyncio.sleep(0)
            if sub.event_counts.get(
                (EventCountKind.BACKPRESSURED, EventType.CREATED.name)
            ):
                break
        pending.cancel()
        try:
            await pending
        except (asyncio.CancelledError, Exception):
            pass

        metrics = list(BusMetricsCollector().collect())

        sample = _sample_for(
            metrics,
            "gpustack:bus_events",
            {
                "topic": topic,
                "source": "slow",
                "kind": EventCountKind.BACKPRESSURED.value,
                "event_type": EventType.CREATED.name,
            },
        )
        assert sample.value >= 1

        sample = _sample_for(
            metrics,
            "gpustack:bus_queue_full",
            {
                "topic": topic,
                "source": "slow",
            },
        )
        assert sample.value == 1

        sample = _sample_for(
            metrics,
            "gpustack:bus_queue_saturation_ratio",
            {
                "topic": topic,
                "source": "slow",
            },
        )
        assert sample.value == 1.0

        sample = _sample_for(
            metrics,
            "gpustack:bus_queue_capacity",
            {
                "topic": topic,
                "source": "slow",
            },
        )
        assert sample.value == 1
    finally:
        event_bus.unsubscribe(topic, sub)


@pytest.mark.asyncio
async def test_received_equals_filtered_plus_coalesced_plus_enqueued():
    """Sanity: ``RECEIVED = FILTERED + COALESCED + ENQUEUED``."""
    topic = "_test_bus_metrics_invariant"
    sub = event_bus.subscribe(
        topic, source="invariant", event_types={EventType.CREATED, EventType.UPDATED}
    )

    try:
        # Mix: 2 CREATED (both enqueued), 3 UPDATED for id=10 (1 enqueued + 2
        # coalesced), 1 DELETED (filtered out by event_types).
        await sub.enqueue(Event(type=EventType.CREATED, data={"id": 1}, id=1))
        await sub.enqueue(Event(type=EventType.CREATED, data={"id": 2}, id=2))
        await sub.enqueue(Event(type=EventType.UPDATED, data={"id": 10}, id=10))
        await sub.enqueue(Event(type=EventType.UPDATED, data={"id": 10}, id=10))
        await sub.enqueue(Event(type=EventType.UPDATED, data={"id": 10}, id=10))
        await sub.enqueue(Event(type=EventType.DELETED, data={"id": 99}, id=99))

        def total(kind: EventCountKind) -> int:
            return sum(
                count for (k, _evt), count in sub.event_counts.items() if k is kind
            )

        assert total(EventCountKind.RECEIVED) == 6
        assert total(EventCountKind.FILTERED) == 1
        assert total(EventCountKind.COALESCED) == 2
        assert total(EventCountKind.ENQUEUED) == 3
        assert total(EventCountKind.RECEIVED) == total(EventCountKind.FILTERED) + total(
            EventCountKind.COALESCED
        ) + total(EventCountKind.ENQUEUED)
    finally:
        event_bus.unsubscribe(topic, sub)
