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

        # 1 subscriber per (topic, source) — the metric is now source-scoped.
        for source in ("test_a", "test_b"):
            sample = _sample_for(
                metrics,
                "gpustack:bus_subscribers",
                {"topic": topic, "source": source},
            )
            assert sample.value == 1

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
async def test_subscribers_sharing_source_collapse_to_one_series():
    """#5902: multiple subscribers with the same (topic, source) — e.g. one
    per open source="streaming" watch stream — must aggregate into a single
    series instead of emitting duplicate label sets.
    """
    topic = "_test_bus_metrics_collapse"
    # Three streaming subscribers on the same topic, mirroring three open
    # watch connections.
    subs = [event_bus.subscribe(topic, source="streaming") for _ in range(3)]

    try:
        # Give each a different queue depth so max-aggregation is observable.
        await subs[0].enqueue(Event(type=EventType.CREATED, data={"id": 1}, id=1))
        await subs[1].enqueue(Event(type=EventType.CREATED, data={"id": 1}, id=1))
        await subs[1].enqueue(Event(type=EventType.CREATED, data={"id": 2}, id=2))
        # subs[2] left empty.

        metrics = list(BusMetricsCollector().collect())

        label = {"topic": topic, "source": "streaming"}

        # Exactly one series per metric for this label set — no duplicates.
        for metric_name in (
            "gpustack:bus_subscribers",
            "gpustack:bus_queue_depth",
            "gpustack:bus_queue_capacity",
            "gpustack:bus_queue_full",
            "gpustack:bus_queue_saturation_ratio",
            "gpustack:bus_subscriber_latest_keys",
        ):
            matching = [
                s
                for s in _all_samples(metrics, metric_name)
                if all(s.labels.get(k) == v for k, v in label.items())
            ]
            assert len(matching) == 1, (
                f"{metric_name} emitted {len(matching)} series for {label}, "
                "expected 1 (duplicate label sets)"
            )

        # bus_subscribers = 3 collapsed subscribers.
        assert _sample_for(metrics, "gpustack:bus_subscribers", label).value == 3
        # queue_depth = max across subscribers (subs[1] has 2 queued).
        assert _sample_for(metrics, "gpustack:bus_queue_depth", label).value == 2
        # bus_events CREATED ENQUEUED summed across all three (1 + 2 + 0 = 3).
        assert (
            _sample_for(
                metrics,
                "gpustack:bus_events",
                {
                    **label,
                    "kind": EventCountKind.ENQUEUED.value,
                    "event_type": EventType.CREATED.name,
                },
            ).value
            == 3
        )
    finally:
        for sub in subs:
            event_bus.unsubscribe(topic, sub)


@pytest.mark.asyncio
async def test_bus_events_total_stays_monotonic_across_unsubscribe():
    """A departing subscriber's counts must not drop the summed counter.

    Short-lived source="streaming" subscribers unsubscribe when their watch
    stream closes; folding their counts into the bus-level accumulator keeps
    bus_events_total monotonic so rate()/increase() stay correct.
    """
    topic = "_test_bus_metrics_monotonic"
    sub_a = event_bus.subscribe(topic, source="streaming")
    sub_b = event_bus.subscribe(topic, source="streaming")
    sub_c = None

    def enqueued_total():
        return _sample_for(
            list(BusMetricsCollector().collect()),
            "gpustack:bus_events",
            {
                "topic": topic,
                "source": "streaming",
                "kind": EventCountKind.ENQUEUED.value,
                "event_type": EventType.CREATED.name,
            },
        ).value

    try:
        await sub_a.enqueue(Event(type=EventType.CREATED, data={"id": 1}, id=1))
        await sub_a.enqueue(Event(type=EventType.CREATED, data={"id": 2}, id=2))
        await sub_b.enqueue(Event(type=EventType.CREATED, data={"id": 3}, id=3))

        before = enqueued_total()
        assert before == 3  # 2 from sub_a + 1 from sub_b

        # Closing one stream must not lower the counter.
        event_bus.unsubscribe(topic, sub_a)
        after = enqueued_total()
        assert after == before == 3

        # A new stream keeps accumulating on top of the retired total.
        sub_c = event_bus.subscribe(topic, source="streaming")
        await sub_c.enqueue(Event(type=EventType.CREATED, data={"id": 4}, id=4))
        assert enqueued_total() == 4
    finally:
        for sub in (sub_a, sub_b, sub_c):
            if sub is not None:
                event_bus.unsubscribe(topic, sub)
        event_bus.retired_event_counts.clear()


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
