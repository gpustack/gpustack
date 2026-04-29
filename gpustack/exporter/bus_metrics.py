"""Prometheus metrics for the in-process event bus, pulled at scrape time."""

from typing import Iterator

from prometheus_client.registry import Collector
from prometheus_client.core import (
    CounterMetricFamily,
    GaugeMetricFamily,
    Metric,
)

from gpustack.server.bus import event_bus
from gpustack.utils.name import metric_name


class BusMetricsCollector(Collector):
    def collect(self) -> Iterator[Metric]:
        subscribers = GaugeMetricFamily(
            metric_name("bus_subscribers"),
            "Number of active bus subscribers per topic.",
            labels=["topic"],
        )
        queue_depth = GaugeMetricFamily(
            metric_name("bus_queue_depth"),
            "Current per-subscriber queue depth at scrape time.",
            labels=["topic", "source"],
        )
        queue_capacity = GaugeMetricFamily(
            metric_name("bus_queue_capacity"),
            "Per-subscriber queue maxsize.",
            labels=["topic", "source"],
        )
        queue_full = GaugeMetricFamily(
            metric_name("bus_queue_full"),
            "1 if the subscriber queue is full at scrape time, 0 otherwise. "
            "A sustained 1 indicates a slow consumer holding back the bus.",
            labels=["topic", "source"],
        )
        queue_saturation = GaugeMetricFamily(
            metric_name("bus_queue_saturation_ratio"),
            "Per-subscriber queue depth as a fraction of capacity "
            "(qsize / maxsize).",
            labels=["topic", "source"],
        )
        latest_keys = GaugeMetricFamily(
            metric_name("bus_subscriber_latest_keys"),
            "Number of distinct ids pending coalesced UPDATED delivery "
            "per subscriber (size of latest_by_key).",
            labels=["topic", "source"],
        )
        events = CounterMetricFamily(
            metric_name("bus_events"),
            "Cumulative event counts per subscriber. Kinds: "
            "received, filtered, coalesced, enqueued, backpressured "
            "(see EventCountKind).",
            labels=["topic", "source", "kind", "event_type"],
        )

        # Copy in case subscribe/unsubscribe races with collection.
        snapshot = {topic: list(subs) for topic, subs in event_bus.subscribers.items()}

        for topic, subs in snapshot.items():
            subscribers.add_metric([topic], len(subs))
            for sub in subs:
                source = sub.source or ""
                qsize = sub.queue.qsize()
                maxsize = sub.queue.maxsize
                queue_depth.add_metric([topic, source], qsize)
                queue_capacity.add_metric([topic, source], maxsize)
                queue_full.add_metric([topic, source], 1.0 if sub.queue.full() else 0.0)
                queue_saturation.add_metric(
                    [topic, source], qsize / maxsize if maxsize else 0.0
                )
                latest_keys.add_metric([topic, source], len(sub.latest_by_key))
                for (kind, event_type_name), count in sub.event_counts.items():
                    events.add_metric(
                        [topic, source, kind.value, event_type_name], count
                    )

        yield subscribers
        yield queue_depth
        yield queue_capacity
        yield queue_full
        yield queue_saturation
        yield latest_keys
        yield events
