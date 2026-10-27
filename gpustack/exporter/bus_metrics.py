"""Prometheus metrics for the in-process event bus, pulled at scrape time."""

from typing import Dict, Iterator, Tuple

from prometheus_client.registry import Collector
from prometheus_client.core import (
    CounterMetricFamily,
    GaugeMetricFamily,
    Metric,
)

from gpustack.server.bus import event_bus
from gpustack.utils.name import metric_name


class BusMetricsCollector(Collector):
    """Expose event-bus health as Prometheus metrics.

    A ``(topic, source)`` pair does NOT uniquely identify a subscriber:
    every open watch/SSE stream subscribes with ``source="streaming"``, so a
    hot topic routinely carries several subscribers sharing one label set.
    Emitting one series per subscriber would then produce duplicate label sets
    (invalid exposition, see #5902), and adding a per-subscriber label would
    blow up cardinality as ephemeral streams come and go. So metrics are
    aggregated per ``(topic, source)``: the worst-case (max) subscriber for the
    gauges — which is what "is a consumer backpressuring?" cares about — and
    the sum for cumulative event counts. ``bus_subscribers`` additionally gains
    a ``source`` dimension so the collapsed subscriber count stays visible.

    The event counter sum spans both live subscribers and the bus-level
    ``retired_event_counts`` accumulator (counts of already-unsubscribed
    subscribers). Without the retired term, a short-lived source's series
    would drop as streams close and Prometheus would read the drop as a
    counter reset, so ``bus_events_total`` stays monotonic.
    """

    def collect(self) -> Iterator[Metric]:
        subscribers = GaugeMetricFamily(
            metric_name("bus_subscribers"),
            "Number of active bus subscribers per topic and source. "
            "Several may share source=streaming (one per open watch stream).",
            labels=["topic", "source"],
        )
        queue_depth = GaugeMetricFamily(
            metric_name("bus_queue_depth"),
            "Max queue depth at scrape time across subscribers sharing "
            "topic+source (worst-case consumer).",
            labels=["topic", "source"],
        )
        queue_capacity = GaugeMetricFamily(
            metric_name("bus_queue_capacity"),
            "Per-subscriber queue maxsize (identical across subscribers "
            "sharing topic+source).",
            labels=["topic", "source"],
        )
        queue_full = GaugeMetricFamily(
            metric_name("bus_queue_full"),
            "1 if any subscriber sharing topic+source has a full queue at "
            "scrape time, 0 otherwise. A sustained 1 indicates a slow "
            "consumer holding back the bus.",
            labels=["topic", "source"],
        )
        queue_saturation = GaugeMetricFamily(
            metric_name("bus_queue_saturation_ratio"),
            "Max queue depth as a fraction of capacity (qsize / maxsize) "
            "across subscribers sharing topic+source.",
            labels=["topic", "source"],
        )
        latest_keys = GaugeMetricFamily(
            metric_name("bus_subscriber_latest_keys"),
            "Max number of distinct ids pending coalesced UPDATED delivery "
            "(size of latest_by_key) across subscribers sharing topic+source.",
            labels=["topic", "source"],
        )
        events = CounterMetricFamily(
            metric_name("bus_events"),
            "Cumulative event counts summed across subscribers sharing "
            "topic+source. Kinds: received, filtered, coalesced, enqueued, "
            "backpressured (see EventCountKind).",
            labels=["topic", "source", "kind", "event_type"],
        )

        # Copy in case subscribe/unsubscribe races with collection.
        snapshot = {topic: list(subs) for topic, subs in event_bus.subscribers.items()}

        # Aggregate per (topic, source) so subscribers sharing a label set
        # (notably source="streaming", one per open watch stream) collapse to
        # a single series instead of emitting duplicate lines.
        count: Dict[Tuple[str, str], int] = {}
        depth_max: Dict[Tuple[str, str], int] = {}
        capacity: Dict[Tuple[str, str], int] = {}
        full: Dict[Tuple[str, str], bool] = {}
        saturation_max: Dict[Tuple[str, str], float] = {}
        latest_keys_max: Dict[Tuple[str, str], int] = {}
        # Seed with the counts of already-unsubscribed subscribers so the
        # counter never drops when a subscriber departs. list() copies in
        # case unsubscribe folds a new entry in mid-collection.
        events_sum: Dict[Tuple[str, str, str, str], int] = dict(
            list(event_bus.retired_event_counts.items())
        )

        for topic, subs in snapshot.items():
            for sub in subs:
                key = (topic, sub.source or "")
                qsize = sub.queue.qsize()
                maxsize = sub.queue.maxsize
                count[key] = count.get(key, 0) + 1
                depth_max[key] = max(depth_max.get(key, 0), qsize)
                capacity[key] = max(capacity.get(key, 0), maxsize)
                full[key] = full.get(key, False) or sub.queue.full()
                saturation_max[key] = max(
                    saturation_max.get(key, 0.0),
                    qsize / maxsize if maxsize else 0.0,
                )
                latest_keys_max[key] = max(
                    latest_keys_max.get(key, 0), len(sub.latest_by_key)
                )
                # Copy the items: ``enqueue`` mutates ``event_counts`` on the
                # event loop while ``/metrics`` collects from the scrape thread,
                # so iterating the live dict can raise "dictionary changed size
                # during iteration".
                for (kind, event_type_name), c in list(sub.event_counts.items()):
                    ekey = (topic, sub.source or "", kind.value, event_type_name)
                    events_sum[ekey] = events_sum.get(ekey, 0) + c

        for (topic, source), n in count.items():
            subscribers.add_metric([topic, source], n)
            queue_depth.add_metric([topic, source], depth_max[(topic, source)])
            queue_capacity.add_metric([topic, source], capacity[(topic, source)])
            queue_full.add_metric(
                [topic, source], 1.0 if full[(topic, source)] else 0.0
            )
            queue_saturation.add_metric(
                [topic, source], saturation_max[(topic, source)]
            )
            latest_keys.add_metric([topic, source], latest_keys_max[(topic, source)])

        for (topic, source, kind, event_type_name), total in events_sum.items():
            events.add_metric([topic, source, kind, event_type_name], total)

        yield subscribers
        yield queue_depth
        yield queue_capacity
        yield queue_full
        yield queue_saturation
        yield latest_keys
        yield events
