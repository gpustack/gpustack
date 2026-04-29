import asyncio
import logging
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from gpustack.envs import EVENT_BUS_SUBSCRIBER_QUEUE_SIZE

# Re-export from coordinator.base for backward compatibility
from gpustack.server.coordinator.base import Event, EventType
from gpustack.server.coordinator.cache import get_change_detector
from gpustack.server.coordinator.models import get_model_for_topic

logger = logging.getLogger(__name__)


class EventCountKind(Enum):
    """Subscriber-side counter buckets surfaced as Prometheus labels.

    On the normal completion path:
    ``RECEIVED = FILTERED + COALESCED + ENQUEUED`` and
    ``BACKPRESSURED ⊆ ENQUEUED``. If a ``put`` is cancelled mid-flight,
    BACKPRESSURED may have been bumped without a matching ENQUEUED — the
    ``latest_by_key`` rollback in ``enqueue`` keeps the queue/dict
    invariant intact, but the counter invariant is best-effort.
    """

    RECEIVED = "received"
    FILTERED = "filtered"
    COALESCED = "coalesced"
    ENQUEUED = "enqueued"
    BACKPRESSURED = "backpressured"


# Re-export for backward compatibility
__all__ = [
    'Event',
    'EventType',
    'EventCountKind',
    'Subscriber',
    'EventBus',
    'event_bus',
    'set_coordinator',
    'event_decoder',
]


def event_decoder(obj):
    if "type" in obj:
        obj["type"] = EventType[obj["type"]]
    return obj


class Subscriber:
    """A bus subscriber owning its own bounded queue.

    UPDATED events for the same id are coalesced via ``latest_by_key``.
    Invariant: ``id ∈ latest_by_key`` iff there is (or will be) a queue
    token whose ``receive()`` will pop it. When the queue is full the
    producer awaits ``put`` rather than dropping. Publish paths spawn
    enqueue in their own tasks (see ``EventBus._route_event``), so
    backpressure stalls only the per-event task, not the caller.
    """

    def __init__(
        self,
        topic: Optional[str] = None,
        source: Optional[str] = None,
        event_types: Optional[Iterable[EventType]] = None,
        queue_size: Optional[int] = None,
    ):
        self.topic = topic
        self.source = source
        self.event_types: Optional[Set[EventType]] = (
            set(event_types) if event_types else None
        )
        self.queue: asyncio.Queue = asyncio.Queue(
            maxsize=(
                queue_size
                if queue_size is not None
                else EVENT_BUS_SUBSCRIBER_QUEUE_SIZE
            )
        )
        self.latest_by_key: Dict[Any, Event] = {}
        self.lock = asyncio.Lock()
        # Read by ``BusMetricsCollector`` and reflected as Prometheus counters.
        self.event_counts: Dict[Tuple[EventCountKind, str], int] = {}

    def _bump(self, kind: EventCountKind, event_type: EventType) -> None:
        key = (kind, event_type.name)
        self.event_counts[key] = self.event_counts.get(key, 0) + 1

    def should_enqueue(self, event: Event) -> bool:
        """Pre-enqueue filter. Drops events the subscriber has opted out of."""
        if self.event_types is not None and event.type not in self.event_types:
            return False
        return True

    async def enqueue(self, event: Event):
        self._bump(EventCountKind.RECEIVED, event.type)

        if not self.should_enqueue(event):
            self._bump(EventCountKind.FILTERED, event.type)
            return

        if event.type == EventType.UPDATED and event.id is not None:
            async with self.lock:
                if event.id in self.latest_by_key:
                    self.latest_by_key[event.id] = event
                    self._bump(EventCountKind.COALESCED, event.type)
                    return
                self.latest_by_key[event.id] = event
            # Release the lock before awaiting put: a full queue would
            # otherwise serialize unrelated ids behind it.
            try:
                await self._put_with_backpressure(event)
            except BaseException:
                # If the put was cancelled or errored before a token reached
                # the queue, neither this event nor any later UPDATED that
                # piggybacked on the same dict entry will ever be popped.
                # Roll back so the next UPDATED for this id can re-enter
                # the queue — otherwise we'd reproduce the #4794 stranded-id
                # bug, just triggered by cancellation instead of QueueFull.
                async with self.lock:
                    self.latest_by_key.pop(event.id, None)
                raise
            return

        await self._put_with_backpressure(event)

    async def _put_with_backpressure(self, event: Event):
        if self.queue.full():
            logger.warning(
                "Subscriber queue full, applying backpressure: "
                "source=%s topic=%s event_type=%s id=%s queue_size=%s",
                self.source,
                self.topic,
                event.type.name,
                event.id,
                self.queue.qsize(),
            )
            self._bump(EventCountKind.BACKPRESSURED, event.type)
        await self.queue.put(event)
        self._bump(EventCountKind.ENQUEUED, event.type)

    async def receive(self) -> Any:
        event = await self.queue.get()
        if event.type == EventType.UPDATED and event.id is not None:
            async with self.lock:
                return self.latest_by_key.pop(event.id, event)

        return event


class EventBus:
    def __init__(self):
        """
        Initialize EventBus.

        Uses coordinator for distributed pub/sub when available,
        otherwise operates in local-only mode.
        """
        self.subscribers: Dict[str, List[Subscriber]] = {}
        self._coordinator = None
        self._listen_task: Optional[asyncio.Task] = None
        self._subscribed_channels: set = set()
        # Holds strong references to fire-and-forget tasks so the GC
        # doesn't reap them mid-execution (Python's create_task only
        # holds a weak reference to the task it returns).
        self._pending_tasks: Set[asyncio.Task] = set()

    def _spawn(self, coro) -> asyncio.Task:
        """``asyncio.create_task`` plus retain-and-discard bookkeeping."""
        task = asyncio.create_task(coro)
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)
        return task

    def set_coordinator(self, coordinator):
        """Set the coordinator for distributed pub/sub."""
        self._coordinator = coordinator

    async def start(self):
        """Start the EventBus listener."""
        if self._coordinator:
            # Register ourselves as a subscriber to coordinator
            for topic in self.subscribers:
                await self._subscribe_to_coordinator(topic)
            logger.info("EventBus started with coordinator")

    async def stop(self):
        """Stop the EventBus."""
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
        logger.info("EventBus stopped")

    def subscribe(
        self,
        topic: str,
        source: Optional[str] = None,
        event_types: Optional[Iterable[EventType]] = None,
    ) -> Subscriber:
        """Subscribe to a topic.

        ``source`` is a free-form label used in queue-full log lines so
        operators can identify which consumer is backpressuring. ``event_types``
        is an optional whitelist applied before enqueue — events not matching
        are dropped without occupying a queue slot.
        """
        subscriber = Subscriber(topic=topic, source=source, event_types=event_types)
        if topic not in self.subscribers:
            self.subscribers[topic] = []
            # Subscribe to coordinator if available
            if self._coordinator:
                self._spawn(self._subscribe_to_coordinator(topic))

        self.subscribers[topic].append(subscriber)
        return subscriber

    async def _subscribe_to_coordinator(self, topic: str):
        """Subscribe to coordinator for a topic."""
        if topic in self._subscribed_channels:
            return

        try:
            # Create a closure that captures the topic
            def on_event(event: Event):
                self._on_coordinator_event(event, topic)

            # Register callback with coordinator
            self._coordinator.subscribe(topic, on_event)
            self._subscribed_channels.add(topic)
            logger.debug(f"Subscribed to coordinator topic: {topic}")
        except Exception as e:
            logger.error(f"Failed to subscribe to coordinator topic {topic}: {e}")

    def _on_coordinator_event(self, event: Event, topic: str):
        """Handle event received from coordinator.

        Coordinator implementations must invoke this callback from the main
        event loop (see Coordinator.subscribe); a coordinator whose driver
        fires events from a background thread is responsible for bridging
        to the main loop itself (e.g. via loop.call_soon_threadsafe).
        """
        try:
            self._spawn(self._process_coordinator_event(event, topic))
        except RuntimeError:
            logger.warning(
                f"No running event loop for coordinator event on topic {topic}, skipping"
            )

    async def _process_coordinator_event(self, event: Event, topic: str):
        """
        Process event from coordinator.

        For cross-instance events (only ID received), this method:
        1. Fetches full data from database
        2. Detects changes using local cache
        3. Reconstructs the event with complete data and changed_fields
        """
        # Delay import to avoid circular imports
        from gpustack.server.db import async_session

        # Check if this is a cross-instance event (only has ID)
        is_id_only = (
            event.data is not None
            and isinstance(event.data, dict)
            and set(event.data.keys()) == {"id"}
        )

        if not is_id_only:
            # Local event or cache event, route directly
            logger.trace(
                f"Routing non-ID-only event for topic {topic}: data type={type(event.data).__name__}, keys={list(event.data.keys()) if isinstance(event.data, dict) else 'N/A'}, id={event.id}"
            )
            self._route_event(event, topic)
            return

        # Skip events with no ID - we can't fetch from database
        if event.id is None:
            logger.warning(
                f"Skipping event for topic {topic}: no ID present, cannot fetch data."
            )
            return

        try:
            model_class = get_model_for_topic(topic)
            if model_class is None:
                # Unknown topic, skip to avoid sending incomplete data
                logger.debug(f"Skipping event for topic {topic}: no model class found.")
                return

            # Use ChangeDetector to detect changes and manage cache
            detector = get_change_detector(topic)
            old_obj = detector.get(event.id)

            async with async_session() as session:
                # Fetch full object from database
                obj = await model_class.one_by_id(session, event.id)

                if event.type == EventType.DELETED:
                    # For DELETED events, object is already gone from DB
                    # Use cached old_obj as the data for the event
                    if old_obj is not None:
                        # Use cached object to provide full data for DELETED event
                        enriched_event = Event(
                            type=event.type,
                            data=old_obj,
                            changed_fields={},
                            id=event.id,
                        )
                        logger.trace(
                            f"Enriched DELETED event for topic {topic}: id={event.id}, "
                            f"using cached {type(old_obj).__name__}"
                        )
                        self._route_event(enriched_event, topic)
                    else:
                        # No cached object, route ID-only event for DELETED
                        # so clients know the object was deleted
                        logger.trace(
                            f"Routing ID-only DELETED event for topic {topic}: id={event.id}, "
                            f"no cached object available"
                        )
                        self._route_event(event, topic)
                    # Always remove from cache on DELETE
                    detector.remove(event.id)
                    return

                if obj is None:
                    # Object not in DB (race condition or already deleted), skip
                    logger.debug(
                        f"Skipping event for topic {topic}: object {event.id} not found in database."
                    )
                    return

                # Detect changes for non-DELETE events
                changed_fields = detector.detect_changes(old_obj, obj)

                # Update cache with new object
                detector.put(event.id, obj)

                # Reconstruct event with full data and detected changes
                enriched_event = Event(
                    type=event.type,
                    data=obj,
                    changed_fields=changed_fields,
                    id=event.id,
                )
                logger.trace(
                    f"Enriched event for topic {topic}: id={event.id}, "
                    f"model={type(obj).__name__}, changed_fields={list(changed_fields.keys())}"
                )

                self._route_event(enriched_event, topic)

        except Exception as e:
            logger.error(
                f"Failed to enrich coordinator event for {topic}: {e}. "
                f"Skipping event to avoid sending incomplete data."
            )
            # Skip the event rather than sending incomplete data
            return

    def _route_event(self, event: Event, topic: str):
        """Route event to subscribers of the specific topic.

        Per-subscriber enqueue runs in its own task so a slow consumer
        cannot head-of-line block its peers under blocking backpressure.
        Trade-off: this fan-out is unbounded — for very hot topics with no
        coalescing protection (CREATED/DELETED), bursts can spawn many
        pending enqueue tasks on slow consumers. UPDATED is naturally
        bounded by ``latest_by_key`` coalescing.
        """
        if topic in self.subscribers:
            for subscriber in self.subscribers[topic]:
                self._spawn(subscriber.enqueue(event))

    def unsubscribe(self, topic: str, subscriber: Subscriber):
        """Unsubscribe from a topic."""
        if topic in self.subscribers:
            self.subscribers[topic].remove(subscriber)
            if not self.subscribers[topic]:
                del self.subscribers[topic]

    async def publish(self, topic: str, event: Event):
        """Publish an event to a topic.

        With a coordinator, distribution flows through it so every instance
        sees the event on the same path. On failure or in standalone mode,
        fall back to ``_route_event`` for local fan-out — each subscriber's
        enqueue runs in its own task, so backpressure on one consumer does
        not head-of-line block its peers.
        """
        if self._coordinator:
            try:
                await self._coordinator.publish(topic, event)
                return
            except Exception as e:
                logger.error(
                    f"Failed to publish event to coordinator, "
                    f"falling back to local delivery: {e}"
                )

        self._route_event(event, topic)


event_bus = EventBus()


def set_coordinator(coordinator):
    """Set the coordinator for the global event bus."""
    event_bus.set_coordinator(coordinator)
