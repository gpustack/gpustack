import asyncio
import logging
from typing import Any, Dict, List, Optional

# Re-export from coordinator.base for backward compatibility
from gpustack.server.coordinator.base import Event, EventType
from gpustack.server.coordinator.cache import get_change_detector
from gpustack.server.coordinator.models import get_model_for_topic

logger = logging.getLogger(__name__)

# Re-export for backward compatibility
__all__ = [
    'Event',
    'EventType',
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
    def __init__(self):
        self.queue = asyncio.Queue(maxsize=1024)
        self.latest_by_key = {}
        self.lock = asyncio.Lock()

    async def enqueue(self, event: Event):
        # Squash UPDATED events by keeping only the latest per key
        if event.type == EventType.UPDATED and event.id is not None:
            async with self.lock:
                if event.id in self.latest_by_key:
                    self.latest_by_key[event.id] = event
                    return
                self.latest_by_key[event.id] = event

            try:
                self.queue.put_nowait(event)
            except asyncio.QueueFull:
                # If the queue is full, skip adding the event, relying on latest_by_key, could receive it later
                logger.warning(
                    "Subscriber:%s queue full, skipping UPDATED event for id=%s",
                    id(self),
                    event.id,
                )
            return

        # For other event types, enqueue directly
        await self.queue.put(event)

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

    def subscribe(self, topic: str) -> Subscriber:
        """Subscribe to a topic."""
        subscriber = Subscriber()
        if topic not in self.subscribers:
            self.subscribers[topic] = []
            # Subscribe to coordinator if available
            if self._coordinator:
                asyncio.create_task(self._subscribe_to_coordinator(topic))

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
            asyncio.create_task(self._process_coordinator_event(event, topic))
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
        """Route event to subscribers of the specific topic."""
        if topic in self.subscribers:
            for subscriber in self.subscribers[topic]:
                asyncio.create_task(subscriber.enqueue(event))

    def unsubscribe(self, topic: str, subscriber: Subscriber):
        """Unsubscribe from a topic."""
        if topic in self.subscribers:
            self.subscribers[topic].remove(subscriber)
            if not self.subscribers[topic]:
                del self.subscribers[topic]

    async def publish(self, topic: str, event: Event):
        """Publish an event to a topic.

        When a coordinator is configured, distribution normally happens via the
        coordinator so all instances (including this one) receive the event on
        the same path. If the coordinator publish fails, fall back to local
        delivery so this instance can still process its own events.
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

        # Local delivery (no coordinator configured, or coordinator publish failed).
        if topic in self.subscribers:
            for subscriber in self.subscribers[topic]:
                await subscriber.enqueue(event)


event_bus = EventBus()


def set_coordinator(coordinator):
    """Set the coordinator for the global event bus."""
    event_bus.set_coordinator(coordinator)
