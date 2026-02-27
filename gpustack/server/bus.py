import asyncio
from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class EventType(Enum):
    CREATED = 1
    UPDATED = 2
    DELETED = 3
    UNKNOWN = 4
    HEARTBEAT = 5

    def __str__(self):
        return self.name


@dataclass
class Event:
    type: EventType
    data: Any
    changed_fields: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)
    id: Optional[Any] = None

    def __post_init__(self):
        if isinstance(self.type, int):
            self.type = EventType(self.type)

        if self.id is None:
            self.id = self._derive_id_from_data()

    def _derive_id_from_data(self) -> Optional[Any]:
        if self.data is None:
            return None

        if self.type in [EventType.DELETED, EventType.CREATED]:
            return None

        # SQLModel
        if hasattr(self.data, "id"):
            return getattr(self.data, "id")

        # Plain dict
        if isinstance(self.data, dict):
            return self.data.get("id")

        return None


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
        self.subscribers: Dict[str, List[Subscriber]] = {}

    def subscribe(self, topic: str) -> Subscriber:
        subscriber = Subscriber()
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(subscriber)
        return subscriber

    def unsubscribe(self, topic: str, subscriber: Subscriber):
        if topic in self.subscribers:
            self.subscribers[topic].remove(subscriber)
            if not self.subscribers[topic]:
                del self.subscribers[topic]

    async def publish(self, topic: str, event: Event):
        if topic in self.subscribers:
            for subscriber in self.subscribers[topic]:
                await subscriber.enqueue(event)


event_bus = EventBus()
