import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
from enum import Enum
import copy


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

    def __post_init__(self):
        if isinstance(self.type, int):
            self.type = EventType(self.type)


def event_decoder(obj):
    if "type" in obj:
        obj["type"] = EventType[obj["type"]]
    return obj


class Subscriber:
    def __init__(self):
        self.queue = asyncio.Queue(maxsize=256)

    async def enqueue(self, event: Event):
        await self.queue.put(event)

    async def receive(self) -> Any:
        return await self.queue.get()


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
                await subscriber.enqueue(copy.deepcopy(event))


event_bus = EventBus()
