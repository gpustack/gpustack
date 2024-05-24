import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List
from enum import Enum

from gpustack.logging import logger


class EventType(Enum):
    CREATED = 1
    UPDATED = 2
    DELETED = 3
    UNKNOWN = 4


@dataclass
class Event:
    type: EventType
    data: Any


class Subscriber:
    def __init__(self):
        self.queue = asyncio.Queue()

    async def enqueue(self, event: Event):
        await self.queue.put(event)

    async def receive(self) -> Any:
        return await self.queue.get()


class EventBus:
    def __init__(self):
        self.subscribers: Dict[str, List[Subscriber]] = {}

    def subscribe(self, topic: str) -> Subscriber:
        logger.debug(f"Subscribing to topic: {topic}")

        subscriber = Subscriber()
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(subscriber)
        return subscriber

    def unsubscribe(self, topic: str, subscriber: Subscriber):
        logger.debug(f"Unsubscribing from topic: {topic}")

        if topic in self.subscribers:
            self.subscribers[topic].remove(subscriber)
            if not self.subscribers[topic]:
                del self.subscribers[topic]

    async def publish(self, topic: str, event: Event):
        if topic in self.subscribers:
            for subscriber in self.subscribers[topic]:
                await subscriber.enqueue(event)


event_bus = EventBus()
