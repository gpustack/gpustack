import asyncio
import logging

logger = logging.getLogger(__name__)


class AsyncUniqueQueue:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.set = set()
        self.lock = asyncio.Lock()

    async def put(self, item):
        async with self.lock:
            if item not in self.set:
                self.set.add(item)
                await self.queue.put(item)
            else:
                logger.debug(f"Item {item.__hash__()} already in queue.")

    async def get(self):
        item = await self.queue.get()
        async with self.lock:
            self.set.remove(item)
        return item

    def qsize(self):
        return self.queue.qsize()

    async def empty(self):
        return self.queue.empty()

    def task_done(self):
        self.queue.task_done()
