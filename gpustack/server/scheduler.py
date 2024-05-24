from sqlmodel import Session
from gpustack.schemas.nodes import Node
from gpustack.schemas.tasks import Task
from gpustack.server.bus import EventType
from gpustack.server.db import get_engine
from gpustack.logging import logger


class Scheduler:

    async def start(self):
        """
        Start the scheduler.
        """

        engine = get_engine()
        with Session(engine) as session:
            tasks = Task.all(session)

        for task in tasks:
            await self._do_schedule(task)

        async for event in Task.subscribe():
            if event.type == EventType.DELETED:
                continue
            await self._do_schedule(event.data)

    async def _do_schedule(self, task: Task) -> bool:
        try:
            if self._should_schedule(task):
                await self.schedule_naively(task)
        except Exception as e:
            logger.error(f"Failed to schedule task {task.id}: {e}")

    def _should_schedule(self, task: Task) -> bool:
        """
        Check if the task should be scheduled.
        """

        return task.node_id is None

    async def schedule_naively(self, task: Task):
        """
        Schedule a task by picking any node.
        """

        engine = get_engine()
        with Session(engine) as session:
            node = Node.first(session)

        if not node:
            return

        task = Task.one_by_id(session, task.id)  # load from the new session
        task.node_id = node.id
        await task.update(session, task)

        logger.debug(f"Scheduled task {task.id} to node {node.id}")
