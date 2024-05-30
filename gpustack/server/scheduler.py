import logging
from sqlmodel import Session

from gpustack.schemas.nodes import Node
from gpustack.schemas.model_instances import ModelInstance
from gpustack.server.bus import EventType
from gpustack.server.db import get_engine


logger = logging.getLogger(__name__)


class Scheduler:

    async def start(self):
        """
        Start the scheduler.
        """

        engine = get_engine()
        with Session(engine) as session:
            model_instances = ModelInstance.all(session)

        for mi in model_instances:
            await self._do_schedule(mi)

        async for event in ModelInstance.subscribe():
            if event.type == EventType.DELETED:
                continue
            await self._do_schedule(event.data)

    async def _do_schedule(self, mi: ModelInstance) -> bool:
        try:
            if self._should_schedule(mi):
                await self.schedule_naively(mi)
        except Exception as e:
            logger.error(f"Failed to schedule model instance {mi.id}: {e}")

    def _should_schedule(self, mi: ModelInstance) -> bool:
        """
        Check if the model instance should be scheduled.
        """

        return mi.node_id is None

    async def schedule_naively(self, mi: ModelInstance):
        """
        Schedule a model instance by picking any node.
        """

        engine = get_engine()
        with Session(engine) as session:
            node = Node.first(session)

        if not node:
            return

        model_instance = ModelInstance.one_by_id(
            session, mi.id
        )  # load from the new session
        model_instance.node_id = node.id
        await model_instance.update(session, model_instance)

        logger.debug(f"Scheduled model instance {model_instance.id} to node {node.id}")
