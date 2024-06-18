import asyncio
import logging
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.nodes import Node
from gpustack.schemas.models import ModelInstance
from gpustack.server.bus import EventType
from gpustack.server.db import get_engine

logger = logging.getLogger(__name__)


class Scheduler:
    def __init__(self):
        self._engine = get_engine()
        self._check_interval = 30

    async def start(self):
        """
        Start the scheduler.
        """

        asyncio.create_task(self.check_pending_instances())

        async for event in ModelInstance.subscribe(self._engine):
            if event.type == EventType.DELETED:
                continue
            await self._do_schedule(event.data)

    async def check_pending_instances(self):
        """
        Periodcally check pending instances and schedule them.
        """

        while True:
            await asyncio.sleep(self._check_interval)
            async with AsyncSession(self._engine) as session:
                instances = await ModelInstance.all_by_field(
                    session, "state", "Pending"
                )
                for instance in instances:
                    await self._do_schedule(instance)

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

        async with AsyncSession(self._engine) as session:
            node = await Node.first(session)

        if not node:
            return

        model_instance = await ModelInstance.one_by_id(
            session, mi.id
        )  # load from the new session
        model_instance.node_id = node.id
        model_instance.node_ip = node.address
        await model_instance.update(session, model_instance)

        logger.debug(f"Scheduled model instance {model_instance.id} to node {node.id}")
