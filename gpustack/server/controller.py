import logging
from sqlmodel import Session
from gpustack.schemas.model_instances import ModelInstance, ModelInstanceCreate
from gpustack.schemas.models import Model
from gpustack.server.bus import EventType
from gpustack.server.db import get_engine


logger = logging.getLogger(__name__)


class ModelController:
    def __init__(self):
        self._engine = get_engine()
        pass

    async def start(self):
        """
        Start the controller.
        """

        async for event in Model.subscribe():
            if event.type == EventType.DELETED:
                continue
            await self._reconcile(event.data)

    async def _reconcile(self, model: Model):
        """
        Reconcile the model.
        """
        try:
            with Session(self._engine) as session:
                instances = ModelInstance.all_by_field(session, "model_id", model.id)

            # TODO replicas
            if len(instances) == 0:
                instance = ModelInstanceCreate(
                    model_id=model.id,
                    state="PENDING",
                )
                await ModelInstance.create(session, instance)
        except Exception as e:
            logger.error(f"Failed to reconcile model {model.id}: {e}")
