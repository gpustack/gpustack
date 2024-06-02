import logging
from sqlmodel import Session
from gpustack.schemas.models import Model, ModelInstance, ModelInstanceCreate
from gpustack.server.bus import Event, EventType
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

        with Session(self._engine) as session:
            async for event in Model.subscribe(session):
                await self._reconcile(event)

    async def _reconcile(self, event: Event):
        """
        Reconcile the model.
        """

        model: Model = event.data
        event_type: EventType = event.type
        try:
            with Session(self._engine) as session:
                instances = ModelInstance.all_by_field(session, "model_id", model.id)

                if event_type == EventType.DELETED:
                    for instance in instances:
                        await instance.delete(session)
                elif len(instances) == 0:  # TODO replicas
                    instance = ModelInstanceCreate(
                        model_id=model.id,
                        source=model.source,
                        huggingface_model_id=model.huggingface_model_id,
                        s3_address=model.s3_address,
                        state="Pending",
                    )
                    await ModelInstance.create(session, instance)

                    logger.debug(f"Created model instance for model {model.id}")
        except Exception as e:
            logger.error(f"Failed to reconcile model {model.id}: {e}")
