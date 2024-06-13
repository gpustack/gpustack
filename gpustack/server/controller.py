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

                elif len(instances) < model.replicas:
                    instance = ModelInstanceCreate(
                        model_id=model.id,
                        model_name=model.name,
                        source=model.source,
                        huggingface_repo_id=model.huggingface_repo_id,
                        huggingface_filename=model.huggingface_filename,
                        ollama_library_model_name=model.ollama_library_model_name,
                        s3_address=model.s3_address,
                        state="Pending",
                    )
                    for _ in range(model.replicas - len(instances)):
                        await ModelInstance.create(session, instance)
                        logger.debug(f"Created model instance for model {model.id}")

                elif len(instances) > model.replicas:
                    for instance in instances[model.replicas :]:
                        await instance.delete(session)
                        logger.debug(f"Deleted model instance {instance.id}")

        except Exception as e:
            logger.error(f"Failed to reconcile model {model.id}: {e}")
