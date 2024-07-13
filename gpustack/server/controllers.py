import logging
import random
import string
from typing import Sequence
from sqlmodel.ext.asyncio.session import AsyncSession
from gpustack.schemas.models import (
    Model,
    ModelInstance,
    ModelInstanceCreate,
    ModelInstanceStateEnum,
)
from gpustack.server.bus import Event
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

        async for event in Model.subscribe(self._engine):
            await self._reconcile(event)

    async def _reconcile(self, event: Event):
        """
        Reconcile the model.
        """

        model: Model = event.data
        try:
            async with AsyncSession(self._engine) as session:
                instances = await ModelInstance.all_by_field(
                    session, "model_id", model.id
                )

                await sync_replicas(session, model, instances)
        except Exception as e:
            logger.error(f"Failed to reconcile model {model.name}: {e}")


class ModelInstanceController:
    def __init__(self):
        self._engine = get_engine()
        pass

    async def start(self):
        """
        Start the controller.
        """

        async for event in ModelInstance.subscribe(self._engine):
            await self._reconcile(event)

    async def _reconcile(self, event: Event):
        """
        Reconcile the model.
        """

        model_instance: ModelInstance = event.data
        try:
            async with AsyncSession(self._engine) as session:
                model = await Model.one_by_id(session, model_instance.model_id)
                if not model:
                    return

                instances = await ModelInstance.all_by_field(
                    session, "model_id", model.id
                )

                await sync_ready_replicas(session, model, instances)

        except Exception as e:
            logger.error(
                f"Failed to reconcile model instance {model_instance.name}: {e}"
            )


async def sync_replicas(
    session: AsyncSession, model: Model, instances: Sequence[ModelInstance]
):
    """
    Synchronize the replicas.
    """

    if model.deleted_at is not None:
        return

    if len(instances) < model.replicas:
        for _ in range(model.replicas - len(instances)):
            name_prefix = ''.join(
                random.choices(string.ascii_letters + string.digits, k=5)
            )
            instance = ModelInstanceCreate(
                name=f"{model.name}-{name_prefix}",
                model_id=model.id,
                model_name=model.name,
                source=model.source,
                huggingface_repo_id=model.huggingface_repo_id,
                huggingface_filename=model.huggingface_filename,
                ollama_library_model_name=model.ollama_library_model_name,
                state=ModelInstanceStateEnum.pending,
            )
            await ModelInstance.create(session, instance)
            logger.debug(f"Created model instance for model {model.name}")

    elif len(instances) > model.replicas:
        for instance in instances[model.replicas :]:
            await instance.delete(session)
            logger.debug(f"Deleted model instance {instance.name}")


async def sync_ready_replicas(
    session: AsyncSession, model: Model, instances: Sequence[ModelInstance]
):
    """
    Synchronize the ready replicas.
    """

    if model.deleted_at is not None:
        return

    ready_replicas: int = 0
    for _, instance in enumerate(instances):
        if instance.state == ModelInstanceStateEnum.running:
            ready_replicas += 1

    if model.ready_replicas != ready_replicas:
        model.ready_replicas = ready_replicas
        await model.update(session)
