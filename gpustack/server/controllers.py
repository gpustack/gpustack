import logging
import random
import string
from typing import List
from sqlmodel.ext.asyncio.session import AsyncSession
from gpustack.config.config import Config
from gpustack.policies.offload_layer_policy import OffloadLayerPolicy
from gpustack.policies.placement_policy import PlacementPolicy, ScaleTypeEnum
from gpustack.policies.policy import ModelInstanceScore
from gpustack.policies.status_policy import StatusPolicy
from gpustack.schemas.models import (
    Model,
    ModelInstance,
    ModelInstanceCreate,
    ModelInstanceStateEnum,
)
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.server.bus import Event, EventType
from gpustack.server.db import get_engine


logger = logging.getLogger(__name__)


class ModelController:
    def __init__(self, cfg: Config):
        self._engine = get_engine()
        self._config = cfg

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
                await sync_replicas(session, model, self._config)
        except Exception as e:
            logger.error(f"Failed to reconcile model {model.name}: {e}")


class ModelInstanceController:
    def __init__(self, cfg: Config):
        self._engine = get_engine()
        self._config = cfg

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

                if event.type == EventType.DELETED:
                    await sync_replicas(session, model, self._config)

                await model.refresh(session)
                await sync_ready_replicas(session, model)

        except Exception as e:
            logger.error(
                f"Failed to reconcile model instance {model_instance.name}: {e}"
            )


async def sync_replicas(session: AsyncSession, model: Model, cfg: Config):
    """
    Synchronize the replicas.
    """

    if model.deleted_at is not None:
        return

    instances = await ModelInstance.all_by_field(session, "model_id", model.id)
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
                state=ModelInstanceStateEnum.PENDING,
            )
            await ModelInstance.create(session, instance)
            logger.debug(f"Created model instance for model {model.name}")

    elif len(instances) > model.replicas:
        candidates = await find_scale_down_candidates(instances, model)

        scale_down_count = len(candidates) - model.replicas
        if scale_down_count > 0:
            for candidate in candidates[:scale_down_count]:
                instance = candidate.model_instance
                await instance.delete(session)
                logger.debug(f"Deleted model instance {instance.name}")


async def policy_score_instances(
    policy, instances: List[ModelInstance]
) -> List[ModelInstanceScore]:
    return await policy.score_instances(instances)


async def find_scale_down_candidates(
    instances: List[ModelInstance], model: Model
) -> List[ModelInstanceScore]:
    try:
        placement_policy = PlacementPolicy(model, scale_type=ScaleTypeEnum.SCALE_DOWN)
        placement_candidates = await policy_score_instances(placement_policy, instances)

        offload_layer_policy = OffloadLayerPolicy(model)
        offload_candidates = await policy_score_instances(
            offload_layer_policy, instances
        )

        status_policy = StatusPolicy(model)
        status_candidates = await policy_score_instances(status_policy, instances)

        offload_cand_map = {cand.model_instance.id: cand for cand in offload_candidates}
        placement_cand_map = {
            cand.model_instance.id: cand for cand in placement_candidates
        }

        for cand in status_candidates:
            score = cand.score * 100
            offload_candidate = offload_cand_map.get(cand.model_instance.id)
            score += offload_candidate.score * 10 if offload_candidate else 0

            placement_candidate = placement_cand_map.get(cand.model_instance.id)
            score += placement_candidate.score if placement_candidate else 0
            cand.score = score / 111

        final_candidates = sorted(
            status_candidates, key=lambda x: x.score, reverse=False
        )
        return final_candidates
    except Exception as e:
        state_message = (
            f"Failed to find scale down candidates for model {model.name}: {e}"
        )
        logger.error(state_message)


async def sync_ready_replicas(session: AsyncSession, model: Model):
    """
    Synchronize the ready replicas.
    """

    if model.deleted_at is not None:
        return

    instances = await ModelInstance.all_by_field(session, "model_id", model.id)

    ready_replicas: int = 0
    for _, instance in enumerate(instances):
        if instance.state == ModelInstanceStateEnum.RUNNING:
            ready_replicas += 1

    if model.ready_replicas != ready_replicas:
        model.ready_replicas = ready_replicas
        await model.update(session)


class WorkerController:
    def __init__(self):
        self._engine = get_engine()
        pass

    async def start(self):
        """
        Start the controller.
        """

        async for event in Worker.subscribe(self._engine):
            if event.type in (EventType.UPDATED, EventType.DELETED):
                try:
                    await self._reconcile(event)
                except Exception as e:
                    logger.error(f"Failed to reconcile worker: {e}")

    async def _reconcile(self, event):
        """
        Delete instances base on the worker state and event type.
        """
        worker: Worker = event.data
        if not worker:
            return

        async with AsyncSession(self._engine) as session:
            instances = await ModelInstance.all_by_field(
                session, "worker_name", worker.name
            )
            if not instances:
                return

            if (
                worker.state == WorkerStateEnum.NOT_READY
                or event.type == EventType.DELETED
            ):
                instance_names = [instance.name for instance in instances]
                for instance in instances:
                    await instance.delete(session)

                if instance_names:
                    state = (
                        worker.state
                        if worker.state == WorkerStateEnum.NOT_READY
                        else "deleted"
                    )
                    logger.debug(
                        f"Delete instance {', '.join(instance_names)} "
                        f"since worker {worker.name} is {state}"
                    )
