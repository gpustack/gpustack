import logging
import random
import string
from typing import Any, Dict, List
import httpx
from sqlmodel.ext.asyncio.session import AsyncSession
from gpustack.config.config import Config
from gpustack.policies.scorers.offload_layer_scorer import OffloadLayerScorer
from gpustack.policies.scorers.placement_scorer import PlacementScorer, ScaleTypeEnum
from gpustack.policies.base import ModelInstanceScore
from gpustack.policies.scorers.status_scorer import StatusScorer
from gpustack.schemas.models import (
    BackendEnum,
    Model,
    ModelInstance,
    ModelInstanceCreate,
    ModelInstanceStateEnum,
    get_backend,
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
            if event.type == EventType.HEARTBEAT:
                continue

            await self._reconcile(event)

    async def _reconcile(self, event: Event):
        """
        Reconcile the model.
        """

        model: Model = event.data
        try:
            async with AsyncSession(self._engine) as session:
                await set_default_worker_selector(session, model)
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
            if event.type == EventType.HEARTBEAT:
                continue

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

                if (
                    event.type == EventType.UPDATED
                    and model_instance.state == ModelInstanceStateEnum.RUNNING
                ):
                    # fetch meta from running instance, if it's different from the model meta update it
                    meta = await get_meta_from_running_instance(model_instance)
                    if meta and meta != model.meta:
                        model.meta = meta

                await sync_ready_replicas(session, model)

        except Exception as e:
            logger.error(
                f"Failed to reconcile model instance {model_instance.name}: {e}"
            )


async def set_default_worker_selector(session: AsyncSession, model: Model):
    if model.deleted_at is not None:
        return

    model = await Model.one_by_id(session, model.id)
    if not model.worker_selector and get_backend(model) == BackendEnum.VLLM:
        # vLLM models are only supported on Linux amd64
        model.worker_selector = {"os": "linux", "arch": "amd64"}
        await model.update(session)


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
                model_scope_model_id=model.model_scope_model_id,
                model_scope_file_path=model.model_scope_file_path,
                local_path=model.local_path,
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


async def find_scale_down_candidates(
    instances: List[ModelInstance], model: Model
) -> List[ModelInstanceScore]:
    try:
        placement_scorer = PlacementScorer(model, scale_type=ScaleTypeEnum.SCALE_DOWN)
        placement_candidates = await placement_scorer.score_instances(instances)

        offload_layer_scorer = OffloadLayerScorer(model)
        offload_candidates = await offload_layer_scorer.score_instances(instances)

        status_scorer = StatusScorer(model)
        status_candidates = await status_scorer.score_instances(instances)

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


async def get_meta_from_running_instance(mi: ModelInstance) -> Dict[str, Any]:
    """
    Get the meta information from the running instance.
    """

    if mi.state != ModelInstanceStateEnum.RUNNING:
        return {}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"http://{mi.worker_ip}:{mi.port}/v1/models")
            response.raise_for_status()

            models = response.json()
            if "data" not in models or not models["data"]:
                return {}

            first_model = models["data"][0]
            meta_info = first_model.get("meta", {})

            # Optional keys from different backends
            optional_keys = [
                "voices",
                "max_model_len",
            ]
            for key in optional_keys:
                if key in first_model:
                    meta_info[key] = first_model[key]

            return meta_info
        except Exception as e:
            logger.error(f"Failed to get meta from running instance {mi.name}: {e}")
            return {}


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

            instance_names = []
            if worker.state == WorkerStateEnum.UNREACHABLE:
                await self.update_instance_states(
                    session,
                    instances,
                    ModelInstanceStateEnum.RUNNING,
                    ModelInstanceStateEnum.UNREACHABLE,
                    "Worker is unreachable from the server",
                    "worker is unreachable from the server",
                )
                return

            if worker.state == WorkerStateEnum.READY:
                await self.update_instance_states(
                    session,
                    instances,
                    ModelInstanceStateEnum.UNREACHABLE,
                    ModelInstanceStateEnum.RUNNING,
                    "",
                    "worker is ready",
                )
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

    async def update_instance_states(
        self,
        session,
        instances,
        old_state,
        new_state,
        new_state_message,
        log_update_reason,
    ):
        instance_names = []
        for instance in instances:
            if instance.state == old_state:
                instance_names.append(instance.name)

                instance.state = new_state
                instance.state_message = new_state_message
                await instance.update(session)
        if instance_names:
            logger.debug(
                f"Marked instance {', '.join(instance_names)} {new_state} "
                f"since {log_update_reason}"
            )
