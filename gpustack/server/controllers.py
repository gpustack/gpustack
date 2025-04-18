import logging
import random
import string
from typing import Any, Dict, List
import httpx
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from gpustack.config.config import Config
from gpustack.policies.scorers.offload_layer_scorer import OffloadLayerScorer
from gpustack.policies.scorers.placement_scorer import PlacementScorer, ScaleTypeEnum
from gpustack.policies.base import ModelInstanceScore
from gpustack.policies.scorers.status_scorer import StatusScorer
from gpustack.schemas.model_files import ModelFile, ModelFileStateEnum
from gpustack.schemas.models import (
    BackendEnum,
    Model,
    ModelInstance,
    ModelInstanceCreate,
    ModelInstanceStateEnum,
    RayActor,
    SourceEnum,
    get_backend,
)
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.server.bus import Event, EventType
from gpustack.server.db import get_engine
from gpustack.server.services import (
    ModelFileService,
    ModelInstanceService,
    ModelService,
)


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

                if model_instance.state == ModelInstanceStateEnum.INITIALIZING:
                    await ensure_instance_model_file(session, model_instance)

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
    if (
        not model.worker_selector
        and not model.gpu_selector
        and get_backend(model) == BackendEnum.VLLM
    ):
        # vLLM models are only supported on Linux
        model.worker_selector = {"os": "linux"}
        await ModelService(session).update(model)


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

            await ModelInstanceService(session).create(instance)
            logger.debug(f"Created model instance for model {model.name}")

    elif len(instances) > model.replicas:
        candidates = await find_scale_down_candidates(instances, model)

        scale_down_count = len(candidates) - model.replicas
        if scale_down_count > 0:
            for candidate in candidates[:scale_down_count]:
                instance = candidate.model_instance
                await ModelInstanceService(session).delete(instance)
                logger.debug(f"Deleted model instance {instance.name}")


async def ensure_instance_model_file(session: AsyncSession, instance: ModelInstance):
    """
    Synchronize the model file of the model instance.
    """
    if instance.worker_id is None:
        # Not scheduled yet
        return

    if len(instance.model_files) > 0:
        instance = await ModelInstance.one_by_id(session, instance.id)
        await sync_instance_files_state(session, instance, instance.model_files)
        return

    model_files = await get_or_create_model_files_for_instance(session, instance)
    for model_file in model_files:
        if model_file.state == ModelFileStateEnum.ERROR:
            # Retry the download
            model_file.state = ModelFileStateEnum.DOWNLOADING
            model_file.download_progress = 0
            model_file.state_message = ""
            await model_file.update(session)

        logger.info(
            f"Retrying download for model file {model_file.readable_source} for model instance {instance.name}"
        )

    instance = await ModelInstance.one_by_id(session, instance.id)
    instance.model_files = model_files
    await sync_instance_files_state(session, instance, model_files)
    logger.debug(
        f"Associated model file {model_file.readable_source}(id: {model_file.id}) with model instance {instance.name}"
    )


async def get_or_create_model_files_for_instance(
    session: AsyncSession, instance: ModelInstance
) -> List[ModelFile]:
    model_files = await get_model_files_for_instance(session, instance)

    worker_ids = [instance.worker_id]
    if instance.distributed_servers and instance.distributed_servers.ray_actors:
        worker_ids += [
            ray_actor.worker_id for ray_actor in instance.distributed_servers.ray_actors
        ]

    if len(model_files) == len(worker_ids):
        return model_files

    existing_worker_ids = []
    for model_file in model_files:
        existing_worker_ids = [model_file.worker_id for model_file in model_files or []]

    for worker_id in worker_ids:
        if worker_id not in existing_worker_ids:
            model_file = ModelFile(
                source=instance.source,
                huggingface_repo_id=instance.huggingface_repo_id,
                huggingface_filename=instance.huggingface_filename,
                ollama_library_model_name=instance.ollama_library_model_name,
                model_scope_model_id=instance.model_scope_model_id,
                model_scope_file_path=instance.model_scope_file_path,
                local_path=instance.local_path,
                state=ModelFileStateEnum.DOWNLOADING,
                worker_id=worker_id,
                source_index=instance.model_source_index,
            )
            model_file = await ModelFile.create(session, model_file)
            logger.info(
                f"Created model file for model instance {instance.name} and worker {worker_id}"
            )

    return await get_model_files_for_instance(session, instance)


async def get_model_files_for_instance(
    session: AsyncSession, instance: ModelInstance
) -> List[ModelFile]:
    worker_ids = [instance.worker_id]
    if instance.distributed_servers and instance.distributed_servers.ray_actors:
        worker_ids += [
            ray_actor.worker_id for ray_actor in instance.distributed_servers.ray_actors
        ]

    model_files = await ModelFileService(session).get_by_source_index(
        instance.model_source_index
    )
    model_files = [
        model_file for model_file in model_files if model_file.worker_id in worker_ids
    ]

    if instance.source == SourceEnum.LOCAL_PATH and instance.local_path:
        # If the source is local path, get the model files with the same local path.
        local_path_model_files = await ModelFileService(session).get_by_resolved_path(
            instance.local_path
        )
        local_path_model_files = [
            model_file
            for model_file in local_path_model_files
            if model_file.worker_id in worker_ids
        ]
        existing_worker_ids = {mf.worker_id for mf in model_files}
        additional_files = [
            model_file
            for model_file in local_path_model_files
            if model_file.worker_id not in existing_worker_ids
        ]
        model_files.extend(additional_files)

    return model_files


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
        await ModelService(session).update(model)


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

            if (
                worker.state == WorkerStateEnum.NOT_READY
                or event.type == EventType.DELETED
            ):
                instance_names = [instance.name for instance in instances]
                for instance in instances:
                    await ModelInstanceService(session).delete(instance)

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
                await ModelInstanceService(session).update(instance)
        if instance_names:
            logger.debug(
                f"Marked instance {', '.join(instance_names)} {new_state} "
                f"since {log_update_reason}"
            )


class ModelFileController:
    """
    Model file controller syncs the model file download status to related model instances.
    """

    def __init__(self):
        self._engine = get_engine()

    async def start(self):
        """
        Start the controller.
        """

        async for event in ModelFile.subscribe(self._engine):
            if event.type == EventType.CREATED or event.type == EventType.UPDATED:
                await self._reconcile(event)

    async def _reconcile(self, event: Event):
        """
        Reconcile the model file.
        """

        file: ModelFile = event.data
        try:
            async with AsyncSession(self._engine) as session:
                file = await ModelFile.one_by_id(session, file.id)

            if not file:
                # In case the file is deleted
                return

            for instance in file.instances:
                async with AsyncSession(self._engine) as session:
                    await sync_instance_files_state(session, instance, [file])
        except Exception as e:
            logger.error(f"Failed to reconcile model file {file.id}: {e}")


async def sync_instance_files_state(
    session: AsyncSession, instance: ModelInstance, files: List[ModelFile]
):
    for file in files:
        if file.worker_id == instance.worker_id:
            await sync_main_model_file_state(session, file, instance)
        else:
            await sync_distributed_model_file_state(session, file, instance)


async def sync_main_model_file_state(
    session: AsyncSession, file: ModelFile, instance: ModelInstance
):
    """
    Sync the model file state to the related model instance.
    """

    if instance.state == ModelInstanceStateEnum.ERROR:
        return

    logger.trace(
        f"Syncing model file {file.id} with model instance {instance.id}, file state: {file.state}, "
        f"progress: {file.download_progress}, message: {file.state_message}, instance state: {instance.state}"
    )

    need_update = False
    if (
        file.state == ModelFileStateEnum.DOWNLOADING
        and instance.state == ModelInstanceStateEnum.INITIALIZING
    ):
        # Download started
        instance.state = ModelInstanceStateEnum.DOWNLOADING
        instance.download_progress = 0
        instance.state_message = ""
        need_update = True
    elif (
        file.state == ModelFileStateEnum.DOWNLOADING
        and instance.state == ModelInstanceStateEnum.DOWNLOADING
        and file.download_progress != instance.download_progress
    ):
        # Update the download progress
        instance.download_progress = file.download_progress
        need_update = True

    elif file.state == ModelFileStateEnum.READY and (
        instance.state == ModelInstanceStateEnum.DOWNLOADING
        or instance.state == ModelInstanceStateEnum.INITIALIZING
    ):
        # Download completed
        instance.download_progress = 100
        instance.resolved_path = file.resolved_paths[0]
        if model_instance_download_completed(instance):
            # All files are downloaded
            instance.state = ModelInstanceStateEnum.STARTING
        need_update = True
    elif file.state == ModelFileStateEnum.ERROR:
        # Download failed
        instance.state = ModelInstanceStateEnum.ERROR
        instance.state_message = file.state_message
        need_update = True

    if need_update:
        await ModelInstanceService(session).update(instance)


async def sync_distributed_model_file_state(
    session: AsyncSession, file: ModelFile, instance: ModelInstance
):
    """
    Sync the model file state to the related model instance.
    """

    if instance.state == ModelInstanceStateEnum.ERROR:
        return

    ray_actors: List[RayActor] = []

    logger.trace(
        f"Syncing distributed model file {file.id} with model instance {instance.name}, file state: {file.state}, "
        f"progress: {file.download_progress}, message: {file.state_message}, instance state: {instance.state}"
    )

    need_update = False
    for ray_actor in instance.distributed_servers.ray_actors:
        if ray_actor.worker_id == file.worker_id:
            if (
                file.state == ModelFileStateEnum.DOWNLOADING
                and file.download_progress != ray_actor.download_progress
            ):
                ray_actor.download_progress = file.download_progress
                need_update = True
            elif (
                file.state == ModelFileStateEnum.READY
                and ray_actor.download_progress != 100
            ):
                ray_actor.download_progress = 100
                if model_instance_download_completed(instance):
                    # All files are downloaded
                    instance.state = ModelInstanceStateEnum.STARTING
                need_update = True
            elif file.state == ModelFileStateEnum.ERROR:
                instance.state = ModelInstanceStateEnum.ERROR
                instance.state_message = file.state_message
                need_update = True
        ray_actors.append(ray_actor)

    if need_update:
        instance.distributed_servers.ray_actors = ray_actors
        flag_modified(instance, "distributed_servers")
        await ModelInstanceService(session).update(instance)


def model_instance_download_completed(instance: ModelInstance):
    if instance.download_progress != 100:
        return False

    if (
        instance.distributed_servers is None
        or not instance.distributed_servers.ray_actors
    ):
        return True

    for ray_actor in instance.distributed_servers.ray_actors:
        if ray_actor.download_progress != 100:
            return False

    return True
