import logging
import random
import string
from typing import Any, Dict, List, Tuple, Optional
import httpx
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import select
from sqlalchemy.orm import selectinload
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
    SourceEnum,
    get_backend,
)
from gpustack.schemas.workers import (
    Worker,
    WorkerStateEnum,
    WorkerStatus,
)
from gpustack.schemas.clusters import (
    Cluster,
    WorkerPool,
    CloudCredential,
    Credential,
    CredentialType,
    ClusterStateEnum,
    SSHKeyOptions,
)
from gpustack.server.bus import Event, EventType, event_bus
from gpustack.server.db import get_engine
from gpustack.server.services import (
    ModelFileService,
    ModelInstanceService,
    ModelService,
    WorkerService,
)
from gpustack.cloud_providers.common import (
    get_client_from_provider,
    construct_cloud_instance,
    construct_user_data,
    generate_ssh_key_pair,
)
from gpustack.cloud_providers.abstract import (
    ProviderClientBase,
    CloudInstance,
    InstanceState,
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
                    worker = await Worker.one_by_id(session, model_instance.worker_id)
                    if not worker:
                        logger.error(
                            f"Failed to find worker {model_instance.worker_id} for model instance {model_instance.name}"
                        )
                        return

                    # fetch meta from running instance, if it's different from the model meta update it
                    meta = await self.get_meta_from_running_instance(
                        model_instance, worker
                    )
                    if meta and meta != model.meta:
                        model.meta = meta

                await sync_ready_replicas(session, model)

        except Exception as e:
            logger.error(
                f"Failed to reconcile model instance {model_instance.name}: {e}"
            )

    async def get_meta_from_running_instance(
        self, mi: ModelInstance, w: Worker
    ) -> Dict[str, Any]:
        """
        Get the meta information from the running instance.
        """
        if mi.state != ModelInstanceStateEnum.RUNNING:
            return {}

        async with httpx.AsyncClient() as client:
            try:
                url = f"http://{w.ip}:{w.port}/proxy/v1/models"
                headers = {
                    "X-Target-Port": str(mi.port),
                    "Authorization": f"Bearer {self._config.token}",
                }
                response = await client.get(
                    url=url,
                    headers=headers,
                )
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
                cluster_id=model.cluster_id,
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
    """
    Get or create model files for the given model instance.
    """

    model_files = await get_model_files_for_instance(session, instance)
    worker_ids = _get_worker_ids_for_file_download(instance)

    # Return early if all model files are already created for the workers
    if len(model_files) == len(worker_ids):
        return model_files

    # Get the worker IDs that are missing model files.
    missing_worker_ids = set(worker_ids) - {
        model_file.worker_id for model_file in model_files
    }
    if not missing_worker_ids:
        return model_files

    # Create model files for the missing worker IDs.
    for worker_id in missing_worker_ids:
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
        await ModelFile.create(session, model_file)
        logger.info(
            f"Created model file for model instance {instance.name} and worker {worker_id}"
        )

    # After creating the model files, fetch them again to return the complete list.
    return await get_model_files_for_instance(session, instance)


async def get_model_files_for_instance(
    session: AsyncSession, instance: ModelInstance
) -> List[ModelFile]:
    worker_ids = _get_worker_ids_for_file_download(instance)

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


class WorkerController:
    def __init__(self, cfg: Config):
        self._engine = get_engine()
        self._provisioning = WorkerProvisioningController(cfg)
        pass

    async def start(self):
        """
        Start the controller.
        """

        async for event in Worker.subscribe(self._engine):
            try:
                await self._reconcile(event)
                await self._provisioning._reconcile(event)
                await self._notify_parents(event)
            except Exception as e:
                logger.error(f"Failed to reconcile worker: {e}")

    async def _reconcile(self, event):
        """
        Delete instances base on the worker state and event type.
        """
        if event.type not in (EventType.UPDATED, EventType.DELETED):
            return
        worker: Worker = event.data
        # Skip reconciliation for provisioning and deleting workers.
        # There is a dedicated controller to handle provisioning.
        if not worker or worker.state.is_provisioning:
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

    async def _notify_parents(self, event: Event):
        if event.type not in (EventType.UPDATED, EventType.DELETED):
            return
        worker: Worker = event.data
        changed_fields = event.changed_fields
        if not worker or not changed_fields:
            return
        state_changed: Optional[Tuple[Any, Any]] = changed_fields.get("state", None)
        if state_changed is None:
            return
        async with AsyncSession(self._engine) as session:
            worker = await Worker.one_by_id(session, worker.id)
            if not worker:
                return
            if worker.worker_pool is not None:
                result = await session.exec(
                    select(WorkerPool)
                    .options(selectinload(WorkerPool.pool_workers))
                    .where(WorkerPool.id == worker.worker_pool_id)
                )
                worker_pool = result.one()
                copied_pool = WorkerPool(**worker_pool.model_dump())
                await event_bus.publish(
                    copied_pool.__class__.__name__.lower(),
                    Event(
                        type=EventType.UPDATED,
                        data=copied_pool,
                    ),
                )
            if worker.cluster is not None:
                result = await session.exec(
                    select(Cluster)
                    .options(
                        selectinload(Cluster.cluster_workers),
                        selectinload(Cluster.cluster_models),
                    )
                    .where(Cluster.id == worker.cluster_id)
                )
                cluster = result.one()
                copied_cluster = Cluster(**cluster.model_dump())
                await event_bus.publish(
                    copied_cluster.__class__.__name__.lower(),
                    Event(
                        type=EventType.UPDATED,
                        data=copied_cluster,
                    ),
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
            instance.state_message = ""
        need_update = True
    elif file.state == ModelFileStateEnum.ERROR:
        # Download failed
        instance.state = ModelInstanceStateEnum.ERROR
        instance.state_message = file.state_message
        need_update = True

    if need_update:
        await ModelInstanceService(session).update(instance)


async def sync_distributed_model_file_state(  # noqa: C901
    session: AsyncSession, file: ModelFile, instance: ModelInstance
):
    """
    Sync the model file state to the related model instance.
    """

    if instance.state == ModelInstanceStateEnum.ERROR:
        return

    if (
        not instance.distributed_servers
        or not instance.distributed_servers.download_model_files
    ):
        return

    logger.trace(
        f"Syncing distributed model file {file.id} with model instance {instance.name}, file state: {file.state}, "
        f"progress: {file.download_progress}, message: {file.state_message}, instance state: {instance.state}"
    )

    need_update = False

    for item in instance.distributed_servers.subordinate_workers or []:
        if item.worker_id == file.worker_id:
            if (
                file.state == ModelFileStateEnum.DOWNLOADING
                and file.download_progress != item.download_progress
            ):
                item.download_progress = file.download_progress
                need_update = True
            elif (
                file.state == ModelFileStateEnum.READY and item.download_progress != 100
            ):
                item.download_progress = 100
                if model_instance_download_completed(instance):
                    # All files are downloaded
                    instance.state = ModelInstanceStateEnum.STARTING
                    instance.state_message = ""
                need_update = True
            elif file.state == ModelFileStateEnum.ERROR:
                instance.state = ModelInstanceStateEnum.ERROR
                instance.state_message = file.state_message
                need_update = True

    if need_update:
        flag_modified(instance, "distributed_servers")
        await ModelInstanceService(session).update(instance)


def model_instance_download_completed(instance: ModelInstance):
    if instance.download_progress != 100:
        return False

    if (
        instance.distributed_servers
        and instance.distributed_servers.download_model_files
    ):
        for subworker in instance.distributed_servers.subordinate_workers or []:
            if subworker.download_progress != 100:
                return False

    return True


def _get_worker_ids_for_file_download(
    instance: ModelInstance,
) -> List[str]:
    """
    Get the all worker IDs of the model instance that are
    responsible for downloading the model files,
    including the main worker and distributed workers.
    """

    worker_ids = [instance.worker_id] if instance.worker_id else []

    if (
        instance.distributed_servers
        and instance.distributed_servers.download_model_files
    ):
        worker_ids += [
            item.worker_id
            for item in instance.distributed_servers.subordinate_workers or []
            if item.worker_id
        ]

    return worker_ids


async def new_workers_from_pool(
    session: AsyncSession, pool: WorkerPool
) -> List[Worker]:
    fields = {"deleted_at": None, "worker_pool_id": pool.id}
    current_workers = await Worker.all_by_fields(session, fields=fields)
    current_workers = [
        worker
        for worker in current_workers
        if worker.state not in [WorkerStateEnum.DELETING]
    ]
    # if has enough workers, no need to create more
    if len(current_workers) >= pool.replicas:
        return []
    delta = pool.replicas - len(current_workers)
    if pool.batch_size is not None and delta > pool.batch_size:
        delta = pool.batch_size
    provisioning_workers = [
        worker
        for worker in current_workers
        if worker.state in [WorkerStateEnum.PROVISIONING]
    ]
    # if has enough provisioning workers, no need to create more
    if pool.batch_size <= len(provisioning_workers):
        return []
    new_workers = []
    for _ in range(delta):
        new_worker = Worker(
            hostname="",
            ip="",
            port=0,
            worker_uuid="",
            cluster=pool.cluster,
            worker_pool=pool,
            provider=pool.cluster.provider,
            name=f"pool-{pool.id}-"
            + ''.join(random.choices(string.ascii_lowercase + string.digits, k=8)),
            labels={
                "provider": pool.cluster.provider.value,
                "instance_type": pool.instance_type or "unknown",
            },
            state=WorkerStateEnum.PENDING,
            status=WorkerStatus.get_default_status(),
        )
        new_workers.append(new_worker)
    return new_workers


class WorkerPoolController:
    def __init__(self):
        self._engine = get_engine()
        pass

    async def start(self):
        async for event in WorkerPool.subscribe(self._engine):
            if event.type == EventType.HEARTBEAT:
                continue
            try:
                await self._reconcile(event)
            except Exception as e:
                logger.error(f"Failed to reconcile worker pool: {e}")

    async def _reconcile(self, event: Event):
        """
        Reconcile the worker pool state with the current event.
        """
        logger.info(f"Reconcile worker pool {event.data.id} with event {event.type}")
        async with AsyncSession(self._engine) as session:
            pool = await WorkerPool.one_by_id(session, event.data.id)
            if pool is None or pool.deleted_at is not None:
                return
            # mark the data to avoid read after commit
            cluster_name = pool.cluster.name
            pool_id = pool.id
            workers = await new_workers_from_pool(session, pool)
            if len(workers) == 0:
                return
            ids = []
            for worker in workers:
                created_worker: Worker = await Worker.create(
                    session=session, source=worker, auto_commit=False
                )
                ids.append(created_worker.id)
            await session.commit()
            logger.info(
                f"Created {len(ids)} new workers {ids} for cluster {cluster_name} worker pool {pool_id}"
            )


class WorkerProvisioningController:
    def __init__(self, cfg: Config):
        self._engine = get_engine()
        self._cfg = cfg

    @classmethod
    async def _create_ssh_key(
        cls,
        session: AsyncSession,
        client: ProviderClientBase,
        worker: Worker,
    ) -> int:
        """
        Generate a new ssh key pair,
        And Create ssh_key in cloud provider.
        Create SSHKey record without commit and returns it.
        """
        logger.info(f"Creating ssh key for worker {worker.name}")
        private_key, public_key = generate_ssh_key_pair()
        ssh_key = Credential(
            credential_type=CredentialType.SSH,
            public_key=public_key,
            encoded_private_key=private_key,
            ssh_key_options=SSHKeyOptions(
                algorithm="ED25519",
                length=0,
            ),
        )
        ssh_key_id = await client.create_ssh_key(worker.name, public_key)
        ssh_key.external_id = str(ssh_key_id)
        ssh_key_rtn = await Credential.create(session, ssh_key, auto_commit=False)
        return ssh_key_rtn.id

    @classmethod
    async def _create_instances(
        cls,
        session: AsyncSession,
        client: ProviderClientBase,
        worker: Worker,
        cfg: Config,
    ) -> str:
        distrubution, public = await client.determine_linux_distribution(
            worker.worker_pool.os_image
        )
        user_data = construct_user_data(
            config=cfg,
            worker=worker,
            distribution=distrubution,
            public=public,
        )
        ssh_key = await Credential.one_by_id(session, worker.ssh_key_id)
        if ssh_key is None:
            raise ValueError(f"SSH key {worker.ssh_key_id} not found")
        to_create = construct_cloud_instance(worker, ssh_key, user_data)
        logger.info(f"Creating cloud instance for worker {worker.name}")
        logger.debug(f"Cloud instance configuration: {to_create}")
        return await client.create_instance(to_create)

    @classmethod
    async def _provisioning_started(
        cls,
        session: AsyncSession,
        client: ProviderClientBase,
        worker: Worker,
        instance: CloudInstance,
    ) -> bool:
        changed = True
        provider_config = worker.provider_config or {}
        volumes = list(
            (getattr(worker.worker_pool.cloud_options, "volumes", None) or [])
        )
        volume_ids = provider_config.get("volume_ids", [])
        if worker.ip is None or worker.ip == "":
            try:
                instance = await client.wait_for_public_ip(worker.external_id)
                worker.ip = instance.ip_address if instance.ip_address else ""
                worker.state_message = "Waiting for volumes to attach"
            except Exception as e:
                logger.warning(
                    f"Failed to wait for instance {worker.external_id} to get public ip: {e}"
                )
        elif len(volumes) != len(volume_ids) and len(volumes) > 0:
            volume_ids = await client.create_volumes_and_attach(
                worker.id, worker.external_id, worker.cluster.region, *volumes
            )
            provider_config["volume_ids"] = volume_ids
            worker.provider_config = provider_config
        elif (
            len(volumes) == len(volume_ids)
            and worker.state == WorkerStateEnum.PROVISIONING
        ):
            if not hasattr(provider_config, "volume_ids"):
                provider_config["volume_ids"] = []
            worker.provider_config = provider_config
            worker.state = WorkerStateEnum.INITIALIZING
            if worker.cluster.state != ClusterStateEnum.PROVISIONED:
                worker.cluster.state = ClusterStateEnum.PROVISIONED
                await worker.cluster.update(session=session, auto_commit=False)
            worker.state_message = "Initializing: installing required drivers and software. The worker will start automatically after setup."
        else:
            changed = False
        return changed

    @classmethod
    async def _provisioning_before_started(
        cls,
        session: AsyncSession,
        client: ProviderClientBase,
        worker: Worker,
        cfg: Config,
    ) -> Tuple[Optional[CloudInstance], bool]:
        """
        return started and changed
        """
        instance = None
        changed = False
        if worker.external_id is not None:
            instance = await client.get_instance(worker.external_id)
            # TODO should handle instance not exist problem
            if instance is None or instance.status == InstanceState.RUNNING:
                return instance, changed
        changed = True
        if worker.state == WorkerStateEnum.PENDING:
            worker.state = WorkerStateEnum.PROVISIONING
            worker.state_message = "Creating SSH key"
        elif worker.ssh_key_id is None:
            worker.ssh_key_id = await cls._create_ssh_key(session, client, worker)
            worker.state_message = "Creating cloud instance"
        elif worker.external_id is None:
            worker.external_id = await cls._create_instances(
                session, client, worker, cfg
            )
            worker.state_message = "Waiting for cloud instance started"
        elif worker.external_id is not None:
            try:
                # depress the timeout exception
                instance = await client.wait_for_started(worker.external_id)
                worker.state_message = "Waiting for instance's public ip"
            except Exception as e:
                logger.warning(
                    f"Failed to wait for instance {worker.external_id} to start: {e}"
                )
        return instance, changed

    @classmethod
    async def _provisioning_instance(
        cls,
        session: AsyncSession,
        client: ProviderClientBase,
        worker: Worker,
        cfg: Config,
    ):
        # provider_config = worker.provider_config or {}
        # Phase I is to ensure instance running.
        instance, changed = await cls._provisioning_before_started(
            session, client, worker, cfg
        )
        if (
            not changed
            and instance is not None
            and instance.status == InstanceState.RUNNING
        ):
            # Phase II is to wait for instance infomation and attach volume.
            changed = await cls._provisioning_started(session, client, worker, instance)
        if changed:
            await WorkerService(session).update(
                worker=worker, source=None, auto_commit=False
            )

    @classmethod
    async def _deleting_instance(
        cls,
        session: AsyncSession,
        client: ProviderClientBase,
        worker: Worker,
    ):
        if worker.external_id is None:
            return
        ssh_key = await Credential.one_by_id(session, worker.ssh_key_id)
        try:
            await client.delete_instance(worker.external_id)
            if ssh_key and ssh_key.external_id:
                await client.delete_ssh_key(ssh_key.external_id)
        except Exception as e:
            logger.error(f"Failed to delete instance {worker.external_id}: {e}")
        # if using soft delete here, skip deletion and remove external_id
        if ssh_key:
            await ssh_key.delete(session, auto_commit=False)
        if worker.deleted_at is not None:
            await WorkerService(session).delete(worker, auto_commit=False)

    async def _reconcile(self, event: Event):
        """
        When provisioning a worker, the state will transition from following steps:
        - PENDING - initial state for worker created by pool, the next state is PROVISIONING
        - PROVISIONING - begin provisioning with related info updated in worker object, the next state is PROVISIONED
        - PROVISIONED - done provisioning and waiting for worker to register
        - DELETING - worker is being deleted
        - ERROR - an error occurred during provisioning
        """
        worker: Worker = event.data
        if not worker:
            return
        if worker.state not in [
            WorkerStateEnum.PENDING,
            WorkerStateEnum.PROVISIONING,
            WorkerStateEnum.DELETING,
        ]:
            return
        logger.info(
            f"Reconcile provisioning worker {event.data.name} with event {event.type}"
        )
        async with AsyncSession(self._engine) as session:
            # Fetch the worker from the database
            worker: Worker = await Worker.one_by_id(session, worker.id)
            if not worker:
                return
            credential: CloudCredential = await CloudCredential.one_by_id(
                session, worker.cluster.credential_id
            )
            client = get_client_from_provider(
                worker.cluster.provider,
                credential=credential,
            )
            try:
                if worker.state in [
                    WorkerStateEnum.PENDING,
                    WorkerStateEnum.PROVISIONING,
                ]:
                    await self._provisioning_instance(
                        session, client, worker, self._cfg
                    )
                if worker.state == WorkerStateEnum.DELETING:
                    await self._deleting_instance(session, client, worker)
                await session.commit()
            except Exception as e:
                message = f"Failed to provision or delete worker {worker.name}: {e}"
                logger.error(message)
                await session.rollback()
                await session.refresh(worker)
                worker.state = WorkerStateEnum.ERROR
                worker.state_message = message
                await WorkerService(session).update(
                    worker=worker, source=None, auto_commit=True
                )


class ClusterController:
    def __init__(self):
        self._engine = get_engine()
        pass

    async def start(self):
        """
        Start the controller.
        """

        async for event in Cluster.subscribe(self._engine):
            if event.type == EventType.HEARTBEAT:
                continue
            try:
                await self._reconcile(event)
            except Exception as e:
                logger.error(f"Failed to reconcile cluster: {e}")

    async def _reconcile(self, event: Event):
        """
        Reconcile the cluster state with the current event.
        """
        logger.info(f"Reconcile cluster {event.data.name} with event {event.type}")
