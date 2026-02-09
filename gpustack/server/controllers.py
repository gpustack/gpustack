import logging
import random
import string
import asyncio
import yaml
from importlib.resources import files
from typing import Any, Dict, List, Tuple, Optional, Set
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy.orm.attributes import flag_modified

from gpustack.config.config import (
    Config,
    get_cluster_image_name,
)
from gpustack.policies.scorers.offload_layer_scorer import OffloadLayerScorer
from gpustack.policies.scorers.placement_scorer import PlacementScorer, ScaleTypeEnum
from gpustack.policies.base import ModelInstanceScore
from gpustack.policies.scorers.status_scorer import StatusScorer
from gpustack.schemas.inference_backend import (
    InferenceBackend,
    get_built_in_backend,
    VersionConfig,
    VersionConfigDict,
)
from gpustack.schemas.model_files import ModelFile, ModelFileStateEnum
from gpustack.schemas.model_routes import (
    ModelRoute,
    ModelRouteTarget,
    MyModel,
    TargetStateEnum,
)
from gpustack.schemas.models import (
    BackendEnum,
    BackendSourceEnum,
    ModelSource,
    Model,
    ModelInstance,
    ModelInstanceCreate,
    ModelInstanceStateEnum,
    SourceEnum,
    get_backend,
)
from gpustack.schemas.config import (
    GatewayModeEnum,
    SensitivePredefinedConfig,
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
    ClusterProvider,
)

from gpustack.schemas.users import (
    User,
    is_default_cluster_user,
)
from gpustack.server.bus import Event, EventType, event_bus
from gpustack.server.catalog import get_catalog_draft_models
from gpustack.server.db import async_session
from gpustack.server.services import (
    ModelFileService,
    ModelInstanceService,
    ModelService,
    WorkerService,
    ModelRouteService,
)
from gpustack.cloud_providers.common import (
    get_client_from_provider,
    construct_cloud_instance,
    generate_ssh_key_pair,
)
from gpustack.cloud_providers.abstract import (
    ProviderClientBase,
    CloudInstance,
    InstanceState,
)
from kubernetes_asyncio import client as k8s_client
from gpustack.gateway.client.networking_higress_io_v1_api import (
    NetworkingHigressIoV1Api,
    McpBridgeRegistry,
)
from gpustack.gateway.client.extensions_higress_io_v1_api import (
    ExtensionsHigressIoV1Api,
    WasmPluginMatchRule,
)
from gpustack.gateway.client.networking_istio_io_v1alpha3_api import (
    NetworkingIstioIoV1Alpha3Api,
)
from gpustack.gateway import utils as mcp_handler
from gpustack.gateway import get_async_k8s_config
from gpustack.schemas.model_provider import (
    ModelProvider,
)
from gpustack.gateway.plugins import get_plugin_url_with_name_and_version


logger = logging.getLogger(__name__)


class ModelController:
    def __init__(self, cfg: Config):
        self._config = cfg

        pass

    async def start(self):
        """
        Start the controller.
        """

        async for event in Model.subscribe(source="model_controller"):
            if event.type == EventType.HEARTBEAT:
                continue

            await self._reconcile(event)

    async def _reconcile(self, event: Event):
        """
        Reconcile the model.
        """

        model: Model = event.data
        try:
            async with async_session() as session:
                await sync_replicas(session, model)
                await notify_model_route_target(
                    session=session, model=model, event=event
                )
                await sync_categories_and_meta(session, model, event)
        except Exception as e:
            logger.error(f"Failed to reconcile model {model.name}: {e}")


class ModelInstanceController:
    def __init__(self, cfg: Config):
        self._config = cfg
        self._k8s_config = get_async_k8s_config(cfg=cfg)
        self._disable_gateway = cfg.gateway_mode == GatewayModeEnum.disabled

        pass

    async def start(self):
        """
        Start the controller.
        """
        if not self._disable_gateway:
            base_client = k8s_client.ApiClient(configuration=self._k8s_config)
            self._higress_network_api = NetworkingHigressIoV1Api(base_client)

        async for event in ModelInstance.subscribe(source="model_instance_controller"):
            if event.type == EventType.HEARTBEAT:
                continue

            await self._reconcile(event)

    async def _reconcile(self, event: Event):
        """
        Reconcile the model.
        """

        model_instance: ModelInstance = event.data
        try:
            async with async_session() as session:
                model = await Model.one_by_id(session, model_instance.model_id)
                if not model:
                    return
                model_deleting = model.deleted_at is not None

                if not self._disable_gateway:
                    worker = (
                        await Worker.one_by_id(session, model_instance.worker_id)
                        if model_instance.worker_id
                        else None
                    )
                    await mcp_handler.ensure_model_instance_mcp_bridge(
                        event_type=event.type,
                        model_instance=model_instance,
                        networking_higress_api=self._higress_network_api,
                        namespace=self._config.gateway_namespace,
                        cluster_id=model.cluster_id,
                        worker=worker,
                    )

                if event.type == EventType.DELETED:
                    # trigger model replica sync
                    copied_model = Model.model_validate(model.model_dump())
                    asyncio.create_task(
                        event_bus.publish(
                            Model.__name__.lower(),
                            Event(type=EventType.UPDATED, data=copied_model),
                        )
                    )
                elif model_instance.state == ModelInstanceStateEnum.INITIALIZING:
                    await ensure_instance_model_file(session, model_instance)
                    return

                if model_deleting:
                    return

                await model.refresh(session)
                await sync_ready_replicas(session, model)
        except Exception as e:
            logger.error(
                f"Failed to reconcile model instance {model_instance.name}: {e}"
            )


async def sync_replicas(session: AsyncSession, model: Model):
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
                model_scope_model_id=model.model_scope_model_id,
                model_scope_file_path=model.model_scope_file_path,
                local_path=model.local_path,
                state=ModelInstanceStateEnum.PENDING,
                cluster_id=model.cluster_id,
                draft_model_source=get_draft_model_source(model),
                backend=get_backend(model),
                backend_version=model.backend_version,
            )

            await ModelInstanceService(session).create(instance)
            logger.debug(f"Created model instance for model {model.name}")

    elif len(instances) > model.replicas:
        # Get instances for update lock, to avoid race condition with scheduler
        instances = await ModelInstance.all_by_field(
            session, "model_id", model.id, for_update=True
        )
        candidates = await find_scale_down_candidates(instances, model)

        scale_down_count = len(candidates) - model.replicas
        if scale_down_count > 0:
            scale_down_instances = []
            for candidate in candidates[:scale_down_count]:
                scale_down_instances.append(candidate.model_instance)

            scale_down_instance_names = await ModelInstanceService(
                session
            ).batch_delete(scale_down_instances)
            if scale_down_instance_names:
                logger.debug(f"Deleted model instances: {scale_down_instance_names}")


async def distribute_models_to_user(
    session: AsyncSession, model: ModelRoute, event: Event
):
    if len(event.changed_fields) == 0 and event.type == EventType.CREATED:
        return
    model_dict = model.model_dump(exclude={"instances", "users", "cluster"})
    model_id = model.id
    to_delete_model_user_ids: Set[int] = set()
    to_update_model_user_ids: Set[int] = set()
    to_create_model_user_ids: Set[int] = set()
    if event.type == EventType.DELETED:
        users = await User.all_by_fields(
            session, fields={"deleted_at": None, "is_admin": False}
        )
        for user in users:
            to_delete_model_user_ids.add(user.id)
    if event.type == EventType.UPDATED:
        changed_fields = event.changed_fields.copy()
        changed_users = changed_fields.pop("users", None)
        if changed_users is not None:
            old_users, new_users = changed_users
            old_user_ids = {user.id for user in old_users}
            new_user_ids = {user.id for user in new_users}
            to_create_model_user_ids = new_user_ids - old_user_ids
            to_delete_model_user_ids = old_user_ids - new_user_ids
        if len(changed_fields) > 0:
            users = await User.all_by_fields(
                session,
                fields={"deleted_at": None, "is_admin": False},
                extra_conditions=[User.routes.any(ModelRoute.id == model.id)],
            )
            current_user_ids = {user.id for user in users}
            to_update_model_user_ids = current_user_ids - to_create_model_user_ids
    if event.type == EventType.CREATED:
        users = await User.all_by_fields(
            session,
            fields={"deleted_at": None, "is_admin": False},
            extra_conditions=[User.routes.any(ModelRoute.id == model.id)],
        )
        for user in users:
            to_create_model_user_ids.add(user.id)
    tasks = []
    for event_type, ids in [
        (EventType.CREATED, to_create_model_user_ids),
        (EventType.DELETED, to_delete_model_user_ids),
        (EventType.UPDATED, to_update_model_user_ids),
    ]:
        for user_id in ids:
            my_model = MyModel(
                pid=f"{model_id}:{user_id}",
                user_id=user_id,
                **model_dict,
            )
            tasks.append(
                event_bus.publish(
                    MyModel.__name__.lower(), Event(type=event_type, data=my_model)
                )
            )
    if tasks:
        await asyncio.gather(*tasks)


async def ensure_instance_model_file(session: AsyncSession, instance: ModelInstance):
    """
    Synchronize the model file of the model instance.
    """
    if instance.worker_id is None:
        # Not scheduled yet
        return

    instance = await ModelInstance.one_by_id(
        session,
        instance.id,
        options=[
            selectinload(ModelInstance.model_files),
        ],
    )
    if not instance:
        return

    if len(instance.model_files) > 0:
        await sync_instance_files_state(session, instance, instance.model_files)
        return

    retry_model_files = []
    model_files = await get_or_create_model_files_for_instance(session, instance)
    draft_model_files = []
    if instance.draft_model_source:
        draft_model_files = await get_or_create_model_files_for_instance(
            session, instance, is_draft_model=True
        )
    for model_file in model_files + draft_model_files:
        if model_file.state == ModelFileStateEnum.ERROR:
            # Retry the download
            retry_model_files.append(model_file.readable_source)

            model_file.state = ModelFileStateEnum.DOWNLOADING
            model_file.download_progress = 0
            model_file.state_message = ""
            await model_file.update(session, auto_commit=False)

    if retry_model_files:
        await session.commit()
        logger.info(
            f"Retrying download for model files {retry_model_files} for model instance {instance.name}"
        )

    instance = await ModelInstance.one_by_id(session, instance.id)
    instance.model_files = model_files
    instance.draft_model_files = draft_model_files
    await sync_instance_files_state(session, instance, model_files + draft_model_files)


async def get_or_create_model_files_for_instance(
    session: AsyncSession, instance: ModelInstance, is_draft_model: bool = False
) -> List[ModelFile]:
    """
    Get or create model files for the given model instance.
    If is_draft_model is True, get or create model files for the draft model.
    """

    model_files = await get_model_files_for_instance(session, instance, is_draft_model)
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

    model_source = instance
    if is_draft_model:
        model_source = instance.draft_model_source
    # Create model files for the missing worker IDs.
    for worker_id in missing_worker_ids:
        model_file = ModelFile(
            source=model_source.source,
            huggingface_repo_id=model_source.huggingface_repo_id,
            huggingface_filename=model_source.huggingface_filename,
            model_scope_model_id=model_source.model_scope_model_id,
            model_scope_file_path=model_source.model_scope_file_path,
            local_path=model_source.local_path,
            state=ModelFileStateEnum.DOWNLOADING,
            worker_id=worker_id,
            source_index=model_source.model_source_index,
        )
        await ModelFile.create(session, model_file)
        logger.info(
            f"Created model file for model instance {instance.name} and worker {worker_id}"
        )

    # After creating the model files, fetch them again to return the complete list.
    return await get_model_files_for_instance(session, instance, is_draft_model)


async def get_model_files_for_instance(
    session: AsyncSession, instance: ModelInstance, is_draft_model: bool = False
) -> List[ModelFile]:
    """
    Get the model files for the given model instance.
    If draft_model is provided, get the model files for the draft model.
    """
    worker_ids = _get_worker_ids_for_file_download(instance)

    model_source: ModelSource = instance
    if is_draft_model:
        model_source = instance.draft_model_source

    model_files = await ModelFileService(session).get_by_source_index(
        model_source.model_source_index
    )
    model_files = [
        model_file for model_file in model_files if model_file.worker_id in worker_ids
    ]

    if model_source.source == SourceEnum.LOCAL_PATH and model_source.local_path:
        # If the source is local path, get the model files with the same local path.
        local_path_model_files = await ModelFileService(session).get_by_resolved_path(
            model_source.local_path
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


def get_draft_model_source(model: Model) -> Optional[ModelSource]:
    """
    Get the model source for the draft model.
    First check the catalog for the draft model.
    If not found, get the model source empirically to support custom draft models.
    """
    if model.speculative_config is None or not model.speculative_config.draft_model:
        return None

    draft_model = model.speculative_config.draft_model
    catalog_draft_models = get_catalog_draft_models()
    for catalog_draft_model in catalog_draft_models:
        if catalog_draft_model.name == draft_model:
            return catalog_draft_model

    # If draft_model looks like a path, assume it's a local path.
    if draft_model.startswith("/"):
        return ModelSource(source=SourceEnum.LOCAL_PATH, local_path=draft_model)

    # Otherwise, assume it comes from the same source as the main model.
    if model.source == SourceEnum.HUGGING_FACE:
        return ModelSource(
            source=SourceEnum.HUGGING_FACE,
            huggingface_repo_id=draft_model,
        )
    elif model.source == SourceEnum.MODEL_SCOPE:
        return ModelSource(
            source=SourceEnum.MODEL_SCOPE,
            model_scope_model_id=draft_model,
        )
    return None


async def find_scale_down_candidates(
    instances: List[ModelInstance], model: Model
) -> List[ModelInstanceScore]:
    try:
        placement_scorer = PlacementScorer(
            model, instances, scale_type=ScaleTypeEnum.SCALE_DOWN
        )
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


async def get_cluster_registry(
    session: AsyncSession, cluster_id: int
) -> Optional[McpBridgeRegistry]:
    cluster_user = await User.one_by_field(
        session=session,
        field="cluster_id",
        value=cluster_id,
        options=[selectinload(User.cluster)],
    )
    if is_default_cluster_user(cluster_user):
        return None
    cluster_registry = mcp_handler.cluster_registry(cluster_user.cluster)
    if cluster_registry is None:
        return None
    return cluster_registry


async def sync_model_route_mapper(
    cfg: Config,
    extensions_api: ExtensionsHigressIoV1Api,
    event_type: EventType,
    ingress_name: str,
    route_name: str,
    destinations: mcp_handler.DestinationTupleList,
    fallback_destinations: mcp_handler.DestinationTupleList,
):
    """
    Synchronize the model route mapper.
    """
    mapper_namespace = cfg.get_namespace()
    if mapper_namespace == cfg.gateway_namespace:
        mapper_namespace = None
    model_name_to_registries: Dict[str, List[str]] = {}
    for _, model_name, registry in destinations:
        if route_name == model_name:
            # Skip self mapping
            continue
        registries = model_name_to_registries.setdefault(model_name, [])
        registries.append(registry.get_service_name())
    fallback_model_name_to_registries: Dict[str, List[str]] = {}
    for _, model_name, registry in fallback_destinations:
        registries = fallback_model_name_to_registries.setdefault(model_name, [])
        registries.append(registry.get_service_name())
    expected_rules = mcp_handler.get_model_route_mapper_config(
        ingress_name=ingress_name,
        route_name=route_name,
        namespace=mapper_namespace,
        model_name_to_registries=model_name_to_registries,
        fallback_model_name_to_registries=fallback_model_name_to_registries,
    )
    try:
        if event_type == EventType.DELETED or expected_rules is None:
            await extensions_api.delete_wasmplugin(
                namespace=cfg.gateway_namespace,
                name=ingress_name,
            )
            return
    except k8s_client.ApiException as e:
        if e.status == 404:
            return
        if e.status != 404:
            logger.error(
                f"Failed to get model route mapper wasmplugin {ingress_name}: {e}"
            )
            raise

    plugin_spec = mcp_handler.model_route_mapper_plugin_spec(
        url=get_plugin_url_with_name_and_version("model-mapper", "2.0.0", cfg),
        match_rules=expected_rules,
    )
    await mcp_handler.ensure_wasm_plugin(
        api=extensions_api,
        namespace=cfg.gateway_namespace,
        name=ingress_name,
        expected=plugin_spec,
        extra_labels=mcp_handler.model_route_selector,
    )


async def ensure_route_ai_proxy_config(
    cfg: Config,
    model_route_id: int,
    extensions_api: ExtensionsHigressIoV1Api,
    route_destinations: mcp_handler.DestinationTupleList,
    fallback_destinations: mcp_handler.DestinationTupleList,
):
    service_namespace_prefix = cfg.get_namespace() + "/"
    if cfg.get_namespace() == cfg.gateway_namespace:
        service_namespace_prefix = ""
    operating_id = mcp_handler.model_route_cleanup_prefix(model_route_id)
    ingress_name = mcp_handler.model_route_ingress_name(model_route_id)
    fallback_ingress_name = mcp_handler.fallback_ingress_name(ingress_name)

    def destination_contains_provider(input: mcp_handler.DestinationTupleList) -> bool:
        for _, _, registry in input:
            # the provider registry starts with the provider id prefix
            if registry.name.startswith(mcp_handler.provider_id_prefix):
                return True
        return False

    has_provider = destination_contains_provider(route_destinations)
    fallback_has_provider = destination_contains_provider(fallback_destinations)
    expected_providers = []
    expected_match_rules = []
    # cross provider needs to configure ai_proxy
    if has_provider != fallback_has_provider:
        unique_registry_services: Set[str] = set(
            registry.get_service_name()
            for _, _, registry in route_destinations
            if (not registry.name.startswith(mcp_handler.provider_id_prefix))
        )
        unique_fallback_registry_services: Set[str] = set(
            registry.get_service_name()
            for _, _, registry in fallback_destinations
            if (not registry.name.startswith(mcp_handler.provider_id_prefix))
        )
        if len(unique_registry_services) > 0:
            expected_providers.append(
                mcp_handler.ai_proxy_openai_provider_config(operating_id)
            )
            expected_match_rules.append(
                WasmPluginMatchRule(
                    config={
                        "activeProviderId": operating_id,
                    },
                    configDisable=False,
                    service=list(unique_registry_services),
                    ingress=[f"{service_namespace_prefix}{ingress_name}"],
                )
            )
        # same logic for fallback
        if len(unique_fallback_registry_services) > 0:
            expected_providers.append(
                mcp_handler.ai_proxy_openai_provider_config(f"{operating_id}-fallback")
            )
            expected_match_rules.append(
                WasmPluginMatchRule(
                    config={
                        "activeProviderId": f"{operating_id}-fallback",
                    },
                    configDisable=False,
                    service=list(unique_fallback_registry_services),
                    ingress=[f"{service_namespace_prefix}{fallback_ingress_name}"],
                )
            )

    await mcp_handler.ensure_gpustack_ai_proxy_config(
        extensions_api=extensions_api,
        namespace=cfg.gateway_namespace,
        expected_providers=expected_providers,
        expected_match_rules=expected_match_rules,
        operating_id_prefix=operating_id,
    )


async def sync_gateway(
    session: AsyncSession,
    event: Event,
    cfg: Config,
    model_route: ModelRoute,
    networking_api: k8s_client.NetworkingV1Api,
    extensions_api: ExtensionsHigressIoV1Api,
    istio_networking_api: NetworkingIstioIoV1Alpha3Api,
):
    event_type = event.type
    model_route_from_db = await ModelRoute.one_by_id(session, model_route.id)
    destinations = []
    fallback_destinations = []
    if not model_route_from_db:
        event_type = EventType.DELETED
    if event.type != EventType.DELETED:
        destinations, fallback_destinations = await calculate_destinations(
            session, model_route
        )
    ingress_name = mcp_handler.model_route_ingress_name(model_route.id)
    await sync_model_route_mapper(
        cfg=cfg,
        extensions_api=extensions_api,
        event_type=event_type,
        ingress_name=ingress_name,
        route_name=model_route.name,
        destinations=destinations,
        fallback_destinations=fallback_destinations,
    )
    if event_type != EventType.DELETED and len(destinations) == 0:
        destinations = fallback_destinations
    await mcp_handler.ensure_model_ingress(
        event_type=event_type,
        ingress_name=ingress_name,
        route_name=model_route.name,
        namespace=cfg.get_namespace(),
        destinations=destinations,
        networking_api=networking_api,
        included_generic_route=False,
        included_proxy_route=model_route.generic_proxy,
    )
    fallback_event_type = event_type
    if len(fallback_destinations) == 0:
        fallback_event_type = EventType.DELETED
    # Fallback ingress
    await mcp_handler.ensure_model_ingress(
        event_type=fallback_event_type,
        ingress_name=mcp_handler.fallback_ingress_name(ingress_name),
        route_name=model_route.name,
        namespace=cfg.get_namespace(),
        destinations=fallback_destinations,
        networking_api=networking_api,
        included_generic_route=False,
        included_proxy_route=model_route.generic_proxy,
        extra_annotations=mcp_handler.higress_http_header_matcher(
            "exact", "x-higress-fallback-from", ingress_name
        ),
    )
    # Fallback filter
    await mcp_handler.ensure_fallback_filter(
        event_type=fallback_event_type,
        ingress_name=ingress_name,
        namespace=cfg.get_namespace(),
        networking_istio_api=istio_networking_api,
    )
    # ensure ai proxy config
    await ensure_route_ai_proxy_config(
        cfg=cfg,
        model_route_id=model_route.id,
        extensions_api=extensions_api,
        route_destinations=destinations,
        fallback_destinations=fallback_destinations,
    )


def flatten_destinations(
    weight_to_count: List[Tuple[int, int, mcp_handler.DestinationTupleList]],
    max_weight: Optional[int] = 0,
) -> mcp_handler.DestinationTupleList:
    persentage_list = mcp_handler.hamilton_calculate_weight(
        [(weight, count) for weight, count, _ in weight_to_count],
        max_weight=max_weight,
    )
    flatten_registry_list: mcp_handler.DestinationTupleList = []
    index = 0
    for _, _, registry_list_part in weight_to_count:
        for count, model_name, registry in registry_list_part:
            total_percentage = sum(persentage_list[index : index + count])
            index += count
            if total_percentage != 0:
                flatten_registry_list.append((total_percentage, model_name, registry))
    return flatten_registry_list


async def calculate_destinations(
    session: AsyncSession,
    model_route: ModelRoute,
) -> Tuple[mcp_handler.DestinationTupleList, mcp_handler.DestinationTupleList]:
    """
    return persentage Tuple for each registry with model name and the fallback registry
    """
    weight_to_count: List[Tuple[int, int, mcp_handler.DestinationTupleList]] = []
    fallback_weight_to_count: List[
        Tuple[int, int, mcp_handler.DestinationTupleList]
    ] = []
    targets = await ModelRouteTarget.all_by_field(session, "route_id", model_route.id)
    for target in targets:
        if target.state != TargetStateEnum.ACTIVE:
            continue
        to_extend: mcp_handler.DestinationTupleList = []
        if target.model_id is not None:
            model = await Model.one_by_id(session, target.model_id)
            if model is None:
                continue
            to_extend = await calculate_model_destinations(session, model)
        elif target.provider_id is not None:
            to_extend = await provider_destinations(
                session=session,
                provider_id=target.provider_id,
                provider_model_name=target.provider_model_name,
            )
        if to_extend is None or len(to_extend) == 0:
            # no valid destination found
            continue
        count = sum([count for count, _, _ in to_extend])
        weight_to_count.append((target.weight, count, to_extend))
        if (
            target.fallback_status_codes is not None
            and len(target.fallback_status_codes) > 0
        ):
            fallback_weight_to_count.append((target.weight, count, to_extend))
    if len(weight_to_count) == 0:
        return [], []

    flatten_registry_list = flatten_destinations(weight_to_count)
    fallback_registry_list = []
    if len(fallback_weight_to_count) > 0:
        # fallback might have 0 weight, so set max_weight to 1
        fallback_registry_list = flatten_destinations(
            fallback_weight_to_count, max_weight=1
        )

    return flatten_registry_list, fallback_registry_list


async def provider_destinations(
    session: AsyncSession,
    provider_id: int,
    provider_model_name: str,
) -> mcp_handler.DestinationTupleList:
    """
    return count dict for provider registry
    """
    provider = await ModelProvider.one_by_id(session, provider_id)
    if provider is None:
        return []
    return [(1, provider_model_name, mcp_handler.provider_registry(provider))]


async def calculate_model_destinations(
    session: AsyncSession,
    model: Model,
) -> mcp_handler.DestinationTupleList:
    """
    return count dict for each registry
    """
    # find out is handling default cluster's model
    cluster_registry = await get_cluster_registry(session, model.cluster_id)
    if cluster_registry is not None:
        return [(1, model.name, cluster_registry)]

    instances = await ModelInstance.all_by_field(session, "model_id", model.id)
    instances = [
        instance
        for instance in instances
        if instance.worker_ip is not None
        and instance.port is not None
        and instance.worker_ip != ""
        and instance.state == ModelInstanceStateEnum.RUNNING
    ]
    worker_list = await Worker.all_by_fields(
        session=session,
        fields={
            "cluster_id": model.cluster_id,
            "deleted_at": None,
        },
        extra_conditions=[
            Worker.id.in_(
                [
                    instance.worker_id
                    for instance in instances
                    if instance.worker_id is not None
                ]
            )
        ],
    )
    workers = {worker.id: worker for worker in worker_list}
    registry_list = mcp_handler.model_instances_registry_list(instances, workers)

    return registry_list


class WorkerController:
    def __init__(self, cfg: Config):
        self._provisioning = WorkerProvisioningController(cfg)

    async def start(self):
        """
        Start the controller.
        """

        async for event in Worker.subscribe(source="worker_controller"):
            if event.type == EventType.HEARTBEAT:
                continue
            try:
                await self._reconcile(event)
                await self._provisioning._reconcile(event)
                await self._notify_parents(event)
            except Exception as e:
                logger.error(f"Failed to reconcile worker: {e}")

    async def _reconcile(self, event: Event):
        """
        Delete instances base on the worker state and event type.
        """
        if event.type not in (EventType.UPDATED, EventType.DELETED):
            return
        worker: Worker = event.data
        if not worker:
            return

        if worker.state.is_provisioning and worker.state != WorkerStateEnum.DELETING:
            # Skip reconciliation for provisioning and deleting workers.
            # There is a dedicated controller to handle provisioning.
            return

        if event.type == EventType.UPDATED:
            changed_fields = event.changed_fields
            if not changed_fields or "state" not in changed_fields:
                # No state change
                return

        async with async_session() as session:
            instances = await ModelInstance.all_by_field(
                session, "worker_name", worker.name
            )
            if not instances:
                return

            if event.type == EventType.DELETED:
                instance_names = await ModelInstanceService(session).batch_delete(
                    instances
                )
                if instance_names:
                    logger.info(
                        f"Delete instance {', '.join(instance_names)} "
                        f"since worker {worker.name} is deleted"
                    )
                return

            if (
                worker.unreachable
                or worker.state == WorkerStateEnum.UNREACHABLE
                or worker.state == WorkerStateEnum.NOT_READY
            ):
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
        update_instances = []

        for instance in instances:
            if instance.state == old_state:
                update_instances.append(instance)

        if update_instances:
            patch = {"state": new_state, "state_message": new_state_message}
            instance_names = await ModelInstanceService(session).batch_update(
                update_instances, patch
            )

        if instance_names:
            logger.info(
                f"Marked instance {', '.join(instance_names)} {new_state} "
                f"since {log_update_reason}"
            )

    async def _notify_parents(self, event: Event):
        if event.type not in (EventType.UPDATED, EventType.DELETED):
            return
        worker: Worker = event.data
        changed_fields = event.changed_fields
        if not worker or (not changed_fields and event.type != EventType.DELETED):
            return
        state_changed: Optional[Tuple[Any, Any]] = (changed_fields or {}).get(
            "state", None
        )
        proxy_mode_changed: Optional[Tuple[Any, Any]] = (changed_fields or {}).get(
            "proxy_mode", None
        )
        should_notify = (
            state_changed is not None
            or proxy_mode_changed is not None
            or event.type == EventType.DELETED
        )
        if not should_notify:
            return
        async with async_session() as session:
            if worker.worker_pool_id is not None:
                worker_pool = await WorkerPool.one_by_id(
                    session,
                    worker.worker_pool_id,
                    options=[selectinload(WorkerPool.pool_workers)],
                )
                if worker_pool is not None:
                    copied_pool = WorkerPool(**worker_pool.model_dump())
                    await event_bus.publish(
                        copied_pool.__class__.__name__.lower(),
                        Event(
                            type=EventType.UPDATED,
                            data=copied_pool,
                        ),
                    )
            if worker.cluster_id is not None:
                cluster = await Cluster.one_by_id(
                    session,
                    worker.cluster_id,
                    options=[
                        selectinload(Cluster.cluster_workers),
                        selectinload(Cluster.cluster_models),
                    ],
                )
                if cluster is not None:
                    copied_cluster = Cluster(**cluster.model_dump())
                    await event_bus.publish(
                        copied_cluster.__class__.__name__.lower(),
                        Event(
                            type=EventType.UPDATED,
                            data=copied_cluster,
                        ),
                    )


class InferenceBackendController:
    """
    Inference backend controller initializes built-in and community backends in the database.
    """

    async def start(self):
        async with async_session() as session:
            # Initialize built-in backends
            await self._init_built_in_backends(session)

            # Initialize community backends
            await self._init_community_backends(session)

    async def _init_built_in_backends(self, session: AsyncSession):
        """Initialize built-in backends in the database."""
        for built_in_backend in get_built_in_backend():
            if built_in_backend.backend_name == BackendEnum.CUSTOM.value:
                continue

            backend = await InferenceBackend.one_by_field(
                session, "backend_name", built_in_backend.backend_name
            )

            if not backend:
                # Create new built-in backend with backend_source
                built_in_backend.backend_source = BackendSourceEnum.BUILT_IN
                built_in_backend.enabled = True
                await InferenceBackend.create(session, built_in_backend)
                logger.info(
                    f"Init built-in backend {built_in_backend.backend_name} in database"
                )
            elif backend.backend_source is None:
                # Update existing backend without backend_source
                backend.backend_source = BackendSourceEnum.BUILT_IN
                if backend.enabled is None:
                    backend.enabled = True
                    await backend.update(
                        session,
                        {
                            "backend_source": BackendSourceEnum.BUILT_IN,
                            "enabled": (
                                backend.enabled if backend.enabled is not None else True
                            ),
                        },
                    )
                    logger.info(
                        f"Updated backend_source for existing built-in backend {backend.backend_name}"
                    )

    async def _init_community_backends(self, session: AsyncSession):
        """Load community backends from community-inference-backends.yaml into database."""
        try:
            # Get the path to community-inference-backends.yaml
            yaml_file = files("gpustack.assets").joinpath(
                "community-inference-backends.yaml"
            )

            if not yaml_file.is_file():
                logger.debug(
                    "community-inference-backends.yaml not found, skipping community backend initialization"
                )
                return

            yaml_data = yaml.safe_load(yaml_file.read_text())

            if not yaml_data:
                logger.debug(
                    "No community backends found in community-inference-backends.yaml"
                )
                return

            if not isinstance(yaml_data, list):
                logger.error(
                    f"Invalid community-inference-backends.yaml format: expected list, got {type(yaml_data).__name__}"
                )
                return

            for backend_config in yaml_data:
                await self._upsert_community_backend(session, backend_config)

            logger.debug(
                "Community backends initialized from community-inference-backends.yaml"
            )

        except (ModuleNotFoundError, FileNotFoundError):
            # community_backends directory or yaml file does not exist
            logger.debug(
                "Community backends directory or file not found, skipping initialization"
            )
        except Exception as e:
            logger.error(f"Failed to initialize community backends: {e}")

    async def _upsert_community_backend(self, session: AsyncSession, config: dict):
        """Create or update a community backend from YAML configuration."""
        backend_name = config.get("backend_name")
        if not backend_name:
            return

        # Prepare backend data
        allowed_keys = [
            "backend_name",
            "version_configs",
            "default_version",
            "default_backend_param",
            "default_run_command",
            "default_entrypoint",
            "health_check_path",
            "description",
            "icon",
            "default_env",
        ]
        backend_data = {k: config[k] for k in allowed_keys if k in config}

        # Set backend source
        backend_data["backend_source"] = BackendSourceEnum.COMMUNITY
        backend_data["enabled"] = False

        # Convert version_configs to VersionConfigDict
        if 'version_configs' in backend_data and backend_data['version_configs']:
            version_config_dict = {}
            for version, ver_config in backend_data['version_configs'].items():
                # All versions loaded from YAML are predefined versions
                # Convert framework information to built_in_frameworks

                frameworks = None
                if 'built_in_frameworks' in ver_config:
                    frameworks = ver_config['built_in_frameworks']
                elif (
                    'custom_framework' in ver_config and ver_config['custom_framework']
                ):
                    # Even if YAML uses custom_framework, convert it to built_in_frameworks
                    frameworks = [ver_config['custom_framework']]

                # Set built_in_frameworks and clear custom_framework
                if frameworks:
                    ver_config['built_in_frameworks'] = (
                        frameworks if isinstance(frameworks, list) else [frameworks]
                    )
                else:
                    # If no framework specified, use empty list to mark as predefined version
                    ver_config['built_in_frameworks'] = []

                # Ensure custom_framework is None (predefined versions should not have custom_framework)
                ver_config['custom_framework'] = None

                version_config_dict[version] = VersionConfig(**ver_config)

            backend_data['version_configs'] = VersionConfigDict(
                root=version_config_dict
            )

        # Upsert: update if exists, create if not
        existing = await InferenceBackend.one_by_field(
            session, "backend_name", backend_name
        )
        if existing:
            # Smart merge logic to preserve user customizations

            # 1. Merge version_configs: preserve user custom versions, update YAML versions
            if 'version_configs' in backend_data and backend_data['version_configs']:
                yaml_versions = backend_data['version_configs'].root
                existing_versions = (
                    existing.version_configs.root if existing.version_configs else {}
                )

                # Create merged version dictionary
                merged_versions = {}

                # First add all YAML versions (overwrite old versions with same name)
                for version, config in yaml_versions.items():
                    merged_versions[version] = config

                # Then add user custom versions (built_in_frameworks is None)
                for version, config in existing_versions.items():
                    if (
                        config.built_in_frameworks is None
                        and version not in yaml_versions
                    ):
                        # This is a user custom version not in YAML, preserve it
                        merged_versions[version] = config

                backend_data['version_configs'] = VersionConfigDict(
                    root=merged_versions
                )

            # 2. Preserve user-modified enabled status (if user enabled it, don't reset to False)
            if existing.enabled:
                backend_data['enabled'] = True

            # 3. Merge default_env (preserve user-added environment variables)
            if existing.default_env:
                if 'default_env' in backend_data and backend_data['default_env']:
                    # Merge: YAML environment variables + user-added environment variables
                    merged_env = dict(existing.default_env)
                    merged_env.update(backend_data['default_env'])
                    backend_data['default_env'] = merged_env
                else:
                    # YAML doesn't define it, preserve user's
                    backend_data['default_env'] = existing.default_env

            # 4. Update database
            await existing.update(session, backend_data)
        else:
            backend = InferenceBackend(**backend_data)
            await InferenceBackend.create(session, backend)


class ModelFileController:
    """
    Model file controller syncs the model file download status to related model instances.
    """

    async def start(self):
        """
        Start the controller.
        """

        async for event in ModelFile.subscribe(source="model_file_controller"):
            if event.type == EventType.CREATED or event.type == EventType.UPDATED:
                await self._reconcile(event)

    async def _reconcile(self, event: Event):
        """
        Reconcile the model file.
        """

        file: ModelFile = event.data
        try:
            async with async_session() as session:
                file = await ModelFile.one_by_id(
                    session,
                    file.id,
                    options=[
                        selectinload(ModelFile.instances),
                        selectinload(ModelFile.draft_instances),
                    ],
                )

            if not file:
                # In case the file is deleted
                return

            for instance in file.instances + file.draft_instances:
                async with async_session() as session:
                    await sync_instance_files_state(session, instance, [file])
        except Exception as e:
            logger.error(f"Failed to reconcile model file {file.id}: {e}")


async def sync_instance_files_state(
    session: AsyncSession, instance: ModelInstance, files: List[ModelFile]
):
    for file in files:
        if file.worker_id == instance.worker_id:
            is_draft_model = _is_draft_model_file(file, instance)
            if is_draft_model:
                await sync_main_worker_model_file_state(
                    session, file, instance, is_draft_model=True
                )
            else:
                await sync_main_worker_model_file_state(session, file, instance)
        else:
            await sync_distributed_model_file_state(session, file, instance)


def _is_draft_model_file(file: ModelFile, instance: ModelInstance) -> bool:
    """
    Check if the model file is the draft model file for the given model instance.
    """
    if not instance.draft_model_source:
        return False

    if file.model_source_index == instance.draft_model_source.model_source_index:
        return True

    # The model uses a local path as its draft source, but the model file may come from a remote source.
    # Match by resolved path.
    if (
        instance.draft_model_source.source == SourceEnum.LOCAL_PATH
        and file.resolved_paths
        and file.resolved_paths[0] == instance.draft_model_source.local_path
    ):
        return True

    return False


async def sync_main_worker_model_file_state(
    session: AsyncSession,
    file: ModelFile,
    instance: ModelInstance,
    is_draft_model: bool = False,
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

    # Downloading
    if file.state == ModelFileStateEnum.DOWNLOADING:
        if instance.state == ModelInstanceStateEnum.INITIALIZING:
            # Download started
            instance.state = ModelInstanceStateEnum.DOWNLOADING
            instance.download_progress = 0
            instance.state_message = ""
            need_update = True
        elif instance.state == ModelInstanceStateEnum.DOWNLOADING:
            # Update download progress
            if (
                is_draft_model
                and file.download_progress != instance.draft_model_download_progress
                and instance.draft_model_download_progress != 100
            ):
                # For the draft model file
                instance.draft_model_download_progress = file.download_progress
                need_update = True
            elif (
                file.download_progress != instance.download_progress
                and instance.download_progress != 100
            ):
                # For the main model file
                instance.download_progress = file.download_progress
                need_update = True

    # Download completed
    elif file.state == ModelFileStateEnum.READY and (
        instance.state == ModelInstanceStateEnum.DOWNLOADING
        or instance.state == ModelInstanceStateEnum.INITIALIZING
    ):
        if is_draft_model and (
            instance.draft_model_download_progress != 100
            or not instance.draft_model_resolved_path
        ):
            # Download completed for the draft model file
            instance.draft_model_download_progress = 100
            instance.draft_model_resolved_path = file.resolved_paths[0]
            need_update = True
        elif not is_draft_model and (
            instance.download_progress != 100 or not instance.resolved_path
        ):
            # Download completed for the main model file
            instance.download_progress = 100
            instance.resolved_path = file.resolved_paths[0]
            need_update = True

        if model_instance_download_completed(instance):
            # All files are downloaded
            instance.state = ModelInstanceStateEnum.STARTING
            instance.state_message = ""
            need_update = True
        elif instance.state == ModelInstanceStateEnum.INITIALIZING:
            # one but not all files downloaded, turn to DOWNLOADING state
            instance.state = ModelInstanceStateEnum.DOWNLOADING
            instance.state_message = ""
            need_update = True

    # Download error
    elif file.state == ModelFileStateEnum.ERROR:
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

    if instance.draft_model_source and instance.draft_model_download_progress != 100:
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
            ifname="",
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
                **pool.labels,
            },
            state=WorkerStateEnum.PENDING,
            status=WorkerStatus.get_default_status(),
        )
        new_workers.append(new_worker)
    return new_workers


class WorkerPoolController:
    """Worker pool controller creates new workers based on the worker pool configuration."""

    async def start(self):
        async for event in WorkerPool.subscribe(source="worker_pool_controller"):
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
        async with async_session() as session:
            pool = await WorkerPool.one_by_id(
                session, event.data.id, options=[selectinload(WorkerPool.cluster)]
            )
            if pool is None or pool.deleted_at is not None:
                return
            # mark the data to avoid read after commit
            cluster_name = pool.cluster.name
            cluster = pool.cluster
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
            if cluster.state == ClusterStateEnum.PENDING:
                cluster.state = ClusterStateEnum.PROVISIONING
                cluster.state_message = None
                await cluster.update(session=session, auto_commit=False)
            await session.commit()
            logger.info(
                f"Created {len(ids)} new workers {ids} for cluster {cluster_name} worker pool {pool_id}"
            )


class WorkerProvisioningController:
    def __init__(self, cfg: Config):
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
        secret_fields = set(SensitivePredefinedConfig.model_fields.keys())
        secret_configs = (
            worker.cluster.worker_config.model_dump(include=secret_fields)
            if worker.cluster.worker_config
            else {}
        )
        user_data = await client.construct_user_data(
            server_url=worker.cluster.server_url or cfg.server_external_url,
            token=worker.cluster.registration_token,
            image_name=get_cluster_image_name(worker.cluster.worker_config),
            os_image=worker.worker_pool.os_image,
            secret_configs=secret_configs,
        )
        ssh_key = await Credential.one_by_id(session, worker.ssh_key_id)
        if ssh_key is None:
            raise ValueError(f"SSH key {worker.ssh_key_id} not found")
        to_create = construct_cloud_instance(worker, ssh_key, user_data.format())
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
        if worker.advertise_address is None or worker.advertise_address == "":
            try:
                instance = await client.wait_for_public_ip(worker.external_id)
                worker.advertise_address = (
                    instance.ip_address if instance.ip_address else ""
                )
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

    async def check_server_external_url(self, cluster_server_url: Optional[str] = None):
        server_url = cluster_server_url or self._cfg.server_external_url
        if server_url is None or server_url == "":
            raise ValueError(
                "Cluster's server_url is not configured, Please edit cluster first."
            )
        import aiohttp
        from yarl import URL

        healthz_url = str(URL(server_url) / "healthz")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(healthz_url, timeout=10) as resp:
                    if resp.status != 200:
                        raise ValueError(
                            f"External server healthz url {healthz_url} is not reachable, status code: {resp.status}"
                        )
        except Exception as e:
            raise ValueError(
                f"Failed to check external server healthz url {healthz_url}: {e}"
            )

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
        async with async_session() as session:
            # Fetch the worker from the database
            worker: Worker = await Worker.one_by_id(
                session,
                worker.id,
                options=[
                    selectinload(Worker.cluster),
                    selectinload(Worker.worker_pool),
                ],
            )
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
                if worker.state == WorkerStateEnum.PENDING:
                    await self.check_server_external_url(worker.cluster.server_url)
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
                logger.exception(message)
                await session.rollback()
                await session.refresh(worker)
                worker.state = WorkerStateEnum.ERROR
                worker.state_message = message
                await WorkerService(session).update(
                    worker=worker, source=None, auto_commit=True
                )


class ClusterController:
    def __init__(self, cfg: Config):
        self._cfg = cfg
        self._disable_gateway = cfg.gateway_mode == GatewayModeEnum.disabled
        self._k8s_config = get_async_k8s_config(cfg=cfg)
        pass

    async def start(self):
        """
        Start the controller.
        """
        if self._cfg.gateway_mode != GatewayModeEnum.disabled:
            base_client = k8s_client.ApiClient(configuration=self._k8s_config)
            self._higress_network_api = NetworkingHigressIoV1Api(base_client)

        async for event in Cluster.subscribe(source="cluster_controller"):
            if event.type == EventType.HEARTBEAT:
                continue
            try:
                await self._reconcile(event)
            except Exception as e:
                logger.error(f"Failed to reconcile cluster: {e}")

    async def _reconcile(self, event: Event):
        """
        Reconcile the cluster state.
        """
        await self._sync_cluster_state(event)
        if self._disable_gateway:
            return
        await self._ensure_worker_mcp_bridge(event)

    async def _sync_cluster_state(self, event: Event):
        if event.type == EventType.DELETED:
            return
        cluster: Cluster = event.data
        if not cluster:
            return
        async with async_session() as session:
            cluster: Cluster = await Cluster.one_by_id(
                session, cluster.id, options=[selectinload(Cluster.cluster_workers)]
            )
            if not cluster or cluster.provider in [
                ClusterProvider.Kubernetes,
                ClusterProvider.Docker,
            ]:
                return
            if cluster.workers == 0 and cluster.state != ClusterStateEnum.PENDING:
                cluster.state = ClusterStateEnum.PENDING
                cluster.state_message = (
                    "No workers have been provisioned for this cluster yet."
                )
                await cluster.update(session=session, auto_commit=True)

    async def _ensure_worker_mcp_bridge(self, event: Event):
        """
        The worker registry list for cluster is no longer needed.
        Use empty list to trigger MCPBridge controller to clean up the worker registries
        and proxies when cluster is created or deleted.
        """
        if self._cfg.gateway_mode == GatewayModeEnum.disabled:
            return
        cluster: Cluster = event.data
        mcp_resource_name = mcp_handler.default_mcp_bridge_name
        desired_registries = []
        to_delete_prefix = mcp_handler.cluster_worker_prefix(cluster.id)
        try:
            await mcp_handler.ensure_mcp_bridge(
                client=self._higress_network_api,
                namespace=self._cfg.gateway_namespace,
                mcp_bridge_name=mcp_resource_name,
                desired_registries=desired_registries,
                to_delete_prefix=to_delete_prefix,
            )
        except Exception as e:
            logger.error(f"Failed to ensure MCPBridge for cluster {cluster.name}: {e}")
            raise


async def notify_model_route_target(session: AsyncSession, model: Model, event: Event):
    if event.type == EventType.DELETED:
        return
    should_notify = False
    if event.changed_fields is not None:
        related_fields = ["ready_replicas", "replicas"]
        for field in related_fields:
            if field in event.changed_fields:
                should_notify = True
                break
    model: Model = await Model.one_by_id(
        session=session,
        id=model.id,
        options=[
            selectinload(Model.model_route_targets),
        ],
    )
    if not model:
        return
    targets = model.model_route_targets
    for target in targets:
        if should_notify:
            target_copy = ModelRouteTarget(**target.model_dump())
            await event_bus.publish(
                target_copy.__class__.__name__.lower(),
                Event(
                    type=EventType.UPDATED,
                    data=target_copy,
                    changed_fields={
                        "model": (
                            {},
                            {
                                "id": model.id,
                                "name": model.name,
                                "ready_replicas": model.ready_replicas,
                                "replicas": model.replicas,
                            },
                        )
                    },
                ),
            )


async def sync_categories_and_meta(session: AsyncSession, model: Model, event: Event):
    if event.type == EventType.DELETED:
        return
    model: Model = await Model.one_by_id(
        session=session,
        id=model.id,
        options=[
            selectinload(Model.model_routes),
        ],
    )
    if not model:
        return
    routes = model.model_routes
    for route in routes:
        # created_by_model default to false if not set
        if not route.created_by_model:
            continue
        if route.categories != model.categories or route.meta != model.meta:
            await ModelRouteService(session).update(
                model_route=route,
                source={"categories": model.categories, "meta": model.meta},
                auto_commit=True,
            )


class ModelProviderController:
    def __init__(self, cfg: Config):
        self._config = cfg
        self._disable_gateway = cfg.gateway_mode == GatewayModeEnum.disabled
        self._k8s_config = get_async_k8s_config(cfg=cfg)

    async def start(self):
        if self._disable_gateway:
            return
        if not self._disable_gateway:
            base_client = k8s_client.ApiClient(configuration=self._k8s_config)
            self._higress_network_api = NetworkingHigressIoV1Api(base_client)
            self._higress_extension_api = ExtensionsHigressIoV1Api(base_client)

        async for event in ModelProvider.subscribe(source="model_provider_controller"):
            try:
                await self._reconcile(event)
            except Exception as e:
                logger.exception(f"Failed to reconcile model provider: {e}")

    async def _ensure_provider_registry(
        self,
        model_provider: ModelProvider,
        event: Event,
    ):
        provider_registry = mcp_handler.provider_registry(model_provider)
        registry_to_remove = (
            provider_registry is None or event.type == EventType.DELETED
        )
        to_delete_prefix = (
            f"{mcp_handler.provider_id_prefix}{model_provider.id}"
            if registry_to_remove
            else None
        )
        desired_registries = [] if registry_to_remove else [provider_registry]

        provider_proxy = mcp_handler.provider_proxy(model_provider)
        proxy_to_remove = provider_proxy is None or event.type == EventType.DELETED
        to_delete_proxy_prefix = (
            f"proxy-{model_provider.id}" if proxy_to_remove else None
        )
        desired_proxies = [] if proxy_to_remove else [provider_proxy]

        try:
            await mcp_handler.ensure_mcp_bridge(
                client=self._higress_network_api,
                namespace=self._config.gateway_namespace,
                mcp_bridge_name=mcp_handler.default_mcp_bridge_name,
                desired_registries=desired_registries,
                desired_proxies=desired_proxies,
                to_delete_prefix=to_delete_prefix,
                to_delete_proxies_prefix=to_delete_proxy_prefix,
            )
        except Exception as e:
            logger.error(
                f"Failed to ensure MCPRegistry for model provider {model_provider.name}: {e}"
            )
            raise

    async def _ensure_provider_ai_proxy_config(self):
        try:
            async with async_session() as session:
                providers = await ModelProvider.all_by_field(
                    session,
                    "deleted_at",
                    None,
                )
                provider_config_list, match_rules = (
                    mcp_handler.provider_proxy_plugin_spec(*providers)
                )
                await mcp_handler.ensure_gpustack_ai_proxy_config(
                    extensions_api=self._higress_extension_api,
                    namespace=self._config.gateway_namespace,
                    expected_providers=provider_config_list,
                    expected_match_rules=match_rules,
                    operating_id_prefix=mcp_handler.provider_id_prefix,
                )
        except Exception as e:
            logger.error(f"Failed to ensure provider's ai_proxy config: {e}")
            raise

    async def _reconcile(self, event: Event):
        """
        Reconcile the model provider.
        """
        model_provider: ModelProvider = event.data
        if not model_provider:
            return
        if event.type == EventType.DELETED:
            await self._ensure_provider_registry(model_provider, event)
            await self._ensure_provider_ai_proxy_config()
            return
        async with async_session() as session:
            model_provider: ModelProvider = await ModelProvider.one_by_id(
                session, model_provider.id
            )
            if not model_provider:
                return
            await self._ensure_provider_registry(model_provider, event)
            await self._ensure_provider_ai_proxy_config()


class ModelRouteTargetController:
    def __init__(self, config: Config):
        self._config = config

    async def start(self):
        async for event in ModelRouteTarget.subscribe(
            source="model_route_target_controller"
        ):
            try:
                await self._reconcile(event)
            except Exception as e:
                logger.exception(f"Failed to reconcile model route target: {e}")

    async def _notify_parents(
        self, session: AsyncSession, target: ModelRouteTarget, event: Event
    ):
        if event.type not in (EventType.UPDATED, EventType.DELETED):
            return
        changed_fields = event.changed_fields
        if not target or (not changed_fields and event.type != EventType.DELETED):
            return
        should_notify_fields = [
            "state",
            "provider_id",
            "model_id",
            "provider_model_name",
            "model",
        ]
        should_notify = event.type == EventType.DELETED
        if not should_notify:
            for field in should_notify_fields:
                if field in (changed_fields or {}):
                    should_notify = True
                    break
        if not should_notify:
            return
        try:
            model_route: ModelRoute = await ModelRoute.one_by_id(
                session, target.route_id
            )
            if not model_route:
                return
            copied_route = ModelRoute.model_validate(model_route.model_dump())
            await event_bus.publish(
                ModelRoute.__name__.lower(),
                Event(type=EventType.UPDATED, data=copied_route),
            )
        except Exception as e:
            logger.error(f"Failed to notify model route for target {target.name}: {e}")

    async def _sync_state(
        self, session: AsyncSession, target: ModelRouteTarget, event: Event
    ):
        if event.type == EventType.DELETED:
            return
        target: ModelRouteTarget = await ModelRouteTarget.one_by_id(session, target.id)
        if not target:
            return
        if target.provider_id is not None:
            target_state = TargetStateEnum.ACTIVE
        if target.model_id is not None:
            model = await Model.one_by_id(session, target.model_id)
            if not model:
                return
            target_state = (
                TargetStateEnum.ACTIVE
                if model.ready_replicas > 0
                else TargetStateEnum.UNAVAILABLE
            )
        if target.state != target_state:
            target.state = target_state
            await target.update(session=session, auto_commit=True)

    async def _reconcile(self, event: Event):
        target: ModelRouteTarget = event.data
        if not target:
            return
        async with async_session() as session:
            await self._notify_parents(session, target, event)
            await self._sync_state(session, target, event)


class ModelRouteController:
    def __init__(self, cfg: Config):
        self._config = cfg
        self._gateway_namespace = cfg.gateway_namespace
        self._k8s_config = get_async_k8s_config(cfg=cfg)
        self._disable_gateway = cfg.gateway_mode == GatewayModeEnum.disabled

    async def start(self):
        if not self._disable_gateway:
            base_client = k8s_client.ApiClient(configuration=self._k8s_config)
            self._networking_api = k8s_client.NetworkingV1Api(base_client)
            self._higress_extension_api = ExtensionsHigressIoV1Api(base_client)
            self._networking_istio_api = NetworkingIstioIoV1Alpha3Api(base_client)

        async for event in ModelRoute.subscribe(source="model_route_controller"):
            try:
                await self._reconcile(event)
            except Exception as e:
                logger.exception(f"Failed to reconcile model route: {e}")

    async def _sync_targets(self, session: AsyncSession, event: Event) -> bool:
        if event.type == EventType.DELETED:
            return False
        model_route: ModelRoute = event.data
        if not model_route:
            return False
        model_route: ModelRoute = await ModelRoute.one_by_id(
            session,
            model_route.id,
            options=[selectinload(ModelRoute.route_targets)],
        )
        if not model_route:
            return False
        target_total = len(model_route.route_targets)
        ready_target_total = len(
            [
                target
                for target in model_route.route_targets
                if target.state == TargetStateEnum.ACTIVE
            ]
        )
        model_route_service = ModelRouteService(session=session)
        if target_total == 0 and model_route.created_by_model:
            await model_route_service.delete(model_route, auto_commit=True)
            return True

        if (
            model_route.targets != target_total
            or model_route.ready_targets != ready_target_total
        ):
            model_route.targets = target_total
            model_route.ready_targets = ready_target_total

            await model_route_service.update(model_route, auto_commit=True)
            return True
        return False

    async def _reconcile(self, event: Event):
        """
        Reconcile the model route.
        """
        model_route: ModelRoute = event.data
        if not model_route:
            return
        async with async_session() as session:
            # sync targets will update model route record so make sure to do it before other operations
            updated = await self._sync_targets(session, event)
            if not self._disable_gateway and not updated:
                await sync_gateway(
                    cfg=self._config,
                    session=session,
                    event=event,
                    networking_api=self._networking_api,
                    extensions_api=self._higress_extension_api,
                    model_route=model_route,
                    istio_networking_api=self._networking_istio_api,
                )
            await distribute_models_to_user(session, model_route, event)
