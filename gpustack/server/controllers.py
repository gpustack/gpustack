import logging
import random
import string
import asyncio
from importlib.resources import files
from functools import partial
from typing import Any, Dict, Iterable, List, Tuple, Optional, Set
from pydantic import BaseModel
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy.orm.attributes import flag_modified

from gpustack.config.config import (
    Config,
    get_cluster_image_name,
)
from gpustack.policies.scorers.offload_layer_scorer import OffloadLayerScorer
from gpustack.policies.scorers.placement_scorer import PlacementScorer, ScaleTypeEnum
from gpustack.policies.scorers.score_chain import (
    ModelInstanceScoreChain,
)
from gpustack.policies.base import ModelInstanceScore
from gpustack.policies.worker_filters.label_matching_filter import label_matching
from gpustack.policies.scorers.status_scorer import StatusScorer
from gpustack.schemas.inference_backend import (
    InferenceBackend,
    get_built_in_backend,
)
from gpustack.schemas.links import ModelRoutePrincipalLink
from gpustack.schemas.model_files import ModelFile, ModelFileStateEnum
from gpustack.schemas.model_routes import (
    ModelRoute,
    ModelRouteTarget,
    MyModel,
    TargetStateEnum,
    effective_route_name,
)
from gpustack.schemas.principals import (
    Principal,
    PrincipalType,
    platform_principal_id,
)
from gpustack.schemas.models import (
    BackendEnum,
    BackendSourceEnum,
    LoraListEntry,
    ModelSource,
    Model,
    ModelInstance,
    ModelInstanceCreate,
    ModelInstanceStateEnum,
    ModelInstanceSubordinateWorker,
    SourceEnum,
    get_backend,
)
from gpustack.schemas.links import (
    ModelInstanceModelFileLink,
    ModelInstanceDraftModelFileLink,
)
from gpustack.utils.lora_model_source import (
    lora_entry_to_model_source,
    lora_route_name_for,
    normalized_lora_list,
    model_base_descriptor,
)
from gpustack.schemas.config import (
    GatewayModeEnum,
    SensitivePredefinedConfig,
)
from gpustack.schemas.cache_services import (
    cache_service_spec_digest,
    CacheService,
    CacheServiceInstance,
    CacheServiceInstanceCreate,
    CacheServiceModeEnum,
    CacheServiceStateEnum,
)
from gpustack.server.cache_provider_catalog import get_cache_provider
from gpustack.server.cache_services import resolve_instance_cache_config_safe
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
    is_default_cluster_principal,
)
from gpustack.schemas.runner_source import (
    InferenceRunnerSource,
    reconcile_runner_overrides,
)
from gpustack.schemas.catalog_source import (
    BUILTIN_CATALOG_SOURCE_NAME,
    CatalogSource,
    normalize_catalog_yaml,
    reconcile_catalog,
)
from gpustack.schemas.inference_backend_source import (
    BUILTIN_BACKEND_SOURCE_NAME,
    InferenceBackendSource,
    normalize_backend_yaml,
    reconcile_backend,
)
from gpustack.server.catalog import read_builtin_catalog_text
from gpustack.schemas.source import SourceTypeEnum
from gpustack.server.sources.core import gather_and_merge, sha256_of
from gpustack.server.bus import (
    Event,
    EventType,
    event_bus,
    event_field,
    resolve_event_id,
)
from gpustack.server.cache import delete_cache_by_key
from gpustack.utils.model_source import get_draft_model_source
from gpustack import envs
from gpustack.server.db import async_session
from gpustack.server.services import (
    ModelFileService,
    ModelInstanceService,
    ModelService,
    WorkerService,
    ModelRouteService,
    collect_route_cache_names,
    revoke_model_access_cache,
)
from gpustack.server.lora_model_routes import cleanup_orphan_lora_routes
from gpustack.utils.model_instance_workers import get_model_instance_worker_match
from gpustack.cloud_providers.common import (
    get_client_from_provider,
    construct_cloud_instance,
    generate_ssh_key_pair,
)
from gpustack.cloud_providers.abstract import (
    ProviderClientBase,
    CloudInstance,
    InstanceProvisioningFailed,
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
    WasmPluginSpec,
)
from gpustack.gateway.client.networking_istio_io_v1alpha3_api import (
    NetworkingIstioIoV1Alpha3Api,
)
from gpustack.gateway import utils as mcp_handler
from gpustack.gateway import get_async_k8s_config
from gpustack.schemas.model_provider import (
    ModelProvider,
)

logger = logging.getLogger(__name__)


class ModelController:
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

        async for event in Model.subscribe(source="model_controller"):
            if event.type == EventType.HEARTBEAT:
                continue

            await self._reconcile(event)

    async def _ensure_model_mcp_bridge(
        self, session: AsyncSession, event_type: EventType, model: Model
    ):
        if self._disable_gateway:
            return
        model_instances = await ModelInstance.all_by_fields(
            session,
            fields={"model_id": model.id, "deleted_at": None},
        )
        worker_by_id = None
        worker_ids = {
            instance.worker_id for instance in model_instances if instance.worker_id
        }
        if worker_ids:
            workers = await Worker.all_by_fields(
                session,
                extra_conditions=[
                    Worker.id.in_(worker_ids),
                ],
            )
            worker_by_id = {worker.id: worker for worker in workers}

        lora_route_names = [
            lora_route_name_for(model.name, entry.lora_name)
            for entry in normalized_lora_list(model)
        ]
        await mcp_handler.ensure_model_mcp_bridge(
            event_type=event_type,
            model_id=model.id,
            model_instances=model_instances,
            networking_higress_api=self._higress_network_api,
            namespace=self._config.gateway_namespace,
            cluster_id=model.cluster_id,
            workers=worker_by_id,
            lora_route_names=lora_route_names,
        )

    async def _reconcile(self, event: Event):
        """
        Reconcile the model.
        """
        model: Model = event.data
        # Unhydrated means a delete whose row is gone (see Event), and every
        # callee below reads model fields. Leader-only controller, so nothing
        # else picks it up -- warn rather than drop it quietly.
        if not isinstance(model, Model):
            logger.warning(
                f"Model {resolve_event_id(event)} {event.type} not reconciled: "
                f"the event carries only an id and the row is gone"
            )
            return
        try:
            async with async_session() as session:
                await sync_replicas(session, model)
                await notify_model_route_target(
                    session=session, model=model, event=event
                )
                await sync_categories_and_meta(session, model, event)
                await self._ensure_model_mcp_bridge(session, event.type, model)
        except Exception as e:
            logger.error(f"Failed to reconcile model {model.name}: {e}")


class ModelInstanceController:
    def __init__(self, cfg: Config):
        self._config = cfg

        pass

    async def start(self):
        """
        Start the controller.
        """

        async for event in ModelInstance.subscribe(source="model_instance_controller"):
            if event.type == EventType.HEARTBEAT:
                continue

            await self._reconcile(event)

    async def _reconcile(self, event: Event):
        """
        Reconcile the model.
        """

        model_instance: ModelInstance = event.data
        # A cross-instance DELETE may carry only the id (see Event), so take
        # what the payload can give: the id always resolves, model_id only
        # when the event is hydrated.
        instance_id = resolve_event_id(event)
        model_id = event_field(model_instance, "model_id")
        try:
            async with async_session() as session:
                if event.type == EventType.DELETED and model_instance is not None:
                    # Cover cascade deletes that bypass ModelInstanceService.
                    #
                    # Known gap: with an id-only payload model_id is unknown,
                    # so the get_running_instances entry -- the one that feeds
                    # routing, and so the more important of the two -- is not
                    # dropped. Deletes through ModelInstanceService are fine
                    # (it invalidates itself, and delete_cache_by_key
                    # broadcasts), but the cascade path this block exists for
                    # is not, and the controller is leader-only so no other
                    # process retries it. Closing it needs the deleted row's
                    # model_id, which no event can carry.
                    instance_service = ModelInstanceService(session)
                    if model_id is not None:
                        await delete_cache_by_key(
                            instance_service.get_running_instances,
                            model_id,
                        )
                    if instance_id is not None:
                        await delete_cache_by_key(
                            instance_service.get_by_id, instance_id
                        )

                # Everything past this point needs the row's fields, and
                # model_id is the way in. Return explicitly rather than
                # letting one_by_id take a None primary key: it warns today
                # ("fully NULL primary key identity cannot load any object")
                # and SQLAlchemy reserves the right to raise on it later.
                # Nothing is lost -- the replica sync is level-triggered, and
                # the deleting instance ran it against the hydrated row.
                if model_id is None:
                    return
                model = await Model.one_by_id(session, model_id)
                if not model:
                    return
                model_deleting = model.deleted_at is not None

                if event.type == EventType.DELETED:
                    # trigger model replica sync, but only if model is not deleted
                    if not model_deleting:
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

                should_cleanup_lora_routes = event.type == EventType.DELETED or (
                    event.type == EventType.UPDATED
                    and "state" in (event.changed_fields or {})
                    and model_instance.state != ModelInstanceStateEnum.RUNNING
                )
                any_lora_route_deleted = False
                if should_cleanup_lora_routes:
                    any_lora_route_deleted = await cleanup_orphan_lora_routes(
                        session, model
                    )

                await model.refresh(session)
                replicas_updated = await sync_ready_replicas(session, model)
                if any_lora_route_deleted and not replicas_updated:
                    await session.commit()
                if any_lora_route_deleted:
                    await revoke_model_access_cache(session=session)
        except Exception as e:
            logger.error(
                "Failed to reconcile model instance "
                f"{event_field(model_instance, 'name', instance_id)}: {e}"
            )


class CacheServiceController:
    """
    Reconciles managed cache services onto their desired CacheServiceInstance
    set and aggregates instance states back onto the service row.

    The provider declaration's topology dictates the desired set: singleton
    services run exactly one instance on the user-picked worker; per_node
    services run one instance per non-deleted worker of the service's cluster
    (narrowed to workers matching the service's worker_selector labels when
    one is set), following workers as they join and leave. Instances whose
    worker left the desired set are hard-deleted; their workloads are
    cleaned up by the worker-side orphan cleaner. External services are
    untouched.
    """

    RESYNC_INTERVAL_SECONDS = 60

    def __init__(self, cfg: Config):
        self._config = cfg

    async def start(self):
        """
        Start the controller.
        """
        await asyncio.gather(
            self._watch_cache_services(),
            self._watch_workers(),
            self._watch_instances(),
            self._resync_loop(),
        )

    async def _watch_cache_services(self):
        async for event in CacheService.subscribe(source="cache_service_controller"):
            if event.type not in (EventType.CREATED, EventType.UPDATED):
                # Service deletion is a hard delete; the instances table's
                # ON DELETE CASCADE drops the rows with it.
                continue
            cache_service: CacheService = event.data
            if (
                cache_service is None
                or cache_service.mode != CacheServiceModeEnum.MANAGED
            ):
                continue
            await self._reconcile_service_by_id(cache_service.id)

    async def _watch_workers(self):
        # The resync loop covers startup, so the initial CREATED replay
        # would only duplicate it.
        async for event in Worker.subscribe(
            source="cache_service_controller_workers", replay_existing=False
        ):
            if event.type not in (
                EventType.CREATED,
                EventType.UPDATED,
                EventType.DELETED,
            ):
                continue
            # An id-only DELETE payload (see Event) cannot tell us which
            # cluster to reconcile, which reads the same as a worker with no
            # cluster: nothing to do here, and the resync loop heals it.
            cluster_id = event_field(event.data, "cluster_id")
            if cluster_id is None:
                continue
            # Of the update stream, only (un)soft-deletion and label
            # changes (which move a worker in or out of per_node services'
            # selector scopes) change the desired instance set.
            changed_fields = event.changed_fields or {}
            if event.type == EventType.UPDATED and not (
                "deleted_at" in changed_fields or "labels" in changed_fields
            ):
                continue
            await self._reconcile_cluster_services(cluster_id)

    async def _watch_instances(self):
        async for event in CacheServiceInstance.subscribe(
            source="cache_service_controller_instances", replay_existing=False
        ):
            if event.type not in (
                EventType.CREATED,
                EventType.UPDATED,
                EventType.DELETED,
            ):
                continue
            instance: CacheServiceInstance = event.data
            if instance is None or instance.cache_service_id is None:
                continue
            try:
                async with async_session() as session:
                    service = await CacheService.one_by_id(
                        session, instance.cache_service_id
                    )
                    if (
                        service is None
                        or service.deleted_at is not None
                        or service.mode != CacheServiceModeEnum.MANAGED
                    ):
                        continue
                    if event.type == EventType.DELETED:
                        # A deleted instance whose worker is still in the
                        # desired set gets a fresh PENDING replacement row
                        # right away instead of on the next resync pass, so
                        # instance deletion doubles as a relaunch from
                        # scratch.
                        await self._reconcile_service(session, service)
                        # Deletion also invalidates engines attached to the
                        # instance (e.g. a narrowed worker_selector).
                        await self._refresh_attached_snapshots(session, service)
                    else:
                        await self._sync_service_aggregate(session, service)
                        changed = set(event.changed_fields or {})
                        # Both directions matter: turning RUNNING attaches
                        # waiting engines, leaving RUNNING (or moving
                        # ports) invalidates attached ones.
                        if event.type == EventType.CREATED or (
                            {"state", "port"} & changed
                        ):
                            await self._refresh_attached_snapshots(session, service)
            except Exception as e:
                logger.error(
                    f"Failed to reconcile cache service "
                    f"{instance.cache_service_id} on instance event: {e}"
                )

    _PRE_START_STATES = frozenset(
        {
            ModelInstanceStateEnum.SCHEDULED,
            ModelInstanceStateEnum.INITIALIZING,
            ModelInstanceStateEnum.DOWNLOADING,
        }
    )
    """Model-instance states whose engine has not consumed the snapshot
    yet: the serve process re-reads the row at the STARTING transition,
    so a rewrite made in any of these states still reaches the engine.
    STARTING itself is excluded — the container launch races the write."""

    _CACHE_READY_HINT = "cache service is now ready; restart the instance to attach"

    async def _refresh_attached_snapshots(
        self, session: AsyncSession, service: CacheService
    ):
        """A cache instance changing (turning RUNNING, moving ports,
        stopping, being deleted) makes the cache_config snapshots
        resolved before it stale. The snapshot has no other owner, so
        this is the convergence point, in both directions:

        - degraded instances whose engine has not started yet get a
          fresh resolve written to the row, closing the
          create-service-then-model window (cache image pull and model
          download overlap);
        - RUNNING engines keep their snapshot — it records what the
          engine actually started with — but its endpoint_live view
          tracks the present: it flips off when the recorded endpoint is
          no longer what a fresh resolve yields (cache gone, or moved to
          another port), and back on upon recovery, so the "attached"
          indicators never report a cache that is not there. A degraded
          RUNNING engine gains a restart hint when a fresh resolve would
          now attach (never for takeover/incompatibility, which a
          restart would not fix)."""
        models = await Model.all_by_fields(
            session,
            fields={"cluster_id": service.cluster_id},
            extra_conditions=[Model.deleted_at.is_(None)],
        )
        attached = [
            model
            for model in models
            if model.extended_kv_cache
            and model.extended_kv_cache.is_shared()
            and model.extended_kv_cache.cache_service_id == service.id
        ]
        if not attached:
            return
        workers_by_id = {
            worker.id: worker
            for worker in await Worker.all_by_fields(
                session,
                fields={"cluster_id": service.cluster_id},
                extra_conditions=[Worker.deleted_at.is_(None)],
            )
        }
        for model in attached:
            instances = await ModelInstance.all_by_fields(
                session, {"model_id": model.id}
            )
            for mi in instances:
                if mi.worker_id is None:
                    # Pre-scheduling instances resolve at placement.
                    continue
                refreshable = mi.state in self._PRE_START_STATES
                running = mi.state == ModelInstanceStateEnum.RUNNING
                if not (refreshable or running):
                    continue
                snapshot = await resolve_instance_cache_config_safe(
                    session,
                    model,
                    workers_by_id.get(mi.worker_id),
                    spans_workers=mi.spans_workers,
                )
                if snapshot is None:
                    continue
                if refreshable:
                    if snapshot == mi.cache_config:
                        continue
                    mi.cache_config = snapshot
                    await ModelInstanceService(session).update(mi)
                    continue
                current = mi.cache_config
                if current is None:
                    continue
                if current.injected:
                    live = bool(
                        snapshot.injected
                        and snapshot.endpoint is not None
                        and current.endpoint is not None
                        and snapshot.endpoint.host == current.endpoint.host
                        and snapshot.endpoint.port == current.endpoint.port
                    )
                    unchanged = (
                        current.endpoint_live is False
                        if not live
                        else current.endpoint_live in (None, True)
                    )
                    if unchanged:
                        continue
                    mi.cache_config = current.model_copy(update={"endpoint_live": live})
                    await ModelInstanceService(session).update(mi)
                elif snapshot.injected and self._CACHE_READY_HINT not in (
                    current.reason or ""
                ):
                    reason = current.reason
                    mi.cache_config = current.model_copy(
                        update={
                            "reason": (
                                f"{reason}; {self._CACHE_READY_HINT}"
                                if reason
                                else self._CACHE_READY_HINT
                            )
                        }
                    )
                    await ModelInstanceService(session).update(mi)

    async def _resync_loop(self):
        """Periodic full pass over managed services, catching drift the
        event paths missed (e.g. events dropped on a full queue)."""
        while True:
            await asyncio.sleep(self.RESYNC_INTERVAL_SECONDS)
            try:
                async with async_session() as session:
                    services = await CacheService.all_by_fields(
                        session,
                        fields={"mode": CacheServiceModeEnum.MANAGED},
                        extra_conditions=[CacheService.deleted_at.is_(None)],
                    )
                for service in services:
                    await self._reconcile_service_by_id(service.id)
                    # The snapshot convergence path is event-driven; the
                    # resync exists to catch dropped events, so it must
                    # cover this path too (idempotent).
                    async with async_session() as session:
                        refreshed = await CacheService.one_by_id(session, service.id)
                        if refreshed is not None and refreshed.deleted_at is None:
                            await self._refresh_attached_snapshots(session, refreshed)
            except Exception as e:
                logger.error(f"Failed to resync cache services: {e}")

    async def _reconcile_cluster_services(self, cluster_id: int):
        try:
            async with async_session() as session:
                services = await CacheService.all_by_fields(
                    session,
                    fields={
                        "mode": CacheServiceModeEnum.MANAGED,
                        "cluster_id": cluster_id,
                    },
                    extra_conditions=[CacheService.deleted_at.is_(None)],
                )
            for service in services:
                await self._reconcile_service_by_id(service.id)
        except Exception as e:
            logger.error(
                f"Failed to reconcile cache services of cluster {cluster_id}: {e}"
            )

    async def _reconcile_service_by_id(self, cache_service_id: int):
        try:
            async with async_session() as session:
                service = await CacheService.one_by_id(session, cache_service_id)
                if (
                    service is None
                    or service.deleted_at is not None
                    or service.mode != CacheServiceModeEnum.MANAGED
                ):
                    return
                await self._reconcile_service(session, service)
        except Exception as e:
            logger.error(f"Failed to reconcile cache service {cache_service_id}: {e}")

    async def _reconcile_service(self, session: AsyncSession, service: CacheService):
        """Drive the service's instance rows to the desired worker set,
        then refresh the service-level aggregate state."""
        desired_worker_ids, error_message, reconcile = await self._desired_worker_ids(
            session, service
        )
        if error_message is not None and not reconcile:
            await self._set_service_state(
                session,
                service,
                state=CacheServiceStateEnum.ERROR,
                state_message=error_message,
                healthy=False,
            )
            return

        instances = await CacheServiceInstance.all_by_fields(
            session, {"cache_service_id": service.id}
        )
        existing_worker_ids = set()
        for instance in instances:
            if instance.worker_id not in desired_worker_ids:
                await instance.delete(session)
                logger.info(
                    f"Deleted instance {instance.id} of cache service "
                    f"{service.name}: worker {instance.worker_id} left "
                    "the desired set"
                )
            else:
                existing_worker_ids.add(instance.worker_id)

        for worker_id in sorted(desired_worker_ids - existing_worker_ids):
            # Same display-name convention as model instances: the parent's
            # name (as of instance creation; a later service rename does not
            # rename instances) plus a short random suffix.
            name_suffix = ''.join(
                random.choices(string.ascii_lowercase + string.digits, k=5)
            )
            await CacheServiceInstance.create(
                session,
                CacheServiceInstanceCreate(
                    name=f"{service.name}-{name_suffix}",
                    cache_service_id=service.id,
                    worker_id=worker_id,
                    cluster_id=service.cluster_id,
                    state=CacheServiceStateEnum.PENDING,
                    spec_digest=cache_service_spec_digest(service),
                ),
            )
            logger.info(
                f"Created instance of cache service {service.name} "
                f"on worker {worker_id}"
            )

        if error_message is not None:
            await self._set_service_state(
                session,
                service,
                state=CacheServiceStateEnum.ERROR,
                state_message=error_message,
                healthy=False,
            )
            return
        await self._sync_service_aggregate(session, service)

    async def _desired_worker_ids(
        self, session: AsyncSession, service: CacheService
    ) -> Tuple[Set[int], Optional[str], bool]:
        """The workers the service should have instances on, an error
        message when the desired set is unsatisfiable, and whether the
        instance rows should still be reconciled onto that set. A per_node
        selector matching nothing is an authoritative empty set — labels
        change only by explicit edits, so the instances follow (the
        selector can scale the service to zero) and the service parks in
        ERROR to say why. A singleton service whose picked worker is gone
        parks without touching its rows: the identity worker vanishing is
        a fault, not a narrowing."""
        provider = get_cache_provider(service.provider_name)
        topology = provider.topology if provider else "singleton"
        if topology == "per_node":
            workers = await Worker.all_by_fields(
                session,
                fields={"cluster_id": service.cluster_id},
                extra_conditions=[Worker.deleted_at.is_(None)],
            )
            if not workers:
                return (
                    set(),
                    "Cluster has no active workers to run cache instances.",
                    True,
                )
            selector = service.worker_selector
            if selector:
                workers = [
                    worker
                    for worker in workers
                    if label_matching(selector, worker.labels or {})
                ]
                if not workers:
                    return (
                        set(),
                        f"No workers match the worker selector: {selector}.",
                        True,
                    )
            return {worker.id for worker in workers}, None, True

        if not service.worker_id:
            return set(), "No worker assigned.", False
        worker = await Worker.one_by_id(session, service.worker_id)
        if (
            worker is None
            or worker.deleted_at is not None
            or worker.cluster_id != service.cluster_id
        ):
            return set(), "Assigned worker no longer exists.", False
        return {service.worker_id}, None, True

    async def _sync_service_aggregate(
        self, session: AsyncSession, service: CacheService
    ):
        """Fold the instances' states into the service row: all RUNNING →
        RUNNING/healthy; some RUNNING → RUNNING/unhealthy with an N/M
        breakdown; none RUNNING but a cache server already launching →
        STARTING; none launched yet → PENDING; otherwise ERROR."""
        instances = await CacheServiceInstance.all_by_fields(
            session, {"cache_service_id": service.id}
        )
        total = len(instances)
        running = sum(
            1
            for instance in instances
            if instance.state == CacheServiceStateEnum.RUNNING
        )
        starting = any(
            instance.state == CacheServiceStateEnum.STARTING for instance in instances
        )
        pending = any(
            instance.state == CacheServiceStateEnum.PENDING for instance in instances
        )

        if total and running == total:
            state, healthy, message = CacheServiceStateEnum.RUNNING, True, None
        elif running:
            state, healthy, message = (
                CacheServiceStateEnum.RUNNING,
                False,
                f"{running}/{total} instances running",
            )
        elif starting:
            state, healthy, message = CacheServiceStateEnum.STARTING, None, None
        elif pending:
            state, healthy, message = CacheServiceStateEnum.PENDING, None, None
        else:
            state, healthy, message = (
                CacheServiceStateEnum.ERROR,
                False,
                f"0/{total} instances running" if total else "no instances running",
            )

        # A spec edit does not touch running containers (the controller
        # reconciles the instance set, not the spec; recovery is
        # delete-to-recreate) — say so instead of silently returning the
        # new spec from the API while containers run the old one.
        # Instances predating the digest (None) are never flagged.
        current_digest = cache_service_spec_digest(service)
        if any(
            instance.spec_digest and instance.spec_digest != current_digest
            for instance in instances
        ):
            drift_message = (
                "configuration changed after instances started; delete "
                "instances to recreate them with the current configuration"
            )
            message = f"{message}; {drift_message}" if message else drift_message

        await self._set_service_state(
            session, service, state=state, state_message=message, healthy=healthy
        )

    async def _set_service_state(
        self,
        session: AsyncSession,
        service: CacheService,
        state: CacheServiceStateEnum,
        state_message: Optional[str],
        healthy: Optional[bool],
    ):
        """Write the aggregate only on change, so steady state produces no
        UPDATE events for watchers to churn on."""
        if (
            service.state == state
            and service.state_message == state_message
            and service.healthy == healthy
        ):
            return
        await service.update(
            session,
            {
                "state": state,
                "state_message": state_message,
                "healthy": healthy,
            },
        )


async def sync_replicas(session: AsyncSession, model: Model):
    """
    Synchronize the replicas.
    """

    # Re-fetch model from database to ensure we have latest state
    # (event data may be from a different session or stale)
    fresh_model = await Model.one_by_id(session, model.id)
    if not fresh_model or fresh_model.deleted_at is not None:
        return
    model = fresh_model

    instances = await ModelInstance.all_by_field(session, "model_id", model.id)
    if len(instances) < model.replicas:
        for _ in range(model.replicas - len(instances)):
            name_prefix = ''.join(
                random.choices(string.ascii_lowercase + string.digits, k=5)
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
                # Inherit the parent Model's tenant binding — the schema
                # default of platform_principal_id() would otherwise
                # land instances of a non-Default-Org Model in Default.
                owner_principal_id=model.owner_principal_id,
                draft_model_source=await get_draft_model_source(session, model),
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
            session,
            fields={
                "kind": PrincipalType.USER,
                "deleted_at": None,
                "is_admin": False,
            },
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
                fields={
                    "kind": PrincipalType.USER,
                    "deleted_at": None,
                    "is_admin": False,
                },
                extra_conditions=[
                    User.id.in_(
                        select(ModelRoutePrincipalLink.principal_id).where(
                            ModelRoutePrincipalLink.route_id == model.id
                        )
                    )
                ],
            )
            current_user_ids = {user.id for user in users}
            to_update_model_user_ids = current_user_ids - to_create_model_user_ids
    if event.type == EventType.CREATED:
        users = await User.all_by_fields(
            session,
            fields={
                "kind": PrincipalType.USER,
                "deleted_at": None,
                "is_admin": False,
            },
            extra_conditions=[
                User.id.in_(
                    select(ModelRoutePrincipalLink.principal_id).where(
                        ModelRoutePrincipalLink.route_id == model.id
                    )
                )
            ],
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
                # Match the view's pid layout (``route_id:user_id:via``).
                # The publisher doesn't know the granting chain, so the
                # via suffix is empty — same shape as the PUBLIC/AUTHED
                # branch of ``non_admin_user_models``.
                pid=f"{model_id}:{user_id}:",
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


def _instance_model_files_complete(
    instance: ModelInstance, model: Optional[Model]
) -> bool:
    """Check if all expected model files already exist and none need retry."""
    worker_ids = _get_worker_ids_for_file_download(instance)
    if not worker_ids:
        return False

    existing = instance.model_files or []
    if any(f.state == ModelFileStateEnum.ERROR for f in existing):
        return False

    # Primary model files: each worker should have a non-LoRA file
    primary_worker_ids = {f.worker_id for f in existing if not f.is_lora}
    if not all(wid in primary_worker_ids for wid in worker_ids):
        return False

    # LoRA files: each LoRA entry × each worker should have a file
    expected_lora_count = len(normalized_lora_list(model)) if model else 0
    if expected_lora_count > 0:
        lora_files = [f for f in existing if f.is_lora]
        if len(lora_files) < expected_lora_count * len(worker_ids):
            return False

    # Draft model files
    if instance.draft_model_source:
        draft_files = instance.draft_model_files or []
        if any(f.state == ModelFileStateEnum.ERROR for f in draft_files):
            return False
        draft_worker_ids = {f.worker_id for f in draft_files}
        if not all(wid in draft_worker_ids for wid in worker_ids):
            return False

    return True


async def _link_instance_primary_model_files(
    session: AsyncSession, instance_id: int, files: List[ModelFile]
):
    """Insert missing instance↔model_file links; caller's session will flush/commit."""
    for f in files:
        if f.id is None:
            continue
        stmt = select(ModelInstanceModelFileLink).where(
            ModelInstanceModelFileLink.model_instance_id == instance_id,
            ModelInstanceModelFileLink.model_file_id == f.id,
        )
        if (await session.exec(stmt)).first() is None:
            session.add(
                ModelInstanceModelFileLink(
                    model_instance_id=instance_id, model_file_id=f.id
                )
            )


async def _link_instance_draft_model_files(
    session: AsyncSession, instance_id: int, files: List[ModelFile]
):
    """Same as primary links but for draft-model file associations."""
    for f in files:
        if f.id is None:
            continue
        stmt = select(ModelInstanceDraftModelFileLink).where(
            ModelInstanceDraftModelFileLink.model_instance_id == instance_id,
            ModelInstanceDraftModelFileLink.model_file_id == f.id,
        )
        if (await session.exec(stmt)).first() is None:
            session.add(
                ModelInstanceDraftModelFileLink(
                    model_instance_id=instance_id, model_file_id=f.id
                )
            )


def _is_primary_instance_model_file(
    file: ModelFile, instance: ModelInstance, is_draft_model: bool
) -> bool:
    if is_draft_model:
        return False
    if file.is_lora:
        return False
    return True


async def get_or_create_lora_model_files_for_instance(
    session: AsyncSession, instance: ModelInstance, model: Model
) -> List[ModelFile]:
    """Ensure ModelFile rows exist for model.lora_list on the instance's workers (same session as caller)."""
    entries = normalized_lora_list(model)
    if not entries:
        return []
    worker_ids = _get_worker_ids_for_file_download(instance)
    base_desc = model_base_descriptor(model)
    worker_scopes = await _get_worker_tenant_scopes(session, worker_ids)
    out: List[ModelFile] = []
    seen_ids: Set[int] = set()

    for entry in entries:
        try:
            lora_src = lora_entry_to_model_source(entry)
        except ValueError as e:
            logger.warning(
                "Skip invalid LoRA entry %r for instance %s; ModelFile will not be created: %s",
                entry.lora_name,
                instance.name,
                e,
            )
            continue
        # Query once per entry, reuse across workers
        existing_list = await ModelFileService(session).get_by_source_index(
            lora_src.model_source_index
        )
        existing_list = existing_list or []
        for worker_id in worker_ids:
            hit = next((f for f in existing_list if f.worker_id == worker_id), None)
            if hit:
                if not hit.is_lora:
                    hit.is_lora = True
                if hit.base_model != base_desc:
                    hit.base_model = base_desc
                    await hit.update(session, auto_commit=False)
                if hit.id is not None and hit.id not in seen_ids:
                    out.append(hit)
                    seen_ids.add(hit.id)
            else:
                cluster_id, owner_principal_id = worker_scopes.get(
                    worker_id, (None, None)
                )
                nf = ModelFile(
                    source=lora_src.source,
                    huggingface_repo_id=lora_src.huggingface_repo_id,
                    huggingface_filename=lora_src.huggingface_filename,
                    model_scope_model_id=lora_src.model_scope_model_id,
                    model_scope_file_path=lora_src.model_scope_file_path,
                    local_path=lora_src.local_path,
                    is_lora=True,
                    base_model=base_desc,
                    state=ModelFileStateEnum.DOWNLOADING,
                    worker_id=worker_id,
                    source_index=lora_src.model_source_index,
                    cluster_id=cluster_id,
                    owner_principal_id=owner_principal_id,
                )
                created = await ModelFile.create(session, nf, auto_commit=False)
                mf = created or nf
                if mf.id is not None and mf.id not in seen_ids:
                    out.append(mf)
                    seen_ids.add(mf.id)
    return out


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
            selectinload(ModelInstance.draft_model_files),
        ],
    )
    if not instance:
        return

    model = await Model.one_by_id(session, instance.model_id)

    # Early-return: skip expensive get_or_create queries when all files are ready
    if _instance_model_files_complete(instance, model):
        all_files = list(instance.model_files) + list(instance.draft_model_files)
        await sync_instance_files_state(session, instance, all_files)
        return

    retry_model_files = []
    model_files = await get_or_create_model_files_for_instance(session, instance)
    draft_model_files = []
    if instance.draft_model_source:
        draft_model_files = await get_or_create_model_files_for_instance(
            session, instance, is_draft_model=True
        )
    lora_model_files: List[ModelFile] = []
    if model:
        lora_model_files = await get_or_create_lora_model_files_for_instance(
            session, instance, model
        )

    for model_file in model_files + draft_model_files + lora_model_files:
        if model_file.state == ModelFileStateEnum.ERROR:
            # Retry the download
            retry_model_files.append(model_file.readable_source)

            model_file.state = ModelFileStateEnum.DOWNLOADING
            model_file.download_progress = 0
            model_file.state_message = ""
            await model_file.update(session, auto_commit=False)

    await _link_instance_primary_model_files(
        session, instance.id, model_files + lora_model_files
    )
    await _link_instance_draft_model_files(session, instance.id, draft_model_files)
    # Commit file creation, retry resets, and instance <--> file links in one
    # transaction.  ModelFile events are published by the after_commit hook,
    # so ModelFileController._reconcile will only see these files after the
    # links already exist — preventing a race where reconcile queries
    # file.instances before the links are committed.
    await session.commit()

    if retry_model_files:
        logger.info(
            f"Retrying download for model files {retry_model_files} for model instance {instance.name}"
        )

    instance = await ModelInstance.one_by_id(session, instance.id)
    await sync_instance_files_state(
        session,
        instance,
        model_files + draft_model_files + lora_model_files,
    )


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
    worker_scopes = await _get_worker_tenant_scopes(session, missing_worker_ids)
    # Create model files for the missing worker IDs.
    for worker_id in missing_worker_ids:
        cluster_id, owner_principal_id = worker_scopes.get(worker_id, (None, None))
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
            cluster_id=cluster_id,
            owner_principal_id=owner_principal_id,
        )
        await ModelFile.create(session, model_file, auto_commit=False)
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


async def find_scale_down_candidates(
    instances: List[ModelInstance],
    model: Model,
    *,
    status_max_score: Optional[float] = None,
    offload_max_score: Optional[float] = None,
    placement_max_score: Optional[float] = None,
    total_max_score: Optional[float] = None,
) -> List[ModelInstanceScore]:
    try:
        if status_max_score is None:
            status_max_score = envs.SCHEDULER_SCALE_DOWN_STATUS_MAX_SCORE
        if offload_max_score is None:
            offload_max_score = envs.SCHEDULER_SCALE_DOWN_OFFLOAD_MAX_SCORE
        if placement_max_score is None:
            placement_max_score = envs.SCHEDULER_SCALE_DOWN_PLACEMENT_MAX_SCORE

        chain = ModelInstanceScoreChain(
            scorers=[
                StatusScorer(model, max_score=status_max_score),
                OffloadLayerScorer(model, max_score=offload_max_score),
                PlacementScorer(
                    model,
                    instances,
                    scale_type=ScaleTypeEnum.SCALE_DOWN,
                    max_score=placement_max_score,
                ),
            ],
            total_max_score=total_max_score,
        )
        final_candidates = await chain.score(instances)
        final_candidates = sorted(
            final_candidates, key=lambda x: x.score, reverse=False
        )
        return final_candidates
    except Exception as e:
        state_message = (
            f"Failed to find scale down candidates for model {model.name}: {e}"
        )
        logger.error(state_message)
        return []


async def sync_ready_replicas(session: AsyncSession, model: Model) -> bool:
    """
    Synchronize the ready replicas.

    Returns True if the model row was updated (and the session was committed).
    """

    if model.deleted_at is not None:
        return False

    instances = await ModelInstance.all_by_field(session, "model_id", model.id)

    ready_replicas: int = 0
    for _, instance in enumerate(instances):
        if instance.state == ModelInstanceStateEnum.RUNNING:
            ready_replicas += 1

    if model.ready_replicas != ready_replicas:
        model.ready_replicas = ready_replicas
        await ModelService(session).update(model)
        return True
    return False


async def get_cluster_registry(
    session: AsyncSession, cluster_id: int
) -> Optional[McpBridgeRegistry]:
    # Resolve the cluster's SYSTEM principal via the inverse FK
    # (``Cluster.system_principal_id``) — that link replaces the old
    # ``User.cluster_id`` lookup after the FK direction was inverted.
    cluster = await Cluster.one_by_id(session, cluster_id)
    if cluster is None or cluster.system_principal_id is None:
        return None
    cluster_principal = await Principal.one_by_id(session, cluster.system_principal_id)
    if cluster_principal is None or is_default_cluster_principal(cluster_principal):
        return None
    cluster_registry = mcp_handler.cluster_registry(cluster)
    if cluster_registry is None:
        return None
    return cluster_registry


async def sync_model_route_mapper(
    cfg: Config,
    extensions_api: ExtensionsHigressIoV1Api,
    ingress_name: str,
    route_name: str,
    destinations: mcp_handler.DestinationTupleList,
    fallback_destinations: mcp_handler.DestinationTupleList,
):
    """
    Synchronize the model route mapper.
    """
    ingress_prefix = f"{cfg.get_namespace()}/"
    if cfg.get_namespace() == cfg.gateway_namespace:
        ingress_prefix = ""
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

    expected_rules = mcp_handler.get_expected_match_list(
        route_name=route_name,
        ingress_prefix=ingress_prefix,
        ingress_name=ingress_name,
        model_name_to_registries=model_name_to_registries,
        fallback_model_name_to_registries=fallback_model_name_to_registries,
    )

    def spec_diff(current_spec: Optional[WasmPluginSpec]) -> WasmPluginSpec:
        # the current spec must exist. If not, it means the plugin has been deleted manually,
        # we should not recreate it until next update event to avoid potential misconfiguration.
        if current_spec is None:
            return current_spec
        to_keep_rules: List[WasmPluginMatchRule] = []
        full_ingress_name = f"{ingress_prefix}{ingress_name}"

        for rule in current_spec.matchRules or []:
            if full_ingress_name not in rule.ingress:
                to_keep_rules.append(rule)
        to_keep_rules.extend(expected_rules)
        to_keep_rules.sort(key=lambda r: r.ingress[0] if r.ingress else "")
        current_spec.matchRules = to_keep_rules
        return current_spec

    await mcp_handler.ensure_wasm_plugin(
        api=extensions_api,
        name=mcp_handler.gpustack_model_mapper_name,
        namespace=cfg.gateway_namespace,
        spec_diff=spec_diff,
    )


async def ensure_route_generic_proxy_router_config(
    cfg: Config,
    model_route: ModelRoute,
    effective_name: str,
    extensions_api: ExtensionsHigressIoV1Api,
    generic_proxy_enabled: bool,
):
    """
    Reconcile the single aliasNameMapping entry that maps /model/proxy/<route_id>/...
    to this route's effective model name. When ``generic_proxy_enabled`` is False
    (generic proxy disabled or route deleted), the entry is removed and other
    routes are untouched.

    ``effective_name`` is the fully-qualified model name including the
    Org name prefix (e.g. ``org1/qwen3-0.6b``) for non-platform Orgs;
    platform Org keeps the unprefixed ``model_route.name``.
    """
    route_name = effective_name if generic_proxy_enabled else None
    await mcp_handler.ensure_wasm_plugin(
        api=extensions_api,
        name=mcp_handler.gpustack_generic_proxy_router_name,
        namespace=cfg.gateway_namespace,
        spec_diff=partial(
            mcp_handler.generic_proxy_router_diff_spec,
            route_id=model_route.id,
            route_name=route_name,
        ),
    )


async def ensure_route_ai_proxy_config(
    cfg: Config,
    model_route_id: int,
    extensions_api: ExtensionsHigressIoV1Api,
    model_groups: List[mcp_handler.ModelAIProxyGroup],
):
    """Reconcile this route's slice of the ai-proxy CR, grouped by deployment.

    Provider entries are keyed by Model, so a Model targeted by several routes
    has exactly one of them and route-level changes never rewrite it. Ownership
    of the *rules* is therefore expressed by ingress rather than by provider id
    — see ``compare_and_append_proxy_match_rules``.

    External model providers are not handled here: ``ModelProviderController``
    owns their ``provider-<id>`` entries, whose rules match on service only.
    """
    prefixed_ingress_name, prefixed_fallback_ingress_name = (
        mcp_handler.route_ingress_names_for_plugins(
            model_route_id=model_route_id,
            resource_namespace=cfg.get_namespace(),
            gateway_namespace=cfg.gateway_namespace,
        )
    )
    expected_providers, expected_match_rules = mcp_handler.model_ai_proxy_plugin_spec(
        groups=model_groups,
        main_ingress=prefixed_ingress_name,
        fallback_ingress=prefixed_fallback_ingress_name,
    )

    await mcp_handler.ensure_wasm_plugin(
        api=extensions_api,
        name=mcp_handler.gpustack_ai_proxy_name,
        namespace=cfg.gateway_namespace,
        spec_diff=partial(
            mcp_handler.ai_proxy_diff_spec,
            expected_providers=expected_providers,
            expected_match_rules=expected_match_rules,
            owned_ingresses={
                prefixed_ingress_name,
                prefixed_fallback_ingress_name,
            },
        ),
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
    model_route_from_db = await ModelRoute.one_by_id(
        session,
        model_route.id,
        options=[selectinload(ModelRoute.route_targets)],
    )
    targets: List[ModelRouteTarget] = (
        getattr(model_route_from_db, "route_targets", []) if model_route_from_db else []
    )
    has_fallback_target = any(
        target
        for target in targets
        if target.fallback_status_codes and len(target.fallback_status_codes) > 0
    )
    destinations = []
    fallback_destinations = []
    model_groups: List[mcp_handler.ModelAIProxyGroup] = []
    if not model_route_from_db:
        event_type = EventType.DELETED
    if event.type != EventType.DELETED:
        destinations, fallback_destinations, model_groups = (
            await calculate_destinations(session, model_route)
        )
    # Effective model name = `<owner-name>/<route.name>` for non-platform
    # Orgs (so two Orgs can use the same `route.name` without colliding
    # in Higress's AI proxy match rules), unprefixed for the platform Org
    # (backward compatible for existing clients).
    route_owner = await Principal.one_by_id(session, model_route.owner_principal_id)
    effective_name = effective_route_name(
        model_route.name,
        getattr(route_owner, "name", None),
        getattr(route_owner, "id", None) == platform_principal_id(),
    )
    ingress_name = mcp_handler.model_route_ingress_name(model_route.id)
    await sync_model_route_mapper(
        cfg=cfg,
        extensions_api=extensions_api,
        ingress_name=ingress_name,
        route_name=effective_name,
        destinations=destinations,
        fallback_destinations=fallback_destinations,
    )
    # FIXME: Copy the fallback destination to the main ingress for now to make sure the fallback
    # route is always hit when fallback is configured, even if the main route has no valid
    # destination. This is to avoid potential misconfiguration that causes the main route to
    # have no destination and the fallback route is not hit at all.
    await mcp_handler.ensure_model_ingress(
        ingress_class_name=cfg.gateway_ingress_class,
        event_type=event_type,
        ingress_name=ingress_name,
        route_name=effective_name,
        namespace=cfg.get_namespace(),
        destinations=destinations if len(destinations) > 0 else fallback_destinations,
        networking_api=networking_api,
        included_generic_route=False,
        included_proxy_route=model_route.generic_proxy,
    )
    fallback_event_type = event_type
    if not has_fallback_target:
        fallback_event_type = EventType.DELETED
    # Fallback ingress
    await mcp_handler.ensure_model_ingress(
        ingress_class_name=cfg.gateway_ingress_class,
        event_type=fallback_event_type,
        ingress_name=mcp_handler.fallback_ingress_name(ingress_name),
        route_name=effective_name,
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
    # Generic-proxy router: inject x-higress-llm-model when /model/proxy/<id>/
    # is hit, so the existing main ingress header matcher + fallback chain apply.
    await ensure_route_generic_proxy_router_config(
        cfg=cfg,
        model_route=model_route,
        effective_name=effective_name,
        extensions_api=extensions_api,
        generic_proxy_enabled=(
            event_type != EventType.DELETED and bool(model_route.generic_proxy)
        ),
    )
    # ensure ai proxy config
    await ensure_route_ai_proxy_config(
        cfg=cfg,
        model_route_id=model_route.id,
        extensions_api=extensions_api,
        model_groups=model_groups,
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
) -> Tuple[
    mcp_handler.DestinationTupleList,
    mcp_handler.DestinationTupleList,
    List[mcp_handler.ModelAIProxyGroup],
]:
    """
    Return the percentage tuple for each registry with model name and the
    fallback registry, plus the route's self-hosted destinations grouped by
    deployment (Model) for the ai-proxy provider config. External provider
    targets are excluded from the groups — they carry their own credentials and
    their own provider entry.
    """
    weight_to_count: List[Tuple[int, int, mcp_handler.DestinationTupleList]] = []
    fallback_weight_to_count: List[
        Tuple[int, int, mcp_handler.DestinationTupleList]
    ] = []
    model_groups: Dict[int, mcp_handler.ModelAIProxyGroup] = {}
    # Routes commonly fan out to several deployments of one cluster; cache the
    # token per cluster so the group build stays at one query per cluster.
    cluster_api_tokens: Dict[Optional[int], List[str]] = {}
    targets = await ModelRouteTarget.all_by_field(session, "route_id", model_route.id)
    for target in targets:
        if target.state != TargetStateEnum.ACTIVE:
            continue
        to_extend: mcp_handler.DestinationTupleList = []
        model: Optional[Model] = None
        if target.model_id is not None:
            model = await Model.one_by_id(session, target.model_id)
            if model is None:
                continue
            to_extend = await calculate_model_destinations(
                session, model, target.overridden_model_name
            )
        elif target.provider_id is not None:
            to_extend = await provider_destinations(
                session=session,
                provider_id=target.provider_id,
                provider_model_name=target.overridden_model_name,
            )
        if to_extend is None or len(to_extend) == 0:
            # no valid destination found
            continue
        is_fallback_target = (
            target.fallback_status_codes is not None
            and len(target.fallback_status_codes) > 0
        )
        if model is not None:
            await accumulate_model_ai_proxy_group(
                session=session,
                model_groups=model_groups,
                cluster_api_tokens=cluster_api_tokens,
                model=model,
                destinations=to_extend,
                is_fallback_target=is_fallback_target,
            )
        count = sum([count for count, _, _ in to_extend])
        weight_to_count.append((target.weight, count, to_extend))
        if is_fallback_target:
            fallback_weight_to_count.append((target.weight, count, to_extend))
    if len(weight_to_count) == 0:
        return [], [], []

    flatten_registry_list = flatten_destinations(weight_to_count)
    fallback_registry_list = []
    if len(fallback_weight_to_count) > 0:
        # fallback might have 0 weight, so set max_weight to 1
        fallback_registry_list = flatten_destinations(
            fallback_weight_to_count, max_weight=1
        )

    return flatten_registry_list, fallback_registry_list, list(model_groups.values())


async def accumulate_model_ai_proxy_group(
    session: AsyncSession,
    model_groups: Dict[int, mcp_handler.ModelAIProxyGroup],
    cluster_api_tokens: Dict[Optional[int], List[str]],
    model: Model,
    destinations: mcp_handler.DestinationTupleList,
    is_fallback_target: bool,
):
    """Fold one route target's upstream services into its deployment's group.

    A Model can appear under several targets of the same route (e.g. one target
    per LoRA, each aliasing the same instances), so services accumulate instead
    of overwriting. The credential is the deployment's cluster registration
    token: it is the only existing credential every worker of that cluster
    accepts, and ai-proxy holds one ``apiTokens`` list per provider while the
    backend worker is picked independently by Envoy.

    ``cluster_api_tokens`` is the caller's per-reconcile cache, so a route with
    many deployments in one cluster resolves the token once.
    """
    group = model_groups.get(model.id)
    if group is None:
        if model.cluster_id not in cluster_api_tokens:
            cluster_api_tokens[model.cluster_id] = await cluster_registration_tokens(
                session, model.cluster_id
            )
        group = mcp_handler.ModelAIProxyGroup(
            model_id=model.id,
            api_tokens=cluster_api_tokens[model.cluster_id],
            native_anthropic_api=model.native_anthropic_api,
        )
        model_groups[model.id] = group
    service_names = {registry.get_service_name() for _, _, registry in destinations}
    group.service_names.update(service_names)
    if is_fallback_target:
        group.fallback_service_names.update(service_names)


async def cluster_registration_tokens(
    session: AsyncSession, cluster_id: Optional[int]
) -> List[str]:
    """The cluster's registration token as an ai-proxy ``apiTokens`` list.

    Empty when the cluster or its token is missing, which leaves ai-proxy on its
    pre-existing behavior of reading the inbound ``Authorization`` header.
    """
    if cluster_id is None:
        return []
    cluster = await Cluster.one_by_id(session, cluster_id)
    if cluster is None or not cluster.registration_token:
        return []
    return [cluster.registration_token]


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
    overridden_model_name: Optional[str] = None,
) -> mcp_handler.DestinationTupleList:
    """Build destinations for a local-model target. LoRA child routes pass
    ``overridden_model_name=<base>:<lora>`` so the gateway's modelMapping
    becomes a self-map (skipped at sync_model_route_mapper), letting the
    LoRA module name reach vLLM intact.
    """
    downstream_model_name = overridden_model_name or model.name
    # LoRA targets share the base model's instances; route them to a per-LoRA
    # aliased service (same address, distinct name) registered in ensure_model_mcp_bridge
    # so the gateway can weight and rewrite per LoRA instead of collapsing onto one.
    registry_name_suffix = (
        mcp_handler.lora_registry_name_suffix(overridden_model_name)
        if overridden_model_name is not None and ":" in overridden_model_name
        else None
    )
    cluster_registry = await get_cluster_registry(session, model.cluster_id)
    if cluster_registry is not None:
        return [(1, downstream_model_name, cluster_registry)]

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
    return mcp_handler.model_instances_registry_list(
        instances,
        workers,
        downstream_model_name=downstream_model_name,
        registry_name_suffix=registry_name_suffix,
    )


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
            # All three handlers below branch on worker fields (state,
            # cluster_id, name), none of which survive an unhydrated payload
            # (see Event). Guarding once keeps one the first handler cannot
            # read from costing the other two as well.
            if not isinstance(event.data, Worker):
                logger.warning(
                    f"Worker {resolve_event_id(event)} {event.type} not "
                    f"reconciled: the event carries only an id and the row is "
                    f"gone"
                )
                continue
            try:
                await self._reconcile(event)
                await self._provisioning._reconcile(event)
                await self._notify_relatives(event)
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
            all_instances = await ModelInstance.all_by_field(
                session, "cluster_id", worker.cluster_id
            )
            if not all_instances:
                return
            matched_instances = []
            for instance in all_instances:
                match = get_model_instance_worker_match(
                    instance,
                    worker_name=worker.name,
                    worker_id=worker.id,
                )
                if match.matched:
                    matched_instances.append((instance, match))
            if not matched_instances:
                return

            if event.type == EventType.DELETED:
                instance_names = await ModelInstanceService(session).batch_delete(
                    [instance for instance, _ in matched_instances]
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
                await self.update_impacted_instance_states_to_unreachable(
                    session,
                    matched_instances,
                    worker.name,
                )
                return

    async def update_impacted_instance_states_to_unreachable(
        self,
        session,
        matched_instances,
        worker_name,
    ):
        instance_names = set()
        subordinate_worker_names = set()
        for instance, match in matched_instances:
            patch = {}
            distributed_servers_changed = False
            if (
                match.is_main_worker
                and instance.state == ModelInstanceStateEnum.RUNNING
            ):
                patch["state"] = ModelInstanceStateEnum.UNREACHABLE
                patch["state_message"] = "Worker is unreachable from the server"
                instance_names.add(instance.name)

            for index in match.subordinate_worker_indexes:
                subordinate_worker = instance.distributed_servers.subordinate_workers[
                    index
                ]
                if subordinate_worker.state == ModelInstanceStateEnum.UNREACHABLE:
                    continue
                subordinate_worker.state = ModelInstanceStateEnum.UNREACHABLE
                subordinate_worker.state_message = (
                    "Worker is unreachable from the server"
                )
                subordinate_worker_names.add(
                    f"{instance.name}:{subordinate_worker.worker_name}"
                )
                distributed_servers_changed = True

            if distributed_servers_changed:
                patch["distributed_servers"] = instance.distributed_servers
                flag_modified(instance, "distributed_servers")

            if patch:
                await ModelInstanceService(session).update(instance, patch)
        if instance_names:
            logger.info(
                f"Marked instance {', '.join(instance_names)} unreachable "
                f"since worker {worker_name} is unreachable from the server"
            )
        if subordinate_worker_names:
            logger.info(
                f"Marked subordinate workers {', '.join(subordinate_worker_names)} unreachable "
                f"since worker {worker_name} is unreachable from the server"
            )

    async def _notify_relatives(self, event: Event):
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
        should_notify_parents = (
            state_changed is not None
            or proxy_mode_changed is not None
            or event.type == EventType.DELETED
        )
        proxy_address_changed: Optional[Tuple[Any, Any]] = (changed_fields or {}).get(
            "proxy_address", None
        )
        should_notify_children = (
            proxy_address_changed is not None or proxy_mode_changed is not None
        )

        if not should_notify_parents and not should_notify_children:
            return
        async with async_session() as session:
            if should_notify_parents and worker.worker_pool_id is not None:
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
            if should_notify_parents and worker.cluster_id is not None:
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

            if should_notify_children:
                instances = await ModelInstance.all_by_fields(
                    session,
                    fields={"worker_id": worker.id},
                    options=[selectinload(ModelInstance.model)],
                )
                notified_model = set()
                for instance in instances:
                    if instance.model_id in notified_model:
                        continue
                    notified_model.add(instance.model_id)
                    copied_model = Model(**instance.model.model_dump())
                    await event_bus.publish(
                        copied_model.__class__.__name__.lower(),
                        Event(
                            type=EventType.UPDATED,
                            data=copied_model,
                        ),
                    )


class InferenceBackendController:
    """
    Leader-only controller that seeds the built-in inference engines and the
    BUILTIN community-backend source, then materializes the Platform-NULL
    community backends from all enabled InferenceBackendSource rows.

    On start it seeds the packaged community-inference-backends.yaml as the
    BUILTIN source, then (like RunnerSourceController) subscribes and
    smart-merges on any source change; the initial replay of existing sources
    drives the first reconcile.
    """

    async def start(self):
        async with async_session() as session:
            await self._init_built_in_backends(session)
        await self._seed_builtin_source()
        async for event in InferenceBackendSource.subscribe(
            source="inference_backend_source_controller"
        ):
            if event.type in (
                EventType.CREATED,
                EventType.UPDATED,
                EventType.DELETED,
            ):
                await self._reconcile()

    async def _init_built_in_backends(self, session: AsyncSession):
        """Initialize built-in backends in the database."""
        for built_in_backend in get_built_in_backend():
            if built_in_backend.backend_name == BackendEnum.CUSTOM.value:
                continue

            # Built-in backends always seed as Platform (owner_principal_id IS NULL).
            # Per-Org overrides live in additional rows created by Org owners /
            # managers; those are managed via the inference_backend routes.
            backend = await InferenceBackend.one_by_fields(
                session,
                {
                    "backend_name": built_in_backend.backend_name,
                    "owner_principal_id": None,
                },
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

    async def _reconcile(self):
        try:
            async with async_session() as session:
                await gather_and_merge(
                    session, InferenceBackendSource, reconcile_backend
                )
        except Exception as e:
            logger.error(f"Failed to reconcile community backends: {e}")

    async def _seed_builtin_source(self):
        """Upsert the BUILTIN InferenceBackendSource from the packaged
        community-inference-backends.yaml.

        The content is refreshed to the newest packaged catalog on every start
        (so a GPUStack upgrade ships it), while the user's ``enabled`` toggle on
        the built-in source is left untouched.
        """
        try:
            yaml_file = files("gpustack.assets").joinpath(
                "community-inference-backends.yaml"
            )
            if not yaml_file.is_file():
                logger.debug(
                    "community-inference-backends.yaml not found, skipping seed"
                )
                return
            raw = await asyncio.to_thread(yaml_file.read_text)
            content = normalize_backend_yaml(raw)
        except Exception as e:
            logger.error(f"Failed to seed built-in inference backend source: {e}")
            return

        content_hash = sha256_of(content)
        async with async_session() as session:
            existing = await InferenceBackendSource.one_by_field(
                session, "name", BUILTIN_BACKEND_SOURCE_NAME
            )
            if existing:
                # Skip the write (and its UPDATED event → redundant reconcile)
                # when a restart re-seeds the same packaged content. Judged by
                # the same hash every other writer in the source layer uses, so
                # "did this change" has one answer everywhere.
                if (
                    existing.content_hash == content_hash
                    and existing.source_type == SourceTypeEnum.BUILTIN
                ):
                    return
                await existing.update(
                    session,
                    {
                        "content": content,
                        "content_hash": content_hash,
                        "source_type": SourceTypeEnum.BUILTIN,
                    },
                )
            else:
                await InferenceBackendSource.create(
                    session,
                    InferenceBackendSource(
                        name=BUILTIN_BACKEND_SOURCE_NAME,
                        source_type=SourceTypeEnum.BUILTIN,
                        content=content,
                        content_hash=content_hash,
                        enabled=True,
                    ),
                )


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


class RunnerSourceController:
    """
    Leader-only controller that materializes RunnerOverrideEntry from
    InferenceRunnerSource. On any source change (and the initial replay), it
    gathers all enabled sources, merges them in a stable order, and
    full-rewrites the override table (pure derived materialization).
    """

    async def start(self):
        async for event in InferenceRunnerSource.subscribe(
            source="runner_source_controller"
        ):
            if event.type in (
                EventType.CREATED,
                EventType.UPDATED,
                EventType.DELETED,
            ):
                await self._reconcile()

    async def _reconcile(self):
        try:
            async with async_session() as session:
                await gather_and_merge(
                    session, InferenceRunnerSource, reconcile_runner_overrides
                )
        except Exception as e:
            logger.error(f"Failed to reconcile runner override entries: {e}")


class CatalogSourceController:
    """
    Leader-only controller that materializes CatalogModelEntry from
    CatalogSource. On start it seeds the BUILTIN source from the packaged
    model-catalog.yaml, then (like RunnerSourceController) subscribes and
    full-rewrites the materialized table on any source change; the initial
    replay of existing sources drives the first reconcile.
    """

    def __init__(self, config: Config):
        self._config = config

    async def start(self):
        await self._seed_builtin_source()
        async for event in CatalogSource.subscribe(source="catalog_source_controller"):
            if event.type in (
                EventType.CREATED,
                EventType.UPDATED,
                EventType.DELETED,
            ):
                await self._reconcile()

    async def _reconcile(self):
        try:
            async with async_session() as session:
                await gather_and_merge(session, CatalogSource, reconcile_catalog)
        except Exception as e:
            logger.error(f"Failed to reconcile catalog model entries: {e}")

    async def _seed_builtin_source(self):
        """Upsert the BUILTIN CatalogSource from the packaged catalog.

        The content is refreshed to the newest packaged catalog on every start
        (so a GPUStack upgrade ships it), while the user's ``enabled`` toggle on
        the built-in source is left untouched.
        """
        try:
            raw = await asyncio.to_thread(
                read_builtin_catalog_text, self._config.model_catalog_file
            )
            content = normalize_catalog_yaml(raw)
        except Exception as e:
            logger.error(f"Failed to seed built-in catalog source: {e}")
            return

        content_hash = sha256_of(content)
        async with async_session() as session:
            existing = await CatalogSource.one_by_field(
                session, "name", BUILTIN_CATALOG_SOURCE_NAME
            )
            if existing:
                # Skip the write (and its UPDATED event → redundant reconcile)
                # when a restart re-seeds the same packaged content. Judged by
                # the same hash every other writer in the source layer uses, so
                # "did this change" has one answer everywhere.
                if (
                    existing.content_hash == content_hash
                    and existing.source_type == SourceTypeEnum.BUILTIN
                ):
                    return
                await existing.update(
                    session,
                    {
                        "content": content,
                        "content_hash": content_hash,
                        "source_type": SourceTypeEnum.BUILTIN,
                    },
                )
            else:
                await CatalogSource.create(
                    session,
                    CatalogSource(
                        name=BUILTIN_CATALOG_SOURCE_NAME,
                        source_type=SourceTypeEnum.BUILTIN,
                        content=content,
                        content_hash=content_hash,
                        enabled=True,
                    ),
                )


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


def _aggregate_instance_download_progress(
    instance: ModelInstance,
    current_file: ModelFile,
    override_progress: Optional[float] = None,
    override_state: Optional[ModelFileStateEnum] = None,
) -> Optional[float]:
    """
    Average progress over the main worker's not-yet-READY files. Subordinate
    files are excluded: instance.download_progress is the main worker's bar (the
    UI shows subordinate progress separately). 100.0 if all READY, None if none.
    override_* = a transition not yet persisted to the DB row.
    """
    # Main worker only; subordinate progress is tracked per-worker elsewhere.
    files = [f for f in instance.model_files or [] if f.worker_id == instance.worker_id]
    if not files:
        return None
    active_values: List[float] = []
    for f in files:
        if current_file.id is not None and f.id == current_file.id:
            p = (
                override_progress
                if override_progress is not None
                else current_file.download_progress
            )
            s = override_state if override_state is not None else current_file.state
        else:
            p = f.download_progress
            s = f.state
        if s == ModelFileStateEnum.READY:
            continue
        active_values.append(float(p) if p is not None else 0.0)
    if not active_values:
        return 100.0
    return sum(active_values) / len(active_values)


def _refresh_instance_download_progress(
    instance: ModelInstance, file: ModelFile, *, file_ready: bool = False
) -> bool:
    """Re-aggregate per-file progress into instance.download_progress (the
    overall bar). `file_ready` treats `file` as 100%/READY so it drops out of
    the active set and the bar can reach 100. Returns True if it changed."""
    if instance.download_progress == 100:
        return False
    if file_ready:
        aggregate = _aggregate_instance_download_progress(
            instance,
            file,
            override_progress=100.0,
            override_state=ModelFileStateEnum.READY,
        )
    else:
        aggregate = _aggregate_instance_download_progress(instance, file)
    if aggregate is not None and aggregate != instance.download_progress:
        instance.download_progress = aggregate
        return True
    return False


def _first_resolved_path(
    files: Optional[List[ModelFile]], *, exclude_lora: bool = False
) -> Optional[str]:
    """Return the first resolved path among `files`, skipping LoRA files when
    `exclude_lora` is set. None if no file carries a resolved path."""
    for file in files or []:
        if exclude_lora and file.is_lora:
            continue
        if file.resolved_paths:
            return file.resolved_paths[0]
    return None


async def _promote_to_starting_if_complete(
    session: AsyncSession, instance: ModelInstance
) -> bool:
    """When all files are ready, attach the LoRA mount list, backfill the
    resolved paths, and move to STARTING. Returns True if promoted."""
    loaded = await ModelInstance.one_by_id_with_model_files(session, instance.id)
    if not _download_completed(loaded):
        return False
    # Promotion is the single choke point into STARTING, so backfill the paths
    # here: the subordinate path never sets them and concurrent events may carry
    # a None snapshot, which crashes the worker on Path(None).
    if not instance.resolved_path:
        instance.resolved_path = _first_resolved_path(
            loaded.model_files, exclude_lora=True
        )
    if instance.draft_model_source and not instance.draft_model_resolved_path:
        instance.draft_model_resolved_path = _first_resolved_path(
            loaded.draft_model_files
        )
    mounted, lora_skipped = await _build_mounted_loras_payload(session, instance)
    if mounted is not None:
        instance.mounted_loras = mounted
    instance.state = ModelInstanceStateEnum.STARTING
    instance.state_message = "; ".join(lora_skipped) if lora_skipped else ""
    return True


def _sync_main_worker_downloading(
    instance: ModelInstance, file: ModelFile, is_draft_model: bool
) -> bool:
    """Handle a main-worker file's DOWNLOADING event. Returns need_update."""
    # First file to start: flip to DOWNLOADING and seed the bar. Draft seeds 0
    # (tracked separately); primary/LoRA seeds the active-file aggregate.
    if instance.state == ModelInstanceStateEnum.INITIALIZING:
        instance.state = ModelInstanceStateEnum.DOWNLOADING
        instance.state_message = ""
        if is_draft_model:
            instance.download_progress = 0
        else:
            aggregate = _aggregate_instance_download_progress(instance, file)
            instance.download_progress = aggregate if aggregate is not None else 0
        return True

    if instance.state != ModelInstanceStateEnum.DOWNLOADING:
        return False

    if is_draft_model:
        if (
            file.download_progress != instance.draft_model_download_progress
            and instance.draft_model_download_progress != 100
        ):
            instance.draft_model_download_progress = file.download_progress
            return True
        return False

    # Primary/LoRA file: feed the aggregate bar.
    return _refresh_instance_download_progress(instance, file)


async def _sync_main_worker_ready(
    session: AsyncSession,
    instance: ModelInstance,
    file: ModelFile,
    is_draft_model: bool,
) -> bool:
    """Handle a main-worker file's READY event. Returns need_update."""
    need_update = False

    if is_draft_model:
        if (
            instance.draft_model_download_progress != 100
            or not instance.draft_model_resolved_path
        ):
            instance.draft_model_download_progress = 100
            if file.resolved_paths:
                instance.draft_model_resolved_path = file.resolved_paths[0]
            need_update = True
    else:
        # Only the primary file owns resolved_path; LoRA files use mounted_loras.
        if (
            _is_primary_instance_model_file(file, instance, is_draft_model)
            and not instance.resolved_path
        ):
            if file.resolved_paths:
                instance.resolved_path = file.resolved_paths[0]
            need_update = True
        if _refresh_instance_download_progress(instance, file, file_ready=True):
            need_update = True

    if await _promote_to_starting_if_complete(session, instance):
        need_update = True
    elif instance.state == ModelInstanceStateEnum.INITIALIZING:
        # Some but not all files done.
        instance.state = ModelInstanceStateEnum.DOWNLOADING
        instance.state_message = ""
        need_update = True

    return need_update


async def sync_main_worker_model_file_state(
    session: AsyncSession,
    file: ModelFile,
    instance: ModelInstance,
    is_draft_model: bool = False,
):
    """Sync a main-worker model file's state onto its model instance."""

    # Re-load (with model_files) to avoid identity-map conflicts with a detached
    # instance from the caller, and to let progress aggregation read sibling rows.
    instance = await ModelInstance.one_by_id_with_model_files(session, instance.id)
    if not instance or instance.state == ModelInstanceStateEnum.ERROR:
        return

    logger.trace(
        f"Syncing model file {file.id} with model instance {instance.id}, file state: {file.state}, "
        f"progress: {file.download_progress}, message: {file.state_message}, instance state: {instance.state}"
    )

    need_update = False
    if file.state == ModelFileStateEnum.DOWNLOADING:
        need_update = _sync_main_worker_downloading(instance, file, is_draft_model)
    elif file.state == ModelFileStateEnum.READY and instance.state in (
        ModelInstanceStateEnum.DOWNLOADING,
        ModelInstanceStateEnum.INITIALIZING,
    ):
        need_update = await _sync_main_worker_ready(
            session, instance, file, is_draft_model
        )
    elif file.state == ModelFileStateEnum.ERROR:
        instance.state = ModelInstanceStateEnum.ERROR
        instance.state_message = file.state_message
        need_update = True

    if need_update:
        await ModelInstanceService(session).update(instance)


async def _sync_subordinate_worker(
    session: AsyncSession,
    instance: ModelInstance,
    subordinate: ModelInstanceSubordinateWorker,
    file: ModelFile,
) -> bool:
    """
    Sync one subordinate worker's file state. subordinate.download_progress is
    display-only (completion is decided by ModelFile state, not this field).
    Returns need_update.
    """
    if file.state == ModelFileStateEnum.DOWNLOADING:
        if file.download_progress == subordinate.download_progress:
            return False
        subordinate.download_progress = file.download_progress
        return True

    if file.state == ModelFileStateEnum.READY:
        # progress may already be 100 from the final DOWNLOADING report, so a
        # READY event must still re-check completion — gating on the progress
        # value would skip the STARTING transition and stick at 100%.
        need_update = subordinate.download_progress != 100
        subordinate.download_progress = 100
        if instance.state in (
            ModelInstanceStateEnum.DOWNLOADING,
            ModelInstanceStateEnum.INITIALIZING,
        ):
            if await _promote_to_starting_if_complete(session, instance):
                need_update = True
        return need_update

    if file.state == ModelFileStateEnum.ERROR:
        instance.state = ModelInstanceStateEnum.ERROR
        instance.state_message = file.state_message
        return True

    return False


async def sync_distributed_model_file_state(
    session: AsyncSession, file: ModelFile, instance: ModelInstance
):
    """Sync a subordinate-worker model file's state onto its model instance."""

    # Re-load to avoid identity-map conflicts with a detached caller instance.
    instance = await ModelInstance.one_by_id(session, instance.id)
    if not instance or instance.state == ModelInstanceStateEnum.ERROR:
        return

    if (
        not instance.distributed_servers
        or not instance.distributed_servers.download_model_files
    ):
        return

    subordinate = next(
        (
            item
            for item in instance.distributed_servers.subordinate_workers or []
            if item.worker_id == file.worker_id
        ),
        None,
    )
    if subordinate is None:
        return

    logger.trace(
        f"Syncing distributed model file {file.id} with model instance {instance.name}, file state: {file.state}, "
        f"progress: {file.download_progress}, message: {file.state_message}, instance state: {instance.state}"
    )

    if await _sync_subordinate_worker(session, instance, subordinate, file):
        flag_modified(instance, "distributed_servers")
        await ModelInstanceService(session).update(instance)


async def _build_mounted_loras_payload(
    session: AsyncSession, instance: ModelInstance
) -> Tuple[Optional[List[LoraListEntry]], List[str]]:
    """
    Build LoraListEntry list for Model.lora_list entries whose LoRA ModelFile
    is READY on this instance. Used once when transitioning to STARTING.

    Returns (mounted_loras, skip_messages). skip_messages collects per-entry
    reasons for any LoRA that could not be resolved (invalid source config),
    so callers can surface them via instance.state_message.
    """
    model = await Model.one_by_id(session, instance.model_id)
    if not model:
        return None, []
    entries = normalized_lora_list(model)
    if not entries:
        return [], []
    inst = await ModelInstance.one_by_id_with_model_files(session, instance.id)
    if not inst:
        return None, []
    out: List[LoraListEntry] = []
    skipped: List[str] = []
    for entry in entries:
        try:
            src = lora_entry_to_model_source(entry)
        except ValueError as e:
            msg = f"LoRA {entry.lora_name!r} skipped: {e}"
            logger.warning("%s (instance=%s, entry=%s)", msg, instance.name, entry)
            skipped.append(msg)
            continue
        for f in inst.model_files or []:
            if not getattr(f, "is_lora", False):
                continue
            if f.model_source_index != src.model_source_index:
                continue
            if f.state != ModelFileStateEnum.READY or not f.resolved_paths:
                continue
            out.append(
                LoraListEntry(
                    lora_name=lora_route_name_for(model.name, entry.lora_name),
                    lora_repo_name=entry.lora_repo_name,
                    source=entry.source,
                    huggingface_filename=entry.huggingface_filename,
                    model_scope_file_path=entry.model_scope_file_path,
                    local_path=entry.local_path,
                    path=f.resolved_paths[0],
                    model_file_id=f.id,
                )
            )
            break
    return out, skipped


def _download_completed(instance: Optional[ModelInstance]) -> bool:
    """True when every ModelFile (primary, LoRA, draft) is READY. Pure check over
    an already-loaded instance — the caller owns the single eager load."""
    if instance is None:
        return False

    if not instance.model_files and not instance.draft_model_source:
        return False

    for model_file in instance.model_files or []:
        if model_file.state != ModelFileStateEnum.READY:
            return False

    if instance.draft_model_source:
        draft_files = instance.draft_model_files or []
        if not draft_files:
            return False
        for draft_file in draft_files:
            if draft_file.state != ModelFileStateEnum.READY:
                return False

    # Subordinate files are in instance.model_files (checked above) — the single
    # source of truth. The old subordinate_workers progress check raced with the
    # distributed sync path and could stick at DOWNLOADING after all hit 100%.
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


async def _get_worker_tenant_scopes(
    session: AsyncSession, worker_ids: Iterable[int]
) -> Dict[int, Tuple[Optional[int], Optional[int]]]:
    """Resolve ``(cluster_id, owner_principal_id)`` for each worker so newly
    created ModelFiles inherit the same tenant scope as their host worker —
    matching the route-side derivation in ``routes/model_files.create_model_file``.
    Without this, ModelFiles created by the controller come back with NULL
    tenant columns and are invisible to org principals."""
    scopes: Dict[int, Tuple[Optional[int], Optional[int]]] = {}
    unique_worker_ids = {wid for wid in worker_ids if wid is not None}
    if not unique_worker_ids:
        return scopes
    workers = await Worker.all_by_fields(
        session,
        extra_conditions=[Worker.id.in_(unique_worker_ids)],
    )
    for worker in workers:
        scopes[worker.id] = (
            worker.cluster_id,
            getattr(worker, "owner_principal_id", None),
        )
    return scopes


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
    #
    # ``batch_size`` is optional and means "provision at most this many at a
    # time", so an unset one caps nothing -- exactly as the delta above already
    # reads it. Comparing it anyway raises TypeError on ``None <= 0``, which
    # the caller catches as a failed reconcile: the pool logs one line and then
    # never creates a worker, for any provider, because the retry hits the same
    # comparison.
    if pool.batch_size is not None and pool.batch_size <= len(provisioning_workers):
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
            # Denormalize from cluster so tenant_list_conditions can match
            # without joining clusters. Mirrors the worker registration
            # path in routes/workers.update_worker_data.
            owner_principal_id=pool.cluster.owner_principal_id,
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
            except Exception:
                # With the traceback: a pool that cannot reconcile creates no
                # workers at all, and the message alone ("'<=' not supported
                # between instances of 'NoneType' and 'int'") says nothing
                # about which pool or which line refused. ``logger.exception``
                # appends the exception itself, so interpolating it here would
                # only print it twice.
                logger.exception("Failed to reconcile worker pool")

    async def _reconcile(self, event: Event):
        """
        Reconcile the worker pool state with the current event.
        """
        # Only the id is needed -- the pool is re-read below either way -- so
        # an id-only payload (see Event) is as good as a hydrated one here.
        # On a DELETE the row is gone and one_by_id returns None.
        pool_event_id = resolve_event_id(event)
        if pool_event_id is None:
            # No row to reconcile. Return rather than passing None to
            # one_by_id, which warns ("fully NULL primary key identity cannot
            # load any object") and may raise in a future SQLAlchemy.
            return
        logger.info(f"Reconcile worker pool {pool_event_id} with event {event.type}")
        async with async_session() as session:
            pool = await WorkerPool.one_by_id(
                session, pool_event_id, options=[selectinload(WorkerPool.cluster)]
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


def _record_ssh_endpoint(
    provider_config: Dict[str, Any], instance: Optional[CloudInstance]
) -> bool:
    """Note where SSH answers, when the provider doesn't serve it on the
    instance's own address. Returns whether anything was written.

    Only providers that publish instances behind a mapped address report one;
    for the rest ``advertise_address:22`` is the answer and nothing is stored.
    Already-recorded endpoints are left alone so a later poll that happens to
    come back without the mapping can't erase it.
    """
    if instance is None or not instance.ssh_endpoint:
        return False
    if "ssh_endpoint" in provider_config:
        return False
    host, port = instance.ssh_endpoint
    ssh_endpoint: Dict[str, Any] = {"host": host, "port": port}
    # Omit an unknown user rather than storing "": the provider only fills it
    # in once the instance is up, and a consumer shouldn't have to tell an
    # empty string apart from an absent key.
    if instance.ssh_user:
        ssh_endpoint["user"] = instance.ssh_user
    provider_config["ssh_endpoint"] = ssh_endpoint
    return True


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
        # None means the provider registered nothing (no SSH key API); leave
        # external_id NULL rather than storing the string "None", so teardown
        # knows there is no upstream key to delete.
        ssh_key.external_id = str(ssh_key_id) if ssh_key_id is not None else None
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
        ssh_key = await Credential.one_by_id(session, worker.ssh_key_id)
        if ssh_key is None:
            raise ValueError(f"SSH key {worker.ssh_key_id} not found")
        user_data = await client.construct_user_data(
            server_url=worker.cluster.server_url or cfg.server_external_url,
            token=worker.cluster.registration_token,
            image_name=get_cluster_image_name(
                worker.cluster.worker_config,
                worker.cluster.system_default_container_registry,
            ),
            os_image=worker.worker_pool.os_image,
            secret_configs=secret_configs,
            worker_name=worker.name,
            # Providers with no SSH key API embed this in the cloud-config;
            # the ones that have one attach it by id instead and ignore it.
            ssh_public_key=ssh_key.public_key,
        )
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
        # Copy rather than alias: provider_config is a plain JSON column, so
        # SQLAlchemy only notices a write when the attribute is set to a
        # different object. Mutating the loaded dict in place and assigning it
        # back would leave the change unsaved.
        provider_config = dict(worker.provider_config or {})
        volumes = list(
            (getattr(worker.worker_pool.cloud_options, "volumes", None) or [])
        )
        volume_ids = provider_config.get("volume_ids", [])
        # Backfill a mapping that wasn't published yet the first time round.
        # wait_for_public_ip returns as soon as an address appears, which for a
        # provider that maps SSH elsewhere can be before the mapping exists —
        # and the branch below only runs while advertise_address is still
        # empty, so without this the hint would stay missing for good.
        backfilled = _record_ssh_endpoint(provider_config, instance)
        if backfilled:
            worker.provider_config = provider_config
        if worker.advertise_address is None or worker.advertise_address == "":
            try:
                instance = await client.wait_for_public_ip(worker.external_id)
                worker.advertise_address = (
                    instance.ip_address if instance.ip_address else ""
                )
                worker.state_message = "Waiting for volumes to attach"
                # Last in the branch on purpose: this is a display aid, and
                # failing to read it must not hold back the state machine.
                if _record_ssh_endpoint(provider_config, instance):
                    worker.provider_config = provider_config
            except InstanceProvisioningFailed:
                # Terminal on the provider side; see _provisioning_before_started.
                raise
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
            # Membership, not hasattr: provider_config is a dict, so hasattr
            # was always False and this overwrote the ids of volumes that had
            # just been created -- losing the only record of them, so teardown
            # could never find the volumes to delete.
            if "volume_ids" not in provider_config:
                provider_config["volume_ids"] = []
            worker.provider_config = provider_config
            worker.state = WorkerStateEnum.INITIALIZING
            if worker.cluster.state != ClusterStateEnum.PROVISIONED:
                worker.cluster.state = ClusterStateEnum.PROVISIONED
                await worker.cluster.update(session=session, auto_commit=False)
            worker.state_message = "Initializing: installing required drivers and software. The worker will start automatically after setup."
        else:
            changed = backfilled
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
            except InstanceProvisioningFailed:
                # The provider gave up on the instance, so polling it again on
                # the next reconcile would spin forever. Let it through to the
                # caller, which parks the worker in ERROR.
                raise
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
                    # Needed by the DELETING branch below: the hard delete in
                    # ``_deleting_instance`` cascades to the worker's SYSTEM
                    # principal, and ``_handle_cascade_delete`` reads that
                    # ``lazy="noload"`` relationship off the instance.
                    selectinload(Worker.system_principal),
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
        # This runs on DELETED too -- cleaning up after a deleted cluster is
        # the point -- and the id is all it needs, so an id-only payload
        # (see Event) serves it just as well as a hydrated row.
        cluster_id = resolve_event_id(event)
        if cluster_id is None:
            return
        mcp_resource_name = mcp_handler.default_mcp_bridge_name
        desired_registries = []
        to_delete_prefix = mcp_handler.cluster_worker_prefix(cluster_id)
        try:
            await mcp_handler.ensure_mcp_bridge(
                client=self._higress_network_api,
                namespace=self._cfg.gateway_namespace,
                mcp_bridge_name=mcp_resource_name,
                desired_registries=desired_registries,
                to_delete_prefix=to_delete_prefix,
            )
        except Exception as e:
            logger.error(
                "Failed to ensure MCPBridge for cluster "
                f"{event_field(event.data, 'name', cluster_id)}: {e}"
            )
            raise


async def notify_model_route_target(session: AsyncSession, model: Model, event: Event):
    if event.type == EventType.DELETED:
        return
    should_notify = False
    if event.changed_fields is not None:
        # ``native_anthropic_api`` is read where the route's ai-proxy provider
        # entry is built, and that only runs off a route event -- so without it
        # here, flipping the selector would change nothing until the deployment
        # happened to scale.
        related_fields = ["ready_replicas", "replicas", "native_anthropic_api"]
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
        if route.created_model_id is None:
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
        # On DELETED both desired sets are empty and only the id is read, so
        # this path works off an id-only payload (see Event) -- which is what
        # a cross-instance delete carries, and skipping it would strand the
        # registry and proxy of a provider that no longer exists. Deriving
        # the two specs first would defeat that: they need a hydrated row and
        # their values are discarded here anyway.
        deleted = event.type == EventType.DELETED
        provider_id = resolve_event_id(event)
        provider_registry = (
            None if deleted else mcp_handler.provider_registry(model_provider)
        )
        registry_to_remove = provider_registry is None or deleted
        # Match by exact name (not prefix) so that deleting provider id "1" does
        # not also drop other providers whose id shares that numeric prefix
        # (e.g. "provider-10", "provider-11").
        to_delete_names = (
            [mcp_handler.provider_registry_name(provider_id)]
            if registry_to_remove
            else None
        )
        desired_registries = [] if registry_to_remove else [provider_registry]

        provider_proxy = None if deleted else mcp_handler.provider_proxy(model_provider)
        proxy_to_remove = provider_proxy is None or deleted
        to_delete_proxy_names = (
            [mcp_handler.provider_proxy_name(provider_id)] if proxy_to_remove else None
        )
        desired_proxies = [] if proxy_to_remove else [provider_proxy]

        try:
            await mcp_handler.ensure_mcp_bridge(
                client=self._higress_network_api,
                namespace=self._config.gateway_namespace,
                mcp_bridge_name=mcp_handler.default_mcp_bridge_name,
                desired_registries=desired_registries,
                desired_proxies=desired_proxies,
                to_delete_names=to_delete_names,
                to_delete_proxies_names=to_delete_proxy_names,
            )
        except Exception as e:
            logger.error(
                "Failed to ensure MCPRegistry for model provider "
                f"{event_field(model_provider, 'name', provider_id)}: {e}"
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
                await mcp_handler.ensure_wasm_plugin(
                    api=self._higress_extension_api,
                    name=mcp_handler.gpustack_ai_proxy_name,
                    namespace=self._config.gateway_namespace,
                    spec_diff=partial(
                        mcp_handler.ai_proxy_diff_spec,
                        expected_providers=provider_config_list,
                        expected_match_rules=match_rules,
                        operating_id_prefix=mcp_handler.provider_id_prefix,
                    ),
                )
        except Exception as e:
            logger.error(f"Failed to ensure provider's ai_proxy config: {e}")
            raise

    async def _notify_provider_model_routes(
        self, session: AsyncSession, model_provider: ModelProvider, event: Event
    ):
        if event.type != EventType.UPDATED:
            return
        changed_fields = event.changed_fields or {}
        should_notify = False
        if "config" not in changed_fields:
            return

        # the changed field "config" must have old and new value, otherwise it's not a valid update event for config change.
        # index 0 of the tuple is the old value, index 1 is the new value.
        # each value must be a list with only 1 element as it is a norman field instead of relationship field.
        old_config = changed_fields["config"][0][0]
        if isinstance(changed_fields["config"][0][0], BaseModel):
            old_config = changed_fields["config"][0][0].model_dump()
        new_config = changed_fields["config"][1][0]
        if isinstance(changed_fields["config"][1][0], BaseModel):
            new_config = changed_fields["config"][1][0].model_dump()

        # use hardcoded fields to determine whether to notify.
        # For ProviderConfigType, including:
        # - openaiCustomUrl
        # - ollamaServerHost
        # - difyApiUrl
        # - claudeCustomUrl
        # The above fields will affect the registry type of the provider_registry,
        # it requires notifying ingress to regenerate registry destination.
        related_fields = [
            "openaiCustomUrl",
            "ollamaServerHost",
            "difyApiUrl",
            "claudeCustomUrl",
        ]
        for field in related_fields:
            if old_config.get(field) != new_config.get(field):
                should_notify = True
                break
        if not should_notify:
            return
        targets = await ModelRouteTarget.all_by_fields(
            session=session,
            fields={"provider_id": model_provider.id},
            options=[selectinload(ModelRouteTarget.model_route)],
        )
        unique_routes = {
            target.model_route.id: target.model_route
            for target in targets
            if target.model_route is not None
        }
        for route in unique_routes.values():
            route_copy = ModelRoute.model_validate(route.model_dump())
            await event_bus.publish(
                route_copy.__class__.__name__.lower(),
                Event(type=EventType.UPDATED, data=route_copy),
            )

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
            await self._notify_provider_model_routes(session, model_provider, event)


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
            "overridden_model_name",
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
        # Handle ID-only events from distributed mode
        target_id = (
            target.id
            if hasattr(target, 'id')
            else target.get('id') if isinstance(target, dict) else None
        )
        if not target_id:
            return
        target: ModelRouteTarget = await ModelRouteTarget.one_by_id(session, target_id)
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

    async def _update_orphan_route(
        self, session: AsyncSession, target: ModelRouteTarget, event: Event
    ) -> bool:
        """
        Update the orphan route if the target is deleted or has no associated model.
        If the target model is not deleted, transfer model_route to a non model-created model.
        """

        if event.type != EventType.DELETED:
            return True
        if target.model_id is None:
            return True
        model = await Model.one_by_id(session, target.model_id)
        if not model or model.deleted_at is not None:
            return True
        # If the model is not deleted, transfer the model route to a non model-created model route to avoid service disruption.
        # The model route will be automatically deleted by the controller after the target is deleted.
        orphan_route = await ModelRoute.one_by_id(session=session, id=target.route_id)
        if (
            not orphan_route
            or orphan_route.deleted_at is not None
            or orphan_route.created_model_id is None
        ):
            # The route is already deleted or not created by model, no need to transfer.
            # returns true to trigger parent notification and state sync to update the route state if needed.
            return True
        try:
            route_service = ModelRouteService(session=session)
            await route_service.update(
                orphan_route, source={"created_model_id": None}, auto_commit=True
            )
        except Exception as e:
            logger.error(f"Failed to transfer model route {orphan_route.id}: {e}")
            return True
        return False

    async def _reconcile(self, event: Event):
        target: ModelRouteTarget = event.data
        if not target:
            return
        # Every branch below keys off target fields (route_name, route_id,
        # model_id, state), none of which an unhydrated payload has (see
        # Event). Known gap: this controller exists to cover cascades that
        # bypass ModelRouteService, and it is leader-only, so nothing else
        # invalidates the route caches or transfers an orphaned route.
        # Closing it needs the deleted row's route_name, which no event can
        # carry, so it belongs in a periodic pass.
        if not isinstance(target, ModelRouteTarget):
            logger.warning(
                f"Model route target {resolve_event_id(event)} {event.type} "
                f"not reconciled: the event carries only an id and the row is "
                f"gone; route caches may be stale until the next write"
            )
            return
        async with async_session() as session:
            # Cover cascade create/delete that bypass ModelRouteService.
            # UPDATED is skipped — it cannot change the resolved target set.
            if event.type in (EventType.CREATED, EventType.DELETED):
                route_name = target.route_name
                if route_name:
                    route_service = ModelRouteService(session=session)
                    names = await collect_route_cache_names(
                        session, target.route_id, route_name
                    )
                    for name in names:
                        await delete_cache_by_key(
                            route_service.resolve_route_targets, name
                        )
                        await delete_cache_by_key(
                            route_service.get_model_auth_info_by_name, name
                        )

            should_notify_parents = await self._update_orphan_route(
                session, target, event
            )
            if should_notify_parents:
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
        # Reached only with a hydrated route -- the caller guards, and a
        # DELETE returned above -- so this is just the canonical way to read
        # the id, not id-only handling.
        model_route_id = resolve_event_id(event)
        if not model_route_id:
            return False
        model_route: ModelRoute = await ModelRoute.one_by_id(
            session,
            model_route_id,
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
        if target_total == 0 and model_route.created_model_id is not None:
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
        # sync_gateway and distribute_models_to_user both need the row's
        # fields (name for the ingress names, model_dump for the per-user
        # copies), which an unhydrated payload does not have (see Event).
        # Known gap, not a safe skip: re-reading is not an option either
        # (ModelRouteService.delete takes the default hard-delete path), and
        # this controller is leader-only. Deletes through the API are still
        # covered -- the service drops the route caches itself, and that
        # invalidation is broadcast -- so what is uncovered is the cascade
        # path this controller exists for.
        if not isinstance(model_route, ModelRoute):
            logger.warning(
                f"Model route {resolve_event_id(event)} {event.type} not "
                f"reconciled: the event carries only an id and the row is "
                f"gone; gateway config for it may be stale"
            )
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
