import hashlib
import json
import logging
import random
import string
import asyncio
from datetime import datetime, timezone
from importlib.resources import files
from functools import partial
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Sequence,
    Tuple,
    Optional,
    Set,
    Awaitable,
    Callable,
)
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
from gpustack.policies.scorers.pairing_affinity_scorer import PairingRetentionScorer
from gpustack.policies.scorers.placement_scorer import PlacementScorer, ScaleTypeEnum
from gpustack.policies.scorers.score_chain import (
    ModelInstanceScoreChain,
)
from gpustack.policies.base import ModelInstanceScore, ModelInstanceScorer
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
    servable_instances,
    BackendEnum,
    BackendSourceEnum,
    DegradationReasonEnum,
    LoraListEntry,
    ModelSource,
    Model,
    ModelInstance,
    ModelInstanceCreate,
    ModelInstanceStateEnum,
    ModelInstanceSubordinateWorker,
    ModelSpecBase,
    ModelStateEnum,
    RoleNameEnum,
    RoleSpec,
    RoleStatus,
    SourceEnum,
    get_backend,
    member_worker_ids,
    role_effective_model,
)
from gpustack.schemas.gpu_instance_types import GPUInstanceType
from gpustack.server.workqueue import WorkEvent, WorkEventType, WorkQueue
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
    CacheServiceStateEnum,
)
from gpustack.schemas.cache_providers import (
    render_optional_template,
    resolved_field_values,
)
from gpustack.schemas.pd_modes import PDTensorParallelPairingEnum
from gpustack.server.cache_provider_catalog import (
    builtin_catalog_text,
    get_cache_provider,
)
from gpustack.server.pd_pairing import (
    PAIRING_ANY_PARALLELISM,
    PAIRING_TP,
    role_parameters,
    tensor_parallel_rule,
    undecidable_factors,
    violates_tensor_parallel_direction,
)
from gpustack.utils.command import find_last_int_parameter, find_last_parameter
from gpustack.server import pd_membership
from gpustack.server.pd_membership import outcome_for as membership_outcome_for
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
from gpustack.schemas.cache_providers import CacheProvider
from gpustack.schemas.cache_provider_source import (
    BUILTIN_CACHE_PROVIDER_SOURCE_NAME,
    CacheProviderSource,
    reconcile_cache_providers,
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
)
from gpustack.gateway.client.networking_istio_io_v1alpha3_api import (
    NetworkingIstioIoV1Alpha3Api,
)
from gpustack.gateway import utils as mcp_handler
from gpustack.routes.plugins import (
    RouteArtifactCollector,
    RouteReconcileContext,
    dispatch_route_reconcile,
)
from gpustack.gateway import get_async_k8s_config
from gpustack.schemas.model_provider import (
    ModelProvider,
)

logger = logging.getLogger(__name__)


def _gateway_registrable_instances(
    model: Model, instances: List[ModelInstance]
) -> List[ModelInstance]:
    """The members whose addresses may become gateway upstreams.

    For a role-bearing group that is the router alone. Every member of a PD
    group serves an OpenAI-shaped API on its own port, so registering them all
    is not a duplicate registration — it is a set of upstreams that answer the
    same requests *wrongly*: a request balanced onto a prefill returns after
    one token, and one onto a decode runs without the prefix its KV was
    supposed to carry. Both return 200 with plausible text, which is the
    failure mode PD is least able to absorb.

    A group with no router role registers nothing here rather than falling
    back to its GPU members, for the same reason: there is no member that can
    correctly answer a whole request on its own.
    """
    return servable_instances(model, instances)


# The bus speaks CREATED/UPDATED/DELETED and the work queue speaks
# ADDED/MODIFIED/DELETED. Mapped rather than unified because the queue's
# DELETED carries queue semantics — it jumps the ready queue and is sticky
# under coalescing — which the bus's has no opinion about.
_WORK_EVENT_TYPE_BY_BUS = {
    EventType.CREATED: WorkEventType.ADDED,
    EventType.UPDATED: WorkEventType.MODIFIED,
    EventType.DELETED: WorkEventType.DELETED,
}


class ModelController:
    """Reconciles a model's replicas, status, routes and gateway registration.

    Events go through a per-model work queue rather than straight into
    `_reconcile`, and for a role-bearing model that is a correctness
    requirement rather than a throughput one. The trigger chain is "instance
    DELETED -> Model UPDATED -> reconcile", so retiring a 4P4D generation is
    nine deletions and nine reconciles — and each of the middle ones sees a
    group short of members. Reconciling on every one of them recreates what
    the deletion is still in the middle of removing. One reconcile per burst,
    reading the settled state, cannot make that mistake.
    """

    def __init__(self, cfg: Config):
        self._config = cfg
        self._k8s_config = get_async_k8s_config(cfg=cfg)
        self._disable_gateway = cfg.gateway_mode == GatewayModeEnum.disabled
        # Keyed by model id, so different models still reconcile concurrently
        # while one model's events serialise.
        self._queue: WorkQueue = WorkQueue(coalesce=self._merge_events)
        self._inflight: Dict[Any, asyncio.Task] = {}
        self._dispatch_task: Optional[asyncio.Task] = None

    @staticmethod
    def _merge_events(existing: WorkEvent, incoming: WorkEvent) -> WorkEvent:
        """Collapse a burst into one reconcile without losing what changed.

        Plain latest-wins would be wrong here even though the row it carries is
        the freshest one: `notify_model_route_target` decides whether to publish
        by asking which fields moved, so dropping an intermediate event's
        `changed_fields` drops the notification that event was carrying. A
        `state` transition followed by an unrelated edit would leave the gateway
        holding a target it was never told to update.

        So the row is the newest and the changed-field set is the union — which
        is what "one reconcile of everything that happened since the last one"
        actually means. Per field the oldest before-value and the newest
        after-value are kept, so the pair still describes the whole span rather
        than its last step.

        A pending DELETED stays sticky (the default policy): a model row that is
        gone must not be reconciled as if it were merely updated.
        """
        if (
            existing.type == WorkEventType.DELETED
            and incoming.type != WorkEventType.DELETED
        ):
            return existing

        old_event: Event = existing.object
        new_event: Event = incoming.object
        if (
            old_event is None
            or new_event is None
            or not old_event.changed_fields
            or new_event.changed_fields is None
        ):
            return incoming

        merged = dict(new_event.changed_fields)
        for field, (before, after) in old_event.changed_fields.items():
            if field in merged:
                merged[field] = (before, merged[field][1])
            else:
                merged[field] = (before, after)
        new_event.changed_fields = merged
        return incoming

    async def start(self):
        """
        Start the controller.
        """
        if not self._disable_gateway:
            base_client = k8s_client.ApiClient(configuration=self._k8s_config)
            self._higress_network_api = NetworkingHigressIoV1Api(base_client)
            self._higress_extension_api = ExtensionsHigressIoV1Api(base_client)

        self._dispatch_task = asyncio.create_task(self._dispatch())
        try:
            async for event in Model.subscribe(source="model_controller"):
                if event.type == EventType.HEARTBEAT:
                    continue
                model = event.data
                if model is None:
                    continue
                self._queue.add(
                    WorkEvent(
                        keys=(model.id,),
                        type=_WORK_EVENT_TYPE_BY_BUS.get(
                            event.type, WorkEventType.MODIFIED
                        ),
                        object=event,
                    )
                )
        finally:
            tasks: List[asyncio.Task] = []
            if self._dispatch_task is not None:
                self._dispatch_task.cancel()
                tasks.append(self._dispatch_task)
            for task in list(self._inflight.values()):
                task.cancel()
                tasks.append(task)
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    async def _dispatch(self):
        while True:
            event = await self._queue.get()
            self._inflight[event.keys] = asyncio.create_task(self._process(event))

    async def _process(self, event: WorkEvent):
        keys = event.keys
        try:
            await self._reconcile(event.object)
            self._queue.forget(keys)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Failed to reconcile model %s", keys)
            self._queue.add_rate_limited(event)
        finally:
            self._queue.done(keys)
            _ = self._inflight.pop(keys, None)

    async def _ensure_model_mcp_bridge(
        self, session: AsyncSession, event_type: EventType, model: Model
    ):
        if self._disable_gateway:
            return
        model_instances = await ModelInstance.all_by_fields(
            session,
            fields={"model_id": model.id, "deleted_at": None},
        )
        model_instances = _gateway_registrable_instances(model, model_instances)
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
                drain_due = await sync_replicas(session, model)
                if drain_due is not None:
                    # The only self-scheduled pass this controller books, and it
                    # exists because nothing else would arrive: a drained member
                    # publishes no further Model event and a settled deployment
                    # publishes none either, so `_reap_drained` would wait on a
                    # reconcile that never comes. `add_after` is
                    # last-schedule-wins per key, so re-booking on every pass
                    # keeps one timer rather than accumulating them.
                    self._queue.add_after(
                        WorkEvent(
                            keys=(model.id,),
                            type=WorkEventType.MODIFIED,
                            object=event,
                        ),
                        drain_due,
                    )
                # The status owner has to run on the spec side too, not only on
                # instance events. `role_status.desired` is read straight off
                # `roles[].replicas`, so a spec edit changes it with no instance
                # changing — and a model with no instances at all (a group
                # parked at `replicas: 0`) would otherwise never have its status
                # computed even once, leaving the UI with no declared shape to
                # show. Safe to call from both sides: the change gate inside
                # means a pass that finds nothing new writes nothing, so the
                # update this may publish converges after one round.
                #
                # Load a session-attached row rather than writing through
                # `event.data`: what arrives on the bus is a detached copy whose
                # identity may already have been collected, and assigning to it
                # raises "parent object of type <Model> has been garbage
                # collected". `notify_model_route_target` below re-fetches for
                # the same reason.
                attached = await Model.one_by_id(session, model.id)
                if attached is not None:
                    await sync_model_status(session, attached)
                await notify_model_route_target(
                    session=session, model=model, event=event
                )
                await sync_categories_and_meta(session, model, event)
                await self._ensure_model_mcp_bridge(session, event.type, model)
                await self._sync_model_ai_proxy(session, model)
        except Exception as e:
            logger.error(f"Failed to reconcile model {model.name}: {e}")

    async def _sync_model_ai_proxy(self, session: AsyncSession, model: Model):
        """The Model controller owns the ai-proxy CR: the entry's content
        is a pure function of the deployment, and instance changes arrive
        here as model events (ready-replica sync). Route CRUD enqueues the
        affected models when a reference appears or disappears; a deleted
        or soft-deleted row strips the deployment's entry."""
        if self._disable_gateway:
            return
        await sync_model_ai_proxy(
            cfg=self._config,
            session=session,
            extensions_api=self._higress_extension_api,
            model_id=model.id,
        )


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
                replicas_updated = await sync_model_status(session, model)
                if any_lora_route_deleted and not replicas_updated:
                    await session.commit()
                if any_lora_route_deleted:
                    await revoke_model_access_cache(session=session)
        except Exception as e:
            logger.error(
                "Failed to reconcile model instance "
                f"{event_field(model_instance, 'name', instance_id)}: {e}"
            )


def _component_replica_count(
    spec, provider, config_fields: Optional[Dict[str, Any]]
) -> int:
    """The component's replica count: its declared one unless a managed
    field sizes it (a value below one, or one that is not a number at
    all, keeps the declaration). The sizing field resolves through its
    visibility gate, so a count offered only with a feature falls back to
    the gated default while the feature is off."""
    count = spec.replicas if spec else 1
    if spec is None or not spec.replicas_by:
        return count
    resolved = resolved_field_values(
        provider.fields if provider else [], config_fields or {}
    )
    configured = resolved.get(spec.replicas_by)
    try:
        if configured is not None and int(configured) >= 1:
            count = int(configured)
    except (TypeError, ValueError):
        pass
    return count


class CacheServiceController:
    """
    Reconciles managed cache services onto their desired CacheServiceInstance
    set and aggregates instance states back onto the service row.

    The provider declaration's topology dictates the desired set: replicas
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
        # Four drivers reconcile the same service — its own events, worker
        # events, instance events and the periodic pass — and each reads
        # the instance rows before deciding what is missing. Without a
        # turn each, two of them observe the same gap and both fill it.
        self._service_locks: Dict[int, asyncio.Lock] = {}

    def _service_lock(self, cache_service_id: int) -> asyncio.Lock:
        lock = self._service_locks.get(cache_service_id)
        if lock is None:
            lock = asyncio.Lock()
            self._service_locks[cache_service_id] = lock
        return lock

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
            if cache_service is None:
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
                    if service is None or service.deleted_at is not None:
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
                        provider = await get_cache_provider(
                            session, service.provider_name
                        )
                        await self._sync_service_aggregate(session, service, provider)
                        changed = set(event.changed_fields or {})
                        # A component turning RUNNING may unblock a
                        # dependent component's creation (the stores
                        # wait for the master's address), so state flips
                        # re-run the reconcile for multi-component
                        # providers.
                        if "state" in changed:
                            if provider and provider.components:
                                # through the id, which is where the
                                # per-service turn is taken: reconciling
                                # here directly is one of the four
                                # drivers the lock exists for
                                await self._reconcile_service_by_id(service.id)
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
                    role=mi.role,
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
                        fields={},
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
                    fields={"cluster_id": cluster_id},
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
            async with self._service_lock(cache_service_id):
                async with async_session() as session:
                    service = await CacheService.one_by_id(session, cache_service_id)
                    if service is None or service.deleted_at is not None:
                        # The lock stays: dropping it here, while holding
                        # it, hands a waiter a lock no later caller will
                        # look up — two reconciles for one service could
                        # then run at once. Ids are never reused, so what
                        # is left behind is one lock per service seen.
                        return
                    await self._reconcile_service(session, service)
        except Exception as e:
            logger.error(f"Failed to reconcile cache service {cache_service_id}: {e}")

    async def _reconcile_service(self, session: AsyncSession, service: CacheService):
        """Drive the service's instance rows to the desired per-component
        worker sets, then refresh the service-level aggregate state. A
        component depending on another (stores needing the master's
        address) only gets instances once a dependency instance is
        RUNNING with its port known; the dependency's RUNNING event
        re-runs this reconcile, so the gate converges without polling."""
        provider = await get_cache_provider(session, service.provider_name)
        instances = await CacheServiceInstance.all_by_fields(
            session, {"cache_service_id": service.id}
        )
        desired_by_component, error_message, reconcile = (
            await self._desired_component_workers(session, service, instances, provider)
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

        # Dependency addresses stamp into dependent instances at creation
        # (the running process bakes them into its config), so they are
        # resolved once per pass: the dependency's RUNNING instance plus
        # its worker's IP.
        config_fields = service.config.fields if service.config else None
        addresses = await self._component_addresses(
            session, provider, instances, config_fields
        )

        surviving: List[CacheServiceInstance] = []
        # How many rows of each (component, worker) pair the desired
        # layout still has room for; a replica beyond that count is
        # surplus (the count dropped, or the pool moved elsewhere).
        room: Dict[Tuple[str, int], int] = {
            (component, worker_id): replicas
            for component, layout in desired_by_component.items()
            for worker_id, replicas in layout.items()
        }
        for instance in instances:
            component = instance.component or ""
            spec = provider.get_component(component) if provider else None
            stale_address = False
            if spec is not None and spec.depends_on:
                expected = {
                    name: address
                    for name, address in addresses.items()
                    if name == spec.depends_on
                }
                # No address to hand down reads two ways: the dependency has
                # not come up yet — leave the dependent alone — or it was
                # turned off, in which case an instance still carrying its
                # address is running against one that will never answer.
                dependency_off = not (
                    provider
                    and provider.component_enabled(spec.depends_on, config_fields)
                )
                stale_address = (
                    (instance.component_addresses or {}) != expected
                    if expected or dependency_off
                    else False
                )
            key = (component, instance.worker_id)
            surplus = room.get(key, 0) <= 0
            if surplus or stale_address:
                await instance.delete(session)
                reason = (
                    "its dependency's address changed"
                    if stale_address
                    else (
                        f"worker {instance.worker_id} holds no more replicas "
                        f"of component '{component}'"
                    )
                )
                logger.info(
                    f"Deleted instance {instance.id} of cache service "
                    f"{service.name}: {reason}"
                )
            else:
                surviving.append(instance)
                room[key] -= 1

        for component, layout in desired_by_component.items():
            spec = provider.get_component(component) if provider else None
            instance_addresses: Optional[Dict[str, str]] = None
            # A dependency turned off by its declared field is not something
            # to wait for: it will never run, so the dependent stands on
            # its own (stores need their master; LMCache's
            # servers only need a coordinator once P2P is on).
            if (
                spec is not None
                and spec.depends_on
                and provider is not None
                and provider.component_enabled(spec.depends_on, config_fields)
            ):
                if spec.depends_on not in addresses:
                    # Converges on the dependency's RUNNING event.
                    continue
                instance_addresses = {spec.depends_on: addresses[spec.depends_on]}
            # Whatever room the surviving rows left over is what to create.
            missing = [
                worker_id
                for worker_id in sorted(layout)
                for _ in range(room.get((component, worker_id), 0))
            ]
            for worker_id in missing:
                # Same display-name convention as model instances: the
                # parent's name (as of instance creation; a later service
                # rename does not rename instances), the component role
                # when there is one, and a short random suffix.
                name_suffix = ''.join(
                    random.choices(string.ascii_lowercase + string.digits, k=5)
                )
                name_role = f"-{component}" if component else ""
                await CacheServiceInstance.create(
                    session,
                    CacheServiceInstanceCreate(
                        name=f"{service.name}{name_role}-{name_suffix}",
                        cache_service_id=service.id,
                        worker_id=worker_id,
                        cluster_id=service.cluster_id,
                        component=component,
                        component_addresses=instance_addresses,
                        state=CacheServiceStateEnum.PENDING,
                        spec_digest=cache_service_spec_digest(service),
                    ),
                )
                logger.info(
                    f"Created instance of cache service {service.name}"
                    f"{name_role} on worker {worker_id}"
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
        await self._sync_service_aggregate(session, service, provider)

    async def _component_addresses(
        self,
        session: AsyncSession,
        provider,
        instances: List[CacheServiceInstance],
        config_fields: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """The address of every depended-on component that is up: the one
        RUNNING instance's host:port, or the component's rendered
        address_template where it declares one (an HA master pool is
        reached through its coordination backend, not through whichever
        replica answered first). A dependent still waits for an instance
        to run either way — the pool has to exist before it is useful."""
        if provider is None or not provider.components:
            return {}
        depended = {
            spec.depends_on for spec in provider.components.values() if spec.depends_on
        }
        resolved_fields = resolved_field_values(provider.fields, config_fields or {})
        addresses: Dict[str, str] = {}
        for name in depended:
            # Lowest worker id, not whatever the query returned first: the
            # address is stamped on every dependent, and a different pick
            # between two passes reads as the dependency having moved — which
            # deletes and recreates them all while the pool is still up.
            running = sorted(
                (
                    candidate
                    for candidate in instances
                    if (candidate.component or "") == name
                    and candidate.state == CacheServiceStateEnum.RUNNING
                    and candidate.port
                ),
                key=lambda candidate: (candidate.worker_id or 0, candidate.id or 0),
            )
            instance = running[0] if running else None
            if instance is None:
                continue
            spec = provider.get_component(name)
            templated = render_optional_template(
                spec.address_template if spec else None, resolved_fields
            )
            if templated:
                addresses[name] = templated
                continue
            worker = await Worker.one_by_id(session, instance.worker_id)
            if worker is None or not worker.ip:
                continue
            addresses[name] = f"{worker.ip}:{instance.port}"
        return addresses

    async def _desired_component_workers(
        self,
        session: AsyncSession,
        service: CacheService,
        instances: List[CacheServiceInstance],
        provider: Optional[CacheProvider] = None,
    ) -> Tuple[Dict[str, Dict[int, int]], Optional[str], bool]:
        """How many instances each provider component should have on each
        worker ("" keys the sole component of single-component
        providers), an error message when a desired layout is
        unsatisfiable, and whether the instance rows should still be
        reconciled. A per_node selector matching nothing is
        an authoritative empty layout — labels change only by explicit
        edits, so the instances follow (the selector can scale the
        service to zero) and the service parks in ERROR to say why. A
        replicas component is scheduler-placed: an explicit service
        worker_id pins one replica, the rest stay sticky to where they
        already run and spread over the matching workers by lowest id,
        one per worker — a cluster smaller than the replica count runs
        what fits rather than stacking two on a node. A pinned worker
        vanishing is a fault that parks the service without touching its
        rows (the user chose it); an auto-placed replica just moves (pool
        reset — the provider self-heals by remounting)."""
        if provider is None:
            provider = await get_cache_provider(session, service.provider_name)
        layouts = provider.component_layouts() if provider else {"": "replicas"}
        multi_component = bool(provider and provider.components)

        workers = await Worker.all_by_fields(
            session,
            fields={"cluster_id": service.cluster_id},
            extra_conditions=[Worker.deleted_at.is_(None)],
        )
        selector = service.worker_selector
        matching = [
            worker
            for worker in workers
            if not selector or label_matching(selector, worker.labels or {})
        ]
        matching_ids = {worker.id for worker in matching}

        config_fields = service.config.fields if service.config else None
        desired: Dict[str, Dict[int, int]] = {}
        for component, topology in layouts.items():
            # A component turned off by its declared field keeps no
            # instances: its desired set is empty, and the diff below
            # deletes any leftovers from before the toggle.
            if provider and not provider.component_enabled(component, config_fields):
                desired[component] = {}
                continue
            if topology == "per_node":
                if not workers:
                    return (
                        {},
                        "Cluster has no active workers to run cache instances.",
                        True,
                    )
                if not matching:
                    return (
                        {},
                        f"No workers match the worker selector: {selector}.",
                        True,
                    )
                desired[component] = {worker_id: 1 for worker_id in matching_ids}
                continue

            spec = provider.get_component(component) if provider else None
            count = _component_replica_count(spec, provider, config_fields)

            pinned: Set[int] = set()
            if service.worker_id:
                if service.worker_id in matching_ids:
                    pinned = {service.worker_id}
                elif not multi_component:
                    # The user chose this worker; its vanishing is a
                    # fault, not a narrowing.
                    return {}, "Assigned worker no longer exists.", False

            if not matching_ids:
                label = component or "cache"
                return (
                    {},
                    f"No workers available to place the '{label}' component.",
                    True,
                )

            current: Dict[int, int] = {}
            for instance in instances:
                if (instance.component or "") != component:
                    continue
                if instance.worker_id in matching_ids:
                    current[instance.worker_id] = current.get(instance.worker_id, 0) + 1

            # Sticky workers first (a moved replica resets its share of
            # the pool), then the rest by id. One replica per worker: a
            # component's instances share whatever the node holds for it
            # — a data directory, a device — so two of them on one worker
            # would collide over it. A cluster smaller than the replica
            # count therefore runs what fits rather than stacking the
            # remainder; the form warns before it comes to that.
            rotation = (
                sorted(pinned)
                + sorted(set(current) - pinned)
                + sorted(matching_ids - pinned - set(current))
            )
            desired[component] = {worker_id: 1 for worker_id in rotation[:count]}
        return desired, None, True

    async def _sync_service_aggregate(
        self,
        session: AsyncSession,
        service: CacheService,
        provider: Optional[CacheProvider] = None,
    ):
        """Fold the instances' states into the service row: all RUNNING →
        RUNNING/healthy; some RUNNING → RUNNING/unhealthy with an N/M
        breakdown; none RUNNING but a cache server already launching →
        STARTING; none launched yet → PENDING; otherwise ERROR.

        ``provider`` is the declaration the caller already resolved; reading
        the catalog is a query, and a reconcile pass would otherwise repeat it
        for the same service."""
        instances = await CacheServiceInstance.all_by_fields(
            session, {"cache_service_id": service.id}
        )

        if provider is None:
            provider = await get_cache_provider(session, service.provider_name)
        config_fields = service.config.fields if service.config else None
        components = (
            [
                name
                for name in provider.components
                if provider.component_enabled(name, config_fields)
            ]
            if provider
            else []
        )
        if provider and provider.components:
            # A component turned off still has its rows until the next
            # reconcile deletes them, and this aggregate also runs straight off
            # an instance event. Counting them would let an intentionally
            # disabled component's leftovers park the service in ERROR, or hold
            # it unhealthy for a container that is on its way out.
            enabled = set(components)
            instances = [
                instance
                for instance in instances
                if (instance.component or "") in enabled
            ]

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
        if components:
            # Multi-component service: available means every component
            # has at least one RUNNING instance (a distributed pool serves
            # with the master and any store up); healthy means all of
            # them are. A component with no rows yet (its creation gated
            # on a dependency) reads as pending, not as a fault.
            tallies = {name: [0, 0] for name in components}
            for instance in instances:
                tally = tallies[instance.component or ""]
                tally[1] += 1
                if instance.state == CacheServiceStateEnum.RUNNING:
                    tally[0] += 1
            breakdown = " · ".join(
                f"{name} {tally[0]}/{tally[1]}" for name, tally in tallies.items()
            )
            # A component takes one worker per replica, so a cluster with
            # fewer matching workers than replicas runs a smaller pool.
            # It serves, and says so: a count that silently stops short
            # of what was asked for reads as the pool being at size.
            short = " · ".join(
                f"{name} {tallies[name][1]}/{requested}"
                for name, requested in (
                    (
                        name,
                        _component_replica_count(
                            provider.get_component(name), provider, config_fields
                        ),
                    )
                    for name in components
                )
                if provider.component_layouts().get(name) != "per_node"
                and tallies[name][1] < requested
            )
            if all(tally[0] > 0 for tally in tallies.values()):
                healthy_all = running == total
                state, healthy, message = (
                    CacheServiceStateEnum.RUNNING,
                    healthy_all,
                    (
                        f"Fewer workers than replicas: {short}"
                        if short
                        else None if healthy_all else breakdown
                    ),
                )
            elif starting:
                state, healthy, message = CacheServiceStateEnum.STARTING, None, None
            elif (
                pending or any(tally[1] == 0 for tally in tallies.values())
            ) and not any(
                instance.state == CacheServiceStateEnum.ERROR for instance in instances
            ):
                # A component still without rows is waiting on the one it
                # depends on — unless that one failed, which is a fault to
                # report rather than a wait to keep showing.
                state, healthy, message = CacheServiceStateEnum.PENDING, None, None
            else:
                state, healthy, message = (
                    CacheServiceStateEnum.ERROR,
                    False,
                    breakdown if total else "no instances running",
                )
        elif total and running == total:
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


async def sync_replicas(session: AsyncSession, model: Model) -> Optional[float]:
    """
    Synchronize the replicas.

    Returns the seconds until this model next needs a pass with nothing else
    prompting one, or None when it does not. Only the role-bearing rule has
    such a deadline today — a drain window — so the role-less path returns
    None and behaves exactly as it did.

    Two convergence rules live behind this one name, and the switch between
    them is `model.roles`:

    - No roles: the pre-PD rule, `model.replicas` interchangeable instances.
      `_sync_replicas_legacy` below is that code unchanged.
    - Roles: convergence is per role, because `Model.replicas` stops being a
      count and becomes a 0/1 deployment switch (the counts move to
      `roles[].replicas`). Running the legacy rule on a role-bearing model is
      an *active* bug, not merely a gap: a 3P1D+router deployment is five
      instance rows against `replicas == 1`, so `5 > 1` deletes four of them
      and the victims are whichever the scale-down scorer ranks lowest.
    """

    # Re-fetch model from database to ensure we have latest state
    # (event data may be from a different session or stale)
    fresh_model = await Model.one_by_id(session, model.id)
    if not fresh_model or fresh_model.deleted_at is not None:
        return None
    model = fresh_model

    if model.roles:
        return await _sync_replicas_per_role(session, model)

    # Turning disaggregation off leaves the group's members behind, and they
    # cannot simply be handed to the role-less rule. Two reasons, and either
    # alone would be enough: that rule ranks every instance in one comparison,
    # which an eight-card prefill and a cpu_only router cannot share, so it
    # would pick a plausible-looking wrong victim; and the gateway filter keys
    # on `model.roles`, so the moment roles are gone every leftover member
    # becomes a registered upstream — and a request balanced onto a former
    # prefill returns after one token, with a 200.
    #
    # So the group is retired first and the plain replicas are built on the
    # next pass. Two passes rather than one because the deletion has to be
    # settled before anything counts what is left.
    orphans = [
        instance
        for instance in await ModelInstance.all_by_field(session, "model_id", model.id)
        if instance.role
    ]
    if orphans:
        logger.info(
            f"Model {model.name} no longer declares roles; retiring "
            f"{len(orphans)} group member(s) before rebuilding plain replicas"
        )
        await _release_and_delete(session, orphans)
        return None

    await _sync_replicas_legacy(session, model)
    return None


async def _sync_replicas_legacy(session: AsyncSession, model: Model):
    """The role-less rule, byte-for-byte what it has always been."""

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


# Spec fields that must NOT enter a generation's digest. Everything else on
# `ModelSpecBase` does, and that direction is deliberate: a field added later
# joins the digest by default, which errs toward restarting a group that did
# not need it rather than toward pairing two generations that must not meet.
# A wrong restart is visible and costs a reload; a cross-generation pair is
# silent and returns wrong answers (F7 3.3 — `max_model_len` mismatched across
# P and D handshakes fine, transfers fine, and only a long prompt reveals it,
# after prefill has already been paid for).
# Distinguishes "no override" from "override with None", which is exactly the
# case that matters here: an unset `backend_version` is a real spec value.
_UNSET = "\x00unset"

_DIGEST_EXCLUDED_SPEC_FIELDS = frozenset(
    {
        # Descriptive. Renaming the description must not restart a group.
        "description",
        "meta",
        "categories",
        # Counts, not shape. Convergence below handles them per role, and
        # folding them in would make scaling 1P1D to 2P1D a full-group
        # restart — exactly what per-role convergence exists to avoid.
        "replicas",
        "ready_replicas",
        "scaling_schedule",
        # Supervisor behaviour and routing, not container shape.
        "restart_on_error",
        "generic_proxy",
        # Mounted at run time against a running engine.
        "lora_list",
        # Placement preference for the *next* scheduling decision, not
        # container shape. Folding it in would make tightening gather a
        # full-group restart that relocates nothing — the members already
        # hold their workers, and nothing re-places a running group. The
        # deployment form says so in as many words ("only affects later
        # scheduling; running groups are not moved"), and a digest bump would
        # make that sentence a lie.
        "gather",
    }
)


def _role_digest_payload(role: RoleSpec) -> Dict[str, Any]:
    """A role's contribution to the digest, minus its replica count.

    Same reason `replicas` is excluded at the model level: a role's count is
    what per-role convergence adjusts, so folding it in would turn every
    scale into a generation change.
    """
    payload = role.model_dump(mode="json", exclude_none=True)
    payload.pop("replicas", None)
    return payload


async def _instance_type_snapshots(
    session: AsyncSession, model: Model
) -> Dict[str, Optional[str]]:
    """The InstanceType identity snapshot behind every type name the model
    selects, keyed by name.

    A `gpu_type_selector` records only the type's *name*,
    while the catalog behind that name is versioned by retire-and-insert — so
    two members admitted at different moments can resolve one name to
    different card specs. Without this in the digest that drift is invisible:
    the group looks like one generation and is two.

    An unresolvable name maps to None rather than being dropped, so "the type
    is gone" is itself a digest input.
    """
    names = set()
    for role in model.roles or []:
        selector = role.gpu_type_selector or model.gpu_type_selector
        if selector is not None and selector.type:
            names.add(selector.type)
    if not names and model.gpu_type_selector and model.gpu_type_selector.type:
        names.add(model.gpu_type_selector.type)

    snapshots: Dict[str, Optional[str]] = {}
    for name in sorted(names):
        matched = await GPUInstanceType.all_by_fields(
            session,
            fields={
                "cluster_id": model.cluster_id,
                "deleted_at": None,
                "name": name,
            },
        )
        snapshots[name] = matched[0].snapshot if matched else None
    return snapshots


async def model_spec_digest(
    session: AsyncSession,
    model: Model,
    backend_version: Optional[str] = _UNSET,
) -> str:
    """The generation identity of `model`'s deployment shape.

    Shaped after `GPUInstanceType.compute_snapshot`: a content
    hash over the definitional spec with the mutable description fields
    excluded, so an unchanged spec keeps its digest across restarts and a
    changed one produces a new generation.

    `backend_version` substitutes for the model's own, and exists for one
    caller: asking whether a *particular member* is out of date with the spec.
    See `_stale_members`.
    """
    payload: Dict[str, Any] = {}
    for field in ModelSpecBase.model_fields:
        if field in _DIGEST_EXCLUDED_SPEC_FIELDS:
            continue
        value = getattr(model, field, None)
        if field == "backend_version" and backend_version is not _UNSET:
            value = backend_version
        if field == "roles":
            value = [_role_digest_payload(role) for role in (value or [])] or None
        elif isinstance(value, BaseModel):
            value = value.model_dump(mode="json", exclude_none=True)
        elif isinstance(value, list):
            value = [
                (
                    item.model_dump(mode="json", exclude_none=True)
                    if isinstance(item, BaseModel)
                    else item
                )
                for item in value
            ]
        payload[field] = value

    payload["_instance_types"] = await _instance_type_snapshots(session, model)

    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return f"sha1:{hashlib.sha1(blob.encode('utf-8')).hexdigest()}"


async def _stale_members(
    session: AsyncSession,
    model: Model,
    instances: Sequence[ModelInstance],
) -> Optional[bool]:
    """Whether any running member predates the config it is shown with.

    None where nothing can be said: a member created before `spec_digest`
    existed carries None, and reading that as "differs" would mark every
    pre-upgrade model stale on the first pass after an upgrade.

    One exemption, and it is not a loophole -- it is the difference between
    a config change and a record of what is already running.

    The case it exists for is a model deployed with no version pinned. Once a
    member starts, the worker reads the engine's version off it and writes it
    back to the Model. That write is deliberate: without it a later replica
    resolves its own, newer build and the group goes heterogeneous. But
    `backend_version` is in the digest -- also correctly, since a different
    engine build needs a new container -- so the digest changes and every
    running member reads as stale seconds after it started, with nobody having
    edited anything. Without the exemption the banner would tell the user to
    restart a group in order to adopt a value read off that very group, and
    restarting does clear it, so the advice appears to work and the reading is
    never questioned.

    So a member is excused when both hold:

    - its stamp matches the spec with `backend_version` unset. Not a guess at
      the old value -- the write-back only fires when the field was falsy, so
      unset is precisely what the member was stamped against.
    - the version now recorded is the one the member is actually running.
      Without this, pinning 0.5.14 onto a group running 0.5.15 would also be
      excused, and that edit genuinely needs a restart. A member that cannot
      say what it runs stays stale, which is the conservative direction.

    The stamps are deliberately *not* rewritten to match. `group_id` is derived
    from the digest and is what the router matches its peers on, so re-stamping
    would rename a running group's generation underneath it.
    """
    digested = [i for i in instances if i.spec_digest]
    if not digested:
        return None

    current = await model_spec_digest(session, model)
    if all(i.spec_digest == current for i in digested):
        return False

    unpinned = await model_spec_digest(session, model, backend_version=None)
    recorded = model.backend_version

    def is_current(instance: ModelInstance) -> bool:
        if instance.spec_digest == current:
            return True
        return (
            instance.spec_digest == unpinned
            and recorded is not None
            and instance.backend_version == recorded
        )

    return not all(is_current(i) for i in digested)


def _generation_group_id(model: Model, digest: str) -> str:
    """One `group_id` is one generation, and a generation is one digest.

    Scoped by model id so the value is unique fleet-wide: `group_id` is what
    the router matches its peers on, and two models that happen to share a
    spec must not be able to resolve each other's members.
    """
    return f"{model.id}-{digest.split(':')[-1][:16]}"


def _gpu_roles(model: Model) -> List[RoleSpec]:
    """Every role that occupies accelerators — that is, everything but the
    router.

    This is the set that forms atomically and the set Kueue's
    `pod-group-total-count` counts: a 4P4D is 8, not 9.
    """
    return [
        role for role in (model.roles or []) if role.name != RoleNameEnum.ROUTER.value
    ]


def _role_dependencies(model: Model, role: RoleSpec) -> List[str]:
    """Roles that must have a ready member before `role` may be created.

    An explicit `dependencies` wins. Absent one, the router depends on every
    GPU role — and that default is load-bearing rather than a convenience:
    the router's command line is rendered from its peers' `ip:port`, ports are
    assigned worker-side at start, so a router created alongside its peers has
    nothing to render (F4 3.6, F3 3.4 ④). Creating it early does not merely
    produce a slower start, it produces a router pointed at nothing.
    """
    if role.dependencies is not None:
        return list(role.dependencies)
    if role.name == RoleNameEnum.ROUTER.value:
        return [gpu_role.name for gpu_role in _gpu_roles(model)]
    return []


def _dependencies_ready(
    model: Model, role: RoleSpec, members: List[ModelInstance]
) -> bool:
    """Whether every role `role` depends on has at least one RUNNING member.

    RUNNING rather than merely created, because what the dependent needs is
    the *address*, and an instance only has one once its worker has assigned
    ports and started it.
    """
    required = _role_dependencies(model, role)
    if not required:
        return True
    running = {
        member.role
        for member in members
        if member.state == ModelInstanceStateEnum.RUNNING and member.role
    }
    return all(name in running for name in required)


async def _build_instance_create(
    session: AsyncSession,
    model: Model,
    role: RoleSpec,
    group_id: str,
    digest: str,
) -> ModelInstanceCreate:
    """One member row of `role` in the generation `group_id`.

    Everything outside the four PD columns is what `_sync_replicas_legacy`
    builds, deliberately: a role's overrides are applied by the read-path
    projection (`role_effective_model`), never written here, so that one
    intent keeps one source of truth.

    The exception is a column whose value is a *decision made once, at
    creation*, and which the read path therefore cannot revisit — `backend`,
    which picks the image, and `draft_model_source`, which picks the weights to
    download. Those are resolved against the role-effective model here because
    there is nowhere later to do it.
    """
    name_prefix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
    # Everything decided from a per-role override is decided from this, not
    # from `model`. Projecting only some of the overrides is what admits a
    # group that configures a draft model on decode alone: the engine argument
    # names the draft model while no draft weights reach the machine. MTP hides
    # that — its head travels inside the main weights — so only an eagle3 or an
    # external draft model shows it.
    effective = role_effective_model(model, role.name)
    return ModelInstanceCreate(
        name=f"{model.name}-{role.name}-{name_prefix}",
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
        owner_principal_id=model.owner_principal_id,
        draft_model_source=await get_draft_model_source(session, effective),
        # The backend is a per-role override, so it is resolved against the
        # role-effective model rather than the Model — a `custom` group may
        # legitimately mix engines.
        backend=get_backend(effective),
        backend_version=role.backend_version or model.backend_version,
        role=role.name,
        group_id=group_id,
        spec_digest=digest,
    )


async def _release_and_delete(
    session: AsyncSession, instances: List[ModelInstance]
) -> List[str]:
    """The single exit for every member deletion — scale-down, generation
    teardown and full teardown all pass through here.

    The Kueue finalizer is NOT yet applied here. Deleting a member of an
    admitted pod-group without marking `kueue.x-k8s.io/retriable-in-group:
    "false"` leaves the Pod in Terminating and the Workload holding its quota.
    That marking belongs to the k8s deployment path, which has no PD wiring
    yet; collecting the deletions behind one function now is what makes adding
    it a one-place change rather than three.
    """
    if not instances:
        return []
    names = await ModelInstanceService(session).batch_delete(instances)
    if names:
        # INFO because this is the other half of the `Formed group` line. A
        # group that loses every member re-forms from scratch, and the solver
        # is free to answer somewhere else -- so a group can move machines
        # with nothing on the model row to show for it. Neither `stale` nor a
        # degradation can carry that: one follows the spec digest, which did
        # not change, and the other reads current state, which looks correct
        # once the rebuild lands. The two log lines are the whole account
        # there is, and they are only an account if both are visible.
        logger.info(f"Deleted model instances: {names}")
    return names


async def _sync_replicas_per_role(
    session: AsyncSession, model: Model
) -> Optional[float]:
    """Converge a role-bearing model, one role at a time.

    Per role rather than per group because the group is not the unit of
    change: turning a 1P1D into a 2P1D under group semantics would mean
    deleting the group and recreating it — a full outage to add one prefill.

    Returns the seconds until the earliest drain window closes, or None when
    nothing is draining. The caller uses it to book the pass that reaps them:
    see `_next_drain_due`.
    """
    instances = await ModelInstance.all_by_field(session, "model_id", model.id)

    if model.replicas == 0:
        # `Model.replicas` is a deployment switch for a role-bearing model,
        # so zero means the whole group is parked, not "zero of each role".
        await _release_and_delete(session, instances)
        return None

    digest = await model_spec_digest(session, model)

    # Turning disaggregation ON leaves the previous single-role deployment's
    # instances behind, and they cannot join a generation: they carry no role,
    # so nothing counts them toward any role's tally, and the gateway filter —
    # which registers only a group's router — has already stopped routing to
    # them. What is left is a member of nothing that still holds its GPUs, and
    # holding them is not passive: it is what keeps the new group's decode
    # from being schedulable. Observed exactly that way on a two-card host.
    #
    # Retired in the same pass rather than left for an explicit restart,
    # because enabling disaggregation is the most complete generation change
    # there is — the deployment's shape, not its parameters — and the new
    # generation is already being formed below.
    orphans = [i for i in instances if not i.group_id]
    if orphans:
        logger.info(
            f"Model {model.name} now declares roles; retiring "
            f"{len(orphans)} instance(s) that predate the group"
        )
        await _release_and_delete(session, orphans)
        instances = [i for i in instances if i.group_id]

    # The live members define the current generation, not the model's present
    # digest. A spec edit makes the running members stale; it does not by
    # itself retire them — F7 3.3 requires the switch to be an explicit
    # all-stop-then-all-start, because restarting members one at a time is
    # precisely how a cross-generation pair is produced. So new members join
    # the generation their peers are already in.
    members = [i for i in instances if i.group_id]
    if members:
        group_id = max(
            {i.group_id for i in members},
            key=lambda gid: (
                len([i for i in members if i.group_id == gid]),
                max(i.created_at for i in members if i.group_id == gid),
            ),
        )
        generation = [i for i in members if i.group_id == group_id]
        generation_digest = next(
            (i.spec_digest for i in generation if i.spec_digest), digest
        )
    else:
        group_id = _generation_group_id(model, digest)
        generation = []
        generation_digest = digest

    if not generation:
        # First formation is atomic and contains ONLY the GPU roles. Two
        # reasons, and they point the same way: Kueue's pod-group admission
        # counts members against a declared total, so a group whose rows
        # appear in batches is repeatedly judged incomplete; and the
        # router cannot be in this transaction at all, since it has no peers
        # to render yet (see `_role_dependencies`).
        pending = []
        for role in _gpu_roles(model):
            for _ in range(role.replicas):
                pending.append(
                    await _build_instance_create(
                        session, model, role, group_id, generation_digest
                    )
                )
        if pending:
            await ModelInstanceService(session).batch_create(pending)
            # INFO rather than debug: a group forming is how a group also
            # re-forms. Nothing on the model row records that its members were
            # torn down and rebuilt elsewhere — not `stale`, which follows the
            # spec digest, and not a degradation, which reads current state —
            # so when an operator asks why a running group moved machines,
            # this line is the only first-hand evidence there is.
            logger.info(
                f"Formed group {group_id} for model {model.name} "
                f"with {len(pending)} members"
            )
            generation = await ModelInstance.all_by_field(session, "model_id", model.id)
            generation = [i for i in generation if i.group_id == group_id]
        # Deliberately no early return: the router still has to be considered
        # below, and it becomes creatable the moment its dependencies report
        # ready — which may already be true on a later pass.

    # Before the per-role arithmetic, not after: a member whose window has
    # passed is gone as far as the ratio is concerned, and leaving it in the
    # count for one more pass would make the role look satisfied and stop the
    # replacement that a re-scale-up is waiting for.
    generation = await _reap_drained(session, generation)

    for role in model.roles:
        have = [i for i in generation if i.role == role.name]
        if len(have) < role.replicas:
            if not _dependencies_ready(model, role, generation):
                continue
            pending = [
                await _build_instance_create(
                    session, model, role, group_id, generation_digest
                )
                for _ in range(role.replicas - len(have))
            ]
            await ModelInstanceService(session).batch_create(pending)
            logger.debug(
                f"Created {len(pending)} {role.name} instance(s) for "
                f"model {model.name} in group {group_id}"
            )
        elif len(have) > role.replicas:
            await _scale_down_role(session, model, role, have, generation)

    return _next_drain_due(generation)


def _next_drain_due(instances: List[ModelInstance]) -> Optional[float]:
    """Seconds until the earliest drain window closes, or None if none is open.

    **Without this the drain has no second half.** `_reap_drained` runs off
    the reconcile loop, and this controller is purely event-driven — nothing
    ticks. A member marked for drain changes no field that `sync_model_status`
    publishes, so the mark itself produces no further Model event, and a
    deployment that has settled produces none either. The window would then
    expire against a pass that never comes: the member stays out of the
    router's registry, serving nothing, holding its accelerators, for as long
    as the deployment goes unedited.

    Floored at zero rather than clamped away, so a window that already elapsed
    (a server that was down through it) books an immediate pass instead of a
    negative delay the queue would read as "now, unconditionally".
    """
    now = datetime.now(timezone.utc)
    window = envs.SCHEDULER_DRAIN_WINDOW_SECONDS
    deadlines = []
    for instance in instances:
        since = instance.draining_since
        if since is None:
            continue
        if since.tzinfo is None:
            since = since.replace(tzinfo=timezone.utc)
        deadlines.append(max(0.0, window - (now - since).total_seconds()))
    return min(deadlines) if deadlines else None


async def _scale_down_role(
    session: AsyncSession,
    model: Model,
    role: RoleSpec,
    have: List[ModelInstance],
    generation: Sequence[ModelInstance],
):
    """Take this role's surplus members out of rotation.

    The excess is measured against the ROLE's count, not the model's. The
    pre-PD line was `len(candidates) - model.replicas`, which assumed
    `candidates` held every instance of the model; scoped to one role of a
    4P4D that arithmetic deletes eight instances in one pass.

    **Marks rather than deletes.** Deleting a prefill outright drops the KV
    blocks the decodes are still fetching, and the engine has no shutdown that
    waits for them. `_reap_drained` deletes it once the window has passed;
    until then the member is out of the router's registry and still running.

    `generation` is the whole group, not just this role: choosing which prefill
    to drop is a question about the decodes beside it, and this role's members
    cannot answer it (`PairingRetentionScorer`).
    """
    draining = [i for i in have if i.draining_since is not None]
    excess = len(have) - len(draining) - role.replicas
    if excess <= 0:
        # Already enough on their way out. Picking a second victim while the
        # first is still draining is how a burst of reconciles scales a role
        # to zero one window at a time — the surplus has been acted on, it
        # just has not finished.
        return

    keep = [i for i in have if i.draining_since is None]
    candidates = await find_scale_down_candidates(keep, model, peers=generation)
    if not candidates:
        # `find_scale_down_candidates` returns [] on its internal exception,
        # so an empty result is indistinguishable from a scoring failure.
        # Not deleting is the fail-safe reading of that ambiguity.
        return

    if not _soft_scale_down_enabled(model):
        await _release_and_delete(
            session, [c.model_instance for c in candidates[:excess]]
        )
        return

    now = datetime.now(timezone.utc)
    for candidate in candidates[:excess]:
        instance = candidate.model_instance
        instance.draining_since = now
        await instance.update(session)
        logger.info(
            f"Draining {instance.name} (role '{role.name}') of model "
            f"{model.name}; it leaves the router now and is deleted in "
            f"{envs.SCHEDULER_DRAIN_WINDOW_SECONDS}s"
        )


def _soft_scale_down_enabled(model: Model) -> bool:
    """Only a group drains, and only when a window is configured.

    A role-less model has no router registry to be removed from, so the wait
    would be a wait with nothing happening during it — the member would keep
    taking new requests for the whole window and then vanish mid-request,
    which is strictly worse than deleting it now.
    """
    return bool(model.roles) and envs.SCHEDULER_DRAIN_WINDOW_SECONDS > 0


async def _reap_drained(
    session: AsyncSession, instances: List[ModelInstance]
) -> List[ModelInstance]:
    """Delete the members whose drain window has passed; return the rest.

    Returns the survivors rather than the reaped so the caller can assign
    straight through. What it must not do is hand back a list still containing
    a deleted member: the per-role arithmetic that follows would count it,
    find the role satisfied, and skip creating the replacement a re-scale-up
    is waiting for.

    Runs off the reconcile loop rather than a timer: a timer would have to be
    re-armed on restart for every draining member, and the row already says
    when each window ends.
    """
    window = envs.SCHEDULER_DRAIN_WINDOW_SECONDS
    if window <= 0:
        return list(instances)
    now = datetime.now(timezone.utc)
    due = []
    for instance in instances:
        since = instance.draining_since
        if since is None:
            continue
        if since.tzinfo is None:
            # `UTCDateTime` puts the zone back on the way out, so this is not
            # the path a stored row takes. Kept for a value that reached here
            # without passing through the column — it was written as UTC either
            # way, and the alternative is a TypeError on the subtraction below.
            since = since.replace(tzinfo=timezone.utc)
        if (now - since).total_seconds() >= window:
            due.append(instance)
    if not due:
        return list(instances)
    logger.info(
        f"Drain window elapsed for {len(due)} member(s): "
        f"{', '.join(i.name for i in due)}"
    )
    await _release_and_delete(session, due)
    return [i for i in instances if i not in due]


async def cancel_drain(session: AsyncSession, instance: ModelInstance):
    """Put a draining member back into rotation.

    One field: the next membership reconcile sees an ordinary RUNNING member and
    re-registers its address. Nothing has to be restarted because nothing was
    stopped — which is what would make a wrong scale-down recoverable rather
    than merely regrettable, *if* anything called this.

    **GPUStack ships no rollback, and this is not one.** There is no rollback
    feature anywhere in the product — not for a scale-down, not automatic, not
    manual — and this function does not add one: it is the mechanism a rollback
    would be built on, with no trigger attached. Read it as "the state is
    reversible in principle", never as "the operator can reverse it".

    No automatic trigger yet, and deliberately not one invented here. The
    obvious one — "roll back if the group's TTFT degrades during the window" —
    needs a threshold, and a threshold picked without data would fire on
    ordinary load variance and undo correct scale-downs. What exists is the
    mechanism and the audit trail; the judgement stays with the operator until
    there are measurements to set it from.

    **No caller in production code, and that is the shipped decision — not a
    route someone forgot to wire.** Nothing but the tests reaches this: there is
    no API endpoint, no UI, no controller path. The window is 60s by default,
    which is not long enough for a human to notice a mis-scale, find the member
    and press something; and the instance list does not surface
    `draining_since`, so such a button would be pressed blind.

    **Putting `replicas` back is not a rollback either.** It recovers the
    *ratio*: the anti-flap arithmetic in `_scale_down_role` will not pick a second victim,
    and the pass that reaps the drained member creates the replacement in the
    same reconcile. But it does not recover the member — that one is still
    deleted on schedule and the replacement is a cold start, which for a PD role
    means loading weights again. Re-scaling up is the only thing an operator can
    do; it is not undo.

    Keep this callable: when a data-backed threshold exists, the trigger
    attaches here.
    """
    instance.draining_since = None
    await instance.update(session)
    logger.info(f"Cancelled the drain of {instance.name}; it rejoins the router")


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
    peers: Optional[Sequence[ModelInstance]] = None,
    status_max_score: Optional[float] = None,
    offload_max_score: Optional[float] = None,
    placement_max_score: Optional[float] = None,
    pairing_max_score: Optional[float] = None,
    total_max_score: Optional[float] = None,
) -> List[ModelInstanceScore]:
    """Rank this role's members worst-first, so the caller can take from the front.

    `peers` is the generation the candidates belong to — every role, not just
    theirs. It is what `PairingRetentionScorer` counts the opposite role from,
    and omitting it drops that scorer: the candidate list itself is single-role
    by the check below, so it can never answer "how many decodes share this
    prefill's worker". A role-less model has no peers and no pairing scorer.
    """
    roles = {instance.role for instance in instances}
    if len(roles) > 1:
        # This is a selector WITHIN a comparable set, not across one. Its
        # `PlacementScorer` reads `_get_worker_model_instance_count()`, which
        # aggregates per worker by `model_id` — so a prefill holding eight
        # cards and a `cpu_only` router land in one distribution and their
        # scores are not comparable. Ranking them together does not fail, it
        # picks a plausible-looking wrong victim, which is the failure mode
        # PD is least able to absorb -- a PD failure must be a hard one.
        raise ValueError(
            "find_scale_down_candidates requires instances of a single role, "
            f"got {sorted(str(r) for r in roles)}"
        )
    try:
        if status_max_score is None:
            status_max_score = envs.SCHEDULER_SCALE_DOWN_STATUS_MAX_SCORE
        if offload_max_score is None:
            offload_max_score = envs.SCHEDULER_SCALE_DOWN_OFFLOAD_MAX_SCORE
        if placement_max_score is None:
            placement_max_score = envs.SCHEDULER_SCALE_DOWN_PLACEMENT_MAX_SCORE
        if pairing_max_score is None:
            pairing_max_score = envs.SCHEDULER_SCALE_DOWN_PAIRING_MAX_SCORE

        scorers: List[ModelInstanceScorer] = [
            StatusScorer(model, max_score=status_max_score),
            OffloadLayerScorer(model, max_score=offload_max_score),
        ]

        # One generation, so one group id — more than one means the caller
        # mixed generations, and a `d_j` counted across two of them would rank
        # a member by peers it can never pair with. Nothing is deleted on that
        # reading; the scorer is simply left off.
        group_ids = {i.group_id for i in instances if i.group_id}
        if peers is not None and len(group_ids) == 1:
            scorers.append(
                PairingRetentionScorer(
                    next(iter(group_ids)),
                    next(iter(roles)),
                    peers,
                    max_score=pairing_max_score,
                )
            )

        scorers.append(
            PlacementScorer(
                model,
                instances,
                scale_type=ScaleTypeEnum.SCALE_DOWN,
                max_score=placement_max_score,
            )
        )

        chain = ModelInstanceScoreChain(
            scorers=scorers,
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


def upstream_registration_ready(model: Model) -> bool:
    """The second half of the RUNNING predicate: whether the group's router
    upstream is registered.

    The predicate has exactly one definition::

        Model.state == RUNNING  <=>  every role has >=1 ready
                                     AND the upstream registration succeeded

    Registration is a *precondition* of RUNNING rather than a consequence of
    it — deriving "group ready, therefore register" the other way round is a
    self-cycle. Every role gets a ready member, the server then tries to
    register the router upstream, and only a successful registration flips the
    group to RUNNING.

    A model with no router role has no upstream to register, so the predicate
    is vacuously true: a plain deployment and a role-less model are servable
    as soon as their members are ready. This seam is the one place the
    other half is decided — when the router registration step lands it
    reports its recorded outcome here, and a group whose registration failed
    stays PARTIAL with a message instead of silently claiming to serve.
    """
    has_router = any(
        role.name == RoleNameEnum.ROUTER.value for role in (model.roles or [])
    )
    if not has_router:
        return True
    outcome = membership_outcome_for(model.id)
    if outcome is None:
        # No attempt recorded yet, and the answer is "servable" rather than
        # "not yet" on purpose. A recipe that does not launch `--enable-igw`
        # has no membership step at all: the router already knows its peers
        # from the command line, so parking such a group in PARTIAL would
        # break every deployment that works today.
        #
        # Under igw the reconcile runs on the same pass that computes this, so
        # an unrecorded outcome there is the first pass only.
        return True
    return outcome.ok


def is_model_servable(model: Model) -> bool:
    """Whether requests may be routed to this model.

    This is the boolean servability gate, and `Model.state` is what it reads
    rather than `ready_replicas > 0`: under PD a count does not imply
    servability (3P1D with the router still down is four RUNNING instances and
    zero service). `ModelRouteTarget.state` is the only place
    in this repo that evaluates it; `ModelRoute.ready_targets`, `/v1/models`
    and `resolve_route_targets` all derive from that target state, so they
    follow from this one predicate.

    One predicate, no per-shape special case: RUNNING means servable and
    nothing else does. That holds because `state` was defined to answer
    exactly this question and running-but-worse-than-asked-for is expressed
    beside it, in `degradations`, rather than inside it — a role-less model
    with 2 of 3 replicas up is RUNNING with `ratio_unmet`, not PARTIAL. So
    PARTIAL keeps a single meaning everywhere: members are up and the
    deployment still cannot serve, which for a group is a role at zero and
    for a role-less model cannot happen at all.

    For a role-less model this is equivalent to `ready_replicas > 0`, which
    `derive_model_state` is what makes true.
    """
    if model.state is None:
        # A row carries NULL between its creation and the first
        # `sync_model_status` pass over it. The migration backfills existing
        # rows, so this is not about the upgrade — it is about newly created
        # models, and about any row a reconcile has not reached yet. For those
        # the replica counter gives the same answer.
        return model.ready_replicas > 0
    return model.state == ModelStateEnum.RUNNING


def derive_route_target_state(
    target: ModelRouteTarget, model: Optional[Model]
) -> TargetStateEnum:
    """The state a target should be in, from what it points at.

    A pure function of the target and its model, so the answer can be
    recomputed at any time from rows that are already loaded — which is what
    makes the state correctable rather than only updatable.

    A provider target is always ACTIVE: its availability belongs to the
    provider, not to us. A target that points at neither a model nor a
    provider is UNAVAILABLE rather than left as it was; the previous code left
    that case's variable unbound.
    """
    if target.provider_id is not None:
        return TargetStateEnum.ACTIVE
    if target.model_id is not None and model is not None:
        return (
            TargetStateEnum.ACTIVE
            if is_model_servable(model)
            else TargetStateEnum.UNAVAILABLE
        )
    return TargetStateEnum.UNAVAILABLE


async def reconcile_route_target_states(session: AsyncSession, model: Model) -> bool:
    """Bring this model's route targets in line with its servability.

    Level-triggered on purpose, and this is the point of the function.
    Transitions alone are not enough: `notify_model_route_target` publishes
    when `state` / `ready_replicas` / `replicas` change and
    `ModelRouteTargetController` reacts, which works right up until the
    transition and its consumer do not overlap in time. The bus does not
    replay, so a transition published while the controller was not subscribed
    is lost, and edge-triggered code never re-derives the answer.

    A worker that goes unreachable and recovers while the `modelroutetarget`
    subscription is being re-established loses its recovery event: the model
    is back at RUNNING with nothing left to transition, so the target sits
    UNAVAILABLE for good — `/v1/models` returns an empty list while the
    deployment serves fine when addressed directly.

    Called from `sync_model_status`, which already runs on every model and
    instance event, so any subsequent event repairs a lost one. Writes only on
    a difference, like every other gate here, so the common case costs one
    comparison.

    The targets are queried, not read off `model.model_route_targets`, and
    that is what makes the repair fire at all. `sync_model_status` is always
    reached with the Model already loaded in the session -- `_reconcile` fetches
    it plainly and hands it over -- so a fetch *with*
    `selectinload(model_route_targets)` hits SQLAlchemy's identity map, returns
    that same instance and never applies the loader option. The relationship
    stays unloaded, and an unloaded collection reads as `[]` rather than
    raising: the emptiness check below would then return early on every pass,
    silently, indistinguishably from "this model has no route targets". A
    query cannot be short-circuited by an object that is already in the
    session.
    """
    targets = await ModelRouteTarget.all_by_fields(
        session, fields={"model_id": model.id, "deleted_at": None}
    )
    if not targets:
        return False

    changed = False
    for target in targets:
        desired = derive_route_target_state(target, model)
        if target.state != desired:
            logger.info(
                "Route target %s of model %s: %s -> %s (re-derived from the "
                "model's state)",
                target.name,
                model.name,
                target.state,
                desired,
            )
            target.state = desired
            await target.update(session=session, auto_commit=True)
            changed = True
    return changed


class PairingLocality(NamedTuple):
    """The locality figure and why it reads the way it does.

    Two fields rather than one because `None` was already carrying two
    meanings and is about to carry a third, and they call for different
    things on screen: "not yet" becomes a number later, "never" does not.
    A code rather than a magic value, following `status` and
    `request_count_source` on the same response -- a reader should not have
    to know that -1 means anything.
    """

    value: Optional[float]
    source: str


#: The question has not been asked yet: no group, or a role with no running
#: member. It will have an answer once the members are up.
LOCALITY_UNKNOWN = "unknown"
#: Measured from where the members landed. `value` is the figure.
LOCALITY_MEASURED = "measured"
#: The roles hold disjoint machines *and* at least one member spans more than
#: one -- so the zero is a fact about the shape of the deployment rather than
#: a placement that could have gone better. Rendering it as a verdict would
#: put a permanent degradation on exactly the deployments that need to span,
#: and it is not one an operator can act on.
LOCALITY_SPANNING = "spanning_members"


def pairing_locality(
    model: Model, instances: Sequence[ModelInstance]
) -> PairingLocality:
    """The chance a request's KV transfer stays inside one host.

    Not "how many pairs are local", because nothing pairs them: the router
    picks a prefill and a decode *independently* (`--prefill-policy
    cache_aware --decode-policy round_robin`), and topology-aware pairing is
    an explicit non-goal of this phase. So the honest figure is the
    probability that two independent picks land on the same worker:

        P(local) = Σ_w  (prefill_w / prefill_total) × (decode_w / decode_total)

    **The driver is how many MACHINES the group landed on, not how many
    replicas it has.** Mixed evenly over `m` workers this comes out at `1/m`,
    so the same 4P4D packed from four hosts onto two goes from 0.25 to 0.5,
    and onto one host to 1.0. `1/x` is simply the case `m == x` — one prefill
    and one decode per host, the most spread-out arrangement that still pairs
    at all — which is why the deploy form, knowing only `x`, can offer it as a
    floor and nothing better.

    That floor holds only while the roles stay MIXED across those hosts.
    Spread further, so a host carries one role and not the other, and this
    falls below `1/x` all the way to 0 — which the group solver's round-robin
    dealing exists to prevent, and which manual card selection still reaches.

    `value` is None when the question does not apply — not a group, or a role
    with no running member, where 0 would read as a verdict rather than as
    silence — and `source` says which kind of silence it is.
    """
    if not model.roles:
        return PairingLocality(None, LOCALITY_UNKNOWN)

    by_role: Dict[str, Dict[int, int]] = {}
    for instance in instances:
        role = instance.role
        if role not in (RoleNameEnum.PREFILL.value, RoleNameEnum.DECODE.value):
            continue
        if instance.state != ModelInstanceStateEnum.RUNNING:
            continue
        for worker_id in member_worker_ids(instance):
            by_role.setdefault(role, {})
            by_role[role][worker_id] = by_role[role].get(worker_id, 0) + 1

    prefill = by_role.get(RoleNameEnum.PREFILL.value) or {}
    decode = by_role.get(RoleNameEnum.DECODE.value) or {}
    if not prefill or not decode:
        return PairingLocality(None, LOCALITY_UNKNOWN)

    prefill_total = sum(prefill.values())
    decode_total = sum(decode.values())
    value = sum(
        (count / prefill_total) * (decode.get(worker_id, 0) / decode_total)
        for worker_id, count in prefill.items()
    )

    # **The formula does not survive a member that spans machines**, and the
    # first cut of this guard only caught the case where it happened to bottom
    # out at zero. It is wrong more widely than that.
    #
    # The sum assumes the router's two picks are independent *and* that each
    # pick lands somewhere — true while an instance is one machine. KV is
    # sharded by TP rank, so a decode rank needs particular prefill ranks'
    # shards; once the ranks of one member are spread over several machines,
    # whether a pair is local depends on the rank mapping, which this sum
    # cannot see. Worked example: prefill on {1,2} and decode on {2,3} at equal
    # TP, ranks laid out in order — every rank pair is remote, and the sum says
    # 0.25.
    #
    # Absent beats wrong, and this figure is not decoration: `pairing_remote`
    # is derived from it. So a spanning member is reported as a kind of
    # silence, whatever the arithmetic came to.
    if _spans_machines(model, instances):
        return PairingLocality(None, LOCALITY_SPANNING)
    return PairingLocality(value, LOCALITY_MEASURED)


def _spans_machines(model: Model, instances: Sequence[ModelInstance]) -> bool:
    """Whether any weight-bearing member occupies more than one worker.

    Read through `member_worker_ids`, the one place that knows a member is not
    only the machine its row is filed under.
    """
    from gpustack.schemas.models import role_takes_no_accelerator

    for instance in instances:
        if role_takes_no_accelerator(model, instance.role):
            continue
        if len(member_worker_ids(instance)) > 1:
            return True
    return False


async def _gather_unmet(
    session: AsyncSession, model: Model, instances: Sequence[ModelInstance]
) -> bool:
    """Whether the group landed looser than the layer it asked for.

    Under `PreferGather` that pair means "aim for this, ship it either way",
    and without an answer afterwards the ask is recorded in the spec while the
    outcome is recorded nowhere.

    **`MustGather` reaches here too.** Admission refuses only the formation,
    which goes through the solver; a scaled-out member and the router are
    placed by the per-instance path, so excluding `MustGather` here would
    leave the one strategy whose point is strictness with no report at all.
    `GatherFloorFilter` is the enforcement; this is what
    catches the cases the filter deliberately declines to force — members
    already spread across domains, or sitting in the unclassified bucket, where
    refusing a new member would not put back a floor that is already gone.
    Under `MustGather` this should therefore always be false, which makes it an
    invariant check rather than a report.

    Computed from where the members actually are, not from what the solver
    decided. The solver's verdict is not kept, and it would go stale anyway:
    a rescheduled member can loosen a group that was placed tightly, and this
    runs on every reconcile.
    """
    from gpustack.schemas.models import GatherStrategyEnum, role_takes_no_accelerator
    from gpustack.topology.tree import ROOT_LAYER, common_layer, order_layers
    from gpustack.topology.view import build_view

    gather = getattr(model, "gather", None)
    layer = getattr(gather, "layer", None)
    strategy = getattr(gather, "strategy", None)
    if not layer or strategy not in (
        GatherStrategyEnum.PREFER_GATHER,
        GatherStrategyEnum.MUST_GATHER,
    ):
        return False

    # Accelerator-bearing members only, which for today's shapes means
    # prefill and decode. The router is excluded for the same reason the
    # solver excludes it (`role_demands`): it holds no weights, so it
    # "neither competes for cards nor constrains which domain the group lands
    # in". Counting it here would report a group as having missed its target
    # because the *proxy* landed on another host — a placement the solver
    # never constrained and would make again.
    worker_ids = {
        worker_id
        for instance in instances
        if instance.state == ModelInstanceStateEnum.RUNNING
        and not role_takes_no_accelerator(model, instance.role)
        for worker_id in member_worker_ids(instance)
    }
    # One member, or none placed yet: there is no distance between members to
    # be wrong about. Silence rather than a pass — the question has not been
    # asked yet.
    if len(worker_ids) < 2:
        return False

    cluster = await Cluster.one_by_id(session, model.cluster_id)
    workers = await Worker.all_by_field(session, "cluster_id", model.cluster_id)
    try:
        view = build_view(getattr(cluster, "topology", None), workers)
    except Exception:
        # A declaration that cannot become a tree is the cluster's problem and
        # is reported there. Claiming a placement degradation off the back of
        # it would point at the wrong thing.
        return False

    placed = [
        node
        for node in _leaves_of(view.root)
        if worker_ids.intersection(node.worker_ids)
    ]
    if len(placed) < 2:
        return False

    # The tightest layer containing every member: fold pairwise and keep the
    # LOOSEST answer, since a layer holding all of them has to hold each pair.
    order = [spec.layer for spec in order_layers(view.specs)] + [ROOT_LAYER]
    rank = {name: index for index, name in enumerate(order)}
    actual = placed[0].layer
    for node in placed[1:]:
        shared = common_layer(placed[0], node) or ROOT_LAYER
        if rank.get(shared, 0) < rank.get(actual, len(order)):
            actual = shared

    # Looser means *earlier* in a root-to-leaf order.
    return rank.get(actual, 0) < rank.get(layer, len(order))


def _gather_blocked_scale_out(model: Model, instances: Sequence[ModelInstance]) -> bool:
    """Whether a member is sitting unplaced under an active `MustGather` floor.

    **The successful refusal had no reporter at all.** `GatherFloorFilter`
    does exactly what `MustGather` asks -- it drops every worker outside the
    domain the group's running members occupy, so a scaled-out member is not
    placed rather than placed elsewhere -- and the entire trace of that is one
    pending instance's `state_message`. The model stays `running` with an empty
    `degradations` list, which is the same thing it says about a scale-up that
    is merely still in flight. Measured in an e2e round: a 1P1D told to grow to
    2P sat with a pending prefill and a model row that reported nothing, and
    the only way to learn why was to open each member in turn.

    `_gather_unmet` above is not this and cannot be made into it. It fires when
    the floor has already been BROKEN, which under `MustGather` is an invariant
    check -- the case where the filter deliberately declines to force. The case
    where the filter succeeds is the common one and was silent.

    **This does not prove the floor is the cause, and must not read as if it
    did.** A cluster with no free cards anywhere presents identically: a
    weight-bearing member placed, a sibling pending, nothing moving. Separating
    the two would mean re-running the filter chain against every worker from
    here, and the answer would be stale by the time it was published. So the
    marker states what is certainly true -- a member is unplaced, and this
    deployment would rather wait than spread -- and leaves the discrimination
    to the member's own `state_message`, which names whichever filter emptied
    the list.

    Three conditions, each excluding a case that would make the marker lie:

    * `MustGather` with a layer. `PreferGather` never refuses, so an unplaced
      member there is a capacity fact with nothing to do with gather.
    * A weight-bearing member already placed. That is what makes the floor
      *active*: the filter anchors on the group's placed members, so with none
      of them placed there is no domain to be kept inside. It is also what
      excludes formation, which is refused by the solver rather than the filter
      and fails scheduling with the shortfall named -- a different report.
    * An unplaced member older than the dwell, so an ordinary scale-up does not
      wear the marker during the seconds between its row being created and the
      scheduler reaching it.
    """
    from gpustack.schemas.models import GatherStrategyEnum, role_takes_no_accelerator

    gather = getattr(model, "gather", None)
    layer = getattr(gather, "layer", None)
    if not layer or getattr(gather, "strategy", None) != GatherStrategyEnum.MUST_GATHER:
        return False

    # Weight-bearing only, and for the same reason the filter anchors on
    # those alone (`role_demands` excludes the router): a router holds no
    # weights, so it is subject to the floor without being what defines it.
    # A group whose router happened to be placed first has established no
    # domain, and treating it as an anchor would arm the marker during
    # formation -- the one case this has to stay quiet through.
    #
    # Collected as generations rather than as a yes/no, because the scope of
    # the second question follows from it: a generation being torn down and
    # rebuilt has unplaced rows of its own, and counting those would put the
    # marker on a restart -- which is a group forming again, i.e. exactly the
    # case above. An orphan carrying no `group_id` anchors nothing for the same
    # reason it is not a member of anything.
    anchored_groups = {
        instance.group_id
        for instance in instances
        if instance.worker_id is not None
        and instance.group_id is not None
        and not role_takes_no_accelerator(model, instance.role)
    }
    if not anchored_groups:
        return False

    now = datetime.now(timezone.utc)
    dwell = envs.SCHEDULER_GATHER_BLOCKED_DWELL_SECONDS
    for instance in instances:
        if instance.worker_id is not None or instance.group_id not in anchored_groups:
            continue
        created = getattr(instance, "created_at", None)
        if created is None:
            # A row with no stamp cannot be shown to have waited. Silence is
            # the honest answer -- the alternative reports every such member as
            # blocked the instant it appears, which is the flashing this dwell
            # exists to prevent.
            continue
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        if (now - created).total_seconds() >= dwell:
            return True
    return False


def _engine_version_below_recipe_floor(model: Model) -> bool:
    """Whether a pinned engine version sits under the recipe's declared floor.

    `backend_versions` in `pd-modes.yaml` was carrying no weight at all: an
    e2e round created PD models pinned to SGLang 0.5.5, vLLM 0.19.0, a
    nonexistent 9.9.9 and the string `not-a-version`, and all four were
    accepted with HTTP 200 and nothing said afterwards.

    The `>=0.5.7` on the two SGLang recipes is a correctness floor, not a
    preference. A member's id stopped being its URL and became a UUID the
    registry mints at that version (2a098200), so on an older build
    `DELETE /workers/{url}` answers 400 -- a scaled-down member stays in the
    router's registry and keeps taking traffic while GPUStack reports it gone.

    **Reported, not refused**, which is the deliberate difference from the
    cache provider's `versions` next door in `create_model`. That one rejects
    with a 400 because an out-of-range engine there receives injected args it
    cannot parse (`--shutdown-timeout`) and never starts, so refusing costs a
    deployment that was not going to run anyway. Here the group runs, and the
    number may belong to a self-built image with a private version that
    carries the fix; a 400 would break those to prevent a failure they do not
    have.

    Same fail-open as that check, and for the same reason: only a version
    `version_in_range` positively reports as OUT of range counts. None,
    unparseable and unpinned all leave the marker unset -- an exotic version
    string must never be the thing that condemns a deployment. A local version
    or a pre-release is let through on top of that, for the reason
    `_is_self_described_build` gives.
    """
    from packaging.version import Version

    from gpustack.schemas.models import role_takes_no_accelerator
    from gpustack.server.pd_mode_catalog import get_pd_mode
    from gpustack.utils.version import version_in_range

    def _is_self_described_build(pinned_version: str) -> bool:
        """Whether the number describes a build of the user's own, which the
        recipe's floor has no standing to rank.

        The self-built image this check promises not to condemn only gets
        that promise kept when its version fails to parse at all
        (`0.23.0-ascend-router-custom`, `latest`). Two forms that do parse are
        the same situation and were being marked anyway, and both say in PEP
        440's own vocabulary that the version is not the release it sorts next
        to: `+ourfix` means the official 0.5.6 with something of the packager's
        applied on top, and `0.5.7rc1` means a build handed out before 0.5.7
        exists. Either one may already carry the fix `>=0.5.7` is asking for --
        backporting it is exactly why someone cuts a `+local` -- and nothing in
        the string can say whether it does.

        So passing them is the platform admitting it cannot tell, not declaring
        them sound. The marker's claim is "below the declared floor", and a
        number that is not describing the floor's release line was never
        measured against it to begin with.

        `.dev0` lands here too, as a pre-release: packaging counts dev releases
        among them, and a build cut off someone's branch is the same
        unanswerable question an rc is.
        """
        try:
            parsed = Version(pinned_version)
        except Exception:
            # Unparseable already failed open a line below, so nothing that
            # cannot be parsed reaches this predicate. Swallowed regardless,
            # because this must never be the thing that turns a version string
            # the check has always tolerated into a raise mid-reconcile.
            return False
        return parsed.local is not None or parsed.is_prerelease

    disaggregation = getattr(model, "disaggregation", None)
    if disaggregation is None:
        return False

    mode_name = disaggregation.mode.value
    mode = get_pd_mode(mode_name)
    if mode is None or not mode.backend_versions:
        # `custom` declares no range because it injects nothing, and a recipe
        # that has not stated a floor has not claimed one. Both are "no answer",
        # never "compatible".
        return False

    # Per role, not per model, and the same expression `_build_instance_create`
    # writes onto the member row -- `role.backend_version or
    # model.backend_version` -- because that row is what actually starts. A
    # group whose model-level pin is fine can still run one decode on a build
    # that cannot be scaled down, and one is enough: that member is the one
    # that keeps serving after GPUStack believes it removed it.
    #
    # Spelled out rather than taken from `role_effective_model`, which resolves
    # the identical value: the projection validates a whole Model per role, and
    # this runs on every reconcile of every PD deployment for one string.
    #
    # **Weight-bearing roles only, and the router is the reason that is not
    # pedantry.** `backend_versions` describes the *engine* -- `>=0.5.7` is a
    # statement about SGLang -- and the router is the one role whose engine is
    # genuinely its own: the deploy form keeps an image-and-version section for
    # it precisely because "a `vllm-router` is not the model's engine". The
    # built-in recipes run it out of the model's own runner image, so its
    # version is usually the engine's and comparing them is merely redundant;
    # a hand-written router image is where it stops being redundant and starts
    # being wrong, because that version number answers a different question and
    # would condemn a deployment whose engine is perfectly in range.
    pinned = {
        getattr(role, "backend_version", None)
        or getattr(model, "backend_version", None)
        for role in (model.roles or [])
        if not role_takes_no_accelerator(model, role.name)
    }
    pinned.add(getattr(model, "backend_version", None))

    return any(
        version_in_range(version, mode.backend_versions) is False
        and not _is_self_described_build(version)
        for version in pinned
        if version
    )


def _pd_pairing_roles(model: Model) -> Tuple[Optional[RoleSpec], Optional[RoleSpec]]:
    """This deployment's prefill and decode, or two Nones for anything that is
    not a disaggregated group carrying both."""
    if getattr(model, "disaggregation", None) is None:
        return None, None
    roles = getattr(model, "roles", None) or []
    prefill = next((r for r in roles if r.name == RoleNameEnum.PREFILL.value), None)
    decode = next((r for r in roles if r.name == RoleNameEnum.DECODE.value), None)
    return prefill, decode


def _pairing_unverified(model: Model) -> bool:
    """Whether a pairing factor was declared on one role and left silent on the
    other, so admission could not judge it.

    Placement has nothing to do with this one: like
    `_engine_version_below_recipe_floor` it is a property of the spec and is
    true from the moment the group is created. What it buys is an honest answer
    to a question the user believes was already settled -- the pairing
    pre-check refuses mismatched context windows and tensor parallelisms, so a
    group that was accepted reads as a group that was checked, and until now a
    single silent role was enough to make that untrue.

    Not "the pair is wrong". `server.pd_pairing` spells out why the silent
    side's value cannot be resolved here for any of these factors, and most
    deployments this marks are correct. The claim is only that nothing verified
    them.
    """
    prefill, decode = _pd_pairing_roles(model)
    if prefill is None or decode is None:
        return False
    return bool(
        undecidable_factors(prefill, decode, getattr(model, "backend_parameters", None))
    )


def _placed_tensor_parallelism(
    model: Model, role: RoleSpec, instances: Sequence[ModelInstance]
) -> List[int]:
    """The tensor parallelism each placed member of this role actually runs.

    A declared `--tensor-parallel-size` is the answer for every member of the
    role. Absent one, a single-worker member runs the cards it was given --
    that is `get_auto_parallelism_arguments` in both backends, not a guess --
    which is precisely the number the spec could not supply at admission.

    Two shapes contribute nothing rather than a number: a member spanning
    workers, where `cal_distributed_parallelism_arguments` splits the world
    size into tp and pp further down, and a role that writes dp or pp without
    tp, which suppresses the injection entirely and falls back to the engine's
    own default.
    """
    parameters = role_parameters(role, getattr(model, "backend_parameters", None))
    declared = find_last_int_parameter(parameters, PAIRING_TP)
    if declared is None and find_last_parameter(parameters, PAIRING_ANY_PARALLELISM):
        return []

    widths: List[int] = []
    for instance in instances:
        if instance.role != role.name or instance.worker_id is None:
            continue
        if declared is not None:
            widths.append(declared)
            continue
        servers = getattr(instance, "distributed_servers", None)
        if servers is not None and getattr(servers, "subordinate_workers", None):
            continue
        cards = len(getattr(instance, "gpu_indexes", None) or [])
        if cards:
            widths.append(cards)
    return widths


def _pairing_tp_misplaced(model: Model, instances: Sequence[ModelInstance]) -> bool:
    """Whether the cards the members actually got break the recipe's
    tensor-parallel direction.

    The admission check reads the spec, and for this one factor the spec is
    routinely silent: a role that writes no parallelism and pins no cards runs
    whatever the scheduler hands it. That is the gap this closes -- the same
    rule, applied where the number finally exists.

    Compared at the extremes rather than pairwise, because the router pairs at
    random: a group is only as good as its narrowest decode against its widest
    prefill, and one member of each is enough for a transfer to land on the
    shape the connector cannot serve.
    """
    from gpustack.server.pd_mode_catalog import get_pd_mode

    prefill, decode = _pd_pairing_roles(model)
    if prefill is None or decode is None:
        return False

    disaggregation = model.disaggregation
    mode_name = disaggregation.mode.value
    rule = tensor_parallel_rule(get_pd_mode(mode_name))
    if rule == PDTensorParallelPairingEnum.ANY:
        return False

    prefill_widths = _placed_tensor_parallelism(model, prefill, instances)
    decode_widths = _placed_tensor_parallelism(model, decode, instances)
    if not prefill_widths or not decode_widths:
        return False

    if rule == PDTensorParallelPairingEnum.DECODE_GE_PREFILL:
        return violates_tensor_parallel_direction(
            rule, prefill_tp=max(prefill_widths), decode_tp=min(decode_widths)
        )
    return violates_tensor_parallel_direction(
        rule, prefill_tp=min(prefill_widths), decode_tp=max(decode_widths)
    )


def _leaves_of(node) -> List:
    """Every host node under `node`."""
    if not node:
        return []
    if not node.children:
        return [node]
    return [leaf for child in node.children for leaf in _leaves_of(child)]


def _pairing_remote(model: Model, instances: Sequence[ModelInstance]) -> bool:
    """Whether *no* request can keep its KV off the network.

    The threshold is zero, not a fraction, and that is the whole design of
    this marker.

    A partial locality is not a misconfiguration — it is the arithmetic of a
    router that pairs at random, where an evenly spread xPxD tops out at 1/x
    however well it was placed. Warning at "below some fraction" would fire on
    every correctly placed 4P4D and teach people to ignore the marker.

    Zero is different in kind: prefill and decode share no host at all, so
    every single transfer crosses the network, and it is reachable by ordinary
    manual selection — pick host A's cards for prefill and host B's for
    decode and nothing today says a word. That is the case worth a marker,
    and on a link without RDMA it is the difference between PD helping and PD
    being strictly worse than not disaggregating.
    """
    locality = pairing_locality(model, instances)
    return locality.value is not None and locality.value == 0


async def _degradation_reasons(
    session: AsyncSession,
    model: Model,
    instances: Sequence[ModelInstance],
    *,
    state: ModelStateEnum,
    ready_replicas: int,
    role_status: Optional[Dict[str, RoleStatus]],
    cache_not_injected: bool,
    cache_reason: Optional[str],
    state_message: Optional[str],
) -> Tuple[List[str], Optional[str]]:
    """Every way this deployment is up but worse than it was asked for.

    Degradations coexist with RUNNING by construction -- `derive_model_state`
    looks at neither the cache nor the ratio -- which is the whole reason they
    are a separate list rather than a state.

    `state` is taken rather than re-derived, and only the ratio reads it: the
    markers below describe placement or configuration and are true whether or
    not the deployment is serving, while "short of the shape you asked for"
    says something about the service itself.

    Returns the reasons and a possibly-extended `state_message`: one of them
    (the cache) carries a detail worth putting in front of the user, and
    threading it back is cheaper than a second pass to recover it.
    """
    reasons: List[str] = []

    if _ratio_unmet(
        model, state=state, ready_replicas=ready_replicas, role_status=role_status
    ):
        reasons.append(DegradationReasonEnum.RATIO_UNMET.value)

    if cache_not_injected:
        # A resolved cache the instance could not attach to only makes it
        # slower, so it is a marker and never a lifecycle value.
        reasons.append(DegradationReasonEnum.CACHE_NOT_INJECTED.value)
        detail = "shared cache not injected"
        if cache_reason:
            detail = f"{detail}: {cache_reason}"
        state_message = "; ".join(m for m in (state_message, detail) if m) or None

    if _pairing_remote(model, instances):
        # Placement-only, so it is knowable the moment the members are placed
        # rather than after traffic has shown it. That is the point: on a link
        # without RDMA an all-remote pairing makes PD strictly worse than not
        # disaggregating, and the user should not have to learn that from a
        # TTFT regression.
        reasons.append(DegradationReasonEnum.PAIRING_REMOTE.value)

    if await _gather_unmet(session, model, instances):
        # Says what the spec cannot: the ask is stored, the outcome was not.
        reasons.append(DegradationReasonEnum.GATHER_UNMET.value)

    if _gather_blocked_scale_out(model, instances):
        # The complement of the marker above, and the case that actually
        # happens: `GatherFloorFilter` refusing a member *successfully*. That
        # refusal is the strategy working, which is why it is a marker and not
        # an error -- but it was reported nowhere on the model, so a scale-up
        # that will never complete looked exactly like one still in flight.
        reasons.append(DegradationReasonEnum.GATHER_BLOCKED_SCALE_OUT.value)

    if _engine_version_below_recipe_floor(model):
        # Placement has nothing to do with this one: it is true of the spec
        # from the moment the group is created. What it buys is that the
        # consequence is invisible until it
        # bites -- on SGLang below 0.5.7 a scaled-down member is never removed
        # from the router's registry and goes on taking traffic, which reads as
        # a routing bug and not as a version pin.
        reasons.append(DegradationReasonEnum.ENGINE_VERSION_BELOW_RECIPE_FLOOR.value)

    if _pairing_unverified(model):
        # Spec-only, like the floor above: true from the moment the group is
        # created. It reports an absence rather than a fault -- one role
        # declared a pairing factor, the other went silent, and the silent
        # side's default is not resolvable without the checkpoint or the
        # placement. Worth saying because the pre-check refuses the mismatches
        # it CAN see, so acceptance reads as verification.
        reasons.append(DegradationReasonEnum.PAIRING_UNVERIFIED.value)

    if _pairing_tp_misplaced(model, instances):
        # The other half of the same gap, and the half that can be answered:
        # once the members are placed their cards are the tensor parallelism,
        # so the direction the recipe declares finally applies to a deployment
        # rather than to a description. Marked and not enforced -- these
        # members are already running, and taking them down to report their
        # shape would cost more than the report is worth.
        reasons.append(DegradationReasonEnum.PAIRING_TP_MISPLACED.value)

    return reasons, state_message


def _ratio_unmet(
    model: Model,
    *,
    state: ModelStateEnum,
    ready_replicas: int,
    role_status: Optional[Dict[str, RoleStatus]],
) -> bool:
    """Whether the deployment is short of the shape it was asked for.

    Reported as a degradation rather than a state, because a deployment short
    of its count **is still serving** — reduced throughput, not an outage.
    That premise is the whole meaning of the marker, and it is why the gate is
    `state == RUNNING`: RUNNING is the one predicate in this module for "can
    this serve" (`_model_is_servable`), so asking it here is asking whether
    the sentence this marker renders is true at all.

    Gated on the state rather than on a count, because a count cannot answer
    it under PD. A 1P1D whose router is down is two RUNNING instances and zero
    service: `ready_replicas` is 2, every arithmetic guard passes, and the
    marker claimed reduced capacity for a group serving nothing — beside a
    PARTIAL state saying the opposite. Not an edge case either: the router is
    created only once every GPU role has a RUNNING member
    (`_role_dependencies`), so that window opens on every PD start.

    For a group the question is per role, since the declared ratio is what
    makes a group a 3P1D rather than a 4P4D, and one role at half staff is
    exactly the case the marker exists to surface.

    Unchanged for a role-less model: `derive_model_state` makes RUNNING
    exactly `ready_replicas > 0` there, which is the guard this replaces.
    """
    if state != ModelStateEnum.RUNNING:
        return False
    if role_status is not None:
        return any(status.ready < status.desired for status in role_status.values())
    return ready_replicas < model.replicas


def _requires_every_member(model: Model) -> bool:
    """Whether this group is servable only at full staffing.

    Guarded rather than read straight through: a role-only deployment
    (multi-role orchestration with no PD) has no `disaggregation` at all, and
    that must mean the default rather than raise.

    Not exposed in the deployment form yet. The semantics live here so that
    the value a user can already set through the API is the value the group
    is judged by -- a stored setting that changes nothing is worse than an
    absent one, because it reads back as if it took effect.
    """
    disaggregation = model.disaggregation
    if disaggregation is None:
        return False
    return disaggregation.readiness == "all"


def derive_model_state(
    model: Model,
    *,
    ready_replicas: int,
    instance_count: int,
    role_status: Optional[Dict[str, RoleStatus]],
    error_count: int,
) -> Tuple[ModelStateEnum, Optional[str]]:
    """Fold one instance scan into the model-level lifecycle value and its
    message (F7 3.1 / 3.2).

    This is an aggregate, not a copy of `ModelInstanceStateEnum`: there are no
    download or start phases here.

    `state` answers one question — can this serve — and nothing else. Being
    up but worse than asked for lives beside it in `degradations`, never
    inside it: a group serving without its shared cache is RUNNING with
    `cache_not_injected`, and a deployment short of its declared replica count
    or role ratio is RUNNING with `ratio_unmet`. That is what lets PARTIAL
    keep one meaning everywhere — members are up and it still cannot serve —
    and lets the servability gate be a plain `state == RUNNING` with no
    per-shape special case.

    Readiness is decided before failure: a deployment with members up is
    serving whatever else has failed, so ERROR is reserved for "nothing is
    ready and something failed". That ordering is what keeps
    `state == RUNNING` equivalent to `ready_replicas > 0` for a role-less
    model, which the servability gate and the `GET /v2/models?state=` filter
    both depend on.
    """
    if role_status is not None:
        # A group is servable when every role has at least one ready member,
        # not when every role is fully staffed: requiring the latter would
        # make a 2P3D deployment unservable for the whole duration of a
        # scale-up (F7 3.1). Falling short of the declared ratio while every
        # role is covered is a degradation, not a lifecycle value.
        #
        # `readiness: all` is the other answer to the same question, for a
        # deployment sized so that a partial group is worse than no group --
        # a ratio tuned to a known load degrades into queueing rather than
        # into reduced throughput. It moves the shortfall from `degradations`
        # into `state`, which is a real behaviour change: the endpoint stops
        # accepting traffic during a scale-up instead of serving through it.
        # Hence per-deployment and defaulting to the forgiving one.
        require_full = _requires_every_member(model)
        roles_missing = sorted(
            name
            for name, status in role_status.items()
            if (status.ready < status.desired if require_full else status.ready == 0)
        )
        if not roles_missing and upstream_registration_ready(model):
            return ModelStateEnum.RUNNING, None
        if ready_replicas > 0:
            # Members up, still not servable — the one meaning PARTIAL has.
            if roles_missing:
                return (
                    ModelStateEnum.PARTIAL,
                    f"roles not ready: {', '.join(roles_missing)}",
                )
            return ModelStateEnum.PARTIAL, "waiting for upstream registration"
        if error_count:
            return (
                ModelStateEnum.ERROR,
                f"{error_count}/{instance_count} members failed",
            )
        return ModelStateEnum.PENDING, None

    # No roles: the backward-compatibility baseline. The model is its own
    # single implicit role, so "every role has a ready member" degenerates to
    # `ready_replicas > 0` — which is therefore exactly when it is RUNNING.
    # PARTIAL is unreachable here on purpose: one ready replica serves, so
    # there is no state in which a role-less model has members up and cannot
    # serve. Short of the requested count is `ratio_unmet`, carried by
    # `sync_model_status`, with the count itself in the message.
    if ready_replicas == 0:
        if error_count:
            # Only ERROR is counted as a failure. UNREACHABLE is a worker
            # comms fault that clears when the worker comes back, so it reads
            # as not-ready-yet, the same way it does per instance today.
            return (
                ModelStateEnum.ERROR,
                f"{error_count}/{instance_count} instances failed",
            )
        return ModelStateEnum.PENDING, None
    if ready_replicas < model.replicas:
        return (
            ModelStateEnum.RUNNING,
            f"{ready_replicas}/{model.replicas} replicas ready",
        )
    return ModelStateEnum.RUNNING, None


async def _router_dial(
    session: AsyncSession, router_instance: ModelInstance
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """`(host, proxy, token)`: how to reach this router.

    `host` is what goes in front of the port, and the two paths want different
    answers:

    - Through the tunnel proxy the request leaves from the WORKER's own side,
      so the worker's `ip` is exactly what resolves there.
    - Direct, the server has to route to the worker itself, and `ip` is
      routinely an address only that worker's network can reach. Measured on a
      cloud worker whose `ip` was a VPC address: every `GET /workers` failed
      to read the registry while the same router answered on the worker's
      `advertise_address`, so the group never got its members registered and
      served 503s while every member reported RUNNING.

    Direct is also the answer for a worker not in `tunnel` mode and for a
    worker row that cannot be loaded — dialling direct is what the product did
    before the proxy existed, so an unreadable row degrades to the old
    behaviour rather than to no attempt at all. Such a row falls back to the
    instance's own `worker_ip`, the only address left once the worker is
    unreadable.

    The token comes back with the address because the proxy authenticates
    every request it forwards. Returning the address alone would produce a
    hop that answers 401, which `reconcile` cannot tell apart from a router
    whose registry is unreadable — and that reading is what orders a router
    restart, so a missing credential would present as a restart loop.
    """
    fallback = router_instance.worker_ip
    if not router_instance.worker_id:
        return fallback, None, None
    try:
        worker = await Worker.one_by_id(session, router_instance.worker_id)
    except Exception as e:
        logger.debug(
            "Could not load worker %s for the router's address: %s",
            router_instance.worker_id,
            e,
        )
        return fallback, None, None
    if worker is None:
        return fallback, None, None
    proxy = worker.get_proxy_address()
    if proxy:
        return fallback, proxy, worker.token
    return worker.get_dial_address() or fallback, None, None


async def _reconcile_router_membership(
    session: AsyncSession, model: Model, instances: List[ModelInstance]
) -> None:
    """Tell every running router of this group who its members are.

    Records the outcome for `upstream_registration_ready`, which is what keeps
    a group whose registration failed out of RUNNING — it stays PARTIAL with
    the router's own words instead of claiming to serve.

    Never raises. A failure here has to read as "not registered yet" and be
    retried on the next pass, because the alternative is a controller loop
    that stops syncing every other status field over one unreachable router.
    """
    from gpustack.server.pd_mode_catalog import get_pd_mode

    if model.disaggregation is None:
        pd_membership.forget(model.id)
        return

    mode_name = model.disaggregation.mode.value
    mode = get_pd_mode(mode_name)
    if mode is None or not mode.router.membership_api_usable:
        # Command-line path: the router knows its peers already. Recording
        # nothing is what lets `upstream_registration_ready` stay vacuously
        # true for these groups.
        pd_membership.forget(model.id)
        return

    routers = pd_membership.router_instances(instances)
    if not routers:
        pd_membership.record(
            model.id,
            pd_membership.MembershipOutcome(
                ok=False, reason="no router is running yet"
            ),
        )
        return

    # Every router, and the worst outcome wins: one router with an empty
    # registry serves 503s while another serves fine, and a group is only
    # servable when the thing in front of it is.
    worst = None
    for router_instance in routers:
        # Per router, because both the proxy and the address belong to the
        # WORKER the router runs on: a `tunnel` worker only ever dials out, so
        # the server cannot reach the router's port directly and every call in
        # `reconcile` has to ride the same forward proxy the gateway uses for
        # that worker's model instances. Off the tunnel the server dials the
        # worker itself, where the address it has to use is the one the worker
        # publishes for outside access rather than the one it sees itself at.
        host, proxy, proxy_token = await _router_dial(session, router_instance)
        address = f"{host}:{router_instance.port}"
        try:
            outcome = await pd_membership.reconcile(
                model, mode, instances, address, proxy=proxy, proxy_token=proxy_token
            )
        except Exception as e:
            logger.warning(
                "Router membership reconcile failed for model %s at %s: %s",
                model.name,
                address,
                e,
            )
            outcome = pd_membership.MembershipOutcome(
                ok=False, reason=f"membership reconcile raised: {e}"
            )
        if worst is None or (worst.ok and not outcome.ok):
            worst = outcome
    if worst is not None:
        pd_membership.record(model.id, _explain_unreadable(model, worst))


def _explain_unreadable(
    model: Model, outcome: "pd_membership.MembershipOutcome"
) -> "pd_membership.MembershipOutcome":
    """Re-word an unreadable registry according to what happens next.

    The bare reason — "the member list could not be read" — is the same
    sentence whether the platform is about to recreate the router, has already
    tried and got nowhere, or has been told not to try at all. Those are three
    different situations for whoever is watching the deployment, so each gets
    its own account; anything else is returned untouched.
    """
    if not outcome.unreadable:
        return outcome

    if pd_membership.restarts_exhausted(model.id):
        # The message changes because the suspicion does. Up to here
        # "unreadable" could have been a wedged router, and the repair was to
        # restart it. Having restarted it and read nothing, what is left is
        # the path: the server cannot reach the router's port. On a
        # `tunnel`-mode worker that is the proxy — the only route inward —
        # and no further restart can discover that for the operator.
        reason = (
            "the router's member list cannot be read from the server, "
            "and restarting the router did not change that — so the "
            "router's port is not reachable rather than the process "
            "being stuck. On a worker in `tunnel` proxy mode the only "
            "route inward is the server's proxy port; check that it is "
            "running and that the worker's tunnel is connected."
        )
    elif not model.restart_on_error:
        # Say that the repair exists and was not taken, rather than leaving
        # a group parked with no account of why. Without this the deployment
        # reads the same whether the platform is about to recreate the router
        # or has decided not to.
        reason = (
            "the router's member list could not be read. Recreating the "
            "router is what usually repairs this, and the platform did NOT "
            "do it because this deployment has «restart on error» off — so "
            "the group stays as it is. Restart it manually once you have "
            "looked, or turn the switch on to let the platform try."
        )
    else:
        return outcome

    return pd_membership.MembershipOutcome(
        ok=False,
        unreadable=True,
        reason=reason,
        registered=outcome.registered,
    )


async def _restart_unreachable_routers(
    session: AsyncSession, model: Model, instances: Sequence[ModelInstance]
) -> None:
    """Delete the router members so convergence recreates them.

    Deleting rather than restarting in place: the router's command line is
    rendered from its peers' live addresses at creation, so a recreated router
    picks up the current ones — which is also the repair for the case this
    path exists for, a member whose port changed under a router that still
    holds its previous address.
    """
    from gpustack.schemas.models import ModelInstanceStateEnum

    for instance in instances:
        if instance.role != RoleNameEnum.ROUTER.value:
            continue
        if instance.state != ModelInstanceStateEnum.RUNNING:
            continue
        logger.warning(
            "Router %s of model %s has been unreachable for %d passes; "
            "deleting it so a fresh one is created with the group's current "
            "member addresses.",
            instance.name,
            model.name,
            pd_membership.RESTART_AFTER_UNREADABLE_PASSES,
        )
        try:
            await instance.delete(session)
        except Exception as e:
            logger.warning("Could not delete router %s: %s", instance.name, e)


async def sync_model_status(session: AsyncSession, model: Model) -> bool:  # noqa: C901
    """
    Synchronize the model's server-owned status from its instances.

    The single owner of every status field on the Model row: the
    counter, the lifecycle, the per-role detail and the degradation markers
    are four different questions about the same scan, so one scan answers
    them, one change gate writes them and one transaction commits them.
    Nothing else writes them.

    Returns True if the model row was updated (and the session was committed).
    """

    if model.deleted_at is not None:
        return False

    instances = await ModelInstance.all_by_field(session, "model_id", model.id)

    # `ready_replicas` keeps its exact pre-PD meaning: a plain count of
    # RUNNING instances, router included. `exporter.py` publishes it as
    # `model_running_instances`, so it is a counter and nothing else —
    # servability is `state`, per-role detail is `role_status`.
    ready_replicas: int = 0
    ready_by_role: Dict[str, int] = {}
    draining_by_role: Dict[str, int] = {}
    error_count: int = 0
    cache_not_injected: bool = False
    cache_reason: Optional[str] = None
    for instance in instances:
        draining = instance.draining_since is not None
        if instance.role and draining:
            draining_by_role[instance.role] = draining_by_role.get(instance.role, 0) + 1
        if instance.state == ModelInstanceStateEnum.RUNNING:
            # The model-level count keeps its pre-PD meaning — a plain count of
            # RUNNING instances, router and draining members included — because
            # `model_running_instances` publishes it and `derive_model_state`
            # reads it for role-less models.
            ready_replicas += 1
            # The per-role one does not: a draining member left the router's
            # member list when its window opened, so it is running without
            # being reachable. Counting it as ready is what made a role scaled
            # from 3 to 2 report `3 / 2` until the window closed.
            if instance.role and not draining:
                ready_by_role[instance.role] = ready_by_role.get(instance.role, 0) + 1
        elif instance.state == ModelInstanceStateEnum.ERROR:
            error_count += 1
        # A resolved cache the instance could not attach to is a degradation,
        # not a failure: the instance starts anyway, just without the shared
        # cache. `cache_config` is None when no cache service was
        # selected at all, which is not a degradation.
        if instance.cache_config is not None and not instance.cache_config.injected:
            cache_not_injected = True
            if cache_reason is None:
                cache_reason = instance.cache_config.reason

    role_status: Optional[Dict[str, RoleStatus]] = None
    if model.roles:
        # `desired` can only come from `roles[].replicas`: an instance that was
        # never created has no state, so the ratio is not derivable from the
        # scan alone (F7 3.2). `ready` is counted per role out of the same
        # scan.
        #
        # A group is one generation at a time -- no blue-green -- so
        # counting a role across the model's instances is counting it within
        # the live `group_id`. If that ever stops being true, scoping the
        # count to a generation belongs right here.
        role_status = {
            role.name: RoleStatus(
                desired=role.replicas,
                ready=ready_by_role.get(role.name, 0),
                draining=draining_by_role.get(role.name, 0),
            )
            for role in model.roles
        }

    # Reconcile BEFORE deriving the state. A router process that is up is
    # not the same as a router that can serve: what its registry already holds
    # is version-dependent and has to be read rather than assumed (see fact 2
    # in `pd_membership`), and a member this group adds through the API takes
    # traffic only once the router's own read-back reports it. Deriving RUNNING
    # first and registering after would publish an upstream that cannot serve.
    #
    # A no-op for any group whose recipe does not launch what its membership
    # API needs: `reconcile` returns ok immediately when
    # `membership_api_usable` is false.
    await _reconcile_router_membership(session, model, instances)

    # The one failure a restart can fix, and only after it has persisted.
    #
    # The shipped recipes run ONE router per group and it is the gateway's only
    # upstream, so recreating it interrupts the whole group until the
    # replacement is up and has finished probing its peers. That price is worth
    # paying only when the router is not answering at all, which is what
    # `unreadable` means; a router that refuses a member is alive and
    # disagreeing, and the replacement would be handed the same members to
    # refuse again.
    #
    # And only when the deployment asked to be repaired at all.
    # `restart_on_error` is the deployment's answer to "recover by yourself or
    # stop and let me look", and recreating the router is a recovery like any
    # other — more disruptive than most, since the replacement is a NEW member
    # with a fresh name and a zeroed restart count, which is precisely why a
    # group with the switch off looked like it was restarting forever and
    # never settled into a state anyone could inspect. With it off the group
    # stays where it failed and says why; `POST /{id}/restart` is the manual
    # way out.
    if model.restart_on_error and pd_membership.should_restart_router(model.id):
        pd_membership.note_restart_ordered(model.id)
        await _restart_unreachable_routers(session, model, instances)

    state, state_message = derive_model_state(
        model,
        ready_replicas=ready_replicas,
        instance_count=len(instances),
        role_status=role_status,
        error_count=error_count,
    )

    reasons, state_message = await _degradation_reasons(
        session,
        model,
        instances,
        state=state,
        ready_replicas=ready_replicas,
        role_status=role_status,
        cache_not_injected=cache_not_injected,
        cache_reason=cache_reason,
        state_message=state_message,
    )
    degradations = reasons or None

    # `stale`: the running members predate the config they are shown with.
    # Orthogonal to `state` — a stale group is usually still serving, which is
    # exactly what makes it worth surfacing: without it, a user who edits a
    # config and sees the model still RUNNING has no way to learn the edit has
    # not taken effect: changing `Model.env` returns the new value from the
    # API while `StartedAt` never moves.
    #
    # Only members that carry a digest count. A row created before this column
    # existed has None, and reading that as "differs" would mark every
    # pre-upgrade model stale on the first pass after an upgrade.
    stale = await _stale_members(session, model, instances)

    # The restart guard's other half. Set by the endpoint before it tears the
    # generation down, released here the moment the rebuilt one is serving —
    # which is the only event that actually means "the replacements are no
    # longer at risk". It rides the same change gate rather than getting its
    # own write, so releasing the guard costs nothing on a pass that was
    # already publishing the transition into RUNNING.
    restarting_since = model.restarting_since
    if restarting_since is not None and state == ModelStateEnum.RUNNING:
        restarting_since = None

    if (
        model.ready_replicas != ready_replicas
        or model.state != state
        or model.state_message != state_message
        or model.role_status != role_status
        or model.stale != stale
        or model.degradations != degradations
        or model.restarting_since != restarting_since
    ):
        model.ready_replicas = ready_replicas
        model.state = state
        model.state_message = state_message
        model.role_status = role_status
        model.stale = stale
        model.degradations = degradations
        model.restarting_since = restarting_since
        await ModelService(session).update(model)
        updated = True
    else:
        updated = False

    # After the state is settled, not inside the gate above: the target has
    # to be corrected even on a pass that found the model unchanged, because
    # the case this exists for is exactly "the model is right and the target
    # is not". Gating it on `updated` would reproduce the edge-triggered
    # behaviour it replaces.
    try:
        await reconcile_route_target_states(session, model)
    except Exception as e:
        # A target left stale is a routing outage, but so is a status pass that
        # raises: this is a repair, and it must not be able to break the thing
        # it rides on.
        logger.warning(
            "Could not re-derive route target states for model %s: %s",
            model.name,
            e,
        )

    return updated


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


async def sync_model_ai_proxy(
    cfg: Config,
    session: AsyncSession,
    extensions_api: ExtensionsHigressIoV1Api,
    model_id: int,
) -> None:
    """Reconcile ONE deployment's ai-proxy entry.

    The content is a pure function of the deployment — its instances'
    registries, its cluster's registration token, its anthropic selector.
    Model names a deployment serves (LoRA aliases, overrides) never enter:
    they are expressed on the mapper/lb CR, while ai-proxy only attaches
    the per-deployment credential to the deployment's own services.

    The route reference read is an EXISTENCE gate only: no live target
    means no rule (and the provider goes unreferenced); one target or
    many, aliased or not, produce the identical rule. Legacy per-route
    entries never ride this write — the startup cleanup pass retires
    them wholesale, which trades a bounded upgrade window for the
    absence of retirement races between sibling deployments.

    The caller is the Model controller: model and instance events own
    the ai-proxy CR, and route CRUD enqueues the affected models when a
    reference appears or disappears (see notify_model_ai_proxy_change).
    """
    owned_provider_ids = {mcp_handler.model_ai_proxy_provider_id(model_id)}

    group: Optional[mcp_handler.ModelAIProxyGroup] = None
    model = await Model.one_by_id(session, model_id)
    targets = await ModelRouteTarget.all_by_field(session, "model_id", model_id)
    live_targets = [
        target
        for target in targets
        if target.deleted_at is None and target.state == TargetStateEnum.ACTIVE
    ]
    # The legacy per-route ids are retired wholesale by the startup
    # cleanup pass (see cleanup_ai_proxy_config); retiring them here
    # instead would race sibling deployments' reconciles on shared
    # routes.
    if model is not None and model.deleted_at is None:
        if live_targets:
            destinations = await calculate_model_destinations(session, model)
            if destinations:
                group = mcp_handler.ModelAIProxyGroup(
                    model_id=model.id,
                    api_tokens=await cluster_registration_tokens(
                        session, model.cluster_id
                    ),
                    native_anthropic_api=model.native_anthropic_api,
                )
                group.service_names.update(
                    {registry.get_service_name() for _, _, registry in destinations}
                )

    expected_providers, expected_match_rules = mcp_handler.model_ai_proxy_plugin_spec(
        groups=[group] if group is not None else [],
    )
    await mcp_handler.ensure_wasm_plugin(
        api=extensions_api,
        name=mcp_handler.gpustack_ai_proxy_name,
        namespace=cfg.gateway_namespace,
        spec_diff=partial(
            mcp_handler.ai_proxy_diff_spec,
            expected_providers=expected_providers,
            expected_match_rules=expected_match_rules,
            owned_provider_ids=owned_provider_ids,
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
    destinations = []
    fallback_destinations = []
    if not model_route_from_db:
        event_type = EventType.DELETED
    if event.type != EventType.DELETED:
        destinations, fallback_destinations = await calculate_destinations(
            session, model_route
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
    # One collector across every route plugin: the flush after dispatch turns
    # all their declarations into a single read-modify-write per shared CR.
    # The mapper's fallback rules and the fallback ingress/filter are the
    # fallback plugin's; this function keeps the shared inputs (the
    # destinations pass, the effective name) and the core-path artifacts.
    collector = RouteArtifactCollector()
    await dispatch_route_reconcile(
        RouteReconcileContext(
            cfg=cfg,
            session=session,
            model_route=model_route,
            ingress_name=ingress_name,
            event_is_delete=event_type == EventType.DELETED,
            extensions_api=extensions_api,
            istio_networking_api=istio_networking_api,
            collector=collector,
            networking_api=networking_api,
            effective_name=effective_name,
            fallback_destinations=fallback_destinations,
            destinations=destinations,
        )
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
]:
    """
    Return the percentage tuple for each registry with model name and the
    fallback registry. The ai-proxy provider config is per deployment and
    is refreshed by ``sync_model_ai_proxy`` from the model's full
    reference set, not from this route's pass.
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
        count = sum([count for count, _, _ in to_extend])
        weight_to_count.append((target.weight, count, to_extend))
        if is_fallback_target:
            fallback_weight_to_count.append((target.weight, count, to_extend))
    if len(weight_to_count) == 0:
        return [], []

    # All-zero weights are the LB scoring / round-robin mode: candidate
    # selection is the gateway plugin's job (the cluster_header
    # EnvoyFilter displaces Envoy's weighted_clusters), but the ingress
    # still needs a destination entry to exist at all — flatten with
    # max_weight=1 so every registry gets an equal placeholder share,
    # the same trick the fallback list already uses for its zero-weight
    # members. Dropping them (a plain flatten) deletes the route's
    # ingress entirely.
    all_zero = all(weight == 0 or weight is None for weight, _, _ in weight_to_count)
    flatten_registry_list = flatten_destinations(
        weight_to_count, max_weight=1 if all_zero else 0
    )
    fallback_registry_list = []
    if len(fallback_weight_to_count) > 0:
        # fallback might have 0 weight, so set max_weight to 1
        fallback_registry_list = flatten_destinations(
            fallback_weight_to_count, max_weight=1
        )

    return flatten_registry_list, fallback_registry_list


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
    becomes a self-map (skipped at the fallback rule render), letting the
    LoRA module name reach vLLM intact.
    """
    downstream_model_name = overridden_model_name or model.name
    # The model name rides the destination tuple and is expressed on the
    # mapper/lb CR (candidate.modelName, modelMappers) — never as a
    # distinct service name. Every name a deployment serves, LoRA
    # included, routes over the deployment's own registries; the wasm
    # plugin weights and rewrites per candidate, so per-name alias
    # clusters are not needed.
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
    # Same narrowing as the registry side in `_ensure_model_mcp_bridge`. The
    # registry decides which addresses *exist* as upstreams; this annotation
    # decides how traffic is *split* across them, and a weight naming a member
    # the registry never registered is what Envoy hangs on. Measured: a 1P1D
    # group got `34% router / 33% prefill / 33% decode`, the router answered
    # correctly and the other two thirds of requests never returned.
    instances = _gateway_registrable_instances(model, instances)
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

    ``on_materialized`` is awaited once the table holds the new set, for a
    reader that has to be current with it rather than with the source it came
    from. Both would see the same source event, and nothing orders two
    subscribers — this says when the rows are actually there.
    """

    def __init__(self, on_materialized: Optional[Callable[[], Awaitable]] = None):
        self._on_materialized = on_materialized

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
            # The table is as it was, so a reader current with it still is.
            return
        if self._on_materialized is not None:
            await self._on_materialized()


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


class CacheProviderSourceController:
    """
    Leader-only controller that materializes CacheProviderEntry from
    CacheProviderSource. On start it seeds the BUILTIN source from the packaged
    asset merged with every asset an installed plugin ships, then (like
    CatalogSourceController) subscribes and full-rewrites the materialized table
    on any source change; the initial replay of existing sources drives the
    first reconcile.
    """

    def __init__(self):
        # Two drivers rewrite this table -- a source change, and the runner
        # overrides landing -- and on a first start they arrive together. Each
        # reads the sources, then writes every row; interleaved, one of them
        # loses its inserts to the name uniqueness constraint, which is caught
        # and logged rather than retried. The loser being the runner one would
        # leave the table holding what was derived from the packaged catalog,
        # with nothing to come back for it: a source that did not move
        # publishes no event, so the next round is 12 hours away at best.
        # (SourceRefresher._round_lock exists for the same reason.)
        self._rewrite_lock = asyncio.Lock()

    async def start(self):
        await self._seed_builtin_source()
        async for event in CacheProviderSource.subscribe(
            source="cache_provider_source_controller"
        ):
            if event.type in (
                EventType.CREATED,
                EventType.UPDATED,
                EventType.DELETED,
            ):
                await self.rebuild()

    async def rebuild(self) -> None:
        """Rewrite the materialized catalog from the sources as they stand.

        Also what the runner materialization calls when it has finished: a
        provider reading its release line off the runner images reads the
        admin's additions to those too, and the version they add appears here
        only once this runs. Driven from there rather than from the source
        those overrides come from, which both controllers see at once with
        nothing to order them — this catalog would read the rows as they were
        before the rewrite, and nothing would come back for it.
        """
        await self._reconcile()

    async def _reconcile(self):
        async with self._rewrite_lock:
            try:
                async with async_session() as session:
                    await gather_and_merge(
                        session, CacheProviderSource, reconcile_cache_providers
                    )
            except Exception as e:
                logger.error(f"Failed to reconcile cache provider entries: {e}")

    async def _seed_builtin_source(self):
        """Upsert the BUILTIN row from the assets this release carries.

        Refreshed on every start, so an upgrade — or a newly installed plugin
        carrying a provider — ships its declarations; the user's ``enabled``
        toggle on the row is left untouched.
        """
        try:
            content = await asyncio.to_thread(builtin_catalog_text)
        except Exception as e:
            logger.error(f"Failed to seed the built-in cache provider source: {e}")
            return

        content_hash = sha256_of(content)
        async with async_session() as session:
            existing = await CacheProviderSource.one_by_field(
                session, "name", BUILTIN_CACHE_PROVIDER_SOURCE_NAME
            )
            if existing:
                # Skip the write when a restart re-seeds the same assets, judged
                # by the same hash every other writer in the source layer uses.
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
                await CacheProviderSource.create(
                    session,
                    CacheProviderSource(
                        name=BUILTIN_CACHE_PROVIDER_SOURCE_NAME,
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


def _changed_scalar(value: Any) -> Any:
    """Normalize a ``changed_fields`` scalar side to its plain value.

    The two producers of change events store different shapes for a
    scalar column: the local ``find_history`` path records
    ``(hist.deleted, hist.added)`` — sequences that are empty on the
    None side of a None↔value transition — while the cross-instance
    ``detect_changes`` path records a flat ``(old, new)``. Both
    collapse to the scalar (None included).
    """
    if isinstance(value, (tuple, list)):
        return value[0] if value else None
    return value


async def notify_model_ai_proxy_change(
    session: AsyncSession, model_ids: "Set[int]"
) -> None:
    """Enqueue the models whose ai-proxy reference set may have changed.

    Route CRUD does not write the ai-proxy CR — the Model controller owns
    it — so when a target appears or disappears (route create/update/
    delete, target add/remove, a state flip through the ACTIVE gate), the
    affected models are enqueued here and rebuilt from their full
    reference set. Deleted models need no event: their own delete event
    strips the entry.
    """
    for model_id in model_ids:
        if model_id is None:
            continue
        model = await Model.one_by_id(session, model_id)
        if model is None or model.deleted_at is not None:
            continue
        copied = Model.model_validate(model.model_dump())
        await event_bus.publish(
            Model.__name__.lower(),
            Event(type=EventType.UPDATED, data=copied),
        )


async def notify_model_route_target(session: AsyncSession, model: Model, event: Event):
    if event.type == EventType.DELETED:
        return
    should_notify = False
    if event.changed_fields is not None:
        # ``native_anthropic_api`` is read where the route's ai-proxy provider
        # entry is built, and that only runs off a route event -- so without it
        # here, flipping the selector would change nothing until the deployment
        # happened to scale.
        #
        # `state` is what the target's ACTIVE gate reads, so a state change
        # has to reach the target even when the RUNNING count did not move
        # (a group whose upstream registration flips, for instance).
        related_fields = [
            "state",
            "ready_replicas",
            "replicas",
            "native_anthropic_api",
        ]
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
                                "state": model.state,
                                "ready_replicas": model.ready_replicas,
                                "replicas": model.replicas,
                            },
                        )
                    },
                ),
            )


async def sync_categories_and_meta(session: AsyncSession, model: Model, event: Event):
    """Propagate a model's derived ``categories`` / ``meta`` onto the route it
    created.

    Neither field is known when the route is born: `POST /models` copies
    whatever the caller sent (the UI sends neither, they are auto-detected)
    and the scheduler fills them in a second later, off `evaluate_gguf_model`
    / `evaluate_pretrained_config`. Only the Model is written there, so
    without this the route keeps the empty list it was created with -- and
    `/v1/models?categories=llm` filters on `ModelRoute.categories`, so the
    deployment silently stops being listed as an LLM.

    Queries `ModelRoute` directly rather than walking `model.model_routes`,
    for the reason `reconcile_route_target_states` spells out at length:
    `_reconcile` has already loaded this Model row twice by the time we get
    here (plainly for `sync_model_status`, then with
    `selectinload(model_route_targets)` inside `notify_model_route_target`),
    so a third fetch asking for `selectinload(model_routes)` hits the
    identity map, returns that same instance and never applies the loader
    option. `model_routes` is `lazy="noload"`, and an unloaded collection
    reads as `[]` rather than raising -- so the loop below found nothing to
    do, every time, for every model, without a single log line. Confirmed
    against a live database: a clean session yields the route, a session in
    `_reconcile`'s state yields none.

    `created_model_id` is also the more honest filter. The relationship goes
    through `ModelRouteTarget`, so a multi-target route reached from model A
    would have had A's categories written onto it even when B created it.
    """
    if event.type == EventType.DELETED:
        return
    model: Model = await Model.one_by_id(session=session, id=model.id)
    if not model:
        return
    routes = await ModelRoute.all_by_fields(
        session,
        fields={"created_model_id": model.id, "deleted_at": None},
    )
    # Plugins keep per-route state in meta under their registered names
    # (the lb base capability). The model-sync wholesale meta replace
    # would clobber those keys, so they are carried over from the
    # current row — model metadata owns the rest.
    from gpustack.routes.plugins import route_plugins

    plugin_meta_keys = {p.name for p in route_plugins()}
    for route in routes:
        merged_meta = {
            # A plugin-named key in model.meta is ordinary user data,
            # not plugin state — it must not be able to forge or
            # resurrect plugin-owned keys on the route row.
            **{
                k: v for k, v in (model.meta or {}).items() if k not in plugin_meta_keys
            },
            **{k: v for k, v in (route.meta or {}).items() if k in plugin_meta_keys},
        }
        if route.categories != model.categories or route.meta != merged_meta:
            await ModelRouteService(session).update(
                model_route=route,
                source={"categories": model.categories, "meta": merged_meta},
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
        # Before the subscription, because the gap between them is the hole
        # this closes. The bus does not replay: a model that changed state
        # while this controller was down published an event nobody consumed,
        # and since the state then stops changing there is nothing left to
        # react to. A target can therefore sit UNAVAILABLE against a RUNNING
        # model indefinitely, with `/v1/models` empty.
        #
        # One sweep at startup answers it for every target at once, and costs
        # one pass over a table with as many rows as there are route targets.
        await self._resync_all_targets()

        async for event in ModelRouteTarget.subscribe(
            source="model_route_target_controller"
        ):
            try:
                await self._reconcile(event)
            except Exception as e:
                logger.exception(f"Failed to reconcile model route target: {e}")

    async def _resync_all_targets(self):
        """Re-derive every target's state from what it points at.

        Idempotent and write-on-difference, so a healthy fleet logs nothing
        and writes nothing.
        """
        try:
            async with async_session() as session:
                targets = await ModelRouteTarget.all(session)
                repaired = 0
                for target in targets:
                    model = None
                    if target.model_id is not None:
                        model = await Model.one_by_id(session, target.model_id)
                    desired = derive_route_target_state(target, model)
                    if target.state != desired:
                        logger.info(
                            "Startup resync: route target %s %s -> %s",
                            target.name,
                            target.state,
                            desired,
                        )
                        target.state = desired
                        await target.update(session=session, auto_commit=True)
                        repaired += 1
                if repaired:
                    logger.info(
                        "Startup resync corrected %d route target(s) whose "
                        "state had drifted from their model's.",
                        repaired,
                    )
        except Exception as e:
            # Never fatal: the controller's steady-state job is more important
            # than this repair, and the per-model pass in `sync_model_status`
            # is a second chance at the same correction.
            logger.warning("Route target startup resync failed: %s", e)

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
            # LB candidate-shaping columns — a change here must re-render
            # the gateway candidates config, same as a weight or name edit
            "weight",
            "max_running_requests",
            # The fallback path (fallback ingress, filter, mapper rules)
            # turns on and off with a target's fallback codes
            "fallback_status_codes",
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
            # A state flip crosses the ai-proxy existence gate, and a
            # model_id change crosses it for both models — enqueue them
            # so the Model controller rebuilds their entries from the new
            # reference set.
            if event.type == EventType.DELETED:
                await notify_model_ai_proxy_change(session, {target.model_id})
            else:
                changed = changed_fields or {}
                if "state" in changed:
                    await notify_model_ai_proxy_change(session, {target.model_id})
                if "model_id" in changed:
                    old_model_id, new_model_id = changed["model_id"]
                    await notify_model_ai_proxy_change(
                        session,
                        {
                            _changed_scalar(old_model_id),
                            _changed_scalar(new_model_id),
                        },
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
            # The servability gate: `Model.state`, not the RUNNING count.
            # `ModelRoute.ready_targets`, `/v1/models` and
            # `resolve_route_targets` are all defined off this target state,
            # so they follow from here. The route's weight / fallback / alias
            # mechanics are untouched.
            target_state = (
                TargetStateEnum.ACTIVE
                if is_model_servable(model)
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
