import asyncio
import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple, Union
from fastapi import APIRouter, Depends, Query, Request, Response
from pydantic import BaseModel
from fastapi.responses import RedirectResponse, StreamingResponse
from gpustack_runtime.detector import ManufacturerEnum
from sqlalchemy.orm import selectinload
from sqlmodel import or_, select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack import envs
from gpustack.api.exceptions import (
    AlreadyExistsException,
    ConflictException,
    InternalServerErrorException,
    BadRequestException,
    ForbiddenException,
    NotFoundException,
)
from gpustack.schemas.common import Pagination
from gpustack.schemas.inference_backend import is_custom_backend
from gpustack.schemas.models import (
    ModelInstance,
    ModelInstanceStateEnum,
    ModelInstancesPublic,
    BackendEnum,
    ModelListParams,
)
from gpustack.schemas.cache_services import (
    CacheService,
    CacheServiceAttachedMetrics,
    ModelCacheMetricsPublic,
)
from gpustack.server.cache_service_metrics import (
    collect_model_cache_metrics,
    parse_window as parse_metrics_window,
)
from gpustack.schemas.deployment_document import (
    OVERWRITABLE_FIELDS,
    DeploymentActionEnum,
    DeploymentExportRequest,
    DeploymentImportRequest,
    DeploymentImportResult,
    DeploymentPlanEntry,
    LoadedEntry,
    deployment_entry,
    diff_entries,
    dump_deployments,
    entry_document_form,
    entry_label,
    load_deployments,
)
from gpustack.schemas.clusters import Cluster, GatherStrategyEnum
from gpustack.schemas.gpu_instance_types import GPUInstanceType
from gpustack.schemas.workers import GPUDeviceStatus, Worker
from gpustack.utils.version import version_in_range
from gpustack.api.tenant import (
    TenantContext,
    bypass_tenant_filter,
    assert_cluster_visible,
    assert_resource_visible,
    cluster_scoped_system,
    scoped_cluster_row_visible,
    tenant_list_conditions,
)
from gpustack.server.db import async_session
from gpustack.server.deps import (
    CurrentUserDep,
    ListParamsDep,
    SessionDep,
    TenantContextDep,
)
from gpustack.schemas.models import (
    PD_BACKENDS,
    ExtendedKVCacheConfig,
    LoraListEntry,
    PDModeEnum,
    Model,
    ModelCreate,
    ModelSpecBase,
    ModelUpdate,
    ModelPublic,
    ModelsPublic,
    RoleNameEnum,
    role_effective_model,
)
from gpustack.schemas.model_routes import (
    AccessPolicyEnum,
    ModelRoute,
    ModelRouteTarget,
    TargetStateEnum,
)
from gpustack.schemas.links import ModelRoutePrincipalLink
from gpustack.schemas.principals import platform_principal_id
from gpustack.server.services import (
    ModelRouteService,
    ModelInstanceService,
    ModelService,
    WorkerService,
    revoke_model_access_cache,
)
from gpustack.server.controllers import model_spec_digest
from gpustack.server.scaling_scheduler import compute_desired_replicas
from gpustack.server.cache_provider_catalog import get_cache_provider
from gpustack.server.lora_adapters_discovery import list_adapters_for_base
from gpustack.server.lora_model_routes import (
    cleanup_orphan_lora_routes,
    create_lora_model_routes,
    is_lora_list_stale,
)
from gpustack.server.pd_pairing import (
    DIFFER,
    PAIRING_MUST_MATCH,
    compare_max_model_len,
    compare_must_match,
    effective_tensor_parallelism,
    role_parameters,
    tensor_parallel_rule,
    violates_tensor_parallel_direction,
)
from gpustack.utils.command import find_last_parameter, find_parameter
from gpustack.utils.export_limits import attachment_headers, sanitize_filename
from gpustack.utils.convert import safe_int
from gpustack.utils.gpu import parse_gpu_id
from gpustack.routes.model_common import (
    ModelStateFilterEnum,
    build_category_conditions,
    categories_filter,
    model_state_condition,
    model_state_stream_filter,
)
from gpustack.config.config import get_global_config
from gpustack.schemas.pd_modes import PDTensorParallelPairingEnum
from gpustack.server.pd_mode_catalog import get_pd_mode
from gpustack.server.pd_mode_resolver import resolve_pd_mode
from gpustack.server.cluster_accelerators import cluster_vendors
from gpustack.utils.grafana import build_model_dashboard_url, resolve_grafana_base_url
from gpustack.utils.lora_model_source import lora_route_name_for

router = APIRouter()

logger = logging.getLogger(__name__)


def _make_model_watch_filter(ctx, categories, state=None):
    """Watch-stream visibility: cluster-bound service accounts only see
    their own cluster's models; everyone keeps the categories and state
    filters. Predicates are pre-built so inactive filters cost nothing on
    the per-event hot path."""
    predicates = []
    if cluster_scoped_system(ctx):
        predicates.append(lambda data: scoped_cluster_row_visible(ctx, data))
    if state is not None:
        predicates.append(lambda data: model_state_stream_filter(data, state))
    if categories:
        predicates.append(lambda data: categories_filter(data, categories))

    def _visible(data) -> bool:
        for p in predicates:
            if not p(data):
                return False
        return True

    return _visible


@router.get("", response_model=ModelsPublic)
async def get_models(
    ctx: TenantContextDep,
    params: ModelListParams = Depends(),
    state: Optional[ModelStateFilterEnum] = Query(
        default=None,
        description="Filter by model state.",
    ),
    search: str = None,
    categories: Optional[List[str]] = Query(None, description="Filter by categories."),
    cluster_id: int = None,
    backend: Optional[str] = Query(None, description="Filter by backend."),
):
    fuzzy_fields = {}
    if search:
        fuzzy_fields = {"name": search}

    fields = {}
    if cluster_id:
        fields["cluster_id"] = cluster_id

    if backend:
        fields["backend"] = backend

    # Streaming uses field-equality only; scope by current org so non-admin
    # users never see cross-org rows via the live stream. Admin without an
    # explicit org context keeps the unfiltered cross-org stream. System
    # users (workers / cluster accounts) bypass owner scoping — they serve
    # every Org's models — but cluster-bound service accounts are narrowed
    # to their own cluster's rows below.
    if ctx.current_principal_id is not None and not bypass_tenant_filter(ctx):
        fields["owner_principal_id"] = ctx.current_principal_id

    if params.watch:
        return StreamingResponse(
            Model.streaming(
                fields=fields,
                fuzzy_fields=fuzzy_fields,
                filter_func=_make_model_watch_filter(ctx, categories, state),
            ),
            media_type="text/event-stream",
        )

    async with async_session() as session:
        extra_conditions = list(tenant_list_conditions(ctx, Model))
        if categories:
            conditions = build_category_conditions(session, Model, categories)
            extra_conditions.append(or_(*conditions))

        state_condition = model_state_condition(state)
        if state_condition is not None:
            extra_conditions.append(state_condition)

        order_by = params.order_by
        if order_by:
            # When sorting by "source", add additional sorting fields for deterministic ordering
            new_order_by = []
            for field, direction in order_by:
                new_order_by.append((field, direction))
                if field == "source":
                    new_order_by.append(("huggingface_repo_id", direction))
                    new_order_by.append(("huggingface_filename", direction))
                    new_order_by.append(("model_scope_model_id", direction))
                    new_order_by.append(("model_scope_file_path", direction))
                    new_order_by.append(("local_path", direction))
            order_by = new_order_by

        return await Model.paginated_by_query(
            session=session,
            fuzzy_fields=fuzzy_fields,
            extra_conditions=extra_conditions,
            page=params.page,
            per_page=params.perPage,
            fields=fields,
            order_by=order_by,
        )


@router.get("/adapters", response_model=Dict[str, Any])
async def get_model_adapters(
    session: SessionDep,
    user: CurrentUserDep,
    base: str = Query(
        ...,
        description=(
            "Base model repo id (e.g. Qwen/Qwen3-8B) for HF/ModelScope adapter discovery; "
            "also used to match local cached LoRAs."
        ),
    ),
    q: Optional[str] = Query(
        None,
        description="Optional keyword (Hugging Face search, ModelScope Search).",
    ),
    limit: int = Query(
        40,
        ge=1,
        le=200,
        description="Max adapter entries per remote source (HF and ModelScope).",
    ),
):
    _ = user
    return await list_adapters_for_base(session, base, q=q, limit=limit)


@router.get("/{id}", response_model=ModelPublic)
async def get_model(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
):
    model = await Model.one_by_id(session, id, options=[selectinload(Model.instances)])
    assert_resource_visible(ctx, model, not_found_message="Model not found")
    public = ModelPublic.model_validate(model)
    public.has_stale_lora_instances = is_lora_list_stale(model)
    return public


@router.get("/{id}/dashboard")
async def get_model_dashboard(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    request: Request,
):
    model = await _get_model(session=session, ctx=ctx, id=id)

    cfg = get_global_config()

    cluster = None
    if model.cluster_id is not None:
        cluster = await Cluster.one_by_id(session, model.cluster_id)

    # Which dashboard a deployment belongs on, and with which variables, is
    # `build_model_dashboard_url`'s to answer: a benchmark report links to the
    # same place and a group must not be sent to the model dashboard from
    # either door.
    dashboard_url = build_model_dashboard_url(
        cfg,
        resolve_grafana_base_url(cfg, request),
        model,
        cluster_name=cluster.name if cluster is not None else None,
    )
    if dashboard_url is None:
        raise InternalServerErrorException(
            message="Grafana dashboard settings are not configured"
        )

    return RedirectResponse(url=dashboard_url, status_code=302)


async def _get_model(
    session: SessionDep,
    ctx,
    id: int,
):
    model = await Model.one_by_id(session, id)
    assert_resource_visible(ctx, model, not_found_message="Model not found")
    return model


@router.get("/{id}/instances", response_model=ModelInstancesPublic)
async def get_model_instances(ctx: TenantContextDep, id: int, params: ListParamsDep):
    if params.watch:
        # Gate the stream on the same visibility check the non-watch
        # branch applies, so a model id outside the caller's scope can't
        # be tailed live.
        async with async_session() as session:
            model = await Model.one_by_id(session, id)
            assert_resource_visible(ctx, model, not_found_message="Model not found")
        fields = {"model_id": id}
        return StreamingResponse(
            ModelInstance.streaming(fields=fields),
            media_type="text/event-stream",
        )

    async with async_session() as session:
        model = await Model.one_by_id(
            session, id, options=[selectinload(Model.instances)]
        )
        assert_resource_visible(ctx, model, not_found_message="Model not found")

        instances = model.instances
        count = len(instances)
        total_page = math.ceil(count / params.perPage)
        pagination = Pagination(
            page=params.page,
            perPage=params.perPage,
            total=count,
            totalPage=total_page,
        )

        return ModelInstancesPublic(items=instances, pagination=pagination)


@router.get("/{id}/cache-metrics", response_model=ModelCacheMetricsPublic)
async def get_model_cache_metrics(
    request: Request,
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    window: str = "1h",
):
    """What the shared cache service is doing for this deployment: the
    engines' own external-cache hit accounting, per instance, over the
    window. Gated on the deployment's visibility — a caller reading its
    own deployment's hit rate is not reading the cache service's
    telemetry, which stays the service owner's."""
    model = await Model.one_by_id(session, id, options=[selectinload(Model.instances)])
    assert_resource_visible(ctx, model, not_found_message="Model not found")
    try:
        window_seconds = parse_metrics_window(window)
    except ValueError as e:
        raise BadRequestException(message=str(e))
    if not (model.extended_kv_cache and model.extended_kv_cache.is_shared()):
        return ModelCacheMetricsPublic(
            available=False,
            reason="The deployment does not use a shared cache service",
        )
    attached = [
        CacheServiceAttachedMetrics(
            model_id=model.id,
            model_name=model.name,
            model_instance_name=instance.name,
            worker_name=instance.worker_name,
        )
        for instance in model.instances
    ]
    attached.sort(key=lambda row: row.model_instance_name or "")
    return await collect_model_cache_metrics(
        model.cluster_id,
        attached,
        window_seconds,
        client=getattr(request.app.state, "http_client_no_proxy", None),
    )


def apply_scaling_schedule_baseline(
    model_in: Union[ModelCreate, ModelUpdate, ModelSpecBase],
) -> None:
    """
    Drive ``replicas`` from an enabled scaling schedule.

    While a schedule is enabled the replica count is owned by the schedule:
    ``baseline_replicas`` plus the window rules are the user's input, and
    ``replicas`` becomes the scheduler-driven value. Set it to the count
    effective right now so the model doesn't run at a stale count until the
    scheduler's next tick. A submitted ``replicas`` is ignored in this mode.

    Call this as a server-side assignment *after* validation. Validating a
    rewritten ``replicas`` would make checks depend on the current wall clock:
    the value is a point-in-time output of the schedule, not caller intent.
    """
    schedule = getattr(model_in, "scaling_schedule", None)
    if not schedule or not schedule.enabled:
        return
    effective = compute_desired_replicas(schedule)
    if effective is not None:
        model_in.replicas = effective


def _max_intended_replicas(
    model_in: Union[ModelCreate, ModelUpdate, ModelSpecBase],
) -> int:
    """Largest replica count this deployment could ever run.

    Only for deciding *whether* placement needs validating. The plain
    ``replicas > 0`` gate assumes a zero count means no instances are ever
    placed, which a schedule breaks: ``replicas`` is then just the count for
    right now, and the scheduler raises it later. Scaling to zero outside
    business hours is a headline use case, so a submitted 0 must still get its
    ``gpu_selector`` checked. Checks *inside* validation keep reading the
    submitted ``replicas`` — that is the caller's intent.
    """
    schedule = getattr(model_in, "scaling_schedule", None)
    if not schedule or not schedule.enabled:
        return model_in.replicas
    # An enabled schedule always carries a baseline and at least one rule.
    return max(
        model_in.replicas,
        schedule.baseline_replicas,
        *(rule.replicas for rule in schedule.rules),
    )


def validate_roles(  # noqa: C901
    model_in: Union[ModelCreate, ModelUpdate, ModelSpecBase],
    stored: Optional[Model] = None,
) -> None:
    """Structural checks on a multi-role deployment.

    These checks deliberately reject rather than reinterpret. Every rule here
    exists because the alternative — silently folding the request into
    something adjacent — is the failure mode that makes a deployment behave
    unlike what the user typed.

    `stored` is the row being updated, and every rule below is judged against
    the *merged* state. A sparse PUT carries only the fields it changes, so
    without this a request that adds a schedule without resending `roles` would
    be judged as a role-less model and the schedule would be accepted onto a
    group — exactly the combination the rule forbids, reached by not mentioning
    the thing that makes it illegal. Same shape for `replicas`. Read-only, so
    nothing here can widen what the request persists.
    """

    def field(name: str):
        submitted = getattr(model_in, name, None)
        if submitted is not None:
            return submitted
        if stored is not None and name not in getattr(
            model_in, "model_fields_set", set()
        ):
            return getattr(stored, name, None)
        return submitted

    roles = field("roles")
    disaggregation = field("disaggregation")

    if not roles:
        if disaggregation is not None:
            raise BadRequestException(
                message="disaggregation requires roles: declare a prefill and a decode role."
            )
        return

    names = [role.name for role in roles]
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        raise BadRequestException(
            message=f"Duplicate role name(s): {', '.join(sorted(duplicates))}."
        )

    allowed = {item.value for item in RoleNameEnum}
    unknown = [name for name in names if name not in allowed]
    if unknown:
        raise BadRequestException(
            message=(
                f"Unsupported role name(s): {', '.join(unknown)}. "
                f"Supported roles are {', '.join(sorted(allowed))}."
            )
        )

    for role in roles:
        if role.name == RoleNameEnum.ROUTER.value and role.replicas != 1:
            raise BadRequestException(
                message="The router role runs exactly one replica."
            )
        # Refused rather than ignored. Prefill and decode get CPU and memory
        # from sizing — the weights and the parallelism decide them — so a
        # hand-written value here would be a second source for the same number
        # that silently disagrees with the estimate.
        if role.resources is not None and role.name != RoleNameEnum.ROUTER.value:
            raise BadRequestException(
                message=(
                    f"Role '{role.name}' cannot declare CPU or memory: only the "
                    "router does, because it holds no weights. Every other "
                    "role's footprint is derived from the model."
                )
            )
        # Refused rather than ignored, and for the same reason `resources` is:
        # a value nothing reads is indistinguishable from one that was never
        # sent. Until this field existed the role's adapters were dropped by
        # `extra="ignore"` and the response was a 200 with no `lora_list` in
        # it — the user's only clue that their configuration had not been
        # stored was reading the record back and noticing an absence.
        if role.lora_list:
            raise BadRequestException(
                message=(
                    f"Role '{role.name}' cannot declare LoRA adapters: a "
                    "multi-role deployment is reached through its router, and "
                    "the router's member table is indexed by served-model "
                    "name, so an adapter name attached to one role never "
                    "resolves to that role. Declare lora_list on the model "
                    "instead, or deploy the adapters as their own model."
                )
            )
        if role.resources is not None:
            if role.resources.cpu is not None and role.resources.cpu <= 0:
                raise BadRequestException(
                    message="The router's CPU request must be greater than zero."
                )
            if role.resources.memory is not None and role.resources.memory <= 0:
                raise BadRequestException(
                    message="The router's memory request must be greater than zero."
                )

    # `dependencies` is a start order, so a cycle is a deployment that never
    # starts. Reject it here rather than letting the controller spin.
    known = set(names)
    graph = {role.name: list(role.dependencies or []) for role in roles}
    for name, deps in graph.items():
        for dep in deps:
            if dep not in known:
                raise BadRequestException(
                    message=f"Role '{name}' depends on '{dep}', which is not declared."
                )
            if dep == name:
                raise BadRequestException(
                    message=f"Role '{name}' cannot depend on itself."
                )
    visiting: set = set()
    done: set = set()

    def _walk(name: str) -> None:
        if name in done:
            return
        if name in visiting:
            raise BadRequestException(
                message=f"Role dependencies form a cycle through '{name}'."
            )
        visiting.add(name)
        for dep in graph.get(name, []):
            _walk(dep)
        visiting.discard(name)
        done.add(name)

    for name in graph:
        _walk(name)

    # `roles[].replicas` is the only scaling truth, so a model-level count
    # above one would be a second one. Refuse instead of quietly reading it as
    # a multiplier — an implicit mode switch is exactly what makes a
    # deployment stop matching its own spec.
    if field("replicas") not in (0, 1):
        raise BadRequestException(
            message=(
                "A model with roles uses replicas as an on/off switch (0 or 1). "
                "Scale a disaggregated deployment through roles[].replicas."
            )
        )

    # The scaling scheduler writes `model.replicas` directly, without passing
    # through this validation, so a window rule holding 3 would break the
    # deployment at its next tick rather than at submit time.
    schedule = field("scaling_schedule")
    if schedule and schedule.enabled:
        raise BadRequestException(
            message="Scheduled scaling is not supported for a model with roles."
        )

    if disaggregation is None:
        return

    counts = {name: names.count(name) for name in allowed}
    if counts[RoleNameEnum.PREFILL.value] != 1:
        raise BadRequestException(
            message="A disaggregated model needs exactly one prefill role."
        )
    if counts[RoleNameEnum.DECODE.value] != 1:
        raise BadRequestException(
            message="A disaggregated model needs exactly one decode role."
        )
    if counts[RoleNameEnum.ROUTER.value] > 1:
        raise BadRequestException(
            message="A disaggregated model has at most one router."
        )

    _reject_lora_under_disaggregation(field)
    _reject_an_engine_that_cannot_be_disaggregated(field, roles)
    _reject_cache_under_a_hand_written_mode(field, roles, disaggregation)
    _reject_a_policy_the_mode_cannot_apply(disaggregation)
    _reject_router_params_the_platform_owns(roles, disaggregation)

    # A recipe injects one engine's connector configuration into every role,
    # so a role on a different engine would receive settings it cannot read.
    #
    # Read off the recipe rather than a table restating it. The catalog is
    # already loaded by the time this line runs -- the two rejections above
    # resolve the same mode -- so the second source of truth was buying a
    # dict lookup and owing a start-up assertion that the two still agreed.
    mode = get_pd_mode(disaggregation.mode.value)
    permitted = list(mode.backends) if mode else []
    if permitted:
        for role in roles:
            role_backend = role.backend or field("backend")
            if role_backend and role_backend not in permitted:
                raise BadRequestException(
                    message=(
                        f"Role '{role.name}' runs backend '{role_backend}', which "
                        f"pd mode '{disaggregation.mode.value}' cannot configure "
                        f"(it targets {', '.join(permitted)}). Mixing engines "
                        f"across roles requires pd mode 'custom', where the "
                        f"connection parameters are yours to supply."
                    )
                )


def _reject_router_params_the_platform_owns(roles, disaggregation) -> None:
    """A router parameter that would collide with an injected one.

    The router's tunable flags are meant to be overridden — appending them is
    last-wins, verified against both shipped wheels. The connection flags are
    not, and refusing them is not tidiness:

    - ``--prefill`` / ``--decode`` are ``action="append"`` in both routers, so
      a second one does not replace the injected peer. It adds one the router
      then forwards to and cannot reach, and the only symptom is a member that
      quietly never gets traffic.
    - ``--host`` / ``--port`` / ``--prometheus-*`` are last-wins, which is
      worse in a different way: the router comes up bound somewhere the
      gateway and the metrics scraper are not looking.

    The list is read off the recipe rather than written here, so adding a mode
    cannot forget to extend it.
    """
    router = next(
        (
            r
            for r in roles
            if r.name == RoleNameEnum.ROUTER.value and r.backend_parameters
        ),
        None,
    )
    if router is None:
        return
    mode = get_pd_mode(disaggregation.mode.value)
    if mode is None or mode.router is None:
        return
    owned = set(mode.router.platform_owned_flags)
    if not owned:
        return
    for param in router.backend_parameters:
        # Both spellings a user can write: `--flag value` and `--flag=value`.
        name = str(param).split("=", 1)[0].strip()
        if name in owned:
            raise BadRequestException(
                message=(
                    f"'{name}' on the router is set by GPUStack from where the "
                    f"group was placed, so it cannot be given here. Adjustable "
                    f"router flags for this mode: "
                    f"{', '.join(a.flag for a in mode.router.tunable_args) or 'none'}."
                )
            )


def _mode_renders_connector_key(mode, key: str) -> bool:
    """Whether any of the mode's roles renders `key` into its connector
    descriptor.

    Reads the descriptors rather than searching the serialized mode for
    `{{key}}`: a full-text search over `model_dump_json()` also matches the
    word appearing in a description, and misses a recipe that supplies a
    literal value instead of a placeholder. Both are the wrong answer for the
    question being asked, which is whether the user's value reaches the engine.
    """
    for role in (mode.roles or {}).values():
        if key in (role.connector or {}):
            return True
    return False


def _reject_a_policy_the_mode_cannot_apply(disaggregation) -> None:
    """`kv_load_failure_policy` is a vLLM/NIXL setting, not a platform one.

    Only `vllm-nixl` renders it. The SGLang modes have no equivalent concept
    at all -- their KV lifecycle is a bootstrap timeout that aborts the
    request, not a load that can fail and be retried -- and Mooncake's
    connector does not read the key. So there is nothing to implement on the
    other three; what there is, is a value the user weighed and set that then
    quietly does nothing.

    Which is why this rejects rather than warns, and only for a non-default
    value. `fail` is what an engine that never sees the setting does anyway,
    so refusing it would break every group on those modes to no purpose;
    `recompute` is the deliberate choice -- trade a 500 for a silent
    recomputation -- and a user who made it and got neither is worse off than
    one who was told the mode cannot honour it.

    Derived from the recipe rather than a list of mode names: a mode that
    starts rendering the key is accepted the moment it does, with nothing here
    to remember to update.
    """
    from gpustack.schemas.models import DisaggregationSpec

    policy = disaggregation.kv_load_failure_policy
    default = DisaggregationSpec.model_fields["kv_load_failure_policy"].default
    if policy == default:
        return

    mode_name = disaggregation.mode.value
    mode = get_pd_mode(mode_name)
    if mode is None or _mode_renders_connector_key(mode, "kv_load_failure_policy"):
        return

    if disaggregation.mode == PDModeEnum.CUSTOM:
        # The one mode where the setting may well be reachable, just not from
        # here: `custom` injects nothing, so every connector key is the user's
        # to write. Pointing them at another mode would be the wrong advice.
        raise BadRequestException(
            message=(
                f"pd mode 'custom' injects no connector configuration, so "
                f"kv_load_failure_policy='{policy}' would be stored and never "
                f"reach the engine. Set it inside your own "
                f"--kv-transfer-config instead."
            )
        )

    raise BadRequestException(
        message=(
            f"pd mode '{mode_name}' cannot apply kv_load_failure_policy="
            f"'{policy}': its KV connector has no such setting, so the value "
            f"would be stored and never reach the engine. Leave it at "
            f"'{default}', or use a mode whose connector reads it."
        )
    )


def _reject_lora_under_disaggregation(field) -> None:
    """LoRA adapters and disaggregation cannot be asked for together yet.

    **The combination deploys and then refuses every request**, which is why
    it is refused here instead of documented. The group reaches `running` with
    all three members, the adapter's route `<base>:<adapter>` is created and
    reports a ready target, a chat against the base name answers 200 — and a
    chat against the adapter name answers 503 `No available workers`. The
    engines are innocent: prefill and decode each list the adapter in their own
    `GET /v1/models` and each answer a direct chat on it.

    The wall is the router in front of the group. Its worker registry is indexed
    by served-model name, and a member registers itself under `model_id:
    "{{model_name}}"` — the base name, once. So the registry has no entry under
    the adapter's name to hand the request to, and the honest reading of that is
    that adapter names are not part of a group's addressing scheme at all.
    Making them so is a change to the router's membership protocol, not a field
    on this model.

    Judged against the merged state, so neither direction of an update slips
    past: adding adapters to a group that already disaggregates, and adding
    disaggregation to a model that already carries adapters, are the same
    combination arriving from opposite sides.

    Admission-time only. A row that already holds both keeps reconciling — this
    is never consulted outside create and update — because refusing to converge
    a deployment that exists would take away the running base model too, and
    the base model is the part that works.
    """
    if not field("lora_list"):
        return

    raise BadRequestException(
        message=(
            "A disaggregated deployment cannot serve LoRA adapters. A group is "
            "addressed through its router, whose member table is indexed by "
            "served-model name, and its members register under the base model's "
            "name only — so a request naming an adapter reaches no member and "
            "fails with 'No available workers', even though both engines have "
            "the adapter loaded. Either remove the adapters from this "
            "deployment and serve them from a non-disaggregated one, which is "
            "unaffected, or remove the disaggregation."
        )
    )


def _reject_an_engine_that_cannot_be_disaggregated(field, roles) -> None:
    """An engine PD does not apply to, whatever the mode.

    Distinct from the per-mode engine check further up, which asks whether a
    *recipe* can be injected into a role and lets `custom` through because it
    injects nothing. This one asks whether the engine can be disaggregated at
    all, and `custom` is not an exemption from it: supplying the connection
    parameters yourself does not give an engine a prompt KV cache to hand
    across. Without this, `custom` mode is a way around the check entirely --
    the `custom` recipe declares no `backends`, so the loop below does not run.

    Judged per role and against the model-level engine each role falls back to,
    because a group is only as disaggregable as the engine each member runs.
    """
    offenders = {
        role.backend or field("backend")
        for role in roles
        if (role.backend or field("backend")) not in PD_BACKENDS
    }
    offenders.discard(None)
    if not offenders:
        return

    raise BadRequestException(
        message=(
            f"Backend {', '.join(sorted(offenders))} cannot be disaggregated. "
            f"Prefill/decode splits the prompt KV cache across two engines, "
            f"which only {', '.join(PD_BACKENDS)} do here. Deploy this model "
            f"without disaggregation, or switch it to one of those backends."
        )
    )


def _reject_cache_under_a_hand_written_mode(field, roles, disaggregation) -> None:
    """`custom` mode and an extended KV cache cannot be asked for together.

    Everywhere else the two compose: GPUStack folds the mode's connector and
    the cache's into one `MultiConnector`, which is what makes a disaggregated
    deployment with a shared cache a supported combination rather than a
    choice between them.

    `custom` is the one mode that injects no connection state at all — its
    whole contract is that the parameters are the user's. So there is nothing
    to compose the cache with, and quietly injecting a connector under a mode
    that promises not to would be the surprise this rejection exists to
    prevent. Written by hand, both still fit in one flag; the engine composes
    connectors and the user is the one holding the pen.
    """
    if disaggregation.mode != PDModeEnum.CUSTOM:
        return

    model_cache = field("extended_kv_cache")
    for role in roles:
        cache = (
            role.extended_kv_cache
            if role.extended_kv_cache is not None
            else model_cache
        )
        if cache is None or not getattr(cache, "enabled", False):
            continue
        raise BadRequestException(
            message=(
                f"Role '{role.name}' enables the extended KV cache under pd "
                "mode 'custom', which injects no connector configuration at "
                "all — so there is nothing for GPUStack to compose the cache "
                "into. Either choose a pd mode that configures a connector, "
                "where the two are combined for you, or keep 'custom' and "
                "write the combined configuration into backend_parameters."
            )
        )


# The tables the two sides are compared through live in `server.pd_pairing`,
# because the placement-time check and the `pairing_unverified` marker read
# exactly the same ones. Keeping a second copy here is how the underscore
# spellings went missing from one of them.


def _toggle_enabled(toggle, parameters: List[str]) -> bool:
    """Where a role's parameters leave one boolean engine switch.

    Last spelling wins, matching argparse, so a role carrying both flags is
    read the way the engine would read it rather than the way the list happens
    to be ordered.
    """
    enabled = toggle.default_enabled
    for token in parameters:
        name = token.split("=", 1)[0]
        if name == toggle.enable_flag:
            enabled = True
        elif name == toggle.disable_flag:
            enabled = False
    return enabled


def _check_pairing_toggles(field, prefill_params, decode_params) -> None:
    """Refuse a pair whose two roles land a declared boolean switch differently.

    The switches are declared per backend in `pd-modes.yaml` under
    `pairing_toggles` rather than written here, for the same reason the
    value-carrying parameters live in `server/pd_pairing.py`'s tables: which
    settings must match is a fact about the engine, and a fact kept in code is
    one nobody finds when adding the next engine.
    """
    from gpustack.schemas.models import BackendEnum
    from gpustack.server.pd_mode_catalog import get_pairing_toggles

    # A deployment that names no backend runs vLLM, the same default
    # `get_backend` applies once the row exists.
    backend = field("backend") or BackendEnum.VLLM
    for toggle in get_pairing_toggles(backend):
        if _toggle_enabled(toggle, prefill_params) == _toggle_enabled(
            toggle, decode_params
        ):
            continue
        detail = toggle.description or f"the two roles disagree on {toggle.key}"
        raise BadRequestException(
            message=(
                f"prefill and decode disagree on {toggle.key}: {detail}. The "
                f"pair is then rejected on contact and the group never serves. "
                f"Note that a KV connector may set this on its own — the "
                f"divergence comes from one role carrying "
                f"{toggle.enable_flag} and the other not."
            )
        )


def _check_tensor_parallel_pairing(disaggregation, *, prefill_tp, decode_tp) -> None:
    """Apply the recipe's declared tensor-parallel direction.

    The direction belongs to the KV connector, so it is read off the mode
    (`PDMode.pairing.tensor_parallel`) rather than written here: NIXL needs
    decode at least as wide as prefill, vllm-ascend's Mooncake needs the
    opposite (Huawei's reference deployment is prefill TP4 / decode TP1), and
    `custom` injects no connector GPUStack knows. A mode the catalog cannot
    resolve is held to the NIXL rule, which is what every mode was held to
    before the rule became declarable.
    """
    mode_name = disaggregation.mode.value
    rule = tensor_parallel_rule(get_pd_mode(mode_name))
    if not violates_tensor_parallel_direction(
        rule, prefill_tp=prefill_tp, decode_tp=decode_tp
    ):
        return

    if rule == PDTensorParallelPairingEnum.DECODE_GE_PREFILL:
        raise BadRequestException(
            message=(
                f"decode runs tensor parallelism {decode_tp}, below prefill's "
                f"{prefill_tp}. A decode narrower than its prefill cannot "
                f"receive that prefill's KV layout, and the engine reports it "
                f"as an IndexError inside decode rather than as a "
                f"configuration error. decode's tensor parallelism must be at "
                f"least prefill's."
            )
        )
    raise BadRequestException(
        message=(
            f"prefill runs tensor parallelism {prefill_tp}, below decode's "
            f"{decode_tp}. pd mode '{mode_name}' gathers each decode rank's "
            f"KV from prefill ranks, which needs prefill's tensor "
            f"parallelism to be at least decode's."
        )
    )


def validate_role_pairing(  # noqa: C901
    model_in: Union[ModelCreate, ModelUpdate, ModelSpecBase],
    stored: Optional[Model] = None,
) -> None:
    """Reject prefill/decode pairs the engines will accept and serve wrongly.

    The division of labour with the engine is deliberate and documented in X1
    3.1: most handshake factors are hashed by the connector and rejected on
    contact, so re-checking them here buys attribution, not safety. Two are
    different.

    `max_model_len` is checked by nothing at all. Measured with prefill at 8192
    and decode at 4096: the handshake passes, KV transfers, short prompts
    answer normally, and only a prompt above decode's window fails — with a 400
    from decode, after prefill has already computed it. The user is left
    believing the deployment serves 8192.

    Tensor parallelism is asserted by the engine at run time, but a decode
    narrower than its prefill surfaces as an `IndexError` inside decode rather
    than as a configuration error, so the hard block is worth more than the
    assertion.

    **Every rule here refuses only what it can prove.** The test is whether
    both sides' effective values are determinable from the spec, not whether
    both sides typed the parameter — `server.pd_pairing` holds that
    distinction and says why substituting the engines' defaults instead would
    have been wrong for every one of them. A factor one side left silent is
    not decided here and not refused here; the deployment carries
    `pairing_unverified` and says so.

    That asymmetry is the point on this path in particular. `evaluate_model_input`
    turns a refusal into the red compatibility error in the deploy form, so a
    rule that guesses does not produce a 400 an operator can argue with — it
    produces a form that refuses to submit a deployment which would have run.

    This is a pre-check, not a mirror of the engine's factor set — vLLM's own
    source says that set is "likely to evolve significantly over time", so the
    engine stays the final judge.
    """

    def field(name: str):
        submitted = getattr(model_in, name, None)
        if submitted is not None:
            return submitted
        if stored is not None and name not in getattr(
            model_in, "model_fields_set", set()
        ):
            return getattr(stored, name, None)
        return submitted

    roles = field("roles")
    if not roles or not field("disaggregation"):
        return

    model_parameters = field("backend_parameters")
    prefill = next((r for r in roles if r.name == RoleNameEnum.PREFILL.value), None)
    decode = next((r for r in roles if r.name == RoleNameEnum.DECODE.value), None)
    if prefill is None or decode is None:
        return

    prefill_params = role_parameters(prefill, model_parameters)
    decode_params = role_parameters(decode, model_parameters)

    verdict, prefill_len, decode_len = compare_max_model_len(
        prefill_params, decode_params
    )
    if verdict == DIFFER:
        raise BadRequestException(
            message=(
                f"prefill and decode declare different context lengths "
                f"({prefill_len} vs {decode_len}). No engine checks this: "
                f"the pair handshakes, transfers KV and answers short "
                f"prompts, and a prompt above "
                f"{min(prefill_len, decode_len)} tokens fails at decode "
                f"after prefill has already computed it. Give both roles "
                f"the same context length."
            )
        )

    # Effective, not declared. A role that writes no tp is not a role with an
    # unknown one when it pins its own cards: the backend injects the card
    # count for a single-worker member, so a decode pinned to one card beside a
    # prefill at TP2 is a real inversion and worth catching. A role that pins
    # nothing stays unknown and is reported as unverified rather than
    # held to a 1 it was never going to run.
    prefill_tp = effective_tensor_parallelism(prefill, prefill_params)
    decode_tp = effective_tensor_parallelism(decode, decode_params)
    if prefill_tp is not None and decode_tp is not None:
        _check_tensor_parallel_pairing(
            field("disaggregation"), prefill_tp=prefill_tp, decode_tp=decode_tp
        )

    _check_pairing_toggles(field, prefill_params, decode_params)

    for label, names in PAIRING_MUST_MATCH.items():
        if compare_must_match(label, names, prefill_params, decode_params) != DIFFER:
            continue
        prefill_value = find_last_parameter(prefill_params, names)
        decode_value = find_last_parameter(decode_params, names)
        raise BadRequestException(
            message=(
                f"prefill and decode declare different {label} "
                f"('{prefill_value}' vs '{decode_value}'). The connector "
                f"rejects the pair on contact, so the group would never "
                f"serve; the roles must agree."
            )
        )


async def validate_model_in(
    session: SessionDep,
    model_in: Union[ModelCreate, ModelUpdate, ModelSpecBase],
    *,
    cluster_id: Optional[int] = None,
    stored: Optional[Model] = None,
):
    # `stored` is the row being updated, so a sparse PUT is judged against the
    # merged state rather than against the handful of fields it happened to
    # send. Absent on create, where there is nothing to merge.
    validate_roles(model_in, stored=stored)
    validate_role_pairing(model_in, stored=stored)
    await validate_pd_mode_runtime(
        session, model_in, cluster_id=cluster_id, stored=stored
    )
    await validate_gather_layer(session, model_in, cluster_id=cluster_id, stored=stored)

    if getattr(model_in, "gpu_type_selector", None) is not None:
        await validate_gpu_type_selector(session, model_in, cluster_id=cluster_id)

    if model_in.gpu_selector is not None and _max_intended_replicas(model_in) > 0:
        await validate_gpu_ids(session, model_in, cluster_id=cluster_id)

    if is_custom_backend(model_in.backend):
        logger.info("Skip model validation for custom backend")
        return

    if model_in.backend_parameters:
        param_gpu_layers = find_parameter(
            model_in.backend_parameters, ["ngl", "gpu-layers", "n-gpu-layers"]
        )

        if param_gpu_layers:
            int_param_gpu_layers = safe_int(param_gpu_layers, None)
            if (
                not param_gpu_layers.isdigit()
                or int_param_gpu_layers < 0
                or int_param_gpu_layers > 999
            ):
                raise BadRequestException(
                    message="Invalid backend parameter --gpu-layers. Please provide an integer in the range 0-999 (inclusive)."
                )

            if (
                int_param_gpu_layers == 0
                and model_in.gpu_selector is not None
                and len(model_in.gpu_selector.gpu_ids) > 0
            ):
                raise BadRequestException(
                    message="Cannot set --gpu-layers to 0 and manually select GPUs at the same time. Setting --gpu-layers to 0 means running on CPU only."
                )

        unsupported_params = [
            (
                ["port"],
                (
                    "Setting the port using --port is not supported. Ports are "
                    "automatically allocated by GPUStack."
                ),
            ),
            (
                ["api-key"],
                (
                    "Setting the API key using --api-key is not supported. API keys "
                    "are managed by GPUStack."
                ),
            ),
            (
                ["served-model-name"],
                (
                    "Setting the served model name using --served-model-name is not "
                    "supported. The model name is automatically set from your "
                    "deployment configuration."
                ),
            ),
        ]

        for param_names, error_message in unsupported_params:
            if find_parameter(model_in.backend_parameters, param_names):
                raise BadRequestException(message=error_message)

    validate_and_normalize_lora_list(model_in)


def validate_and_normalize_lora_list(
    model_in: Union[ModelCreate, ModelUpdate, ModelSpecBase],
) -> None:
    """Normalize each lora_name to the stored "<base>:<short>" form.

    Accepts a bare short name and prepends the base prefix; a correct "<base>:"
    prefix is kept as-is. Rejects wrong prefixes, embedded colons, empty names,
    and duplicates. The API strips the prefix again on the way out (see
    ModelPublic._strip_lora_prefix).
    """
    lora_list = getattr(model_in, "lora_list", None)
    if not lora_list:
        return

    expected_prefix = f"{model_in.name}:"
    seen: set = set()
    for i, item in enumerate(lora_list):
        entry = LoraListEntry.model_validate(item) if isinstance(item, dict) else item
        short_name = (entry.lora_name or "").strip()
        if not short_name:
            raise BadRequestException(
                message="lora_name must not be empty in lora_list."
            )
        if ":" in short_name:
            if not short_name.startswith(expected_prefix):
                raise BadRequestException(
                    message=(
                        f"lora_name '{short_name}' must not contain ':'. Set "
                        f"lora_name to the bare adapter name (e.g. 'my-adapter'); "
                        f"the '{expected_prefix}' prefix is added automatically."
                    )
                )
            short_name = short_name[len(expected_prefix) :]
            if not short_name:
                raise BadRequestException(
                    message=(
                        f"lora_name is missing the suffix after the base model "
                        f"prefix '{expected_prefix}'."
                    )
                )
            if ":" in short_name:
                raise BadRequestException(
                    message=(
                        f"lora_name '{entry.lora_name}' must not contain a nested "
                        f"':' after the base model prefix."
                    )
                )
        entry.lora_name = lora_route_name_for(model_in.name, short_name)
        lora_list[i] = entry
        if short_name in seen:
            raise BadRequestException(
                message=f"Duplicate lora_name '{short_name}' in lora_list."
            )
        seen.add(short_name)


async def validate_pd_mode_runtime(
    session: SessionDep,
    model_in: Union[ModelCreate, ModelUpdate, ModelSpecBase],
    *,
    cluster_id: Optional[int] = None,
    stored: Optional[Model] = None,
):
    """Reject a pd mode no accelerator in the cluster can run.

    Every built-in recipe is accelerator-specific: `vllm-ascend-mooncake`
    injects an Ascend-only connector plus HCCL variables, and the NVIDIA
    recipes inject connectors no other runtime can read. Injecting one into
    the wrong accelerator fails inside the connector rather than at submit
    time. `PDModeRuntimeFilter` also drops the mismatched workers during
    scheduling; this check exists so the answer is a readable refusal instead
    of an empty candidate list.

    `custom` is never rejected -- it declares no `gpu_filters`, so an
    unsupported engine × accelerator pair means "no built-in recipe", never
    "no PD".

    Shares `resolve_pd_mode` with the resolve endpoint the form reads, so the
    API cannot refuse a combination the form just told the user was fine.

    Accelerator-less clusters are left to scheduling: a cluster whose workers
    have not reported devices yet must not be judged as unable to run
    anything.
    """
    disaggregation = getattr(model_in, "disaggregation", None)
    if disaggregation is None and stored is not None:
        if "disaggregation" not in getattr(model_in, "model_fields_set", set()):
            disaggregation = getattr(stored, "disaggregation", None)
    if not disaggregation:
        return

    mode_name = disaggregation.mode.value
    mode = get_pd_mode(mode_name)
    if mode is None or mode.gpu_filters is None or not mode.gpu_filters.vendor:
        return

    effective_cluster_id = cluster_id or getattr(model_in, "cluster_id", None)
    vendors = await cluster_vendors(session, effective_cluster_id)
    if not vendors:
        return

    backend = getattr(model_in, "backend", None) or (
        getattr(stored, "backend", None) if stored else None
    )
    resolution = resolve_pd_mode(
        backend, vendors, vendor=getattr(disaggregation, "vendor", None)
    )
    verdict = next(
        (option for option in resolution.options if option.name == mode_name), None
    )
    if verdict is not None and not verdict.eligible:
        raise BadRequestException(
            message=(
                f"pd mode '{mode_name}' cannot run here: "
                f"{verdict.ineligible_reason}"
            )
        )


async def validate_gather_layer(
    session: SessionDep,
    model_in: Union[ModelCreate, ModelUpdate, ModelSpecBase],
    *,
    cluster_id: Optional[int] = None,
    stored: Optional[Model] = None,
):
    """Refuse a `gather.layer` this cluster has no rung for.

    The failure this prevents is silence, not a crash. The solver's
    `_enforced_gather` stands an unknown layer down and places the group as if
    nothing had been asked for — correct behaviour there (a layer renamed under
    a *running* deployment must not take it down) and exactly the wrong
    behaviour at submit time, where it would accept a `MustGather` under a
    promise nothing enforces.

    It is checked here rather than on `GatherSpec` because the answer depends
    on the cluster: the layer names are the cluster's own declaration, which a
    field validator on the model has no access to. `accelerator_domain` is
    subject to the same rule as every other name — valid only for a cluster
    that actually declared a layer called that.

    A cluster that cannot be read is left alone rather than guessed at:
    scheduling still stands the requirement down, so the cost is a missed
    refusal, not a wrong one.
    """
    gather = getattr(model_in, "gather", None)
    if gather is None and stored is not None:
        if "gather" not in getattr(model_in, "model_fields_set", set()):
            gather = getattr(stored, "gather", None)
    layer = getattr(gather, "layer", None)
    if not layer:
        return

    from gpustack.schemas.clusters import Cluster
    from gpustack.topology.tree import NODE_LAYER, TopologyError
    from gpustack.topology.vocabulary import (
        gather_layer_names,
        validate_declaration,
    )

    if layer == NODE_LAYER:
        await _refuse_a_floor_no_member_can_meet(session, model_in, cluster_id, stored)
        return

    effective_cluster_id = cluster_id or getattr(model_in, "cluster_id", None)
    if effective_cluster_id is None:
        return
    cluster = await Cluster.one_by_id(session, effective_cluster_id)
    if cluster is None:
        return

    try:
        names = gather_layer_names(validate_declaration(cluster.topology))
    except TopologyError:
        # The cluster's own declaration is broken; refusing the *model* for it
        # would send the operator to the wrong page.
        return
    if layer not in names:
        raise BadRequestException(
            message=(
                f"gather layer {layer!r} is not a layer of this cluster. "
                f"Available: {', '.join(names)}."
            )
        )


async def _refuse_a_floor_no_member_can_meet(
    session: SessionDep,
    model_in: Union[ModelCreate, ModelUpdate, ModelSpecBase],
    cluster_id: Optional[int],
    stored: Optional[Model],
):
    """Refuse «all members on one machine» when a member cannot fit on one.

    **The one contradiction cross-machine members introduce.** A floor at the
    host rung says every member of the group sits on a single machine; a role
    whose tensor-parallel width exceeds any machine in the cluster needs two.
    Both are things the operator asked for, and no placement satisfies them
    together.

    Caught here rather than left to the solver because the two answers read
    completely differently. At placement time it surfaces as a group that never
    leaves PENDING with a capacity sentence beside it -- and the cluster is not
    short of capacity, so the reader goes looking for cards that are already
    there. At submit time it is one sentence naming the two settings that
    disagree, while both are still on screen.

    Silent about everything it cannot be sure of: a parallel width the engine
    was never told, a cluster whose workers cannot be read, a backend whose
    width this does not know how to ask for. A refusal is only worth issuing
    when the contradiction is certain.
    """
    gather = getattr(model_in, "gather", None) or (
        getattr(stored, "gather", None) if stored is not None else None
    )
    if getattr(gather, "strategy", None) != GatherStrategyEnum.MUST_GATHER:
        # `PreferGather` at the host rung is a target, not a promise. A member
        # that has to span simply misses it and says so afterwards.
        return

    spanning, widest = await roles_that_must_span(session, model_in, cluster_id)
    for name, width in spanning:
        raise BadRequestException(
            message=(
                f"Role {name!r} needs {width} GPUs and the widest worker "
                f"in this cluster has {widest}, so it has to span machines — "
                f"but this deployment also asks for every member to be on one "
                f"machine, and to be refused rather than placed outside it. "
                f"Lower the parallel size, or choose a looser topology floor."
            )
        )


async def roles_that_must_span(
    session: SessionDep,
    model_in: Union[ModelCreate, ModelUpdate, ModelSpecBase],
    cluster_id: Optional[int] = None,
) -> Tuple[List[Tuple[str, int]], int]:
    """Which roles cannot fit one machine, and how wide the widest machine is.

    Arithmetic, not a placement solve — which is the whole reason it is safe to
    ask while a form is still being typed. It compares a width the operator has
    already stated against a fact about the cluster; it does not select
    candidates, does not consult free capacity, and its answer does not move
    while the rest of the form is filled in. It must not become the other
    thing: a verdict about a finished configuration, asked of a half finished
    one, which conflates «this tier does not fit» with «I cannot tell yet».

    Empty is the answer for everything it cannot be sure of — a width the
    engine was never told, a cluster whose workers cannot be read, a backend
    whose width this does not know how to ask for. Both callers treat «not
    sure» as «not spanning».
    """
    effective_cluster_id = cluster_id or getattr(model_in, "cluster_id", None)
    if effective_cluster_id is None:
        return [], 0

    # Widths first: a deployment that never stated one cannot span anything,
    # and asking the cluster about it would be a query for nothing.
    widths = [
        (spec.name, _role_gpu_width(model_in, spec.name))
        for spec in getattr(model_in, "roles", None) or []
    ]
    widths = [(name, width) for name, width in widths if width]
    if not widths:
        return [], 0

    workers = await Worker.all_by_field(session, "cluster_id", effective_cluster_id)
    widest = max(
        (len((w.status.gpu_devices or []) if w.status else []) for w in workers),
        default=0,
    )
    if widest == 0:
        return [], 0

    return [(name, width) for name, width in widths if width > widest], widest


def _role_gpu_width(model_in, role_name: str) -> Optional[int]:
    """How many GPUs one member of this role wants, or None if not stated.

    Asked of the selectors rather than re-derived: they already turn a mixed
    bag of `--tensor-parallel-size` / `--tp-size` / `--pipeline-parallel-size`
    / data parallelism into a world size, per engine, and a second reading of
    those flags here is a second answer to one question.
    """
    from gpustack.policies.candidate_selectors import (
        SGLangResourceFitSelector,
        VLLMResourceFitSelector,
    )
    from gpustack.schemas.models import get_backend

    try:
        projected = role_effective_model(model_in, role_name)
        backend = get_backend(projected)
    except Exception:
        return None
    selectors = {
        BackendEnum.VLLM: VLLMResourceFitSelector,
        BackendEnum.SGLANG: SGLangResourceFitSelector,
    }
    selector = selectors.get(backend)
    if selector is None:
        return None
    try:
        world_size, _strategies = selector.get_world_size_from_backend_parameters(
            projected
        )
    except Exception:
        return None
    return world_size


async def validate_gpu_type_selector(
    session: SessionDep,
    model_in: Union[ModelCreate, ModelUpdate, ModelSpecBase],
    *,
    cluster_id: Optional[int] = None,
):
    """Validate a model's ``gpu_type_selector`` against the local projection.

    Reads only the local ``GPUInstanceType`` projection table (synced from the
    operator watch stream); no Kubernetes client calls. Fails closed: anything
    that cannot be verified from the projection is rejected.
    """
    selector = model_in.gpu_type_selector

    gpu_selector = model_in.gpu_selector
    if gpu_selector is not None and gpu_selector.gpu_ids:
        raise BadRequestException(
            message="gpu_type_selector cannot be combined with gpu_selector: "
            "manual GPU selection and InstanceType-based sliced GPU selection "
            "are mutually exclusive."
        )

    # Narrowed to the slicing and partition modes, and it needs to be.
    #
    # A slice is a fraction of one card the node's device plugin picks at
    # allocation time, so "more than one" has no meaning: the caller cannot say
    # which card the first one landed on. A *whole-card* claim has no such
    # difficulty — the operator's resource model hands out several at once, and
    # the container sees exactly the devices allocated, so an engine told tp=4
    # finds four. Refusing that would refuse something the layer below can do.
    #
    # Note this check does not fire on the common path:
    # `set_model_gpus_per_replica` returns early unless `gpu_selector.gpu_ids`
    # is set, and manual ids are mutually exclusive with `gpu_type_selector`
    # above — so `gpus_per_replica` is `None` for every InstanceType claim.
    # Where it does fire it prevents a claim being accepted and then scheduled
    # onto one card while the engine expected several, which is worse than a
    # refusal. Whole-card multi-card is handled by
    # `InstanceTypeWholeCardSelector`.
    sliced_or_partitioned = (
        (selector.accelerator_sliced_memory_percentage or 0) > 0
        or (selector.accelerator_sliced_cores_percentage or 0) > 0
        or bool(selector.accelerator_partitioned_profile)
    )
    if (
        sliced_or_partitioned
        and gpu_selector is not None
        and gpu_selector.gpus_per_replica is not None
        and gpu_selector.gpus_per_replica > 1
    ):
        raise BadRequestException(
            message="gpus_per_replica must be 1 when a sliced or partitioned "
            "gpu_type_selector is set: one slice is a fraction of one card, so "
            "asking for several has no meaning. Use a whole-card claim (all "
            "slicing percentages zero) for a member that needs several cards."
        )

    memory_pct = selector.accelerator_sliced_memory_percentage
    cores_pct = selector.accelerator_sliced_cores_percentage
    memory_sliced = memory_pct is not None and memory_pct > 0
    cores_sliced = cores_pct is not None and cores_pct > 0
    if (memory_sliced or cores_sliced) and selector.accelerator_partitioned_profile:
        raise BadRequestException(
            message="accelerator_partitioned_profile cannot be combined with "
            "accelerator_sliced_memory_percentage or "
            "accelerator_sliced_cores_percentage: hardware partitioning and "
            "software slicing cannot both apply to one card."
        )

    effective_cluster_id = (
        cluster_id if cluster_id is not None else getattr(model_in, "cluster_id", None)
    )
    if effective_cluster_id is None:
        raise BadRequestException(
            message="A cluster must be specified when gpu_type_selector is set: "
            "the InstanceType projection is scoped per cluster."
        )

    matched_list = await GPUInstanceType.all_by_fields(
        session,
        fields={
            "cluster_id": effective_cluster_id,
            "deleted_at": None,
            "name": selector.type,
        },
    )
    if not matched_list:
        # Keep the two errors distinct: no synced types in the cluster at
        # all vs. this type missing.
        instance_types = await GPUInstanceType.all_by_fields(
            session,
            fields={"cluster_id": effective_cluster_id, "deleted_at": None},
        )
        if not instance_types:
            raise BadRequestException(
                message=f"Cluster {effective_cluster_id} has no synced GPU InstanceTypes: "
                "gpu_type_selector requires a Kubernetes cluster managed by "
                "gpustack-operator."
            )
        raise BadRequestException(
            message=f"GPU InstanceType '{selector.type}' not found in cluster "
            f"{effective_cluster_id}."
        )
    matched = matched_list[0]

    # A cluster also publishes non-accelerated InstanceTypes (a CPU-only pool),
    # and nothing about their name says so. The deploy form filters them out of
    # its GPU Type list, but an API caller can still name one — and it would
    # only fail later at scheduling, reported as a type that "does not report
    # its accelerator memory" rather than as the wrong kind of type.
    if not matched.spec.acceleratable:
        raise BadRequestException(
            message=f"GPU InstanceType '{selector.type}' in cluster "
            f"{effective_cluster_id} is not an accelerator type: "
            "gpu_type_selector requires one backed by GPUs."
        )

    # spec is projected as soon as the InstanceType appears, but status.detail is
    # backfilled by the operator afterwards, so a type can be nameable before its
    # hardware is known. Every mode needs the card's memory to size the claim
    # (a percentage of it, a profile out of it, or the whole card), so without it
    # the model would be accepted here and then never schedule — the fit would
    # report the type as unavailable or as not reporting its memory.
    detail = matched.status.detail if matched.status else None
    if detail is None or not detail.memory:
        raise BadRequestException(
            message=f"GPU InstanceType '{selector.type}' in cluster "
            f"{effective_cluster_id} does not report its accelerator memory yet: "
            "gpustack-operator has not finished backfilling the type. Retry once "
            "the type reports its hardware detail."
        )

    if selector.accelerator_partitioned_profile:
        sliced_detail = detail.sliced_detail
        physical = sliced_detail.physical if sliced_detail else None
        profiles = physical.profiles if physical and physical.profiles else []
        profile_names = {p.name for p in profiles if p.name}
        if selector.accelerator_partitioned_profile not in profile_names:
            raise BadRequestException(
                message=f"Profile '{selector.accelerator_partitioned_profile}' "
                f"is not offered by GPU InstanceType '{selector.type}' in "
                f"cluster {effective_cluster_id}. Available profiles: "
                f"{sorted(profile_names) or 'none'}."
            )


async def validate_gpu_ids(  # noqa: C901
    session: SessionDep,
    model_in: Union[ModelCreate, ModelUpdate, ModelSpecBase],
    *,
    cluster_id: Optional[int] = None,
):
    effective_cluster_id = (
        cluster_id if cluster_id is not None else getattr(model_in, "cluster_id", None)
    )

    # A selector can legitimately carry `gpus_per_replica` and no `gpu_ids`:
    # that's what it looks like when the card is picked by the operator's
    # device plugin rather than by index, which is also the shape of a
    # per-role selector. Everything below reads `gpu_ids` as a sequence, so
    # normalise it once here instead of guarding at each use.
    gpu_ids = model_in.gpu_selector.gpu_ids or []

    if gpu_ids and model_in.gpu_selector.gpus_per_replica:
        if len(gpu_ids) < model_in.gpu_selector.gpus_per_replica:
            raise BadRequestException(
                message="The number of selected GPUs must be greater than or equal to gpus_per_replica."
            )

    model_backend = model_in.backend

    if model_backend == BackendEnum.VOX_BOX and (
        len(gpu_ids) > 1
        or (
            model_in.gpu_selector.gpus_per_replica is not None
            and model_in.gpu_selector.gpus_per_replica > 1
        )
    ):
        raise BadRequestException(
            message="The vox-box backend is restricted to execution on a single NVIDIA GPU."
        )

    worker_name_set = set()
    for gpu_id in gpu_ids:
        is_valid, matched = parse_gpu_id(gpu_id)
        if not is_valid:
            raise BadRequestException(message=f"Invalid GPU ID: {gpu_id}")

        worker_name = matched.get("worker_name")
        gpu_index = safe_int(matched.get("gpu_index"), -1)
        worker_name_set.add(worker_name)

        if effective_cluster_id is None:
            raise BadRequestException(
                message=f"A cluster context is required for manual GPU selection, but was not provided. Cannot validate worker '{worker_name}'."
            )

        worker = await WorkerService(session).get_by_cluster_id_name(
            effective_cluster_id, worker_name
        )
        if not worker:
            raise BadRequestException(message=f"Worker {worker_name} not found")

        gpu = (
            next(
                (gpu for gpu in worker.status.gpu_devices if gpu.index == gpu_index),
                None,
            )
            if worker.status and worker.status.gpu_devices
            else None
        )
        if gpu:
            validate_gpu(gpu, model_backend=model_backend)

        if model_backend == BackendEnum.VLLM and len(worker_name_set) > 1:
            await validate_distributed_vllm_limit_per_worker(session, model_in, worker)

    if (
        is_custom_backend(model_backend)
        and len(worker_name_set) > 1
        and model_in.replicas == 1
    ):
        raise BadRequestException(
            message="Distributed inference across multiple workers is not supported for custom backends."
        )


def validate_gpu(gpu_device: GPUDeviceStatus, model_backend: str = ""):
    if (
        model_backend == BackendEnum.VOX_BOX
        and gpu_device.vendor != ManufacturerEnum.NVIDIA.value
    ):
        raise BadRequestException(
            "The vox-box backend is supported only on NVIDIA GPUs."
        )

    if (
        model_backend == BackendEnum.ASCEND_MINDIE
        and gpu_device.vendor != ManufacturerEnum.ASCEND.value
    ):
        raise BadRequestException(
            f"Ascend MindIE backend requires Ascend NPUs. Selected {gpu_device.vendor} GPU is not supported."
        )


async def validate_distributed_vllm_limit_per_worker(
    session: AsyncSession, model: Union[ModelCreate, ModelUpdate], worker: Worker
):
    """
    Validate that there is no more than one distributed vLLM instance per worker.
    """
    instances = await ModelInstance.all_by_field(session, "worker_id", worker.id)
    for instance in instances:
        if (
            instance.distributed_servers
            and instance.distributed_servers.subordinate_workers
            and instance.model_name != model.name
        ):
            raise BadRequestException(
                message=f"Each worker can run only one distributed vLLM instance. Worker '{worker.name}' already has '{instance.name}'."
            )


async def assert_cluster_belongs_to_org(
    ctx: TenantContext,
    session: AsyncSession,
    cluster_id: Optional[int],
    owner_principal_id: int,
    cluster: Optional[Cluster] = None,
):
    """Ensure a chosen cluster is visible to the caller and owned by the
    given Org.

    A model runs on infrastructure owned by its Org, so its cluster must
    belong to that Org — otherwise a tenant could target the platform's
    (or another Org's) cluster, stamping a cross-tenant model. A cluster the
    caller can't see is reported as missing (404), so cross-tenant cluster
    ids can't be probed via a 403-vs-404 difference; a visible cluster owned
    by another Org is a 403. No cluster chosen (``cluster_id is None``)
    leaves default-cluster resolution to pick the Org's own cluster.

    ``cluster`` may be passed pre-fetched to avoid a duplicate lookup.
    """
    if cluster_id is None:
        return
    if cluster is None:
        cluster = await Cluster.one_by_id(session, cluster_id)
    not_found = f"Cluster {cluster_id} not found"
    assert_cluster_visible(ctx, cluster, not_found_message=not_found)
    if cluster.deleted_at is not None:
        raise NotFoundException(message=not_found)
    if cluster.owner_principal_id != owner_principal_id:
        raise ForbiddenException(
            message="The selected cluster does not belong to the current organization."
        )


class _CacheDeclaration(NamedTuple):
    """One extended-KV-cache configuration this deployment will actually run.

    A role-bearing deployment has more than one, because `extended_kv_cache`,
    `backend` and `backend_version` are all per-role overrides and the
    injection resolver reads them through the role's projection
    (`resolve_instance_cache_config`). Judging the Model's values alone would
    check a configuration no member runs.
    """

    role: Optional[str]
    ext: "ExtendedKVCacheConfig"
    backend: Optional[str]
    backend_version: Optional[str]

    @property
    def where(self) -> str:
        """The clause that says which member a refusal is about, empty at the
        model level so a role-less deployment's messages are unchanged."""
        return f" on role '{self.role}'" if self.role else ""


def _cache_declarations(model_in) -> List[_CacheDeclaration]:
    """Every distinct cache configuration `model_in` would deploy.

    The inherit-when-None merge is spelled out rather than taken from
    `role_effective_model`, following the rest of this module: validation runs
    on a `ModelCreate` / `ModelUpdate` / `ModelSpec`, and building a
    `RoleEffectiveModel` out of a request body to read three fields off it is a
    conversion the projection was not written for.

    Deduplicated on the three fields that decide the answer, so a group whose
    roles all inherit the Model's cache on the Model's engine is checked exactly
    once and cannot start reporting a refusal against a role name for a value
    the user wrote at the model level.
    """
    model_ext = getattr(model_in, "extended_kv_cache", None)
    model_backend = getattr(model_in, "backend", None)
    model_version = getattr(model_in, "backend_version", None)

    declarations = [
        (
            _CacheDeclaration(None, model_ext, model_backend, model_version)
            if model_ext
            else None
        )
    ]
    for role in getattr(model_in, "roles", None) or []:
        ext = (
            role.extended_kv_cache if role.extended_kv_cache is not None else model_ext
        )
        if not ext:
            continue
        declarations.append(
            _CacheDeclaration(
                role.name,
                ext,
                role.backend or model_backend,
                role.backend_version or model_version,
            )
        )

    seen = set()
    out: List[_CacheDeclaration] = []
    for declaration in declarations:
        if declaration is None:
            continue
        key = (
            declaration.ext.model_dump_json(),
            declaration.backend,
            declaration.backend_version,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(declaration)
    return out


def _reject_a_split_cache_pool(declarations: List[_CacheDeclaration]) -> None:
    """Two roles of one deployment may not attach to two different cache
    services.

    The question this answers is whether `extended_kv_cache` is a per-member
    setting or a property of the deployment, and for the *identity of the
    service* it is the latter. A shared cache is one pool that the members
    write into and read out of; naming two makes it two pools, prefill stores a
    prefix into one and decode looks for it in the other, and nothing anywhere
    reports a miss — the requests all succeed, at the hit rate the feature
    exists to raise. That is the failure mode this module refuses on principle.

    Deliberately narrower than symmetry. Whether one side may take a cache while
    the other takes none is a *different* question, and the answer there does
    not depend on this one: this rule needs two roles that are both
    shared-enabled and cannot fire on a one-sided configuration at all.

    On that other question, note which way round the risk actually runs — a
    one-sided configuration is the *safe* one. Cache on prefill alone serves
    normally and the cache does its job. Both sides enabled is the combination
    that kills the decode engine once the pool actually hits, because vLLM's
    `MultiConnector` then runs two async loads per request and only
    deduplicates saves. Nor does NIXL's compatibility hash refuse a one-sided
    pair: what that hash carries is `is_hma_enabled`, which differs between the
    two sides only when the cache connector does not support HMA — a property
    of the `lmcache` build inside the engine image, not of the configuration.
    """
    services = {
        declaration.ext.cache_service_id
        for declaration in declarations
        # `is_shared()` reads the mode, not the id, so a shared declaration may
        # still be missing its `cache_service_id` here -- the misconfiguration
        # the per-declaration check rejects with a 400. Left in, `sorted()`
        # would compare None with an int and turn that into a 500.
        if declaration.role
        and declaration.ext.is_shared()
        and declaration.ext.cache_service_id is not None
    }
    if len(services) < 2:
        return

    raise BadRequestException(
        message=(
            f"The roles of this deployment name different cache services "
            f"({', '.join(str(service) for service in sorted(services))}). A "
            f"shared KV cache is one pool the whole deployment reads and "
            f"writes; two services are two pools, so a prefix stored by one "
            f"role is never found by the other and the only symptom is a cache "
            f"that never hits. Point every role at the same cache service, or "
            f"leave the roles' extended_kv_cache unset so they inherit the "
            f"model's."
        )
    )


async def validate_shared_kv_cache(
    session: AsyncSession,
    model_in: Union[ModelCreate, ModelUpdate],
    owner_principal_id: int,
    effective_cluster_id: Optional[int],
) -> None:
    """Validate every extended-KV-cache configuration this deployment declares.

    "shared" mode attaches the model's inference engine to a CacheService
    row, so the service must exist, belong to the model's Org (a
    cross-tenant id is reported as missing so service ids can't be probed),
    run in the model's cluster (the engine connects over the cluster
    network), and have a provider that knows how to inject connector
    config for the model's backend. "local" mode uses no service, so a
    stray cache_service_id is rejected as a mis-configuration rather than
    silently ignored.

    **Every rule below applies to the role-level value as well as the
    Model's**, because `extended_kv_cache` is a per-role override: checking
    only the Model's would accept the identical configuration under
    `roles[].extended_kv_cache` in every case this function exists to refuse —
    a service id that names nothing, 'local' carrying a service id, 'shared'
    carrying none, a service in another cluster, another tenant's service, a
    provider that cannot configure the backend, an engine version under the
    provider's floor. The one role-aware check in this area,
    `_reject_cache_under_a_hand_written_mode`, does refuse a role-level
    `enabled` — a role's cache was always meant to be seen here.

    Mutates `model_in` in place: `_drop_local_only_cache_knobs` clears the
    local-cache sizing fields off a shared declaration.
    """
    declarations = _cache_declarations(model_in)
    _reject_a_split_cache_pool(declarations)
    for declaration in declarations:
        await _validate_one_cache_declaration(
            session, declaration, owner_principal_id, effective_cluster_id
        )
    _drop_local_only_cache_knobs(declarations)


def _drop_local_only_cache_knobs(declarations: List[_CacheDeclaration]) -> None:
    """Clear the local-cache sizing knobs off every shared declaration.

    `ram_size` and `ram_ratio` size the cache the *engine process* offloads
    into host memory, and only a "local" cache has one. In "shared" mode the
    cache is a separate service with its own memory, which both worker
    backends say outright by returning before they apply either knob —
    `vllm.py`'s `_set_lmcache_env` never sets `LMCACHE_MAX_LOCAL_CPU_SIZE`,
    `sglang.py` never passes `--hicache-ratio`.

    Left on the row they are not merely inert. The scheduler books host RAM
    from them (`get_computed_ram_claim`), and `ram_ratio` **defaults to 1.2**,
    so a deployment whose author only ever picked a cache service reserves
    1.2x its VRAM claim of memory that no process will take. On a tight host
    that surplus is enough to leave a group's small router without room while
    both engines fit, and the refusal names the router — the only member whose
    RAM was checked honestly.

    **Cleared rather than refused**, unlike the `cache_service_id`-under-local
    rule above. That one rejects a value only a user can have written; these
    two arrive as a schema default on rows whose author never set them, so
    refusing would make those deployments impossible to update without first
    editing a field they never touched. The RAM accounting does not depend on
    this either way — `get_computed_ram_claim` gates on `is_local()` — so this
    is about not storing a number that means nothing.
    """
    for declaration in declarations:
        ext = declaration.ext
        if not ext or not ext.is_shared():
            continue
        if ext.ram_size is None and ext.ram_ratio is None:
            continue
        logger.debug(
            "Dropping local-cache sizing (ram_size=%s, ram_ratio=%s) from a "
            "shared KV cache declaration%s; the cache service owns that memory.",
            ext.ram_size,
            ext.ram_ratio,
            declaration.where,
        )
        ext.ram_size = None
        ext.ram_ratio = None


async def _validate_one_cache_declaration(
    session: AsyncSession,
    declaration: _CacheDeclaration,
    owner_principal_id: int,
    effective_cluster_id: Optional[int],
) -> None:
    ext = declaration.ext
    where = declaration.where
    if not ext or not ext.enabled:
        return

    if ext.is_local():
        if ext.cache_service_id:
            raise BadRequestException(
                message=f"cache_service_id is only valid when mode is 'shared'{where}"
            )
        return

    if not ext.cache_service_id:
        raise BadRequestException(
            message=(
                f"cache_service_id is required when extended KV cache "
                f"mode is 'shared'{where}"
            )
        )

    cache_service = await CacheService.one_by_id(session, ext.cache_service_id)
    if (
        cache_service is None
        or cache_service.deleted_at is not None
        or cache_service.owner_principal_id != owner_principal_id
    ):
        raise NotFoundException(message=f"Cache service not found{where}")

    if (
        effective_cluster_id is not None
        and cache_service.cluster_id != effective_cluster_id
    ):
        raise BadRequestException(
            message=(
                f"The cache service must be in the same cluster as the "
                f"model{where}."
            )
        )

    provider = await get_cache_provider(session, cache_service.provider_name)
    backend = declaration.backend or BackendEnum.VLLM.value
    if provider is None or provider.integration_for(backend) is None:
        raise BadRequestException(
            message=(
                f"Cache service provider '{cache_service.provider_name}' is "
                f"not compatible with backend '{backend}'{where}."
            )
        )

    # Every built-in integration is framework-scoped, so a cluster whose
    # accelerators are all outside the provider's support matrix would
    # pass the framework-less check above and then degrade on every
    # instance. Pre-check against the cluster's actual accelerators;
    # accelerator-less clusters are left to scheduling.
    workers = await Worker.all_by_fields(
        session,
        fields={"cluster_id": cache_service.cluster_id},
        extra_conditions=[Worker.deleted_at.is_(None)],
    )
    frameworks = {
        device.type
        for worker in workers
        for device in (
            worker.status.gpu_devices
            if worker.status and worker.status.gpu_devices
            else []
        )
        if device.type
    }
    if frameworks and not any(
        provider.integration_for(backend, framework) for framework in frameworks
    ):
        raise BadRequestException(
            message=(
                f"Cache service provider '{cache_service.provider_name}' "
                f"has no '{backend}' integration for the cluster's "
                f"accelerators ({', '.join(sorted(frameworks))}){where}."
            )
        )

    # A pinned engine version below an integration's declared floor would
    # receive injected args the engine does not accept (e.g.
    # --shutdown-timeout) and fail to start. Reject when the version
    # falls outside every candidate integration's range; unparseable
    # versions fail open, and an unpinned version is resolved at deploy
    # time (the injection resolver re-checks it there).
    engine_version = declaration.backend_version
    if engine_version:
        candidates = (
            [provider.integration_for(backend, framework) for framework in frameworks]
            if frameworks
            else [provider.integration_for(backend)]
        )
        ranged = [c for c in candidates if c is not None and c.versions]
        if ranged and all(
            version_in_range(engine_version, c.versions) is False for c in ranged
        ):
            ranges = ", ".join(sorted({c.versions for c in ranged}))
            raise BadRequestException(
                message=(
                    f"Backend version {engine_version} is outside the "
                    f"cache provider's supported '{backend}' range "
                    f"({ranges}){where}."
                )
            )


class SpanningRole(BaseModel):
    name: str
    gpus: int


class SpanningPreview(BaseModel):
    """Which members of this draft cannot fit on one machine.

    Empty `roles` means «no, or not knowable», and the caller must treat those
    two the same: everything here is an addition to what a form could already
    say, never a precondition for saying it.
    """

    roles: List[SpanningRole]
    widest_worker_gpus: int


@router.post("/spanning-roles", response_model=SpanningPreview)
async def preview_spanning_roles(
    session: SessionDep, ctx: TenantContextDep, model_in: ModelCreate
):
    """Answer «will a member of this have to occupy more than one machine».

    The deploy form's «at least about X% of requests pair on one host» is a
    FLOOR, and cross-machine members turn it the wrong way round rather than
    merely loosening it: a member too wide for any machine takes whole machines
    (the engine selectors allocate every GPU of every worker they pick), so no
    machine holds both a prefill and a decode and the true figure is exactly
    zero. A note reading «at least 25%» beside a real 0 is worse than no note.

    Asked of the server rather than computed in the form because the width is
    the engine's own arithmetic -- vLLM spells it `--tensor-parallel-size`,
    SGLang `--tp-size`, and both fold in pipeline and data parallelism. A
    second reading of those flags in TypeScript is a second answer to one
    question, and the two would drift.

    Safe to ask mid-typing because it is arithmetic and not a solve: see
    `roles_that_must_span`.
    """
    # Visibility, not ownership: nothing is created here, and the answer is
    # about the cluster's machines rather than about anything the caller owns.
    if model_in.cluster_id is not None:
        assert_cluster_visible(
            ctx, await Cluster.one_by_id(session, model_in.cluster_id)
        )
    spanning, widest = await roles_that_must_span(session, model_in)
    return SpanningPreview(
        roles=[SpanningRole(name=name, gpus=width) for name, width in spanning],
        widest_worker_gpus=widest,
    )


async def _resolve_target_org(
    ctx: TenantContext, session: AsyncSession, cluster_id: Optional[int]
) -> Tuple[int, Optional[Cluster]]:
    """Resolve the Org a new model lands in, plus the cluster row if it
    had to be fetched on the way.

    Admin in "All" mode (no current principal) inherits the chosen
    cluster's Org, or falls back to the platform Org. The same value
    drives both the uniqueness pre-check and the row stamped on insert;
    resolving it up front keeps them in sync so the pre-check actually
    catches a collision in the Org the model will land in.
    """
    target_org_id = ctx.current_principal_id
    cluster = None
    if target_org_id is None and cluster_id is not None:
        # Admin "All" mode has no principal context; derive the owning Org
        # from the chosen cluster. Returned so the ownership check can
        # reuse it instead of doing a second lookup.
        cluster = await Cluster.one_by_id(session, cluster_id)
        if cluster is None:
            raise NotFoundException(message=f"Cluster {cluster_id} not found")
        target_org_id = cluster.owner_principal_id
    if target_org_id is None:
        target_org_id = platform_principal_id()
    return target_org_id, cluster


async def _assert_route_name_available(
    session: AsyncSession, model_in: ModelCreate, target_org_id: int
) -> None:
    """Reject a route name already taken in the target Org, when one is asked
    for. Separate from the model-name check so a path that already knows the
    model's fate can run just this half, and report it in the same words."""
    if not model_in.enable_model_route:
        return
    existing_route = await ModelRoute.one_by_fields(
        session,
        {"name": model_in.name, "owner_principal_id": target_org_id},
    )
    if existing_route:
        raise AlreadyExistsException(
            message=f"Model route with name '{model_in.name}' already exists."
        )


async def _assert_model_name_available(
    session: AsyncSession, model_in: ModelCreate, target_org_id: int
) -> None:
    """Reject a name already taken in the target Org.

    Only a new model has to clear this; overwriting an existing one collides
    with itself by definition.
    """
    # Model & ModelRoute names are unique within their Org. Two Orgs
    # can each have a "llama3" without colliding.
    existing = await Model.one_by_fields(
        session,
        {"name": model_in.name, "owner_principal_id": target_org_id},
    )
    if existing:
        raise AlreadyExistsException(
            message=f"Model with name '{model_in.name}' already exists."
        )
    await _assert_route_name_available(session, model_in, target_org_id)


async def _validate_model_spec(
    session: AsyncSession, model_in: ModelCreate, target_org_id: int
) -> None:
    """Validate the spec itself, independent of whether it is new or replaces
    an existing row.

    Mutates ``model_in`` in place: LoRA names are normalized to the stored
    ``<base>:<short>`` form and the scaling-schedule baseline is applied.
    """
    await validate_model_in(session, model_in)
    # Server-side assignment, after validation: validation must see the replica
    # count the caller submitted, not the schedule-driven one.
    apply_scaling_schedule_baseline(model_in)
    await validate_shared_kv_cache(
        session, model_in, target_org_id, model_in.cluster_id
    )


async def _check_model_create(
    session: AsyncSession,
    ctx: TenantContext,
    model_in: ModelCreate,
    target_org_id: int,
    cluster: Optional[Cluster],
) -> None:
    """Run every pre-insert check for a new model without writing anything.

    Mutates ``model_in`` in place, via :func:`_validate_model_spec`.
    """
    # The chosen cluster must exist, be visible to the caller, and be owned
    # by the target Org. In admin "All" mode target_org_id was derived from
    # the cluster, so the ownership check is trivially satisfied and this
    # mainly rejects a missing/deleted or non-visible cluster_id.
    await assert_cluster_belongs_to_org(
        ctx, session, model_in.cluster_id, target_org_id, cluster=cluster
    )
    await _assert_model_name_available(session, model_in, target_org_id)
    await _validate_model_spec(session, model_in, target_org_id)


async def _persist_model_create(
    session: AsyncSession, model_in: ModelCreate, target_org_id: int
) -> Model:
    """Insert the model and, when ``enable_model_route`` is set, its
    route, target, Org grant and LoRA child routes. Never commits, so a
    caller can batch several models into one transaction.
    """
    model_in_dict = model_in.model_dump(exclude={"enable_model_route"})

    # Stamp tenant scope. ModelBase has owner_principal_id defaulted to
    # PLATFORM_PRINCIPAL_ID, so `model_dump()` always emits the key —
    # `setdefault` would silently leave it at 1 even when the caller is
    # acting under a different Org. Override directly with the resolved
    # value.
    model_in_dict["owner_principal_id"] = target_org_id

    # Multi-tenant default: a non-platform Org's new model (and the
    # route(s) it spawns) is scoped to that Org via ALLOWED_PRINCIPALS
    # with the owning Org auto-granted below. The Default (platform) Org
    # keeps AUTHED. Caller's explicit ``access_policy`` always wins and
    # then manages its own grants via /principals. ``model_dump`` always
    # emits ``access_policy`` (it has a default), so override directly.
    org_scoped_default = (
        target_org_id is not None
        and target_org_id != platform_principal_id()
        and "access_policy" not in model_in.model_fields_set
    )
    if org_scoped_default:
        model_in_dict["access_policy"] = AccessPolicyEnum.ALLOWED_PRINCIPALS

    model: Model = await Model.create(session, source=model_in_dict, auto_commit=False)
    if not model_in.enable_model_route:
        return model
    await _create_model_route(session, model, grant_owning_org=org_scoped_default)
    return model


async def _create_model_route(
    session: AsyncSession, model: Model, *, grant_owning_org: bool
) -> None:
    """Create the deployment's primary route, its target and its LoRA child
    routes. Never commits.
    """
    model_route = ModelRoute(
        name=model.name,
        description=model.description,
        categories=model.categories,
        generic_proxy=model.generic_proxy,
        created_model_id=model.id,
        access_policy=model.access_policy,
        owner_principal_id=model.owner_principal_id,
    )
    model_route: ModelRoute = await ModelRoute.create(
        session, source=model_route, auto_commit=False
    )
    model_route_target = ModelRouteTarget(
        name=f"{model.name}-deployment",
        route_name=model_route.name,
        generic_proxy=model.generic_proxy,
        model_route=model_route,
        model=model,
        # Policy load-balancing out of the box: with more than one
        # replica the gateway picks among the deployment's instances
        # per request, so the auto-created target carries no split
        # weight (capability scoring decides the instance). Single
        # replica deployments keep the weighted shape — there is
        # nothing to schedule between instances. This default is a
        # creation-time convenience only: scaling an existing deployment
        # across the 1-replica boundary later leaves the stored shape
        # alone, and an explicit edit wins either way.
        weight=0 if (model.replicas or 0) > 1 else 100,
        state=TargetStateEnum.UNAVAILABLE,
    )
    await ModelRouteTarget.create(
        session,
        source=model_route_target,
        auto_commit=False,
    )
    if (model.replicas or 0) > 1:
        from gpustack.routes.plugins import get_route_plugin

        least_load = get_route_plugin("least-load")
        if least_load is not None:
            # Best-effort by design: through the plugin's own write path
            # so the default lands exactly as a client's plugins section
            # would, but a missing or partially-migrated plugin table must
            # not turn the whole deployment create into a 500 — the
            # route works without the capability, and the next write can
            # add it back. The savepoint isolates the plugin write: a
            # DB-level failure rolls back to here without poisoning the
            # caller's uncommitted transaction.
            try:
                async with session.begin_nested():
                    await least_load.on_route_write(
                        "create", model_route, {"enabled": True}, session
                    )
            except Exception:
                logger.warning(
                    "Failed to enable the least-load capability on the route "
                    "for model %s; the route is created without it",
                    model.name,
                    exc_info=True,
                )
    if grant_owning_org:
        # Auto-grant the owning Org on the primary route so its
        # members see it out of the box. The route is brand new,
        # so no existence check is needed; LoRA child routes get
        # their own grants inside create_lora_model_routes.
        session.add(
            ModelRoutePrincipalLink(
                route_id=model_route.id,
                principal_id=model.owner_principal_id,
            )
        )
    await create_lora_model_routes(
        session,
        model,
        access_policy=model.access_policy,
        generic_proxy=model.generic_proxy,
    )


@router.post(
    "",
    response_model=ModelPublic,
)
async def create_model(
    session: SessionDep, ctx: TenantContextDep, model_in: ModelCreate
):
    target_org_id, cluster = await _resolve_target_org(
        ctx, session, model_in.cluster_id
    )
    await _check_model_create(session, ctx, model_in, target_org_id, cluster)

    try:
        model = await _persist_model_create(session, model_in, target_org_id)
        await session.commit()
        if model_in.enable_model_route:
            await revoke_model_access_cache(session=session)
    except BadRequestException:
        await session.rollback()
        raise
    except Exception as e:
        await session.rollback()
        raise InternalServerErrorException(message=f"Failed to create model: {e}")

    return model


async def _models_to_export(
    session: AsyncSession, ctx: TenantContext, export_in: DeploymentExportRequest
) -> List[Model]:
    """The rows an export covers, by id so a re-export is byte-stable.

    With ``ids`` given, every id must resolve to a row the caller can see
    (and match ``cluster_id`` when set); otherwise 404 with no partial
    result, mirroring ``GET /models/{id}`` for cross-tenant ids.

    A document covers one Org: names are unique per Org, so two Orgs' rows
    could collide in one file, and an import lands every entry in one Org.
    """
    conditions = list(tenant_list_conditions(ctx, Model))
    if export_in.ids is not None:
        conditions.append(Model.id.in_(export_in.ids))
    fields = {}
    if export_in.cluster_id is not None:
        fields["cluster_id"] = export_in.cluster_id
    models = list(
        await Model.all_by_fields(session, fields=fields, extra_conditions=conditions)
    )
    if export_in.ids is not None:
        found = {model.id for model in models}
        missing = [str(id) for id in export_in.ids if id not in found]
        if missing:
            raise NotFoundException(message=f"Model not found: {', '.join(missing)}")
    orgs = {model.owner_principal_id for model in models}
    if len(orgs) > 1:
        raise BadRequestException(
            message="the selection spans more than one organization; export "
            "one organization at a time"
        )
    models.sort(key=lambda model: model.id)
    return models


async def _route_backed_model_ids(
    session: AsyncSession, models: List[Model]
) -> Set[int]:
    """Ids among ``models`` whose primary model route still exists.

    LoRA child routes carry ``created_model_id`` too, so only the live route
    named after the model counts: that is the one ``enable_model_route``
    re-creates on import.
    """
    names_by_id = {model.id: model.name for model in models}
    if not names_by_id:
        return set()
    result = await session.exec(
        select(ModelRoute.created_model_id, ModelRoute.name).where(
            ModelRoute.created_model_id.in_(list(names_by_id)),
            ModelRoute.deleted_at.is_(None),
        )
    )
    return {
        created_model_id
        for created_model_id, name in result.all()
        if names_by_id.get(created_model_id) == name
    }


async def _cluster_names_by_id(
    session: AsyncSession, models: List[Model]
) -> Dict[int, str]:
    """Names of the clusters ``models`` sit in, by id.

    The document names clusters rather than numbering them, and
    ``Model.cluster`` does not load with the row.
    """
    ids = {model.cluster_id for model in models if model.cluster_id is not None}
    if not ids:
        return {}
    result = await session.exec(
        select(Cluster.id, Cluster.name).where(Cluster.id.in_(list(ids)))
    )
    return {cluster_id: name for cluster_id, name in result.all()}


@router.post("/export")
async def export_models(
    session: SessionDep, ctx: TenantContextDep, export_in: DeploymentExportRequest
):
    """Download deployments as a YAML document (``schemas/deployment_document``).

    POST rather than GET: ``/export`` is a fixed segment inside the ``/{id}``
    namespace, and only a different method keeps it independent of route
    registration order.
    """
    models = await _models_to_export(session, ctx, export_in)
    route_backed_ids = await _route_backed_model_ids(session, models)
    cluster_names = await _cluster_names_by_id(session, models)
    exported_at = datetime.now(timezone.utc)
    content = dump_deployments(models, route_backed_ids, cluster_names, exported_at)
    if len(models) == 1:
        filename = sanitize_filename(f"{models[0].name}.yaml", "deployment.yaml")
    else:
        filename = f"gpustack-deployments-{exported_at:%Y%m%d-%H%M%S}.yaml"
    return Response(
        content=content,
        media_type="application/x-yaml",
        headers=attachment_headers(filename),
    )


class _ImportItem(NamedTuple):
    """One document entry: what would happen to it, and what it acts on."""

    plan: DeploymentPlanEntry
    entry: Optional[ModelCreate]
    """None when the entry did not parse."""
    existing: Optional[Model]
    """The row this entry would replace, if there is one."""


def _apply_replica_override(model_in: ModelCreate, replicas: int) -> None:
    """Use the caller's replica count for this entry.

    Held to the same rules as the deploy form, which lets the count be set
    beside a manual GPU selection and validates the pair afterwards. An
    enabled scaling schedule is the one case where ``replicas`` is not the
    user's to set -- ``apply_scaling_schedule_baseline`` recomputes it from
    the schedule -- so the count lands on ``baseline_replicas``, which is what
    the form's Replicas field edits in that mode too.
    """
    model_in.replicas = replicas
    if model_in.scaling_schedule and model_in.scaling_schedule.enabled:
        model_in.scaling_schedule.baseline_replicas = replicas


async def _own_model_routes(session: AsyncSession, model: Model) -> List[ModelRoute]:
    """The live routes this deployment created -- its primary route and its
    LoRA children. The same test the export reads ``enable_model_route`` off,
    so what a document says about routes and what an import settles agree."""
    return list(
        await ModelRoute.all_by_fields(
            session, {"created_model_id": model.id, "deleted_at": None}
        )
    )


async def _routes_serving_others(
    session: AsyncSession, routes: List[ModelRoute], model_id: int
) -> List[str]:
    """Names of ``routes`` that also target a deployment other than this one."""
    if not routes:
        return []
    targets = await ModelRouteTarget.all_by_fields(
        session,
        {"deleted_at": None},
        extra_conditions=[
            ModelRouteTarget.route_id.in_([route.id for route in routes])
        ],
    )
    shared = {target.route_id for target in targets if target.model_id != model_id}
    return sorted(route.name for route in routes if route.id in shared)


async def _overwrite_blockers(
    session: AsyncSession,
    existing: Model,
    model_in: ModelCreate,
    own_routes: List[ModelRoute],
    target_org_id: int,
) -> List[str]:
    """Why this deployment cannot be replaced, if it cannot.

    Overwriting is deliberately narrow, because a document is an easy thing to
    apply by accident: only a deployment that is both meant to be stopped and
    actually stopped.

    ``replicas`` alone is the operator's intent, not the cluster's state --
    instances are torn down asynchronously after a scale to zero, so a
    deployment can read as stopped while it is still serving. Both are
    checked, because the rest of the overwrite path is only safe once nothing
    is running -- dropping this deployment's model routes, and moving it to
    another cluster, both assume there is nothing left to cut off. A move is
    the entry's ``cluster`` to make: instances are only ever placed in the
    cluster the row names at the time they are created, so a stopped
    deployment has nothing to carry over.
    """
    blockers = []
    if existing.replicas > 0:
        blockers.append(
            f"already exists and is running (replicas={existing.replicas}); "
            "stop it before overwriting"
        )
    else:
        live = await ModelInstance.all_by_fields(
            session, {"model_id": existing.id, "deleted_at": None}
        )
        if live:
            blockers.append(
                f"already exists and still has {len(live)} instance(s) "
                "shutting down; wait for them to stop before overwriting"
            )

    primary = next((route for route in own_routes if route.name == existing.name), None)
    if model_in.enable_model_route:
        if primary is None:
            try:
                await _assert_route_name_available(session, model_in, target_org_id)
            except AlreadyExistsException as e:
                blockers.append(e.message)
        # Keeping the route still reaps the children the document drops.
        doomed = _dropped_lora_routes(own_routes, existing, model_in)
        detach_before = "removing the adapter(s)"
    else:
        # Disabling it drops every route this deployment owns.
        doomed, detach_before = own_routes, "disabling the route"

    # Deleting these is safe only because this deployment is stopped; another
    # one attached to the same route need not be.
    shared = await _routes_serving_others(session, doomed, existing.id)
    if shared:
        blockers.append(
            f"model route(s) {', '.join(shared)} also target other "
            f"deployments; detach them before {detach_before}"
        )
    return blockers


def _dropped_lora_routes(
    own_routes: List[ModelRoute], existing: Model, model_in: ModelCreate
) -> List[ModelRoute]:
    """The LoRA child routes :func:`cleanup_orphan_lora_routes` would reap once
    this entry is written: those whose adapter the new ``lora_list`` drops.

    ``lora_route_name_for`` is idempotent, so this reads the same before and
    after LoRA names are normalized in place.
    """
    desired = {
        lora_route_name_for(existing.name, entry.lora_name)
        for entry in (model_in.lora_list or [])
        if entry.lora_name
    }
    return [
        route for route in own_routes if ":" in route.name and route.name not in desired
    ]


class _ClusterMaps(NamedTuple):
    """The target Org's clusters, in the shapes the import reads them.

    A name the Org holds twice is left out of ``ids_by_name`` and kept in
    ``ambiguous``: it has to be reported, not picked between.
    """

    ids_by_name: Dict[str, int]
    names_by_id: Dict[int, str]
    ambiguous: Set[str]


async def _org_cluster_maps(session: AsyncSession, target_org_id: int) -> _ClusterMaps:
    """Fetched once for the whole document. A model's cluster must belong to
    its Org, so the Org's own are the whole of what an entry may name -- which
    is what keeps another Org's unreachable from here.
    """
    clusters = await Cluster.all_by_fields(
        session, {"owner_principal_id": target_org_id, "deleted_at": None}
    )
    ids_by_name: Dict[str, List[int]] = {}
    for each in clusters:
        ids_by_name.setdefault(each.name, []).append(each.id)
    return _ClusterMaps(
        ids_by_name={
            name: ids[0] for name, ids in ids_by_name.items() if len(ids) == 1
        },
        names_by_id={each.id: each.name for each in clusters},
        ambiguous={name for name, ids in ids_by_name.items() if len(ids) > 1},
    )


async def _document_org_id(session: AsyncSession, loaded: List[LoadedEntry]) -> int:
    """The Org an import lands in when the caller's context does not say.

    An admin in "All" mode has no principal context, and defaulting to the
    platform Org would land another Org's export in Default on a name match.
    Clusters are owned rows, so the document's ``cluster_name`` values answer
    it; anything but exactly one Org is refused.
    """
    named = sorted({entry.cluster_name for entry in loaded if entry.cluster_name})
    if not named:
        raise BadRequestException(
            message="no organization to import into: choose a target cluster, or "
            'switch to an organization, or name a "cluster_name" in the document'
        )
    clusters = await Cluster.all_by_fields(
        session, {"deleted_at": None}, extra_conditions=[Cluster.name.in_(named)]
    )
    if not clusters:
        raise BadRequestException(
            message=f"no cluster named {', '.join(named)} in this installation; "
            "rename them to clusters that exist here, or choose a target cluster "
            "for the import"
        )
    orgs = {cluster.owner_principal_id for cluster in clusters}
    if len(orgs) > 1:
        raise BadRequestException(
            message="the clusters this document names belong to more than one "
            "organization; import one organization at a time, or choose a target "
            "cluster for the import"
        )
    return orgs.pop()


def _entry_cluster_id(
    entry: LoadedEntry,
    forced_cluster_id: Optional[int],
    existing: Optional[Model],
    clusters: _ClusterMaps,
) -> Tuple[Optional[int], Optional[str]]:
    """Where an entry lands, and what is wrong if nowhere.

    In order: the cluster the request forces, the one the entry names, the one
    the deployment is already in -- the last keeps a document written before
    clusters were recorded working. Either of the first two may disagree with
    the row, and then the overwrite moves the deployment.

    ``clusters`` holds only the target Org's, so a name outside it is as good
    as absent.
    """
    if forced_cluster_id is not None:
        return forced_cluster_id, None
    if entry.cluster_name is not None:
        cluster_id = clusters.ids_by_name.get(entry.cluster_name)
        if cluster_id is not None:
            return cluster_id, None
        if entry.cluster_name in clusters.ambiguous:
            return None, (
                f"more than one cluster is named '{entry.cluster_name}'; rename "
                "them apart, or choose a target cluster for the import"
            )
        return None, (
            f"cluster '{entry.cluster_name}' not found; rename it to a cluster in "
            "this installation, or choose a target cluster for the import"
        )
    if existing is not None:
        return existing.cluster_id, None
    return None, (
        'no cluster; choose a target cluster for the import, or add a "cluster_name" '
        "field to this deployment"
    )


async def _plan_import(
    session: AsyncSession,
    loaded: List[LoadedEntry],
    import_in: DeploymentImportRequest,
    target_org_id: int,
    clusters: _ClusterMaps,
) -> List[_ImportItem]:
    """Work out what importing this document would do, writing nothing.

    Every problem lands on the entry that caused it rather than aborting the
    pass, so one round trip tells the user about all of them at once.

    Rows are looked up scoped to ``target_org_id``, which is the caller's own
    Org (or, for an admin with no principal context, the target cluster's or
    the document's). That is the same boundary ``assert_resource_visible``
    enforces, so an entry can never name its way onto another Org's
    deployment. The cluster maps cover that same Org, so an entry naming
    another Org's cluster simply resolves to nothing.
    """
    items: List[_ImportItem] = []
    for entry in loaded:
        plan = DeploymentPlanEntry(
            index=entry.index,
            name=entry.name,
            errors=list(entry.errors),
            raw=entry.raw,
        )
        # The row this entry acts on, whether or not the entry parsed: an
        # entry that failed to validate still names a deployment, and the one
        # it would have replaced is what the user reads beside it to see what
        # they broke.
        existing = (
            await Model.one_by_fields(
                session, {"name": entry.name, "owner_principal_id": target_org_id}
            )
            if entry.name
            else None
        )
        existing_routes = (
            await _own_model_routes(session, existing) if existing is not None else []
        )
        if existing is not None:
            plan.current = deployment_entry(
                existing,
                any(route.name == existing.name for route in existing_routes),
                clusters.names_by_id.get(existing.cluster_id),
            )

        model_in = entry.entry
        if model_in is None:
            items.append(_ImportItem(plan, None, None))
            continue

        model_in.cluster_id, cluster_error = _entry_cluster_id(
            entry, import_in.cluster_id, existing, clusters
        )
        if cluster_error is not None:
            plan.errors.append(cluster_error)

        # Before the snapshot, so an adjusted count shows up in the diff and
        # is what gets written -- the preview and the write read the same
        # request.
        if model_in.name in import_in.replica_overrides:
            _apply_replica_override(
                model_in, import_in.replica_overrides[model_in.name]
            )
        # Snapshot before the checks below normalize LoRA names and the
        # replica count in place: the diff is against what the user wrote.
        desired = entry_document_form(
            model_in, clusters.names_by_id.get(model_in.cluster_id)
        )
        plan.desired = desired
        if existing is None:
            plan.action = DeploymentActionEnum.CREATE
            # The model name is settled by the lookup above; only the route
            # name is still open.
            try:
                await _assert_route_name_available(session, model_in, target_org_id)
            except AlreadyExistsException as e:
                plan.errors.append(e.message)
        else:
            plan.changes = diff_entries(plan.current, desired)
            # The diff spells a cluster by name, and two of an Org's can share
            # one; only the id says whether the deployment actually moves.
            moves = model_in.cluster_id != existing.cluster_id
            plan.action = (
                DeploymentActionEnum.UPDATE
                if plan.changes or moves
                else DeploymentActionEnum.UNCHANGED
            )
            # Only an entry that would actually be written has to clear these.
            # An unchanged one writes nothing, so holding it to the overwrite
            # rules would reject re-importing an untouched export of a running
            # deployment -- the very thing a backup is for. A cluster that
            # disagrees is never unchanged: ``cluster`` is a document field,
            # so it lands in ``changes`` like any other.
            if plan.action is DeploymentActionEnum.UPDATE:
                plan.errors.extend(
                    await _overwrite_blockers(
                        session,
                        existing,
                        model_in,
                        existing_routes,
                        target_org_id,
                    )
                )

        try:
            await _validate_model_spec(session, model_in, target_org_id)
        except (BadRequestException, NotFoundException) as e:
            plan.errors.append(e.message)
        items.append(_ImportItem(plan, model_in, existing))
    return items


@router.post("/import", response_model=DeploymentImportResult)
async def import_models(
    session: SessionDep,
    ctx: TenantContextDep,
    import_in: DeploymentImportRequest,
):
    """Apply a deployment document, or none of it.

    A dry run always answers with the plan — which entries would be created,
    which overwritten and how, which are already as the document describes —
    so the client can show it and ask. Writing is a single transaction.

    A document lands in one Org: the caller's, the forced cluster's, or -- for
    an admin in "All" mode following the file -- the document's own clusters'.
    """
    try:
        loaded = await asyncio.to_thread(load_deployments, import_in.content)
    except ValueError as e:
        raise BadRequestException(message=str(e))

    # A misspelled name would otherwise just not take effect, and the caller
    # would be told a count they never asked for is what will be written.
    unmatched = sorted(
        set(import_in.replica_overrides) - {entry.name for entry in loaded}
    )
    if unmatched:
        raise BadRequestException(
            message="replica_overrides names no deployment in the document: "
            f"{', '.join(unmatched)}"
        )

    if ctx.current_principal_id is None and import_in.cluster_id is None:
        # Nothing in the request says which Org, so the document decides.
        target_org_id = await _document_org_id(session, loaded)
        cluster = None
    else:
        target_org_id, cluster = await _resolve_target_org(
            ctx, session, import_in.cluster_id
        )
        if cluster is None and import_in.cluster_id is not None:
            cluster = await Cluster.one_by_id(session, import_in.cluster_id)
    # Once, up front: a bad cluster is one 404/403, not one error per entry.
    # A request that forces no cluster has nothing to check here; the clusters
    # the entries name are resolved against the Org's own, below.
    await assert_cluster_belongs_to_org(
        ctx, session, import_in.cluster_id, target_org_id, cluster=cluster
    )

    clusters = await _org_cluster_maps(session, target_org_id)
    items = await _plan_import(session, loaded, import_in, target_org_id, clusters)
    plans = [item.plan for item in items]
    valid = not any(plan.errors for plan in plans)
    if import_in.dry_run:
        return DeploymentImportResult(dry_run=True, valid=valid, entries=plans)
    if not valid:
        raise BadRequestException(
            message="\n".join(
                f"{entry_label(plan.index, plan.name)}: {error}"
                for plan in plans
                for error in plan.errors
            )
        )

    # Every unconfirmed overwrite at once, like every other problem the import
    # reports -- not just the first one the write loop happens to reach.
    unconfirmed = [
        f"{entry_label(item.plan.index, item.plan.name)}: already exists; "
        "confirm the overwrite before importing"
        for item in items
        if item.plan.action is DeploymentActionEnum.UPDATE
        and item.plan.name not in import_in.overwrite
    ]
    if unconfirmed:
        raise BadRequestException(message="\n".join(unconfirmed))

    try:
        models = await _persist_deployments(session, items, target_org_id)
        await session.commit()
    except BadRequestException:
        await session.rollback()
        raise
    except Exception as e:
        await session.rollback()
        # Only the type: the exception text carries SQLAlchemy's bound
        # parameters, and for this route those include a deployment's `env`.
        logger.error(f"Failed to import deployments: {type(e).__name__}")
        raise InternalServerErrorException(message="Failed to import deployments")

    # After the commit, and outside its try: a cache that fails to clear has
    # not undone the write, and reporting 500 here would tell the caller
    # nothing happened when everything did.
    # An overwrite can create or drop routes either way, so it always
    # invalidates; a plain create only when it asked for a route.
    if any(
        item.plan.action is DeploymentActionEnum.UPDATE
        or (
            item.plan.action is DeploymentActionEnum.CREATE
            and item.entry.enable_model_route
        )
        for item in items
    ):
        await revoke_model_access_cache(session=session)

    return DeploymentImportResult(
        dry_run=False,
        valid=True,
        entries=plans,
        items=[ModelPublic.model_validate(model) for model in models],
    )


async def _persist_model_update(
    session: AsyncSession, existing: Model, model_in: ModelCreate
) -> Model:
    """Replace ``existing`` with the entry and settle its routes. Never
    commits.

    A whole replacement, not a merge: the document is the desired state, so a
    field left out of it goes back to its default -- deleting a
    ``gpu_selector`` from the file is how a deployment returns to automatic
    placement. Only the document's own fields are written; see
    ``OVERWRITABLE_FIELDS`` for why that list is a whitelist.

    ``cluster_id`` is written beside them rather than from that list, which
    holds the document's field names: the document says ``cluster``, by name,
    and the route has already resolved it.
    """
    patch = {field: getattr(model_in, field) for field in OVERWRITABLE_FIELDS}
    patch["cluster_id"] = model_in.cluster_id
    await ModelService(session).update(existing, patch, auto_commit=False)

    own = await _own_model_routes(session, existing)
    primary = next((route for route in own if route.name == existing.name), None)
    if not model_in.enable_model_route:
        # Safe because an overwrite only reaches a stopped deployment, so
        # nothing is being served through these.
        for route in own:
            await ModelRouteService(session).delete(route, auto_commit=False)
        return existing
    if primary is None:
        await _create_model_route(
            session,
            existing,
            grant_owning_org=existing.access_policy
            == AccessPolicyEnum.ALLOWED_PRINCIPALS,
        )
    else:
        await create_lora_model_routes(
            session,
            existing,
            access_policy=existing.access_policy,
            generic_proxy=existing.generic_proxy,
        )
    # Either way the new lora_list decides which child routes survive: a
    # rebuilt primary route can still have children left from the old one.
    await cleanup_orphan_lora_routes(session, existing)
    return existing


async def _persist_deployments(
    session: AsyncSession,
    items: List[_ImportItem],
    target_org_id: int,
) -> List[Model]:
    """Write every entry without committing, and answer with the row each one
    settled on, in document order.

    Callers must have rejected the plan already if any entry carries errors,
    so every item here has an ``action`` and a parsed ``entry``.

    A LoRA route name conflict only surfaces while the routes are created, so
    it is relabelled here with the entry it belongs to, like every other error
    the import reports.
    """
    models = []
    for item in items:
        label = entry_label(item.plan.index, item.plan.name)
        if item.plan.action is DeploymentActionEnum.UNCHANGED:
            # The document already describes this row. Nothing to write, and
            # so nothing to confirm either.
            models.append(item.existing)
            continue
        try:
            if item.plan.action is DeploymentActionEnum.UPDATE:
                # Re-checked inside the transaction: the plan was built before
                # it opened, and the gate it enforces -- nothing running -- is
                # the whole reason dropping routes here is safe.
                own_routes = await _own_model_routes(session, item.existing)
                blockers = await _overwrite_blockers(
                    session,
                    item.existing,
                    item.entry,
                    own_routes,
                    target_org_id,
                )
                if blockers:
                    raise BadRequestException(message=blockers[0])
                models.append(
                    await _persist_model_update(session, item.existing, item.entry)
                )
            else:
                models.append(
                    await _persist_model_create(session, item.entry, target_org_id)
                )
        except BadRequestException as e:
            raise BadRequestException(message=f"{label}: {e.message}")
    return models


@router.put(
    "/{id}",
    response_model=ModelPublic,
)
async def update_model(
    session: SessionDep, ctx: TenantContextDep, id: int, model_in: ModelUpdate
):
    model = await Model.one_by_id(session, id)
    assert_resource_visible(ctx, model, not_found_message="Model not found")

    # Block re-pointing a model at another Org's (e.g. the Default org's
    # shared) or a non-visible cluster: its cluster must stay owned by the
    # model's Org.
    await assert_cluster_belongs_to_org(
        ctx, session, model_in.cluster_id, model.owner_principal_id
    )

    # Validate against the merged state: a sparse update carries only the
    # fields being changed, so validation would otherwise check
    # gpu_selector/gpu_type_selector mutual exclusion against half the
    # picture (e.g. setting gpu_selector on a model that already has
    # gpu_type_selector). object.__setattr__ bypasses pydantic's
    # fields-set tracking, keeping the backfill out of the persisted patch.
    for field in ("gpu_type_selector", "gpu_selector", "cluster_id"):
        if field not in model_in.model_fields_set:
            object.__setattr__(model_in, field, getattr(model, field))

    await validate_model_in(session, model_in, stored=model)
    # Server-side assignment, after validation: validation must see the replica
    # count the caller submitted, not the schedule-driven one.
    apply_scaling_schedule_baseline(model_in)
    await validate_shared_kv_cache(
        session,
        model_in,
        model.owner_principal_id,
        model_in.cluster_id or model.cluster_id,
    )

    if model_in.backend != BackendEnum.CUSTOM.value and (
        model.run_command or model.image_name
    ):
        patch = model_in.model_dump(exclude_unset=True)
        patch["run_command"] = None
        patch["image_name"] = None
        model_in = patch

    try:
        await ModelService(session).update(model, model_in, auto_commit=False)
        updated = await Model.one_by_id(session, id)
        if not updated:
            raise RuntimeError("Model not found after update")
        base_route = await ModelRoute.one_by_field(session, "name", updated.name)
        if base_route:
            await create_lora_model_routes(
                session,
                updated,
                access_policy=updated.access_policy,
                generic_proxy=updated.generic_proxy,
            )
            await cleanup_orphan_lora_routes(session, updated)
        await session.commit()
        await revoke_model_access_cache(session=session)
    except BadRequestException:
        await session.rollback()
        raise
    except Exception as e:
        await session.rollback()
        raise InternalServerErrorException(message=f"Failed to update model: {e}")

    return updated


class ModelRestartResult(BaseModel):
    """What a restart request did, so the caller can tell "converged" from
    "nothing to do" without a second read."""

    spec_digest: str
    """The generation the group is being brought onto."""
    restarted: bool
    deleted_instances: List[str] = []
    message: Optional[str] = None


@router.post("/{id}/restart", response_model=ModelRestartResult)
async def restart_model(session: SessionDep, ctx: TenantContextDep, id: int):
    """Retire the running generation so the current spec takes effect.

    Atomic by construction, and that is the point rather than an optimisation.
    "Restart" has until now meant deleting an instance and letting replica
    convergence rebuild it, which for a group produces a window holding a
    new-generation prefill beside an old-generation decode — and the engines do
    not reject that pairing. A `max_model_len` mismatch handshakes, transfers,
    and only fails on a long prompt, after prefill has already been paid for.
    So the whole generation stops before any of it starts again.

    There is deliberately no role parameter. "Restart only the decodes" is the
    request that produces exactly the cross-generation window above, and the
    strongest way to reject it is to have no way to express it.

    Not idempotent, deliberately. A group already wholly on the current spec is
    still torn down and rebuilt: "restart" is the word this operation is
    offered under, and the state an operator reaches for it in — a process
    wedged behind a socket while the control plane still calls it RUNNING — is
    precisely the one no digest comparison can detect. `restarted` is
    therefore true whenever there was anything to tear down, and
    `deleted_instances` says what that was.

    A restart still in flight is a 409 — the members are mid-replacement and a
    second teardown would delete the replacements. That is recorded on the
    model (`restarting_since`) rather than inferred from the rows: see
    `_restart_in_flight`, and the field's own note for why the digest
    comparison this replaces could never fire.

    """
    model = await Model.one_by_id(session, id)
    assert_resource_visible(ctx, model, not_found_message="Model not found")

    target = await model_spec_digest(session, model)
    instances = await ModelInstance.all_by_fields(
        session, fields={"model_id": model.id, "deleted_at": None}
    )

    if not instances:
        return ModelRestartResult(
            spec_digest=target,
            restarted=False,
            message="No instances to restart; the model has none running.",
        )

    if _restart_in_flight(model):
        # Tearing down again here would delete the replacements the previous
        # restart just created, and cost the group a second full startup.
        raise ConflictException(
            message="A restart is already in progress for this model: its "
            "members are still being rebuilt. Retry once it is running again."
        )

    failed = [
        instance.name
        for instance in instances
        if instance.state == ModelInstanceStateEnum.ERROR
    ]

    # No short-circuit on a converged group. Returning `restarted: false`
    # when the members already carry the target digest reads as "converge to
    # the current spec, and a converged group has nothing to converge".
    #
    # That would make the button mean two different things depending on the
    # row. Only a group's members are stamped with a `spec_digest` — a
    # role-less deployment's instances carry None, `{None} != {target}` is
    # always true, and so a plain model would always rebuild while a PD group
    # on its current spec answered with a sentence and did nothing. Same menu
    # entry, same wording, opposite behaviour — and the half that does nothing
    # would be the half whose members are hardest to cycle by hand.
    #
    # Between making both idempotent and making both act, act wins: "restart"
    # is the word on the menu, and the state an operator reaches for it in —
    # a wedged process that is RUNNING as far as the control plane knows — is
    # exactly the one a digest comparison cannot see. `deleted_instances` in
    # the result still reports what was actually torn down, so a caller that
    # cares can tell.
    #
    # No thrash risk: this endpoint is only ever reached by an explicit
    # request. Automatic recovery of a crashed member is the worker's, and it
    # has its own crash-loop brake.

    # Marked before the teardown, not after: if the process dies between the
    # two, a guard left on is recoverable (it lapses) while a guard never set
    # leaves the replacements exposed to the next click.
    await ModelService(session).update(
        model, {"restarting_since": datetime.now(timezone.utc)}
    )

    try:
        deleted = await ModelInstanceService(session).batch_delete(list(instances))
    except Exception as e:
        # Nothing was torn down, so there are no replacements to protect and no
        # reason to make the operator wait out the lapse. Released explicitly
        # rather than left to expire: the failure they now have to retry is the
        # worst moment to answer the retry with a 409.
        await ModelService(session).update(model, {"restarting_since": None})
        raise InternalServerErrorException(message=f"Failed to restart model: {e}")

    # Rebuilding is left to replica convergence rather than done here: it is
    # the one place that knows a group forms its GPU roles atomically and holds
    # the router back until they run, and duplicating that here would be a
    # second implementation of the rule that matters most.
    return ModelRestartResult(
        spec_digest=target,
        restarted=True,
        deleted_instances=deleted,
        message=_restart_message(failed),
    )


def _restart_in_flight(model: Model) -> bool:
    """Whether a previous restart is still rebuilding this deployment.

    Read off `Model.restarting_since`, which the status pass clears on RUNNING.
    Two things it is deliberately not:

    - **Not a digest comparison.** The teardown is synchronous and the reconcile
      rebuilds from the same target digest, so the generations never coexist and
      `len(digests) > 1` — the test this replaces — was never true. What it was
      written to prevent happened anyway, measured: a second click 4.5s after
      the first deleted the three replacements the first had just created.
    - **Not "the group is not RUNNING".** That would refuse the restart of a
      group wedged in `starting`, which is the state operators reach for this
      endpoint in and the reason its no-op short-circuit was removed.

    The lapse makes the refusal bounded. A restart that never converges must not
    become a deployment that can never be restarted.
    """
    since = model.restarting_since
    if since is None:
        return False
    if since.tzinfo is None:
        since = since.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - since).total_seconds()
    return age < envs.RESTART_IN_FLIGHT_LAPSE_SECONDS


def _restart_message(failed: List[str]) -> str:
    """Say whether a member was in error, because that leads to a different
    next step: a spec change is expected to fix itself, while a failed member
    usually means the reason it failed is still there."""
    if failed:
        return (
            f"Instances retired, including {len(failed)} in error "
            f"({', '.join(sorted(failed))}); the group will re-form on the "
            "current configuration. A member that failed for a reason still "
            "present will fail again — check its log before retrying."
        )
    return "Instances retired; the group will re-form on the current configuration."


@router.delete(
    "/{id}",
)
async def delete_model(session: SessionDep, ctx: TenantContextDep, id: int):
    model = await Model.one_by_id(
        session,
        id,
        options=[
            selectinload(Model.instances),
            selectinload(Model.model_route_targets),
        ],
    )
    assert_resource_visible(ctx, model, not_found_message="Model not found")

    try:
        await ModelService(session).delete(model)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to delete model: {e}")
