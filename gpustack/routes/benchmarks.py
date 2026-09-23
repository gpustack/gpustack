from sqlmodel import col
import re
import yaml
from datetime import timedelta
from typing import Dict, List, Optional, Sequence
import aiohttp
from fastapi import APIRouter, Depends, Query, Request, status
from fastapi.responses import (
    PlainTextResponse,
    RedirectResponse,
    StreamingResponse,
)
from sqlmodel import func, or_
from gpustack import envs
from gpustack.api.exceptions import (
    AlreadyExistsException,
    InternalServerErrorException,
    NotFoundException,
    BadRequestException,
)
from gpustack.api.responses import StreamingResponseWithStatusCode
from gpustack.api.tenant import (
    bypass_tenant_filter,
    assert_resource_visible,
    tenant_list_conditions,
    cluster_scoped_system,
    scoped_cluster_row_visible,
)
from gpustack.mixins.active_record import fuzzy_like
from gpustack.schemas.cache_services import CacheService
from gpustack.schemas.models import (
    Model,
    ModelInstance,
    ModelInstanceStateEnum,
    RoleNameEnum,
    is_audio_model,
    is_embedding_model,
    is_image_model,
    is_reranker_model,
    role_effective_model,
    servable_instances,
)
from gpustack.schemas.clusters import Cluster
from gpustack.schemas.model_routes import (
    ModelRoute,
    ModelRouteTarget,
    TargetStateEnum,
    effective_route_name,
)
from gpustack.schemas.principals import Principal, platform_principal_id
from gpustack.schemas.workers import Worker
from gpustack.config.config import get_global_config
from gpustack.server.db import async_session
from gpustack.server.deps import SessionDep, TenantContextDep
from gpustack.schemas.benchmark import (
    BenchmarkTargetModeEnum,
    DATASET_RANDOM,
    DATASET_SEED_MAX,
    DATASET_SEED_MIN,
    DATASET_SHAREGPT,
    SLO_THRESHOLDS,
    Benchmark,
    BenchmarkCreate,
    BenchmarkFullPublic,
    BenchmarkListParams,
    BenchmarkLoadTypeEnum,
    BenchmarkMetrics,
    BenchmarkResult,
    BenchmarkResultCreate,
    BenchmarkResultPublic,
    BenchmarkSnapshot,
    BenchmarkStateEnum,
    BenchmarkStateUpdate,
    BenchmarkUpdate,
    BenchmarkPublic,
    BenchmarksPublic,
    generate_dataset_seed,
)

from gpustack.server.services import (
    WorkerService,
)
from gpustack.server.worker_request import stream_to_worker, request_to_worker
from gpustack.utils.gpu import summary_gpu_snapshots
from gpustack.utils.grafana import (
    build_model_dashboard_url,
    resolve_grafana_base_url,
)
from gpustack.utils.snapshot import (
    create_model_instance_snapshot,
    create_worker_snapshot,
)
from gpustack.worker.logs import LogOptionsDep
from sqlalchemy.orm import defer

MAX_EXPORT_RECORDS = 20
# Upper bound on the per-point grid a single benchmark may upload. The ramp's own
# budget is max_points and a manual stage list is user-sized, so this sits far
# above any real curve — it exists so a malformed upload is rejected at the
# boundary instead of turning into an unbounded write.
MAX_BENCHMARK_RESULT_POINTS = 500
BENCHMARK_EXPORT_FIELD_ORDER = [
    "name",
    "model_name",
    "model_instance_name",
    "profile",
    "dataset_name",
    "request_rate",
    "total_requests",
    "dataset_input_tokens",
    "dataset_output_tokens",
    "dataset_seed",
]

router = APIRouter()


def order_benchmark_export_fields(benchmark: dict) -> dict:
    ordered = {}
    for field in BENCHMARK_EXPORT_FIELD_ORDER:
        if field in benchmark:
            ordered[field] = benchmark[field]

    for field, value in benchmark.items():
        if field not in ordered:
            ordered[field] = value

    return ordered


@router.get("", response_model=BenchmarksPublic)
async def get_benchmarks(
    ctx: TenantContextDep,
    params: BenchmarkListParams = Depends(),
    search: str = None,
    state: Optional[BenchmarkStateEnum] = Query(
        default=None,
        description="Filter by benchmark state.",
    ),
    model_name: Optional[str] = Query(None, description="Filter by model name."),
    gpu_summary: Optional[str] = Query(None, description="Filter by GPU summary."),
    dataset_name: Optional[str] = Query(None, description="Filter by dataset name."),
    profile: Optional[str] = Query(None, description="Filter by profile."),
    load_type: Optional[BenchmarkLoadTypeEnum] = Query(
        None, description="Filter by load type (fixed_rate / concurrency)."
    ),
    target_mode: Optional[BenchmarkTargetModeEnum] = Query(
        None, description="Filter by target mode (instance / route)."
    ),
    worker_id: Optional[int] = Query(
        None,
        description=(
            "Filter by the worker that owns the run. Every worker's 3-second "
            "state poll sends this; without it each worker reconciles the "
            "whole cluster's runs, and a run it does not own looks failed to "
            "it because the workload and its result files live on the owning "
            "host, so it reports a healthy run as failed."
        ),
    ),
):
    return await _get_benchmarks(
        ctx=ctx,
        params=params,
        state=state,
        search=search,
        model_name=model_name,
        gpu_summary=gpu_summary,
        dataset_name=dataset_name,
        profile=profile,
        load_type=load_type,
        target_mode=target_mode,
        worker_id=worker_id,
    )


def _fuzzy_contains(value: Optional[str], target: Optional[str]) -> bool:
    """Return False only when the filter value is set but not contained in target."""
    if not value:
        return True
    if not target:
        return False
    return value.lower() in target.lower()


def split_search_terms(search: Optional[str]) -> List[str]:
    """``search`` as the list of needles to match, any one of which qualifies a row.

    Comparing runs means pulling up exactly the handful being compared, and one
    substring rarely spans them. A benchmark name admits neither a comma nor a
    space, so both read unambiguously as separators.

    Args:
        search: The raw `search` query parameter.

    Returns:
        The needles, in the order given, without duplicates or blanks.
    """
    if not search:
        return []
    return list(dict.fromkeys(term for term in re.split(r"[,\s]+", search) if term))


def name_search_filter(data: Benchmark, search_terms: Sequence[str]) -> bool:
    """Whether `data` matches any needle — the watch stream's half of the filter.

    Mirrors the SQL the paginated branch builds, so a searched page and the
    stream that updates it agree on which rows belong to it.
    """
    if not search_terms:
        return True
    name = (data.name or "").lower()
    return any(term.lower() in name for term in search_terms)


def gpu_summary_filter(data: Benchmark, gpu_summary: Optional[str]) -> bool:
    return _fuzzy_contains(gpu_summary, data.gpu_summary)


def _make_benchmark_visibility_filter(ctx):
    def _visible(b: Benchmark) -> bool:
        if cluster_scoped_system(ctx):
            return scoped_cluster_row_visible(ctx, b)
        if bypass_tenant_filter(ctx):
            return True
        org_id = getattr(b, "owner_principal_id", None)
        if (
            ctx.current_principal_id is not None
            and org_id is not None
            and org_id == ctx.current_principal_id
        ):
            return True
        return False

    return _visible


async def _get_benchmarks(  # noqa: C901
    ctx,
    params: BenchmarkListParams,
    search: str = None,
    state: Optional[BenchmarkStateEnum] = None,
    model_name: Optional[str] = None,
    gpu_summary: Optional[str] = None,
    dataset_name: Optional[str] = None,
    profile: Optional[str] = None,
    load_type: Optional[BenchmarkLoadTypeEnum] = None,
    target_mode: Optional[BenchmarkTargetModeEnum] = None,
    worker_id: Optional[int] = None,
):
    # Name search is built here rather than handed to `fuzzy_fields`, which
    # takes one needle per column: several names have to OR together.
    search_terms = split_search_terms(search)

    fields = {}
    if state:
        fields["state"] = state

    if dataset_name:
        fields["dataset_name"] = dataset_name

    # Exact match, and deliberately in `fields` rather than `extra_conditions`:
    # `fields` is the only one of the two that the `watch` streaming branch
    # below also applies, and the worker watches as well as polls. Putting it
    # in `extra_conditions` would fix the poll and silently leave the stream
    # cluster-wide.
    if worker_id is not None:
        fields["worker_id"] = worker_id

    # `load_type` (fixed_rate / concurrency) filter (exact match; every row
    # carries a load_type).
    def _load_type_match(data) -> bool:
        return not load_type or data.load_type == load_type

    # `target_mode` (instance / route) filter. The column is backfilled by the
    # migration and written on every insert, so a NULL here means a row someone
    # put in by hand — and what such a row measured was an instance. Matching
    # NULL as `instance` keeps the two modes a partition of the list rather
    # than losing rows between them.
    def _target_mode_match(data) -> bool:
        if not target_mode:
            return True
        if target_mode == BenchmarkTargetModeEnum.INSTANCE:
            return data.target_mode in (None, BenchmarkTargetModeEnum.INSTANCE)
        return data.target_mode == target_mode

    extra_conditions = list(tenant_list_conditions(ctx, Benchmark))
    if search_terms:
        extra_conditions.append(
            or_(*[fuzzy_like(Benchmark.name, term) for term in search_terms])
        )
    if gpu_summary:
        extra_conditions.append(
            func.lower(Benchmark.gpu_summary).like(f"%{gpu_summary.lower()}%")
        )
    if profile:
        extra_conditions.append(
            func.lower(Benchmark.profile).like(f"%{profile.lower()}%")
        )
    if model_name:
        extra_conditions.append(
            func.lower(Benchmark.model_name).like(f"%{model_name.lower()}%")
        )
    if load_type:
        extra_conditions.append(Benchmark.load_type == load_type)
    if target_mode == BenchmarkTargetModeEnum.INSTANCE:
        extra_conditions.append(
            or_(
                Benchmark.target_mode == target_mode,
                col(Benchmark.target_mode).is_(None),
            )
        )
    elif target_mode:
        extra_conditions.append(Benchmark.target_mode == target_mode)

    _benchmark_visible = _make_benchmark_visibility_filter(ctx)

    if params.watch:
        return StreamingResponse(
            Benchmark.streaming(
                fields=fields,
                filter_func=lambda data: _benchmark_visible(data)
                and name_search_filter(data, search_terms)
                and gpu_summary_filter(data, gpu_summary)
                and _fuzzy_contains(profile, data.profile)
                and _fuzzy_contains(model_name, data.model_name)
                and _load_type_match(data)
                and _target_mode_match(data),
            ),
            media_type="text/event-stream",
        )

    order_by = params.order_by
    if order_by:
        new_order_by = []
        for field, direction in order_by:
            new_order_by.append((field, direction))
            if field in [
                "dataset_name",
                "cluster_id",
                "model_id",
                "model_name",
                "state",
            ]:
                # add additional sorting fields for deterministic ordering
                new_order_by.append(("created_at", direction))
        order_by = new_order_by

    async with async_session() as session:
        return await Benchmark.paginated_by_query(
            session=session,
            fields=fields,
            page=params.page,
            per_page=params.perPage,
            order_by=order_by,
            extra_conditions=extra_conditions,
            options=[defer(Benchmark.raw_metrics)],
        )


@router.get("/{id}", response_model=BenchmarkFullPublic)
async def get_benchmark(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
):
    benchmark = await Benchmark.one_by_id(session, id)
    assert_resource_visible(
        ctx, benchmark, not_found_message=f"Benchmark {id} not found"
    )
    return benchmark


# How far either side of a run the dashboard window is opened. A run's row is
# created before its container is (the image pull alone took ~100s on a first
# run), and the last write lands as the run reports its final state, so the
# recorded interval is a little wider than the traffic on both ends anyway.
# Padding costs a flat region on each side and buys the ramp-up not being
# clipped off the left edge.
_DASHBOARD_PADDING = timedelta(minutes=2)


@router.get("/{id}/dashboard")
async def get_benchmark_dashboard(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    request: Request,
):
    """Redirect to the dashboard for what this run measured, at the time it ran.

    The client-side numbers say a run was slow; they cannot say where the time
    went. Under PD that is three separate places -- the prefill queue, the
    decode queue, and the KV transfer between them -- and all three are on the
    PD dashboard, which is why a group's report links there rather than to the
    model dashboard whose request counters double for it.

    The window is the run's own, not `now`: a report read a day later is about
    an interval that has passed, and a link that opens on the last six hours
    shows an idle deployment.
    """
    benchmark = await Benchmark.one_by_id(session, id)
    assert_resource_visible(
        ctx, benchmark, not_found_message=f"Benchmark {id} not found"
    )

    model = None
    if benchmark.model_id is not None:
        model = await Model.one_by_id(session, benchmark.model_id)
    if model is None:
        # The deployment is gone; its metrics are not, but nothing here can say
        # which dashboard they belong on or which role counts.
        raise BadRequestException(
            message=(
                f"Model {benchmark.model_id} no longer exists, so the run's "
                f"dashboard cannot be resolved"
            )
        )

    cluster = None
    if benchmark.cluster_id is not None:
        cluster = await Cluster.one_by_id(session, benchmark.cluster_id)

    cfg = get_global_config()
    dashboard_url = build_model_dashboard_url(
        cfg,
        resolve_grafana_base_url(cfg, request),
        model,
        cluster_name=cluster.name if cluster is not None else None,
        extra_params=_dashboard_window(benchmark),
    )
    if dashboard_url is None:
        raise InternalServerErrorException(
            message="Grafana dashboard settings are not configured"
        )

    return RedirectResponse(url=dashboard_url, status_code=302)


def _dashboard_window(benchmark: Benchmark) -> Dict[str, str]:
    """Grafana's `from`/`to` for the interval this run occupied.

    A run that is still going gets an open right edge (`now`) rather than its
    last progress write: the reader is watching it happen, and a window that
    ends a few seconds ago would look frozen.
    """
    started = benchmark.created_at
    if started is None:
        return {}

    window: Dict[str, str] = {
        "from": str(int((started - _DASHBOARD_PADDING).timestamp() * 1000))
    }
    finished = benchmark.state in (
        BenchmarkStateEnum.COMPLETED,
        BenchmarkStateEnum.STOPPED,
        BenchmarkStateEnum.ERROR,
        # A run whose worker went away stopped producing traffic then, whatever
        # the row says next.
        BenchmarkStateEnum.UNREACHABLE,
    )
    if finished and benchmark.updated_at is not None:
        window["to"] = str(
            int((benchmark.updated_at + _DASHBOARD_PADDING).timestamp() * 1000)
        )
    else:
        window["to"] = "now"
    return window


def apply_progress_invariant(
    state_update: BenchmarkStateUpdate, benchmark: Benchmark
) -> None:
    """Keep `progress` monotonic within a run, in place on `state_update`.

    A multi-stage run reports each stage's slice of the overall bar, so a later
    stage's early percentage can be lower than the previous stage's end; the server
    holds the floor rather than letting the bar jump backwards.

    Two exceptions to the floor, both about a run STARTING:

    * entering RUNNING with no progress stated -> reset to 0. This is a fresh run
      (or a re-run), so the previous run's progress is not a floor for it.
    * entering RUNNING WITH a progress stated -> take it as given. The caller is
      reporting where the run actually is; overwriting that with 0 would discard a
      real measurement, and clamping it to the old run's value would be worse.
    """
    entering_running = (
        state_update.state == BenchmarkStateEnum.RUNNING
        and benchmark.state != BenchmarkStateEnum.RUNNING
    )
    if entering_running:
        if state_update.progress is None:
            state_update.progress = 0.0
            state_update.__pydantic_fields_set__.add("progress")
        return
    if (
        state_update.progress is not None
        and benchmark.progress is not None
        and state_update.progress < benchmark.progress
    ):
        state_update.progress = benchmark.progress


def _validate_load_config(benchmark_in: BenchmarkCreate) -> None:
    """Reject load configurations the runner cannot act on.

    `load_type` itself is checked by pydantic (it is an enum), but the numeric
    knobs and the auto_tune/stages combination are not. Every value here reaches a
    container as a CLI argument, so an unchecked one surfaces as an opaque ERROR
    minutes later — or, for the auto_tune + stages pair, as a run that silently
    ignores half of what the user configured.
    """
    if benchmark_in.auto_tune and benchmark_in.stages:
        raise BadRequestException(
            message=(
                "Fields auto_tune and stages are mutually exclusive: auto_tune "
                "searches the load axis itself, while stages runs the loads you "
                "specify. Set one of them."
            )
        )

    _validate_stages(benchmark_in)
    _validate_search_range(benchmark_in)
    _validate_positive_knobs(benchmark_in)
    _validate_point_budget(benchmark_in)
    _validate_non_negative_knobs(benchmark_in)
    _validate_dataset_seed(benchmark_in)
    _validate_token_windows(benchmark_in)


def _validate_stages(benchmark_in: BenchmarkCreate) -> None:
    if benchmark_in.stages is None:
        return
    if not benchmark_in.stages:
        raise BadRequestException(message="Field stages must not be empty")
    # Each item being an object is pydantic's contract (List[Dict[str, Any]]), so
    # only the contents need checking here.
    for i, stage in enumerate(benchmark_in.stages):
        rate = stage.get("rate")
        if not isinstance(rate, (int, float)) or isinstance(rate, bool):
            raise BadRequestException(
                message=f"Stage {i} must specify a numeric 'rate'"
            )
        if rate <= 0:
            raise BadRequestException(message=f"Stage {i} 'rate' must be > 0")
        # The per-stage constraints ride into `--stages` as JSON verbatim, so an
        # unusable one is not caught until the container acts on it — the same
        # reason the top-level budgets are checked in _validate_positive_knobs.
        for key in ("max_requests", "max_seconds"):
            value = stage.get(key)
            if value is None:
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise BadRequestException(message=f"Stage {i} '{key}' must be numeric")
            if value <= 0:
                raise BadRequestException(message=f"Stage {i} '{key}' must be > 0")


def _validate_search_range(benchmark_in: BenchmarkCreate) -> None:
    lower, upper = benchmark_in.lower_bound, benchmark_in.upper_bound
    if lower is not None and lower <= 0:
        raise BadRequestException(message="Field lower_bound must be > 0")
    if upper is not None and upper <= 0:
        raise BadRequestException(message="Field upper_bound must be > 0")
    if lower is not None and upper is not None and lower >= upper:
        raise BadRequestException(
            message=(
                f"Field lower_bound ({lower}) must be less than upper_bound "
                f"({upper}); the search range would otherwise be empty."
            )
        )


def _validate_positive_knobs(benchmark_in: BenchmarkCreate) -> None:
    """Budgets and SLO thresholds are all "<= N"-style quantities: 0 or less is not
    a stricter setting, it is a value the runner cannot act on."""
    fields = ["max_points", "max_total_seconds", "max_seconds", "turns"]
    fields += [t.attr for t in SLO_THRESHOLDS]
    for field in fields:
        value = getattr(benchmark_in, field, None)
        if value is not None and value <= 0:
            raise BadRequestException(message=f"Field {field} must be > 0")


def _validate_point_budget(benchmark_in: BenchmarkCreate) -> None:
    """The point budget cannot exceed what the results upload will accept.

    `POST /{id}/results` refuses a grid larger than MAX_BENCHMARK_RESULT_POINTS, and
    that check runs at the END of the run: a benchmark configured with 600 points
    would ramp for as long as its time budget allowed and then have its entire curve
    rejected by the terminal sync, leaving a `state_message` and nothing else.
    Same ceiling, applied while it is still a fixable typo.

    A manual `stages` list is checked too, for the same reason — it is the other way
    to ask for more rows than the grid can hold.
    """
    if (
        benchmark_in.max_points is not None
        and benchmark_in.max_points > MAX_BENCHMARK_RESULT_POINTS
    ):
        raise BadRequestException(
            message=(
                f"Field max_points ({benchmark_in.max_points}) must not exceed "
                f"{MAX_BENCHMARK_RESULT_POINTS}, the most result points a benchmark "
                "can hold."
            )
        )
    if (
        benchmark_in.stages is not None
        and len(benchmark_in.stages) > MAX_BENCHMARK_RESULT_POINTS
    ):
        raise BadRequestException(
            message=(
                f"Field stages ({len(benchmark_in.stages)} stages) must not exceed "
                f"{MAX_BENCHMARK_RESULT_POINTS}, the most result points a benchmark "
                "can hold."
            )
        )


def _validate_non_negative_knobs(benchmark_in: BenchmarkCreate) -> None:
    """Knobs where 0 IS a setting but a negative value is not.

    Kept apart from :func:`_validate_positive_knobs` because zero is meaningful
    here and must stay accepted: no warmup, no cooldown, tolerate no errors at
    all. All three are forwarded to guidellm on any load shape (see
    ``BenchmarkRunner._build_command_args``), so a negative one only surfaces as
    a container that dies mid-run.
    """
    for field in ("warmup", "cooldown", "max_errors"):
        value = getattr(benchmark_in, field, None)
        if value is not None and value < 0:
            raise BadRequestException(message=f"Field {field} must be >= 0")


def _validate_dataset_seed(benchmark_in: BenchmarkCreate) -> None:
    """A client-pinned seed has to fit the range the generator reserves.

    The bound is not cosmetic: a multi-stage run derives each stage's seed as
    base + stage_index, so a base near numpy's 2**32 ceiling overflows on the last
    stages, and a negative one is rejected outright. Both surface as a container
    that dies mid-run, which is why they are refused here instead.

    Only meaningful for the synthetic Random dataset — that is the only one whose
    prompts are generated from the seed.
    """
    seed = benchmark_in.dataset_seed
    if seed is None or benchmark_in.dataset_name != DATASET_RANDOM:
        return
    if not DATASET_SEED_MIN <= seed <= DATASET_SEED_MAX:
        raise BadRequestException(
            message=(
                f"Field dataset_seed ({seed}) must be between {DATASET_SEED_MIN} "
                f"and {DATASET_SEED_MAX}; a multi-stage run offsets it per stage, "
                "so the range leaves room for that."
            )
        )


def _validate_token_windows(benchmark_in: BenchmarkCreate) -> None:
    """An inverted token-length window yields no valid length."""
    for lo_field, hi_field in (
        ("dataset_input_min", "dataset_input_max"),
        ("dataset_output_min", "dataset_output_max"),
    ):
        lo, hi = getattr(benchmark_in, lo_field), getattr(benchmark_in, hi_field)
        if lo is not None and hi is not None and lo > hi:
            raise BadRequestException(
                message=f"Field {lo_field} ({lo}) must not exceed {hi_field} ({hi})"
            )


async def _resolve_target_model(
    session: SessionDep, benchmark_in: BenchmarkCreate
) -> Model:
    """The deployment a run measures.

    A benchmark targets a MODEL, not one of its members. Under PD "pick an
    instance" is not a choice the user can make correctly — a group answers
    only through its router — and even without PD it is the wrong question:
    picking one replica of three measures a replica while the report says it
    measured the model.

    `model_instance_name` is still accepted, because a client that names a
    member is naming its model too, and the instance list page's "run
    benchmark" action does exactly that.
    """
    if benchmark_in.model_id is not None:
        model = await Model.one_by_id(session, benchmark_in.model_id)
        if not model:
            raise BadRequestException(
                message=f"Model {benchmark_in.model_id} not found"
            )
        return model

    if benchmark_in.model_name:
        fields = {"name": benchmark_in.model_name}
        if benchmark_in.cluster_id is not None:
            # A model name is unique within a cluster, not across the fleet.
            fields["cluster_id"] = benchmark_in.cluster_id
        model = await Model.one_by_fields(session, fields)
        if not model:
            raise BadRequestException(
                message=f"Model '{benchmark_in.model_name}' not found"
            )
        return model

    name = (benchmark_in.model_instance_name or "").strip()
    if not name:
        raise BadRequestException(
            message="Field model_id, model_name or model_instance_name must be specified"
        )
    instance = await ModelInstance.one_by_field(session, "name", name)
    if not instance:
        raise BadRequestException(message=f"Model instance '{name}' not found")
    model = await Model.one_by_id(session, instance.model_id)
    if not model:
        raise BadRequestException(message=f"Model {instance.model_id} not found")
    return model


async def _resolve_target_endpoint(
    session: SessionDep, model: Model, requested_name: Optional[str]
) -> ModelInstance:
    """The member the load is actually sent to.

    `servable_instances()` decides, which makes this its third caller: the
    gateway's upstream registration, the direct proxy, and now the load
    generator. The rule it carries is that no member of a group can answer on
    its own — a prefill returns after a single token, a decode runs without
    the prefix its KV was meant to carry, and both answer 200 with plausible
    text. A benchmark pointed at one of them does not fail; it reports
    latencies for a request path no user request ever takes.
    """
    members = await ModelInstance.all_by_field(session, "model_id", model.id)
    running = [m for m in members if m.state == ModelInstanceStateEnum.RUNNING]
    servable = servable_instances(model, running)

    requested = (requested_name or "").strip()
    if requested:
        named = next((m for m in members if m.name == requested), None)
        if named is None:
            raise BadRequestException(message=f"Model instance '{requested}' not found")
        if named.state != ModelInstanceStateEnum.RUNNING:
            raise BadRequestException(
                message=f"Model instance '{requested}' not in RUNNING state"
            )
        if any(m.id == named.id for m in servable):
            return named
        if getattr(model, "roles", None):
            raise BadRequestException(
                message=(
                    f"Model instance '{requested}' has role "
                    f"'{named.role}', which cannot serve a whole request: a "
                    f"disaggregated group answers only through its router. "
                    f"Target the model instead of one of its members."
                )
            )

    if not servable:
        if getattr(model, "roles", None):
            raise BadRequestException(
                message=(
                    f"Model '{model.name}' has no running router, so the group "
                    f"cannot serve a request and there is nothing to measure."
                )
            )
        raise BadRequestException(
            message=f"Model '{model.name}' has no running instance to benchmark"
        )
    # Lowest id: the same member across repeated runs of one deployment, so
    # two runs of the same configuration are comparable.
    return sorted(servable, key=lambda m: m.id)[0]


async def _effective_name(session: SessionDep, route: ModelRoute) -> str:
    owner = await Principal.one_by_id(session, route.owner_principal_id)
    return effective_route_name(
        route.name,
        getattr(owner, "name", None),
        getattr(owner, "id", None) == platform_principal_id(),
    )


async def _resolve_named_route(
    session: SessionDep, model: Model, requested: str
) -> str:
    """The route the caller named, once it is confirmed to front this model.

    A model can sit behind several routes — an alias, a canary — and they are
    not the same measurement: a canary sends a share of the load to a
    different model entirely. So the form names the one it listed rather than
    letting the server pick, and this checks that the pair actually holds
    instead of trusting it: a route that does not front this model would
    measure something else under this model's name, and the report would say
    nothing about the mismatch.
    """
    targets = await ModelRouteTarget.all_by_fields(
        session,
        {"model_id": model.id, "state": TargetStateEnum.ACTIVE, "deleted_at": None},
    )
    for target in targets:
        route = await ModelRoute.one_by_id(session, target.route_id)
        if route is None:
            continue
        name = await _effective_name(session, route)
        if requested in (name, route.name):
            return name

    raise BadRequestException(
        message=(
            f"Route '{requested}' does not have an active target for model "
            f"'{model.name}', so a run through it would not measure this "
            f"deployment."
        )
    )


async def _resolve_route_name(session: SessionDep, model: Model) -> str:
    """The name a client calls this deployment by.

    `route` mode drives the deployment through the same door as production
    traffic, and that door is addressed by the effective route name — prefixed
    with the owning Org for everyone but the platform Org, exactly as the
    gateway and `/v1` resolve it. Deriving it here rather than asking the
    caller for it keeps the two from disagreeing: a benchmark that named the
    wrong route would measure a different deployment and say nothing.

    An ACTIVE target is required. A route whose target is UNAVAILABLE resolves
    to nothing at request time, so a run against it would measure a 503 —
    refusing with the reason is the honest answer.
    """
    targets = await ModelRouteTarget.all_by_fields(
        session,
        {"model_id": model.id, "state": TargetStateEnum.ACTIVE, "deleted_at": None},
    )
    if not targets:
        raise BadRequestException(
            message=(
                f"Model '{model.name}' is not reachable through any active route, "
                f"so it cannot be benchmarked in route mode. Deploy a route for "
                f"it, or benchmark an instance instead."
            )
        )

    # Lowest id: a model reachable through several routes (an alias, a canary)
    # is measured through the oldest, which is the one that has been in front
    # of it the longest. The name is recorded on the run either way, since a
    # route's targets and weights are editable and a canary can send part of
    # the load elsewhere.
    target = sorted(targets, key=lambda t: t.id)[0]
    route = await ModelRoute.one_by_id(session, target.route_id)
    if route is None:
        raise BadRequestException(
            message=f"Route '{target.route_name}' no longer exists"
        )
    return await _effective_name(session, route)


async def _resolve_placement(
    session: SessionDep, model: Model, endpoint: ModelInstance
) -> ModelInstance:
    """The member whose worker the run's container is deployed on.

    Not the endpoint, under PD. `--processor` is a path on the HOST — the
    tokenizer is read from the model directory the worker downloaded — and the
    router holds no weights, so the machine serving the endpoint is not
    necessarily a machine that has the files. (Today it happens to have them,
    because `ensure_instance_model_file` creates a ModelFile for every
    scheduled instance without looking at its role, so a router's worker
    downloads a full set of weights it never reads. That is worth fixing, and
    this must land first or fixing it takes the benchmark down with it.)

    Decode is preferred among the members that do have the files: the load
    generator is CPU-bound and steals from whatever it sits next to, and
    prefill is the role that spends its CPU on tokenization. Sitting on the
    router's worker would be worse than either — the router is the group's
    only entrance, so throttling it throttles the very thing being measured.
    """
    if not getattr(model, "roles", None):
        return endpoint

    members = await ModelInstance.all_by_field(session, "model_id", model.id)
    with_weights = [
        m
        for m in members
        if m.state == ModelInstanceStateEnum.RUNNING
        and m.role != RoleNameEnum.ROUTER.value
        and m.resolved_path
        and m.worker_id is not None
    ]
    if not with_weights:
        # Nothing better to say than "run it where the endpoint is". The
        # tokenizer may not resolve there, and the run reports that itself.
        return endpoint

    decode = [m for m in with_weights if m.role == RoleNameEnum.DECODE.value]
    return sorted(decode or with_weights, key=lambda m: m.id)[0]


async def validate_and_mutate_benchmark_in(  # noqa: C901
    session: SessionDep, benchmark_in: BenchmarkCreate
) -> Benchmark:

    mutated = Benchmark(**benchmark_in.model_dump())
    model = await _resolve_target_model(session, benchmark_in)
    instance = await _resolve_target_endpoint(
        session, model, benchmark_in.model_instance_name
    )
    placement = await _resolve_placement(session, model, instance)
    # Route mode still resolves an instance above: the run has to be recorded
    # against something, the placement still has to land where the weights are,
    # and the snapshot still describes the deployment. What changes is only
    # where the load is SENT, which the runner reads off `target_mode`.
    route_name = None
    if mutated.target_mode == BenchmarkTargetModeEnum.ROUTE:
        # Named by the caller when it had a list to pick from (the form does),
        # derived when it did not (an API client naming only a model).
        route_name = (
            await _resolve_named_route(session, model, benchmark_in.route_name)
            if benchmark_in.route_name
            else await _resolve_route_name(session, model)
        )

    mutated.model_id = model.id
    mutated.model_name = model.name
    # Server-derived: the endpoint is resolved from the model, so a client's
    # value is a request rather than the answer.
    mutated.model_instance_name = instance.name

    if benchmark_in.dataset_name is None:
        raise BadRequestException(message="Field dataset_name must be specified")

    if benchmark_in.dataset_name not in [
        DATASET_RANDOM,
        DATASET_SHAREGPT,
    ]:
        raise BadRequestException(
            message=f"Dataset '{benchmark_in.dataset_name}' is not supported. Supported datasets are '{DATASET_RANDOM}' and '{DATASET_SHAREGPT}'."
        )

    if benchmark_in.dataset_name == DATASET_RANDOM and (
        benchmark_in.dataset_input_tokens is None
        or benchmark_in.dataset_output_tokens is None
    ):
        raise BadRequestException(
            message="Fields dataset_input_tokens and dataset_output_tokens must be specified for 'Random' dataset"
        )

    # dataset_seed is the seed the run actually uses, so make it concrete here
    # (only the Random dataset generates from it): a client that states just the
    # intent (dataset_seed_random, no value) gets one generated, while a pinned
    # seed is always honored. Without this the run falls back to the benchmark
    # runner's fixed default and replays the previous run's prompts — and the
    # prefix cache they left behind.
    if mutated.dataset_name == DATASET_RANDOM:
        if mutated.dataset_seed is None:
            mutated.dataset_seed = generate_dataset_seed()
            mutated.dataset_seed_random = True
        elif "dataset_seed_random" not in benchmark_in.model_fields_set:
            # An explicit seed with no stated provenance is a pinned one.
            mutated.dataset_seed_random = False

    if (
        is_image_model(model)
        or is_audio_model(model)
        or is_embedding_model(model)
        or is_reranker_model(model)
    ):
        raise BadRequestException(
            message=f"Benchmarking is not supported for model type '{model.type.value}'"
        )

    if benchmark_in.request_rate <= 0:
        mutated.request_rate = (
            benchmark_in.total_requests
            if benchmark_in.total_requests is not None
            else 1000
        )  # treat non-positive request_rate as unlimited

    _validate_load_config(benchmark_in)

    snapshot = await get_benchmark_snapshot(
        session, instance, model, route_name=route_name
    )
    mutated.snapshot = snapshot
    mutated.gpu_summary, mutated.gpu_vendor_summary = summary_gpu_snapshots(
        snapshot.gpus
    )
    # The worker the container runs on is the PLACEMENT member's, which is the
    # endpoint's for everything but a group. The two are separate facts and
    # conflating them is what puts a run on a machine without the weights.
    mutated.worker_id = placement.worker_id
    # Server-derive tenant scope from the target instance so client-supplied
    # cluster_id can't smuggle a benchmark into another tenant, and so the
    # row is visible to the owning Org via cluster_resource_visibility.
    mutated.cluster_id = instance.cluster_id
    mutated.owner_principal_id = instance.owner_principal_id
    return mutated


@router.post(
    "",
    response_model=BenchmarkPublic,
)
async def create_benchmark(
    session: SessionDep, ctx: TenantContextDep, benchmark_in: BenchmarkCreate
):
    existing = await Benchmark.one_by_field(session, "name", benchmark_in.name)
    if existing:
        raise AlreadyExistsException(
            message=f"Benchmark with name '{benchmark_in.name}' already exists."
        )

    mutated = await validate_and_mutate_benchmark_in(session, benchmark_in)
    try:
        benchmark = await Benchmark.create(session, mutated)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to create benchmark: {e}")

    return benchmark


@router.put(
    "/{id}",
    response_model=BenchmarkPublic,
)
async def update_benchmark(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    benchmark_in: BenchmarkUpdate,
):
    benchmark = await Benchmark.one_by_id(session, id)
    assert_resource_visible(ctx, benchmark, not_found_message="Benchmark not found")
    try:
        await benchmark.update(session, benchmark_in)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to update benchmark: {e}")

    return benchmark


@router.patch(
    "/{id}/state",
    response_model=BenchmarkPublic,
)
async def update_benchmark_state(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    state_update: BenchmarkStateUpdate,
):
    benchmark = await Benchmark.one_by_id(session, id)
    assert_resource_visible(ctx, benchmark, not_found_message="Benchmark not found")

    if (
        state_update.state is not None
        and state_update.state == BenchmarkStateEnum.STOPPED
        and benchmark.state
        not in [
            BenchmarkStateEnum.QUEUED,
            BenchmarkStateEnum.PENDING,
            BenchmarkStateEnum.RUNNING,
        ]
    ):
        raise BadRequestException(
            message="Only benchmarks in QUEUED, PENDING, or RUNNING state can be stopped."
        )

    apply_progress_invariant(state_update, benchmark)

    try:
        await benchmark.update(session, state_update)
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to update benchmark state: {e}"
        )

    return benchmark


async def _snapshot_members(
    session: SessionDep, mi: ModelInstance, model: Model
) -> List[ModelInstance]:
    """The members a run's snapshot has to account for.

    For a group that is every running member, not the one the load is sent to.
    The endpoint of a group is its router, which holds no accelerator at all —
    snapshotting it alone reports a run on zero cards, and every per-card
    figure derived from it (`gpu_summary`, tokens per GPU) is then either empty
    or wrong. What the reader wants to compare is what the deployment cost,
    which is the whole group.
    """
    group_id = getattr(mi, "group_id", None)
    if not group_id:
        return [mi]

    members = await ModelInstance.all_by_field(session, "model_id", model.id)
    same_group = [
        m
        for m in members
        if getattr(m, "group_id", None) == group_id
        and m.state == ModelInstanceStateEnum.RUNNING
    ]
    # The endpoint is always in, even if it stopped running between resolution
    # and here: the run is about to be pointed at it.
    if not any(m.id == mi.id for m in same_group):
        same_group.append(mi)
    return sorted(same_group, key=lambda m: m.id)


async def _attached_cache_service_name(
    session: SessionDep,
    model: Model,
    resolved: Dict[int, Optional[str]],
) -> Optional[str]:
    """The name of the shared cache service `model` attaches to, if any.

    `model` here is the *effective* model of one member: `extended_kv_cache` is
    a per-role override, so a group can point prefill and decode at different
    services, or only one of them at a service at all.

    The config carries only the id; the snapshot keeps the name so a report
    still says what it ran against after the service is deleted. `resolved`
    memoizes across the members of one snapshot.
    """
    ext = model.extended_kv_cache
    if not (ext and ext.is_shared() and ext.cache_service_id):
        return None
    if ext.cache_service_id not in resolved:
        cache_service = await CacheService.one_by_id(session, ext.cache_service_id)
        resolved[ext.cache_service_id] = (
            cache_service.name if cache_service is not None else None
        )
    return resolved[ext.cache_service_id]


async def get_benchmark_snapshot(
    session: SessionDep,
    mi: ModelInstance,
    model: Model,
    route_name: Optional[str] = None,
) -> BenchmarkSnapshot:
    # instance snapshot

    worker_snapshots = {}
    gpu_snapshots = {}
    instance_snapshots = {}

    cache_service_names: Dict[int, Optional[str]] = {}

    for member in await _snapshot_members(session, mi, model):
        # Project the role's overrides before snapshotting, rather than handing
        # every member the deployment's own values. `env` and
        # `backend_parameters` are overridable per role precisely because
        # prefill and decode need different ones -- on Ascend they differ down
        # to `HCCL_CONNECT_TIMEOUT` (120 vs 1200). Reading the
        # Model's copy for every member would freeze one intent as all of them,
        # and the report would then attribute settings to a member that never
        # ran with them.
        #
        # `role_effective_model` returns the model itself when there is no role
        # to project, so a plain deployment snapshots the model unchanged.
        effective = role_effective_model(model, getattr(member, "role", None))
        instance_snapshots[member.name] = create_model_instance_snapshot(
            member,
            effective,
            cache_service_name=await _attached_cache_service_name(
                session, effective, cache_service_names
            ),
        )

        if member.worker_id is None:
            continue
        w: Worker = await WorkerService(session).get_by_id(member.worker_id)
        w_snapshot, gpus_snapshots = create_worker_snapshot(
            w, member.gpu_type, member.gpu_indexes
        )
        if w_snapshot is not None:
            worker_snapshots[w.name] = w_snapshot
        if gpus_snapshots is not None:
            gpu_snapshots.update(gpus_snapshots)

        if (
            member.distributed_servers
            and member.distributed_servers.subordinate_workers
        ):
            for sub in member.distributed_servers.subordinate_workers:
                sw: Worker = await WorkerService(session).get_by_id(sub.worker_id)
                w_snapshot, gpus_snapshots = create_worker_snapshot(
                    sw, sub.gpu_type, sub.gpu_indexes
                )
                if w_snapshot is not None:
                    worker_snapshots[sw.name] = w_snapshot
                if gpus_snapshots is not None:
                    gpu_snapshots.update(gpus_snapshots)

    return BenchmarkSnapshot(
        instances=instance_snapshots,
        workers=worker_snapshots,
        gpus=gpu_snapshots,
        # Which generation of the deployment's configuration this run measured,
        # read off the member the load is sent to — one group_id is one
        # spec_digest, so any member answers the same. PD tuning is a sweep
        # over P:D ratio x per-role TP x chunked prefill, so two runs of "the
        # same model" routinely measure different things; without this the
        # reports cannot be told apart after the fact.
        spec_digest=getattr(mi, "spec_digest", None),
        # Set in route mode only: the door the load went through, recorded
        # because a route's targets and weights are editable and a canary can
        # send part of the load somewhere else entirely.
        route_name=route_name,
    )


@router.post(
    "/{id}/metrics",
    response_model=BenchmarkPublic,
)
async def update_benchmark_metrics(
    session: SessionDep, ctx: TenantContextDep, id: int, metrics: BenchmarkMetrics
):
    benchmark = await Benchmark.one_by_id(session, id)
    assert_resource_visible(ctx, benchmark, not_found_message="Benchmark not found")
    try:
        await benchmark.update(session, metrics)
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to update benchmark metrics: {e}"
        )

    return benchmark


@router.post(
    "/{id}/results",
    response_model=BenchmarkPublic,
)
async def update_benchmark_results(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    results: List[BenchmarkResultCreate],
):
    """
    Replace the benchmark's per-point results (one row per (input_tokens, rate)
    grid cell). Idempotent: existing rows for this benchmark are removed first so
    a re-run overwrites cleanly.

    The body is typed (rather than a list of free dicts) so the payload is
    validated at the boundary and carries only the measurement: `benchmark_id`
    comes from the path here, and `id` / timestamps / `deleted_at` stay the
    server's to assign.
    """
    benchmark = await Benchmark.one_by_id(session, id)
    assert_resource_visible(ctx, benchmark, not_found_message="Benchmark not found")
    if len(results) > MAX_BENCHMARK_RESULT_POINTS:
        raise BadRequestException(
            message=(
                f"A benchmark cannot have more than {MAX_BENCHMARK_RESULT_POINTS} "
                f"result points (got {len(results)})."
            )
        )
    try:
        existing = await BenchmarkResult.all_by_field(session, "benchmark_id", id)
        for row in existing:
            await row.delete(session, auto_commit=False)
        for data in results:
            await BenchmarkResult.create(
                session,
                source={**data.model_dump(), "benchmark_id": id},
                auto_commit=False,
            )
        await session.commit()
    except Exception as e:
        await session.rollback()
        raise InternalServerErrorException(
            message=f"Failed to update benchmark results: {e}"
        )

    return benchmark


@router.get(
    "/{id}/results",
    response_model=List[BenchmarkResultPublic],
)
async def get_benchmark_results(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    include_raw: bool = Query(
        True,
        description=(
            "Include each point's raw_metrics dump. ON by default because it is not "
            "an extra: the detail page reads its percentile charts, per-point "
            "duration and early-stop reason straight out of it. Pass false for a "
            "caller that only needs the load curve — the column is then left out of "
            "the query, which matters because a multi-point grid runs to megabytes "
            "and the page re-pulls the whole thing on every partial-sync event."
        ),
    ),
):
    benchmark = await Benchmark.one_by_id(session, id)
    assert_resource_visible(ctx, benchmark, not_found_message="Benchmark not found")
    # When the caller opts out, mirror the list route's defer(Benchmark.raw_metrics)
    # so the heavy column is never read rather than fetched and discarded.
    options = None if include_raw else [defer(BenchmarkResult.raw_metrics)]
    results = await BenchmarkResult.all_by_fields(
        session, fields={"benchmark_id": id}, options=options
    )
    results = sorted(results, key=lambda r: (r.input_tokens or 0, r.sequence))
    return [_result_to_public(r, include_raw=include_raw) for r in results]


def _result_to_public(row: BenchmarkResult, include_raw: bool) -> BenchmarkResultPublic:
    """Build the response object field by field.

    `raw_metrics` is read ONLY when it was actually loaded. Letting the serializer
    reach for a deferred column would be a lazy load, and on an async session that
    raises instead of quietly fetching — so the response is assembled here rather
    than by handing the ORM row to `response_model`.
    """
    data = {
        name: getattr(row, name)
        for name in BenchmarkResultPublic.model_fields
        if name != "raw_metrics"
    }
    data["raw_metrics"] = row.raw_metrics if include_raw else None
    return BenchmarkResultPublic.model_validate(data)


@router.delete(
    "/{id}",
)
async def delete_benchmark(session: SessionDep, ctx: TenantContextDep, id: int):
    benchmark = await Benchmark.one_by_id(session, id)
    assert_resource_visible(ctx, benchmark, not_found_message="Benchmark not found")

    try:
        await benchmark.delete(session)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to delete benchmark: {e}")


@router.get("/{id}/logs")
async def get_benchmark_logs(  # noqa: C901
    request: Request,
    ctx: TenantContextDep,
    id: int,
    log_options: LogOptionsDep,
):
    # Inline session released after the initial lookups so a long-lived
    # follow-log stream doesn't hold a database connection for its duration.
    async with async_session() as session:
        benchmark = await Benchmark.one_by_id(session, id)
        assert_resource_visible(ctx, benchmark, not_found_message="Benchmark not found")

        worker = await Worker.one_by_id(session, benchmark.worker_id)
        if not worker:
            raise NotFoundException(message="Benchmark's worker not found")

        if benchmark.state in [
            BenchmarkStateEnum.ERROR,
            BenchmarkStateEnum.STOPPED,
            BenchmarkStateEnum.COMPLETED,
        ]:
            log_options.follow = False

    timeout = aiohttp.ClientTimeout(total=envs.PROXY_TIMEOUT, sock_connect=5)

    if log_options.follow:

        def on_exception(e: Exception, t: aiohttp.ClientTimeout) -> tuple[str, int]:
            msg = (
                str(e)
                if not isinstance(e, TimeoutError)
                else f"Log stream timed out ({t.total} seconds). Please reopen the log page."
            )
            return f"\x1b[999;1H{msg}\n", status.HTTP_500_INTERNAL_SERVER_ERROR

        return StreamingResponseWithStatusCode(
            stream_to_worker(
                worker=worker,
                method="GET",
                path=f"benchmark_logs/{benchmark.id}",
                proxy_client=request.app.state.http_client,
                no_proxy_client=request.app.state.http_client_no_proxy,
                params={
                    "tail": log_options.tail,
                    "follow": log_options.follow,
                    "benchmark_name": benchmark.name,
                },
                timeout=timeout,
                on_exception=on_exception,
                raw=True,
            ),
            media_type="application/octet-stream",
        )
    else:
        resp, body = await request_to_worker(
            worker=worker,
            method="GET",
            path=f"benchmark_logs/{benchmark.id}",
            proxy_client=request.app.state.http_client,
            no_proxy_client=request.app.state.http_client_no_proxy,
            params={
                "tail": log_options.tail,
                "follow": log_options.follow,
                "benchmark_name": benchmark.name,
            },
            timeout=timeout,
        )
        return PlainTextResponse(
            content=body.decode() if body else "", status_code=resp.status
        )


@router.post("/export")
async def export_benchmarks(
    session: SessionDep,
    ctx: TenantContextDep,
    ids: list[int],
):
    if not ids:
        raise BadRequestException(message="No benchmark ids provided.")

    if len(ids) > MAX_EXPORT_RECORDS:
        raise BadRequestException(
            message=f"Export up to {MAX_EXPORT_RECORDS} records at most."
        )

    exclude_fields = [
        "id",
        "cluster_id",
        "owner_principal_id",
        "model_id",
        "worker_id",
        "created_at",
        "updated_at",
        "pid",
        "progress",
        "state_message",
        "state",
        "deleted_at",
    ]
    extra_conditions = [
        col(Benchmark.id).in_(ids),
        *tenant_list_conditions(ctx, Benchmark),
    ]
    benchmarks: Sequence[Benchmark] = await Benchmark.all_by_fields(
        session, fields={}, extra_conditions=extra_conditions
    )
    exported_benchmarks = []
    for b in benchmarks:
        eb = b.model_dump(exclude=set(exclude_fields))
        exported_benchmarks.append(order_benchmark_export_fields(eb))

    export_data = {"benchmarks": exported_benchmarks}
    yaml_str = yaml.safe_dump(export_data, allow_unicode=True, sort_keys=False)
    return PlainTextResponse(content=yaml_str, media_type="application/x-yaml")
