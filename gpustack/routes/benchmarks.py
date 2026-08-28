from sqlmodel import col
import yaml
from typing import List, Optional, Sequence
import aiohttp
from fastapi import APIRouter, Depends, Query, Request, status
from fastapi.responses import PlainTextResponse, StreamingResponse
from sqlmodel import func
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
from gpustack.schemas.models import (
    Model,
    ModelInstance,
    ModelInstanceStateEnum,
    is_audio_model,
    is_embedding_model,
    is_image_model,
    is_reranker_model,
)
from gpustack.schemas.workers import Worker
from gpustack.server.db import async_session
from gpustack.server.deps import SessionDep, TenantContextDep
from gpustack.schemas.benchmark import (
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
    )


def _fuzzy_contains(value: Optional[str], target: Optional[str]) -> bool:
    """Return False only when the filter value is set but not contained in target."""
    if not value:
        return True
    if not target:
        return False
    return value.lower() in target.lower()


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


async def _get_benchmarks(
    ctx,
    params: BenchmarkListParams,
    search: str = None,
    state: Optional[BenchmarkStateEnum] = None,
    model_name: Optional[str] = None,
    gpu_summary: Optional[str] = None,
    dataset_name: Optional[str] = None,
    profile: Optional[str] = None,
    load_type: Optional[BenchmarkLoadTypeEnum] = None,
):
    fuzzy_fields = {}
    if search:
        fuzzy_fields["name"] = search

    fields = {}
    if state:
        fields["state"] = state

    if dataset_name:
        fields["dataset_name"] = dataset_name

    # `load_type` (fixed_rate / concurrency) filter (exact match; every row
    # carries a load_type).
    def _load_type_match(data) -> bool:
        return not load_type or data.load_type == load_type

    extra_conditions = list(tenant_list_conditions(ctx, Benchmark))
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

    _benchmark_visible = _make_benchmark_visibility_filter(ctx)

    if params.watch:
        return StreamingResponse(
            Benchmark.streaming(
                fields=fields,
                fuzzy_fields=fuzzy_fields,
                filter_func=lambda data: _benchmark_visible(data)
                and gpu_summary_filter(data, gpu_summary)
                and _fuzzy_contains(profile, data.profile)
                and _fuzzy_contains(model_name, data.model_name)
                and _load_type_match(data),
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
            fuzzy_fields=fuzzy_fields,
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


async def validate_and_mutate_benchmark_in(  # noqa: C901
    session: SessionDep, benchmark_in: BenchmarkCreate
) -> Benchmark:

    if not benchmark_in.model_instance_name.strip():
        raise BadRequestException(message="Field model_instance_name must be specified")

    mutated = Benchmark(**benchmark_in.model_dump())
    instance = await ModelInstance.one_by_field(
        session, "name", benchmark_in.model_instance_name
    )
    if not instance:
        raise BadRequestException(
            message=f"Model instance '{benchmark_in.model_instance_name}' not found"
        )

    if instance.state != ModelInstanceStateEnum.RUNNING:
        raise BadRequestException(
            message=f"Model instance '{benchmark_in.model_instance_name}' not in RUNNING state"
        )

    if benchmark_in.model_id is None:
        mutated.model_id = instance.model_id
        mutated.model_name = instance.model_name

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

    model = await Model.one_by_id(session, mutated.model_id)
    if not model:
        raise BadRequestException(message=f"Model {mutated.model_id} not found")

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

    snapshot = await get_benchmark_snapshot(session, instance, model)
    mutated.snapshot = snapshot
    mutated.gpu_summary, mutated.gpu_vendor_summary = summary_gpu_snapshots(
        snapshot.gpus
    )
    mutated.worker_id = instance.worker_id
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


async def get_benchmark_snapshot(
    session: SessionDep, mi: ModelInstance, model: Model
) -> BenchmarkSnapshot:
    # instance snapshot

    worker_snapshots = {}
    gpu_snapshots = {}
    instance_snapshots = {}

    instance_snapshots[mi.name] = create_model_instance_snapshot(mi, model)

    w: Worker = await WorkerService(session).get_by_id(mi.worker_id)
    w_snapshot, gpus_snapshots = create_worker_snapshot(w, mi.gpu_type, mi.gpu_indexes)
    if w_snapshot is not None:
        worker_snapshots[w.name] = w_snapshot
    if gpus_snapshots is not None:
        gpu_snapshots.update(gpus_snapshots)

    if mi.distributed_servers and mi.distributed_servers.subordinate_workers:
        for sub in mi.distributed_servers.subordinate_workers:
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
