from sqlmodel import col
import yaml
from typing import Optional, Sequence
import aiohttp
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
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
from gpustack.schemas.models import (
    Model,
    ModelInstance,
    ModelInstanceStateEnum,
    is_audio_model,
    is_embedding_model,
    is_image_model,
    is_renaker_model,
)
from gpustack.schemas.workers import Worker
from gpustack.server.deps import SessionDep
from gpustack.schemas.benchmark import (
    DATASET_RANDOM,
    DATASET_SHAREGPT,
    Benchmark,
    BenchmarkCreate,
    BenchmarkFullPublic,
    BenchmarkListParams,
    BenchmarkMetrics,
    BenchmarkSnapshot,
    BenchmarkStateEnum,
    BenchmarkStateUpdate,
    BenchmarkUpdate,
    BenchmarkPublic,
    BenchmarksPublic,
)

from gpustack.server.services import (
    WorkerService,
)
from gpustack.utils.gpu import summary_gpu_snapshots
from gpustack.utils.network import use_proxy_env_for_url
from gpustack.utils.snapshot import (
    create_model_instance_snapshot,
    create_worker_snapshot,
)
from gpustack.worker.logs import LogOptionsDep
from sqlalchemy.orm import defer

MAX_EXPORT_RECORDS = 20

router = APIRouter()


@router.get("", response_model=BenchmarksPublic)
async def get_benchmarks(
    session: SessionDep,
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
):
    return await _get_benchmarks(
        session=session,
        params=params,
        state=state,
        search=search,
        model_name=model_name,
        gpu_summary=gpu_summary,
        dataset_name=dataset_name,
        profile=profile,
    )


def gpu_summary_filter(data: Benchmark, gpu_summary: Optional[str]) -> bool:
    if (
        gpu_summary
        and data.gpu_summary
        and gpu_summary.lower() not in data.gpu_summary.lower()
    ):
        return False
    return True


async def _get_benchmarks(
    session: SessionDep,
    params: BenchmarkListParams,
    search: str = None,
    state: Optional[BenchmarkStateEnum] = None,
    model_name: Optional[str] = None,
    gpu_summary: Optional[str] = None,
    dataset_name: Optional[str] = None,
    profile: Optional[str] = None,
):
    fuzzy_fields = {}
    if search:
        fuzzy_fields["name"] = search

    if profile:
        fuzzy_fields["profile"] = profile

    fields = {}
    if state:
        fields["state"] = state

    if model_name:
        fields["model_name"] = model_name

    if dataset_name:
        fields["dataset_name"] = dataset_name

    extra_conditions = []
    if gpu_summary:
        extra_conditions.append(
            func.lower(Benchmark.gpu_summary).like(f"%{gpu_summary}%")
        )

    if params.watch:
        return StreamingResponse(
            Benchmark.streaming(
                fields=fields,
                fuzzy_fields=fuzzy_fields,
                filter_func=lambda data: gpu_summary_filter(data, gpu_summary),
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
    id: int,
):
    benchmark = await Benchmark.one_by_id(session, id)
    if not benchmark:
        raise NotFoundException(message=f"Benchmark {id} not found")
    return benchmark


async def validate_and_mutate_benchmark_in(
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

    if benchmark_in.dataset_name not in [DATASET_RANDOM, DATASET_SHAREGPT]:
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

    model = await Model.one_by_id(session, mutated.model_id)
    if not model:
        raise BadRequestException(message=f"Model {mutated.model_id} not found")

    if (
        is_image_model(model)
        or is_audio_model(model)
        or is_embedding_model(model)
        or is_renaker_model(model)
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

    snapshot = await get_benchmark_snapshot(session, instance, model)
    mutated.snapshot = snapshot
    mutated.gpu_summary, mutated.gpu_vendor_summary = summary_gpu_snapshots(
        snapshot.gpus
    )
    mutated.worker_id = instance.worker_id
    return mutated


@router.post("", response_model=BenchmarkPublic)
async def create_benchmark(session: SessionDep, benchmark_in: BenchmarkCreate):
    existing = await Benchmark.one_by_field(session, "name", benchmark_in.name)
    if existing:
        raise AlreadyExistsException(
            message=f"Benchmark '{benchmark_in.name}' already exists. "
            "Please choose a different name or check the existing benchmark."
        )

    mutated = await validate_and_mutate_benchmark_in(session, benchmark_in)
    try:
        benchmark = await Benchmark.create(session, mutated)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to create benchmark: {e}")

    return benchmark


@router.put("/{id}", response_model=BenchmarkPublic)
async def update_benchmark(session: SessionDep, id: int, benchmark_in: BenchmarkUpdate):
    benchmark = await Benchmark.one_by_id(session, id)
    if not benchmark:
        raise NotFoundException(message="Benchmark not found")
    try:
        await benchmark.update(session, benchmark_in)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to update benchmark: {e}")

    return benchmark


@router.patch("/{id}/state", response_model=BenchmarkPublic)
async def update_benchmark_state(
    session: SessionDep, id: int, state_update: BenchmarkStateUpdate
):
    benchmark = await Benchmark.one_by_id(session, id)
    if not benchmark:
        raise NotFoundException(message="Benchmark not found")

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


@router.post("/{id}/metrics", response_model=BenchmarkPublic)
async def update_benchmark_metrics(
    session: SessionDep, id: int, metrics: BenchmarkMetrics
):
    benchmark = await Benchmark.one_by_id(session, id)
    if not benchmark:
        raise NotFoundException(message="Benchmark not found")
    try:
        await benchmark.update(session, metrics)
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to update benchmark metrics: {e}"
        )

    return benchmark


@router.delete("/{id}")
async def delete_benchmark(session: SessionDep, id: int):
    benchmark = await Benchmark.one_by_id(session, id)
    if not benchmark:
        raise NotFoundException(message="Benchmark not found")

    try:
        await benchmark.delete(session)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to delete benchmark: {e}")


@router.get("/{id}/logs")
async def get_benchmark_logs(  # noqa: C901
    request: Request, session: SessionDep, id: int, log_options: LogOptionsDep
):
    benchmark = await Benchmark.one_by_id(session, id)
    if not benchmark:
        raise NotFoundException(message="Benchmark not found")

    worker = await Worker.one_by_id(session, benchmark.worker_id)
    if not worker:
        raise NotFoundException(message="Benchmark's worker not found")

    if benchmark.state in [
        BenchmarkStateEnum.ERROR,
        BenchmarkStateEnum.STOPPED,
        BenchmarkStateEnum.COMPLETED,
    ]:
        log_options.follow = False

    benchmark_log_url = (
        f"http://{worker.advertise_address}:{worker.port}/benchmark_logs"
        f"/{benchmark.id}?{log_options.url_encode()}&benchmark_name={benchmark.name}"
    )

    timeout = aiohttp.ClientTimeout(total=envs.PROXY_TIMEOUT, sock_connect=5)

    use_proxy_env = use_proxy_env_for_url(benchmark_log_url)
    client: aiohttp.ClientSession = (
        request.app.state.http_client
        if use_proxy_env
        else request.app.state.http_client_no_proxy
    )

    if log_options.follow:

        async def proxy_stream():
            try:
                async with client.get(benchmark_log_url, timeout=timeout) as resp:
                    if resp.status != 200:
                        body = await resp.read()
                        yield body, resp.headers, resp.status
                        return

                    async for chunk in resp.content.iter_any():
                        yield chunk, resp.headers, resp.status
            except TimeoutError:
                yield "\x1b[999;1H" + f"Log stream timed out ({timeout.total} seconds). Please reopen the log page.\n", {}, status.HTTP_500_INTERNAL_SERVER_ERROR
            except Exception as e:
                yield "\x1b[999;1H" + f"Error fetching benchmark logs: {str(e)}\n", {}, status.HTTP_500_INTERNAL_SERVER_ERROR

        return StreamingResponseWithStatusCode(
            proxy_stream(),
            media_type="application/octet-stream",
        )
    else:
        try:
            async with client.get(benchmark_log_url, timeout=timeout) as resp:
                if resp.status != 200:
                    raise HTTPException(
                        status_code=resp.status,
                        detail="Error fetching benchmark logs",
                    )
                content = await resp.text()
            return PlainTextResponse(content=content, status_code=resp.status)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error fetching benchmark logs: {str(e)}\n"
            )


@router.post("/export")
async def export_benchmarks(
    session: SessionDep,
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
    ]
    benchmarks: Sequence[Benchmark] = await Benchmark.all_by_fields(
        session, fields={}, extra_conditions=extra_conditions
    )
    exported_benchmarks = []
    for b in benchmarks:
        eb = b.model_dump(exclude=set(exclude_fields))
        exported_benchmarks.append(eb)

    export_data = {"benchmarks": exported_benchmarks}
    yaml_str = yaml.safe_dump(export_data, allow_unicode=True, sort_keys=False)
    return PlainTextResponse(content=yaml_str, media_type="application/x-yaml")
