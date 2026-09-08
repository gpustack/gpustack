import asyncio
import json
from typing import AsyncIterator, Dict, List, Optional, Tuple
import aiohttp
from fastapi import APIRouter, Request, status, HTTPException
from fastapi.responses import (
    PlainTextResponse,
    StreamingResponse,
    RedirectResponse,
)
from urllib.parse import quote, urlencode

from gpustack.api.responses import StreamingResponseWithStatusCode
from gpustack import envs
from gpustack.server.services import ModelInstanceService
from gpustack.server.worker_request import request_to_worker, stream_to_worker
from gpustack.utils.command import resolve_executor_backend
from gpustack.worker.logs import LogOptionsDep
from gpustack.api.exceptions import (
    InternalServerErrorException,
    NotFoundException,
)
from gpustack.schemas.workers import Worker
from gpustack.schemas.clusters import Cluster
from gpustack.api.tenant import (
    bypass_tenant_filter,
    assert_resource_visible,
    cluster_scoped_system,
    scoped_cluster_row_visible,
    tenant_list_conditions,
)
from gpustack.server.db import async_session
from gpustack.server.deps import ListParamsDep, SessionDep, TenantContextDep
from gpustack.schemas.models import (
    BackendEnum,
    Model,
    ModelInstance,
    ModelInstanceCreate,
    ModelInstanceLogOptions,
    ModelInstanceLogWorkerOption,
    ModelInstancePublic,
    ModelInstanceUpdate,
    ModelInstancesPublic,
    ModelInstanceStateEnum,
    ServeLogOptionsResponse,
    is_dp_node_per_instance,
)
from gpustack.schemas.model_files import ModelFileStateEnum
from gpustack.config.config import get_global_config
from gpustack.utils.grafana import resolve_grafana_base_url
from gpustack.utils.tabular_export import stream_zip

router = APIRouter()


def _is_vllm_ray_subordinate(model_instance: ModelInstance) -> bool:
    """Whether a vLLM subordinate node is part of the Ray sidecar path.

    Native multi-node (mp) subordinates — regardless of dp_only/mp_only/nested
    shape — fall through to ``sub-vllm`` because the log viewer doesn't need
    that level of granularity.
    """
    if model_instance.backend != BackendEnum.VLLM:
        return False
    model = model_instance.model
    backend_parameters = model.backend_parameters if model else None
    backend_version = model.backend_version if model else None
    return resolve_executor_backend(backend_parameters, backend_version) == "ray"


def _default_display_name(model_instance: ModelInstance, is_main_worker: bool) -> str:
    """Resolve the UI display name for the internal 'default' container."""
    backend = model_instance.backend
    backend_name = backend.value if hasattr(backend, "value") else backend
    if is_main_worker:
        return backend_name or "default"
    if _is_vllm_ray_subordinate(model_instance):
        return "ray-worker"
    # Generic subordinate: "sub-<backend>" or just "subordinate".
    return f"sub-{backend_name}" if backend_name else "subordinate"


def _map_container_display_name(
    internal_name: str, model_instance: ModelInstance, is_main_worker: bool
) -> str:
    """Forward-map an internal container name to its UI display name."""
    if internal_name != "default":
        return internal_name
    return _default_display_name(model_instance, is_main_worker)


def _unmap_container_display_name(
    display_name: str, model_instance: ModelInstance, is_main_worker: bool
) -> str:
    """Reverse-map a UI display name back to the internal container name."""
    if display_name == _default_display_name(model_instance, is_main_worker):
        return "default"
    return display_name


@router.get("", response_model=ModelInstancesPublic)
async def get_model_instances(
    ctx: TenantContextDep,
    params: ListParamsDep,
    id: Optional[int] = None,
    model_id: Optional[int] = None,
    worker_id: Optional[int] = None,
    cluster_id: Optional[int] = None,
    state: Optional[str] = None,
    search: Optional[str] = None,
):
    fields = {}
    search = search.strip() if search else None
    fuzzy_fields = {"name": search} if search else {}
    if id:
        fields["id"] = id

    if model_id:
        fields["model_id"] = model_id

    if worker_id:
        fields["worker_id"] = worker_id

    if cluster_id:
        fields["cluster_id"] = cluster_id

    if state:
        fields["state"] = state

    # System principals (workers, cluster service accounts) and admin in
    # "All" mode must see every Org's instances regardless of their
    # ``principal_id`` — otherwise a worker's awatch stream
    # would silently filter out instances scheduled to it on clusters
    # outside its Personal Org.
    if ctx.current_principal_id is not None and not bypass_tenant_filter(ctx):
        fields["owner_principal_id"] = ctx.current_principal_id

    if params.watch:
        # Cluster-bound service accounts (worker / cluster bootstrap)
        # only stream instances of their own cluster.
        filter_func = (
            (lambda data: scoped_cluster_row_visible(ctx, data))
            if cluster_scoped_system(ctx)
            else None
        )
        return StreamingResponse(
            ModelInstance.streaming(
                fields=fields,
                fuzzy_fields=fuzzy_fields,
                filter_func=filter_func,
            ),
            media_type="text/event-stream",
        )

    async with async_session() as session:
        extra_conditions = tenant_list_conditions(ctx, ModelInstance)
        return await ModelInstance.paginated_by_query(
            session=session,
            fields=fields,
            fuzzy_fields=fuzzy_fields,
            extra_conditions=extra_conditions,
            page=params.page,
            per_page=params.perPage,
        )


@router.get("/{id}", response_model=ModelInstancePublic)
async def get_model_instance(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
):
    model_instance = await ModelInstance.one_by_id(session, id)
    assert_resource_visible(
        ctx,
        model_instance,
        not_found_message="Model instance not found",
    )
    return model_instance


@router.get("/{id}/dashboard")
async def get_model_instance_dashboard(
    session: SessionDep,
    id: int,
    request: Request,
):
    model_instance = await ModelInstance.one_by_id(session, id)
    if not model_instance:
        raise NotFoundException(message="Model instance not found")

    cfg = get_global_config()
    if not cfg.get_grafana_url() or not cfg.grafana_model_dashboard_uid:
        raise InternalServerErrorException(
            message="Grafana dashboard settings are not configured"
        )

    cluster = None
    if model_instance.cluster_id is not None:
        cluster = await Cluster.one_by_id(session, model_instance.cluster_id)

    query_params = {}
    if cluster is not None:
        query_params["var-cluster_name"] = cluster.name
    query_params["var-model_name"] = model_instance.model_name
    query_params["var-model_instance_name"] = model_instance.name

    grafana_base = resolve_grafana_base_url(cfg, request)
    slug = "gpustack-model"
    dashboard_url = f"{grafana_base}/d/{cfg.grafana_model_dashboard_uid}/{slug}"
    if query_params:
        dashboard_url = f"{dashboard_url}?{urlencode(query_params)}"

    return RedirectResponse(url=dashboard_url, status_code=302)


async def fetch_model_instance(session, ctx, id):
    model_instance = await ModelInstance.one_by_id_with_model_files(session, id)
    # Check visibility before the worker-assignment check so a missing
    # instance and a cross-tenant instance return the same "not found"
    # response instead of a distinguishable "not assigned to a worker" one.
    assert_resource_visible(
        ctx,
        model_instance,
        not_found_message="Model instance not found",
    )
    if not model_instance.worker_id:
        raise NotFoundException(message="Model instance not assigned to a worker")
    return model_instance


async def fetch_worker(session, worker_id):
    worker = await Worker.one_by_id(session, worker_id)
    if not worker:
        raise NotFoundException(message="Model instance's worker not found")
    return worker


@router.get("/{id}/logs")
async def get_serving_logs(  # noqa: C901
    request: Request,
    ctx: TenantContextDep,
    id: int,
    log_options: LogOptionsDep,
    worker_id: Optional[int] = None,
    container_name: Optional[str] = None,
):
    # Inline session released after the initial lookups so a long-lived
    # follow-log stream doesn't hold a database connection for its duration.
    async with async_session() as session:
        model_instance = await fetch_model_instance(session, ctx, id)

        # Reverse-map: convert UI display name back to internal container name.
        if container_name:
            is_main = (
                worker_id or model_instance.worker_id
            ) == model_instance.worker_id
            container_name = _unmap_container_display_name(
                container_name, model_instance, is_main
            )

        # Build valid worker IDs (main worker + subordinate workers for distributed instances)
        valid_worker_ids = {model_instance.worker_id}
        if (
            model_instance.distributed_servers
            and model_instance.distributed_servers.subordinate_workers
        ):
            valid_worker_ids.update(
                sw.worker_id
                for sw in model_instance.distributed_servers.subordinate_workers
            )

        # Determine target worker ID
        target_worker_id = worker_id or model_instance.worker_id
        if target_worker_id not in valid_worker_ids:
            raise NotFoundException(
                message=f"Worker {target_worker_id} not found for model instance {id}"
            )

        worker = await fetch_worker(session, target_worker_id)

        params = _build_serve_log_params(
            model_instance,
            tail=log_options.tail,
            follow=log_options.follow,
            previous=log_options.previous,
            container_name=container_name,
        )

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
                path=f"serveLogs/{model_instance.id}",
                proxy_client=request.app.state.http_client,
                no_proxy_client=request.app.state.http_client_no_proxy,
                params=params,
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
            path=f"serveLogs/{model_instance.id}",
            proxy_client=request.app.state.http_client,
            no_proxy_client=request.app.state.http_client_no_proxy,
            params=params,
            timeout=timeout,
        )
        return PlainTextResponse(
            content=body.decode() if body else "", status_code=resp.status
        )


@router.get(
    "/{id}/logs/download",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Plain text for one log stream, a zip archive for several.",
            "content": {"text/plain": {}, "application/zip": {}},
        }
    },
)
async def download_serving_logs(
    request: Request,
    ctx: TenantContextDep,
    id: int,
):
    """Download a model instance's complete current logs as a file.

    Collects every worker (main + subordinates) and every container per worker
    (the default main workload plus Ray-style sidecars). A single log stream
    downloads as a plain-text .log; multiple streams as a flat zip with one file
    per worker/container. Always the full current logs (no tail/follow/previous).
    Both shapes stream, so neither a log nor the archive is ever buffered whole.
    """
    # Session released before streaming so a download doesn't hold a connection.
    async with async_session() as session:
        model_instance = await fetch_model_instance(session, ctx, id)
        targets = await resolve_instance_log_worker_targets(session, model_instance)

    # Discover every (worker, container) log stream once (one options call/worker).
    streams = await _plan_log_streams(request, model_instance, targets)

    # A dead end has to fail before streaming fixes the status. Only structural
    # ones qualify: a failed discovery still falls back to the main workload.
    if all(stream["worker"] is None for stream in streams):
        detail = "; ".join(
            f"{stream['worker_label']}: {stream['error']}" for stream in streams
        )
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch logs from all workers: {detail}",
        )

    if len(streams) == 1:
        return _stream_single_worker_log(request, model_instance, streams[0])
    return _stream_zipped_worker_logs(request, model_instance, streams)


def _stream_single_worker_log(
    request: Request,
    model_instance: ModelInstance,
    stream: dict,
) -> StreamingResponse:
    """Stream one worker/container's logs straight through, without buffering."""
    filename = _sanitize_filename(f"{model_instance.name or model_instance.id}.log")
    return StreamingResponse(
        _worker_log_chunks(request, model_instance, stream),
        media_type="text/plain; charset=utf-8",
        headers=_attachment_headers(filename),
    )


def _stream_zipped_worker_logs(
    request: Request,
    model_instance: ModelInstance,
    streams: List[dict],
) -> StreamingResponse:
    """Stream a flat zip with one member per log; peak memory is one chunk."""
    zip_name = _sanitize_filename(
        f"{model_instance.name or model_instance.id}.logs.zip"
    )
    return StreamingResponse(
        stream_zip(_zip_members(request, model_instance, streams)),
        media_type="application/zip",
        headers=_attachment_headers(zip_name),
    )


async def _zip_members(
    request: Request,
    model_instance: ModelInstance,
    streams: List[dict],
) -> AsyncIterator[Tuple[str, AsyncIterator[bytes]]]:
    """Name each stream, suffixing a repeated worker/container label."""
    seen: set[str] = set()
    for stream in streams:
        label = f"{stream['worker_label']}.{stream['container_display']}"
        name = _sanitize_filename(f"{label}.log")
        index = 1
        while name in seen:
            name = _sanitize_filename(f"{label}.{index}.log")
            index += 1
        seen.add(name)
        yield name, _worker_log_chunks(request, model_instance, stream)


async def _worker_log_chunks(
    request: Request,
    model_instance: ModelInstance,
    stream: dict,
) -> AsyncIterator[bytes]:
    """One stream's complete logs; its own failures become file content."""
    worker = stream["worker"]
    if worker is None:
        yield f"Failed to fetch logs: {stream['error']}\n".encode()
        return

    # Discovery failed but the fallback still reads the main workload. Say so,
    # or a sidecar log missing from the archive looks like it never existed.
    if stream["error"]:
        yield (
            f"Note: container discovery failed ({stream['error']}); "
            "this log covers the main workload only.\n"
        ).encode()

    params = _build_serve_log_params(
        model_instance, container_name=stream["container_internal"]
    )
    timeout = aiohttp.ClientTimeout(total=envs.PROXY_TIMEOUT, sock_connect=5)
    try:
        async for chunk, _, status_code in stream_to_worker(
            worker=worker,
            method="GET",
            path=f"serveLogs/{model_instance.id}",
            proxy_client=request.app.state.http_client,
            no_proxy_client=request.app.state.http_client_no_proxy,
            params=params,
            timeout=timeout,
            raw=True,
        ):
            payload = chunk if isinstance(chunk, bytes) else chunk.encode()
            # Label an error body so it can't be read as real log output.
            if status_code >= 400:
                yield f"Failed to fetch logs: HTTP {status_code}: ".encode() + payload
                return
            yield payload
    except Exception as e:
        yield f"\nFailed to fetch logs: {e}\n".encode()


def _sanitize_filename(name: str) -> str:
    """Make a name safe as a download filename / zip entry name.

    Non-ASCII stays: zip entries are UTF-8, and the header percent-encodes it.
    """
    cleaned = name.replace("/", "_").replace("\\", "_").replace('"', "_")
    cleaned = "".join(ch for ch in cleaned if ch.isprintable()).strip(". ")
    return cleaned or "logs"


def _attachment_headers(filename: str) -> Dict[str, str]:
    """A Content-Disposition that survives a non-ASCII filename (RFC 6266).

    Starlette encodes headers as latin-1, so a CJK name needs ``filename*``.
    """
    quoted = quote(filename, safe="")
    if quoted == filename:
        return {"Content-Disposition": f'attachment; filename="{filename}"'}
    ascii_filename = "".join(
        ch if ch.isascii() and ch.isprintable() else "_" for ch in filename
    )
    return {
        "Content-Disposition": (
            f'attachment; filename="{ascii_filename}"; filename*=UTF-8\'\'{quoted}'
        )
    }


def _build_serve_log_params(
    model_instance: ModelInstance,
    tail: int = -1,
    follow: bool = False,
    previous: bool = False,
    container_name: Optional[str] = None,
) -> dict:
    """Proxy params for one container's logs; defaults to a full, non-follow read."""
    params = {
        "tail": tail,
        "follow": follow,
        "model_instance_name": model_instance.name,
        "previous": previous,
    }
    if container_name:
        params["container_name"] = container_name
    if (
        model_instance.state != ModelInstanceStateEnum.RUNNING
        and model_instance.model_files
        and model_instance.model_files[0].state != ModelFileStateEnum.READY
    ):
        params["model_file_id"] = model_instance.model_files[0].id
    return params


async def _discover_worker_containers(
    request: Request, worker: Worker, model_instance_id: int
) -> Tuple[List[str], Optional[str]]:
    """Internal container names for one worker, plus any discovery error.

    A failed discovery still falls back to ["default"] (the main workload logs).
    """
    try:
        payload = await fetch_serve_log_options_from_worker(
            request, worker, model_instance_id
        )
    except Exception as e:
        return ["default"], str(e)
    current = next((entry for entry in payload.restarts if not entry.previous), None)
    containers = current.containers if current else []
    return list(containers) or ["default"], None


async def _plan_log_streams(
    request: Request,
    model_instance: ModelInstance,
    targets: List[Tuple[int, str, Optional[Worker]]],
) -> List[dict]:
    """Discover every (worker, container) log stream, one options call per worker.

    Returns flat stream descriptors {"worker_label", "worker", "container_internal",
    "container_display", "error"}. A worker missing from the DB still yields one
    placeholder stream so its absence is reported instead of silently dropped.
    """

    async def per_worker(target: Tuple[int, str, Optional[Worker]]) -> List[dict]:
        worker_id, worker_name, worker = target
        label = worker_name or f"worker-{worker_id}"
        if worker is None:
            return [
                {
                    "worker_label": label,
                    "worker": None,
                    "container_internal": "default",
                    "container_display": "default",
                    "error": "Worker not found in database",
                }
            ]
        is_main = worker_id == model_instance.worker_id
        containers, error = await _discover_worker_containers(
            request, worker, model_instance.id
        )
        return [
            {
                "worker_label": label,
                "worker": worker,
                "container_internal": container_internal,
                "container_display": _map_container_display_name(
                    container_internal, model_instance, is_main
                ),
                "error": error,
            }
            for container_internal in containers
        ]

    grouped = await asyncio.gather(*[per_worker(t) for t in targets])
    return [stream for group in grouped for stream in group]


async def resolve_instance_log_worker_targets(
    session, model_instance: ModelInstance
) -> List[Tuple[int, str, Optional[Worker]]]:
    """
    Ordered targets: main worker, then distributed subordinate workers.
    Worker may be None if the subordinate id is not present in DB (cannot proxy HTTP).
    """
    targets: List[Tuple[int, str, Optional[Worker]]] = []
    seen: set[int] = set()

    main_id = model_instance.worker_id
    if main_id is not None and main_id not in seen:
        main_worker = await fetch_worker(session, main_id)
        targets.append((main_id, main_worker.name or "", main_worker))
        seen.add(main_id)

    dservers = model_instance.distributed_servers
    if dservers and dservers.subordinate_workers:
        for sw in dservers.subordinate_workers:
            wid = sw.worker_id
            if wid is None or wid in seen:
                continue
            name = sw.worker_name or ""
            w = await Worker.one_by_id(session, wid)
            if not name:
                name = w.name if w else ""
            targets.append((wid, name or "", w))
            seen.add(wid)

    return targets


async def fetch_serve_log_options_from_worker(
    request: Request,
    worker: Worker,
    model_instance_id: int,
) -> ServeLogOptionsResponse:
    timeout = aiohttp.ClientTimeout(total=envs.PROXY_TIMEOUT, sock_connect=5)
    try:
        resp, body = await request_to_worker(
            worker=worker,
            method="GET",
            path=f"serveLogOptions/{model_instance_id}",
            proxy_client=request.app.state.http_client,
            no_proxy_client=request.app.state.http_client_no_proxy,
            timeout=timeout,
        )
    except Exception as e:
        raise ValueError(str(e)) from e

    if resp.status != 200:
        raise ValueError(
            f"HTTP {resp.status}: error fetching model instance log options"
        )

    try:
        data = json.loads(body) if body else {}
    except Exception as e:
        raise ValueError(str(e)) from e

    return ServeLogOptionsResponse.model_validate(
        data if isinstance(data, dict) else {}
    )


@router.get("/{id}/log-options", response_model=ModelInstanceLogOptions)
async def get_model_instance_log_options(
    request: Request,
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
):
    """Return per-worker restart_count values that exist on disk for this model instance."""
    model_instance = await fetch_model_instance(session, ctx, id)
    targets = await resolve_instance_log_worker_targets(session, model_instance)

    async def fetch_one(
        target: Tuple[int, str, Optional[Worker]],
    ) -> ModelInstanceLogWorkerOption:
        wid, name, worker = target
        display_name = name
        if worker is None:
            return ModelInstanceLogWorkerOption(
                worker_id=wid,
                name=display_name,
                restarts=[],
                error="Worker not found in database",
            )
        if not display_name:
            display_name = worker.name or ""
        try:
            payload = await fetch_serve_log_options_from_worker(
                request, worker, model_instance.id
            )
            return ModelInstanceLogWorkerOption(
                worker_id=wid,
                name=display_name,
                restarts=payload.restarts,
                error=None,
            )
        except Exception as e:
            return ModelInstanceLogWorkerOption(
                worker_id=wid,
                name=display_name,
                restarts=[],
                error=str(e),
            )

    worker_options = await asyncio.gather(
        *[fetch_one(t) for t in targets],
    )

    for wo in worker_options:
        is_main = wo.worker_id == model_instance.worker_id
        for entry in wo.restarts:
            entry.containers = [
                _map_container_display_name(c, model_instance, is_main)
                for c in entry.containers
            ]

    if worker_options and all(o.error for o in worker_options):
        detail = "; ".join(
            f"{o.worker_id}: {o.error}" for o in worker_options if o.error
        )
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch log options from all workers: {detail}",
        )

    return ModelInstanceLogOptions(
        main_worker_id=model_instance.worker_id,
        workers=list(worker_options),
    )


@router.post("", response_model=ModelInstancePublic)
async def create_model_instance(
    session: SessionDep, model_instance_in: ModelInstanceCreate
):
    # Inherit the parent Model's tenant binding. The schema default of
    # PLATFORM_PRINCIPAL_ID would otherwise persist `owner_principal_id=1`
    # for instances of a non-platform Model whenever the caller (worker /
    # API client) doesn't echo the field back.
    if model_instance_in.model_id is not None:
        parent = await Model.one_by_id(session, model_instance_in.model_id)
        if parent is not None:
            model_instance_in.owner_principal_id = parent.owner_principal_id
    try:
        model_instance = await ModelInstance.create(session, model_instance_in)
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to create model instance: {e}"
        )
    return model_instance


@router.put("/{id}", response_model=ModelInstancePublic)
async def update_model_instance(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    model_instance_in: ModelInstanceUpdate,
):
    model_instance = await ModelInstance.one_by_id(session, id, for_update=True)
    assert_resource_visible(
        ctx,
        model_instance,
        not_found_message="Model instance not found",
    )

    try:
        await ModelInstanceService(session).update(model_instance, model_instance_in)
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to update model instance: {e}"
        )
    return model_instance


@router.delete("/{id}")
async def delete_model_instance(session: SessionDep, ctx: TenantContextDep, id: int):
    model_instance = await ModelInstance.one_by_id(session, id, for_update=True)
    assert_resource_visible(
        ctx,
        model_instance,
        not_found_message="Model instance not found",
    )

    model = (
        await Model.one_by_id(session, model_instance.model_id)
        if model_instance.model_id is not None
        else None
    )

    try:
        # Deleting any node of a DP group tears down the whole group and lets
        # sync_replicas rebuild it intact: a lone member (dp_rank > 0) can't
        # self-schedule — only the rank-0 coordinator fans the group out — and a
        # group that lost its coordinator has nothing to fan out from.
        if model is not None and is_dp_node_per_instance(model):
            siblings = await ModelInstance.all_by_field(session, "model_id", model.id)
            await ModelInstanceService(session).batch_delete(siblings)
        else:
            await ModelInstanceService(session).delete(model_instance)
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to delete model instance: {e}"
        )
