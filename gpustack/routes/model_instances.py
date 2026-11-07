import asyncio
import json
import sys
from contextlib import aclosing, suppress
from typing import AsyncIterator, List, Optional, Tuple
import aiohttp
from fastapi import APIRouter, Request, Response, status, HTTPException
from fastapi.responses import (
    StreamingResponse,
    RedirectResponse,
)
from urllib.parse import urlencode

from gpustack.api.streaming import tenant_streaming
from gpustack.api.responses import StreamingResponseWithStatusCode
from gpustack import envs
from gpustack.server.services import ModelInstanceService
from gpustack.server.controllers import model_members_stale
from gpustack.server.worker_request import request_to_worker, stream_to_worker
from gpustack.utils.command import resolve_executor_backend
from gpustack.worker.logs import LogOptionsDep
from gpustack.api.exceptions import (
    HTTPException as APIHTTPException,
    BadRequestException,
    ConflictException,
    ForbiddenException,
    InternalServerErrorException,
    InvalidException,
    NotFoundException,
)
from gpustack.schemas.workers import Worker
from gpustack.schemas.clusters import Cluster
from gpustack.api.tenant import (
    TenantContext,
    assert_resource_visible,
    tenant_list_conditions,
)
from gpustack.server.db import async_session
from gpustack.server.deps import ListParamsDep, SessionDep, TenantContextDep
from gpustack.server.bus import Event
from gpustack.schemas.models import (
    BackendEnum,
    DistributedServers,
    Model,
    ModelInstance,
    ModelInstanceCreate,
    ModelInstanceLogOptions,
    ModelInstanceLogWorkerOption,
    ModelInstancePublic,
    ModelInstanceMemberReport,
    ModelInstanceStopReport,
    ModelInstanceWorkload,
    ModelInstanceWorkloadStateEnum,
    member_worker_ids,
    ModelInstanceUpdate,
    ModelInstancesPublic,
    ModelInstanceStateEnum,
    ServeLogOptionsResponse,
    get_backend,
    role_effective_model,
)
from gpustack.schemas.model_files import ModelFileStateEnum
from gpustack.schemas.principals import PrincipalType
from gpustack.config.config import get_global_config
from gpustack.utils.grafana import (
    model_dashboard_scope_params,
    resolve_grafana_base_url,
    resolve_server_base_url,
)
from gpustack.utils.export_limits import attachment_headers, sanitize_filename
from gpustack.utils.tabular_export import stream_zip
from gpustack.utils.vllm_topology import (
    vllm_lb_mode,
    managed_lb_parallelism,
    external_lb_fragments,
)
from gpustack.utils.model_instance_workers import (
    get_model_instance_worker_match,
    get_model_instance_workloads,
    workload_topology,
    model_instance_legacy_stop,
    model_instance_metrics_source_inventory,
    requires_group_recovery,
    calculate_group_recovery_policy,
)

router = APIRouter()


async def _strip_private_snapshot_changes(event: Event) -> None:
    event.changed_fields = {
        name: values
        for name, values in event.changed_fields.items()
        if name != "model_config_snapshot"
    }


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
    image_name = model.image_name if model else None
    return (
        resolve_executor_backend(backend_parameters, backend_version, image_name)
        == "ray"
    )


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

    if params.watch:
        return StreamingResponse(
            tenant_streaming(
                ModelInstance,
                ctx,
                fields=fields,
                fuzzy_fields=fuzzy_fields,
                event_transform=_strip_private_snapshot_changes,
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
    check_configuration: bool = False,
):
    model_instance = await ModelInstance.one_by_id(session, id)
    assert_resource_visible(
        ctx,
        model_instance,
        not_found_message="Model instance not found",
    )
    if (
        calculate_group_recovery_policy(model_instance, None) is not None
        and model_instance.distributed_servers.group_recovery is None
    ):
        model_instance = await ModelInstance.one_by_id(
            session, id, for_update=True, refresh=True
        )
        assert_resource_visible(ctx, model_instance)
        await ModelInstanceService(session).ensure_recovery_policy(model_instance)
    if check_configuration:
        model = await Model.one_by_id(session, model_instance.model_id, refresh=True)
        if model is None or await model_members_stale(session, model, [model_instance]):
            raise ConflictException(
                message="The deployment configuration changed before launch"
            )
    return model_instance


def _dashboard_query_target(raw: dict) -> dict:
    target = {}
    for key in ('restart_count', 'dp_rank', 'worker_id', 'component', 'attempt'):
        value = raw.get(key)
        if value is not None:
            if key == 'component':
                if value not in ('sglang_engine', 'sglang_smg', 'vllm_hybrid'):
                    raise BadRequestException(message='Invalid dashboard component')
            else:
                if not value or not value.isascii() or not value.isdigit():
                    raise BadRequestException(message=f'Invalid dashboard {key}')
                try:
                    value = int(value)
                except ValueError as exc:
                    raise BadRequestException(
                        message=f'Invalid dashboard {key}'
                    ) from exc
                if key == 'worker_id' and value == 0:
                    raise BadRequestException(message='Invalid dashboard worker_id')
        target[key] = value
    if (target['attempt'] is not None and target['component'] != 'sglang_smg') or (
        target['component'] in ('sglang_smg', 'vllm_hybrid')
        and target['dp_rank'] is not None
    ):
        raise BadRequestException(message='Invalid dashboard source filters')
    return target


def _dashboard_source_scope(instance: ModelInstance, target: dict) -> dict:
    """Bind monitoring filters to registered listeners, regardless of readiness."""
    unavailable = NotFoundException(message='Dashboard source is unavailable')
    if get_model_instance_workloads(instance) is None:
        if any(value is not None for value in target.values()):
            raise unavailable
        return {}
    epoch = instance.restart_count or 0
    if target['restart_count'] not in (None, epoch):
        raise unavailable
    scope = dict(restart_count=epoch)
    inventory = model_instance_metrics_source_inventory(instance)
    component, rank, worker_id = (
        target['component'],
        target['dp_rank'],
        target['worker_id'],
    )
    candidates = inventory
    if component is not None:
        candidates = [source for source in candidates if source.kind == component]
    if rank is not None:
        candidates = [
            source
            for source in candidates
            if (
                source.kind == 'vllm_engine'
                and source.dp_rank == rank
                or source.kind == 'sglang_engine'
                and source.dp_size is not None
                and 0 <= rank < source.dp_size
            )
        ]
    if worker_id is not None:
        candidates = [source for source in candidates if source.worker_id == worker_id]
    if component is not None or rank is not None:
        if len(candidates) != 1:
            raise unavailable
        source = candidates[0]
        if target['attempt'] not in (None, source.attempt):
            raise unavailable
        scope.update(
            source_kind=source.kind,
            source_name=source.name,
            worker_id=source.worker_id,
            dp_rank=rank,
            component_attempt=source.attempt,
        )
    elif worker_id is not None:
        if not candidates:
            raise unavailable
        scope['worker_id'] = worker_id
    return scope


async def _validate_dashboard_owner(session, instance: ModelInstance, worker_id: int):
    if worker_id not in member_worker_ids(instance):
        raise NotFoundException(message='Dashboard source is unavailable')
    worker = await Worker.one_by_id(session, worker_id, refresh=True)
    if (
        worker is None
        or worker.deleted_at is not None
        or worker.id != worker_id
        or worker.cluster_id != instance.cluster_id
    ):
        raise NotFoundException(message='Dashboard source is unavailable')


@router.get("/{id}/dashboard")
async def get_model_instance_dashboard(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    request: Request,
    restart_count: Optional[str] = None,
    dp_rank: Optional[str] = None,
    worker_id: Optional[str] = None,
    component: Optional[str] = None,
    attempt: Optional[str] = None,
):
    model_instance = await ModelInstance.one_by_id(session, id, refresh=True)
    assert_resource_visible(
        ctx,
        model_instance,
        not_found_message="Model instance not found",
    )

    target = _dashboard_query_target(
        dict(
            restart_count=restart_count,
            dp_rank=dp_rank,
            worker_id=worker_id,
            component=component,
            attempt=attempt,
        )
    )
    scope = _dashboard_source_scope(model_instance, target)
    if scope.get('worker_id') is not None:
        await _validate_dashboard_owner(session, model_instance, scope['worker_id'])

    cfg = get_global_config()
    if not cfg.get_grafana_url() or not cfg.grafana_model_dashboard_uid:
        raise InternalServerErrorException(
            message="Grafana dashboard settings are not configured"
        )

    cluster = None
    if model_instance.cluster_id is not None:
        cluster = await Cluster.one_by_id(session, model_instance.cluster_id)

    query_params = model_dashboard_scope_params(
        cluster_id=model_instance.cluster_id,
        model_id=model_instance.model_id,
        model_instance_id=model_instance.id,
        cluster_name=cluster.name if cluster is not None else None,
        model_name=model_instance.model_name,
        model_instance_name=model_instance.name,
        **scope,
    )
    query_params['var-gpustack_base_url'] = resolve_server_base_url(cfg, request)

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
    source: Optional[str] = None,
):
    # Inline session released after the initial lookups so a long-lived
    # follow-log stream doesn't hold a database connection for its duration.
    async with async_session() as session:
        model_instance = await fetch_model_instance(session, ctx, id)

        if source is not None:
            if worker_id is None or container_name or log_options.previous:
                raise BadRequestException(
                    message="A source requires worker_id and cannot be combined with container or previous"
                )
        # Reverse-map: convert UI display name back to internal container name.
        if container_name:
            is_main = (
                worker_id or model_instance.worker_id
            ) == model_instance.worker_id
            container_name = _unmap_container_display_name(
                container_name, model_instance, is_main
            )

        valid_worker_ids = set(member_worker_ids(model_instance))
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
            offset=log_options.offset,
            limit=log_options.limit,
            source=source,
        )

    if source is not None:
        await _resolve_visible_log_source(request, model_instance, worker, source)

    timeout = aiohttp.ClientTimeout(total=envs.PROXY_TIMEOUT, sock_connect=5)

    def on_exception(e: Exception, t: aiohttp.ClientTimeout) -> tuple[str, int]:
        msg = (
            f"Unable to read logs: {type(e).__name__}"
            if not isinstance(e, TimeoutError)
            else f"Log stream timed out ({t.total} seconds). Please reopen the log page."
        )
        return f"\x1b[999;1H{msg}\n", status.HTTP_500_INTERNAL_SERVER_ERROR

    # Both modes stream. Reading the whole body first would hold an entire
    # serving log in the server process, and these run to gigabytes.
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
    worker_id: Optional[int] = None,
    source: Optional[str] = None,
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
        if worker_id is not None and source is None:
            raise BadRequestException(
                message="worker_id requires a source for downloads"
            )
        if source is not None:
            if worker_id is None:
                raise BadRequestException(message="A source requires worker_id")
            if worker_id not in member_worker_ids(model_instance):
                raise NotFoundException(message="Log source is unavailable")
            worker = await fetch_worker(session, worker_id)
        else:
            targets = await resolve_instance_log_worker_targets(session, model_instance)

    if source is not None:
        _, descriptor = await _resolve_visible_log_source(
            request, model_instance, worker, source
        )
        return await _stream_single_worker_log(
            request,
            model_instance,
            dict(
                worker=worker,
                worker_label=worker.name or str(worker_id),
                container_internal=None,
                container_display=descriptor.label,
                kind=descriptor.kind,
                source=source,
                error=None,
            ),
        )

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
        return await _stream_single_worker_log(request, model_instance, streams[0])
    return _stream_zipped_worker_logs(request, model_instance, streams)


async def _stream_single_worker_log(
    request: Request,
    model_instance: ModelInstance,
    stream: dict,
) -> StreamingResponse:
    """Stream one worker/container's logs straight through, without buffering.

    A stream the worker can measure is served with a Content-Length, which is
    what turns a browser's download into a progress bar instead of a growing
    byte count. The worker measures it on the same look at the files that its
    reads are pinned to, so output arriving meanwhile cannot move it.
    """
    filename = sanitize_filename(
        f"{model_instance.name or model_instance.id}.log", "logs"
    )
    headers = attachment_headers(filename)
    length, chunks = await _worker_log_download(request, model_instance, stream)
    if length is not None:
        headers["Content-Length"] = str(length)
        chunks = _exact_length_chunks(chunks, length)
    return StreamingResponse(
        chunks,
        media_type="text/plain; charset=utf-8",
        headers=headers,
    )


def _stream_zipped_worker_logs(
    request: Request,
    model_instance: ModelInstance,
    streams: List[dict],
) -> StreamingResponse:
    """Stream a flat zip with one member per log; peak memory is one chunk."""
    zip_name = sanitize_filename(
        f"{model_instance.name or model_instance.id}.logs.zip", "logs"
    )
    return StreamingResponse(
        stream_zip(_zip_members(request, model_instance, streams)),
        media_type="application/zip",
        headers=attachment_headers(zip_name),
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
        name = sanitize_filename(f"{label}.log", "logs")
        index = 1
        while name in seen:
            name = sanitize_filename(f"{label}.{index}.log", "logs")
            index += 1
        seen.add(name)
        yield name, _worker_log_chunks(request, model_instance, stream)


async def _worker_log_download(
    request: Request,
    model_instance: ModelInstance,
    stream: dict,
) -> Tuple[Optional[int], AsyncIterator[bytes]]:
    """Start one stream's download, learning its length if the worker states it.

    The whole stream is asked for as one line range. The worker answers with
    its exact length on X-Log-Total-Bytes, measured when it planned the reads
    the body comes from. A length measured any earlier, in a separate call,
    would promise the files as they were then: a main log that grew meanwhile
    pushes the newest container lines past the end. An older worker ignores
    the range and sends the whole log, without a length.

    Args:
        request: The incoming request, for its HTTP clients.
        model_instance: The instance being downloaded.
        stream: One stream descriptor from _plan_log_streams.

    Returns:
        The length, or None when there is none to promise, and the body. With a
        length, a transfer failing midway raises, so the body stops short of
        what was promised; without one, the failure ends the file as a line.
    """
    # Nothing to ask, or a note of our own goes first: no length either way.
    if stream["worker"] is None or stream["error"]:
        return None, _worker_log_chunks(request, model_instance, stream)
    upstream = _stream_worker_log(
        request, model_instance, stream, offset=0, limit=_WHOLE_STREAM_LINES
    )
    try:
        chunk, headers, status_code = await anext(upstream)
    except Exception as e:
        return None, _chunks_of(f"Failed to fetch logs: {e}\n".encode())
    payload = chunk if isinstance(chunk, bytes) else chunk.encode()
    if status_code >= 400:
        failure = f"Failed to fetch logs: HTTP {status_code}: ".encode() + payload
        with suppress(Exception):
            await upstream.aclose()
        return None, _chunks_of(failure)
    declared = {name.lower(): value for name, value in headers.items()}.get(
        "x-log-total-bytes"
    )
    length = (
        int(declared)
        if declared and declared.isascii() and declared.isdigit()
        else None
    )
    note = _empty_log_note(stream)
    if length == 0 and not payload and note:
        with suppress(Exception):
            await upstream.aclose()
        return len(note), _chunks_of(note)
    body = _download_body(payload, upstream, label_failure=length is None)
    if note and length is None:
        body = _label_empty_log(body, note)
    return length, body


def _empty_log_note(stream: dict) -> bytes:
    if stream.get('kind') != 'workload':
        return b''
    return (
        f"No output has been captured for {stream['container_display']}; "
        "the log stream may not be available yet.\n"
    ).encode()


async def _label_empty_log(chunks: AsyncIterator[bytes], note: bytes):
    wrote = False
    async for chunk in chunks:
        wrote = wrote or bool(chunk)
        yield chunk
    if not wrote:
        yield note


async def _download_body(
    first: bytes, upstream: AsyncIterator, label_failure: bool
) -> AsyncIterator[bytes]:
    if first:
        yield first
    try:
        async for chunk, _, _ in upstream:
            yield chunk if isinstance(chunk, bytes) else chunk.encode()
    except Exception as e:
        if not label_failure:
            raise
        yield f"\nFailed to fetch logs: {e}\n".encode()


async def _chunks_of(*chunks: bytes) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


def _stream_worker_log(
    request: Request,
    model_instance: ModelInstance,
    stream: dict,
    offset: Optional[int] = None,
    limit: int = 1000,
):
    """The worker's (chunk, headers, status) stream of one log, for a download."""
    return stream_to_worker(
        worker=stream["worker"],
        method="GET",
        path=f"serveLogs/{model_instance.id}",
        proxy_client=request.app.state.http_client,
        no_proxy_client=request.app.state.http_client_no_proxy,
        params=_build_serve_log_params(
            model_instance,
            container_name=stream["container_internal"],
            source=stream.get("source"),
            offset=offset,
            limit=limit,
        ),
        timeout=aiohttp.ClientTimeout(
            total=None, sock_connect=5, sock_read=_LOG_DOWNLOAD_IDLE_TIMEOUT
        ),
        raw=True,
    )


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
    if stream.get('source') and stream['error']:
        yield f"Failed to fetch logs: {stream['error']}\n".encode()
        return

    # Discovery failed but the fallback still reads the main workload. Say so,
    # or a sidecar log missing from the archive looks like it never existed.
    if stream["error"]:
        yield (
            f"Note: container discovery failed ({stream['error']}); "
            "this log covers the main workload only.\n"
        ).encode()

    wrote = False
    try:
        async with aclosing(
            _stream_worker_log(request, model_instance, stream)
        ) as upstream:
            async for chunk, _, status_code in upstream:
                payload = chunk if isinstance(chunk, bytes) else chunk.encode()
                # Label an error body so it can't be read as real log output.
                if status_code >= 400:
                    with suppress(Exception):
                        await upstream.aclose()
                    yield f"Failed to fetch logs: HTTP {status_code}: ".encode() + payload
                    return
                wrote = wrote or bool(payload)
                yield payload
    except Exception as e:
        yield f"\nFailed to fetch logs: {e}\n".encode()
        return
    if not wrote:
        note = _empty_log_note(stream)
        if note:
            yield note


def _build_serve_log_params(
    model_instance: ModelInstance,
    tail: int = -1,
    follow: bool = False,
    previous: bool = False,
    container_name: Optional[str] = None,
    offset: Optional[int] = None,
    limit: int = 1000,
    source: Optional[str] = None,
) -> dict:
    """Proxy params for one container's logs; defaults to a full, non-follow read."""
    params = {
        "tail": tail,
        "follow": follow,
        "model_instance_name": model_instance.name,
        "previous": previous,
        "current_restart_count": model_instance.restart_count or 0,
    }
    if source is not None:
        params["source"] = source
    if container_name:
        params["container_name"] = container_name
    if offset is not None:
        params["offset"] = offset
        params["limit"] = limit
    model_file_id = _downloading_model_file_id(model_instance)
    if model_file_id is not None:
        params["model_file_id"] = model_file_id
    return params


# A download has no deadline: a log the size cap does not bound legitimately
# takes longer than any total timeout worth setting. What replaces it is a
# liveness check -- this long without a byte and the transfer is dead, not slow.
_LOG_DOWNLOAD_IDLE_TIMEOUT = 120

_DOWNLOAD_SHORTFALL_NOTE = b"\n... the log changed while it was downloading ...\n"

# Past this much missing, the transfer failed rather than the log moved, and
# padding out the rest would hand the client a whole-looking file of filler.
_MAX_SHORTFALL_PADDING = 64 * 1024

# The line limit that asks for a whole stream: a range has no other way to say
# it runs to the end.
_WHOLE_STREAM_LINES = sys.maxsize


def _downloading_model_file_id(model_instance: ModelInstance) -> Optional[int]:
    """The model file whose download log belongs in this instance's logs.

    Args:
        model_instance: The instance being read.

    Returns:
        The model file id, or None once the weights are in place and the
        download log has nothing left to say.
    """
    if (
        model_instance.state != ModelInstanceStateEnum.RUNNING
        and model_instance.model_files
        and model_instance.model_files[0].state != ModelFileStateEnum.READY
    ):
        return model_instance.model_files[0].id
    return None


async def _exact_length_chunks(chunks: AsyncIterator[bytes], expected: int):
    """Hold the body to the length Content-Length promised, within reason.

    The reads are pinned to the length, so the body can only fall short of it:
    a file the reads had not reached yet was cut or rewritten under them. That
    is a log that changed, not a broken connection, which is what a client
    reads a short body as. A shortfall too large to be movement is a transfer
    that failed, and there the broken connection is the honest answer -- as it
    is for a transfer that raises.

    Args:
        chunks: The stream's bytes.
        expected: The length already promised to the client.

    Yields:
        Exactly `expected` bytes, unless the stream fell so far short that the
        body is better left incomplete.
    """
    written = 0
    async for chunk in chunks:
        if written + len(chunk) > expected:
            chunk = chunk[: expected - written]
        if chunk:
            yield chunk
            written += len(chunk)
        if written >= expected:
            return
    shortfall = expected - written
    if 0 < shortfall <= _MAX_SHORTFALL_PADDING:
        note = _DOWNLOAD_SHORTFALL_NOTE[:shortfall]
        yield note + b" " * (shortfall - len(note))


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
        if worker.cluster_id != model_instance.cluster_id:
            return [
                dict(
                    worker_label=label,
                    worker=None,
                    container_internal=None,
                    container_display="Logs",
                    error="Log Worker does not belong to this cluster",
                )
            ]
        is_main = worker_id == model_instance.worker_id
        try:
            payload = await fetch_serve_log_options_from_worker(
                request,
                worker,
                model_instance.id,
                current_restart_count=model_instance.restart_count or 0,
            )
            current = next(
                (entry for entry in payload.restarts if not entry.previous), None
            )
            sources = (
                [
                    s
                    for s in current.sources
                    if s.kind in ("preparation", "workload")
                    and _log_source_visible(model_instance, worker_id, s)
                ]
                if current
                else []
            )
            if sources:
                return [
                    dict(
                        worker_label=label,
                        worker=worker,
                        container_internal=None,
                        container_display=s.label,
                        source=s.id,
                        kind=s.kind,
                        error=None if s.available else "Log source is unavailable",
                    )
                    for s in sources
                ] + _missing_workload_log_streams(
                    model_instance, worker_id, label, sources
                )
            if get_model_instance_workloads(model_instance) is not None:
                return [
                    dict(
                        worker_label=label,
                        worker=None,
                        container_internal=None,
                        container_display="Logs",
                        error="No trusted log sources are available",
                    )
                ]
            containers, error = (list(current.containers) if current else []) or [
                "default"
            ], None
        except Exception as exc:
            containers, error = ["default"], str(exc)
            if get_model_instance_workloads(model_instance) is not None:
                return [
                    dict(
                        worker_label=label,
                        worker=None,
                        container_internal=None,
                        container_display="Logs",
                        error=str(exc),
                    )
                ]
        return [
            {
                "worker_label": label,
                "worker": worker,
                "container_internal": container_internal,
                "kind": "workload",
                "container_display": _map_container_display_name(
                    container_internal, model_instance, is_main
                ),
                "error": error,
            }
            for container_internal in containers
        ]

    grouped = await asyncio.gather(*[per_worker(t) for t in targets])
    return [stream for group in grouped for stream in group]


def _missing_workload_log_streams(instance, worker_id, label, sources):
    """Keep an expected Engine's absence visible beside preparation output."""
    tasks = get_model_instance_workloads(instance, worker_id)
    workloads = [s for s in sources if s.kind == 'workload']
    if tasks is None:
        missing = (
            []
            if any(
                s.component_kind != 'sglang_smg'
                and s.container_name in (None, 'default')
                for s in workloads
            )
            else [('Engine', 'Engine log source is unavailable')]
        )
    else:
        missing = []
        for task in tasks:
            if any(
                s.workload_name == task.name
                or (
                    task.component
                    and task.component.kind == 'sglang_smg'
                    and s.component_kind == 'sglang_smg'
                )
                for s in workloads
            ):
                continue
            kind = (
                'SMG'
                if task.component and task.component.kind == 'sglang_smg'
                else 'Engine'
            )
            missing.append(
                (
                    f'{kind} · {task.name}',
                    f'{kind} log source is unavailable for {task.name}',
                )
            )
    return [
        dict(
            worker_label=label,
            worker=None,
            container_internal=None,
            container_display=name,
            error=error,
        )
        for name, error in missing
    ]


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
    current_restart_count: Optional[int] = None,
) -> ServeLogOptionsResponse:
    timeout = aiohttp.ClientTimeout(total=envs.PROXY_TIMEOUT, sock_connect=5)
    try:
        resp, body = await request_to_worker(
            worker=worker,
            method="GET",
            path=f"serveLogOptions/{model_instance_id}",
            params=(
                {"current_restart_count": current_restart_count}
                if current_restart_count is not None
                else None
            ),
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
    ctx: TenantContextDep,
    id: int,
):
    """Return per-worker restart_count values that exist on disk for this model instance."""
    async with async_session() as session:
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
        if worker.cluster_id != model_instance.cluster_id:
            return ModelInstanceLogWorkerOption(
                worker_id=wid,
                name=display_name,
                restarts=[],
                error="Log Worker does not belong to this cluster",
            )
        if not display_name:
            display_name = worker.name or ""
        try:
            payload = await fetch_serve_log_options_from_worker(
                request,
                worker,
                model_instance.id,
                current_restart_count=model_instance.restart_count or 0,
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
            entry.sources = [
                source
                for source in entry.sources
                if _log_source_visible(model_instance, wo.worker_id, source)
            ]
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
    session: SessionDep,
    ctx: TenantContextDep,
    model_instance_in: ModelInstanceCreate,
):
    if model_instance_in.model_id is None:
        raise NotFoundException(message="Model not found")
    parent = await Model.one_by_id(session, model_instance_in.model_id)
    assert_resource_visible(ctx, parent, not_found_message="Model not found")
    # Instances inherit the authorized model's tenant and cluster scope.
    # Explicit bindings must agree with the parent, including a NULL cluster.
    for field in ("owner_principal_id", "cluster_id"):
        value = getattr(parent, field)
        if (
            field in model_instance_in.model_fields_set
            and getattr(model_instance_in, field) != value
        ):
            raise InvalidException(message=f"{field} must match the parent model")
        setattr(model_instance_in, field, value)
    if (
        get_model_instance_workloads(model_instance_in) is not None
        or model_instance_legacy_stop(model_instance_in) is not None
        or (
            model_instance_in.distributed_servers is not None
            and (
                model_instance_in.distributed_servers.stop_requested
                or model_instance_in.distributed_servers.group_recovery is not None
            )
        )
    ):
        raise BadRequestException(
            message="Managed workloads are initialized by the Server"
        )
    try:
        model_instance = await ModelInstance.create(session, model_instance_in)
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to create model instance: {e}"
        )
    return model_instance


def _validate_local_devices(
    instance: ModelInstance, worker_id: int, workloads: List[ModelInstanceWorkload]
) -> None:
    if workloads and workloads[0].component is not None:
        if workloads[0].component.kind == "vllm_hybrid":
            if len(workloads) != 1:
                raise BadRequestException(
                    message="A Worker must report exactly one Hybrid component"
                )
            return
        engines = [w for w in workloads if w.component.kind == "sglang_engine"]
        smgs = [w for w in workloads if w.component.kind == "sglang_smg"]
        primary = worker_id == instance.worker_id
        if (
            len(engines) != 1
            or len(smgs) != int(primary)
            or (engines[0].component.api_port is not None) != primary
        ):
            raise BadRequestException(
                message="Invalid local SGLang component allocation"
            )
        return
    match = get_model_instance_worker_match(instance, worker_id=worker_id)
    allocated = set(instance.gpu_indexes or []) if match.is_main_worker else set()
    for index in match.subordinate_worker_indexes:
        allocated.update(
            instance.distributed_servers.subordinate_workers[index].gpu_indexes or []
        )
    reported = {
        index
        for item in workloads
        for placement in item.dp_rank_placements
        for index in placement.gpu_indexes
    }
    if reported != allocated:
        raise BadRequestException(
            message="Rank devices must match the Worker's allocation"
        )


def _validate_complete_manifest(
    instance: ModelInstance, workloads: List[ModelInstanceWorkload]
) -> None:
    # Partial registration is expected: a headless Worker may report before
    # the Worker hosting its main API. Check completeness only once all report.
    if {item.worker_id for item in workloads} != set(member_worker_ids(instance)):
        return
    if workloads and workloads[0].component is not None:
        for worker_id in member_worker_ids(instance):
            _validate_local_devices(
                instance, worker_id, [w for w in workloads if w.worker_id == worker_id]
            )
        return
    nodes = {}
    for item in workloads:
        for placement in item.dp_rank_placements:
            nodes.setdefault(placement.dp_rank, set()).add(placement.node_rank)
    if set(nodes) != set(range(len(nodes))) or any(
        ranks != set(range(len(ranks))) for ranks in nodes.values()
    ):
        raise BadRequestException(
            message="Complete rank placements must be contiguous and include each main"
        )


def _merge_component_observation(instance, item, old) -> None:
    """Worker observations cannot replace the Server's component recovery intent."""
    component = item.component
    if component is None:
        return
    if old is None:
        if (
            component.attempt != 0
            or component.stop_requested
            or component.last_restart_time is not None
        ):
            raise ConflictException(
                message="Only the Server may initialize SMG recovery"
            )
    else:
        component.stop_requested = old.component.stop_requested
        component.last_restart_time = old.component.last_restart_time
    if component.api_port is not None and (
        instance.state == ModelInstanceStateEnum.ERROR
        or instance.draining_since is not None
    ):
        component.ready = False


def _apply_workload_observations(local, observations):
    observed = {item.name: item for item in observations}
    if len(observed) != len(observations) or set(observed) != {
        item.name for item in local
    }:
        raise ConflictException(
            message="Observations must match registered local workloads"
        )
    workloads = []
    for old in local:
        observation = observed[old.name]
        item = old.model_copy(deep=True)
        item.state = observation.state
        if item.component is not None:
            if observation.component_attempt != item.component.attempt:
                raise ConflictException(
                    message="Observation does not match the current component attempt"
                )
            if (
                observation.rank_readiness
                or (
                    item.component.api_port is None
                    and observation.component_ready is not None
                )
                or (
                    item.component.api_port is not None
                    and observation.component_ready is None
                )
            ):
                raise BadRequestException(
                    message="Invalid component readiness observation"
                )
            item.component.ready = observation.component_ready
        else:
            mains = {p.dp_rank for p in item.dp_rank_placements if p.node_rank == 0}
            if set(observation.rank_readiness) != mains or (
                observation.component_ready is not None
                or observation.component_attempt is not None
            ):
                raise BadRequestException(message="Invalid rank readiness observation")
            for placement in item.dp_rank_placements:
                if placement.node_rank == 0:
                    placement.ready = observation.rank_readiness[placement.dp_rank]
        workloads.append(item)
    try:
        return [
            ModelInstanceWorkload.model_validate(item.model_dump())
            for item in workloads
        ]
    except ValueError as exc:
        raise BadRequestException(message=str(exc)) from exc


def _validate_reported_workloads(worker_id, local, incoming, registration):
    if any(item.worker_id != worker_id for item in incoming):
        raise ForbiddenException(message="A Worker may only report its own workloads")
    if local and workload_topology(local) != workload_topology(incoming):
        raise ConflictException(
            message="Registered workload topology is immutable within a restart_count"
        )
    if not registration or not local:
        return
    exclude = {'component': {'stop_requested', 'last_restart_time'}}
    if [item.model_dump(exclude=exclude) for item in local] != [
        item.model_dump(exclude=exclude) for item in incoming
    ]:
        raise ConflictException(message="Use observations for registered workloads")


def _merge_workload_report(
    instance: ModelInstance, ctx: TenantContext, report: ModelInstanceMemberReport
) -> DistributedServers:
    """Validate and merge a local report while the caller holds the instance row lock."""
    worker = ctx.user.worker
    if ctx.user.kind != PrincipalType.SYSTEM or worker is None:
        raise ForbiddenException(
            message="Workload reports require a bound Worker principal"
        )
    if worker.cluster_id != instance.cluster_id or worker.id not in member_worker_ids(
        instance
    ):
        raise ForbiddenException(
            message="Worker is not assigned to this instance and cluster"
        )
    stored = get_model_instance_workloads(instance)
    if stored is None:
        raise ConflictException(
            message="Legacy instances do not accept workload reports"
        )
    if report.restart_count != (instance.restart_count or 0):
        raise ConflictException(
            message="Workload report does not match the current restart_count"
        )
    local = get_model_instance_workloads(instance, worker.id)
    if report.workloads is not None:
        workloads = [
            ModelInstanceWorkload.model_validate(item.model_dump())
            for item in report.workloads
        ]
    elif report.observations is not None:
        workloads = _apply_workload_observations(local, report.observations)
    else:
        return instance.distributed_servers
    _validate_reported_workloads(
        worker.id, local, workloads, report.workloads is not None
    )
    terminal = {
        ModelInstanceWorkloadStateEnum.STOPPED,
        ModelInstanceWorkloadStateEnum.FAILED,
    }
    previous = {item.name: item for item in local}
    for item in workloads:
        old = previous.get(item.name)
        _merge_component_observation(instance, item, old)
        if (
            old
            and old.state in terminal
            and item.state not in (old.state, ModelInstanceWorkloadStateEnum.STOPPED)
        ):
            raise ConflictException(
                message="A terminal workload cannot restart within the same generation"
            )
        if (
            instance.state == ModelInstanceStateEnum.ERROR
            or instance.draining_since is not None
        ):
            for placement in item.dp_rank_placements:
                if placement.node_rank == 0:
                    placement.ready = False
    if local or workloads:
        _validate_local_devices(instance, worker.id, workloads)
    incoming = {item.name: item for item in workloads}
    merged = [
        incoming[item.name] if item.worker_id == worker.id else item for item in stored
    ]
    merged.extend(item for item in workloads if item.name not in previous)
    try:
        servers = DistributedServers.model_validate(
            {**instance.distributed_servers.model_dump(), "workloads": merged}
        )
    except ValueError as exc:
        raise BadRequestException(message=str(exc)) from exc
    _validate_complete_manifest(instance, servers.workloads)
    return servers


def _managed_preparation_fields(instance, worker_id, preparation) -> dict:
    """Publish one Worker's preparation without replacing its allocation or tasks."""
    if (
        get_model_instance_workloads(instance, worker_id)
        or instance.draining_since is not None
        or instance.state
        not in (
            ModelInstanceStateEnum.SCHEDULED,
            ModelInstanceStateEnum.INITIALIZING,
            ModelInstanceStateEnum.DOWNLOADING,
            ModelInstanceStateEnum.STARTING,
        )
    ):
        raise ConflictException(
            message="Instance is no longer preparing this generation"
        )
    ports = {
        key: getattr(preparation, key)
        for key in ("port", "ports", "named_ports")
        if getattr(preparation, key) is not None
    }
    if worker_id != instance.worker_id:
        if not instance.ports:
            raise ConflictException(message="Wait for the primary port allocation")
        if ports:
            raise ForbiddenException(
                message="Only the primary may publish shared ports"
            )
        servers = instance.distributed_servers.model_copy(deep=True)
        subordinate = next(
            w for w in servers.subordinate_workers if w.worker_id == worker_id
        )
        subordinate.pid = preparation.pid
        subordinate.ports = list(instance.ports or [])
        subordinate.state = ModelInstanceStateEnum.INITIALIZING
        return {"distributed_servers": servers}
    if ports:
        if instance.ports and any(
            getattr(instance, key) != value for key, value in ports.items()
        ):
            raise ConflictException(message="Published preparation ports are immutable")
        if not instance.ports and (
            not preparation.ports or preparation.port not in preparation.ports
        ):
            raise BadRequestException(
                message="Preparation must publish a complete port allocation"
            )
    elif not instance.ports:
        raise BadRequestException(
            message="Preparation requires a published port allocation"
        )
    source = dict(ports)
    if preparation.pid is not None:
        source["pid"] = preparation.pid
    if instance.state == ModelInstanceStateEnum.SCHEDULED:
        source["state"] = ModelInstanceStateEnum.INITIALIZING
    return source


def _validate_sglang_component_layout(instance, model, workloads) -> None:
    """Match component APIs and logical DP to the selected ordinary Engine."""
    from gpustack.utils.model_instance_workers import (
        sglang_smg_mode,
        sglang_parallelism,
    )

    try:
        if instance.role is not None or not sglang_smg_mode(model):
            raise ValueError("Components require ordinary multi-rank SGLang DP/DPA")
        expected_dp, _, _ = sglang_parallelism(model.backend_parameters)
    except ValueError as exc:
        raise BadRequestException(message=str(exc)) from exc
    for workload in workloads:
        component = workload.component
        if component is None:
            raise BadRequestException(message="Mixed component/rank report")
        if component.kind == "sglang_engine" and component.api_port is not None:
            if component.api_port != instance.port or component.dp_size != expected_dp:
                raise BadRequestException(
                    message="Engine API/DP size differs from allocation"
                )
        if any(
            port is not None and port not in (instance.ports or [])
            for port in (component.api_port, component.metrics_port)
        ):
            raise BadRequestException(message="Component port was not reserved")


def _validate_hybrid_component_layout(instance, model, workloads) -> None:
    """Match each shared API to its allocated rank owner and local DP count."""
    try:
        if (
            get_backend(model) != BackendEnum.VLLM
            or vllm_lb_mode(model.backend_parameters) != "hybrid"
        ):
            raise ValueError("Hybrid components require vLLM Hybrid-LB")
    except ValueError as exc:
        raise BadRequestException(message=str(exc)) from exc
    api = (instance.named_ports or {}).get("_vllm_external_api")
    if api is None or api.count != 1 or api.base not in (instance.ports or []):
        raise BadRequestException(message="Hybrid API port was not reserved")
    for workload in workloads:
        component = workload.component
        if component is None or component.kind != "vllm_hybrid":
            raise BadRequestException(message="Mixed Hybrid component report")
        members = [instance] + list(
            instance.distributed_servers.subordinate_workers or []
        )
        try:
            index = [member.worker_id for member in members].index(workload.worker_id)
            counts = [len(member.gpu_indexes or []) for member in members]
            topology = managed_lb_parallelism(model.backend_parameters, counts)
            fragments = external_lb_fragments(counts, topology, index)
        except ValueError as exc:
            raise BadRequestException(message=str(exc)) from exc
        owns_api = fragments[0].node_rank == 0
        expected = (api.base, len(fragments)) if owns_api else (None, None)
        if (component.api_port, component.dp_size) != expected:
            raise BadRequestException(
                message="Hybrid API/local DP differs from allocation"
            )


async def _validate_workload_layout(session, instance, workloads) -> None:
    """Validate a first registration against the allocated, deterministic layout."""
    if not workloads or get_model_instance_workloads(instance, workloads[0].worker_id):
        return
    model = await Model.one_by_id(session, instance.model_id, refresh=True)
    if model is None:
        raise BadRequestException(message="Model configuration is unavailable")
    if await model_members_stale(session, model, [instance]):
        raise ConflictException(
            message="The deployment configuration changed before registration"
        )
    model = role_effective_model(model, instance.role)
    if (
        get_backend(model) == BackendEnum.VLLM
        and vllm_lb_mode(model.backend_parameters) == "hybrid"
    ):
        _validate_hybrid_component_layout(instance, model, workloads)
        return
    if any(w.component is not None for w in workloads):
        if workloads[0].component.kind == "vllm_hybrid":
            _validate_hybrid_component_layout(instance, model, workloads)
        else:
            _validate_sglang_component_layout(instance, model, workloads)
        return
    _validate_external_rank_layout(instance, model, workloads)


def _validate_external_rank_layout(instance, model, workloads) -> None:
    """Match rank devices and APIs to the allocated External-LB layout."""
    if get_backend(model) != BackendEnum.VLLM or vllm_lb_mode(
        model.backend_parameters
    ) not in ("external", "multi_port_external"):
        return
    members = [instance] + list(instance.distributed_servers.subordinate_workers or [])
    counts = [len(w.gpu_indexes or []) for w in members]
    worker_id = workloads[0].worker_id
    index = [w.worker_id for w in members].index(worker_id)
    try:
        topology = managed_lb_parallelism(model.backend_parameters, counts)
        fragments = external_lb_fragments(counts, topology, index)
    except ValueError as exc:
        raise BadRequestException(message=str(exc)) from exc
    devices = sorted(members[index].gpu_indexes)
    api = (instance.named_ports or {}).get("_vllm_external_api")
    expected = {
        (f.dp_rank, f.node_rank): (
            [devices[i] for i in f.positions],
            api.base + f.dp_rank if api and f.node_rank == 0 else None,
        )
        for f in fragments
    }
    actual = {
        (p.dp_rank, p.node_rank): (p.gpu_indexes, p.api_port)
        for w in workloads
        for p in w.dp_rank_placements
    }
    if api is None or api.count != topology.dp or actual != expected:
        raise BadRequestException(
            message="Rank placements do not match the allocated External-LB layout"
        )


def _merge_legacy_stop_report(instance, ctx, report, current_worker):
    guard = model_instance_legacy_stop(instance)
    worker = ctx.user.worker
    if (
        ctx.user.kind != PrincipalType.SYSTEM
        or worker is None
        or current_worker is None
        or current_worker.deleted_at is not None
        or current_worker.system_principal_id != ctx.user.id
        or current_worker.cluster_id != instance.cluster_id
        or current_worker.state not in ("ready", "maintenance")
        or current_worker.unreachable
        or guard is None
        or worker.id not in guard.worker_ids
        or guard.worker_principal_ids.get(worker.id) != ctx.user.id
    ):
        raise ForbiddenException(
            message="Stop confirmation requires the frozen bound Worker identity"
        )
    if (
        report.token != guard.token
        or report.restart_count != guard.restart_count
        or (instance.restart_count or 0) != guard.restart_count
        or set(member_worker_ids(instance)) != set(guard.worker_ids)
        or (
            instance.draining_since is None
            and instance.state != ModelInstanceStateEnum.ERROR
        )
    ):
        raise ConflictException(
            message="Stop confirmation does not match the current stop barrier"
        )
    servers = instance.distributed_servers.model_copy(deep=True)
    servers.legacy_stop.confirmed_worker_ids = sorted(
        set(guard.confirmed_worker_ids) | {worker.id}
    )
    return servers


def _preserve_managed_fields(
    instance: ModelInstance, update: ModelInstanceUpdate, *, legacy: bool = False
) -> None:
    """Keep ordinary instance snapshots from changing managed runtime ownership."""
    stored = get_model_instance_workloads(instance)
    incoming = get_model_instance_workloads(update)
    guard = model_instance_legacy_stop(instance)
    if (
        update.distributed_servers is None
        and instance.distributed_servers is not None
        and instance.distributed_servers.group_recovery is not None
    ):
        update.distributed_servers = instance.distributed_servers.model_copy(deep=True)
    if update.distributed_servers is not None:
        policy = update.distributed_servers.group_recovery
        stored_policy = (
            instance.distributed_servers.group_recovery
            if instance.distributed_servers is not None
            else None
        )
        if policy is not None and policy != stored_policy:
            raise ConflictException(message='Recovery policy is server-owned')
        update.distributed_servers.group_recovery = stored_policy
    if (
        update.distributed_servers is not None
        and update.distributed_servers.stop_requested
        and not (
            instance.distributed_servers is not None
            and instance.distributed_servers.stop_requested
        )
    ):
        raise ConflictException(message="Stop requests are server-owned")
    if (
        model_instance_legacy_stop(update) is not None
        and model_instance_legacy_stop(update) != guard
    ):
        raise ConflictException(message="Stop barriers are server-owned")
    if incoming is not None and incoming != stored:
        raise ConflictException(
            message="Workloads can only be changed through workload_report"
        )
    if legacy:
        for field in (
            'restart_count',
            'last_restart_time',
            'state',
            'state_message',
            'pid',
            'port',
            'ports',
            'named_ports',
            'draining_since',
            'gpu_indexes',
            'gpu_addresses',
            'computed_resource_claim',
        ):
            setattr(update, field, getattr(instance, field))
        update.distributed_servers = instance.distributed_servers
        update.model_fields_set.discard('model_config_snapshot')
        update.model_config_snapshot = None
        return
    if stored is None and guard is None:
        return
    # A legacy snapshot may omit the manifest. Preserve it even if the snapshot
    # carries distributed_servers=null or an old subordinate status list.
    update.distributed_servers = instance.distributed_servers
    for field in update.model_fields_set - {"distributed_servers"}:
        if getattr(update, field) != getattr(instance, field):
            raise ConflictException(
                message=f"Managed instance field {field} cannot be changed by a snapshot"
            )


def _preserve_download_state(
    instance: ModelInstance, update: ModelInstanceUpdate
) -> None:
    """Preserve controller-owned download results across worker state reports."""
    for field in (
        "download_progress",
        "draft_model_download_progress",
        "resolved_path",
        "draft_model_resolved_path",
        "mounted_loras",
    ):
        setattr(update, field, getattr(instance, field))
    if instance.state in (
        ModelInstanceStateEnum.STARTING,
        ModelInstanceStateEnum.RUNNING,
        ModelInstanceStateEnum.ERROR,
    ) and update.state in (
        ModelInstanceStateEnum.INITIALIZING,
        ModelInstanceStateEnum.DOWNLOADING,
    ):
        # A download snapshot must not undo completion or erase a failure.
        # Runtime updates and explicit restarts (SCHEDULED) remain valid transitions.
        update.state = instance.state
        update.state_message = instance.state_message
    if not update.distributed_servers:
        return
    stored_progress = {
        worker.worker_id: worker.download_progress
        for worker in (
            instance.distributed_servers.subordinate_workers or []
            if instance.distributed_servers
            else []
        )
    }
    for worker in update.distributed_servers.subordinate_workers or []:
        worker.download_progress = stored_progress.get(worker.worker_id)


def _preserve_model_config_snapshot(instance, update, source, ctx):
    """Keep absent startup inputs and fence managed snapshots to the main Worker."""
    snapshot = getattr(update, "model_config_snapshot", None)
    if snapshot is None:
        update.model_fields_set.discard("model_config_snapshot")
        return
    managed = get_model_instance_workloads(instance) is not None or isinstance(
        update, ModelInstanceMemberReport
    )
    worker_id = getattr(getattr(ctx.user, "worker", None), "id", None)
    if getattr(ctx.user, "kind", None) != PrincipalType.SYSTEM or (
        managed and worker_id != instance.worker_id
    ):
        raise ForbiddenException(
            message=(
                "Only the main system Worker can update managed configuration snapshots"
                if managed
                else "Only system principals can update model configuration snapshots"
            )
        )
    if (
        instance.distributed_servers is not None
        and instance.distributed_servers.group_recovery is not None
        and calculate_group_recovery_policy(instance, snapshot)
        != instance.distributed_servers.group_recovery
    ):
        raise ConflictException(
            message='Launch configuration differs from recovery policy'
        )
    if isinstance(source, dict):
        source["model_config_snapshot"] = snapshot


async def _locked_reporting_worker(session, ctx):
    worker = ctx.user.worker
    if ctx.user.kind != PrincipalType.SYSTEM or worker is None:
        return None
    return await Worker.one_by_id(session, worker.id, for_update=True, refresh=True)


async def _persist_instance_update(session, instance, source):
    """Suppress duplicate reports after authorization and generation checks."""
    if isinstance(source, dict) and all(
        getattr(instance, key) == value for key, value in source.items()
    ):
        return
    await ModelInstanceService(session).update(instance, source)


@router.put("/{id}", response_model=ModelInstancePublic)
async def update_model_instance(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    model_instance_in: ModelInstanceUpdate,
):
    return await _update_model_instance(session, ctx, id, model_instance_in)


@router.put("/{id}/workload-report", response_model=ModelInstancePublic)
async def report_model_instance_workloads(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    model_instance_in: ModelInstanceMemberReport,
):
    return await _update_model_instance(session, ctx, id, model_instance_in)


@router.post("/{id}/stop-confirmation", response_model=ModelInstancePublic)
async def confirm_model_instance_stop(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    model_instance_in: ModelInstanceStopReport,
):
    return await _update_model_instance(session, ctx, id, model_instance_in)


async def _update_model_instance(session, ctx, id, model_instance_in):
    member_report = isinstance(model_instance_in, ModelInstanceMemberReport)
    stop_report = isinstance(model_instance_in, ModelInstanceStopReport)
    worker = ctx.user.worker if member_report or stop_report else None
    current_worker = (
        await _locked_reporting_worker(session, ctx)
        if member_report or stop_report
        else None
    )
    if member_report or stop_report:
        model_instance = await ModelInstance.one_by_id(
            session, id, for_update=True, refresh=True
        )
    else:
        model_instance = await ModelInstance.one_by_id(session, id, for_update=True)
    assert_resource_visible(
        ctx,
        model_instance,
        not_found_message="Model instance not found",
    )

    if stop_report:
        source = {
            "distributed_servers": _merge_legacy_stop_report(
                model_instance, ctx, model_instance_in, current_worker
            )
        }
    elif member_report:
        # Authentication can precede a concurrent Worker deletion or rebind.
        # The instance lock serializes registration with the deletion guard.
        if (
            ctx.user.kind == PrincipalType.SYSTEM
            and worker is not None
            and (
                current_worker is None
                or current_worker.deleted_at is not None
                or current_worker.system_principal_id != ctx.user.id
                or current_worker.cluster_id != model_instance.cluster_id
            )
        ):
            raise ForbiddenException(
                message="Worker identity is no longer bound to this instance and cluster"
            )
        if get_model_instance_workloads(model_instance) is None:
            source = await _legacy_runtime_fields(
                session, model_instance, ctx, model_instance_in
            )
        else:
            source = await _managed_report_fields(
                session, model_instance, ctx, model_instance_in
            )
    else:
        _preserve_download_state(model_instance, model_instance_in)
        legacy = False
        if requires_group_recovery(model_instance):
            model = await Model.one_by_id(
                session, model_instance.model_id, refresh=True
            )
            legacy = model is not None and requires_group_recovery(
                model_instance, model
            )
        _preserve_managed_fields(model_instance, model_instance_in, legacy=legacy)
        source = model_instance_in
    _preserve_model_config_snapshot(model_instance, model_instance_in, source, ctx)
    try:
        await _persist_instance_update(session, model_instance, source)
    except APIHTTPException:
        raise
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to update model instance: {e}"
        )
    return model_instance


async def _legacy_runtime_fields(session, instance, ctx, report):
    worker = ctx.user.worker
    if (
        ctx.user.kind != PrincipalType.SYSTEM
        or worker is None
        or worker.id not in member_worker_ids(instance)
        or worker.cluster_id != instance.cluster_id
    ):
        raise ForbiddenException(
            message='Runtime reports require an assigned bound Worker'
        )
    model = await Model.one_by_id(session, instance.model_id, refresh=True)
    if model is None or not requires_group_recovery(instance, model):
        raise ConflictException(message='Instance does not use legacy group recovery')
    if (
        report.restart_count != (instance.restart_count or 0)
        or instance.draining_since is not None
        or model_instance_legacy_stop(instance) is not None
    ):
        raise ConflictException(
            message='Runtime report does not match the active round'
        )
    if report.workloads is not None or report.observations is not None:
        raise BadRequestException(message='Instance does not use managed workloads')
    main = worker.id == instance.worker_id
    source = report.model_dump(
        exclude_unset=True,
        exclude={
            'restart_count',
            'model_config_snapshot',
            'workloads',
            'observations',
        },
    )
    source.update(source.pop('preparation', None) or {})
    if source.get('state') is None:
        source.pop('state', None)
    if main and 'arguments' in source:
        raise BadRequestException(message='Arguments belong to subordinate preparation')
    if not main and any(
        key in source for key in ('port', 'named_ports', 'injected_backend_parameters')
    ):
        raise ForbiddenException(
            message='Only the primary may publish shared startup fields'
        )
    if 'backend_version' in source:
        if instance.backend_version not in (None, source['backend_version']):
            raise ConflictException(message='Backend version differs within this round')
    if main:
        member = instance
    else:
        servers = instance.distributed_servers.model_copy(deep=True)
        member = next(
            w for w in servers.subordinate_workers if w.worker_id == worker.id
        )
    if (
        instance.state == ModelInstanceStateEnum.ERROR
        or member.state == ModelInstanceStateEnum.ERROR
    ) and report.state != ModelInstanceStateEnum.ERROR:
        raise ConflictException(
            message='An unhealthy round cannot be resumed by a runtime report'
        )
    order = {
        state: index
        for index, state in enumerate(
            (
                ModelInstanceStateEnum.PENDING,
                ModelInstanceStateEnum.SCHEDULED,
                ModelInstanceStateEnum.INITIALIZING,
                ModelInstanceStateEnum.DOWNLOADING,
                ModelInstanceStateEnum.STARTING,
                ModelInstanceStateEnum.RUNNING,
            )
        )
    }
    if report.state in order and order[report.state] < order.get(member.state, -1):
        source.pop('state', None)
        source.pop('state_message', None)
    if main:
        return source
    version = source.pop('backend_version', None)
    for field, value in source.items():
        setattr(member, field, value)
    return {
        'distributed_servers': servers,
        **({'backend_version': version} if version else {}),
    }


async def _managed_report_fields(session, model_instance, ctx, model_instance_in):
    if model_instance_in.state not in (None, ModelInstanceStateEnum.ERROR):
        raise BadRequestException(
            message='Managed instance state belongs to the Server'
        )
    if (
        model_instance_in.state == ModelInstanceStateEnum.ERROR
        and not model_instance_in.state_message
    ):
        raise BadRequestException(
            message='A rank health failure requires a diagnostic message'
        )
    if any(
        getattr(model_instance_in, field) is not None
        for field in (
            'arguments',
            'gpu_indexes',
            'gpu_addresses',
            'computed_resource_claim',
        )
    ):
        raise BadRequestException(
            message='Managed members report workload observations'
        )
    worker = ctx.user.worker
    source = {
        "distributed_servers": _merge_workload_report(
            model_instance, ctx, model_instance_in
        )
    }
    await _validate_workload_layout(
        session, model_instance, model_instance_in.workloads
    )
    if model_instance_in.preparation is not None:
        source.update(
            _managed_preparation_fields(
                model_instance, worker.id, model_instance_in.preparation
            )
        )
    if model_instance_in.backend_version is not None:
        if model_instance.backend_version not in (
            None,
            model_instance_in.backend_version,
        ):
            raise ConflictException(
                message="Backend version differs within this generation"
            )
        source["backend_version"] = model_instance_in.backend_version
    if (
        model_instance_in.injected_backend_parameters is not None
        and worker.id == model_instance.worker_id
    ):
        source["injected_backend_parameters"] = (
            model_instance_in.injected_backend_parameters
        )
    # Diagnostics share the manifest's Worker and generation fence. A probe
    # can report failure, but cannot recover an ERROR or request RUNNING.
    if (
        model_instance.state
        in (
            ModelInstanceStateEnum.SCHEDULED,
            ModelInstanceStateEnum.INITIALIZING,
            ModelInstanceStateEnum.STARTING,
            ModelInstanceStateEnum.RUNNING,
        )
        and model_instance.draining_since is None
    ):
        if model_instance_in.state is not None:
            source["state"] = model_instance_in.state
        if model_instance_in.state_message is not None:
            source["state_message"] = model_instance_in.state_message
    return source


@router.delete(
    "/{id}",
    response_model=None,
    responses={202: {"description": "Cleanup is pending"}},
)
async def delete_model_instance(
    session: SessionDep, ctx: TenantContextDep, id: int, response: Response = None
) -> None:
    model_instance = await ModelInstance.one_by_id(session, id, for_update=True)
    assert_resource_visible(
        ctx,
        model_instance,
        not_found_message="Model instance not found",
    )

    if (
        get_model_instance_workloads(model_instance) is not None
        and model_instance.group_id
    ):
        model = await Model.one_by_id(session, model_instance.model_id)
        instances = await ModelInstance.all_by_field(
            session, "model_id", model_instance.model_id
        )
        if model is not None and await model_members_stale(session, model, instances):
            raise BadRequestException(
                message="The PD configuration has changed; restart the model instead."
            )

    try:
        deleted = await ModelInstanceService(session).delete(model_instance)
    except APIHTTPException:
        raise
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to delete model instance: {e}"
        )
    completed = deleted is not False
    if response is not None and not completed:
        response.status_code = 202


async def _resolve_visible_log_source(request, instance, worker, selector):
    """Authorize a source identically for viewing and selected-source download."""
    worker_id = worker.id
    if (
        worker_id not in member_worker_ids(instance)
        or worker.cluster_id != instance.cluster_id
    ):
        raise NotFoundException(message="Log source is unavailable")
    try:
        payload = await fetch_serve_log_options_from_worker(
            request,
            worker,
            instance.id,
            current_restart_count=instance.restart_count or 0,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail="Log sources are unavailable on this Worker"
        ) from exc
    for entry in payload.restarts:
        for source in entry.sources:
            if (
                source.id == selector
                and source.available
                and _log_source_visible(instance, worker_id, source)
            ):
                return worker, source
    raise NotFoundException(message="Log source is unavailable")


def _log_source_visible(instance, worker_id, source) -> bool:
    """Use one generation and task allowlist for options, reads and downloads."""
    current = instance.restart_count or 0
    if (
        source.worker_id != worker_id
        or source.restart_count < 0
        or source.restart_count not in (current, current - 1)
    ):
        return False
    if source.previous != (source.restart_count != current):
        return False
    if source.restart_count != current or source.kind == 'preparation':
        return True
    tasks = get_model_instance_workloads(instance, worker_id)
    if source.kind == 'rank':
        return any(
            task.name == source.workload_name
            and any(
                p.dp_rank == source.dp_rank and p.node_rank == source.node_rank
                for p in task.dp_rank_placements
            )
            for task in tasks or []
        )
    if source.component_kind == 'sglang_smg':
        return any(
            task.component and task.component.kind == 'sglang_smg'
            for task in tasks or []
        )
    if tasks is not None:
        return any(task.name == source.workload_name for task in tasks)
    return True
