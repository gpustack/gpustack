import asyncio
from typing import Dict, List, Optional, Tuple
import aiohttp
from fastapi import APIRouter, Request, status, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse, RedirectResponse
from urllib.parse import urlencode
from sqlalchemy.orm import selectinload

from gpustack.api.responses import StreamingResponseWithStatusCode
from gpustack import envs
from gpustack.server.services import ModelInstanceService
from gpustack.server.worker_request import request_to_worker, stream_to_worker
from gpustack.utils.network import use_proxy_env_for_url
from gpustack.worker.logs import LogOptionsDep
from gpustack.api.exceptions import (
    InternalServerErrorException,
    NotFoundException,
)
from gpustack.schemas.workers import Worker
from gpustack.schemas.clusters import Cluster
from gpustack.server.db import async_session
from gpustack.server.deps import ListParamsDep, SessionDep
from gpustack.schemas.models import (
    BackendEnum,
    ModelInstance,
    ModelInstanceCreate,
    ModelInstanceLogOptions,
    ModelInstanceLogWorkerOption,
    ModelInstancePublic,
    ModelInstanceUpdate,
    ModelInstancesPublic,
    ModelInstanceStateEnum,
    ServeLogOptionsResponse,
)
from gpustack.schemas.model_files import ModelFileStateEnum
from gpustack.config.config import get_global_config
from gpustack.utils.grafana import resolve_grafana_base_url

router = APIRouter()


# Subordinate-worker display names, keyed by BackendEnum values.
_SUBORDINATE_DISPLAY_NAMES: Dict[str, str] = {
    BackendEnum.VLLM: "ray-worker",
}


def _default_display_name(backend: Optional[str], is_main_worker: bool) -> str:
    """Resolve the UI display name for the internal 'default' container."""
    if is_main_worker:
        return backend or "default"
    if backend and backend in _SUBORDINATE_DISPLAY_NAMES:
        return _SUBORDINATE_DISPLAY_NAMES[backend]
    # Generic subordinate: "sub-<backend>" or just "subordinate".
    return f"sub-{backend}" if backend else "subordinate"


def _map_container_display_name(
    internal_name: str, backend: Optional[str], is_main_worker: bool
) -> str:
    """Forward-map an internal container name to its UI display name."""
    if internal_name != "default":
        return internal_name
    return _default_display_name(backend, is_main_worker)


def _unmap_container_display_name(
    display_name: str, backend: Optional[str], is_main_worker: bool
) -> str:
    """Reverse-map a UI display name back to the internal container name."""
    if display_name == _default_display_name(backend, is_main_worker):
        return "default"
    return display_name


@router.get("", response_model=ModelInstancesPublic)
async def get_model_instances(
    params: ListParamsDep,
    id: Optional[int] = None,
    model_id: Optional[int] = None,
    worker_id: Optional[int] = None,
    state: Optional[str] = None,
):
    fields = {}
    if id:
        fields["id"] = id

    if model_id:
        fields["model_id"] = model_id

    if worker_id:
        fields["worker_id"] = worker_id

    if state:
        fields["state"] = state

    if params.watch:
        return StreamingResponse(
            ModelInstance.streaming(fields=fields),
            media_type="text/event-stream",
        )

    async with async_session() as session:
        return await ModelInstance.paginated_by_query(
            session=session,
            fields=fields,
            page=params.page,
            per_page=params.perPage,
        )


@router.get("/{id}", response_model=ModelInstancePublic)
async def get_model_instance(
    session: SessionDep,
    id: int,
):
    model_instance = await ModelInstance.one_by_id(session, id)
    if not model_instance:
        raise NotFoundException(message="Model instance not found")
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


async def fetch_model_instance(session, id):
    model_instance = await ModelInstance.one_by_id(
        session, id, options=[selectinload(ModelInstance.model_files)]
    )
    if not model_instance:
        raise NotFoundException(message="Model instance not found")
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
    session: SessionDep,
    id: int,
    log_options: LogOptionsDep,
    worker_id: Optional[int] = None,
    container_name: Optional[str] = None,
):
    model_instance = await fetch_model_instance(session, id)

    # Reverse-map: convert UI display name back to internal container name.
    if container_name:
        is_main = (worker_id or model_instance.worker_id) == model_instance.worker_id
        container_name = _unmap_container_display_name(
            container_name, model_instance.backend, is_main
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

    params = {
        "tail": log_options.tail,
        "follow": log_options.follow,
        "model_instance_name": model_instance.name,
        "previous": log_options.previous,
    }
    if container_name:
        params["container_name"] = container_name
    if (
        model_instance.state != ModelInstanceStateEnum.RUNNING
        and model_instance.model_files
        and model_instance.model_files[0].state != ModelFileStateEnum.READY
    ):
        params["model_file_id"] = model_instance.model_files[0].id

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
    log_options_url = (
        f"http://{worker.advertise_address}:{worker.port}/serveLogOptions"
        f"/{model_instance_id}"
    )
    timeout = aiohttp.ClientTimeout(total=envs.PROXY_TIMEOUT, sock_connect=5)
    use_proxy_env = use_proxy_env_for_url(log_options_url)
    client: aiohttp.ClientSession = (
        request.app.state.http_client
        if use_proxy_env
        else request.app.state.http_client_no_proxy
    )
    try:
        async with client.get(log_options_url, timeout=timeout) as resp:
            if resp.status != 200:
                raise ValueError(
                    f"HTTP {resp.status}: error fetching model instance log options"
                )
            data = await resp.json()
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(str(e)) from e

    return ServeLogOptionsResponse.model_validate(
        data if isinstance(data, dict) else {}
    )


@router.get("/{id}/log-options", response_model=ModelInstanceLogOptions)
async def get_model_instance_log_options(
    request: Request,
    session: SessionDep,
    id: int,
):
    """Return per-worker restart_count values that exist on disk for this model instance."""
    model_instance = await fetch_model_instance(session, id)
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
                _map_container_display_name(c, model_instance.backend, is_main)
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
    try:
        model_instance = await ModelInstance.create(session, model_instance_in)
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to create model instance: {e}"
        )
    return model_instance


@router.put("/{id}", response_model=ModelInstancePublic)
async def update_model_instance(
    session: SessionDep, id: int, model_instance_in: ModelInstanceUpdate
):
    model_instance = await ModelInstance.one_by_id(session, id, for_update=True)
    if not model_instance:
        raise NotFoundException(message="Model instance not found")

    try:
        await ModelInstanceService(session).update(model_instance, model_instance_in)
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to update model instance: {e}"
        )
    return model_instance


@router.delete("/{id}")
async def delete_model_instance(session: SessionDep, id: int):
    model_instance = await ModelInstance.one_by_id(session, id, for_update=True)
    if not model_instance:
        raise NotFoundException(message="Model instance not found")

    try:
        await ModelInstanceService(session).delete(model_instance)
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to delete model instance: {e}"
        )
