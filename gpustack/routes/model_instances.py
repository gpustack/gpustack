from typing import Optional
import aiohttp
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import PlainTextResponse, StreamingResponse, RedirectResponse
from urllib.parse import urlencode
from sqlalchemy.orm import selectinload

from gpustack.api.responses import StreamingResponseWithStatusCode
from gpustack import envs
from gpustack.server.services import ModelInstanceService
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
    ModelInstance,
    ModelInstanceCreate,
    ModelInstancePublic,
    ModelInstanceUpdate,
    ModelInstancesPublic,
    ModelInstanceStateEnum,
)
from gpustack.schemas.model_files import ModelFileStateEnum
from gpustack.config.config import get_global_config
from gpustack.utils.grafana import resolve_grafana_base_url

router = APIRouter()


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
    request: Request, session: SessionDep, id: int, log_options: LogOptionsDep
):
    model_instance = await fetch_model_instance(session, id)
    worker = await fetch_worker(session, model_instance.worker_id)

    model_instance_log_url = (
        f"http://{worker.advertise_address}:{worker.port}/serveLogs"
        f"/{model_instance.id}?{log_options.url_encode()}"
        f"&model_instance_name={model_instance.name}"
    )
    if (
        model_instance.state != ModelInstanceStateEnum.RUNNING
        and model_instance.model_files
        and model_instance.model_files[0].state != ModelFileStateEnum.READY
    ):
        # Get model file ID for injected download logs if instance is downloading
        model_instance_log_url += f"&model_file_id={model_instance.model_files[0].id}"

    timeout = aiohttp.ClientTimeout(total=envs.PROXY_TIMEOUT, sock_connect=5)

    use_proxy_env = use_proxy_env_for_url(model_instance_log_url)
    client: aiohttp.ClientSession = (
        request.app.state.http_client
        if use_proxy_env
        else request.app.state.http_client_no_proxy
    )

    if log_options.follow:

        async def proxy_stream():
            try:
                async with client.get(model_instance_log_url, timeout=timeout) as resp:
                    if resp.status != 200:
                        body = await resp.read()
                        yield body, resp.headers, resp.status
                        return

                    async for chunk in resp.content.iter_any():
                        yield chunk, resp.headers, resp.status
            except TimeoutError:
                yield "\x1b[999;1H" + f"Log stream timed out ({timeout.total} seconds). Please reopen the log page.\n", {}, status.HTTP_500_INTERNAL_SERVER_ERROR
            except Exception as e:
                yield "\x1b[999;1H" + f"Error fetching serving logs: {str(e)}\n", {}, status.HTTP_500_INTERNAL_SERVER_ERROR

        return StreamingResponseWithStatusCode(
            proxy_stream(),
            media_type="application/octet-stream",
        )
    else:
        try:
            async with client.get(model_instance_log_url, timeout=timeout) as resp:
                if resp.status != 200:
                    raise HTTPException(
                        status_code=resp.status,
                        detail="Error fetching serving logs",
                    )
                content = await resp.text()
            return PlainTextResponse(content=content, status_code=resp.status)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error fetching serving logs: {str(e)}\n"
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
