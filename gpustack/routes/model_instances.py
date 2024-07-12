from typing import Optional
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
import httpx

from gpustack.config.config import Config
from gpustack.worker.logs import LogOptionsDep
from gpustack.api.exceptions import (
    InternalServerErrorException,
    NotFoundException,
)
from gpustack.schemas.workers import Worker
from gpustack.server.deps import ListParamsDep, SessionDep
from gpustack.schemas.models import (
    ModelInstance,
    ModelInstanceCreate,
    ModelInstancePublic,
    ModelInstanceUpdate,
    ModelInstancesPublic,
)

router = APIRouter()


@router.get("", response_model=ModelInstancesPublic)
async def get_model_instances(
    session: SessionDep,
    params: ListParamsDep,
    model_id: Optional[int] = None,
    worker_id: Optional[int] = None,
    state: Optional[str] = None,
):
    fields = {}
    if model_id:
        fields["model_id"] = model_id

    if worker_id:
        fields["worker_id"] = worker_id

    if state:
        fields["state"] = state

    if params.watch:
        return StreamingResponse(
            ModelInstance.streaming(session, fields=fields),
            media_type="text/event-stream",
        )

    return await ModelInstance.paginated_by_query(
        session=session,
        fields=fields,
        page=params.page,
        per_page=params.perPage,
    )


@router.get("/{id}", response_model=ModelInstancePublic)
async def get_model_instance(session: SessionDep, id: int):
    model_instance = await ModelInstance.one_by_id(session, id)
    if not model_instance:
        raise NotFoundException(message="Model instance not found")
    return model_instance


@router.get("/{id}/logs")
async def get_serving_logs(
    request: Request, session: SessionDep, id: int, log_options: LogOptionsDep
):
    model_instance = await ModelInstance.one_by_id(session, id)
    if not model_instance:
        raise NotFoundException(message="Model instance not found")

    if not model_instance.worker_id:
        raise NotFoundException(message="Model instance not assigned to a worker")

    # proxy to worker's model_instance logs endpoint
    worker = await Worker.one_by_id(session, model_instance.worker_id)
    if not worker:
        raise NotFoundException(message="Model instance's worker not found")

    server_config: Config = request.app.state.server_config

    model_instance_log_url = (
        f"http://{worker.ip}:{server_config.worker_port}/serveLogs"
        f"/{model_instance.id}?{log_options.url_encode()}"
    )

    timeout = httpx.Timeout(10.0, read=None)
    client: httpx.AsyncClient = request.app.state.http_client

    if log_options.follow:

        async def proxy_stream():
            async with client.stream(
                "GET", model_instance_log_url, timeout=timeout
            ) as response:
                if response.status_code != 200:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail="Error fetching serving logs",
                    )
                async for chunk in response.aiter_bytes():
                    yield chunk

        return StreamingResponse(proxy_stream(), media_type="application/octet-stream")
    else:
        response = await client.get(model_instance_log_url)
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail="Error fetching serving logs",
            )

        return PlainTextResponse(
            content=response.text, status_code=response.status_code
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
    model_instance = await ModelInstance.one_by_id(session, id)
    if not model_instance:
        raise NotFoundException(message="Model instance not found")

    try:
        await model_instance.update(session, model_instance_in)
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to update model instance: {e}"
        )

    return model_instance


@router.delete("/{id}")
async def delete_model_instance(session: SessionDep, id: int):
    model_instance = await ModelInstance.one_by_id(session, id)
    if not model_instance:
        raise NotFoundException(message="Model instance not found")

    try:
        await model_instance.delete(session)
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to delete model instance: {e}"
        )
