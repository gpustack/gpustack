import httpx
import logging

from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse
from openai.types import Model as OAIModel
from openai.pagination import SyncPage

from gpustack.api.exceptions import (
    InvalidException,
    NotFoundException,
    ServiceUnavailableException,
)
from gpustack.schemas.models import Model, ModelInstance, ModelInstanceStateEnum
from gpustack.server.deps import SessionDep

router = APIRouter()

logger = logging.getLogger(__name__)


@router.get("/models")
async def list_models(session: SessionDep):
    result = SyncPage[OAIModel](data=[], object="list")
    models = await Model.all(session)
    for model in models:
        result.data.append(
            OAIModel(
                id=model.name,
                object="model",
                created=int(model.created_at.timestamp()),
                owned_by="gpustack",
            )
        )
    return result


@router.post("/chat/completions")
async def chat_completion(session: SessionDep, request: Request):
    body = await request.json()
    model_name = body.get("model")
    if not model_name:
        raise InvalidException(message="Missing 'model' field")

    model = await Model.one_by_field(session=session, field="name", value=model_name)
    if not model:
        raise NotFoundException(message="Model not found")

    stream = body.get("stream", False)

    request.state.model = model
    request.state.stream = stream

    model_instances = await ModelInstance.all_by_field(
        session=session, field="model_id", value=model.id
    )

    instance = next(
        (
            inst
            for inst in model_instances
            if inst.state == ModelInstanceStateEnum.running
        ),
        None,
    )
    if not instance:
        raise ServiceUnavailableException(message="No running instances available")

    url = f"http://{instance.worker_ip}:{instance.port}/v1/chat/completions"
    logger.debug(f"proxying to {url}")

    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() != "content-length" and key.lower() != "host"
    }

    timeout = 120

    try:
        if stream:

            async def stream():
                async with httpx.AsyncClient() as client:
                    async with client.stream(
                        method=request.method,
                        url=url,
                        headers=headers,
                        json=body,
                        timeout=timeout,
                    ) as resp:
                        async for chunk in resp.aiter_text():
                            yield chunk

            return StreamingResponse(stream())
        else:
            async with httpx.AsyncClient() as client:
                resp = await client.request(
                    method=request.method,
                    url=url,
                    headers=headers,
                    json=body,
                    timeout=timeout,
                )
                return Response(content=resp.content, headers=dict(resp.headers))
    except Exception as e:
        raise ServiceUnavailableException(message=f"An unexpected error occurred: {e}")
