from typing import Optional
import httpx
import logging

from fastapi import APIRouter, Request, Response, status
from openai.types import Model as OAIModel
from openai.pagination import SyncPage
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from starlette.datastructures import UploadFile

from gpustack.api.exceptions import (
    BadRequestException,
    NotFoundException,
    OpenAIAPIError,
    OpenAIAPIErrorResponse,
    ServiceUnavailableException,
)
from gpustack.api.responses import StreamingResponseWithStatusCode
from gpustack.http_proxy.load_balancer import LoadBalancer
from gpustack.schemas.models import Model, ModelInstance, ModelInstanceStateEnum
from gpustack.server.deps import SessionDep


logger = logging.getLogger(__name__)

load_balancer = LoadBalancer()


aliasable_router = APIRouter()


@aliasable_router.post("/chat/completions")
async def chat_completions(session: SessionDep, request: Request):
    return await proxy_request_by_model(request, session, "chat/completions")


@aliasable_router.post("/completions")
async def completions(session: SessionDep, request: Request):
    return await proxy_request_by_model(request, session, "completions")


@aliasable_router.post("/embeddings")
async def embeddings(session: SessionDep, request: Request):
    return await proxy_request_by_model(request, session, "embeddings")


@aliasable_router.post("/images/generations")
async def images_generations(session: SessionDep, request: Request):
    return await proxy_request_by_model(request, session, "images/generations")


@aliasable_router.post("/audio/speech")
async def audio_speech(session: SessionDep, request: Request):
    return await proxy_request_by_model(request, session, "audio/speech")


@aliasable_router.post("/audio/transcriptions")
async def audio_transcriptions(session: SessionDep, request: Request):
    return await proxy_request_by_model(request, session, "audio/transcriptions")


router = APIRouter()
router.include_router(aliasable_router)


@router.get("/models")
async def list_models(
    session: SessionDep,
    embedding_only: Optional[bool] = None,
    image_only: Optional[bool] = None,
    reranker: Optional[bool] = None,
):
    statement = select(Model).where(Model.ready_replicas > 0)

    if embedding_only is not None:
        statement = statement.where(Model.embedding_only == embedding_only)

    if image_only is not None:
        statement = statement.where(Model.image_only == image_only)

    if reranker is not None:
        statement = statement.where(Model.reranker == reranker)

    models = (await session.exec(statement)).all()
    result = SyncPage[OAIModel](data=[], object="list")
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


async def proxy_request_by_model(request: Request, session: SessionDep, endpoint: str):
    """
    Proxy the request to the model instance that is running the model specified in the
    request body.
    """
    model, stream, body_json, form_data, form_files = await parse_request_body(
        request, session
    )
    if not model:
        raise NotFoundException(
            message="Model not found",
            is_openai_exception=True,
        )

    request.state.model = model
    request.state.stream = stream

    instance = await get_running_instance(session, model.id)

    url = f"http://{instance.worker_ip}:{instance.port}/v1/{endpoint}"
    logger.debug(f"proxying to {url}")

    try:
        if stream:
            return await handle_streaming_request(request, url, body_json)
        else:
            return await handle_standard_request(
                request, url, body_json, form_data, form_files
            )
    except Exception as e:
        raise ServiceUnavailableException(
            message=f"An unexpected error occurred: {e}",
            is_openai_exception=True,
        )


async def parse_request_body(request: Request, session: SessionDep):
    model_name = None
    stream = False
    body_json = None
    form_data = None
    form_files = None
    content_type = request.headers.get("content-type", "application/json").lower()

    if request.method == "GET":
        model_name = request.query_params.get("model")
    elif content_type.startswith("multipart/form-data"):
        form_data, form_files, model_name = await parse_form_data(request)
    else:
        body_json, model_name, stream = await parse_json_body(request)

    if not model_name:
        raise BadRequestException(
            message="Missing 'model' field",
            is_openai_exception=True,
        )

    model = await get_model(session, model_name)
    return model, stream, body_json, form_data, form_files


async def parse_form_data(request: Request):
    try:
        form = await request.form()
        model_name = form.get("model")

        form_files = []
        form_data = {}
        for key, value in form.items():
            if isinstance(value, UploadFile):
                form_files.append(
                    (key, (value.filename, await value.read(), value.content_type))
                )
            else:
                form_data[key] = value

        return form_data, form_files, model_name
    except Exception as e:
        raise BadRequestException(
            message=f"We could not parse the form body of your request: {e}",
            is_openai_exception=True,
        )


async def parse_json_body(request: Request):
    try:
        body_json = await request.json()
        model_name = body_json.get("model")
        stream = body_json.get("stream", False)
        return body_json, model_name, stream
    except Exception as e:
        raise BadRequestException(
            message=f"We could not parse the JSON body of your request: {e}",
            is_openai_exception=True,
        )


async def get_model(session: SessionDep, model_name: Optional[str]):
    return await Model.one_by_field(session=session, field="name", value=model_name)


async def handle_streaming_request(
    request: Request, url: str, body_json: Optional[dict]
):
    timeout = 120
    headers = filter_headers(request.headers)

    if body_json and "stream_options" not in body_json:
        # Defaults to include usage.
        # TODO Record usage without client awareness.
        body_json["stream_options"] = {"include_usage": True}

    async def stream_generator():
        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    method=request.method,
                    url=url,
                    headers=headers,
                    json=body_json,
                    timeout=timeout,
                ) as resp:
                    chunk = ""
                    async for line in resp.aiter_lines():
                        if line != "":
                            chunk = line + "\n"
                        else:
                            chunk += "\n"
                            yield chunk, resp.status_code
        except httpx.RequestError:
            error_response = OpenAIAPIErrorResponse(
                error=OpenAIAPIError(
                    message="Service unavailable. Please retry your requests after a brief wait.",
                    code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    type="ServiceUnavailable",
                ),
            )
            yield error_response.model_dump_json(), status.HTTP_503_SERVICE_UNAVAILABLE
        except Exception as e:
            error_response = OpenAIAPIErrorResponse(
                error=OpenAIAPIError(
                    message=f"Internal server error: {e}",
                    code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    type="InternalServerError",
                ),
            )
            yield error_response.model_dump_json(), status.HTTP_500_INTERNAL_SERVER_ERROR

    return StreamingResponseWithStatusCode(
        stream_generator(), media_type="text/event-stream"
    )


async def handle_standard_request(
    request: Request,
    url: str,
    body_json: Optional[dict],
    form_data: Optional[dict],
    form_files: Optional[list],
):
    timeout = 120
    headers = filter_headers(request.headers)

    async with httpx.AsyncClient() as client:
        resp = await client.request(
            method=request.method,
            url=url,
            headers=headers,
            json=body_json if body_json else None,
            data=form_data if form_data else None,
            files=form_files if form_files else None,
            timeout=timeout,
        )
        return Response(
            status_code=resp.status_code,
            headers=dict(resp.headers),
            content=resp.content,
        )


def filter_headers(headers):
    return {
        key: value
        for key, value in headers.items()
        if key.lower() != "content-length"
        and key.lower() != "host"
        and key.lower() != "content-type"
    }


async def get_running_instance(session: AsyncSession, model_id: int):
    model_instances = await ModelInstance.all_by_field(
        session=session, field="model_id", value=model_id
    )
    running_instances = [
        inst for inst in model_instances if inst.state == ModelInstanceStateEnum.RUNNING
    ]
    if not running_instances:
        raise ServiceUnavailableException(
            message="No running instances available",
            is_openai_exception=True,
        )
    return await load_balancer.get_instance(running_instances)
