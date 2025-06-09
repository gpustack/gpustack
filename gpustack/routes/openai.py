import asyncio
import os
from typing import List, Optional
import aiohttp
import logging

from fastapi import APIRouter, Query, Request, Response, status
from openai.types import Model as OAIModel
from openai.pagination import SyncPage
from sqlmodel import or_, select
from sqlmodel.ext.asyncio.session import AsyncSession
from starlette.datastructures import UploadFile

from gpustack.api.exceptions import (
    BadRequestException,
    NotFoundException,
    InternalServerErrorException,
    OpenAIAPIError,
    OpenAIAPIErrorResponse,
    ServiceUnavailableException,
    GatewayTimeoutException,
)
from gpustack.api.responses import StreamingResponseWithStatusCode
from gpustack.http_proxy.load_balancer import LoadBalancer
from gpustack.routes.models import build_category_conditions
from gpustack.schemas.models import (
    CategoryEnum,
    Model,
)
from gpustack.server.db import get_session_context
from gpustack.server.deps import SessionDep
from gpustack.server.services import ModelInstanceService, ModelService, WorkerService


logger = logging.getLogger(__name__)

load_balancer = LoadBalancer()


aliasable_router = APIRouter()

PROXY_TIMEOUT = int(os.getenv("GPUSTACK_PROXY_TIMEOUT_SECONDS", 1800))


@aliasable_router.post("/chat/completions")
async def chat_completions(request: Request):
    return await proxy_request_by_model(request, "chat/completions")


@aliasable_router.post("/completions")
async def completions(request: Request):
    return await proxy_request_by_model(request, "completions")


@aliasable_router.post("/embeddings")
async def embeddings(request: Request):
    return await proxy_request_by_model(request, "embeddings")


@aliasable_router.post("/images/generations")
async def images_generations(request: Request):
    return await proxy_request_by_model(request, "images/generations")


@aliasable_router.post("/images/edits")
async def images_edits(request: Request):
    return await proxy_request_by_model(request, "images/edits")


@aliasable_router.post("/audio/speech")
async def audio_speech(request: Request):
    return await proxy_request_by_model(request, "audio/speech")


@aliasable_router.post("/audio/transcriptions")
async def audio_transcriptions(request: Request):
    return await proxy_request_by_model(request, "audio/transcriptions")


router = APIRouter()
router.include_router(aliasable_router)


@router.get("/models")
async def list_models(
    session: SessionDep,
    embedding_only: Optional[bool] = Query(
        None,
        deprecated=True,
        description="This parameter is deprecated and will be removed in a future version.",
    ),
    image_only: Optional[bool] = Query(
        None,
        deprecated=True,
        description="This parameter is deprecated and will be removed in a future version.",
    ),
    reranker: Optional[bool] = Query(
        None,
        deprecated=True,
        description="This parameter is deprecated and will be removed in a future version.",
    ),
    speech_to_text: Optional[bool] = Query(
        None,
        deprecated=True,
        description="This parameter is deprecated and will be removed in a future version.",
    ),
    text_to_speech: Optional[bool] = Query(
        None,
        deprecated=True,
        description="This parameter is deprecated and will be removed in a future version.",
    ),
    categories: List[str] = Query(
        [],
        description="Model categories to filter by.",
    ),
    with_meta: Optional[bool] = Query(
        None,
        description="Include model meta information.",
    ),
):
    all_categories = set(categories)
    if embedding_only:
        all_categories.add(CategoryEnum.EMBEDDING.value)
    if image_only:
        all_categories.add(CategoryEnum.IMAGE.value)
    if reranker:
        all_categories.add(CategoryEnum.RERANKER.value)
    if speech_to_text:
        all_categories.add(CategoryEnum.SPEECH_TO_TEXT.value)
    if text_to_speech:
        all_categories.add(CategoryEnum.TEXT_TO_SPEECH.value)
    all_categories = list(all_categories)

    statement = select(Model).where(Model.ready_replicas > 0)

    if all_categories:
        conditions = build_category_conditions(session, all_categories)
        statement = statement.where(or_(*conditions))

    models = (await session.exec(statement)).all()
    result = SyncPage[OAIModel](data=[], object="list")
    for model in models:
        result.data.append(
            OAIModel(
                id=model.name,
                object="model",
                created=int(model.created_at.timestamp()),
                owned_by="gpustack",
                meta=model.meta if with_meta else None,
            )
        )
    return result


async def proxy_request_by_model(request: Request, endpoint: str):
    """
    Proxy the request to the model instance that is running the model specified in the
    request body.
    """
    async with get_session_context() as session:
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
        worker = await WorkerService(session).get_by_id(instance.worker_id)
        if not worker:
            raise InternalServerErrorException(
                message=f"Worker with ID {instance.worker_id} not found",
                is_openai_exception=True,
            )

    url = f"http://{instance.worker_ip}:{worker.port}/proxy/v1/{endpoint}"
    token = request.app.state.server_config.token
    extra_headers = {
        "X-Target-Port": str(instance.port),
        "Authorization": f"Bearer {token}",
    }

    logger.debug(f"proxying to {url}, instance port: {instance.port}")

    try:
        if stream:
            return await handle_streaming_request(
                request, url, body_json, form_data, form_files, extra_headers
            )
        else:
            return await handle_standard_request(
                request, url, body_json, form_data, form_files, extra_headers
            )
    except asyncio.TimeoutError as e:
        error_message = f"Request to {url} timed out"
        if str(e):
            error_message += f": {e}"
        raise GatewayTimeoutException(
            message=error_message,
            is_openai_exception=True,
        )
    except Exception as e:
        error_message = "An unexpected error occurred"
        if str(e):
            error_message += f": {e}"
        raise ServiceUnavailableException(
            message=error_message,
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

    if form_data and form_data.get("stream", False):
        # stream may be set in form data, e.g., image edits.
        stream = True

    model = await ModelService(session).get_by_name(model_name)
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


async def handle_streaming_request(
    request: Request,
    url: str,
    body_json: Optional[dict],
    form_data: Optional[dict],
    form_files: Optional[list],
    extra_headers: Optional[dict] = None,
):
    timeout = aiohttp.ClientTimeout(total=300)
    headers = filter_headers(request.headers)
    if extra_headers:
        headers.update(extra_headers)

    if body_json and "stream_options" not in body_json:
        # Defaults to include usage.
        # TODO Record usage without client awareness.
        body_json["stream_options"] = {"include_usage": True}

    async def stream_generator():
        try:
            http_client: aiohttp.ClientSession = request.app.state.http_client
            async with http_client.request(
                method=request.method,
                url=url,
                headers=headers,
                json=body_json if body_json else None,
                data=form_data if form_data else None,
                timeout=timeout,
            ) as resp:
                if resp.status >= 400:
                    yield await resp.read(), resp.headers, resp.status
                    return

                chunk = ""
                async for line in resp.content:
                    line = line.decode("utf-8").strip()
                    if line != "":
                        chunk = line + "\n"
                    else:
                        chunk += "\n"
                        yield chunk, resp.headers, resp.status
        except aiohttp.ClientError as e:
            error_response = OpenAIAPIErrorResponse(
                error=OpenAIAPIError(
                    message=f"Service unavailable. Please retry your requests after a brief wait. Original error: {e}",
                    code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    type="ServiceUnavailable",
                ),
            )
            yield error_response.model_dump_json(), {}, status.HTTP_503_SERVICE_UNAVAILABLE
        except Exception as e:
            error_response = OpenAIAPIErrorResponse(
                error=OpenAIAPIError(
                    message=f"Internal server error: {e}",
                    code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    type="InternalServerError",
                ),
            )
            yield error_response.model_dump_json(), {}, status.HTTP_500_INTERNAL_SERVER_ERROR

    return StreamingResponseWithStatusCode(
        stream_generator(), media_type="text/event-stream"
    )


async def handle_standard_request(
    request: Request,
    url: str,
    body_json: Optional[dict],
    form_data: Optional[dict],
    form_files: Optional[list],
    extra_headers: Optional[dict] = None,
):
    headers = filter_headers(request.headers)
    if extra_headers:
        headers.update(extra_headers)
    data = None
    if form_data or form_files:
        form = aiohttp.FormData()
        for key, value in (form_data or {}).items():
            form.add_field(key, value)
        for key, (filename, content, content_type) in form_files or []:
            form.add_field(key, content, filename=filename, content_type=content_type)
        data = form

    http_client: aiohttp.ClientSession = request.app.state.http_client
    timeout = aiohttp.ClientTimeout(total=PROXY_TIMEOUT)
    async with http_client.request(
        method=request.method,
        url=url,
        headers=headers,
        json=body_json if body_json else None,
        data=data if data else None,
        timeout=timeout,
    ) as response:
        content = await response.read()
        return Response(
            status_code=response.status,
            headers=dict(response.headers),
            content=content,
        )


def filter_headers(headers):
    return {
        key: value
        for key, value in headers.items()
        if key.lower() != "content-length"
        and key.lower() != "host"
        and key.lower() != "content-type"
        and key.lower() != "transfer-encoding"
        and key.lower() != "authorization"
    }


async def get_running_instance(session: AsyncSession, model_id: int):
    running_instances = await ModelInstanceService(session).get_running_instances(
        model_id
    )
    if not running_instances:
        raise ServiceUnavailableException(
            message="No running instances available",
            is_openai_exception=True,
        )
    return await load_balancer.get_instance(running_instances)
