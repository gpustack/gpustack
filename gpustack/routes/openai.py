import json
import re
import random
import asyncio
from typing import AsyncGenerator, List, Optional, Tuple, Union, Dict
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
from gpustack import envs
from gpustack.http_proxy.load_balancer import LoadBalancer
from gpustack.routes.model_common import build_category_conditions
from gpustack.schemas.models import Model
from gpustack.schemas.model_routes import (
    ModelRoute,
    MyModel,
)
from gpustack.schemas.workers import Worker
from gpustack.server.deps import SessionDep, CurrentUserDep
from gpustack.server.services import (
    ModelInstanceService,
    ModelRouteService,
    WorkerService,
    UserService,
)
from gpustack.server.worker_request import request_to_worker, stream_to_worker


logger = logging.getLogger(__name__)

load_balancer = LoadBalancer()


router = APIRouter()


@router.get("/models")
async def list_models(
    user: CurrentUserDep,
    session: SessionDep,
    categories: List[str] = Query(
        [],
        description="Model categories to filter by.",
    ),
    with_meta: Optional[bool] = Query(
        None,
        description="Include model meta information.",
    ),
):
    target_class = ModelRoute if user.is_admin else MyModel
    statement = select(target_class).where(target_class.ready_targets > 0)
    if target_class == MyModel:
        # Non-admin users should only see their own private models when filtering by categories.
        statement = statement.where(target_class.user_id == user.id)

    if categories:
        conditions = build_category_conditions(session, target_class, categories)
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


@router.post("/completions")
@router.post("/chat/completions")
@router.post("/responses")
@router.post("/embeddings")
@router.post("/images/generations")
@router.post("/images/edits")
@router.post("/audio/speech")
@router.post("/audio/transcriptions")
async def proxy_request_by_model(
    request: Request,
    user: CurrentUserDep,
    session: SessionDep,
):
    endpoint = re.sub(r"^/(v1|v1-openai)/", "", request.url.path)
    """
    Proxy the request to the model instance that is running the model specified in the
    request body.
    """
    model_name, stream, body_json, form_data = await parse_request_body(request)
    if not await UserService(session).model_allowed_for_user(
        model_name=model_name,
        user_id=user.id,
        api_key=getattr(request.state, "api_key", None),
    ):
        raise NotFoundException(
            message="Model not found",
            is_openai_exception=True,
        )
    models: List[Model] = await ModelRouteService(
        session
    ).get_model_ids_by_model_route_name(model_name)
    if len(models) == 0:
        raise NotFoundException(
            message="Model not found or no running instances available",
            is_openai_exception=True,
        )
    request.state.stream = stream
    model = random.choice(models)
    request.state.model = model

    mutate_request(request, model_name, body_json, form_data)

    instance = await get_running_instance(session, model.id)
    worker: Worker = await WorkerService(session).get_by_id(instance.worker_id)
    if not worker:
        raise InternalServerErrorException(
            message=f"Worker with ID {instance.worker_id} not found",
            is_openai_exception=True,
        )
    extra_headers = {
        "X-Target-Port": str(instance.port),
    }
    path = f"proxy/v1/{endpoint}"
    logger.debug(
        f"proxying to {instance.worker_ip}:{instance.port}, instance port: {instance.port}"
    )

    try:
        headers, data = _prepare_proxy_request(
            request,
            body_json,
            form_data,
            extra_headers,
            add_stream_options=stream,
        )
        if stream:
            return StreamingResponseWithStatusCode(
                _stream_response(
                    worker,
                    request.method,
                    path,
                    headers,
                    data,
                    request.app.state.http_client,
                    request.app.state.http_client_no_proxy,
                ),
                media_type="text/event-stream",
            )
        else:
            resp, body = await request_to_worker(
                worker=worker,
                method=request.method,
                path=path,
                proxy_client=request.app.state.http_client,
                no_proxy_client=request.app.state.http_client_no_proxy,
                data=data,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=envs.PROXY_TIMEOUT),
            )
            return Response(
                status_code=resp.status,
                headers=dict(resp.headers),
                content=body,
            )
    except asyncio.TimeoutError as e:
        error_message = f"Request to worker {worker.id} timed out"
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


async def parse_request_body(request: Request):
    model_name = None
    stream = False
    body_json = None
    form_data = None
    content_type = request.headers.get("content-type", "application/json").lower()

    if request.method == "GET":
        model_name = request.query_params.get("model")
    elif content_type.startswith("multipart/form-data"):
        form_data, model_name, stream = await parse_form_data(request)
    else:
        body_json, model_name, stream = await parse_json_body(request)

    if not model_name:
        raise BadRequestException(
            message="Missing 'model' field",
            is_openai_exception=True,
        )

    return model_name, stream, body_json, form_data


async def parse_form_data(request: Request) -> Tuple[aiohttp.FormData, str, bool]:
    try:
        form = await request.form()
        model_name = form.get("model")
        stream = form.get("stream", False)

        form_data = aiohttp.FormData()
        for key, value in form.items():
            if isinstance(value, UploadFile):
                form_data.add_field(
                    key,
                    await value.read(),
                    filename=value.filename,
                    content_type=value.content_type,
                )
            else:
                form_data.add_field(key, value)

        return form_data, model_name, stream
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


def _prepare_proxy_request(
    request: Request,
    body_json: Optional[dict],
    form_data: Optional[aiohttp.FormData],
    extra_headers: Optional[dict] = None,
    add_stream_options: bool = False,
) -> Tuple[Dict[str, str], Optional[Union[bytes, aiohttp.FormData]]]:
    """
    Prepare headers and body for proxy requests.

    Returns (headers, data) tuple.
    """
    headers = filter_headers(request.headers)
    if extra_headers:
        headers.update(extra_headers)

    if add_stream_options and body_json and "stream_options" not in body_json:
        # Defaults to include usage.
        # TODO Record usage without client awareness.
        body_json["stream_options"] = {"include_usage": True}

    # Convert body to data
    data = (
        form_data
        if form_data
        else (json.dumps(body_json).encode() if body_json else None)
    )

    # When using data=bytes (instead of json=), aiohttp doesn't set Content-Type
    if body_json and not form_data:
        headers["Content-Type"] = "application/json"

    return headers, data


async def _stream_response(
    worker: Worker,
    method: str,
    path: str,
    headers: Dict[str, str],
    data: Optional[Union[bytes, aiohttp.FormData]],
    proxy_client: aiohttp.ClientSession,
    no_proxy_client: aiohttp.ClientSession,
) -> AsyncGenerator[Tuple[Union[bytes, str], Dict[str, str], int], None]:
    """
    Stream response from worker. Yields (chunk, headers, status) tuples.
    """
    try:
        async for chunk, resp_headers, resp_status in stream_to_worker(
            worker=worker,
            method=method,
            path=path,
            proxy_client=proxy_client,
            no_proxy_client=no_proxy_client,
            data=data,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=envs.PROXY_TIMEOUT),
        ):
            yield chunk, resp_headers, resp_status
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


def mutate_request(
    request: Request,
    model_name: str,
    body_json: Optional[dict],
    form_data: Optional[aiohttp.FormData],
):
    path = request.url.path
    model: Model = request.state.model
    if (
        path == "/v1/rerank"
        and body_json
        and model.env
        and model.env.get("GPUSTACK_APPLY_QWEN3_RERANKER_TEMPLATES", False)
    ):
        apply_qwen3_reranker_templates(body_json)
    if model_name != model.name:
        if body_json is not None:
            body_json["model"] = model.name
        elif form_data is not None:
            form_data.add_field("model", model.name)


def apply_qwen3_reranker_templates(body_json: dict):
    """
    Apply Qwen3 reranker templates to the request body.
    See instructions in https://huggingface.co/Qwen/Qwen3-Reranker-0.6B.
    Note: Once vLLM supports built-in template rendering for this model, this can be removed.
    """
    prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    query_template = "{prefix}<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n<Query>: {query}\n"
    document_template = "<Document>: {doc}{suffix}"
    if "query" in body_json and "documents" in body_json:
        query = body_json["query"]
        documents = body_json["documents"]
        body_json["query"] = query_template.format(prefix=prefix, query=query)
        body_json["documents"] = [
            document_template.format(doc=doc, suffix=suffix) for doc in documents
        ]
