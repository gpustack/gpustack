import codecs
import json
import re
import random
import asyncio
import time
from typing import (
    Any,
    AsyncGenerator,
    Coroutine,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)
import aiohttp
import logging

from fastapi import APIRouter, HTTPException, Query, Request, Response, status
from openai.types import Model as OAIModel
from openai.pagination import SyncPage
from sqlmodel import or_, select
from sqlmodel.ext.asyncio.session import AsyncSession
from starlette.datastructures import UploadFile
from gpustack_runtime.envs import to_bool

from gpustack.api.exceptions import (
    BadRequestException,
    NotFoundException,
    InternalServerErrorException,
    OpenAIAPIError,
    OpenAIAPIErrorResponse,
    OpenAIAPIException,
    ServiceUnavailableException,
    GatewayTimeoutException,
)
from gpustack.api.responses import StreamingResponseWithStatusCode
from gpustack import envs
from gpustack.http_proxy.load_balancer import LoadBalancer
from gpustack.routes.model_common import build_category_conditions
from gpustack.schemas.models import Model, servable_instances
from gpustack.schemas.model_provider import (
    ANTHROPIC_API_VERSION,
    ModelProvider,
    ModelProviderTypeEnum,
)
from gpustack.utils.model_instance_workers import (
    get_model_instance_workloads,
    model_instance_endpoints,
)
from gpustack.utils.network import use_proxy_env_for_url
from gpustack.schemas.model_routes import (
    ModelRoute,
    MyModel,
    effective_route_name,
)
from gpustack.schemas.principals import Principal, platform_principal_id
from gpustack.schemas.workers import Worker
from gpustack.routes.model_routes import (
    _model_route_grant_conditions,
    _my_model_visibility_sql,
)
from gpustack.server.db import async_session
from gpustack.server.deps import SessionDep, CurrentUserDep, TenantContextDep
from gpustack.server.services import (
    ModelInstanceService,
    ModelRouteService,
    ModelService,
    RouteTargetResolution,
    WorkerService,
    UserService,
)
from gpustack.server.worker_request import request_to_worker, stream_to_worker
from gpustack.gateway.utils import (
    model_instance_prefix,
    openai_model_prefixes,
    router_header_key,
)

logger = logging.getLogger(__name__)

load_balancer = LoadBalancer()


# Endpoints served by a dedicated server router (e.g. rerank.router), so the
# auto-registration must skip them to avoid duplicate /v1/<endpoint> registration.
_server_managed_elsewhere = {"/rerank"}


def get_api_router() -> APIRouter:
    """Full OpenAI-compatible endpoint set, mounted at /v1."""
    router = APIRouter()
    router.add_api_route("/models", list_models, methods=["GET"])
    for rp in openai_model_prefixes:
        for endpoint in rp.prefixes:
            if endpoint in _server_managed_elsewhere:
                continue
            router.add_api_route(endpoint, proxy_request_by_model, methods=["POST"])
    return router


def get_legacy_api_router() -> APIRouter:
    """Legacy subset, mounted at /v1-openai. Frozen — new endpoints go to /v1 only."""
    router = APIRouter()
    router.add_api_route("/models", list_models, methods=["GET"])
    for rp in openai_model_prefixes:
        if not rp.support_legacy:
            continue
        for endpoint in rp.prefixes:
            if endpoint in _server_managed_elsewhere:
                continue
            router.add_api_route(endpoint, proxy_request_by_model, methods=["POST"])
    return router


async def list_models(
    request: Request,
    user: CurrentUserDep,
    session: SessionDep,
    ctx: TenantContextDep,
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
        # Apply the same Personal vs Org-act-as partition as
        # ``/v2/my-models``: otherwise the Playground happily lists
        # (and lets the user invoke) a model that the My Models page
        # — under the same view — has hidden. Org-mediated grants
        # surface only in the matching Org context; user/group
        # grants surface in Personal.
        vis = _my_model_visibility_sql(ctx)
        if vis is not None:
            statement = statement.where(vis)
    else:
        # Admin act-as: mirror what a member of the current Org would see
        # (own + PUBLIC/AUTHED + cross-tenant grants) rather than the whole
        # ModelRoute table. Admin "All" mode (no current_principal_id) yields
        # no conditions, so every ready route stays visible there.
        for condition in _model_route_grant_conditions(ctx):
            statement = statement.where(condition)

    if categories:
        conditions = build_category_conditions(session, target_class, categories)
        statement = statement.where(or_(*conditions))

    routes = (await session.exec(statement)).all()
    principal_by_id = await _prefetch_owner_principals(session, routes)
    models = [
        _route_to_oai_model(route, principal_by_id, with_meta) for route in routes
    ]
    api_key = getattr(request.state, "api_key", None)
    if api_key is not None and api_key.allowed_model_names:
        allowed_names = set(api_key.allowed_model_names)
        models = [model for model in models if model.id in allowed_names]
    return SyncPage[OAIModel](data=models, object="list")


def _route_to_oai_model(
    route, principal_by_id: Dict[int, Principal], with_meta: Optional[bool]
) -> OAIModel:
    # Wrap the route name through ``effective_route_name`` so non-platform
    # owners get a `<name>/` prefix in the published id — this matches the
    # gateway's ingress dispatch.
    owner = (
        principal_by_id.get(route.owner_principal_id)
        if route.owner_principal_id
        else None
    )
    return OAIModel(
        id=effective_route_name(
            route.name,
            getattr(owner, "name", None),
            getattr(owner, "id", None) == platform_principal_id(),
        ),
        object="model",
        created=int(route.created_at.timestamp()),
        owned_by="gpustack",
        meta=route.meta if with_meta else None,
    )


async def _prefetch_owner_principals(
    session: AsyncSession, routes
) -> Dict[int, Principal]:
    """Bulk-load owner principals so each route's published id is
    owner-name-prefixed for non-platform owners. Empty when all routes
    belong to the platform Org."""
    principal_ids = {
        r.owner_principal_id for r in routes if r.owner_principal_id is not None
    }
    if not principal_ids:
        return {}
    rows = (
        await session.exec(select(Principal).where(Principal.id.in_(principal_ids)))
    ).all()
    return {p.id: p for p in rows}


async def proxy_request_by_model(
    request: Request,
    user: CurrentUserDep,
):
    """
    Proxy the request to the model instance that is running the model specified in the
    request body.

    Uses an inline session instead of SessionDep so the session is released
    after the initial lookups, preventing long-lived streaming inference
    responses from holding a database connection.
    """
    endpoint = re.sub(r"^.*/(?:v1|v1-openai)/", "", request.url.path)
    model_name, stream, body_json, form_data = await parse_request_body(request)

    async with async_session() as session:
        # The per-user access list is a USER question, and one caller is not a
        # user: the benchmark proxy is entered with a worker token, by a load
        # generator this server told a worker to start. Its own prefix carries
        # the exemption (see `internal_inference` in routes.py) rather than the
        # check learning about principal kinds -- so nothing about `/v1`'s gate
        # changes, and the exemption cannot be reached from `/v1`.
        if not getattr(request.state, "internal_inference", False) and not (
            await UserService(session).model_allowed_for_user(
                model_name=model_name,
                user_id=user.id,
                api_key=getattr(request.state, "api_key", None),
            )
        ):
            raise NotFoundException(
                message="Model not found",
                is_openai_exception=True,
            )
        model_route_service = ModelRouteService(session)
        route_targets: List[RouteTargetResolution] = (
            await model_route_service.resolve_route_targets(model_name)
        )
        if not route_targets:
            # resolve_route_targets filters on TargetStateEnum.ACTIVE, so an
            # empty result means either no such route or every target sitting
            # UNAVAILABLE because its model is not servable (any worker that
            # misses /healthz does that to every model on it; a group with a
            # role at zero ready members does it too). 404 would tell the
            # caller a deployed model is gone and not to retry, so split them.
            if await model_route_service.get_by_name(model_name) is None:
                raise NotFoundException(
                    message="Model not found",
                    is_openai_exception=True,
                )
            raise ServiceUnavailableException(
                message="No running instances available",
                is_openai_exception=True,
            )
        request.state.stream = stream
        target = _select_weighted_target(route_targets)

        # Resolve the route id so downstream middleware (usage recording) can
        # attribute the request to the route it entered through. The lookup
        # is @locked_cached so repeat hits within the same session are cheap.
        model_route = await model_route_service.get_by_name(model_name)
        request.state.model_route_id = model_route.id if model_route else None

        provider: Optional[ModelProvider] = None
        target_provider_id = getattr(target, "provider_id", None)
        if target_provider_id is not None:
            provider = await _load_active_provider(session, target_provider_id)
        else:
            instance, rank_endpoint, worker = await _resolve_local_model_target(
                session, request, model_name, target, body_json, form_data
            )

    if provider is not None:
        return await _proxy_to_provider(
            request=request,
            provider=provider,
            target=target,
            model_name=model_name,
            endpoint=endpoint,
            stream=stream,
            body_json=body_json,
            form_data=form_data,
        )

    extra_headers = {
        router_header_key: (
            f"{rank_endpoint.registry_name}.static"
            if rank_endpoint
            else f"{model_instance_prefix(instance)}.static"
        ),
    }
    path = f"v1/{endpoint}"
    logger.debug(
        "Proxying instance %s to Worker %s at %s:%s",
        instance.id,
        worker.id,
        worker.ip,
        rank_endpoint.port if rank_endpoint else instance.port,
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
            # raise_on_error=False: a non-2xx here is an answer, not a transport
            # failure. The engine's own rejection (an unsupported parameter, a
            # malformed body) carries the status and message the caller needs,
            # and raising would flatten every 4xx into the 503 below. The
            # streaming branch forwards the same statuses for the same reason.
            result = await _request_until_disconnect(
                request,
                request_to_worker(
                    worker=worker,
                    method=request.method,
                    path=path,
                    proxy_client=request.app.state.http_client,
                    no_proxy_client=request.app.state.http_client_no_proxy,
                    data=data,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=envs.PROXY_TIMEOUT),
                    raise_on_error=False,
                ),
            )
            if result is None:
                return Response(status_code=499)
            resp, body = result
            # The worker's headers must not be forwarded verbatim: aiohttp
            # hands over an already-decoded body, so a Content-Encoding or
            # Content-Length from the worker leg describes a body the caller
            # never receives, and Transfer-Encoding/Connection are hop-by-hop.
            # Starlette recomputes Content-Length for the body actually sent.
            return Response(
                status_code=resp.status,
                headers=_worker_response_headers(resp.headers),
                content=body,
            )
    except HTTPException as exc:
        raise OpenAIAPIException(
            status_code=exc.status_code, reason="InvalidRequest", message=exc.detail
        ) from exc
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


_T = TypeVar("_T")


async def _resolve_local_model_target(
    session: AsyncSession,
    request: Request,
    model_name: str,
    target: RouteTargetResolution,
    body_json: Optional[dict],
    form_data: Optional[aiohttp.FormData],
):
    model = await ModelService(session).get_by_id(target.model_id)
    if not model:
        raise NotFoundException(
            message="Model not found",
            is_openai_exception=True,
        )
    request.state.model = model
    request.state.overridden_model_name = target.overridden_model_name

    mutate_request(request, model_name, body_json, form_data)

    instance = await get_running_instance(session, model, target.overridden_model_name)
    instance, rank_endpoint = await _select_managed_endpoint(session, instance)
    lookup_worker_id = rank_endpoint.worker_id if rank_endpoint else instance.worker_id
    worker: Worker = await WorkerService(session).get_by_id(lookup_worker_id)
    if not worker or getattr(worker, "deleted_at", None) is not None:
        raise InternalServerErrorException(
            message=f"Worker with ID {lookup_worker_id} not found",
            is_openai_exception=True,
        )
    return instance, rank_endpoint, worker


async def _request_until_disconnect(
    request: Request,
    upstream: Coroutine[Any, Any, _T],
) -> Optional[_T]:
    # The inbound body has already been buffered, so receiving here cannot
    # consume upload data needed by the upstream request.
    async def wait_for_disconnect() -> None:
        while (await request.receive())["type"] != "http.disconnect":
            pass

    work = asyncio.create_task(upstream)
    watcher = asyncio.create_task(wait_for_disconnect())
    try:
        done, _ = await asyncio.wait(
            {work, watcher}, return_when=asyncio.FIRST_COMPLETED
        )
        if watcher in done:
            watcher.result()
            return None
        return work.result()
    finally:
        work.cancel()
        watcher.cancel()
        await asyncio.gather(work, watcher, return_exceptions=True)


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
        stream = to_bool(form.get("stream"))

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


async def _select_managed_endpoint(session, instance):
    rank_endpoint = None
    if get_model_instance_workloads(instance) is not None:
        from gpustack.schemas.models import ModelInstance

        instance = await ModelInstance.one_by_id(session, instance.id, refresh=True)
        endpoints = model_instance_endpoints(instance) if instance is not None else []
        if not endpoints:
            raise ServiceUnavailableException(
                message="No running rank endpoints available",
                is_openai_exception=True,
            )
        rank_endpoint = random.choice(endpoints)
    return instance, rank_endpoint


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
    yielded_any = False
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
            yielded_any = True
            yield chunk, resp_headers, resp_status
    except aiohttp.ClientError as e:
        logger.error(f"Error streaming from worker {worker.id}: {e}", exc_info=True)
        error_response = OpenAIAPIErrorResponse(
            error=OpenAIAPIError(
                message="Service unavailable. Please retry your requests after a brief wait.",
                code=status.HTTP_503_SERVICE_UNAVAILABLE,
                type="ServiceUnavailable",
            ),
        )
        yield _error_chunk(
            error_response, yielded_any
        ), {}, status.HTTP_503_SERVICE_UNAVAILABLE
    except Exception as e:
        logger.error(f"Error streaming from worker {worker.id}: {e}", exc_info=True)
        error_response = OpenAIAPIErrorResponse(
            error=OpenAIAPIError(
                message="Internal server error.",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                type="InternalServerError",
            ),
        )
        yield _error_chunk(
            error_response, yielded_any
        ), {}, status.HTTP_500_INTERNAL_SERVER_ERROR


def _error_chunk(error_response: OpenAIAPIErrorResponse, mid_stream: bool) -> str:
    """
    Render an error body for the streaming proxy. Once the SSE stream has
    started (mid_stream), the error must be sent as an SSE data event so the
    client keeps parsing; before any chunk is sent it is returned as raw JSON
    with an error status code.
    """
    error_json = error_response.model_dump_json()
    return f"data: {error_json}\n\n" if mid_stream else error_json


_STRIPPED_REQUEST_HEADERS = frozenset(
    {
        "authorization",
        "content-length",
        "content-type",
        "cookie",
        "host",
        "transfer-encoding",
        "x-api-key",
    }
)


def filter_headers(headers):
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in _STRIPPED_REQUEST_HEADERS
    }


# Headers that describe the worker-to-server transfer of the body. They must be
# dropped when relaying the answer: aiohttp auto-decompresses, which strands a
# stale Content-Encoding and Content-Length on the decoded bytes, and the
# remaining two are hop-by-hop and belong to the server-to-worker leg only.
_EXCLUDED_WORKER_RESPONSE_HEADERS = frozenset(
    {
        "content-length",
        "content-encoding",
        "transfer-encoding",
        "connection",
    }
)


def _worker_response_headers(headers) -> Dict[str, str]:
    """Worker response headers that are safe to relay to the caller as-is."""
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in _EXCLUDED_WORKER_RESPONSE_HEADERS
    }


async def get_running_instance(
    session: AsyncSession,
    model: Model,
    overridden_model_name: Optional[str] = None,
):
    """Pick a RUNNING instance, narrowing by ``mounted_loras`` when a
    LoRA ``overridden_model_name`` is given. The filter is needed
    because ``mounted_loras`` is a one-shot snapshot taken at STARTING
    and never hot-reloaded
    """
    running_instances = await ModelInstanceService(session).get_running_instances(
        model.id
    )
    # Narrow to the members that can answer a whole request. For a
    # disaggregated group that is the router alone: balancing across the group
    # does not fail, it answers most requests wrongly — a prefill returns after
    # one token and a decode runs without the prefix its KV was meant to carry,
    # both with a 200.
    running_instances = servable_instances(model, running_instances)
    if not running_instances:
        raise ServiceUnavailableException(
            message="No running instances available",
            is_openai_exception=True,
        )
    if overridden_model_name:
        running_instances = [
            inst
            for inst in running_instances
            if inst.mounted_loras
            and any(m.lora_name == overridden_model_name for m in inst.mounted_loras)
        ]
        if not running_instances:
            raise ServiceUnavailableException(
                message=(
                    f"No running instances with LoRA '{overridden_model_name}' mounted. "
                    f"Restart instances after updating lora_list to apply changes."
                ),
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

    override = getattr(request.state, "overridden_model_name", None) or model.name
    if model_name != override:
        if body_json is not None:
            body_json["model"] = override
        elif form_data is not None:
            form_data.add_field("model", override)


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


_EXCLUDED_PROVIDER_RESPONSE_HEADERS = frozenset(
    {
        "connection",
        "content-encoding",
        "content-length",
        "keep-alive",
        "set-cookie",
        "transfer-encoding",
    }
)

# Sampling and diffusion extensions sent by GPUStack's Playground or local
# inference engines (vLLM, llama-box, vox-box) that hosted OpenAI-compatible
# providers reject as unknown fields.
_LOCAL_ENGINE_ONLY_FIELDS = frozenset(
    {
        "cfg_scale",
        "chat_template_kwargs",
        "guided_choice",
        "guided_json",
        "guided_regex",
        "min_p",
        "negative_prompt",
        "repetition_penalty",
        "sample_steps",
        "sampler",
        "sampling_method",
        "schedule",
        "strength",
        "top_k",
    }
)

# OpenAI Chat Completions fields that Gemini's /v1beta/openai/chat/completions
# protobuf parser rejects with HTTP 400 INVALID_ARGUMENT ("Unknown name ...").
_GEMINI_UNSUPPORTED_CHAT_FIELDS = frozenset(
    {
        "frequency_penalty",
        "logit_bias",
        "logprobs",
        "metadata",
        "presence_penalty",
        "seed",
        "service_tier",
        "store",
        "top_logprobs",
        "user",
    }
)


def _select_weighted_target(
    route_targets: List[RouteTargetResolution],
) -> RouteTargetResolution:
    # Weighted target selection mirrors the Higress gateway path (ModelRouteTarget.weight).
    # random.choices raises on an all-zero weight vector, so fall back to uniform there.
    target_weights = [t.weight for t in route_targets]
    if sum(target_weights) > 0:
        return random.choices(route_targets, weights=target_weights, k=1)[0]
    return random.choice(route_targets)


def _strip_upstream_hop_headers(headers) -> Dict[str, str]:
    return {
        k: v
        for k, v in headers.items()
        if k.lower() not in _EXCLUDED_PROVIDER_RESPONSE_HEADERS
    }


async def _load_active_provider(
    session: AsyncSession, provider_id: int
) -> ModelProvider:
    provider = await ModelProvider.one_by_id(session, provider_id)
    if not provider or provider.deleted_at is not None or not provider.api_tokens:
        raise ServiceUnavailableException(
            message="No running instances available",
            is_openai_exception=True,
        )
    return provider


def _resolve_provider_upstream_url(provider: ModelProvider, endpoint: str) -> str:
    base_url, chat_uri = provider.config.get_chat_url()
    if not base_url:
        raise ServiceUnavailableException(
            message=f"Provider '{provider.name}' has no upstream endpoint configured",
            is_openai_exception=True,
        )
    prefix = "/v1"
    if chat_uri and chat_uri.endswith("/chat/completions"):
        prefix = chat_uri[: -len("/chat/completions")]
    elif chat_uri and chat_uri.endswith("/messages"):
        prefix = chat_uri[: -len("/messages")]
    return f"{base_url.rstrip('/')}{prefix}/{endpoint.lstrip('/')}"


def _build_provider_auth_headers(provider: ModelProvider) -> Dict[str, str]:
    token = random.choice(provider.api_tokens)
    if provider.config.type == ModelProviderTypeEnum.CLAUDE:
        version = (
            getattr(provider.config, "claudeVersion", None) or ANTHROPIC_API_VERSION
        )
        return {
            "x-api-key": token,
            "anthropic-version": version,
        }
    return {"Authorization": f"Bearer {token}"}


def _sanitize_provider_body(
    provider_type: ModelProviderTypeEnum,
    endpoint: str,
    body_json: Optional[dict],
) -> Optional[dict]:
    if body_json is None:
        return None
    disallowed = _LOCAL_ENGINE_ONLY_FIELDS
    if (
        provider_type == ModelProviderTypeEnum.GEMINI
        and endpoint.lstrip("/") == "chat/completions"
    ):
        disallowed = disallowed | _GEMINI_UNSUPPORTED_CHAT_FIELDS
    return {k: v for k, v in body_json.items() if v is not None and k not in disallowed}


def _rewrite_provider_model(
    model_name: str,
    overridden_model_name: Optional[str],
    body_json: Optional[dict],
    form_data: Optional[aiohttp.FormData],
) -> str:
    target_model = overridden_model_name or model_name
    if body_json is not None:
        body_json["model"] = target_model
    elif form_data is not None and target_model != model_name:
        form_data.add_field("model", target_model)
    return target_model


def _select_provider_http_client(
    request: Request, provider: ModelProvider, upstream_url: str
) -> aiohttp.ClientSession:
    no_proxy_client = getattr(request.app.state, "http_client_no_proxy", None)
    if (
        provider.proxy_url
        or use_proxy_env_for_url(upstream_url)
        or no_proxy_client is None
    ):
        return request.app.state.http_client
    return no_proxy_client


async def _proxy_to_provider(
    request: Request,
    provider: ModelProvider,
    target: RouteTargetResolution,
    model_name: str,
    endpoint: str,
    stream: bool,
    body_json: Optional[dict],
    form_data: Optional[aiohttp.FormData],
):
    target_model = _rewrite_provider_model(
        model_name, target.overridden_model_name, body_json, form_data
    )
    request.state.model = None
    request.state.model_name = target_model
    request.state.provider_id = provider.id
    request.state.provider_name = provider.name
    request.state.provider_type = provider.config.type.value

    upstream_url = _resolve_provider_upstream_url(provider, endpoint)
    auth_headers = _build_provider_auth_headers(provider)
    body_json = _sanitize_provider_body(provider.config.type, endpoint, body_json)
    headers, data = _prepare_proxy_request(
        request,
        body_json,
        form_data,
        auth_headers,
        add_stream_options=stream,
    )
    timeout = aiohttp.ClientTimeout(total=provider.timeout or envs.PROXY_TIMEOUT)
    client = _select_provider_http_client(request, provider, upstream_url)
    is_chat_completion = endpoint.lstrip("/") == "chat/completions"

    if stream:
        return StreamingResponseWithStatusCode(
            _stream_provider_response(
                client=client,
                method=request.method,
                url=upstream_url,
                headers=headers,
                data=data,
                proxy_url=provider.proxy_url,
                timeout=timeout,
                fallback_model=target_model,
                is_chat_completion=is_chat_completion,
            ),
            media_type="text/event-stream",
        )

    result = await _request_until_disconnect(
        request,
        _execute_provider_json_request(
            client=client,
            provider=provider,
            method=request.method,
            url=upstream_url,
            headers=headers,
            data=data,
            timeout=timeout,
        ),
    )
    if result is None:
        return Response(status_code=499)
    return result


async def _execute_provider_json_request(
    client: aiohttp.ClientSession,
    provider: ModelProvider,
    method: str,
    url: str,
    headers: Dict[str, str],
    data: Optional[Union[bytes, aiohttp.FormData]],
    timeout: aiohttp.ClientTimeout,
) -> Response:
    try:
        async with client.request(
            method=method,
            url=url,
            headers=headers,
            data=data,
            proxy=provider.proxy_url,
            timeout=timeout,
        ) as resp:
            body = await resp.read()
            return Response(
                status_code=resp.status,
                headers=_strip_upstream_hop_headers(resp.headers),
                content=body,
            )
    except asyncio.TimeoutError as e:
        msg = f"Request to provider '{provider.name}' timed out"
        if str(e):
            msg += f": {e}"
        raise GatewayTimeoutException(message=msg, is_openai_exception=True)
    except Exception as e:
        msg = f"Provider request failed: {e}" if str(e) else "Provider request failed"
        raise ServiceUnavailableException(message=msg, is_openai_exception=True)


def _fill_chat_chunk_defaults(parsed: dict, fallback_model: str) -> None:
    now_ts = int(time.time())
    if not parsed.get("id"):
        parsed["id"] = f"chatcmpl-{now_ts}"
    if not parsed.get("object"):
        parsed["object"] = "chat.completion.chunk"
    if parsed.get("created") is None:
        parsed["created"] = now_ts
    if not parsed.get("model"):
        parsed["model"] = fallback_model
    if parsed.get("choices") is None:
        parsed["choices"] = []


def _normalize_sse_frame(
    raw_event: str, fallback_model: str, is_chat_completion: bool = True
) -> bytes:
    stripped = raw_event.strip()
    if not stripped.startswith("data:"):
        return f"{raw_event}\n\n".encode("utf-8")
    payload = stripped[5:].lstrip()
    if payload == "[DONE]":
        return b"data: [DONE]\n\n"
    try:
        parsed = json.loads(payload)
    except Exception:
        return f"data: {payload}\n\n".encode("utf-8")
    if is_chat_completion and isinstance(parsed, dict) and "error" not in parsed:
        _fill_chat_chunk_defaults(parsed, fallback_model)
        serialized = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
        return f"data: {serialized}\n\n".encode("utf-8")
    return f"data: {payload}\n\n".encode("utf-8")


async def _iter_normalized_sse_frames(
    resp, fallback_model: str, is_chat_completion: bool = True
) -> AsyncGenerator[bytes, None]:
    """Reassemble arbitrary TCP chunks into complete LF-delimited SSE frames.

    ModelUsageMiddleware splits each yielded chunk on ``\\n\\n`` and drops the
    trailing segment, so CRLF-delimited provider streams (e.g. Gemini) or SSE
    events split across socket reads must be reassembled before yielding.
    """
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    buf = ""
    async for raw_chunk in resp.content.iter_any():
        if not raw_chunk:
            continue
        buf += decoder.decode(raw_chunk, final=False)
        if buf.endswith("\r"):
            processable, buf = buf[:-1], "\r"
        else:
            processable, buf = buf, ""
        buf = processable.replace("\r\n", "\n") + buf
        while "\n\n" in buf:
            event, buf = buf.split("\n\n", 1)
            if event.strip():
                yield _normalize_sse_frame(event, fallback_model, is_chat_completion)
    buf += decoder.decode(b"", final=True)
    buf = buf.replace("\r\n", "\n")
    if buf.strip():
        yield _normalize_sse_frame(buf, fallback_model, is_chat_completion)


async def _stream_provider_response(
    client: aiohttp.ClientSession,
    method: str,
    url: str,
    headers: Dict[str, str],
    data: Optional[Union[bytes, aiohttp.FormData]],
    proxy_url: Optional[str],
    timeout: aiohttp.ClientTimeout,
    fallback_model: str = "unknown",
    is_chat_completion: bool = True,
) -> AsyncGenerator[Tuple[Union[bytes, str], Dict[str, str], int], None]:
    yielded_any = False
    try:
        async with client.request(
            method=method,
            url=url,
            headers=headers,
            data=data,
            proxy=proxy_url,
            timeout=timeout,
        ) as resp:
            clean_headers = _strip_upstream_hop_headers(resp.headers)
            if resp.status >= 400:
                err_body = await resp.read()
                yield err_body, clean_headers, resp.status
                return
            async for frame in _iter_normalized_sse_frames(
                resp, fallback_model, is_chat_completion
            ):
                yielded_any = True
                yield frame, clean_headers, resp.status
    except aiohttp.ClientError as e:
        logger.error(f"Error streaming from provider {url}: {e}", exc_info=True)
        error_response = OpenAIAPIErrorResponse(
            error=OpenAIAPIError(
                message="Provider unavailable. Please retry after a brief wait.",
                code=status.HTTP_503_SERVICE_UNAVAILABLE,
                type="ServiceUnavailable",
            ),
        )
        yield _error_chunk(
            error_response, yielded_any
        ), {}, status.HTTP_503_SERVICE_UNAVAILABLE
    except Exception as e:
        logger.error(f"Error streaming from provider {url}: {e}", exc_info=True)
        error_response = OpenAIAPIErrorResponse(
            error=OpenAIAPIError(
                message="Internal server error.",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                type="InternalServerError",
            ),
        )
        yield _error_chunk(
            error_response, yielded_any
        ), {}, status.HTTP_500_INTERNAL_SERVER_ERROR
