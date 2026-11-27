from datetime import datetime, timezone
import functools
import json
import logging
import re
import time
from typing import Set, Type, Union
from uuid import uuid4
from fastapi import Request, Response, status
from fastapi.responses import (
    FileResponse,
    StreamingResponse,
    JSONResponse,
    PlainTextResponse,
)
from jwt import DecodeError, ExpiredSignatureError
from starlette._utils import get_route_path
from starlette.datastructures import MutableHeaders
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import ClientDisconnect
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types import CompletionUsage
from openai.types.audio.transcription_create_response import (
    Transcription,
)
from openai.types.create_embedding_response import (
    Usage as EmbeddingUsage,
)
from gpustack.api.exceptions import ErrorResponse, request_error_log_path
from gpustack.observability.tracing import (
    current_inference_span,
    set_http_status,
    trace_inference_http,
    trace_log_fields,
)
from gpustack.routes.rerank import RerankResponse, RerankUsage
from gpustack.schemas.images import ImageGenerationChunk, ImagesResponse
from gpustack.schemas.model_usage import OperationEnum
from gpustack.schemas.api_keys import ApiKey
from gpustack.schemas.models import Model
from gpustack.schemas.users import User
from gpustack.security import JWTManager
from gpustack import envs
from gpustack.api.auth import SESSION_COOKIE_NAME

from gpustack.server.metrics_collector import (
    ModelUsageMetrics,
    accumulate_gateway_metrics,
)
from gpustack.api.types.openai_ext import CreateEmbeddingResponseExt, CompletionExt

logger = logging.getLogger(__name__)
inference_access_logger = logging.getLogger("gpustack.inference.access")

GPUSTACK_REQUEST_ID_HEADER = "X-GPUStack-Request-Id"
_VLLM_PD_RESPONSE_ID = re.compile(
    r"(?:chatcmpl-|cmpl-)"
    r"(___prefill_addr_[A-Za-z0-9_.:\[\]-]{1,256}"
    r"___decode_addr_[A-Za-z0-9_.:\[\]-]{1,256}_[0-9a-fA-F]{32})"
)


class InferenceAccessMiddleware:
    """Assign a request ID before authentication and log the response lifecycle."""

    def __init__(self, app: ASGIApp, paths: Set[str]):
        self.app = app
        self.paths = paths

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return
        path = get_route_path(scope)
        if path not in self.paths:
            await self.app(scope, receive, send)
            return

        request_id = str(uuid4())
        headers = MutableHeaders(scope=scope)
        headers["X-Request-Id"] = request_id
        del headers[GPUSTACK_REQUEST_ID_HEADER]
        access = {"request_id": request_id, "method": "POST", "path": path}
        scope.setdefault("state", {})["inference_access"] = access
        start = time.monotonic()

        async def send_with_request_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                access["response_code"] = message["status"]
                MutableHeaders(scope=message)[GPUSTACK_REQUEST_ID_HEADER] = request_id
            await send(message)

        async def record_access(scope: Scope, receive: Receive, send: Send) -> None:
            try:
                await self.app(scope, receive, send)
            except ClientDisconnect:
                raise
            except Exception:
                if "response_code" not in access:
                    access["response_code"] = status.HTTP_500_INTERNAL_SERVER_ERROR
                    set_http_status(current_inference_span(), access["response_code"])
                raise
            finally:
                access["duration"] = round((time.monotonic() - start) * 1000)
                inference_access_logger.info(
                    "POST %s %s request_id=%s",
                    path,
                    access.get("response_code", "-"),
                    request_id,
                    extra={**access, **trace_log_fields()},
                )

        tracer = getattr(scope["app"].state, "inference_tracer", None)
        if tracer is None:
            await record_access(scope, receive, send_with_request_id)
        else:
            await trace_inference_http(
                record_access,
                scope,
                receive,
                send_with_request_id,
                tracer,
                f"POST {path}",
                {
                    "http.request.method": "POST",
                    "http.route": path,
                    "gpustack.request_id": request_id,
                },
            )


async def inference_server_error(request: Request, _exc: Exception) -> Response:
    """Attach the request ID to the outer middleware's fallback 500."""
    access = getattr(request.state, "inference_access", None)
    return PlainTextResponse(
        "Internal Server Error",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        headers={GPUSTACK_REQUEST_ID_HEADER: access["request_id"]} if access else None,
    )


@functools.lru_cache(maxsize=1)
def _warn_about_missing_start_time() -> None:
    """Per-process warn-once for the RequestTimeMiddleware misconfiguration.

    ``lru_cache`` keeps the latch encapsulated inside the function — there's
    no module-level mutable state to track or reset in tests.
    """
    logger.warning(
        "request.state.start_time missing in record_model_usage; "
        "RequestTimeMiddleware may not be registered or runs after "
        "ModelUsageMiddleware. Falling back to now() — started_at and "
        "completed_at on the audit row will be equal until this is fixed."
    )


class RequestTimeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request.state.start_time = datetime.now(timezone.utc)
        try:
            response = await call_next(request)
        except Exception as e:
            # Log the full traceback so unexpected errors don't disappear
            # behind the generic 500 response. The exception is otherwise
            # serialized only via str(e), which often hides the real cause
            # (validation errors, attribute errors with terse repr, etc.).
            logger.exception(
                "Unhandled exception in request %s %s",
                request.method,
                request_error_log_path(request),
            )
            response = JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    reason="Internal Server Error",
                    message=f"Unexpected error occurred: {e}",
                ).model_dump(),
            )
        return response


class ModelUsageMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if response.status_code == 200:
            path = request.url.path
            if path == "/v1-openai/chat/completions" or path == "/v1/chat/completions":
                return await process_request(
                    request, response, ChatCompletion, OperationEnum.CHAT_COMPLETION
                )
            elif path == "/v1-openai/completions" or path == "/v1/completions":
                return await process_request(
                    request, response, CompletionExt, OperationEnum.COMPLETION
                )
            elif path == "/v1-openai/embeddings" or path == "/v1/embeddings":
                return await process_request(
                    request,
                    response,
                    CreateEmbeddingResponseExt,
                    OperationEnum.EMBEDDING,
                )
            elif (
                path == "/v1-openai/images/generations"
                or path == "/v1/images/generations"
                or path == "/v1-openai/images/edits"
                or path == "/v1/images/edits"
            ):
                return await process_request(
                    request,
                    response,
                    ImagesResponse,
                    OperationEnum.IMAGE_GENERATION,
                )
            elif path == "/v1-openai/audio/speech" or path == "/v1/audio/speech":
                return await process_request(
                    request,
                    response,
                    FileResponse,
                    OperationEnum.AUDIO_SPEECH,
                )
            elif (
                path == "/v1-openai/audio/transcriptions"
                or path == "/v1/audio/transcriptions"
            ):
                return await process_request(
                    request,
                    response,
                    Transcription,
                    OperationEnum.AUDIO_TRANSCRIPTION,
                )
            elif request.url.path == "/v1/rerank":
                return await process_request(
                    request,
                    response,
                    RerankResponse,
                    OperationEnum.RERANK,
                )

        return response


def _capture_pd_coordinator_id(request: Request, response: Union[dict, str]) -> None:
    access = getattr(request.state, "inference_access", None)
    if access is None or "pd_coordinator_id" in access:
        return
    config = getattr(request.app.state, "server_config", None)
    if not getattr(config, "log_otlp_endpoint", None):
        return
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except (ValueError, RecursionError):
            return

    response_id = response.get("id") if isinstance(response, dict) else None
    if isinstance(response_id, str):
        match = _VLLM_PD_RESPONSE_ID.fullmatch(response_id)
        if match:
            access["pd_coordinator_id"] = match[1]


async def process_request(
    request: Request,
    response: StreamingResponse,
    response_class: Type[
        Union[
            ChatCompletion,
            CompletionExt,
            CreateEmbeddingResponseExt,
            RerankResponse,
            ImagesResponse,
            FileResponse,
            Transcription,
        ]
    ],
    operation: OperationEnum,
):
    stream: bool = getattr(request.state, "stream", False)
    if stream:
        if response_class == ChatCompletion:
            response_class = ChatCompletionChunk
        if response_class == ImagesResponse:
            response_class = ImageGenerationChunk
        return await handle_streaming_response(
            request, response, response_class, operation
        )
    else:
        response_body = b"".join([chunk async for chunk in response.body_iterator])
        try:
            usage = None
            if (
                response.headers.get("content-type")
                .lower()
                .startswith("application/json")
            ):
                response_dict = json.loads(response_body)
                _capture_pd_coordinator_id(request, response_dict)
                response_instance = response_class(**response_dict)
                if hasattr(response_instance, "usage"):
                    usage = response_instance.usage

            await record_model_usage(request, usage, operation)
        except Exception as e:
            # Parser and validation errors can include the model response.
            logger.error("Error processing model usage (%s)", type(e).__name__)
        response = Response(
            content=response_body,
            status_code=response.status_code,
            headers=dict(response.headers),
        )

    return response


async def _resolve_direct_consumer_org(
    request: Request, user: User, raw_header: str
) -> Union[str, None]:
    """Validate a cookie-authed request's ``X-Organization-Id`` before it can
    become ``consumer_principal_id`` on a usage row.

    The direct (non-gateway) inference path has no api_key, so the raw client
    header would otherwise flow straight into ``consumer_principal_id`` — and
    a spoofed / stale id (a principal that doesn't exist) would violate the FK
    to ``principals`` and roll back the entire usage flush batch, dropping
    every user's buffered usage for that window.

    Resolve it through the same ``get_tenant_context`` the API routes use: it
    validates existence + membership (and applies api_key precedence), and its
    ``current_principal_id`` is the validated active principal. Return it as a
    string, or ``None`` on any failure (bad id / not a member) so the collector
    falls back to attributing the usage to the caller themselves. Best-effort —
    attribution must never break usage recording.
    """
    from gpustack.api.tenant import resolve_tenant_context
    from gpustack.server.db import async_session

    try:
        async with async_session() as session:
            ctx = await resolve_tenant_context(
                request, user, x_organization_id=raw_header, session=session
            )
        cpid = ctx.current_principal_id
        return str(cpid) if cpid is not None else None
    except Exception as e:
        logger.debug(f"Ignoring unresolved X-Organization-Id for usage: {e}")
        return None


async def record_model_usage(
    request: Request,
    usage: Union[CompletionUsage, EmbeddingUsage, RerankUsage, None],
    operation: OperationEnum,
):
    total_tokens = getattr(usage, 'total_tokens', 0) or 0
    prompt_tokens = getattr(usage, 'prompt_tokens', total_tokens) or total_tokens
    completion_tokens = (
        getattr(usage, 'completion_tokens', total_tokens - prompt_tokens)
        or total_tokens - prompt_tokens
    )
    prompt_token_details = (
        getattr(usage, "prompt_tokens_details", None) if usage else None
    )
    input_cached_tokens = 0
    if prompt_token_details:
        if isinstance(prompt_token_details, dict):
            input_cached_tokens = prompt_token_details.get("cached_tokens", 0) or 0
        else:
            input_cached_tokens = getattr(prompt_token_details, "cached_tokens", 0) or 0

    user: User = request.state.user
    model: Model = request.state.model
    api_key: ApiKey | None = getattr(request.state, "api_key", None)

    # Active tenant of this call. An api_key pins the consumer to its owner
    # (resolved downstream from ``api_key.owner_principal_id``), so it takes
    # precedence — when a key is present we ignore the header entirely. That
    # also means the gateway path never depends on this: its inference is
    # api_key-authed, so ``organization_id`` is ignored downstream and needs no
    # guarding here. Only the direct (cookie-authed) path — the Playground,
    # whose fetch wrapper attaches ``X-Organization-Id`` — reaches the header
    # branch below, and there the value is client-controlled. Validate it
    # before it becomes ``consumer_principal_id`` (which FKs ``principals``): an
    # unvalidated id straight from the header would let a spoofed / stale value
    # violate that FK and roll back the whole usage flush batch.
    organization_id = None
    if api_key is None:
        raw_organization_id = request.headers.get("x-organization-id")
        if raw_organization_id:
            organization_id = await _resolve_direct_consumer_org(
                request, user, raw_organization_id
            )

    # Reaching this function means the canonical usage chunk was observed,
    # so the report is ``completed=True``. Wall-clock anchors come from
    # RequestTimeMiddleware (start) and now (completion); the unified
    # flusher uses ``completed_at`` to choose the billing period.
    now = datetime.now(timezone.utc)
    started_at = getattr(request.state, "start_time", None)
    first_response_at = getattr(request.state, "first_token_time", None)
    ttft_ms = None
    if started_at is not None and first_response_at is not None:
        ttft_ms = max(0, int((first_response_at - started_at).total_seconds() * 1000))
    access = getattr(request.state, "inference_access", None)
    if started_at is None:
        # Falling back to ``now`` means request duration collapses to ~0,
        # which silently breaks SLO/latency analytics built off of
        # (completed_at - started_at). Surface the misconfiguration once
        # without flooding the log on every request.
        _warn_about_missing_start_time()
        started_at = now
    metric = ModelUsageMetrics(
        model=model.name,
        input_token=prompt_tokens,
        output_token=completion_tokens,
        total_token=total_tokens,
        input_cached_token=input_cached_tokens,
        request_count=1,
        completed=True,
        started_at=int(started_at.timestamp() * 1000),
        completed_at=int(now.timestamp() * 1000),
        ttft_ms=ttft_ms,
        request_id=access["request_id"] if access is not None else None,
        user_id=user.id if user is not None else None,
        model_id=model.id,
        model_route_id=getattr(request.state, "model_route_id", None),
        # Capture cluster_id at request time so it survives a later model
        # delete; the unified flusher prefers this over re-reading the
        # live model row.
        cluster_id=getattr(model, "cluster_id", None),
        access_key=api_key.access_key if api_key is not None else None,
        operation=operation,
        organization_id=organization_id,
    )
    await accumulate_gateway_metrics([metric])


async def handle_streaming_response(
    request: Request,
    response: StreamingResponse,
    response_class: Type[
        Union[ChatCompletionChunk, CompletionExt, ImageGenerationChunk]
    ],
    operation: OperationEnum,
):
    async def streaming_generator():
        async for chunk in response.body_iterator:
            try:
                async for processed_chunk in process_chunk(
                    chunk, request, response_class, operation
                ):
                    yield processed_chunk
            except Exception as e:
                logger.error(
                    "Error processing streaming response (%s)", type(e).__name__
                )
                yield chunk

    return StreamingResponse(streaming_generator(), headers=response.headers)


async def process_chunk(
    chunk,
    request,
    response_class,
    operation: OperationEnum,
):
    if not hasattr(request.state, 'first_token_time'):
        request.state.first_token_time = datetime.now(timezone.utc)

    # each chunk may contain multiple data lines
    lines = chunk.decode("utf-8").split("\n\n")
    for line in lines[:-1]:
        if not line.startswith('data: '):
            # skip non-data SSE messages
            yield f"{line}\n\n".encode("utf-8")
            continue

        data = line.split('data: ')[-1]
        if data.startswith('[DONE]'):
            yield "data: [DONE]\n\n".encode("utf-8")
            continue

        if '"usage":' in data:
            response_dict = json.loads(data)
            _capture_pd_coordinator_id(request, response_dict)
            response_chunk = response_class(**response_dict)

            if is_usage_chunk(response_chunk):
                await record_model_usage(request, response_chunk.usage, operation)

                # Fill rate metrics. These are extended info not included in OAI APIs.
                # llama-box provides them out-of-the-box. Align with other backends here.
                if should_add_metrics(response_dict):
                    add_metrics(response_dict, request, response_chunk)

            yield f"data: {json.dumps(response_dict, separators=(',', ':'))}\n\n".encode(
                "utf-8"
            )
        else:
            if "___prefill_addr_" in line:
                _capture_pd_coordinator_id(request, line.removeprefix("data: "))
            yield f"{line}\n\n".encode("utf-8")


def should_add_metrics(response_dict):
    if not isinstance(response_dict, dict):
        return False

    usage = response_dict.get('usage', {})

    return 'prompt_tokens' in usage and 'tokens_per_second' not in usage


def add_metrics(response_dict, request, response_chunk):
    now = datetime.now(timezone.utc)
    time_to_first_token_ms = (
        request.state.first_token_time - request.state.start_time
    ).total_seconds() * 1000

    tokens_after_first = max(response_chunk.usage.completion_tokens - 1, 1)
    time_per_output_token_ms = (
        (now - request.state.first_token_time).total_seconds()
        * 1000
        / tokens_after_first
    )

    tokens_per_second = (
        1000 / time_per_output_token_ms if time_per_output_token_ms > 0 else 0
    )

    response_dict['usage'].update(
        {
            "time_to_first_token_ms": time_to_first_token_ms,
            "time_per_output_token_ms": time_per_output_token_ms,
            "tokens_per_second": tokens_per_second,
        }
    )


class RefreshTokenMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        jwt_manager: JWTManager = request.app.state.jwt_manager
        token = request.cookies.get(SESSION_COOKIE_NAME)

        if token:
            try:
                payload = jwt_manager.decode_jwt_token(token)
                if payload:
                    # Check if the token is about to expire (less than 15 minutes left)
                    if payload['exp'] - time.time() < 15 * 60:
                        new_token = jwt_manager.create_jwt_token(
                            username=payload['sub']
                        )
                        response.set_cookie(
                            key=SESSION_COOKIE_NAME,
                            value=new_token,
                            httponly=True,
                            max_age=envs.JWT_TOKEN_EXPIRE_MINUTES * 60,
                            expires=envs.JWT_TOKEN_EXPIRE_MINUTES * 60,
                        )
            except (ExpiredSignatureError, DecodeError):
                pass

        return response


def is_usage_chunk(
    chunk: Union[ChatCompletionChunk, CompletionExt, ImageGenerationChunk],
) -> bool:
    choices = getattr(chunk, "choices", None)

    if not choices and chunk.usage:
        return True

    for choice in choices or []:
        if choice.finish_reason is not None and chunk.usage:
            return True

    return False
