import asyncio
import json
import logging
import aiohttp
from typing import Callable, List, Optional, Tuple
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from starlette.background import BackgroundTask

from gpustack.api.auth import worker_auth
from gpustack.api.exceptions import (
    BadRequestException,
    GatewayTimeoutException,
    ServiceUnavailableException,
    NotFoundException,
    ErrorResponse,
)
from gpustack import envs
from gpustack.utils.network import use_proxy_env_for_url
from gpustack.gateway.utils import get_instance_id_from_header, router_header_key
from gpustack.scheduler.meta_registry import QWEN3_TTS_ARCHITECTURE

router = APIRouter(dependencies=[Depends(worker_auth)])

logger = logging.getLogger(__name__)

# Strip hop-by-hop and server-regenerated headers before forwarding the upstream
# response; otherwise the ASGI server appends its own Server/Date/Content-Length
# and clients (e.g. aiohttp) reject the response with "Duplicate 'Server' header".
# Content-Encoding is dropped because the aiohttp ClientSession auto-decompresses
# the body, so the bytes we stream out are already decoded.
_EXCLUDED_RESPONSE_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "server",
        "date",
        "content-length",
        "content-encoding",
    }
)


def _filter_response_headers(resp_headers) -> List[Tuple[str, str]]:
    # Return a list of tuples (not a dict) so multi-value headers such as
    # Set-Cookie are preserved; aiohttp's CIMultiDictProxy emits one item per
    # occurrence.
    return [
        (k, v)
        for k, v in resp_headers.items()
        if k.lower() not in _EXCLUDED_RESPONSE_HEADERS
    ]


def _extract_requested_task_type(content: bytes, content_type: str) -> Optional[str]:
    if "application/json" not in content_type.lower() or not content:
        return None
    try:
        body = json.loads(content)
    except (ValueError, TypeError):
        return None
    requested = body.get("task_type") if isinstance(body, dict) else None
    return requested if isinstance(requested, str) else None


async def _validate_speech_task_type(request: Request, path: str):
    # A Qwen3-TTS variant is a single-task checkpoint; a mismatched task_type
    # crashes the whole vllm engine (issue #5351), so reject it before forwarding.
    if request.method != "POST" or not path.rstrip("/").endswith("audio/speech"):
        return
    model_instance_id = getattr(request.state, "x_target_instance_id", None)
    get_model = getattr(request.app.state, "get_model_by_model_instance_id", None)
    if model_instance_id is None or get_model is None:
        return
    model = get_model(model_instance_id)
    meta = (model.meta or {}) if model else {}
    if meta.get("architecture") != QWEN3_TTS_ARCHITECTURE:
        return
    supported = meta.get("task_type")
    if not supported:
        return

    requested = _extract_requested_task_type(
        await request.body(), request.headers.get("content-type", "")
    )
    if requested and requested.strip().lower() != supported.lower():
        raise BadRequestException(
            message=(
                f"Model '{model.name}' only supports task_type '{supported}', "
                f"but '{requested}' was requested. "
                f"Deploy the matching Qwen3-TTS model for that task."
            ),
            is_openai_exception=True,
        )


@router.api_route(
    "/proxy/{path:path}",
    methods=["GET", "POST", "OPTIONS", "HEAD"],
)
async def proxy(path: str, request: Request):  # noqa: C901
    worker_ip_getter: Callable[[], str] = request.app.state.worker_ip_getter
    if worker_ip_getter is None:
        worker_ip_getter = localhost_fallback
    target_service_port = getattr(request.state, "x_target_port", None)
    if not target_service_port:
        raise HTTPException(
            status_code=400,
            detail="Missing target port; ensure the request includes the routing header",
        )

    # Reject incompatible TTS requests before the try below, so they surface as
    # 400 instead of being swallowed into the catch-all 503.
    await _validate_speech_task_type(request, path)

    try:
        logger.debug(
            f"Proxying request to worker at port {target_service_port} for path: {path}"
        )
        url = f"http://{worker_ip_getter()}:{target_service_port}/{path}"
        if request.url.query:
            url = f"{url}?{request.url.query}"
        headers = dict(request.headers)
        headers.pop("host", None)
        headers.pop("transfer-encoding", None)
        if headers.get("transfer-encoding", "").lower() == "chunked":

            async def body_generator():
                async for chunk in request.stream():
                    yield chunk

            content = body_generator()
        else:
            content = await request.body()

        async def stream_response(resp):
            async for chunk in resp.content.iter_chunked(1024):
                yield chunk

        use_proxy_env = use_proxy_env_for_url(url)
        http_client: aiohttp.ClientSession = (
            request.app.state.http_client
            if use_proxy_env
            else request.app.state.http_client_no_proxy
        )
        timeout = aiohttp.ClientTimeout(total=envs.PROXY_TIMEOUT)
        resp = await http_client.request(
            method=request.method,
            url=url,
            headers=headers,
            data=content,
            timeout=timeout,
        )

        # Heuristic: treat a non-error HTTP status as a successful inference
        # signal so the active health-check loop can skip this instance.
        # For streaming responses the status is available before body
        # transfer, so a mid-stream failure will still be counted — this is
        # acceptable as a best-effort optimisation.
        target_instance_id = getattr(request.state, "x_target_instance_id", None)
        if resp.status < 400 and target_instance_id:
            record_fn = getattr(request.app.state, "record_successful_inference", None)
            if record_fn:
                record_fn(int(target_instance_id))

        response = StreamingResponse(
            stream_response(resp),
            status_code=resp.status,
            background=BackgroundTask(resp.close),
        )
        # Use append (not the headers= constructor kwarg) so duplicate header
        # names like Set-Cookie survive instead of being overwritten by
        # Starlette's MutableHeaders.update.
        for k, v in _filter_response_headers(resp.headers):
            response.headers.append(k, v)
        return response

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


def localhost_fallback() -> str:
    return "127.0.0.1"


def get_model_instance_info_from_model_name(
    request: Request,
) -> Tuple[int, int]:
    """
    Get model instance port and instance id from model name in header
    "x-gpustack-model-instance".

    Return (port, model_instance_id).
    """
    model_instance_id = get_instance_id_from_header(request.headers)
    port: Optional[int] = request.app.state.get_instance_port_by_model_instance_id(
        model_instance_id
    )
    if not port:
        raise NotFoundException(
            message=f"No running model instance found for model name: {model_instance_id}",
        )
    logger.debug(f"Found port {port} from model instance id {model_instance_id}")
    return port, model_instance_id


async def set_port_from_model_name(request: Request, call_next):
    model_name = request.headers.get(router_header_key, None)
    if model_name is None:
        return await call_next(request)
    try:
        port, model_instance_id = get_model_instance_info_from_model_name(request)
        request.scope["path"] = f"/proxy{request.url.path}"
        request.state.x_target_port = str(port)
        request.state.x_target_instance_id = model_instance_id
        return await call_next(request)
    except NotFoundException as e:
        logger.debug("failed to find model instance for proxying: %s", e.message)
        return JSONResponse(
            status_code=e.status_code,
            content=ErrorResponse(
                code=e.status_code,
                reason=e.reason,
                message=e.message,
            ).model_dump(),
        )
    except HTTPException as e:
        logger.debug("failed to find model instance for proxying: %s", e.detail)
        return await call_next(request)
