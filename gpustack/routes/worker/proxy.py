import asyncio
import logging
import aiohttp
from typing import AsyncIterator, Callable, List, Optional, Tuple
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from starlette.background import BackgroundTask

from gpustack.api.auth import worker_auth
from gpustack.api.exceptions import (
    GatewayTimeoutException,
    ServiceUnavailableException,
    NotFoundException,
    ErrorResponse,
)
from gpustack import envs
from gpustack.utils.network import use_proxy_env_for_url
from gpustack.utils.stall import stall_error_sse
from gpustack.gateway.utils import get_instance_id_from_header, router_header_key

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


_EVENT_STREAM_CONTENT_TYPE = "text/event-stream"


def _is_event_stream(content_type: Optional[str]) -> bool:
    return _EVENT_STREAM_CONTENT_TYPE in (content_type or "").lower()


async def _read_first_chunk(
    body_iter: AsyncIterator[bytes], timeout: Optional[float]
) -> Optional[bytes]:
    """Wait for the first body chunk, or None when the body is empty.

    ``timeout`` is None for a non-streaming response: it is only sent once
    generation has finished, so its first chunk *is* the end of generation and
    a tight budget here would kill a legitimate long completion. The absolute
    ceiling (``PROXY_TIMEOUT``) still applies via the aiohttp client timeout.

    An empty body (204/304, HEAD, ``content-length: 0``) reaches EOF right
    away, so it never waits out the budget.
    """
    try:
        return await asyncio.wait_for(body_iter.__anext__(), timeout)
    except StopAsyncIteration:
        return None


async def _stream_body(
    body_iter: AsyncIterator[bytes],
    first_chunk: Optional[bytes],
    idle_timeout: Optional[float],
) -> AsyncIterator[bytes]:
    """Stream the response body, bounding the gap between chunks.

    ``idle_timeout`` is only set for a streaming response, where a gap of tens
    of seconds means the upstream is wedged: a healthy engine emits tokens
    continuously. When the gap is exceeded the headers are long committed, so
    the stall is reported as a terminal in-stream error rather than by dropping
    the connection, which would look exactly like a truncated generation.

    Anything other than a timeout (upstream reset, for one) propagates as
    before.
    """
    if first_chunk is not None:
        yield first_chunk
    while True:
        try:
            yield await asyncio.wait_for(body_iter.__anext__(), idle_timeout)
        except StopAsyncIteration:
            return
        except (asyncio.TimeoutError, TimeoutError):
            if idle_timeout is None:
                # Non-streaming response: nothing meaningful can be appended to
                # a half-delivered JSON body, so keep the existing behavior of
                # letting the connection break.
                raise
            logger.error(
                f"Upstream stopped sending data for {idle_timeout}s mid-stream, "
                "terminating the response with an error event"
            )
            yield stall_error_sse()
            return


def _filter_response_headers(resp_headers) -> List[Tuple[str, str]]:
    # Return a list of tuples (not a dict) so multi-value headers such as
    # Set-Cookie are preserved; aiohttp's CIMultiDictProxy emits one item per
    # occurrence.
    return [
        (k, v)
        for k, v in resp_headers.items()
        if k.lower() not in _EXCLUDED_RESPONSE_HEADERS
    ]


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

        use_proxy_env = use_proxy_env_for_url(url)
        http_client: aiohttp.ClientSession = (
            request.app.state.http_client
            if use_proxy_env
            else request.app.state.http_client_no_proxy
        )
        # ``total`` remains the absolute ceiling. It is a bad failure detector on
        # its own, so what actually bounds a stalled upstream is the segmented
        # budget below. Note that ``sock_read`` cannot be used for it: it starts
        # counting when the request is sent and covers the wait for response
        # headers, which has to stay generous.
        timeout = aiohttp.ClientTimeout(total=envs.PROXY_TIMEOUT)
        resp = await http_client.request(
            method=request.method,
            url=url,
            headers=headers,
            data=content,
            timeout=timeout,
        )

        streaming = _is_event_stream(resp.headers.get("content-type"))
        body_iter = resp.content.iter_chunked(1024)

        # Hold the downstream headers until the first body chunk arrives. vLLM
        # answers an SSE request with 200 + text/event-stream *before* prefill,
        # so committing headers on arrival would fix the status code before we
        # can tell whether a first token is ever coming, leaving a pre-first-
        # token stall reportable only in-stream. Deferring turns it into a real
        # 504 that SDKs retry; for an SSE client the delay is invisible, since
        # it is already waiting for the first event.
        try:
            first_chunk = await _read_first_chunk(
                body_iter, envs.PROXY_TTFT_TIMEOUT if streaming else None
            )
        except (asyncio.TimeoutError, TimeoutError):
            # close(), not release(): data may still be in flight, so the
            # connection must not go back to the pool.
            resp.close()
            raise GatewayTimeoutException(
                message=(
                    f"Upstream {url} sent no response data within "
                    f"{envs.PROXY_TTFT_TIMEOUT}s"
                ),
                is_openai_exception=True,
            )

        # Heuristic: treat a started response body as a successful inference
        # signal so the active health-check loop can skip this instance.
        # Recorded here rather than off the status code alone because an SSE 200
        # arrives before prefill, which would let a wedged instance report
        # itself healthy.
        target_instance_id = getattr(request.state, "x_target_instance_id", None)
        if resp.status < 400 and target_instance_id:
            record_fn = getattr(request.app.state, "record_successful_inference", None)
            if record_fn:
                record_fn(int(target_instance_id))

        response = StreamingResponse(
            _stream_body(
                body_iter,
                first_chunk,
                envs.PROXY_STREAM_IDLE_TIMEOUT if streaming else None,
            ),
            status_code=resp.status,
            background=BackgroundTask(resp.close),
        )
        # Use append (not the headers= constructor kwarg) so duplicate header
        # names like Set-Cookie survive instead of being overwritten by
        # Starlette's MutableHeaders.update.
        for k, v in _filter_response_headers(resp.headers):
            response.headers.append(k, v)
        return response

    except GatewayTimeoutException:
        # Already carries the right status code and message; keep the catch-all
        # below from downgrading a detected stall to a 503.
        raise
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
