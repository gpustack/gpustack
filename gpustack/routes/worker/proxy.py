import asyncio
import logging
import aiohttp
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Coroutine,
    List,
    Optional,
    Tuple,
    TypeVar,
)
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse, JSONResponse
from starlette.background import BackgroundTask
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

from gpustack.api.auth import worker_proxy_auth
from gpustack.api.exceptions import (
    GatewayTimeoutException,
    ServiceUnavailableException,
    NotFoundException,
    ErrorResponse,
)
from gpustack import envs
from gpustack.observability.tracing import (
    PUBLIC_INFERENCE_PATHS,
    current_inference_span,
    inference_request_id,
    inject_trace_context,
    set_span_error,
    trace_inference_http,
    trace_log_fields,
)
from gpustack.utils.network import use_proxy_env_for_url
from gpustack.utils.stall import stall_error_sse
from gpustack.gateway.utils import router_header_key
from gpustack.utils.model_instance_workers import parse_model_instance_target

router = APIRouter(dependencies=[Depends(worker_proxy_auth)])

logger = logging.getLogger(__name__)

T = TypeVar("T")

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


class WorkerProxyTracingMiddleware:
    """Trace routed inference across Worker lookup, auth and response relay."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope["method"] != "POST"
            or not scope["path"].startswith(("/v1/", "/proxy/v1/"))
        ):
            await self.app(scope, receive, send)
            return
        tracer = getattr(scope["app"].state, "inference_tracer", None)
        headers = Headers(scope=scope)
        if tracer is None or router_header_key not in headers:
            await self.app(scope, receive, send)
            return
        attributes = {"http.request.method": "POST"}
        request_id = inference_request_id(headers)
        if request_id:
            attributes["gpustack.request_id"] = request_id
        await trace_inference_http(
            self.app, scope, receive, send, tracer, "POST inference proxy", attributes
        )


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
        if timeout is None:
            return await body_iter.__anext__()
        return await asyncio.wait_for(body_iter.__anext__(), timeout)
    except StopAsyncIteration:
        return None


async def _stream_body(
    body_iter: AsyncIterator[bytes],
    first_chunk: Optional[bytes],
    idle_timeout: Optional[float],
    *,
    log_context: Optional[dict] = None,
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
            if idle_timeout is None:
                # Non-streaming response: no idle bound applies, and nothing
                # meaningful can be appended to a half-delivered JSON body, so
                # keep the existing behavior of letting the connection break.
                yield await body_iter.__anext__()
            else:
                yield await asyncio.wait_for(body_iter.__anext__(), idle_timeout)
        except StopAsyncIteration:
            return
        except (asyncio.TimeoutError, TimeoutError):
            set_span_error(current_inference_span(), "timeout")
            logger.error(
                "Upstream stopped sending data for %ss mid-stream, "
                "terminating the response with an error event",
                idle_timeout,
                extra={**(log_context or {}), **trace_log_fields()},
            )
            yield stall_error_sse()
            return


class _ClientDisconnected(Exception):
    """The client went away before the upstream response started."""


async def _wait_for_client_disconnect(
    request: Request, body_consumed: asyncio.Event
) -> None:
    """Return once the ASGI server reports that the client has disconnected.

    Waits for ``body_consumed`` first: until the request body has been fully
    read, receiving here would steal body messages from the upstream request.
    """
    await body_consumed.wait()
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            return


async def _cancel_on_client_disconnect(
    request: Request,
    coro: Coroutine[Any, Any, T],
    body_consumed: asyncio.Event,
    discard: Callable[[T], None],
) -> T:
    """Await ``coro``, cancelling it if the client disconnects first.

    Until the downstream response starts, nothing else notices a disconnect:
    the ASGI server does not cancel the handler, and Starlette only watches for
    it once a StreamingResponse is running. A non-streaming upstream sends
    nothing until generation has finished, so without this the whole
    generation would run for a client that is gone. Cancelling ``coro`` closes
    the upstream connection, which is what makes the engine (e.g. vLLM) abort
    the request and free the GPU.

    ``discard`` releases a result that completed but can no longer be used.
    """

    def discard_if_completed(task: asyncio.Task[T]) -> None:
        if not task.cancelled() and task.exception() is None:
            discard(task.result())

    work = asyncio.create_task(coro)
    watcher = asyncio.create_task(_wait_for_client_disconnect(request, body_consumed))
    try:
        await asyncio.wait({work, watcher}, return_when=asyncio.FIRST_COMPLETED)
        if not watcher.done():
            watcher.cancel()
            # An outer cancellation must propagate even while the watcher stops.
            await asyncio.gather(watcher, return_exceptions=True)
            if watcher.cancelled():
                return work.result()

        # A successful watcher consumed the disconnect, including when it raced
        # cancellation. If listening failed, propagate that error instead.
        watcher.result()
        raise _ClientDisconnected()
    except BaseException:
        work.cancel()
        watcher.cancel()
        if work.done():
            discard_if_completed(work)
        else:
            # A second cancellation can interrupt the wait, but not result cleanup.
            work.add_done_callback(discard_if_completed)
        await asyncio.gather(work, watcher, return_exceptions=True)
        raise


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
    methods=["GET", "POST", "DELETE", "OPTIONS", "HEAD"],
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
        span = current_inference_span()
        inject_trace_context(headers, span)
        target_instance_id = getattr(request.state, "x_target_instance_id", None)
        log_context = {
            "request_id": inference_request_id(request.headers),
            "model_instance_id": target_instance_id,
        }
        if span.is_recording() and target_instance_id:
            span.set_attribute("gpustack.model_instance_id", int(target_instance_id))
            endpoint = getattr(request.state, "model_instance_endpoint", None)
            if endpoint is not None:
                span.set_attribute("gpustack.model_id", endpoint.model_id)
            if f"/{path}" in PUBLIC_INFERENCE_PATHS:
                span.set_attribute("http.route", f"/{path}")
        # Read the header while popping it: aiohttp sets its own framing, and
        # checking after the pop would never see a chunked upload.
        transfer_encoding = headers.pop("transfer-encoding", "")
        body_consumed = asyncio.Event()
        if transfer_encoding.lower() == "chunked":

            async def body_generator():
                async for chunk in request.stream():
                    yield chunk
                body_consumed.set()

            content = body_generator()
        else:
            content = await request.body()
            body_consumed.set()

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

        async def open_upstream():
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
            except Exception:
                # Any other read failure happens before the StreamingResponse (and
                # its background close) exists, so the connection must be closed
                # here or it leaks out of the pool.
                resp.close()
                raise
            except asyncio.CancelledError:
                # Cancellation (a client disconnect caught by
                # _cancel_on_client_disconnect, or a server shutdown) bypasses
                # ``except Exception``, and would leak the connection for the
                # same reason.
                resp.close()
                raise
            return resp, streaming, body_iter, first_chunk

        try:
            upstream = await _cancel_on_client_disconnect(
                request,
                open_upstream(),
                body_consumed,
                discard=lambda upstream: upstream[0].close(),
            )
        except _ClientDisconnected:
            logger.debug(
                f"Client disconnected before {url} responded; "
                "cancelled the upstream request"
            )
            # Nobody is listening; 499 is the conventional "client closed" code.
            return Response(status_code=499)
        resp, streaming, body_iter, first_chunk = upstream

        # Heuristic: treat a started response body as a successful inference
        # signal so the active health-check loop can skip this instance.
        # Recorded here rather than off the status code alone because an SSE 200
        # arrives before prefill, which would let a wedged instance report
        # itself healthy.
        if resp.status < 400 and target_instance_id:
            record_fn = getattr(request.app.state, "record_successful_inference", None)
            if record_fn:
                record_fn(int(target_instance_id))

        async def stream_response():
            try:
                async for chunk in _stream_body(
                    body_iter,
                    first_chunk,
                    envs.PROXY_STREAM_IDLE_TIMEOUT if streaming else None,
                    log_context=log_context,
                ):
                    yield chunk
            finally:
                # A failed or cancelled stream may never run the background task.
                resp.close()

        response = StreamingResponse(
            stream_response(),
            status_code=resp.status,
            # Also close when disconnect handling prevents iteration from starting.
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
    except HTTPException:
        raise
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
    try:
        target = parse_model_instance_target(request.headers.get(router_header_key, ""))
    except ValueError as exc:
        raise NotFoundException(message="Invalid model destination") from exc
    model_instance_id = target.instance_id
    endpoint = request.app.state.get_instance_endpoint_by_model_instance_id(
        model_instance_id,
        model_id=target.model_id,
        dp_rank=target.dp_rank,
        worker_id=target.worker_id,
    )
    if endpoint is None:
        raise NotFoundException(
            message=f"No running model instance found for model name: {model_instance_id}",
        )
    request.state.model_instance_endpoint = endpoint
    logger.debug(
        f"Found port {endpoint.port} from model instance id {model_instance_id}"
    )
    return endpoint.port, model_instance_id


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
    except (NotFoundException, ServiceUnavailableException) as e:
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
