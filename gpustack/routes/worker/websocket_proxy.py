import asyncio
import logging
from typing import Coroutine, List, Optional, Tuple

import aiohttp
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    WebSocketException,
)
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.security.utils import get_authorization_scheme_param
from fastapi.responses import PlainTextResponse
from starlette.websockets import WebSocketState

from gpustack import envs
from gpustack.api.auth import worker_auth
from gpustack.api.exceptions import NotFoundException, UnauthorizedException
from gpustack.routes.worker.proxy import (
    get_model_instance_info_from_model_name,
    localhost_fallback,
)
from gpustack.utils.network import use_proxy_env_for_url

logger = logging.getLogger(__name__)
router = APIRouter()

_HANDSHAKE_HEADERS = frozenset(
    {
        "host",
        "connection",
        "keep-alive",
        "upgrade",
        "proxy-authorization",
        "proxy-authenticate",
        "te",
        "trailer",
        "transfer-encoding",
        "content-length",
        "sec-websocket-key",
        "sec-websocket-accept",
        "sec-websocket-version",
        "sec-websocket-protocol",
        "sec-websocket-extensions",
    }
)


async def websocket_worker_auth(websocket: WebSocket) -> None:
    """Apply the worker's credential checks without HTTP-only dependencies."""
    scheme, credentials = get_authorization_scheme_param(
        websocket.headers.get("authorization")
    )
    bearer = (
        HTTPAuthorizationCredentials(scheme=scheme, credentials=credentials)
        if scheme.lower() == "bearer" and credentials
        else None
    )
    try:
        await worker_auth(
            request=websocket,
            bearer_token=bearer,
            x_api_key=websocket.headers.get("x-api-key"),
        )
    except (
        UnauthorizedException,
        aiohttp.ClientError,
        asyncio.TimeoutError,
        OSError,
    ) as error:
        logger.debug("WebSocket worker authentication failed: %s", type(error).__name__)
        raise WebSocketException(code=1008, reason="Authentication failed")


def _request_headers(websocket: WebSocket) -> dict[str, str]:
    # Each hop negotiates its own WebSocket handshake; forwarding a key,
    # compression extension or a Connection-nominated header mixes the hops.
    excluded = _HANDSHAKE_HEADERS | {
        header.strip().lower()
        for header in websocket.headers.get("connection", "").split(",")
    }
    return {
        key: value
        for key, value in websocket.headers.items()
        if key.lower() not in excluded
    }


def _close_code(code: int) -> int:
    # 1005/1006/1015 describe absent/abnormal closes and cannot go on the wire.
    if code in (1005, 1006, 1015):
        return 1011
    return code or 1000


def _close_reason(reason: Optional[str]) -> str:
    """Fit a close reason into the 123-byte RFC 6455 control-frame limit."""
    if not reason:
        return ""
    return reason.encode("utf-8")[:123].decode("utf-8", errors="ignore")


def _format_host(host: str) -> str:
    """Format an IPv6 literal for a URL authority component."""
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


async def _connect_upstream(
    websocket: WebSocket, handshake: Coroutine
) -> Tuple[aiohttp.ClientWebSocketResponse, List[dict]]:
    """Cancel a pending handshake if the downstream client goes away."""
    connect_task = asyncio.create_task(handshake)
    pending_messages: List[dict] = []
    disconnect_task = asyncio.create_task(websocket.receive())
    try:
        while True:
            done, _ = await asyncio.wait(
                (connect_task, disconnect_task), return_when=asyncio.FIRST_COMPLETED
            )
            if connect_task in done:
                if disconnect_task in done:
                    message = disconnect_task.result()
                    if message["type"] == "websocket.disconnect":
                        raise WebSocketDisconnect(
                            message.get("code", 1006), message.get("reason", "")
                        )
                    pending_messages.append(message)
                else:
                    disconnect_task.cancel()
                    await asyncio.gather(disconnect_task, return_exceptions=True)
                return connect_task.result(), pending_messages

            message = disconnect_task.result()
            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect(
                    message.get("code", 1006), message.get("reason", "")
                )
            pending_messages.append(message)
            disconnect_task = asyncio.create_task(websocket.receive())
    except BaseException:
        # A successful handshake can race either a disconnect or cancellation.
        # Drain the task before inspecting it so a late result is closed too.
        if not connect_task.done():
            connect_task.cancel()
        await asyncio.gather(connect_task, return_exceptions=True)
        if not connect_task.cancelled() and connect_task.exception() is None:
            await connect_task.result().close()
        raise
    finally:
        disconnect_task.cancel()
        await asyncio.gather(disconnect_task, return_exceptions=True)


async def _from_client(
    websocket: WebSocket,
    upstream: aiohttp.ClientWebSocketResponse,
    pending_messages: List[dict],
) -> Tuple[int, str]:
    while True:
        message = (
            pending_messages.pop(0) if pending_messages else await websocket.receive()
        )
        if message["type"] == "websocket.disconnect":
            return _close_code(message.get("code", 1000)), message.get("reason", "")
        if message.get("text") is not None:
            await upstream.send_str(message["text"])
        elif message.get("bytes") is not None:
            await upstream.send_bytes(message["bytes"])


async def _from_upstream(
    websocket: WebSocket, upstream: aiohttp.ClientWebSocketResponse
) -> Tuple[int, str]:
    while True:
        # aiohttp's async iterator hides CLOSE frames; receive() keeps the code
        # and reason so a backend's deliberate close reaches the caller.
        message = await upstream.receive()
        if message.type == aiohttp.WSMsgType.TEXT:
            await websocket.send_text(message.data)
        elif message.type == aiohttp.WSMsgType.BINARY:
            await websocket.send_bytes(message.data)
        elif message.type == aiohttp.WSMsgType.CLOSE:
            return _close_code(message.data), message.extra or ""
        elif message.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING):
            return _close_code(upstream.close_code), ""
        elif message.type == aiohttp.WSMsgType.ERROR:
            raise upstream.exception() or aiohttp.ClientError(
                "WebSocket receive failed"
            )


async def _relay(
    websocket: WebSocket,
    upstream: aiohttp.ClientWebSocketResponse,
    pending_messages: List[dict],
) -> Tuple[int, str]:
    tasks = {
        asyncio.create_task(_from_client(websocket, upstream, pending_messages)),
        asyncio.create_task(_from_upstream(websocket, upstream)),
    }
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        return next(iter(done)).result()
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def _deny_upstream_handshake(websocket: WebSocket, status_code: int) -> None:
    send_denial_response = getattr(websocket, "send_denial_response", None)
    if "websocket.http.response" in websocket.scope.get("extensions", {}) and callable(
        send_denial_response
    ):
        await send_denial_response(
            PlainTextResponse("Upstream WebSocket unavailable", status_code=status_code)
        )
    else:
        await websocket.close(code=1011, reason="Upstream WebSocket unavailable")


@router.websocket("/{path:path}", dependencies=[Depends(websocket_worker_auth)])
async def proxy(path: str, websocket: WebSocket) -> None:
    """Relay a model WebSocket selected by the gateway's instance header."""
    try:
        port, instance_id = get_model_instance_info_from_model_name(websocket)
    except (HTTPException, NotFoundException):
        await websocket.close(code=1008, reason="No running model instance selected")
        return

    worker_ip_getter = websocket.app.state.worker_ip_getter or localhost_fallback
    target_path = path.removeprefix("proxy/")
    url = f"ws://{_format_host(worker_ip_getter())}:{port}/{target_path}"
    if websocket.url.query:
        url = f"{url}?{websocket.url.query}"
    client = (
        websocket.app.state.http_client
        if use_proxy_env_for_url(url)
        else websocket.app.state.http_client_no_proxy
    )
    protocols = tuple(websocket.scope.get("subprotocols", ()))
    # Consume the initial websocket.connect before watching for a disconnect.
    initial = await websocket.receive()
    if initial["type"] == "websocket.disconnect":
        return
    try:
        upstream, pending_messages = await _connect_upstream(
            websocket,
            asyncio.wait_for(
                client.ws_connect(
                    url,
                    headers=_request_headers(websocket),
                    protocols=protocols,
                    autoclose=False,
                    timeout=aiohttp.ClientWSTimeout(ws_close=5),
                ),
                timeout=envs.PROXY_TTFT_TIMEOUT,
            ),
        )
    except WebSocketDisconnect:
        return
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as error:
        # URLs can carry API credentials in their query; log identity and the
        # failure type, not the URL or an exception that embeds it.
        logger.warning(
            "Model instance %s WebSocket handshake failed (%s)",
            instance_id,
            type(error).__name__,
        )
        status_code = 503
        if isinstance(error, aiohttp.WSServerHandshakeError):
            status_code = error.status if error.status >= 400 else 502
        elif isinstance(error, asyncio.TimeoutError):
            status_code = 504
        await _deny_upstream_handshake(websocket, status_code)
        return

    code, reason = 1000, ""
    try:
        await websocket.accept(subprotocol=upstream.protocol)
        code, reason = await _relay(websocket, upstream, pending_messages)
    except WebSocketDisconnect as error:
        code, reason = _close_code(error.code), error.reason
    except asyncio.CancelledError:
        code, reason = 1001, "Worker shutting down"
        raise
    except (aiohttp.ClientError, OSError, RuntimeError) as error:
        code, reason = 1011, "WebSocket relay failed"
        logger.warning(
            "Model instance %s WebSocket relay failed (%s)",
            instance_id,
            type(error).__name__,
        )
    finally:
        reason = _close_reason(reason)
        try:
            await upstream.close(code=code, message=reason.encode("utf-8"))
        finally:
            if (
                websocket.application_state == WebSocketState.CONNECTED
                and websocket.client_state != WebSocketState.DISCONNECTED
            ):
                try:
                    await websocket.close(code=code, reason=reason)
                except (OSError, RuntimeError):
                    pass
