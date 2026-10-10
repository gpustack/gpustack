import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiohttp
import pytest
from fastapi import FastAPI, WebSocket, WebSocketException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.websockets import WebSocketDisconnect

from gpustack.api import auth
from gpustack.routes.worker import proxy as http_proxy
from gpustack.routes.worker import websocket_proxy


class Upstream:
    def __init__(self, protocol=None):
        self.protocol = protocol
        self.close_code = None
        self.messages = asyncio.Queue()
        self.sent = []
        self.closes = []
        self.receive_cancelled = False
        self.receiving = asyncio.Event()

    async def receive(self):
        self.receiving.set()
        try:
            return await self.messages.get()
        except asyncio.CancelledError:
            self.receive_cancelled = True
            raise

    async def send_str(self, value):
        self.sent.append(("text", value))
        await self.messages.put(aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, value, ""))

    async def send_bytes(self, value):
        self.sent.append(("bytes", value))
        await self.messages.put(aiohttp.WSMessage(aiohttp.WSMsgType.BINARY, value, ""))

    async def close(self, **kwargs):
        self.closes.append(kwargs)

    def exception(self):
        return aiohttp.ClientConnectionError("connection lost")


class Downstream:
    def __init__(self):
        self.messages = asyncio.Queue()
        self.sent = asyncio.Queue()
        self.task = None
        self.receive_cancelled = False

    async def receive(self):
        try:
            return await self.messages.get()
        except asyncio.CancelledError:
            self.receive_cancelled = True
            raise

    async def next_message(self):
        return await asyncio.wait_for(self.sent.get(), timeout=1)

    async def finish(self):
        await asyncio.wait_for(self.task, timeout=1)


@asynccontextmanager
async def connection(
    monkeypatch,
    upstream=None,
    *,
    path="/v1/realtime",
    credentials=None,
    destination="model-1-7.static",
    protocols=(),
    connect_error=None,
    proxy_env=False,
    denial_response=False,
    worker_ip="127.0.0.1",
    early_messages=None,
):
    upstream = upstream or Upstream()
    connect = AsyncMock(return_value=upstream, side_effect=connect_error)
    session = SimpleNamespace(ws_connect=connect)
    app = FastAPI()
    app.include_router(http_proxy.router)
    app.include_router(websocket_proxy.router)
    app.add_middleware(BaseHTTPMiddleware, dispatch=http_proxy.set_port_from_model_name)
    app.state.config = SimpleNamespace(
        token="registration-token", get_server_url=lambda: "http://server"
    )
    app.state.token = "worker-token"
    app.state.worker_ip_getter = lambda: worker_ip
    app.state.get_instance_port_by_model_instance_id = lambda id: (
        40123 if id == 7 else None
    )
    app.state.http_client = session if proxy_env else SimpleNamespace()
    app.state.http_client_no_proxy = session if not proxy_env else SimpleNamespace()
    monkeypatch.setattr(websocket_proxy, "use_proxy_env_for_url", lambda _: proxy_env)

    headers = {
        "host": "worker",
        "connection": "Upgrade, x-hop",
        "upgrade": "websocket",
        "sec-websocket-key": "downstream-key",
        "sec-websocket-version": "13",
        "sec-websocket-extensions": "permessage-deflate",
        "origin": "https://client.example",
        "x-hop": "hop-value",
        "x-extra": "end-to-end",
        **(
            {"authorization": "Bearer worker-token"}
            if credentials is None
            else credentials
        ),
    }
    if destination is not None:
        headers["x-gpustack-model-instance"] = destination
    scope = {
        "type": "websocket",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "scheme": "ws",
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "query_string": b"model=m&token=private-query",
        "headers": [(key.encode(), value.encode()) for key, value in headers.items()],
        "subprotocols": list(protocols),
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 80),
    }
    if denial_response:
        scope["extensions"] = {"websocket.http.response": {}}
    downstream = Downstream()
    await downstream.messages.put({"type": "websocket.connect"})
    for message in early_messages or []:
        await downstream.messages.put(message)
    downstream.task = asyncio.create_task(
        app(scope, downstream.receive, downstream.sent.put)
    )
    try:
        yield downstream, upstream, connect
    finally:
        if not downstream.task.done():
            downstream.task.cancel()
        await asyncio.gather(downstream.task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/v1/realtime", "/proxy/v1/realtime"])
@pytest.mark.parametrize("proxy_env", [True, False])
async def test_handshake_preserves_routing_query_protocol_and_headers(
    monkeypatch, path, proxy_env
):
    async with connection(
        monkeypatch,
        Upstream(protocol="realtime"),
        path=path,
        protocols=("realtime", "other"),
        proxy_env=proxy_env,
    ) as (downstream, _, connect):
        assert (await downstream.next_message())["subprotocol"] == "realtime"
        args, kwargs = connect.call_args
        assert args == ("ws://127.0.0.1:40123/v1/realtime?model=m&token=private-query",)
        assert kwargs["protocols"] == ("realtime", "other")
        assert kwargs["autoclose"] is False
        assert isinstance(kwargs["timeout"], aiohttp.ClientWSTimeout)
        assert kwargs["headers"] == {
            "origin": "https://client.example",
            "x-extra": "end-to-end",
            "authorization": "Bearer worker-token",
            "x-gpustack-model-instance": "model-1-7.static",
        }


@pytest.mark.asyncio
async def test_handshake_brackets_ipv6_worker_address(monkeypatch):
    async with connection(monkeypatch, worker_ip="2001:db8::1") as (
        downstream,
        _,
        connect,
    ):
        assert (await downstream.next_message())["type"] == "websocket.accept"
        assert (
            connect.call_args.args[0]
            == "ws://[2001:db8::1]:40123/v1/realtime?model=m&token=private-query"
        )


@pytest.mark.asyncio
async def test_frame_received_while_handshake_is_pending_is_relayed(monkeypatch):
    async with connection(
        monkeypatch,
        early_messages=[{"type": "websocket.receive", "text": "early"}],
    ) as (downstream, upstream, _):
        assert (await downstream.next_message())["type"] == "websocket.accept"
        assert await downstream.next_message() == {
            "type": "websocket.send",
            "text": "early",
        }
        assert upstream.sent == [("text", "early")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,value", [("text", "hello"), ("bytes", b"\0\1"), ("text", ""), ("bytes", b"")]
)
async def test_frames_relay_in_both_directions(monkeypatch, kind, value):
    async with connection(monkeypatch) as (downstream, upstream, _):
        assert (await downstream.next_message())["type"] == "websocket.accept"
        await downstream.messages.put({"type": "websocket.receive", kind: value})
        assert await downstream.next_message() == {
            "type": "websocket.send",
            kind: value,
        }
        assert upstream.sent == [(kind, value)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "credentials",
    [{}, {"authorization": "Bearer wrong"}, {"authorization": "Basic wrong"}],
)
async def test_unauthenticated_handshake_never_connects_upstream(
    monkeypatch, credentials
):
    async with connection(monkeypatch, credentials=credentials) as (
        downstream,
        _,
        connect,
    ):
        assert (await downstream.next_message())["code"] == 1008
        await downstream.finish()
        connect.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "credentials",
    [
        {"authorization": "Bearer registration-token"},
        {"x-api-key": "worker-token"},
        {"authorization": "Bearer", "x-api-key": "worker-token"},
    ],
)
async def test_worker_credentials_match_http_auth(monkeypatch, credentials):
    async with connection(monkeypatch, credentials=credentials) as (downstream, _, _):
        assert (await downstream.next_message())["type"] == "websocket.accept"


@pytest.mark.asyncio
@pytest.mark.parametrize("valid", [True, False])
async def test_model_credential_uses_server_auth_before_connect(monkeypatch, valid):
    validate = AsyncMock(return_value=valid)
    monkeypatch.setattr(auth, "make_auth_token_via_server", lambda _: validate)
    async with connection(
        monkeypatch,
        credentials={
            "authorization": "Bearer model-token",
            "x-higress-llm-model": "allowed-model",
        },
    ) as (downstream, _, connect):
        message = await downstream.next_message()
        assert message["type"] == ("websocket.accept" if valid else "websocket.close")
        validate.assert_awaited_once_with(
            "http://server", "model-token", "allowed-model"
        )
        if not valid:
            await downstream.finish()
            connect.assert_not_called()


@pytest.mark.asyncio
async def test_model_credential_auth_server_failure_is_a_clean_rejection(monkeypatch):
    async def unavailable(*args, **kwargs):
        raise aiohttp.ClientConnectionError("auth server unavailable")

    monkeypatch.setattr(auth, "make_auth_token_via_server", lambda _: unavailable)
    async with connection(
        monkeypatch,
        credentials={
            "authorization": "Bearer model-token",
            "x-higress-llm-model": "allowed-model",
        },
    ) as (downstream, _, connect):
        assert (await downstream.next_message())["code"] == 1008
        await downstream.finish()
        connect.assert_not_called()


@pytest.mark.asyncio
async def test_model_credential_auth_timeout_is_a_clean_rejection(monkeypatch):
    async def unavailable(*args, **kwargs):
        raise asyncio.TimeoutError()

    monkeypatch.setattr(auth, "make_auth_token_via_server", lambda _: unavailable)
    async with connection(
        monkeypatch,
        credentials={
            "authorization": "Bearer model-token",
            "x-higress-llm-model": "allowed-model",
        },
    ) as (downstream, _, connect):
        assert (await downstream.next_message())["code"] == 1008
        await downstream.finish()
        connect.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("destination", [None, "invalid", "model-1-99.static"])
async def test_missing_or_nonrunning_instance_does_not_connect(
    monkeypatch, destination
):
    async with connection(monkeypatch, destination=destination) as (
        downstream,
        _,
        connect,
    ):
        assert (await downstream.next_message())["code"] == 1008
        await downstream.finish()
        connect.assert_not_called()


@pytest.mark.asyncio
async def test_upstream_failure_does_not_accept_or_log_credentials(monkeypatch, caplog):
    async with connection(
        monkeypatch,
        connect_error=aiohttp.ClientConnectionError("ws://backend?token=private-query"),
    ) as (downstream, _, _):
        assert (await downstream.next_message())["code"] == 1011
        await downstream.finish()
    assert "private-query" not in caplog.text
    assert "worker-token" not in caplog.text


@pytest.mark.asyncio
async def test_client_close_reaches_upstream_and_cancels_receiver(monkeypatch):
    async with connection(monkeypatch) as (downstream, upstream, _):
        await downstream.next_message()
        await upstream.receiving.wait()
        await downstream.messages.put(
            {"type": "websocket.disconnect", "code": 3001, "reason": "done"}
        )
        await downstream.finish()
        assert upstream.closes == [{"code": 3001, "message": b"done"}]
        assert upstream.receive_cancelled


@pytest.mark.asyncio
async def test_close_reason_is_limited_to_rfc6455_payload_size(monkeypatch):
    reason = "🙂" * 100
    async with connection(monkeypatch) as (downstream, upstream, _):
        await downstream.next_message()
        await upstream.receiving.wait()
        await downstream.messages.put(
            {"type": "websocket.disconnect", "code": 3001, "reason": reason}
        )
        await downstream.finish()

    assert upstream.closes[0]["code"] == 3001
    assert len(upstream.closes[0]["message"]) <= 123


@pytest.mark.asyncio
async def test_backend_close_reaches_client_with_code_and_reason(monkeypatch):
    async with connection(monkeypatch) as (downstream, upstream, _):
        await downstream.next_message()
        await upstream.messages.put(
            aiohttp.WSMessage(aiohttp.WSMsgType.CLOSE, 3002, "backend done")
        )
        assert await downstream.next_message() == {
            "type": "websocket.close",
            "code": 3002,
            "reason": "backend done",
        }
        await downstream.finish()
        assert upstream.closes == [{"code": 3002, "message": b"backend done"}]


@pytest.mark.asyncio
async def test_shutdown_closes_upstream_and_drains_relay_tasks(monkeypatch):
    async with connection(monkeypatch) as (downstream, upstream, _):
        await downstream.next_message()
        await upstream.receiving.wait()
        downstream.task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await downstream.finish()
        assert upstream.receive_cancelled
        assert upstream.closes == [{"code": 1001, "message": b"Worker shutting down"}]


@pytest.mark.asyncio
async def test_abnormal_upstream_close_uses_a_valid_wire_code(monkeypatch):
    async with connection(monkeypatch) as (downstream, upstream, _):
        await downstream.next_message()
        upstream.close_code = 1006
        await upstream.messages.put(
            aiohttp.WSMessage(aiohttp.WSMsgType.CLOSED, None, None)
        )
        assert (await downstream.next_message())["code"] == 1011
        await downstream.finish()


@pytest.mark.asyncio
async def test_upstream_receive_error_closes_client(monkeypatch):
    async with connection(monkeypatch) as (downstream, upstream, _):
        await downstream.next_message()
        await upstream.messages.put(
            aiohttp.WSMessage(aiohttp.WSMsgType.ERROR, None, None)
        )
        assert (await downstream.next_message())["code"] == 1011
        await downstream.finish()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200, 401, 404, 500])
async def test_upstream_handshake_denial_preserves_http_error(monkeypatch, status):
    error = aiohttp.WSServerHandshakeError(None, (), status=status)
    async with connection(monkeypatch, connect_error=error, denial_response=True) as (
        downstream,
        _,
        _,
    ):
        message = await downstream.next_message()
        assert message["type"] == "websocket.http.response.start"
        assert message["status"] == (status if status >= 400 else 502)
        assert (await downstream.next_message())[
            "body"
        ] == b"Upstream WebSocket unavailable"
        await downstream.finish()


@pytest.mark.asyncio
async def test_upstream_denial_falls_back_when_starlette_has_no_denial_response():
    websocket = SimpleNamespace(
        scope={"extensions": {"websocket.http.response": {}}},
        close=AsyncMock(),
    )

    await websocket_proxy._deny_upstream_handshake(websocket, 503)

    websocket.close.assert_awaited_once_with(
        code=1011, reason="Upstream WebSocket unavailable"
    )


@pytest.mark.asyncio
async def test_handshake_timeout_cancels_upstream_connect(monkeypatch):
    cancelled = asyncio.Event()

    async def stalled(*args, **kwargs):
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    monkeypatch.setattr(websocket_proxy.envs, "PROXY_TTFT_TIMEOUT", 0.01)
    async with connection(monkeypatch, connect_error=stalled) as (downstream, _, _):
        assert (await downstream.next_message())["code"] == 1011
        await downstream.finish()
        assert cancelled.is_set()


@pytest.mark.asyncio
async def test_disconnect_during_handshake_cancels_connect(monkeypatch):
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def stalled(*args, **kwargs):
        started.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    async with connection(monkeypatch, connect_error=stalled) as (downstream, _, _):
        await asyncio.wait_for(started.wait(), 1)
        await downstream.messages.put({"type": "websocket.disconnect", "code": 1000})
        await downstream.finish()
        assert cancelled.is_set()
        assert downstream.sent.empty()


@pytest.mark.asyncio
async def test_completed_handshake_racing_a_disconnect_is_closed():
    from fastapi import WebSocket

    upstream = Upstream()
    messages = asyncio.Queue()
    await messages.put({"type": "websocket.connect"})
    await messages.put({"type": "websocket.disconnect", "code": 1000})
    websocket = WebSocket({"type": "websocket"}, messages.get, AsyncMock())
    await websocket.receive()

    async def handshake():
        return upstream

    with pytest.raises(WebSocketDisconnect):
        await websocket_proxy._connect_upstream(websocket, handshake())
    assert len(upstream.closes) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "messages,byte_limit,frame_limit",
    [
        (
            [
                {"type": "websocket.receive", "text": "ééé"},
                {"type": "websocket.receive", "bytes": b"abc"},
            ],
            8,
            64,
        ),
        ([{"type": "websocket.receive", "bytes": b""}] * 4, 1024, 3),
    ],
)
async def test_handshake_buffer_overflow_rejects_and_cancels_connect(
    monkeypatch, messages, byte_limit, frame_limit
):
    monkeypatch.setattr(
        websocket_proxy, "_MAX_HANDSHAKE_BYTES", byte_limit, raising=False
    )
    monkeypatch.setattr(
        websocket_proxy, "_MAX_HANDSHAKE_MESSAGES", frame_limit, raising=False
    )
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def stalled(*args, **kwargs):
        started.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    async with connection(monkeypatch, connect_error=stalled) as (downstream, _, _):
        await asyncio.wait_for(started.wait(), 1)
        for message in messages:
            await downstream.messages.put(message)
        assert (await downstream.next_message()) == {
            "type": "websocket.close",
            "code": 1009,
            "reason": "Too much data before upstream handshake",
        }
        await downstream.finish()
        assert cancelled.is_set()


@pytest.mark.asyncio
async def test_buffer_overflow_closes_handshake_that_completes_at_the_same_time(
    monkeypatch,
):
    monkeypatch.setattr(websocket_proxy, "_MAX_HANDSHAKE_BYTES", 1, raising=False)
    messages = asyncio.Queue()
    await messages.put({"type": "websocket.connect"})
    await messages.put({"type": "websocket.receive", "bytes": b"ab"})
    websocket = WebSocket({"type": "websocket"}, messages.get, AsyncMock())
    await websocket.receive()
    upstream = Upstream()

    async def handshake():
        return upstream

    with pytest.raises(WebSocketException) as error:
        await websocket_proxy._connect_upstream(websocket, handshake())
    assert error.value.code == 1009
    assert len(upstream.closes) == 1


@pytest.mark.asyncio
async def test_slow_handshake_preserves_mixed_frames_at_buffer_limits(monkeypatch):
    monkeypatch.setattr(websocket_proxy, "_MAX_HANDSHAKE_BYTES", 8, raising=False)
    monkeypatch.setattr(websocket_proxy, "_MAX_HANDSHAKE_MESSAGES", 3, raising=False)
    messages = asyncio.Queue()
    await messages.put({"type": "websocket.connect"})
    early = [
        {"type": "websocket.receive", "text": "ééé"},
        {"type": "websocket.receive", "bytes": b"xy"},
        {"type": "websocket.receive", "text": ""},
    ]
    for message in early:
        await messages.put(message)
    consumed, release, waiting = asyncio.Event(), asyncio.Event(), asyncio.Event()
    reads = 0

    async def receive():
        nonlocal reads
        reads += 1
        if reads == 5:
            consumed.set()
        try:
            return await messages.get()
        except asyncio.CancelledError:
            waiting.set()
            raise

    websocket = WebSocket({"type": "websocket"}, receive, AsyncMock())
    await websocket.receive()
    upstream = Upstream()

    async def handshake():
        await release.wait()
        return upstream

    task = asyncio.create_task(
        websocket_proxy._connect_upstream(websocket, handshake())
    )
    try:
        await asyncio.wait_for(consumed.wait(), 1)
        release.set()
        response, pending = await asyncio.wait_for(task, 1)
        assert response is upstream
        assert pending == early
        assert waiting.is_set()
        assert upstream.closes == []
        await messages.put({"type": "websocket.receive", "bytes": b"later"})
        await messages.put({"type": "websocket.disconnect", "code": 1000})
        assert await websocket_proxy._from_client(websocket, upstream, pending) == (
            1000,
            "",
        )
        assert upstream.sent == [
            ("text", "ééé"),
            ("bytes", b"xy"),
            ("text", ""),
            ("bytes", b"later"),
        ]
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_handshake_cancel_after_buffering_drains_both_tasks(monkeypatch):
    consumed, cancelled, reader_cancelled = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
    reads = 0

    async def receive():
        nonlocal reads
        reads += 1
        if reads == 1:
            return {"type": "websocket.connect"}
        if reads == 2:
            return {"type": "websocket.receive", "text": "early"}
        consumed.set()
        try:
            await asyncio.Future()
        finally:
            reader_cancelled.set()

    async def handshake():
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    websocket = WebSocket({"type": "websocket"}, receive, AsyncMock())
    await websocket.receive()
    task = asyncio.create_task(
        websocket_proxy._connect_upstream(websocket, handshake())
    )
    try:
        await asyncio.wait_for(consumed.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
        assert cancelled.is_set() and reader_cancelled.is_set()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("disconnect", [False, True])
async def test_handshake_preserves_receive_that_wins_watcher_cancellation(disconnect):
    waiting = asyncio.Event()
    message = (
        {"type": "websocket.disconnect", "code": 1000}
        if disconnect
        else {"type": "websocket.receive", "text": "racing-frame"}
    )

    async def receive():
        waiting.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            return message

    upstream = Upstream()

    async def handshake():
        await waiting.wait()
        return upstream

    websocket = SimpleNamespace(receive=receive)
    if disconnect:
        with pytest.raises(WebSocketDisconnect):
            await websocket_proxy._connect_upstream(websocket, handshake())
        assert len(upstream.closes) == 1
    else:
        result, pending = await websocket_proxy._connect_upstream(
            websocket, handshake()
        )
        assert result is upstream
        assert pending == [message]
        assert upstream.closes == []
