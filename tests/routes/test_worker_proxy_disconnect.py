"""Verify Worker disconnect handling with ASGI messages and client-boundary mocks."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from multidict import CIMultiDict
from starlette.middleware.base import BaseHTTPMiddleware

from gpustack.api import exceptions
from gpustack.api.auth import worker_auth
from gpustack.routes.worker import proxy


def worker(client):
    app = FastAPI()
    app.include_router(proxy.router)
    exceptions.register_handlers(app)
    app.dependency_overrides[worker_auth] = lambda: None
    app.state.worker_ip_getter = lambda: "127.0.0.1"
    app.state.http_client = client
    app.state.http_client_no_proxy = client

    async def route(request, call_next):
        request.state.x_target_port = "12345"
        return await proxy.set_port_from_model_name(request, call_next)

    app.add_middleware(BaseHTTPMiddleware, dispatch=route)
    return app


async def call_worker(client, disconnect=None, first_chunk=None):
    body_sent = False
    disconnect = disconnect if disconnect is not None else asyncio.Event()
    messages = []

    async def receive():
        nonlocal body_sent
        if not body_sent:
            body_sent = True
            return {"type": "http.request", "body": b"{}", "more_body": False}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        messages.append(message)
        if first_chunk is not None and message.get("body"):
            first_chunk.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/proxy/v1/chat/completions",
        "query_string": b"",
        "headers": [(b"content-length", b"2")],
    }
    await asyncio.wait_for(worker(client)(scope, receive, send), 2)
    return messages


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["headers", "first_chunk", "stream"])
async def test_disconnect_cancels_upstream_work(mode):
    started = asyncio.Event()
    cancelled = asyncio.Event()
    disconnect = asyncio.Event()
    first_chunk = asyncio.Event()

    async def body():
        started.set()
        if mode == "stream":
            yield b"data: x\n\n"
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    upstream = SimpleNamespace(
        status=200,
        headers={"content-type": "text/event-stream"},
        content=SimpleNamespace(iter_chunked=lambda size: body()),
        close=Mock(),
    )

    async def request(**kwargs):
        if mode == "headers":
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        return upstream

    client = SimpleNamespace(request=AsyncMock(side_effect=request))
    task = asyncio.create_task(call_worker(client, disconnect, first_chunk))
    try:
        await asyncio.wait_for(started.wait(), 1)
        if mode == "stream":
            await asyncio.wait_for(first_chunk.wait(), 1)
        disconnect.set()
        await task
        assert cancelled.is_set()
        if mode != "headers":
            upstream.close.assert_called()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200, 503])
async def test_response_preserves_status_body_and_repeated_headers(status):
    async def body():
        yield b"OK"

    upstream = SimpleNamespace(
        status=status,
        headers=CIMultiDict([("Set-Cookie", "a=1"), ("Set-Cookie", "b=2")]),
        content=SimpleNamespace(iter_chunked=lambda size: body()),
        close=Mock(),
    )
    client = SimpleNamespace(request=AsyncMock(return_value=upstream))
    messages = await call_worker(client)
    response = next(m for m in messages if m["type"] == "http.response.start")
    assert response["status"] == status
    assert [v for k, v in response["headers"] if k == b"set-cookie"] == [b"a=1", b"b=2"]
    assert b"".join(m.get("body", b"") for m in messages) == b"OK"
    upstream.close.assert_called()


@pytest.mark.asyncio
async def test_backend_timeout_still_returns_504(monkeypatch):
    monkeypatch.setattr(proxy.envs, "PROXY_TIMEOUT", 0.01)
    cancelled = asyncio.Event()

    async def request(*, timeout, **kwargs):
        try:
            await asyncio.wait_for(asyncio.Event().wait(), timeout.total)
        finally:
            cancelled.set()

    client = SimpleNamespace(request=AsyncMock(side_effect=request))
    messages = await call_worker(client)
    response = next(m for m in messages if m["type"] == "http.response.start")
    assert response["status"] == 504
    assert b"timed out" in b"".join(m.get("body", b"") for m in messages)
    assert cancelled.is_set()
