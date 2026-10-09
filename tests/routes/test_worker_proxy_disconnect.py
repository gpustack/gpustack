"""Verify disconnect propagation over loopback TCP with the real ASGI server."""

import asyncio
import socket
from contextlib import asynccontextmanager

import aiohttp
import pytest
import uvicorn
from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware

from gpustack.api import exceptions
from gpustack.api.auth import worker_auth
from gpustack.routes.worker import proxy


@asynccontextmanager
async def backend(mode):
    started = asyncio.Event()
    disconnected = asyncio.Event()
    handlers = set()

    async def handle(reader, writer):
        handlers.add(asyncio.current_task())
        try:
            headers = await reader.readuntil(b"\r\n\r\n")
            for line in headers.split(b"\r\n"):
                if line.lower().startswith(b"content-length:"):
                    await reader.readexactly(int(line.split(b":", 1)[1]))
            started.set()
            if mode in ("normal", "error"):
                status = b"200 OK" if mode == "normal" else b"503 Service Unavailable"
                writer.write(
                    b"HTTP/1.1 " + status + b"\r\nContent-Length: 2\r\n"
                    b"Set-Cookie: a=1\r\nSet-Cookie: b=2\r\n\r\nOK"
                )
                await writer.drain()
            else:
                if mode == "stream":
                    writer.write(
                        b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n"
                        b"Content-Type: text/event-stream\r\n\r\n"
                        b"9\r\ndata: x\n\n\r\n"
                    )
                    await writer.drain()
                await reader.read()
                disconnected.set()
        finally:
            writer.close()
            await writer.wait_closed()
            handlers.discard(asyncio.current_task())

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    try:
        yield server.sockets[0].getsockname()[1], started, disconnected
    finally:
        server.close()
        await server.wait_closed()
        for task in list(handlers):
            task.cancel()
        await asyncio.gather(*handlers, return_exceptions=True)


@asynccontextmanager
async def worker(backend_port):
    ready = asyncio.Event()

    class ReadyServer(uvicorn.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets=sockets)
            ready.set()

    app = FastAPI()
    app.include_router(proxy.router)
    exceptions.register_handlers(app)
    app.dependency_overrides[worker_auth] = lambda: None
    app.state.worker_ip_getter = lambda: "127.0.0.1"

    async def route(request, call_next):
        request.state.x_target_port = backend_port
        return await proxy.set_port_from_model_name(request, call_next)

    # Keep the middleware used by the real Worker in the reproduction.
    app.add_middleware(BaseHTTPMiddleware, dispatch=route)
    async with aiohttp.ClientSession() as session:
        app.state.http_client = session
        app.state.http_client_no_proxy = session
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            server = ReadyServer(
                uvicorn.Config(
                    app,
                    log_level="error",
                    access_log=False,
                    timeout_graceful_shutdown=1,
                )
            )
            task = asyncio.create_task(server.serve(sockets=[sock]))
            try:
                await asyncio.wait_for(ready.wait(), 5)
                yield port
            finally:
                server.should_exit = True
                await asyncio.wait_for(task, 5)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["headers", "stream"])
async def test_disconnect_closes_backend_connection(mode):
    async with backend(mode) as (backend_port, started, disconnected):
        async with worker(backend_port) as port:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            try:
                writer.write(
                    b"POST /proxy/v1/chat/completions HTTP/1.1\r\n"
                    b"Host: worker\r\nContent-Length: 2\r\n\r\n{}"
                )
                await writer.drain()
                await asyncio.wait_for(started.wait(), 2)
                if mode == "stream":
                    await asyncio.wait_for(reader.readuntil(b"data: x\n\n"), 2)
                writer.close()
                await writer.wait_closed()
                # A disconnected client must not leave model work running.
                await asyncio.wait_for(disconnected.wait(), 2)
            finally:
                writer.close()
                await writer.wait_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,status", [("normal", 200), ("error", 503)])
async def test_response_preserves_status_body_and_repeated_headers(mode, status):
    async with backend(mode) as (backend_port, _, _):
        async with worker(backend_port) as port:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"http://127.0.0.1:{port}/proxy/v1/chat/completions", json={}
                ) as response:
                    assert response.status == status
                    assert await response.read() == b"OK"
                    assert response.headers.getall("Set-Cookie") == ["a=1", "b=2"]


@pytest.mark.asyncio
async def test_backend_timeout_still_returns_504_and_closes_connection(monkeypatch):
    monkeypatch.setattr(proxy.envs, "PROXY_TIMEOUT", 0.1)
    async with backend("headers") as (backend_port, _, disconnected):
        async with worker(backend_port) as port:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"http://127.0.0.1:{port}/proxy/v1/chat/completions", json={}
                ) as response:
                    assert response.status == 504
                    assert "timed out" in await response.text()
                await asyncio.wait_for(disconnected.wait(), 2)
