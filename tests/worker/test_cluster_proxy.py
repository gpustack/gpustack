import gzip
import json
from contextlib import asynccontextmanager

import pytest
from aiohttp import web
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from gpustack.api.auth import worker_auth
from gpustack.routes.worker import cluster_proxy


@asynccontextmanager
async def _fake_apiserver(payload: bytes):
    """A stand-in for the kube-apiserver that gzips the response whenever the
    client sends Accept-Encoding: gzip. This isolates the encoding-negotiation
    behaviour the fix cares about; it deliberately does not model the real
    apiserver's ~128 KiB size threshold (which is only why the bug reproduces on
    large bodies in production, not something the proxy logic depends on).
    Yields the http base URL."""

    async def handler(request):
        if "gzip" in request.headers.get("Accept-Encoding", ""):
            return web.Response(
                body=gzip.compress(payload),
                headers={
                    "Content-Encoding": "gzip",
                    "Content-Type": "application/json",
                },
            )
        return web.Response(body=payload, content_type="application/json")

    app = web.Application()
    app.router.add_route("*", "/{path:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    # AppRunner.addresses is the public way to read the bound (host, port) after
    # binding to port 0; avoids depending on aiohttp internals.
    port = runner.addresses[0][1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        await runner.cleanup()


def _worker_app(monkeypatch, base_url: str) -> FastAPI:
    # Bypass real pod ServiceAccount credentials / TLS. ssl_ctx=None means the
    # connector does no TLS, which is correct for the http:// fake upstream.
    monkeypatch.setattr(cluster_proxy, "_resolve_kube_target", lambda: (base_url, None))
    monkeypatch.setattr(cluster_proxy, "_read_token", lambda: "test-token")

    app = FastAPI()
    app.include_router(cluster_proxy.router)
    app.dependency_overrides[worker_auth] = lambda: None
    return app


@asynccontextmanager
async def _worker_client(monkeypatch, base_url: str):
    app = _worker_app(monkeypatch, base_url)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://worker") as client:
        yield client
    # Close the cached upstream aiohttp session created by the route.
    session = getattr(app.state, "kube_api_session", None)
    if session is not None and not session.closed:
        await session.close()


@pytest.mark.asyncio
async def test_cluster_proxy_drops_content_encoding_on_gzip(monkeypatch):
    """Regression for the gzip Content-Encoding mismatch: aiohttp
    auto-decompresses the upstream body, so the worker must not keep advertising
    Content-Encoding: gzip — otherwise the server-side aiohttp client fails with
    'Can not decode content-encoding: gzip'."""
    payload = json.dumps({"items": ["x" * 1000 for _ in range(500)]}).encode()

    async with _fake_apiserver(payload) as base_url:
        async with _worker_client(monkeypatch, base_url) as client:
            resp = await client.get(
                "/cluster-proxy/api/v1/namespaces/kube-system/secrets",
                headers={"Accept-Encoding": "gzip"},
            )

    assert resp.status_code == 200
    # Body was auto-decompressed by aiohttp, so gzip must not be advertised.
    assert "content-encoding" not in {k.lower() for k in resp.headers}
    # The forwarded body is the real decompressed JSON, not gzip bytes.
    assert json.loads(resp.content) == json.loads(payload)


@pytest.mark.asyncio
async def test_cluster_proxy_uncompressed_passthrough(monkeypatch):
    """Guard against over-fixing: an uncompressed (small) response must still be
    forwarded verbatim."""
    payload = json.dumps({"items": []}).encode()

    async with _fake_apiserver(payload) as base_url:
        async with _worker_client(monkeypatch, base_url) as client:
            resp = await client.get(
                "/cluster-proxy/api/v1/namespaces/kube-system/secrets",
                # Force the (forwarded) upstream request to skip gzip; httpx
                # otherwise injects Accept-Encoding: gzip by default.
                headers={"Accept-Encoding": "identity"},
            )

    assert resp.status_code == 200
    assert "content-encoding" not in {k.lower() for k in resp.headers}
    assert json.loads(resp.content) == json.loads(payload)


@asynccontextmanager
async def _fake_apiserver_stream_then_fail(first_chunk: bytes):
    """A stand-in kube-apiserver that writes one chunk of a streamed response
    (mimicking a `?watch=true` long-lived call) and then aborts the connection
    mid-body, e.g. a network blip or the apiserver process dying. Never sends a
    second chunk and never completes normally."""

    async def handler(request):
        resp = web.StreamResponse(
            headers={"Content-Type": "application/json"},
        )
        await resp.prepare(request)
        await resp.write(first_chunk)
        # Simulate the transport failing mid-stream: abort the connection
        # instead of returning normally, so nothing terminates the body
        # cleanly. aiohttp's client (used by the proxy) sees this as a
        # mid-read failure on resp.content.iter_any(), not a clean EOF.
        request.transport.abort()
        return resp

    app = web.Application()
    app.router.add_route("*", "/{path:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_cluster_proxy_surfaces_error_when_upstream_stream_fails_mid_read(
    monkeypatch,
):
    """Regression for #5939: streamer() must not swallow a mid-body transport
    failure into a clean-looking end of stream. The status code and headers
    are already committed to the caller by the time the body starts (they were
    set from the upstream's initial 200 response), so the only way left to
    signal "this body is incomplete" is to fail the ASGI response instead of
    letting the generator return normally. Before the fix, this raises nothing
    and the client observes a truncated body under a 200 with no indication
    anything went wrong."""
    first_chunk = json.dumps({"items": ["ok"]}).encode() + b"\n"

    async with _fake_apiserver_stream_then_fail(first_chunk) as base_url:
        async with _worker_client(monkeypatch, base_url) as client:
            with pytest.raises(Exception):
                await client.get("/cluster-proxy/api/v1/namespaces/kube-system/pods")
