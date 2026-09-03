"""Tests for the segmented proxy timeouts in the worker reverse proxy (#6012).

The timeouts here exist to separate two failure shapes: a serving process that
dies (socket closes, already fast) and one that stays alive but stops responding.
The tests below drive the second shape -- an upstream that holds the connection
open and sends nothing -- on both sides of the first response chunk, plus the two
cases a naive single idle timeout would break: a long prefill and a long
non-streaming generation.
"""

import asyncio
import json
from contextlib import asynccontextmanager
from typing import AsyncIterator, List, Optional, Tuple, Union

import pytest
from aiohttp import ClientSession, web
from fastapi import FastAPI, Request
from httpx import ASGITransport, AsyncClient

from gpustack import envs
from gpustack.api.auth import worker_auth
from gpustack.api.exceptions import register_handlers
from gpustack.routes.worker import proxy as worker_proxy
from gpustack.utils.stall import stall_error_sse

# Small enough to keep the suite fast, large enough not to be flaky on a loaded
# machine.
IDLE_TIMEOUT = 0.2
TTFT_TIMEOUT = 0.6
# Longer than the inter-chunk budget but inside the first-chunk one: a prefill
# that would be killed if the two were collapsed into one idle value.
PREFILL_DELAY = 0.4
# Longer than either budget, so a wrongly-applied budget fails loudly instead of
# depending on the default being generous.
SLOW_GENERATION = 1.0
# Any stall long enough to outlast the budget under test; bounded so a leaked
# handler cannot hang the suite.
STALL = 2


class ScriptedBody:
    """Async iterator over scripted body steps, as ``resp.content.iter_chunked``.

    Each step is ``(delay, chunk)``. A ``None`` chunk stalls indefinitely (the
    caller's budget is what ends it); an exception chunk is raised.
    """

    def __init__(self, steps: List[Tuple[float, Union[bytes, None, Exception]]]):
        self._steps = list(steps)

    def __aiter__(self) -> "ScriptedBody":
        return self

    async def __anext__(self) -> bytes:
        if not self._steps:
            raise StopAsyncIteration
        delay, chunk = self._steps.pop(0)
        if chunk is None:
            await asyncio.sleep(STALL)
        await asyncio.sleep(delay)
        if isinstance(chunk, Exception):
            raise chunk
        return chunk


async def _collect(stream: AsyncIterator[bytes]) -> List[bytes]:
    return [chunk async for chunk in stream]


class TestReadFirstChunk:
    @pytest.mark.asyncio
    async def test_stall_before_first_chunk_times_out(self):
        with pytest.raises((asyncio.TimeoutError, TimeoutError)):
            await worker_proxy._read_first_chunk(
                ScriptedBody([(0, None)]), TTFT_TIMEOUT
            )

    @pytest.mark.asyncio
    async def test_empty_body_is_not_a_stall(self):
        """204/304/HEAD reach EOF immediately and must not wait out the budget."""
        assert (
            await worker_proxy._read_first_chunk(ScriptedBody([]), TTFT_TIMEOUT) is None
        )

    @pytest.mark.asyncio
    async def test_long_prefill_within_budget_is_not_terminated(self):
        chunk = await worker_proxy._read_first_chunk(
            ScriptedBody([(PREFILL_DELAY, b"data: first\n\n")]), TTFT_TIMEOUT
        )
        assert chunk == b"data: first\n\n"

    @pytest.mark.asyncio
    async def test_no_budget_waits_out_a_long_generation(self):
        """The non-streaming path passes no budget: its first chunk *is* the end
        of generation, so a first-chunk budget would kill a legitimate long
        completion."""
        chunk = await worker_proxy._read_first_chunk(
            ScriptedBody([(SLOW_GENERATION, b'{"id":"x"}')]), None
        )
        assert chunk == b'{"id":"x"}'


class TestStreamBody:
    @pytest.mark.asyncio
    async def test_mid_stream_stall_ends_with_error_event(self):
        chunks = await _collect(
            worker_proxy._stream_body(
                ScriptedBody([(0, b"data: b\n\n"), (0, None)]),
                b"data: a\n\n",
                IDLE_TIMEOUT,
            )
        )
        assert chunks == [b"data: a\n\n", b"data: b\n\n", stall_error_sse()]

    @pytest.mark.asyncio
    async def test_normal_stream_is_unchanged(self):
        chunks = await _collect(
            worker_proxy._stream_body(
                ScriptedBody([(0, b"data: b\n\n"), (0, b"data: [DONE]\n\n")]),
                b"data: a\n\n",
                IDLE_TIMEOUT,
            )
        )
        assert chunks == [b"data: a\n\n", b"data: b\n\n", b"data: [DONE]\n\n"]

    @pytest.mark.asyncio
    async def test_non_streaming_body_is_not_bound_by_idle_budget(self):
        chunks = await _collect(
            worker_proxy._stream_body(
                ScriptedBody([(SLOW_GENERATION, b'"}')]), b'{"id":"x', None
            )
        )
        assert chunks == [b'{"id":"x', b'"}']

    @pytest.mark.asyncio
    async def test_upstream_reset_still_propagates(self):
        """Only a stall is converted into an error event; an upstream that drops
        the connection must keep its current behavior."""
        stream = worker_proxy._stream_body(
            ScriptedBody([(0, ConnectionResetError("peer reset"))]), b"data: a\n\n", 5
        )
        with pytest.raises(ConnectionResetError):
            await _collect(stream)


@asynccontextmanager
async def _fake_inference_server(handler):
    app = web.Application()
    app.router.add_route("*", "/{path:.*}", handler)
    # Short shutdown timeout so a still-stalling handler does not hold up cleanup.
    runner = web.AppRunner(app, shutdown_timeout=0.2)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        yield runner.addresses[0][1]
    finally:
        await runner.cleanup()


@asynccontextmanager
async def _worker_client(port: int, instance_id: Optional[int] = None):
    app = FastAPI()
    register_handlers(app)
    app.include_router(worker_proxy.router)
    app.dependency_overrides[worker_auth] = lambda: None
    app.state.worker_ip_getter = lambda: "127.0.0.1"

    @app.middleware("http")
    async def route_to_instance(request: Request, call_next):
        # Stands in for set_port_from_model_name, which resolves the port from
        # the gateway's routing header and rewrites the path onto /proxy.
        request.scope["path"] = f"/proxy{request.url.path}"
        request.state.x_target_port = str(port)
        if instance_id is not None:
            request.state.x_target_instance_id = instance_id
        return await call_next(request)

    async with ClientSession() as session:
        app.state.http_client = session
        app.state.http_client_no_proxy = session
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://worker") as client:
            yield client, app


async def _sse_response(request, chunks: List[bytes], stall_after: bool):
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    # vLLM commits 200 + text/event-stream before prefill; the proxy must not
    # pass that on to the client until a first chunk actually arrives.
    await resp.prepare(request)
    for chunk in chunks:
        await resp.write(chunk)
    if stall_after:
        await asyncio.sleep(STALL)
    await resp.write_eof()
    return resp


@pytest.fixture
def tight_budgets(monkeypatch):
    monkeypatch.setattr(envs, "PROXY_TTFT_TIMEOUT", TTFT_TIMEOUT)
    monkeypatch.setattr(envs, "PROXY_STREAM_IDLE_TIMEOUT", IDLE_TIMEOUT)


@pytest.mark.asyncio
async def test_stall_before_first_token_returns_504(tight_budgets):
    """The reported symptom -- "no error, no status code, just silence" -- becomes
    a status code, which is what makes it retryable by an SDK."""

    async def handler(request):
        return await _sse_response(request, [], stall_after=True)

    async with _fake_inference_server(handler) as port:
        async with _worker_client(port) as (client, _):
            resp = await client.post("/v1/chat/completions", json={"stream": True})

    assert resp.status_code == 504
    assert resp.json()["error"]["code"] == 504


@pytest.mark.asyncio
async def test_stall_mid_stream_ends_stream_with_error_event(tight_budgets):
    """Headers are already committed here, so the only honest ending is a
    terminal error event -- a bare disconnect looks like a finished generation."""

    async def handler(request):
        return await _sse_response(
            request, [b"data: one\n\n", b"data: two\n\n"], stall_after=True
        )

    async with _fake_inference_server(handler) as port:
        async with _worker_client(port) as (client, _):
            resp = await client.post("/v1/chat/completions", json={"stream": True})

    assert resp.status_code == 200
    body = resp.content
    assert body.startswith(b"data: one\n\ndata: two\n\n")
    assert body.endswith(stall_error_sse())


@pytest.mark.asyncio
async def test_streaming_response_completes_normally(tight_budgets):
    async def handler(request):
        return await _sse_response(
            request, [b"data: one\n\n", b"data: [DONE]\n\n"], stall_after=False
        )

    async with _fake_inference_server(handler) as port:
        async with _worker_client(port) as (client, _):
            resp = await client.post("/v1/chat/completions", json={"stream": True})

    assert resp.status_code == 200
    assert resp.content == b"data: one\n\ndata: [DONE]\n\n"


@pytest.mark.asyncio
async def test_long_prefill_is_not_terminated(tight_budgets):
    """Silence before the first token can be a queued request or the prefill of a
    long prompt, so the inter-chunk budget must not apply to it."""

    async def handler(request):
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await asyncio.sleep(PREFILL_DELAY)
        await resp.write(b"data: one\n\n")
        await resp.write_eof()
        return resp

    async with _fake_inference_server(handler) as port:
        async with _worker_client(port) as (client, _):
            resp = await client.post("/v1/chat/completions", json={"stream": True})

    assert resp.status_code == 200
    assert resp.content == b"data: one\n\n"


@pytest.mark.asyncio
async def test_long_non_streaming_generation_is_not_terminated(tight_budgets):
    """A non-streaming response only arrives once generation has finished, so
    neither the first-chunk nor the inter-chunk budget may apply to it."""

    async def handler(request):
        await asyncio.sleep(SLOW_GENERATION)
        return web.json_response({"id": "done"})

    async with _fake_inference_server(handler) as port:
        async with _worker_client(port) as (client, _):
            resp = await client.post("/v1/chat/completions", json={"stream": False})

    assert resp.status_code == 200
    assert resp.json() == {"id": "done"}


@pytest.mark.asyncio
async def test_empty_body_response_is_forwarded(tight_budgets):
    """A response with no body must not be mistaken for a stalled one."""

    async def handler(request):
        return web.Response(status=204)

    async with _fake_inference_server(handler) as port:
        async with _worker_client(port) as (client, _):
            resp = await client.get("/v1/models")

    assert resp.status_code == 204
    assert resp.content == b""


@pytest.mark.asyncio
async def test_successful_inference_recorded_after_first_chunk(tight_budgets):
    """The health-check hook must not count a 200 that never produced data."""
    recorded: List[int] = []

    async def handler(request):
        if request.query.get("stall") == "true":
            return await _sse_response(request, [], stall_after=True)
        return await _sse_response(request, [b"data: one\n\n"], stall_after=False)

    async with _fake_inference_server(handler) as port:
        async with _worker_client(port, instance_id=7) as (client, app):
            app.state.record_successful_inference = recorded.append

            stalled = await client.post(
                "/v1/chat/completions?stall=true", json={"stream": True}
            )
            assert stalled.status_code == 504
            assert recorded == []

            ok = await client.post("/v1/chat/completions", json={"stream": True})
            assert ok.status_code == 200
            assert recorded == [7]


def test_stall_error_sse_is_a_terminated_stream():
    """The injected frames have to be parseable as SSE and end the stream, or a
    client still cannot tell a stall from a completed generation."""
    frames = stall_error_sse().decode().strip().split("\n\n")
    assert len(frames) == 2
    error = json.loads(frames[0].removeprefix("data: "))
    assert error["error"]["code"] == 504
    assert frames[1] == "data: [DONE]"


def test_event_stream_detection():
    assert worker_proxy._is_event_stream("text/event-stream; charset=utf-8")
    assert worker_proxy._is_event_stream("TEXT/EVENT-STREAM")
    assert not worker_proxy._is_event_stream("application/json")
    assert not worker_proxy._is_event_stream(None)


def test_defaults_are_ordered():
    """A first-chunk budget below the inter-chunk one, or either above the
    ceiling, would make the segmentation meaningless."""
    assert envs.PROXY_STREAM_IDLE_TIMEOUT <= envs.PROXY_TTFT_TIMEOUT
    assert envs.PROXY_TTFT_TIMEOUT <= envs.PROXY_TIMEOUT
