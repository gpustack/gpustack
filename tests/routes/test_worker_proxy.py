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
from gpustack.api.auth import worker_proxy_auth
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


def _build_worker_app(port: int, instance_id: Optional[int] = None) -> FastAPI:
    app = FastAPI()
    register_handlers(app)
    app.include_router(worker_proxy.router)
    app.dependency_overrides[worker_proxy_auth] = lambda: None
    app.state.worker_ip_getter = lambda: "127.0.0.1"

    @app.middleware("http")
    async def route_to_instance(request: Request, call_next):
        # Stands in for set_port_from_model_name, which resolves the port from
        # the gateway's routing header and rewrites the path onto /proxy.
        request.scope["path"] = f"/proxy{request.url.path}"
        request.state.x_target_port = str(port)
        request.state.model_instance_endpoint = getattr(
            app.state, "routed_target", None
        )
        if instance_id is not None:
            request.state.x_target_instance_id = instance_id
        return await call_next(request)

    return app


@asynccontextmanager
async def _worker_client(port: int, instance_id: Optional[int] = None):
    app = _build_worker_app(port, instance_id)
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
async def test_first_chunk_failure_closes_upstream_response(monkeypatch):
    """A non-timeout failure while waiting for the first chunk must still close
    the upstream response: the StreamingResponse background close never gets
    constructed, so the pooled connection would otherwise leak."""

    import aiohttp

    closed = []
    original_close = aiohttp.ClientResponse.close

    def _record_close(self):
        closed.append(True)
        return original_close(self)

    monkeypatch.setattr(aiohttp.ClientResponse, "close", _record_close)

    async def _fail(body_iter, timeout):
        raise aiohttp.ClientPayloadError("connection reset mid-preflight")

    monkeypatch.setattr(worker_proxy, "_read_first_chunk", _fail)

    async def handler(request):
        return await _sse_response(request, [b"data: one\n\n"], stall_after=False)

    async with _fake_inference_server(handler) as port:
        async with _worker_client(port) as (client, _):
            resp = await client.post("/v1/chat/completions", json={"stream": True})
            # Asserted inside the client context so a later close during
            # session teardown cannot satisfy it spuriously.
            assert resp.status_code == 503
            assert (
                closed
            ), "upstream response must be closed on a non-timeout read failure"


@pytest.mark.asyncio
async def test_cancellation_while_awaiting_first_chunk_closes_response(
    monkeypatch,
):
    """CancelledError derives from BaseException, so it bypasses the generic
    handler: without an explicit catch the upstream response leaks when the
    coroutine is cancelled (client disconnect, server shutdown) while waiting
    for the first chunk."""

    import aiohttp

    closed = []
    original_close = aiohttp.ClientResponse.close

    def _record_close(self):
        closed.append(True)
        return original_close(self)

    monkeypatch.setattr(aiohttp.ClientResponse, "close", _record_close)

    async def _cancel(body_iter, timeout):
        raise asyncio.CancelledError()

    monkeypatch.setattr(worker_proxy, "_read_first_chunk", _cancel)

    async def handler(request):
        return await _sse_response(request, [b"data: one\n\n"], stall_after=False)

    async with _fake_inference_server(handler) as port:
        async with _worker_client(port) as (client, _):
            # Starlette's BaseHTTPMiddleware (the test fixture's router) wraps
            # the cancellation as RuntimeError; either way the close must have
            # happened before the error left the handler.
            with pytest.raises((asyncio.CancelledError, RuntimeError)):
                await client.post("/v1/chat/completions", json={"stream": True})
            assert closed, (
                "upstream response must be closed when cancelled before "
                "streaming starts"
            )


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


async def _watch_for_upstream_close(request, closed: asyncio.Event, sse: bool):
    """Fake engine: keeps "generating" until its client connection closes.

    This is how vLLM decides to abort: it notices that the connection from the
    proxy has closed. ``closed`` is set when that happens.
    """
    if sse:
        # Headers committed before prefill, as vLLM does, but no token yet.
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
    for _ in range(int(STALL / 0.02)):
        if request.transport is None or request.transport.is_closing():
            closed.set()
            break
        await asyncio.sleep(0.02)
    return web.Response()


async def _call_then_disconnect(
    app: FastAPI, body: bytes, disconnect_after: float, chunked: bool = False
):
    """Drive the ASGI app directly with a client that sends its request and
    then goes away, the way a user cancelling a generation does."""
    if chunked:
        half = len(body) // 2
        messages = [
            {"type": "http.request", "body": body[:half], "more_body": True},
            {"type": "http.request", "body": body[half:], "more_body": False},
        ]
        framing = (b"transfer-encoding", b"chunked")
    else:
        messages = [{"type": "http.request", "body": body, "more_body": False}]
        framing = (b"content-length", str(len(body)).encode())

    async def receive():
        if messages:
            return messages.pop(0)
        await asyncio.sleep(disconnect_after)
        return {"type": "http.disconnect"}

    async def send(message):
        pass

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"worker"),
            (b"content-type", b"application/json"),
            framing,
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 80),
    }
    await app(scope, receive, send)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stream, chunked",
    [
        # vLLM sends nothing for a non-streaming request until generation is
        # done, so the whole generation used to run for a departed client.
        (False, False),
        # Same gap for a stream still queued or in prefill: no first chunk yet.
        (True, False),
        # A chunked upload is streamed to the upstream; the disconnect watch
        # must start once it has been read, not before.
        (False, True),
    ],
    ids=["non_streaming", "streaming_before_first_token", "chunked_upload"],
)
async def test_client_disconnect_before_response_closes_upstream(stream, chunked):
    """A client that disconnects before the response starts must make the proxy
    close its upstream connection promptly, so the engine aborts the request
    and frees the GPU instead of finishing a generation nobody will read."""
    closed = asyncio.Event()

    async def handler(request):
        return await _watch_for_upstream_close(request, closed, sse=stream)

    body = json.dumps({"stream": stream}).encode()
    async with _fake_inference_server(handler) as port:
        app = _build_worker_app(port)
        async with ClientSession() as session:
            app.state.http_client = session
            app.state.http_client_no_proxy = session
            await asyncio.wait_for(
                _call_then_disconnect(app, body, disconnect_after=0.1, chunked=chunked),
                timeout=STALL,
            )
            # Well under STALL: the fake engine would otherwise run it out.
            await asyncio.wait_for(closed.wait(), timeout=1.0)


@pytest.mark.asyncio
async def test_chunked_upload_is_streamed_to_upstream(tight_budgets):
    """A chunked request body reaches the upstream intact. The branch handling
    it used to be unreachable, which sent every body through request.body()."""
    received = {}

    async def handler(request):
        received["body"] = await request.read()
        received["transfer_encoding"] = request.headers.get("Transfer-Encoding")
        return web.json_response({"id": "done"})

    async def upload():
        yield b'{"stream": '
        yield b"false}"

    async with _fake_inference_server(handler) as port:
        async with _worker_client(port) as (client, _):
            resp = await client.post(
                "/v1/chat/completions",
                content=upload(),
                headers={"content-type": "application/json"},
            )

    assert resp.status_code == 200
    assert resp.json() == {"id": "done"}
    assert received["body"] == b'{"stream": false}'
    assert received["transfer_encoding"] == "chunked"


class _ScriptedReceive:
    """Stands in for a Request in _cancel_on_client_disconnect: only
    ``receive`` is used."""

    def __init__(self, receive):
        self.receive = receive


def _consumed() -> asyncio.Event:
    event = asyncio.Event()
    event.set()
    return event


class TestCancelOnClientDisconnect:
    @pytest.mark.asyncio
    async def test_result_is_returned_while_client_stays(self):
        async def never_disconnects():
            await asyncio.sleep(STALL)

        async def work():
            return "upstream"

        result = await worker_proxy._cancel_on_client_disconnect(
            _ScriptedReceive(never_disconnects), work(), _consumed(), discard=None
        )
        assert result == "upstream"

    @pytest.mark.asyncio
    async def test_result_finishing_with_the_disconnect_is_discarded(self):
        """The watcher has consumed the disconnect, so nothing downstream would
        notice it: the finished result must be released, not relayed."""
        discarded = []

        async def disconnects_at_once():
            return {"type": "http.disconnect"}

        async def work():
            return "upstream"

        with pytest.raises(worker_proxy._ClientDisconnected):
            await worker_proxy._cancel_on_client_disconnect(
                _ScriptedReceive(disconnects_at_once),
                work(),
                _consumed(),
                discard=discarded.append,
            )
        assert discarded == ["upstream"]

    @pytest.mark.asyncio
    async def test_watcher_failure_propagates_instead_of_disconnect(self):
        cancelled = asyncio.Event()

        async def fails():
            raise RuntimeError("receive failed")

        async def work():
            try:
                await asyncio.sleep(STALL)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        with pytest.raises(RuntimeError, match="receive failed"):
            await worker_proxy._cancel_on_client_disconnect(
                _ScriptedReceive(fails), work(), _consumed(), discard=None
            )
        assert cancelled.is_set()

    @pytest.mark.asyncio
    async def test_no_listening_before_body_is_consumed(self):
        """Receiving early would steal body messages from the upload."""
        received = []
        body_consumed = asyncio.Event()

        async def receive():
            received.append(True)
            return {"type": "http.disconnect"}

        async def work():
            await asyncio.sleep(0.1)
            return "upstream"

        result = await worker_proxy._cancel_on_client_disconnect(
            _ScriptedReceive(receive), work(), body_consumed, discard=None
        )
        assert result == "upstream"
        assert received == []


@pytest.fixture
def rank_instance():
    from gpustack.schemas.models import (
        DistributedServers,
        ModelInstanceWorkload,
        ModelInstanceStateEnum,
        SourceEnum,
    )
    from tests.utils.model import new_model_instance, new_model_instance_workload

    instance = new_model_instance(
        7, 'moe', 1, worker_id=1, state=ModelInstanceStateEnum.RUNNING
    )
    instance.source, instance.local_path = SourceEnum.LOCAL_PATH, '/model'
    instance.worker_ip, instance.port, instance.ports = '10.0.0.1', 49999, [49999]
    instance.restart_count = 3
    instance.distributed_servers = DistributedServers(
        workloads=[
            ModelInstanceWorkload.model_validate(
                new_model_instance_workload(
                    name='engine',
                    state='running',
                    dp_rank_placements=[
                        dict(
                            dp_rank=r,
                            node_rank=0,
                            gpu_indexes=[r],
                            api_port=41000 + r,
                            ready=True,
                        )
                        for r in (0, 1)
                    ],
                )
            )
        ]
    )
    return instance


@pytest.mark.parametrize('suffix', ['static', 'dns'])
@pytest.mark.parametrize('selector', ['', '-dp0', '-w2'])
def test_model_target_contains_only_endpoint_identity(suffix, selector):
    from gpustack.utils.model_instance_workers import parse_model_instance_target

    target = parse_model_instance_target(f'model-1-7{selector}.{suffix}')
    assert target.model_id == 1 and target.instance_id == 7
    assert target.dp_rank == (0 if selector == '-dp0' else None)
    assert target.worker_id == (2 if selector == '-w2' else None)


@pytest.mark.parametrize(
    'selector',
    ['-r3-dp0', '-r3-smg-a2', '-engine-dp0', '-hybrid-w2', '-labcd', '-dp0-w2'],
)
def test_model_target_rejects_noncanonical_names(selector):
    from gpustack.utils.model_instance_workers import parse_model_instance_target

    with pytest.raises(ValueError):
        parse_model_instance_target(f'model-1-7{selector}.static')


def test_rank_destination_resolves_selected_port_and_epoch(rank_instance):
    from gpustack.utils.model_instance_workers import (
        parse_model_instance_target,
        resolve_model_instance_endpoint,
    )

    target = parse_model_instance_target('model-1-7-dp1.static')
    endpoint = resolve_model_instance_endpoint(rank_instance, target, worker_id=1)
    assert endpoint.port == 41001
    assert endpoint.dp_rank == 1
    assert endpoint.restart_count == 3
    for name in (
        'model-1-7-r2-dp1.static',
        'model-1-7-dp9.static',
        'model-2-7-r3-dp1.static',
        'model-1-7.static',
    ):
        with pytest.raises(ValueError):
            resolve_model_instance_endpoint(
                rank_instance, parse_model_instance_target(name), worker_id=1
            )
    with pytest.raises(ValueError):
        resolve_model_instance_endpoint(rank_instance, target, worker_id=2)


@pytest.mark.parametrize('change', ['empty', 'draining', 'error', 'unready', 'unknown'])
def test_rank_destination_never_falls_back_to_primary(rank_instance, change):
    from datetime import datetime, timezone
    from gpustack.schemas.models import (
        ModelInstanceStateEnum,
        ModelInstanceWorkloadStateEnum,
    )
    from gpustack.utils.model_instance_workers import (
        parse_model_instance_target,
        resolve_model_instance_endpoint,
    )

    if change == 'empty':
        rank_instance.distributed_servers.workloads = []
        rank_instance.distributed_servers.orchestration_mode = "managed"
    elif change == 'draining':
        rank_instance.draining_since = datetime.now(timezone.utc)
    elif change == 'error':
        rank_instance.state = ModelInstanceStateEnum.ERROR
    elif change == 'unready':
        rank_instance.distributed_servers.workloads[0].dp_rank_placements[
            0
        ].ready = False
    else:
        rank_instance.distributed_servers.workloads[0].state = (
            ModelInstanceWorkloadStateEnum.UNKNOWN
        )
    with pytest.raises(ValueError):
        resolve_model_instance_endpoint(
            rank_instance, parse_model_instance_target('model-1-7-dp1.static')
        )


@pytest.mark.parametrize('cache_present', [True, False])
def test_rank_getter_reads_watch_generation(rank_instance, cache_present):
    from unittest.mock import MagicMock
    from gpustack.worker.serve_manager import ServeManager

    manager = ServeManager.__new__(ServeManager)
    manager._worker_id_getter = lambda: 1
    manager._model_instance_by_instance_id = (
        {7: rank_instance.model_copy(deep=True)} if cache_present else {}
    )
    clientset = MagicMock()
    manager._clientset_getter = lambda: clientset
    manager._clientset.model_instances.get_cached.return_value = rank_instance
    assert (
        getattr(
            manager.get_instance_endpoint_by_model_instance_id(
                7, model_id=1, dp_rank=1
            ),
            'port',
            None,
        )
        == 41001
    )
    manager._clientset.model_instances.get_cached.assert_called_once_with(7)
    rank_instance.restart_count = 4
    endpoint = manager.get_instance_endpoint_by_model_instance_id(
        7, model_id=1, dp_rank=1
    )
    assert (endpoint is None) == cache_present
    if endpoint is not None:
        assert endpoint.restart_count == 4
    manager._worker_id_getter = lambda: 2
    assert (
        getattr(
            manager.get_instance_endpoint_by_model_instance_id(
                7, model_id=1, dp_rank=1
            ),
            'port',
            None,
        )
        is None
    )


@pytest.mark.asyncio
async def test_rank_middleware_uses_selected_port(rank_instance):
    from gpustack.utils.model_instance_workers import (
        ModelInstanceTarget,
        resolve_model_instance_endpoint,
    )

    app = FastAPI()
    register_handlers(app)
    app.middleware('http')(worker_proxy.set_port_from_model_name)
    calls = []

    def get_port(instance_id, **kwargs):
        calls.append((instance_id, kwargs))
        return resolve_model_instance_endpoint(
            rank_instance,
            ModelInstanceTarget(
                kwargs['model_id'],
                instance_id,
                dp_rank=kwargs['dp_rank'],
            ),
            worker_id=1,
        )

    app.state.get_instance_endpoint_by_model_instance_id = get_port

    @app.get('/proxy/v1/models')
    def target(request: Request):
        return {'port': request.state.x_target_port}

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url='http://worker'
    ) as client:
        response = await client.get(
            '/v1/models',
            headers={'X-GPUStack-Model-Instance': 'model-1-7-dp1.static'},
        )
    assert response.json() == {'port': '41001'}
    assert calls == [(7, dict(model_id=1, dp_rank=1, worker_id=None))]


@pytest.mark.parametrize('mode', ['direct', 'worker', 'tunnel'])
def test_rank_registry_construction_uses_actual_owner(rank_instance, mode):
    from gpustack.gateway.utils import model_instance_registry
    from gpustack.schemas.workers import Worker, ModelInstanceProxyModeEnum
    from gpustack.utils.model_instance_workers import model_instance_endpoints

    worker = Worker(
        id=1,
        ip='10.0.0.2',
        port=10150,
        proxy_address='http://127.0.0.1:10160',
        proxy_mode=ModelInstanceProxyModeEnum(mode),
    )
    endpoint = model_instance_endpoints(rank_instance)[1]
    registry = model_instance_registry(rank_instance, worker=worker, endpoint=endpoint)
    assert registry.name == 'model-1-7-dp1'
    if mode == 'direct':
        assert registry.domain == '10.0.0.2:41001'
    assert model_instance_registry(rank_instance, worker=worker) is None
    worker.id = 2
    assert (
        model_instance_registry(rank_instance, worker=worker, endpoint=endpoint) is None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'case',
    [
        'worker',
        'cluster',
        'user',
        'foreign_worker',
        'foreign_cluster',
        'stale_binding',
        'stale_epoch',
        'null_cluster',
        'foreign_owner',
        'null_owner',
        'deleted_worker',
        'missing_worker',
        'unbound_system',
        'server_token',
        'stale_cluster_binding',
    ],
)
@pytest.mark.parametrize(
    'kind',
    [
        'rank_instance',
        'hybrid_instance',
        'sglang_component_instance',
        'sglang_engine',
        'legacy_instance',
        'legacy_distributed_instance',
    ],
)
async def test_tunnel_model_resolver_checks_identity_and_releases_session(
    request, monkeypatch, case, kind
):
    fixture = {
        'sglang_engine': 'sglang_component_instance',
        'legacy_instance': 'rank_instance',
        'legacy_distributed_instance': 'rank_instance',
    }.get(kind, kind)
    rank_instance = request.getfixturevalue(fixture)
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from gpustack.gateway import utils
    from gpustack.schemas.principals import PrincipalType
    from gpustack.schemas.models import DistributedServers, ModelInstance
    from gpustack.schemas.workers import Worker
    from gpustack.schemas.clusters import Cluster
    from fastapi import HTTPException

    rank_instance.distributed_servers = {
        'legacy_instance': None,
        'legacy_distributed_instance': DistributedServers(),
    }.get(kind, rank_instance.distributed_servers)
    rank_instance.cluster_id = 11
    rank_instance.owner_principal_id = 20
    worker = SimpleNamespace(
        id=1,
        ip='10.0.0.2',
        cluster_id=11,
        deleted_at=None,
        system_principal_id=9,
        owner_principal_id=20,
    )
    principal = SimpleNamespace(
        id=None if case == 'server_token' else 9,
        name='system/worker/server-token',
        kind=PrincipalType.SYSTEM,
        is_admin=False,
        worker=None if case in ('unbound_system', 'server_token') else worker,
        cluster=None,
    )
    if case == 'null_cluster':
        rank_instance.cluster_id = None
    elif case == 'foreign_owner':
        rank_instance.owner_principal_id = 21
    elif case == 'null_owner':
        rank_instance.owner_principal_id = None
    elif case == 'deleted_worker':
        worker.deleted_at = 'deleted'
    elif case in ('cluster', 'stale_cluster_binding'):
        principal.worker = None
        principal.cluster = SimpleNamespace(id=11, owner_principal_id=20)
    elif case == 'user':
        principal.kind = PrincipalType.USER
    elif case == 'foreign_worker':
        principal.worker = SimpleNamespace(id=2, cluster_id=11, owner_principal_id=20)
    elif case == 'foreign_cluster':
        principal.worker = SimpleNamespace(id=1, cluster_id=12, owner_principal_id=20)
    elif case == 'stale_binding':
        worker.system_principal_id = 10
    elif case == 'stale_epoch':
        rank_instance.restart_count = 4
    closed = []

    @asynccontextmanager
    async def session():
        try:
            yield object()
        finally:
            closed.append(True)

    monkeypatch.setattr(utils, 'async_session', session)
    monkeypatch.setattr(
        'gpustack.api.auth.authenticate_worker_by_request_headers',
        AsyncMock(return_value=principal),
    )
    monkeypatch.setattr(
        ModelInstance, 'one_by_id', AsyncMock(return_value=rank_instance)
    )
    monkeypatch.setattr(
        Worker,
        'one_by_id',
        AsyncMock(return_value=None if case == 'missing_worker' else worker),
    )
    monkeypatch.setattr(
        Cluster,
        'one_by_id',
        AsyncMock(
            return_value=SimpleNamespace(
                system_principal_id=10 if case == 'stale_cluster_binding' else 9
            )
        ),
    )
    headers = {
        utils.router_header_key: {
            'rank_instance': 'model-1-7-dp1.static',
            'hybrid_instance': 'model-1-7-w1.static',
            'sglang_component_instance': 'model-1-7.static',
            'sglang_engine': 'model-1-7-dp1.static',
            'legacy_instance': 'model-1-7.static',
            'legacy_distributed_instance': 'model-1-7.static',
        }[kind]
    }
    if (
        case in ('worker', 'cluster', 'stale_epoch', 'server_token')
        and kind != 'sglang_engine'
    ):
        resolved = await utils.resolve_instance_address_from_model_header(headers)
        assert resolved == (
            '10.0.0.2',
            {
                'sglang_component_instance': 42001,
                'legacy_instance': 49999,
                'legacy_distributed_instance': 49999,
            }.get(kind, 41001),
        )
        assert type(resolved) is tuple
    else:
        with pytest.raises(HTTPException) as exc:
            await utils.resolve_instance_address_from_model_header(headers)
        if case in (
            'null_cluster',
            'foreign_owner',
            'null_owner',
            'unbound_system',
            'stale_cluster_binding',
        ):
            assert exc.value.status_code == {
                'unbound_system': 403,
                'stale_cluster_binding': 404 if kind == 'sglang_engine' else 403,
            }.get(case, 404)
    assert closed == ([] if case == 'user' else [True])


@pytest.mark.asyncio
async def test_legacy_header_checks_visibility_before_rejecting_managed_instance(
    rank_instance, monkeypatch
):
    from unittest.mock import AsyncMock
    from types import SimpleNamespace
    from fastapi import HTTPException
    from gpustack.gateway import utils
    from gpustack.schemas.models import ModelInstance
    from gpustack.schemas.principals import PrincipalType
    from gpustack.schemas.workers import Worker
    from tests.utils.mock import mock_async_session

    assert utils.model_instance_endpoints(rank_instance)
    session = mock_async_session()
    authenticate = AsyncMock(return_value=SimpleNamespace(kind=PrincipalType.SYSTEM))
    visibility = AsyncMock()
    worker_lookup = AsyncMock()
    monkeypatch.setattr(utils, 'async_session', lambda: session)
    monkeypatch.setattr(
        ModelInstance, 'one_by_id', AsyncMock(return_value=rank_instance)
    )
    monkeypatch.setattr(Worker, 'one_by_id', worker_lookup)
    monkeypatch.setattr(
        'gpustack.api.auth.authenticate_worker_by_request_headers', authenticate
    )
    monkeypatch.setattr(utils, '_assert_rank_destination_visible', visibility)
    with pytest.raises(HTTPException) as error:
        await utils.resolve_instance_address_from_model_header(
            {utils.router_header_key: 'model-1-7.static'}
        )
    assert error.value.status_code == 404
    authenticate.assert_awaited_once()
    visibility.assert_awaited_once()
    worker_lookup.assert_not_awaited()
    session.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize('credential', ['missing', 'user'])
@pytest.mark.parametrize(
    'destination', ['model-invalid', 'model-1-7.static', 'model-1-7-dp1.static']
)
async def test_tunnel_rejects_credentials_before_processing_model_targets(
    monkeypatch, credential, destination
):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock
    from fastapi import HTTPException
    from gpustack.gateway import utils
    from gpustack.schemas.models import ModelInstance
    from gpustack.schemas.principals import PrincipalType

    principal = (
        SimpleNamespace(kind=PrincipalType.USER) if credential == 'user' else None
    )
    authenticate = AsyncMock(return_value=principal)
    session = MagicMock()
    lookup = AsyncMock()
    resolve = MagicMock()
    monkeypatch.setattr(
        'gpustack.api.auth.authenticate_worker_by_request_headers', authenticate
    )
    monkeypatch.setattr(utils, 'async_session', session)
    monkeypatch.setattr(ModelInstance, 'one_by_id', lookup)
    monkeypatch.setattr(utils, 'resolve_model_instance_endpoint', resolve)
    headers = {utils.router_header_key: destination}
    with pytest.raises(HTTPException) as error:
        await utils.resolve_instance_address_from_model_header(headers)
    assert error.value.status_code == 403
    assert error.value.detail == 'Model routing requires internal credentials'
    authenticate.assert_awaited_once_with(headers)
    session.assert_not_called()
    lookup.assert_not_awaited()
    resolve.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'principal_kind, scope_mismatch',
    [
        ('worker', 'cluster'),
        ('worker', 'owner'),
        ('cluster', 'cluster'),
        ('cluster', 'owner'),
        ('unbound', None),
    ],
)
@pytest.mark.parametrize(
    'target_state',
    [
        'missing',
        'ready',
        'unready',
        'unknown_rank',
        'missing_worker',
        'deleted_worker',
        'no_address',
        'draining',
    ],
)
async def test_tunnel_hides_invisible_model_state(
    rank_instance, monkeypatch, principal_kind, scope_mismatch, target_state
):
    from datetime import datetime, timezone
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock
    from gpustack.gateway import utils
    from gpustack.schemas.models import ModelInstance
    from gpustack.schemas.principals import PrincipalType
    from gpustack.schemas.workers import Worker
    from gpustack.websocket_proxy.proxy_server import HTTPSProxyServer
    from tests.utils.mock import mock_async_session

    rank_instance.cluster_id, rank_instance.owner_principal_id = 11, 20
    scope = SimpleNamespace(
        id=2 if principal_kind == 'worker' else 11,
        cluster_id=12 if scope_mismatch == 'cluster' else 11,
        owner_principal_id=21 if scope_mismatch == 'owner' else 20,
    )
    if principal_kind == 'cluster':
        scope.id = scope.cluster_id
    principal = SimpleNamespace(
        id=9,
        kind=PrincipalType.SYSTEM,
        is_admin=False,
        worker=scope if principal_kind == 'worker' else None,
        cluster=scope if principal_kind == 'cluster' else None,
    )
    worker = SimpleNamespace(
        id=1,
        cluster_id=11,
        owner_principal_id=20,
        system_principal_id=9,
        ip=None if target_state == 'no_address' else '10.0.0.1',
        deleted_at=(
            datetime.now(timezone.utc) if target_state == 'deleted_worker' else None
        ),
    )
    if target_state == 'unready':
        for workload in rank_instance.distributed_servers.workloads:
            for placement in workload.dp_rank_placements:
                placement.ready = False
    if target_state == 'draining':
        rank_instance.draining_since = datetime.now(timezone.utc)
    session = mock_async_session()
    monkeypatch.setattr(utils, 'async_session', lambda: session)
    monkeypatch.setattr(
        'gpustack.api.auth.authenticate_worker_by_request_headers',
        AsyncMock(return_value=principal),
    )
    monkeypatch.setattr(
        ModelInstance,
        'one_by_id',
        AsyncMock(return_value=None if target_state == 'missing' else rank_instance),
    )
    resolve = MagicMock(wraps=utils.resolve_model_instance_endpoint)
    worker_lookup = AsyncMock(
        return_value=None if target_state == 'missing_worker' else worker
    )
    monkeypatch.setattr(utils, 'resolve_model_instance_endpoint', resolve)
    monkeypatch.setattr(Worker, 'one_by_id', worker_lookup)

    reader = asyncio.StreamReader()
    rank = 8 if target_state == 'unknown_rank' else 1
    reader.feed_data(
        (
            'GET http://ignored/v1/models HTTP/1.1\r\n'
            f'X-GPUStack-Model-Instance: model-1-7-dp{rank}.static\r\n\r\n'
        ).encode()
    )
    reader.feed_eof()
    response = bytearray()
    writer = MagicMock()
    writer.write.side_effect = response.extend
    writer.drain, writer.wait_closed = AsyncMock(), AsyncMock()
    connections = MagicMock()
    proxy = HTTPSProxyServer(
        '127.0.0.1',
        0,
        connections,
        authenticator=AsyncMock(return_value=True),
        header_router=utils.resolve_instance_address_from_model_header,
    )
    await proxy._handle_client(reader, writer)
    headers, body = bytes(response).split(b'\r\n\r\n', 1)
    if principal_kind == 'unbound':
        assert headers.startswith(b'HTTP/1.1 403 ')
        assert body == b'System principal has no cluster scope'
    else:
        assert headers.startswith(b'HTTP/1.1 404 ')
        assert body == b'Model destination not found'
    resolve.assert_not_called()
    worker_lookup.assert_not_awaited()
    connections.assert_not_called()
    session.__aexit__.assert_awaited_once()


def test_headless_worker_never_becomes_a_rank_api(rank_instance):
    from gpustack.schemas.models import (
        ModelInstanceSubordinateWorker,
        ModelInstanceWorkload,
    )
    from gpustack.utils.model_instance_workers import model_instance_endpoints
    from tests.utils.model import new_model_instance_workload

    rank_instance.distributed_servers.subordinate_workers = [
        ModelInstanceSubordinateWorker(worker_id=2)
    ]
    headless = ModelInstanceWorkload.model_validate(
        new_model_instance_workload(worker_id=2, node_rank=1, state='running')
    )
    rank_instance.distributed_servers.workloads.append(headless)
    assert [endpoint.port for endpoint in model_instance_endpoints(rank_instance)] == [
        41000,
        41001,
    ]
    headless.state = 'unknown'
    assert model_instance_endpoints(rank_instance) == []


@pytest.mark.parametrize(
    'fixture_name', ['rank_instance', 'sglang_component_instance', 'hybrid_instance']
)
def test_runtime_degradation_withdraws_the_generation_until_full_recovery(
    request, fixture_name
):
    from gpustack.schemas.models import ModelInstanceStateEnum
    from gpustack.utils.model_instance_workers import model_instance_endpoints

    instance = request.getfixturevalue(fixture_name)
    healthy = instance.model_copy(deep=True)
    expected = len(model_instance_endpoints(instance))
    assert expected > 0
    task = instance.distributed_servers.workloads[0]
    if task.component is None:
        task.dp_rank_placements[-1].ready = False
    else:
        task.state = 'unknown'
        task.component.ready = False
    assert model_instance_endpoints(instance) == []

    instance = healthy.model_copy(deep=True)
    instance.restart_count += 1
    instance.state = ModelInstanceStateEnum.STARTING
    assert model_instance_endpoints(instance) == []
    instance.state = ModelInstanceStateEnum.RUNNING
    assert len(model_instance_endpoints(instance)) == expected


def test_rank_without_main_api_is_not_partially_routable(rank_instance):
    from gpustack.schemas.models import ModelInstanceWorkload
    from gpustack.utils.model_instance_workers import model_instance_endpoints
    from tests.utils.model import new_model_instance_workload

    headless = ModelInstanceWorkload.model_validate(
        new_model_instance_workload(
            dp_rank=2,
            node_rank=1,
            name='headless',
            state='running',
            dp_rank_placements=[dict(dp_rank=2, node_rank=1, gpu_indexes=[2])],
        )
    )
    rank_instance.distributed_servers.workloads.append(headless)
    assert model_instance_endpoints(rank_instance) == []


@pytest.fixture
def sglang_component_instance(rank_instance):
    from gpustack.schemas.models import ModelInstanceWorkload

    rank_instance.distributed_servers.workloads = [
        ModelInstanceWorkload(
            name='engine',
            worker_id=1,
            state='running',
            component={
                'kind': 'sglang_engine',
                'api_port': 42000,
                'dp_size': 4,
                'ready': True,
            },
        ),
        ModelInstanceWorkload(
            name='smg',
            worker_id=1,
            state='running',
            component={
                'kind': 'sglang_smg',
                'api_port': 42001,
                'metrics_port': 42002,
                'attempt': 2,
                'ready': True,
            },
        ),
    ]
    rank_instance.distributed_servers.orchestration_mode = "managed"
    return rank_instance


def test_sglang_default_endpoint_is_single_smg(sglang_component_instance):
    from gpustack.utils.model_instance_workers import (
        model_instance_endpoints,
        parse_model_instance_target,
        resolve_model_instance_endpoint,
    )

    instance = sglang_component_instance
    endpoints = model_instance_endpoints(instance)
    assert len(endpoints) == 1
    assert endpoints[0].registry_name == 'model-1-7'
    assert endpoints[0].port == 42001
    target = parse_model_instance_target('model-1-7.static')
    assert resolve_model_instance_endpoint(instance, target) == endpoints[0]
    instance.restart_count += 1
    instance.distributed_servers.workloads[1].component.attempt += 1
    recovered = resolve_model_instance_endpoint(instance, target)
    assert recovered.registry_name == endpoints[0].registry_name
    assert recovered.restart_count == 4 and recovered.attempt == 3
    with pytest.raises(ValueError):
        resolve_model_instance_endpoint(
            instance, parse_model_instance_target('model-1-7-r3-smg-a1.static')
        )


@pytest.mark.parametrize('state', ['running', 'starting'])
def test_sglang_engine_rank_targets_are_not_platform_routes(
    sglang_component_instance, state
):
    from gpustack.utils.model_instance_workers import (
        parse_model_instance_target,
        resolve_model_instance_endpoint,
    )

    instance = sglang_component_instance
    instance.state = state
    for rank in (0, 3, 4):
        with pytest.raises(ValueError, match='unavailable'):
            resolve_model_instance_endpoint(
                instance, parse_model_instance_target(f'model-1-7-dp{rank}.static')
            )


@pytest.mark.parametrize(
    'destination',
    [
        'model-1-7-r3-smg.static',
        'model-1-7-r3-smg-a.static',
        'model-1-7-r3-smg-a-1.static',
        'model-1-7-dp.static',
        'model-1-7-engine-dp1.static',
        'model-1-7-smg-a0.static',
        'model-1-7-dp0-alias.static',
        'model-1-7-hybrid-w1.static',
        'model-1-7-r3-hybrid.static',
        'model-1-7-w.static',
        'model-1-7-w-1.static',
        'model-1-7-w1-alias.static',
    ],
)
def test_component_reserved_target_shapes_cannot_be_legacy_aliases(destination):
    from gpustack.utils.model_instance_workers import parse_model_instance_target

    with pytest.raises(ValueError):
        parse_model_instance_target(destination)


@pytest.mark.parametrize(
    'change',
    ['missing_engine', 'unready_engine', 'unknown_engine', 'draining', 'stopping_smg'],
)
def test_sglang_components_fail_closed_for_default_entry(
    sglang_component_instance, change
):
    from datetime import datetime, timezone
    from gpustack.schemas.models import ModelInstanceSubordinateWorker
    from gpustack.utils.model_instance_workers import model_instance_endpoints

    instance = sglang_component_instance
    if change == 'missing_engine':
        instance.distributed_servers.subordinate_workers = [
            ModelInstanceSubordinateWorker(worker_id=2, gpu_indexes=[0])
        ]
    elif change == 'unready_engine':
        instance.distributed_servers.workloads[0].component.ready = False
    elif change == 'unknown_engine':
        instance.distributed_servers.workloads[0].state = 'unknown'
    elif change == 'draining':
        instance.draining_since = datetime.now(timezone.utc)
    else:
        instance.distributed_servers.workloads[1].component.stop_requested = True
    assert model_instance_endpoints(instance) == []


@pytest.fixture
def component_getter(sglang_component_instance):
    from unittest.mock import MagicMock
    from gpustack.worker.serve_manager import ServeManager

    manager = ServeManager.__new__(ServeManager)
    manager._worker_id_getter = lambda: 1
    manager._model_instance_by_instance_id = {}
    client = MagicMock()
    manager._clientset_getter = lambda: client
    client.model_instances.get_cached.return_value = sglang_component_instance
    return manager, client, sglang_component_instance


@pytest.mark.parametrize('cached', [True, False])
def test_component_getter_resolves_smg_without_rank_from_watch(
    component_getter, cached
):
    manager, client, instance = component_getter
    if cached:
        manager._model_instance_by_instance_id[instance.id] = instance.model_copy(
            deep=True
        )
    assert (
        getattr(
            manager.get_instance_endpoint_by_model_instance_id(
                7,
                model_id=1,
            ),
            'port',
            None,
        )
        == 42001
    )
    client.model_instances.get_cached.assert_called_once_with(7)


@pytest.mark.parametrize('change', ['attempt', 'generation', 'worker', 'model'])
def test_component_getter_rejects_obsolete_or_foreign_smg(component_getter, change):
    manager, client, instance = component_getter
    local = instance.model_copy(deep=True)
    manager._model_instance_by_instance_id[instance.id] = local
    model_id = 1
    if change == 'attempt':
        local.distributed_servers.workloads[1].component.attempt = 1
    elif change == 'generation':
        local.restart_count = 2
    elif change == 'worker':
        manager._worker_id_getter = lambda: 2
    else:
        model_id = 2
    assert (
        manager.get_instance_endpoint_by_model_instance_id(7, model_id=model_id) is None
    )
    client.model_instances.get_cached.assert_called_once_with(7)


@pytest.mark.parametrize('rank', [0, 3, 4, -1])
def test_component_getter_rejects_engine_rank_during_smg_starting(
    component_getter, rank
):
    manager, client, instance = component_getter
    instance.state = 'starting'
    instance.distributed_servers.workloads[1].state = 'failed'
    instance.distributed_servers.workloads[1].component.ready = False
    assert (
        manager.get_instance_endpoint_by_model_instance_id(
            7,
            model_id=1,
            dp_rank=rank,
        )
        is None
    )
    assert (
        manager.get_instance_endpoint_by_model_instance_id(
            7,
            model_id=1,
        )
        is None
    )


def test_versioned_getter_cannot_fall_back_to_cached_legacy_port(component_getter):
    manager, client, instance = component_getter
    instance.distributed_servers = None
    manager._model_instance_by_instance_id[instance.id] = instance
    assert (
        getattr(
            manager.get_instance_endpoint_by_model_instance_id(
                7, model_id=1, dp_rank=0
            ),
            'port',
            None,
        )
        is None
    )
    client.model_instances.get_cached.assert_called_once_with(7)
    client.model_instances.get_cached.reset_mock()
    assert (
        getattr(
            manager.get_instance_endpoint_by_model_instance_id(7, model_id=1),
            'port',
            None,
        )
        == 49999
    )
    client.model_instances.get.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('component', ['smg-a2', 'engine-dp3', 'hybrid-w1'])
@pytest.mark.parametrize(
    'token,status',
    [('worker-token', 200), ('registration-token', 200), ('user-token', 401)],
)
async def test_component_middleware_carries_validated_target_through_existing_auth(
    component_getter, monkeypatch, component, token, status
):
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from fastapi import Depends

    if component == 'engine-dp3':
        status = 404
    manager, authority, instance = component_getter
    if component == 'hybrid-w1':
        authority.model_instances.get_cached.return_value = hybrid_instance.__wrapped__(
            instance
        )
    app = FastAPI()
    register_handlers(app)
    app.state.token = 'worker-token'
    app.state.config = SimpleNamespace(token='registration-token')
    app.middleware('http')(worker_proxy.set_port_from_model_name)
    getter = MagicMock(side_effect=manager.get_instance_endpoint_by_model_instance_id)
    app.state.get_instance_endpoint_by_model_instance_id = getter
    forwards = []

    @app.get('/proxy/v1/models', dependencies=[Depends(worker_proxy_auth)])
    def target(request: Request):
        parsed = request.state.model_instance_endpoint
        forwards.append(parsed)
        return dict(
            port=request.state.x_target_port,
            model_id=parsed.model_id,
            instance_id=parsed.instance_id,
            restart_count=parsed.restart_count,
            dp_rank=parsed.dp_rank,
            component=parsed.component,
            attempt=parsed.attempt,
        )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url='http://worker'
    ) as client:
        response = await client.get(
            '/v1/models',
            headers={
                'X-GPUStack-Model-Instance': {
                    'smg-a2': 'model-1-7.static',
                    'engine-dp3': 'model-1-7-dp3.static',
                    'hybrid-w1': 'model-1-7-w1.static',
                }[component],
                'Authorization': f'Bearer {token}',
                'X-Higress-Llm-Model': 'sg',
            },
        )
    assert response.status_code == status
    authority.model_instances.get_cached.assert_called_once_with(7)
    expected = dict(
        model_id=1,
        restart_count=3,
        dp_rank=3 if component == 'engine-dp3' else None,
        component={
            'smg-a2': 'sglang_smg',
            'engine-dp3': 'sglang_engine',
            'hybrid-w1': 'vllm_hybrid',
        }[component],
        attempt=2 if component == 'smg-a2' else None,
    )
    getter.assert_called_once_with(
        7,
        model_id=1,
        dp_rank=expected['dp_rank'],
        worker_id=1 if component == 'hybrid-w1' else None,
    )
    assert len(forwards) == int(status == 200)
    if status == 200:
        assert response.json() == dict(
            port={'smg-a2': '42001', 'engine-dp3': '42000', 'hybrid-w1': '41001'}[
                component
            ],
            instance_id=7,
            **expected,
        )


@pytest.mark.parametrize('changed', ['attempt', 'generation'])
def test_component_getter_ignores_cached_obsolete_identity(component_getter, changed):
    manager, client, instance = component_getter
    manager._model_instance_by_instance_id[instance.id] = instance.model_copy(deep=True)
    if changed == 'attempt':
        instance.distributed_servers.workloads[1].component.attempt = 3
    else:
        instance.restart_count = 4
    assert (
        getattr(
            manager.get_instance_endpoint_by_model_instance_id(
                7,
                model_id=1,
            ),
            'port',
            None,
        )
        is None
    )
    client.model_instances.get_cached.assert_called_once_with(7)


@pytest.mark.asyncio
async def test_component_middleware_does_not_forward_unavailable_default_smg(
    component_getter,
):
    from unittest.mock import MagicMock

    manager, authority, instance = component_getter
    instance.state = 'starting'
    instance.distributed_servers.workloads[1].component.ready = False
    app = FastAPI()
    register_handlers(app)
    app.middleware('http')(worker_proxy.set_port_from_model_name)
    app.state.get_instance_endpoint_by_model_instance_id = (
        manager.get_instance_endpoint_by_model_instance_id
    )
    forward = MagicMock()

    @app.get('/proxy/v1/models')
    def target():
        forward()
        return {}

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url='http://worker'
    ) as client:
        response = await client.get(
            '/v1/models',
            headers={'X-GPUStack-Model-Instance': 'model-1-7.static'},
        )
    assert response.status_code == 404
    forward.assert_not_called()
    authority.model_instances.get_cached.assert_called_once_with(7)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'component', ['sglang_smg', 'sglang_engine', 'vllm_hybrid', None]
)
@pytest.mark.parametrize('chunked', [False, True])
async def test_component_proxy_preserves_native_unicode_body(component, chunked):
    from gpustack.utils.model_instance_workers import ModelInstanceEndpoint

    received = []

    async def handler(request):
        raw = await request.read()
        received.append(
            (
                json.loads(raw),
                {key.lower(): value for key, value in request.headers.items()},
                raw,
            )
        )
        return web.json_response({'ok': True})

    async with _fake_inference_server(handler) as port:
        async with _worker_client(port) as (client, app):
            if component:
                app.state.routed_target = ModelInstanceEndpoint(
                    1,
                    7,
                    1,
                    port,
                    3,
                    1 if component == 'sglang_engine' else None,
                    component,
                    2 if component == 'sglang_smg' else None,
                )
            payload = {'prompt': '你好', 'routed_dp_rank': 9, 'data_parallel_rank': 8}
            raw = json.dumps(payload, ensure_ascii=False).encode()

            async def chunks():
                yield raw[:5]
                yield raw[5:]

            response = await client.post(
                '/v1/chat/completions',
                content=chunks() if chunked else raw,
                headers={
                    'Content-Type': 'application/json',
                    'X-Data-Parallel-Rank': '9',
                    'X-SMG-Routing-Key': 'conversation-123',
                },
            )
    assert response.status_code == 200
    data, headers, sent = received[0]
    assert headers['x-smg-routing-key'] == 'conversation-123'
    assert data == payload
    assert headers['x-data-parallel-rank'] == '9'
    assert sent == raw


@pytest.mark.asyncio
@pytest.mark.parametrize('chunked', [False, True])
@pytest.mark.parametrize('body', [b'not-json', b'x' * 33])
@pytest.mark.parametrize('component', ['sglang_smg', 'sglang_engine'])
async def test_component_proxy_preserves_body_and_backend_error(
    component, chunked, body
):
    from gpustack.utils.model_instance_workers import ModelInstanceEndpoint

    received = []
    error = {'error': {'message': 'native backend validation error'}}

    async def handler(request):
        received.append(await request.read())
        return web.json_response(error, status=422)

    async def chunks():
        yield body[:3]
        yield body[3:]

    async with _fake_inference_server(handler) as port:
        async with _worker_client(port) as (client, app):
            app.state.routed_target = ModelInstanceEndpoint(
                1,
                7,
                1,
                port,
                3,
                component=component,
                attempt=2 if component == 'sglang_smg' else None,
            )
            response = await client.post(
                '/v1/chat/completions',
                content=chunks() if chunked else body,
                headers={'Content-Type': 'application/octet-stream'},
            )
    assert received == [body]
    assert response.status_code == 422
    assert response.json() == error


@pytest.mark.asyncio
async def test_removed_engine_rank_route_does_not_forward_during_smg_recovery(
    component_getter,
):
    from gpustack.schemas.models import (
        ModelInstanceStateEnum,
        ModelInstanceWorkloadStateEnum,
    )

    manager, clientset, instance = component_getter
    instance.state = ModelInstanceStateEnum.STARTING
    main = next(
        w
        for w in instance.distributed_servers.workloads
        if w.component.kind == 'sglang_engine' and w.component.api_port is not None
    )
    smg = next(
        w
        for w in instance.distributed_servers.workloads
        if w.component.kind == 'sglang_smg'
    )
    smg.state = ModelInstanceWorkloadStateEnum.FAILED
    smg.component.ready = False
    smg.component.stop_requested = True
    received = []

    async def handler(request):
        received.append(
            (await request.json(), request.headers.get('X-Data-Parallel-Rank'))
        )
        return web.json_response({'ok': True})

    async with _fake_inference_server(handler) as port:
        main.component.api_port = port
        app = FastAPI()
        register_handlers(app)
        app.include_router(worker_proxy.router)
        app.dependency_overrides[worker_proxy_auth] = lambda: None
        app.state.worker_ip_getter = lambda: '127.0.0.1'
        app.state.get_instance_endpoint_by_model_instance_id = (
            manager.get_instance_endpoint_by_model_instance_id
        )
        app.middleware('http')(worker_proxy.set_port_from_model_name)
        async with ClientSession() as upstream:
            app.state.http_client = app.state.http_client_no_proxy = upstream
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url='http://worker'
            ) as client:
                response = await client.post(
                    '/v1/completions',
                    headers={
                        'X-GPUStack-Model-Instance': f'model-{instance.model_id}-{instance.id}-dp1.static',
                        'X-Data-Parallel-Rank': '9',
                    },
                    json={'prompt': 'hi', 'routed_dp_rank': 8, 'data_parallel_rank': 9},
                )
    assert response.status_code == 404
    assert received == []
    clientset.model_instances.get_cached.assert_called_once_with(7)


@pytest.mark.parametrize('watching', [False, True])
@pytest.mark.parametrize('present', [False, True])
def test_managed_routing_uses_only_live_watch_cache(rank_instance, watching, present):
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from gpustack.api.exceptions import ServiceUnavailableException
    from gpustack.client.model_instance_client import ModelInstanceClient
    from gpustack.schemas.models import ModelInstancePublic
    from gpustack.worker.serve_manager import ServeManager

    transport = MagicMock()
    client = ModelInstanceClient(transport)
    client._watch_started = watching
    from datetime import datetime, timezone

    cached = ModelInstancePublic.model_validate(
        rank_instance,
        update={
            'created_at': datetime.now(timezone.utc),
            'updated_at': datetime.now(timezone.utc),
        },
    )
    client._cache = {7: cached} if present else {}
    manager = ServeManager.__new__(ServeManager)
    manager._worker_id_getter = lambda: 1
    manager._clientset_getter = lambda: SimpleNamespace(model_instances=client)
    # A lifecycle snapshot cannot override watch deletion or disconnect.
    manager._model_instance_by_instance_id = {7: rank_instance}
    if watching:
        assert getattr(
            manager.get_instance_endpoint_by_model_instance_id(
                7, model_id=1, dp_rank=1
            ),
            'port',
            None,
        ) == (41001 if present else None)
        cached.restart_count = 4
        assert (
            getattr(
                manager.get_instance_endpoint_by_model_instance_id(
                    7, model_id=1, dp_rank=1
                ),
                'port',
                None,
            )
            is None
        )
    else:
        with pytest.raises(ServiceUnavailableException):
            getattr(
                manager.get_instance_endpoint_by_model_instance_id(
                    7, model_id=1, dp_rank=1
                ),
                'port',
                None,
            )
    transport.get_httpx_client.assert_not_called()


@pytest.mark.asyncio
async def test_rank_middleware_reports_unsynchronized_watch_as_503():
    from unittest.mock import MagicMock
    from gpustack.api.exceptions import ServiceUnavailableException

    app = FastAPI()
    app.middleware('http')(worker_proxy.set_port_from_model_name)
    app.state.get_instance_endpoint_by_model_instance_id = MagicMock(
        side_effect=ServiceUnavailableException(message='Routing cache unavailable')
    )
    forwarded = MagicMock()

    @app.get('/proxy/v1/models')
    async def target():
        forwarded()
        return {}

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url='http://worker'
    ) as client:
        response = await client.get(
            '/v1/models',
            headers={'X-GPUStack-Model-Instance': 'model-1-7-dp1.static'},
        )
    assert response.status_code == 503
    forwarded.assert_not_called()


@pytest.mark.parametrize(
    'local_change', ['generation', 'attempt', 'stopping', 'draining', 'error']
)
def test_routing_watch_cannot_override_newer_local_runtime_identity(
    component_getter, local_change
):
    manager, client, instance = component_getter
    local = instance.model_copy(deep=True)
    manager._model_instance_by_instance_id[instance.id] = local
    if local_change == 'generation':
        local.restart_count += 1
    elif local_change == 'attempt':
        local.distributed_servers.workloads[1].component.attempt += 1
    else:
        manager._managed_stopping = {(instance.id, instance.restart_count): False}
    assert (
        getattr(
            manager.get_instance_endpoint_by_model_instance_id(
                instance.id,
                model_id=1,
            ),
            'port',
            None,
        )
        is None
    )
    client.model_instances.get.assert_not_called()


@pytest.fixture
def hybrid_instance(rank_instance):
    from gpustack.schemas.models import (
        ModelInstanceSubordinateWorker,
        ModelInstanceWorkload,
    )

    rank_instance.distributed_servers.subordinate_workers = [
        ModelInstanceSubordinateWorker(worker_id=worker_id) for worker_id in (2, 3)
    ]
    rank_instance.distributed_servers.workloads = [
        ModelInstanceWorkload(
            name=f'engine-{worker_id}',
            worker_id=worker_id,
            state='running',
            component={
                'kind': 'vllm_hybrid',
                'api_port': 41000 + worker_id if worker_id != 3 else None,
                'dp_size': 2 if worker_id != 3 else None,
                'ready': True if worker_id != 3 else None,
            },
        )
        for worker_id in (1, 2, 3)
    ]
    rank_instance.distributed_servers.orchestration_mode = "managed"
    return rank_instance


def test_hybrid_target_identity_roundtrip_and_actual_api_owners(hybrid_instance):
    from gpustack.utils.model_instance_workers import (
        model_instance_endpoints,
        parse_model_instance_target,
        resolve_model_instance_endpoint,
    )

    endpoints = model_instance_endpoints(hybrid_instance)
    assert [endpoint.registry_name for endpoint in endpoints] == [
        'model-1-7-w1',
        'model-1-7-w2',
    ]
    for endpoint in endpoints:
        target = parse_model_instance_target(endpoint.registry_name + '.static')
        assert target.worker_id == endpoint.worker_id
        assert target.dp_rank is None
        assert resolve_model_instance_endpoint(hybrid_instance, target) == endpoint
        assert (
            resolve_model_instance_endpoint(
                hybrid_instance, target, worker_id=endpoint.worker_id
            )
            == endpoint
        )
        with pytest.raises(ValueError):
            resolve_model_instance_endpoint(hybrid_instance, target, worker_id=3)
    for name in (
        'model-1-7-r2-hybrid-w1.static',
        'model-1-7-w3.static',
        'model-2-7-r3-hybrid-w1.static',
        'model-1-7.static',
    ):
        with pytest.raises(ValueError):
            resolve_model_instance_endpoint(
                hybrid_instance, parse_model_instance_target(name)
            )


@pytest.mark.parametrize(
    'change', ['headless', 'unready', 'generation', 'worker', 'stopping', 'local_epoch']
)
def test_hybrid_getter_checks_watch_worker_and_local_fences(hybrid_instance, change):
    from unittest.mock import MagicMock
    from gpustack.worker.serve_manager import ServeManager

    manager = ServeManager.__new__(ServeManager)
    manager._worker_id_getter = lambda: 2
    manager._model_instance_by_instance_id = {7: hybrid_instance.model_copy(deep=True)}
    client = MagicMock()
    manager._clientset_getter = lambda: client
    client.model_instances.get_cached.return_value = hybrid_instance
    kwargs = dict(model_id=1, restart_count=3, component='vllm_hybrid', worker_id=2)
    assert (
        getattr(
            manager.get_instance_endpoint_by_model_instance_id(
                7,
                model_id=kwargs['model_id'],
                dp_rank=kwargs.get('dp_rank'),
                worker_id=kwargs.get('worker_id'),
            ),
            'port',
            None,
        )
        == 41002
    )
    if change == 'headless':
        manager._worker_id_getter = lambda: 3
        kwargs['worker_id'] = 3
    elif change == 'unready':
        hybrid_instance.distributed_servers.workloads[0].component.ready = False
    elif change == 'generation':
        hybrid_instance.restart_count = 4
    elif change == 'worker':
        kwargs['worker_id'] = 1
    elif change == 'stopping':
        manager._managed_stopping = {(7, 3): True}
    else:
        manager._model_instance_by_instance_id[7].restart_count = 2
    assert (
        getattr(
            manager.get_instance_endpoint_by_model_instance_id(
                7,
                model_id=kwargs['model_id'],
                dp_rank=kwargs.get('dp_rank'),
                worker_id=kwargs.get('worker_id'),
            ),
            'port',
            None,
        )
        is None
    )
