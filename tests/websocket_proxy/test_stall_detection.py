"""Tests for stall detection on the WebSocket tunnel response path (#6012).

In tunnel proxy mode the gateway talks to the server's HTTP proxy port, which
relays raw bytes to the instance through the tunnel. ``wait_for_complete_response``
is the only place in that path holding a timeout, so it is where a
stopped-but-alive serving process has to be detected and reported.

The same proxy port also carries internal worker APIs (log following, for one),
which is why the segmented budget is opt-in per request rather than global: the
tests below pin both that gating and the behavior when it is on.
"""

import asyncio
from typing import List, Optional, Tuple, Union

import pytest

from gpustack.websocket_proxy.connection import IOConnection
from gpustack.websocket_proxy.proxy_server import (
    StallPolicy,
    _parse_response_head,
    wait_for_complete_response,
)

IDLE_TIMEOUT = 0.2
TTFT_TIMEOUT = 0.6
# Longer than either budget: a legitimate wait that must survive both.
SLOW_GENERATION = 1.0
# Bounded so a leaked step cannot hang the suite.
STALL = 2

ERROR_BODY = b'{"error":{"message":"stalled"}}'
ERROR_SSE = b'data: {"error":{"message":"stalled"}}\n\ndata: [DONE]\n\n'

SSE_HEAD = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n"
JSON_HEAD = (
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 5\r\n\r\n"
)
LOG_HEAD = (
    b"HTTP/1.1 200 OK\r\n"
    b"Content-Type: application/octet-stream\r\n"
    b"Transfer-Encoding: chunked\r\n\r\n"
)


def policy(total_timeout: float = 30) -> StallPolicy:
    return StallPolicy(
        ttft_timeout=TTFT_TIMEOUT,
        idle_timeout=IDLE_TIMEOUT,
        total_timeout=total_timeout,
        error_body=ERROR_BODY,
        error_sse=ERROR_SSE,
    )


class ScriptedRemote(IOConnection):
    """Upstream side of the tunnel, driven by scripted ``(delay, chunk)`` steps.

    A ``None`` chunk stalls: the connection stays open and no data arrives, which
    is the failure shape a wedged-but-alive serving process produces. Exhausting
    the steps returns b"" (EOF). The timeout each read was given is recorded, so
    a test can assert which budget was in force.
    """

    def __init__(self, steps: List[Tuple[float, Union[bytes, None]]]):
        self._steps = list(steps)
        self.timeouts: List[Optional[float]] = []

    async def read(self, n: int = -1, timeout: Optional[float] = None) -> bytes:
        self.timeouts.append(timeout)

        async def _next() -> bytes:
            if not self._steps:
                return b""
            delay, chunk = self._steps.pop(0)
            if chunk is None:
                await asyncio.sleep(STALL)
                return b""
            await asyncio.sleep(delay)
            return chunk

        if timeout is None:
            return await _next()
        return await asyncio.wait_for(_next(), timeout)

    async def write(self, data: bytes) -> None:  # pragma: no cover - unused
        raise AssertionError("the response relay never writes to the remote")

    async def close(self) -> None:
        pass


class RecordingClient(IOConnection):
    """Downstream side: records everything written back to the client."""

    def __init__(self):
        self.writes: List[bytes] = []
        self.closed = False

    async def read(self, n: int = -1, timeout: Optional[float] = None) -> bytes:
        return b""

    async def write(self, data: bytes) -> None:
        self.writes.append(data)

    async def close(self) -> None:
        self.closed = True

    @property
    def body(self) -> bytes:
        return b"".join(self.writes)


async def _relay(
    steps: List[Tuple[float, Union[bytes, None]]],
    stall_policy: Optional[StallPolicy] = None,
    expect_body: bool = True,
) -> Tuple[RecordingClient, ScriptedRemote]:
    remote = ScriptedRemote(steps)
    client = RecordingClient()
    await wait_for_complete_response(
        remote, client, stall_policy=stall_policy, expect_body=expect_body
    )
    assert client.closed
    return client, remote


class TestStallBeforeFirstBodyByte:
    @pytest.mark.asyncio
    async def test_returns_504_and_never_commits_the_head(self):
        """vLLM sends 200 + text/event-stream before prefill. Forwarding that head
        on arrival would fix the status code before we know whether a first token
        is coming, leaving the stall reportable only in-stream."""
        client, _ = await _relay([(0, SSE_HEAD), (0, None)], policy())

        assert client.body.startswith(b"HTTP/1.1 504 Gateway Timeout\r\n")
        assert client.body.endswith(ERROR_BODY)
        assert b"text/event-stream" not in client.body
        assert f"Content-Length: {len(ERROR_BODY)}".encode() in client.body

    @pytest.mark.asyncio
    async def test_first_chunk_budget_applies_only_after_the_head(self):
        remote = ScriptedRemote([(0, SSE_HEAD), (0, None)])
        client = RecordingClient()
        await wait_for_complete_response(remote, client, stall_policy=policy())

        # Head phase: only the absolute ceiling. Body phase: the first-chunk
        # budget, since a non-streaming response cannot be told apart from a
        # stalled one before its head arrives.
        assert remote.timeouts[0] > TTFT_TIMEOUT
        assert remote.timeouts[1] == pytest.approx(TTFT_TIMEOUT, abs=0.05)

    @pytest.mark.asyncio
    async def test_total_timeout_bounds_a_head_that_never_arrives(self):
        client, _ = await _relay([(0, None)], policy(total_timeout=0.3))
        assert client.body.startswith(b"HTTP/1.1 504 Gateway Timeout\r\n")


class TestStallMidStream:
    @pytest.mark.asyncio
    async def test_appends_a_terminal_error_event(self):
        client, _ = await _relay(
            [(0, SSE_HEAD), (0, b"data: one\n\n"), (0, None)], policy()
        )

        assert client.body == SSE_HEAD + b"data: one\n\n" + ERROR_SSE

    @pytest.mark.asyncio
    async def test_inter_chunk_budget_is_in_force(self):
        remote = ScriptedRemote([(0, SSE_HEAD), (0, b"data: one\n\n"), (0, None)])
        client = RecordingClient()
        await wait_for_complete_response(remote, client, stall_policy=policy())

        assert remote.timeouts[-1] == pytest.approx(IDLE_TIMEOUT, abs=0.05)

    @pytest.mark.asyncio
    async def test_non_streaming_response_is_closed_without_an_injected_body(self):
        """Nothing meaningful can be appended to a half-delivered JSON body."""
        client, _ = await _relay(
            [(0, JSON_HEAD), (0, b"ab"), (0, None)], policy(total_timeout=0.5)
        )

        assert client.body == JSON_HEAD + b"ab"


class TestLegitimateSilence:
    @pytest.mark.asyncio
    async def test_long_prefill_is_not_terminated(self):
        client, _ = await _relay(
            [(0, SSE_HEAD), (TTFT_TIMEOUT / 2, b"data: one\n\n")], policy()
        )
        assert client.body == SSE_HEAD + b"data: one\n\n"

    @pytest.mark.asyncio
    async def test_slow_non_streaming_generation_is_not_terminated(self):
        """Its head only arrives once generation has finished, so the head phase
        has to outlast the whole generation."""
        client, _ = await _relay(
            [(SLOW_GENERATION, JSON_HEAD), (0, b"12345")], policy()
        )
        assert client.body == JSON_HEAD + b"12345"

    @pytest.mark.asyncio
    async def test_idle_log_stream_is_not_terminated(self):
        """A followed log stream is legitimately idle for minutes. It reaches this
        relay as a non-event-stream body, so no tight budget may apply even when a
        policy is set."""
        client, remote = await _relay(
            [(0, LOG_HEAD), (0, b"line one\n"), (SLOW_GENERATION, b"line two\n")],
            policy(),
        )

        assert client.body == LOG_HEAD + b"line one\nline two\n"
        assert all(t > SLOW_GENERATION for t in remote.timeouts)


class TestWithoutPolicy:
    @pytest.mark.asyncio
    async def test_no_timeout_is_imposed(self):
        """Unsegmented traffic keeps the connection's own default bound."""
        client, remote = await _relay(
            [(0, LOG_HEAD), (SLOW_GENERATION, b"line one\n")], stall_policy=None
        )

        assert client.body == LOG_HEAD + b"line one\n"
        assert remote.timeouts == [None, None, None]


class TestResponsesWithoutABody:
    @pytest.mark.asyncio
    async def test_204_head_is_flushed_immediately(self):
        client, remote = await _relay(
            [(0, b"HTTP/1.1 204 No Content\r\n\r\n"), (STALL, b"never read")], policy()
        )

        assert client.body == b"HTTP/1.1 204 No Content\r\n\r\n"
        # Returned right after the head; the trailing step is never reached.
        assert len(remote.timeouts) == 1

    @pytest.mark.asyncio
    async def test_content_length_zero_head_is_flushed_immediately(self):
        head = b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n"
        client, _ = await _relay([(0, head), (STALL, b"never read")], policy())
        assert client.body == head

    @pytest.mark.asyncio
    async def test_head_request_does_not_wait_for_a_body(self):
        client, _ = await _relay(
            [(0, JSON_HEAD), (STALL, b"never read")], policy(), expect_body=False
        )
        assert client.body == JSON_HEAD

    @pytest.mark.asyncio
    async def test_upstream_closing_after_the_head_still_forwards_it(self):
        """The head is held back, not dropped: a response that turns out to have
        no body must still reach the client."""
        client, _ = await _relay([(0, LOG_HEAD)], policy())
        assert client.body == LOG_HEAD


class TestUnchangedRelayBehavior:
    @pytest.mark.asyncio
    async def test_content_length_caps_the_forwarded_body(self):
        client, _ = await _relay([(0, JSON_HEAD), (0, b"12345trailing")], policy())
        assert client.body == JSON_HEAD + b"12345"

    @pytest.mark.asyncio
    async def test_head_split_across_reads(self):
        client, _ = await _relay(
            [
                (0, b"HTTP/1.1 200 OK\r\nContent-Type: text/event"),
                (0, b"-stream\r\n\r\n"),
                (0, b"data: one\n\n"),
            ],
            policy(),
        )
        assert client.body == SSE_HEAD + b"data: one\n\n"

    @pytest.mark.asyncio
    async def test_chunked_body_streams_until_upstream_closes(self):
        client, _ = await _relay(
            [(0, LOG_HEAD), (0, b"9\r\nline one\n\r\n"), (0, b"0\r\n\r\n")], policy()
        )
        assert client.body == LOG_HEAD + b"9\r\nline one\n\r\n0\r\n\r\n"


class TestParseResponseHead:
    def test_event_stream_detected_case_insensitively(self):
        content_length, is_event_stream, status_code = _parse_response_head(
            b"HTTP/1.1 200 OK\r\nCONTENT-TYPE: Text/Event-Stream\r\n\r\n"
        )
        assert (content_length, is_event_stream, status_code) == (None, True, 200)

    def test_content_length_and_status(self):
        assert _parse_response_head(JSON_HEAD) == (5, False, 200)

    def test_malformed_content_length_falls_back_to_streaming(self):
        content_length, _, status_code = _parse_response_head(
            b"HTTP/1.1 200 OK\r\nContent-Length: abc\r\n\r\n"
        )
        assert content_length is None
        assert status_code == 200

    def test_unparseable_status_line(self):
        assert _parse_response_head(b"garbage\r\n\r\n") == (None, False, 0)
