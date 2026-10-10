from unittest.mock import MagicMock

import pytest

from gpustack.server.worker_request import _stream_response_chunks


class _FakeContent:
    def __init__(self, chunks):
        self._chunks = chunks

    async def iter_chunked(self, _size):
        for chunk in self._chunks:
            yield chunk


def _worker_response(*chunks: bytes):
    resp = MagicMock()
    resp.content = _FakeContent(chunks)
    return resp


async def _relay(*chunks: bytes) -> str:
    return "".join(
        [c async for c in _stream_response_chunks(_worker_response(*chunks))]
    )


RESPONSES_STREAM = (
    b'event: response.created\ndata: {"type":"response.created"}\n\n'
    b'event: response.output_text.delta\n'
    b'data: {"type":"response.output_text.delta","delta":"hi"}\n\n'
    b'event: response.completed\ndata: {"type":"response.completed"}\n\n'
)


@pytest.mark.asyncio
async def test_named_events_keep_event_line_with_their_data():
    relayed = await _relay(RESPONSES_STREAM)

    assert relayed == RESPONSES_STREAM.decode()


@pytest.mark.asyncio
async def test_named_events_survive_chunk_boundaries_inside_a_line():
    pieces = [RESPONSES_STREAM[i : i + 7] for i in range(0, len(RESPONSES_STREAM), 7)]

    relayed = await _relay(*pieces)

    assert relayed == RESPONSES_STREAM.decode()


@pytest.mark.asyncio
async def test_event_id_and_retry_fields_stay_in_the_event_they_describe():
    upstream = b'id: 7\nretry: 1000\nevent: message\ndata: {"a":1}\n\n'

    relayed = await _relay(upstream)

    assert relayed == upstream.decode()


@pytest.mark.asyncio
async def test_data_only_stream_still_gets_one_event_per_line():
    upstream = b'data: {"id":"1"}\ndata: {"id":"2"}\n\ndata: [DONE]\n\n'

    relayed = await _relay(upstream)

    assert relayed == 'data: {"id":"1"}\n\ndata: {"id":"2"}\n\ndata: [DONE]\n\n'


@pytest.mark.asyncio
async def test_comment_line_inside_an_event_does_not_end_it():
    upstream = b'event: message\n: keep-alive\ndata: {"a":1}\n\n'

    relayed = await _relay(upstream)

    assert relayed == upstream.decode()


@pytest.mark.asyncio
async def test_comment_line_between_events_stays_a_comment():
    upstream = b'data: {"id":"1"}\n\n: ping\n\ndata: {"id":"2"}\n\n'

    relayed = await _relay(upstream)

    assert relayed == 'data: {"id":"1"}\n\n: ping\ndata: {"id":"2"}\n\n'
