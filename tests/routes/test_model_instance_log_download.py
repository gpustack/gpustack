"""Route tests for downloading a model instance's serving logs.

The endpoint's shape follows how many (worker, container) log streams it finds,
so these drive the route directly with the worker calls faked. The body is
consumed inside the patch scope: past the first answer a single stream needs
for its length, it only talks to the workers once iterated.
"""

import contextlib
import io
import sys
import zipfile
from types import SimpleNamespace
from typing import Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from gpustack.routes.model_instances import download_serving_logs, get_serving_logs
from gpustack.schemas.models import (
    BackendEnum,
    ModelInstanceLogRestartEntry,
    ModelInstanceStateEnum,
    ServeLogOptionsResponse,
)
from gpustack.server import worker_request
from gpustack.worker.logs import LogOptions

MODULE = "gpustack.routes.model_instances"


@contextlib.asynccontextmanager
async def _fake_async_session():
    yield MagicMock()


def _instance(name="llama-abc"):
    return SimpleNamespace(
        id=7,
        name=name,
        worker_id=1,
        state=ModelInstanceStateEnum.RUNNING,
        model_files=[],
        backend=BackendEnum.VLLM,
        model=None,
        distributed_servers=None,
    )


def _target(worker_id, name, present=True):
    worker = SimpleNamespace(id=worker_id, name=name) if present else None
    return (worker_id, name, worker)


async def _download(instance, targets, *, options, logs, calls=None):
    """Run the endpoint and return (response, body).

    ``options`` maps a worker name to its container list or to the exception its
    discovery raises; ``logs`` maps (worker name, internal container name) to
    (status, chunks) or (status, chunks, headers) — the main workload's internal
    name is "default", and an exception among the chunks is raised there.
    ``calls`` collects every kwargs dict the worker was called with. A body
    that breaks off is returned as far as it got, with the error on
    ``response.transfer_error``.
    """

    async def fake_options(_request, worker, _instance_id):
        planned = options[worker.name]
        if isinstance(planned, Exception):
            raise planned
        return ServeLogOptionsResponse(
            restarts=[
                ModelInstanceLogRestartEntry(previous=False, containers=planned),
                ModelInstanceLogRestartEntry(previous=True, containers=["stale"]),
            ]
        )

    fetched = []

    async def fake_stream(**kwargs):
        container = kwargs["params"].get("container_name")
        fetched.append((kwargs["worker"].name, container))
        if calls is not None:
            calls.append(kwargs)
        status_code, chunks, *headers = logs[(kwargs["worker"].name, container)]
        for chunk in chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk, (headers[0] if headers else {}), status_code

    with (
        patch(f"{MODULE}.async_session", _fake_async_session),
        patch(f"{MODULE}.fetch_model_instance", AsyncMock(return_value=instance)),
        patch(
            f"{MODULE}.resolve_instance_log_worker_targets",
            AsyncMock(return_value=targets),
        ),
        patch(f"{MODULE}.fetch_serve_log_options_from_worker", fake_options),
        patch(f"{MODULE}.stream_to_worker", fake_stream),
    ):
        response = await download_serving_logs(
            request=MagicMock(), ctx=MagicMock(), id=7
        )
        # At most a single stream's first answer, which carries its length:
        # iterating the body is what pulls each log.
        assert len(fetched) <= 1
        chunks = []
        response.transfer_error = None
        try:
            async for chunk in response.body_iterator:
                chunks.append(chunk if isinstance(chunk, bytes) else chunk.encode())
        except Exception as e:
            response.transfer_error = e
    return response, b"".join(chunks)


async def _serve_logs(instance, *, follow, chunks, status_code=200, consume=True):
    """Run the serving-logs endpoint and return (response, body, fetch journal).

    The body is consumed inside the patch scope: nothing reaches the worker
    until it is iterated, which is what "streamed rather than buffered" means
    from the caller's side. Pass consume=False to drive the response through
    ASGI instead, which is the only path that promotes the status code.
    """
    fetched = []

    async def fake_stream(**kwargs):
        fetched.append(kwargs["params"])
        for chunk in chunks:
            yield chunk, {"content-type": "application/octet-stream"}, status_code

    with (
        patch(f"{MODULE}.async_session", _fake_async_session),
        patch(f"{MODULE}.fetch_model_instance", AsyncMock(return_value=instance)),
        patch(
            f"{MODULE}.fetch_worker",
            AsyncMock(return_value=SimpleNamespace(id=1, name="worker-a")),
        ),
        patch(f"{MODULE}.stream_to_worker", fake_stream),
    ):
        response = await get_serving_logs(
            request=MagicMock(),
            ctx=MagicMock(),
            id=7,
            log_options=LogOptions(tail=-1, follow=follow),
        )
        assert fetched == []
        body = b""
        if consume:
            async for chunk, _headers, _status in response.body_iterator:
                body += chunk
    return response, body, fetched


@pytest.mark.asyncio
@pytest.mark.parametrize("follow", [False, True])
async def test_serving_logs_are_streamed_not_buffered(follow):
    """A full read used to be pulled into the server process before any of it
    reached the client, which puts a whole serving log in memory. Both modes
    now hand the worker's bytes straight through, unchanged."""
    response, body, fetched = await _serve_logs(
        _instance(),
        follow=follow,
        chunks=[b"first chunk\n", b"second chunk\n"],
    )

    assert body == b"first chunk\nsecond chunk\n"
    assert fetched[0]["follow"] is follow
    assert response.media_type == "application/octet-stream"


async def _send_asgi(response) -> Tuple[int, bytes]:
    """Drive the response through ASGI and return its status and body.

    The status rides the first yielded chunk and is only promoted once the
    response is actually sent, so reading body_iterator alone would never see it.
    """
    messages = []

    async def send(message):
        messages.append(message)

    await response.stream_response(send)
    start = next(m for m in messages if m["type"] == "http.response.start")
    body = b"".join(
        m.get("body", b"") for m in messages if m["type"] == "http.response.body"
    )
    return start["status"], body


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [404, 500])
async def test_serving_logs_keep_a_worker_failure_intact(status_code):
    """The status and the body the worker answered with both have to survive the
    extra hop, or a caller cannot tell a failure from an empty log."""
    response, _body, _fetched = await _serve_logs(
        _instance(),
        follow=False,
        chunks=[b'{"message":"Log file not found"}'],
        status_code=status_code,
        consume=False,
    )

    sent_status, sent_body = await _send_asgi(response)

    assert sent_status == status_code
    assert sent_body == b'{"message":"Log file not found"}'


class _FakeWorkerResponse:
    """Stand-in for the worker's aiohttp response, yielding fixed chunks."""

    def __init__(self, status_code: int, chunks: list):
        self.status = status_code
        self.headers = {"content-type": "application/octet-stream"}
        self._chunks = chunks

    @property
    def content(self):
        outer = self

        class _Content:
            async def iter_any(self):
                for chunk in outer._chunks:
                    yield chunk

        return _Content()

    async def read(self):
        return b"".join(self._chunks)


async def _serve_logs_for_real(instance, *, chunks, status_code=200):
    """Run the endpoint over the real stream_to_worker, faking only the HTTP hop.

    The guarantee under test lives inside stream_to_worker, so a test that
    replaces it -- as the others here do -- cannot see this class of bug at all.
    """

    @contextlib.asynccontextmanager
    async def fake_request(**_kwargs):
        yield _FakeWorkerResponse(status_code, chunks)

    with (
        patch(f"{MODULE}.async_session", _fake_async_session),
        patch(f"{MODULE}.fetch_model_instance", AsyncMock(return_value=instance)),
        patch(
            f"{MODULE}.fetch_worker",
            AsyncMock(return_value=SimpleNamespace(id=1, name="worker-a")),
        ),
        patch.object(worker_request, "_request_to_worker", fake_request),
    ):
        response = await get_serving_logs(
            request=MagicMock(),
            ctx=MagicMock(),
            id=7,
            log_options=LogOptions(tail=-1, follow=False),
        )
        return await _send_asgi(response)


@pytest.mark.asyncio
async def test_an_empty_log_reads_as_an_empty_log():
    """An instance whose log file exists but holds nothing yet answers 200 with
    no bytes. Streaming that as zero chunks leaves the response with nothing to
    take a status from, which surfaces as "service unavailable" -- the everyday
    case of opening the logs of a freshly started instance."""
    sent_status, sent_body = await _serve_logs_for_real(_instance(), chunks=[])

    assert sent_status == 200
    assert sent_body == b""


@pytest.mark.asyncio
async def test_a_non_empty_log_still_streams_every_chunk():
    sent_status, sent_body = await _serve_logs_for_real(
        _instance(), chunks=[b"first\n", b"second\n"]
    )

    assert sent_status == 200
    assert sent_body == b"first\nsecond\n"


@pytest.mark.asyncio
async def test_serving_logs_pass_bytes_no_decode_can_mangle():
    """Logs carry whatever the backend printed. Decoding them on the way through
    would raise on the first invalid sequence; streaming bytes cannot."""
    _response, body, _fetched = await _serve_logs(
        _instance(),
        follow=False,
        chunks=[b"before \xff\xfe after\n"],
    )

    assert body == b"before \xff\xfe after\n"


@pytest.mark.asyncio
async def test_single_stream_downloads_plain_text_log():
    response, body = await _download(
        _instance(),
        [_target(1, "worker-a")],
        options={"worker-a": ["default"]},
        logs={("worker-a", "default"): (200, [b"first chunk\n", b"second chunk\n"])},
    )

    assert response.media_type == "text/plain; charset=utf-8"
    assert (
        response.headers["content-disposition"]
        == 'attachment; filename="llama-abc.log"'
    )
    assert body == b"first chunk\nsecond chunk\n"


@pytest.mark.asyncio
async def test_non_ascii_instance_name_stays_header_encodable():
    # Starlette encodes headers as latin-1, so a bare filename= would raise.
    response, _ = await _download(
        _instance(name="模型-abc"),
        [_target(1, "worker-a")],
        options={"worker-a": ["default"]},
        logs={("worker-a", "default"): (200, [b"log\n"])},
    )

    disposition = response.headers["content-disposition"]
    assert disposition == (
        "attachment; filename=\"__-abc.log\"; filename*=UTF-8''%E6%A8%A1%E5%9E%8B-abc.log"
    )
    disposition.encode("latin-1")


@pytest.mark.asyncio
async def test_multiple_streams_download_as_a_zip_with_failures_captured():
    instance = _instance()
    targets = [
        _target(1, "worker-a"),
        _target(2, "worker-b"),
        _target(3, "worker-c", present=False),
    ]
    response, body = await _download(
        instance,
        targets,
        options={
            # "default" displays as the backend name, so this label collides.
            "worker-a": ["default", "vLLM"],
            "worker-b": ValueError("HTTP 404: no log options"),
        },
        logs={
            ("worker-a", "default"): (200, [b"main\n"]),
            ("worker-a", "vLLM"): (200, [b"sidecar\n"]),
            ("worker-b", "default"): (500, [b"boom"]),
        },
    )

    assert response.media_type == "application/zip"
    archive = zipfile.ZipFile(io.BytesIO(body))
    assert archive.testzip() is None
    assert archive.namelist() == [
        "worker-a.vLLM.log",
        "worker-a.vLLM.1.log",
        "worker-b.ray-worker.log",
        "worker-c.default.log",
    ]
    assert archive.read("worker-a.vLLM.log") == b"main\n"
    assert archive.read("worker-a.vLLM.1.log") == b"sidecar\n"
    # A failed discovery is noted, and an error body must not read as real log
    # output.
    worker_b_log = archive.read("worker-b.ray-worker.log")
    assert worker_b_log.startswith(b"Note: container discovery failed (HTTP 404: ")
    assert b"Failed to fetch logs: HTTP 500: boom" in worker_b_log
    assert b"Worker not found in database" in archive.read("worker-c.default.log")


@pytest.mark.asyncio
async def test_discovery_failure_still_serves_the_main_workload():
    # Discovery is only how containers are found: losing it falls back to the
    # main workload, whose logs a single worker can still serve on its own.
    response, body = await _download(
        _instance(),
        [_target(1, "worker-a")],
        options={"worker-a": ValueError("HTTP 404: no log options")},
        logs={("worker-a", "default"): (200, [b"real log\n"])},
    )

    assert response.media_type == "text/plain; charset=utf-8"
    # Noted inline, so a sidecar log missing from the download can't pass for
    # one that never existed.
    assert body == (
        b"Note: container discovery failed (HTTP 404: no log options); "
        b"this log covers the main workload only.\n"
        b"real log\n"
    )


@pytest.mark.asyncio
async def test_no_proxyable_worker_fails_with_502():
    # Not in the database, so nothing can be proxied and no fallback can help.
    with pytest.raises(HTTPException) as raised:
        await _download(
            _instance(), [_target(1, "worker-a", present=False)], options={}, logs={}
        )

    assert raised.value.status_code == 502
    assert "Failed to fetch logs from all workers" in raised.value.detail


_MEASURED = {"x-log-total-bytes": "5"}


@pytest.mark.asyncio
async def test_a_download_states_the_length_the_worker_measured():
    """A browser can only show a progress bar for a download whose size it was
    told. The worker measures the whole stream on the same look at its files
    that its reads are pinned to, so the length it states is the body's."""
    calls = []

    response, body = await _download(
        _instance(),
        [_target(1, "worker-a")],
        options={"worker-a": ["default"]},
        logs={("worker-a", "default"): (200, [b"a\n", b"bb\n"], _MEASURED)},
        calls=calls,
    )

    assert response.headers["content-length"] == "5"
    assert body == b"a\nbb\n"
    params = calls[0]["params"]
    assert (params["offset"], params["limit"]) == (0, sys.maxsize)


@pytest.mark.asyncio
async def test_a_download_cut_short_is_filled_out_rather_than_left_hanging():
    """A file rewritten under the reads leaves the body shorter than the length
    already sent -- which a client reads as a broken connection rather than
    as a log that moved."""
    response, body = await _download(
        _instance(),
        [_target(1, "worker-a")],
        options={"worker-a": ["default"]},
        logs={
            ("worker-a", "default"): (200, [b"a\n"], {"X-Log-Total-Bytes": "80"}),
        },
    )

    assert response.headers["content-length"] == "80"
    assert len(body) == 80
    assert b"the log changed while it was downloading" in body


@pytest.mark.asyncio
@pytest.mark.parametrize("length", [8 * 1024, 4 * 1024 * 1024])
async def test_a_download_that_dies_midway_breaks_rather_than_pads(length):
    """Filling a failed transfer out, or writing the failure into it, would
    hand back a file the client cannot tell from a whole one, however small
    the log; stopping short of the promised length is what tells it the
    download failed."""
    response, body = await _download(
        _instance(),
        [_target(1, "worker-a")],
        options={"worker-a": ["default"]},
        logs={
            ("worker-a", "default"): (
                200,
                [b"a\n", ConnectionResetError("worker went away")],
                {"x-log-total-bytes": str(length)},
            ),
        },
    )

    assert response.headers["content-length"] == str(length)
    assert isinstance(response.transfer_error, ConnectionResetError)
    assert body == b"a\n"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "discovery, answer",
    [
        # A worker from before the range read streams the whole log unmeasured.
        (["default"], (200, [b"a\n", b"bb\n"])),
        # Discovery failed, so a note of its own goes into the file first.
        (ValueError("worker down"), (200, [b"a\n", b"bb\n"], _MEASURED)),
        # The worker refused: its answer is labelled, not passed off as a log.
        (["default"], (500, [b"boom"])),
        # A length that is not one is no length at all.
        (["default"], (200, [b"a\n", b"bb\n"], {"x-log-total-bytes": "-5"})),
    ],
    ids=["older-worker", "discovery-failed", "worker-error", "unreadable-length"],
)
async def test_a_download_that_cannot_be_measured_streams_without_a_length(
    discovery, answer
):
    """Better no progress bar than a wrong one: a length that does not match
    what arrives breaks the download outright."""
    response, body = await _download(
        _instance(),
        [_target(1, "worker-a")],
        options={"worker-a": discovery},
        logs={("worker-a", "default"): answer},
    )

    assert "content-length" not in response.headers
    assert response.transfer_error is None
    if answer[0] >= 400:
        assert body == b"Failed to fetch logs: HTTP 500: boom"
    else:
        assert body.endswith(b"a\nbb\n")


@pytest.mark.asyncio
async def test_a_download_waits_on_liveness_rather_than_on_a_deadline():
    """A log past the size cap's reach takes longer to transfer than any total
    timeout worth setting; what a stalled transfer looks like is silence."""
    calls = []

    await _download(
        _instance(),
        [_target(1, "worker-a")],
        options={"worker-a": ["default"]},
        logs={("worker-a", "default"): (200, [b"a\n"])},
        calls=calls,
    )

    timeout = calls[0]["timeout"]
    assert timeout.total is None
    assert timeout.sock_read == 120
