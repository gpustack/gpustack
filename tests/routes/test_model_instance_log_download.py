"""Route tests for downloading a model instance's serving logs.

The endpoint's shape follows how many (worker, container) log streams it finds,
so these drive the route directly with the worker calls faked. The body is
consumed inside the patch scope: it only talks to the workers once iterated.
"""

import contextlib
import io
import zipfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from gpustack.routes.model_instances import download_serving_logs
from gpustack.schemas.models import (
    BackendEnum,
    ModelInstanceLogRestartEntry,
    ModelInstanceStateEnum,
    ServeLogOptionsResponse,
)

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


async def _download(instance, targets, *, options, logs):
    """Run the endpoint and return (response, body).

    ``options`` maps a worker name to its container list or to the exception its
    discovery raises; ``logs`` maps (worker name, internal container name) to
    (status, chunks) — the main workload's internal name is "default".
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
        status_code, chunks = logs[(kwargs["worker"].name, container)]
        for chunk in chunks:
            yield chunk, {}, status_code

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
        # Nothing fetched yet: iterating the body is what pulls each log.
        assert fetched == []
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk if isinstance(chunk, bytes) else chunk.encode())
    return response, b"".join(chunks)


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
