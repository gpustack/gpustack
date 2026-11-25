"""Log endpoints keep worker exception details out of client-visible streams."""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gpustack.routes import benchmarks, cache_services, model_instances
from gpustack.schemas.benchmark import BenchmarkStateEnum
from gpustack.schemas.models import BackendEnum, ModelInstanceStateEnum
from gpustack.server import worker_request
from gpustack.worker.logs import LogOptions


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["model", "cache", "benchmark"])
@pytest.mark.parametrize("error_type", [RuntimeError, TimeoutError])
@pytest.mark.parametrize("after_chunk", [False, True])
@pytest.mark.parametrize("timeout_seconds", [3600, 0])
async def test_log_errors_keep_safe_hints_and_status(
    monkeypatch, caplog, route, error_type, after_chunk, timeout_seconds
):
    sentinel = "credential=private-test-value; /internal/config.yaml"
    worker = SimpleNamespace(id=1)
    request = MagicMock()
    request.app.state.server_config.log_stream_timeout_seconds = timeout_seconds

    class FailedResponse:
        status = 200
        headers = {}

        @property
        def content(self):
            return self

        async def iter_any(self):
            if after_chunk:
                yield b"model output\n"
            raise error_type(sentinel)

    @asynccontextmanager
    async def failed_request(**kwargs):
        assert kwargs["timeout"].total == (timeout_seconds or None)
        assert kwargs["timeout"].sock_connect == 5
        yield FailedResponse()

    @asynccontextmanager
    async def session():
        yield MagicMock()

    monkeypatch.setattr(worker_request, "_request_to_worker", failed_request)
    options = LogOptions(follow=True)
    if route == "cache":
        response = await cache_services._proxy_instance_logs(
            request, SimpleNamespace(id=7, cache_service_id=2), worker, options
        )
    elif route == "benchmark":
        benchmark = SimpleNamespace(
            id=7, name="benchmark", worker_id=1, state=BenchmarkStateEnum.RUNNING
        )
        monkeypatch.setattr(benchmarks, "async_session", session)
        monkeypatch.setattr(benchmarks, "assert_resource_visible", MagicMock())
        monkeypatch.setattr(
            benchmarks.Benchmark, "one_by_id", AsyncMock(return_value=benchmark)
        )
        monkeypatch.setattr(
            benchmarks.Worker, "one_by_id", AsyncMock(return_value=worker)
        )
        response = await benchmarks.get_benchmark_logs(request, MagicMock(), 7, options)
    else:
        monkeypatch.setattr(model_instances, "async_session", session)
        monkeypatch.setattr(
            model_instances,
            "fetch_model_instance",
            AsyncMock(
                return_value=SimpleNamespace(
                    id=7,
                    name="model",
                    restart_count=0,
                    worker_id=1,
                    state=ModelInstanceStateEnum.RUNNING,
                    backend=BackendEnum.VLLM,
                    model_files=[],
                    model=None,
                    distributed_servers=None,
                )
            ),
        )
        monkeypatch.setattr(
            model_instances, "fetch_worker", AsyncMock(return_value=worker)
        )
        response = await model_instances.get_serving_logs(
            request, MagicMock(), 7, options
        )

    sent = []

    async def send(message):
        sent.append(message)

    await response.stream_response(send)
    # Once data starts, HTTP headers are committed; failures remain in the body.
    assert sent[0]["status"] == (200 if after_chunk else 500)
    body = b"".join(message.get("body", b"") for message in sent)
    assert sentinel.encode() not in body
    assert (
        b"timed out"
        if error_type is TimeoutError
        else f"Unable to read logs: {error_type.__name__}".encode()
    ) in body
    assert (b"model output" in body) is after_chunk
    assert b"None seconds" not in body
    records = [r for r in caplog.records if r.exc_info]
    assert len(records) == 1
    assert records[0].name == worker_request.logger.name
    assert isinstance(records[0].exc_info[1], error_type)
    assert records[0].exc_info[2] is not None
    assert sentinel in caplog.text
