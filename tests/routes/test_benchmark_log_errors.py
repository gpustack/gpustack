from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gpustack.routes import benchmarks
from gpustack.schemas.benchmark import BenchmarkStateEnum
from gpustack.server import worker_request
from tests.utils.mock import mock_async_session


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [False, True])
async def test_following_benchmark_logs_hides_private_worker_errors(
    monkeypatch, caplog, timeout
):
    benchmark = SimpleNamespace(
        id=7, worker_id=3, name="test-benchmark", state=BenchmarkStateEnum.RUNNING
    )
    monkeypatch.setattr(benchmarks, "async_session", mock_async_session)
    monkeypatch.setattr(
        benchmarks.Benchmark, "one_by_id", AsyncMock(return_value=benchmark)
    )
    monkeypatch.setattr(
        benchmarks.Worker, "one_by_id", AsyncMock(return_value=SimpleNamespace(id=3))
    )
    monkeypatch.setattr(benchmarks, "assert_resource_visible", lambda *a, **kw: None)
    secret = "private worker address and credentials"
    connection = MagicMock()
    error = TimeoutError(secret) if timeout else RuntimeError(secret)
    connection.__aenter__ = AsyncMock(side_effect=error)
    connection.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(worker_request, "_request_to_worker", lambda **kw: connection)
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(http_client=object(), http_client_no_proxy=object())
        )
    )
    response = await benchmarks.get_benchmark_logs(
        request, object(), benchmark.id, SimpleNamespace(follow=True, tail=50)
    )
    messages = []

    async def send(message):
        messages.append(message)

    await response.stream_response(send)
    assert messages[0]["status"] == 500
    body = b"".join(message.get("body", b"") for message in messages)
    assert secret.encode() not in body
    assert secret in caplog.text
    assert (b"timed out" if timeout else b"unavailable") in body
    assert b"Please reopen the log page." in body
