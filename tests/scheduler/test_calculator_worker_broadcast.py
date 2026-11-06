"""read_local_path_file_from_workers must not leave broadcast tasks running
after it has already returned a result to its caller."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest

from gpustack.policies.worker_filters.local_path_filter import LocalPathFilter
from gpustack.schemas.filesystem import FileExistsResponse
from gpustack.schemas.models import GPUSelector, Model, SourceEnum
from gpustack.scheduler import evaluator
from gpustack.schemas.model_evaluations import ModelEvaluationResult
from gpustack.scheduler.calculator import read_local_path_file_from_workers
from gpustack.scheduler.evaluator import evaluate_model_metadata
from gpustack.server.worker_selector import WorkerPathCheckError, WorkerSelector


def make_local_path_model() -> Model:
    return Model(
        name="test-model",
        replicas=1,
        source=SourceEnum.LOCAL_PATH,
        local_path="/mnt/models/foo",
    )


@pytest.mark.asyncio
async def test_read_local_path_file_from_workers_cancels_losers():
    model = make_local_path_model()
    workers = [SimpleNamespace(name=f"worker-{i}") for i in range(5)]

    async def fake_read_model_config(self, worker, path):
        if worker.name == "worker-0":
            return {"ok": True}
        await asyncio.sleep(2)
        return {"ok": True}

    before = asyncio.all_tasks()
    with patch(
        "gpustack.client.worker_filesystem_client.WorkerFilesystemClient.read_model_config",
        new=fake_read_model_config,
    ):
        result = await read_local_path_file_from_workers(model, "config.json", workers)

    spawned = asyncio.all_tasks() - before
    assert result == {"ok": True}
    assert all(t.done() for t in spawned)


@pytest.mark.asyncio
async def test_read_local_path_file_from_workers_reports_all_errors():
    model = make_local_path_model()
    workers = [SimpleNamespace(name=f"worker-{i}") for i in range(3)]

    async def fake_read_model_config(self, worker, path):
        raise RuntimeError(f"boom from {worker.name}")

    with patch(
        "gpustack.client.worker_filesystem_client.WorkerFilesystemClient.read_model_config",
        new=fake_read_model_config,
    ):
        with pytest.raises(ValueError, match="Failed to read 'config.json'"):
            await read_local_path_file_from_workers(model, "config.json", workers)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        aiohttp.ClientConnectionError("connection refused"),
        asyncio.TimeoutError(),
        aiohttp.ClientError("Worker returned HTTP 403"),
        ValueError("Invalid filesystem response"),
    ],
)
async def test_local_path_filter_distinguishes_failed_checks_from_missing_paths(error):
    model = make_local_path_model()
    model.gpu_selector = GPUSelector(gpu_ids=["ready:cuda:0"])
    workers = [SimpleNamespace(name=name) for name in ("ready", "missing", "unchecked")]
    probe = AsyncMock(
        side_effect=[
            FileExistsResponse(exists=True, path=model.local_path),
            FileExistsResponse(exists=False, path=model.local_path),
            error,
        ]
    )

    with patch(
        "gpustack.client.worker_filesystem_client.WorkerFilesystemClient.path_exists",
        probe,
    ):
        candidates, messages = await LocalPathFilter(model).filter(workers)

    assert candidates == workers[:1]
    assert len(messages) == 2
    missing = next(m for m in messages if "does not exist" in m)
    failed = next(m for m in messages if "could not be checked" in m)
    assert "missing" in missing and "unchecked" not in missing
    assert "unchecked" in failed and "missing" not in failed
    assert "Worker connectivity" in failed
    assert probe.await_count == len(workers)


@pytest.mark.asyncio
async def test_local_path_filter_recovers_when_the_worker_filesystem_api_is_ready():
    model = make_local_path_model()
    model.gpu_selector = GPUSelector(gpu_ids=["worker-0:cuda:0"])
    workers = [SimpleNamespace(name="worker-0")]
    probe = AsyncMock(
        side_effect=[
            aiohttp.ClientConnectionError("connection refused"),
            FileExistsResponse(exists=True, path=model.local_path),
        ]
    )
    path_filter = LocalPathFilter(model)

    with patch(
        "gpustack.client.worker_filesystem_client.WorkerFilesystemClient.path_exists",
        probe,
    ):
        first_candidates, first_messages = await path_filter.filter(workers)
        candidates, messages = await path_filter.filter(workers)

    assert first_candidates == []
    assert len(first_messages) == 1
    assert "could not be checked" in first_messages[0]
    assert "does not exist" not in first_messages[0]
    assert candidates == workers
    assert messages == []
    assert probe.await_count == 2


@pytest.mark.asyncio
async def test_local_path_filter_summarizes_missing_and_unchecked_workers_separately():
    model = make_local_path_model()
    model.gpu_selector = GPUSelector(gpu_ids=["missing-0:cuda:0"])
    workers = [
        SimpleNamespace(name=f"{kind}-{i}")
        for kind in ("missing", "unchecked")
        for i in reversed(range(4))
    ]

    async def probe(worker, path):
        if worker.name.startswith("unchecked"):
            raise aiohttp.ClientConnectionError("connection refused")
        return FileExistsResponse(exists=False, path=path)

    with patch(
        "gpustack.client.worker_filesystem_client.WorkerFilesystemClient.path_exists",
        AsyncMock(side_effect=probe),
    ):
        candidates, messages = await LocalPathFilter(model).filter(workers)

    assert candidates == []
    assert len(messages) == 2
    assert "missing-0, missing-1, missing-2... (+1 more)" in messages[0]
    assert "unchecked" not in messages[0]
    assert "unchecked-0, unchecked-1, unchecked-2... (+1 more)" in messages[1]
    assert "missing" not in messages[1]


@pytest.mark.asyncio
async def test_config_broadcast_preserves_worker_path_check_failure_reason():
    model = make_local_path_model()
    model.gpu_selector = GPUSelector(gpu_ids=["worker-0:cuda:0"])
    workers = [SimpleNamespace(name="worker-0")]
    probe = AsyncMock(side_effect=aiohttp.ClientConnectionError("connection refused"))
    read_config = AsyncMock()
    with (
        patch(
            "gpustack.scheduler.calculator.GPUMatchingFilter.filter",
            AsyncMock(return_value=(workers, [])),
        ),
        patch(
            "gpustack.client.worker_filesystem_client.WorkerFilesystemClient.path_exists",
            probe,
        ),
        patch(
            "gpustack.client.worker_filesystem_client.WorkerFilesystemClient.read_model_config",
            read_config,
        ),
    ):
        with pytest.raises(ValueError) as exc:
            await read_local_path_file_from_workers(model, "config.json", workers)

    assert "could not be checked" in str(exc.value)
    assert "does not exist" not in str(exc.value)
    read_config.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("specified_worker", [False, True])
async def test_worker_path_selector_reports_failed_checks(specified_worker):
    workers = [SimpleNamespace(id=1, name="worker-0")]
    filesystem = SimpleNamespace(
        path_exists=AsyncMock(side_effect=aiohttp.ClientConnectionError("unavailable"))
    )
    selector = WorkerSelector(filesystem)
    with pytest.raises(RuntimeError, match="Failed to check path"):
        await selector.find_worker_with_path(
            workers, worker_id=1 if specified_worker else None, path="/models/foo"
        )


@pytest.mark.asyncio
async def test_worker_path_selector_finishes_pending_probes_before_returning():
    workers = [SimpleNamespace(id=i, name=f"worker-{i}") for i in range(2)]
    started = asyncio.Event()
    finished = asyncio.Event()

    async def probe(worker, path):
        if worker.id == 0:
            await started.wait()
            return FileExistsResponse(exists=True, path=path)
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            finished.set()

    selector = WorkerSelector(SimpleNamespace(path_exists=probe))
    found = await selector.find_worker_with_path(workers, path="/models/foo")
    assert found is workers[0]
    assert finished.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("skip_evaluation", [False, True])
@pytest.mark.parametrize(
    "outcomes",
    [
        [False, False],
        [False, aiohttp.ClientConnectionError("unavailable")],
        [aiohttp.ClientConnectionError("unavailable"), asyncio.TimeoutError()],
        [False, True],
        [aiohttp.ClientConnectionError("unavailable"), True],
    ],
)
async def test_metadata_check_preserves_missing_unknown_and_accessible_paths(
    outcomes, skip_evaluation
):
    model = make_local_path_model()
    model.env = {"GPUSTACK_SKIP_MODEL_EVALUATION": "1"} if skip_evaluation else {}
    workers = [SimpleNamespace(id=i, name=f"worker-{i}") for i in range(2)]

    async def probe(worker, path):
        result = outcomes[worker.id]
        if isinstance(result, Exception):
            raise result
        return FileExistsResponse(exists=result, path=path)

    with (
        patch("gpustack.scheduler.evaluator.os.path.exists", return_value=False),
        patch(
            "gpustack.client.worker_filesystem_client.WorkerFilesystemClient.path_exists",
            AsyncMock(side_effect=probe),
        ),
        patch(
            "gpustack.scheduler.evaluator.scheduler.evaluate_pretrained_config",
            AsyncMock(),
        ),
    ):
        if not any(x is True for x in outcomes) and any(
            isinstance(x, Exception) for x in outcomes
        ):
            if skip_evaluation:
                assert await evaluate_model_metadata(
                    SimpleNamespace(), model, workers
                ) == (True, [])
            else:
                with pytest.raises(WorkerPathCheckError, match="could not be checked"):
                    await evaluate_model_metadata(SimpleNamespace(), model, workers)
            return
        valid, messages = await evaluate_model_metadata(
            SimpleNamespace(), model, workers
        )

    if True in outcomes:
        assert valid and messages == []
    else:
        assert not valid and len(messages) == 1
        assert "does not exist" in messages[0]


@pytest.mark.asyncio
async def test_metadata_check_without_workers_does_not_claim_the_path_is_missing():
    with patch("gpustack.scheduler.evaluator.os.path.exists", return_value=False):
        with pytest.raises(WorkerPathCheckError, match="could not be checked"):
            await evaluate_model_metadata(
                SimpleNamespace(), make_local_path_model(), []
            )


@pytest.mark.asyncio
async def test_cancelling_worker_path_selection_finishes_the_probe():
    started = asyncio.Event()
    finished = asyncio.Event()

    async def probe(worker, path):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            finished.set()

    selector = WorkerSelector(SimpleNamespace(path_exists=probe))
    task = asyncio.create_task(
        selector.find_worker_with_path(
            [SimpleNamespace(id=1, name="worker-0")], path="/models/foo"
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set()


@pytest.mark.asyncio
async def test_worker_path_check_failure_is_not_cached_as_model_incompatibility(
    monkeypatch,
):
    model = make_local_path_model()
    workers = [SimpleNamespace(id=1, name="worker-0")]
    probe = AsyncMock(
        side_effect=[
            aiohttp.ClientConnectionError("unavailable"),
            FileExistsResponse(exists=True, path=model.local_path),
        ]
    )

    async def evaluate(*args, **kwargs):
        compatible, messages = await evaluate_model_metadata(
            SimpleNamespace(), model, workers
        )
        return ModelEvaluationResult(
            compatible=compatible, compatibility_messages=messages
        )

    monkeypatch.setattr(evaluator, "evaluate_cache", {})
    monkeypatch.setattr(evaluator, "make_hashable_key", lambda *a, **kw: "same-input")
    monkeypatch.setattr(
        evaluator, "visible_cache_service", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(evaluator, "cache_instances_in", AsyncMock(return_value=[]))
    monkeypatch.setattr(evaluator, "evaluate_model", evaluate)
    monkeypatch.setattr(evaluator.os.path, "exists", lambda path: False)
    monkeypatch.setattr(evaluator.scheduler, "evaluate_pretrained_config", AsyncMock())
    with patch(
        "gpustack.client.worker_filesystem_client.WorkerFilesystemClient.path_exists",
        probe,
    ):
        failed = await evaluator.evaluate_model_with_cache(
            None, None, model, workers, []
        )
        assert failed.error and not failed.compatible
        assert "could not be checked" in failed.error_message
        assert evaluator.evaluate_cache == {}
        recovered = await evaluator.evaluate_model_with_cache(
            None, None, model, workers, []
        )
        cached = await evaluator.evaluate_model_with_cache(
            None, None, model, workers, []
        )

    assert recovered.compatible and not recovered.error
    assert cached is recovered
    assert probe.await_count == 2
