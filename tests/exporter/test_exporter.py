import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gpustack.exporter.exporter import MetricExporter
from gpustack.schemas.models import ModelInstanceStateEnum


def _sample_value(metrics, metric_name):
    for metric in metrics:
        if metric.name == metric_name:
            assert len(metric.samples) == 1
            return metric.samples[0].value
    raise AssertionError(f"Metric {metric_name} not found")


def _sample_labels(metrics, metric_name):
    for metric in metrics:
        if metric.name == metric_name:
            assert len(metric.samples) == 1
            return metric.samples[0].labels
    raise AssertionError(f"Metric {metric_name} not found")


@pytest.mark.asyncio
async def test_model_instance_restart_metrics_are_collected():
    exporter = MetricExporter(SimpleNamespace(metrics_port=10161))
    latest_restart_time = datetime(2026, 4, 17, 8, 30, tzinfo=timezone.utc)

    instance = SimpleNamespace(
        worker_id=2,
        worker_name="worker-2",
        name="qwen-1",
        state=ModelInstanceStateEnum.RUNNING,
        restart_count=3,
        last_restart_time=latest_restart_time,
    )
    model = SimpleNamespace(
        id=10,
        name="qwen",
        backend="vllm",
        backend_version="0.8.0",
        source="huggingface",
        model_source_key="Qwen/Qwen2.5-0.5B-Instruct",
        replicas=1,
        ready_replicas=1,
        instances=[instance],
    )
    cluster = SimpleNamespace(
        id=1,
        name="default",
        provider="docker",
        state="ready",
        cluster_workers=[],
        cluster_models=[model],
    )

    with patch("gpustack.exporter.exporter.Cluster.all", return_value=[cluster]):
        metrics = await exporter._collect_metrics(session=SimpleNamespace())

    assert _sample_value(metrics, "gpustack:model_instance_restart_count") == 3
    assert (
        _sample_value(metrics, "gpustack:model_instance_latest_restart_time")
        == latest_restart_time.timestamp()
    )
    assert _sample_labels(metrics, "gpustack:model_instance_restart_count") == {
        "cluster_id": "1",
        "cluster_name": "default",
        "worker_id": "2",
        "worker_name": "worker-2",
        "model_id": "10",
        "model_name": "qwen",
        "model_instance_name": "qwen-1",
    }


class _NoopSession:
    async def __aenter__(self):
        return SimpleNamespace()

    async def __aexit__(self, *exc):
        return False


@pytest.mark.asyncio
async def test_generate_metrics_cache_survives_transient_db_error(monkeypatch):
    """A transient DB error while refreshing the cache must not escape the
    loop. If it did, the exception would propagate through the server's
    asyncio.gather and take the whole process down (the #5839 restart). The
    loop should keep the last cache and retry on the next tick.
    """
    exporter = MetricExporter(SimpleNamespace(metrics_port=10162))
    exporter._cache_metrics = ["stale"]

    collect_calls = {"n": 0}

    async def _boom(session):
        collect_calls["n"] += 1
        raise TimeoutError(
            "QueuePool limit of size 30 overflow 20 reached, connection timed out"
        )

    async def _sleep_then_stop(_seconds):
        # Break the otherwise-infinite loop the way a real shutdown would.
        raise asyncio.CancelledError()

    monkeypatch.setattr(exporter, "_collect_metrics", _boom)
    monkeypatch.setattr(
        "gpustack.exporter.exporter.async_session", lambda: _NoopSession()
    )
    monkeypatch.setattr("gpustack.exporter.exporter.asyncio.sleep", _sleep_then_stop)

    # The transient error is swallowed; the loop proceeds to the sleep, where
    # our stand-in raises CancelledError to end the test. CancelledError itself
    # must propagate (clean shutdown), unlike the DB error.
    with pytest.raises(asyncio.CancelledError):
        await exporter.generate_metrics_cache()

    assert collect_calls["n"] == 1  # ran once, error swallowed, reached sleep
    assert exporter._cache_metrics == ["stale"]  # kept last cache, no crash
