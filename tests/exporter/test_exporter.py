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
