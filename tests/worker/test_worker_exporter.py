from types import SimpleNamespace

from prometheus_client.core import GaugeMetricFamily

from gpustack.worker.exporter import MetricExporter


def _exporter(cache=None):
    return MetricExporter(
        cfg=SimpleNamespace(worker_metrics_port=10151),
        collector=None,
        worker_name_getter=lambda: "w",
        worker_ip_getter=lambda: "10.0.0.1",
        worker_id_getter=lambda: 1,
        cache=cache,
    )


def _boom():
    raise RuntimeError("collector exploded")
    yield  # unreachable, but makes this a generator like the real collectors


def test_worker_metric_failure_keeps_runtime_metrics():
    """Both halves are collected independently, so a failure in one must not
    empty the whole /metrics response."""
    exporter = _exporter(
        cache={"unified": {"g": GaugeMetricFamily("g", "d", value=1.0)}}
    )
    exporter.collect_worker_metrics = _boom

    assert [m.name for m in exporter.collect()] == ["g"]


def test_runtime_metric_failure_keeps_worker_metrics():
    exporter = _exporter(cache={})
    exporter.collect_worker_metrics = lambda: iter(
        [GaugeMetricFamily("w", "d", value=1.0)]
    )
    exporter.collect_runtime_metrics = _boom

    assert [m.name for m in exporter.collect()] == ["w"]
