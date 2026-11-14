from types import SimpleNamespace

from prometheus_client.core import GaugeMetricFamily

from gpustack.worker.exporter import MetricExporter


def _exporter(cache=None):
    return MetricExporter(
        cfg=SimpleNamespace(worker_metrics_port=10151, disable_openapi_docs=False),
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


def test_worker_metric_families_use_the_gpustack_prefix():
    """Every worker metric family is published under the ``gpustack:``
    namespace the observability docs list, the GPU info family included."""
    gpu = SimpleNamespace(
        index=0,
        name="NVIDIA A100",
        core=None,
        temperature=None,
        power=None,
        power_used=None,
        memory=None,
    )
    status = SimpleNamespace(
        os=None,
        kernel=None,
        uptime=None,
        cpu=None,
        memory=None,
        gpu_devices=[gpu],
        filesystem=None,
    )
    exporter = MetricExporter(
        cfg=SimpleNamespace(worker_metrics_port=10151, disable_openapi_docs=False),
        collector=SimpleNamespace(
            timed_collect=lambda clientset=None: SimpleNamespace(status=status)
        ),
        worker_name_getter=lambda: "w",
        worker_ip_getter=lambda: "10.0.0.1",
        worker_id_getter=lambda: 1,
        clientset_getter=lambda: None,
    )

    families = {m.name: m for m in exporter.collect_worker_metrics()}

    assert all(name.startswith("gpustack:") for name in families)
    gpu_info = families["gpustack:worker_node_gpu"]
    assert [s.name for s in gpu_info.samples] == ["gpustack:worker_node_gpu_info"]
    assert gpu_info.samples[0].labels["gpu_name"] == "NVIDIA A100"
