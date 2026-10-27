from types import SimpleNamespace
import gpustack.worker.runtime_metrics_aggregator as agg


class _Client:
    def __init__(self):
        self.calls = []

    def fetch_runtime_version_from_endpoint(self, endpoint, backend):
        self.calls.append(endpoint)
        return "0.27.1"


def _mgr(client, updated):
    m = object.__new__(agg.RuntimeMetricsAggregator)
    m._metrics_client = client
    m._update_model_instance = lambda i, **kw: updated.append((i, kw))
    return m


def _router():
    """A PD router: API on 40020, Prometheus exposition on a separate band."""
    return SimpleNamespace(
        id=74,
        worker_ip="192.168.13.3",
        port=40020,
        ports=[40020, 40000],
        backend="vLLM",
        backend_version=None,
        api_detected_backend_version=None,
        named_ports={"prometheus": {"base": 40000}},
    )


def test_router_version_is_probed_on_the_api_port_not_the_exposition_port():
    client, updated = _Client(), []
    mi = _router()
    exposition = f"{mi.worker_ip}:{agg._metrics_port(mi)}"
    assert exposition == "192.168.13.3:40000"

    _mgr(client, updated).fetch_and_update_api_backend_version(mi, exposition)

    assert client.calls == ["192.168.13.3:40020"]
    assert updated == [(74, {"api_detected_backend_version": "0.27.1"})]


def test_an_engine_probes_the_one_port_it_has():
    client, updated = _Client(), []
    mi = SimpleNamespace(
        id=67,
        worker_ip="192.168.50.15",
        port=40046,
        ports=[40046, 40000],
        backend="vLLM",
        backend_version=None,
        api_detected_backend_version=None,
        named_ports=None,
    )
    exposition = f"{mi.worker_ip}:{agg._metrics_port(mi)}"
    assert exposition == "192.168.50.15:40046"

    _mgr(client, updated).fetch_and_update_api_backend_version(mi, exposition)

    assert client.calls == ["192.168.50.15:40046"]
