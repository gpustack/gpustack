import logging
from types import SimpleNamespace
from unittest.mock import patch

from prometheus_client.core import GaugeMetricFamily

from gpustack.utils.metrics import get_builtin_metrics_config
from gpustack.worker.exporter import MetricExporter
from gpustack.worker.runtime_metrics_aggregator import (
    RuntimeMetricsAggregator,
    create_prom_metric_family,
)
from gpustack.worker.runtime_metrics_client import parse_metrics_text

# vLLM serving OpenMetrics (the Rust frontend) declares a counter by its base
# name and suffixes the samples with _total.
OPENMETRICS_TEXT = """\
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="m",engine="0"} 3.0
# HELP vllm:prompt_tokens Number of prefill tokens processed.
# TYPE vllm:prompt_tokens counter
vllm:prompt_tokens_total{model_name="m",engine="0"} 1234.0
# EOF
"""

# vLLM serving the Prometheus text format declares the full name on the TYPE line.
PROMETHEUS_TEXT = """\
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="m",engine="0"} 3.0
# HELP vllm:prompt_tokens_total Number of prefill tokens processed.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{model_name="m",engine="0"} 1234.0
"""

OPENMETRICS_CONTENT_TYPE = "application/openmetrics-text; version=1.0.0; charset=utf-8"
PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


def _families(text, content_type):
    return {f.name: f for f in parse_metrics_text(text, content_type)}


def test_parse_openmetrics_keeps_counter_samples():
    """The Prometheus text parser reports the _total samples as a separate
    untyped family, which leaves vllm:prompt_tokens empty and unmappable."""
    families = _families(OPENMETRICS_TEXT, OPENMETRICS_CONTENT_TYPE)

    assert "vllm:prompt_tokens" in families
    counter = families["vllm:prompt_tokens"]
    assert counter.type == "counter"
    assert [s.value for s in counter.samples] == [1234.0]
    assert not any(f.type == "unknown" for f in families.values())


def test_parse_prometheus_text_is_unchanged():
    families = _families(PROMETHEUS_TEXT, PROMETHEUS_CONTENT_TYPE)

    # The 0.0.4 parser strips the _total suffix, so both formats agree on the
    # family name the metrics config maps against.
    assert "vllm:prompt_tokens" in families
    assert [s.value for s in families["vllm:prompt_tokens"].samples] == [1234.0]
    assert [s.value for s in families["vllm:num_requests_running"].samples] == [3.0]


def test_parse_openmetrics_falls_back_when_malformed():
    """A runtime advertising OpenMetrics but omitting the # EOF trailer should
    still yield metrics rather than failing the whole scrape."""
    families = _families(PROMETHEUS_TEXT, OPENMETRICS_CONTENT_TYPE)

    assert [s.value for s in families["vllm:num_requests_running"].samples] == [3.0]


def test_parse_without_content_type_uses_prometheus_text():
    families = _families(PROMETHEUS_TEXT, None)

    assert [s.value for s in families["vllm:num_requests_running"].samples] == [3.0]


def test_unparsable_payload_yields_no_families():
    """A malformed body must not escape as an exception: the caller retries the
    endpoint whenever one is raised, and no retry can fix a bad payload."""
    malformed = "this is not a metrics payload at all\n"

    assert parse_metrics_text(malformed, PROMETHEUS_CONTENT_TYPE) == []
    # OpenMetrics first, then the text format as a fallback — both fail here.
    assert parse_metrics_text(malformed, OPENMETRICS_CONTENT_TYPE) == []


def test_create_prom_metric_family_supports_untyped():
    family = create_prom_metric_family(
        type="unknown", name="x", description="d", labels=["a"]
    )
    family.add_metric(labels=["1"], value=2.0)

    assert [s.value for s in family.samples] == [2.0]


def _sample(name, value, labels=None):
    return SimpleNamespace(name=name, labels=labels or {}, value=value, timestamp=None)


def _family(name, type_, samples):
    return SimpleNamespace(name=name, type=type_, documentation="d", samples=samples)


# prometheus_client emits _created alongside every counter by default, and the
# OpenMetrics parser keeps it inside the counter family rather than splitting it
# into one of its own.
OPENMETRICS_WITH_CREATED = """\
# HELP vllm:prompt_tokens Number of prefill tokens processed.
# TYPE vllm:prompt_tokens counter
vllm:prompt_tokens_total{model_name="m",engine="0"} 1234.0
vllm:prompt_tokens_created{model_name="m",engine="0"} 1700000000.0
# EOF
"""


def _aggregate(metrics, aggregator=None):
    aggregator = aggregator or RuntimeMetricsAggregator(
        cache={}, worker_id_getter=lambda: 1
    )
    unified, raw = {}, {}
    aggregator._process_endpoint_metrics(
        metrics,
        {"worker_id": "1", "model_name": "m", "model_instance_id": "1"},
        "vLLM",
        "0.11.0",
        unified,
        raw,
        aggregator._get_metrics_config(),
    )
    return unified, raw


def test_created_samples_do_not_duplicate_counter_series():
    """A _created sample carries a creation timestamp, not a count. Passing it to
    add_metric — which appends _total itself — would emit a second _total series
    under identical labels, and Prometheus rejects a whole scrape over that."""
    families = _families(OPENMETRICS_WITH_CREATED, OPENMETRICS_CONTENT_TYPE)
    # The parser really does hand us both samples in one family.
    assert len(families["vllm:prompt_tokens"].samples) == 2

    unified, raw = _aggregate(families)

    assert [(s.name, s.value) for s in raw["vllm:prompt_tokens"].samples] == [
        ("vllm:prompt_tokens_total", 1234.0)
    ]
    assert [(s.name, s.value) for s in unified["gpustack:prompt_tokens"].samples] == [
        ("gpustack:prompt_tokens_total", 1234.0)
    ]


# An info family is named without the suffix its samples carry.
OPENMETRICS_INFO = """\
# HELP vllm:cache_config Cache configuration.
# TYPE vllm:cache_config info
vllm:cache_config_info{block_size="16"} 1.0
# EOF
"""


def test_info_samples_keep_their_own_name():
    """Info samples are named <family>_info, so they must survive the filter and
    be added under that name — InfoMetricFamily.add_metric takes a mapping, not
    the float the exposition gives us."""
    families = _families(OPENMETRICS_INFO, OPENMETRICS_CONTENT_TYPE)
    assert families["vllm:cache_config"].type == "info"

    _, raw = _aggregate(families)

    samples = [(s.name, s.value) for s in raw["vllm:cache_config"].samples]
    assert samples == [("vllm:cache_config_info", 1.0)]
    assert samples[0][0].startswith("vllm:cache_config")


def test_content_type_match_is_case_insensitive():
    """Media types are case-insensitive. Matching case-sensitively would send an
    OpenMetrics payload down the text path, which is the original bug."""
    families = _families(
        OPENMETRICS_TEXT, "Application/OpenMetrics-Text; version=1.0.0"
    )

    assert [s.value for s in families["vllm:prompt_tokens"].samples] == [1234.0]


def test_persistent_family_failure_warns_once(caplog):
    """Aggregation runs every few seconds; a family that always fails must not
    warn on every pass, but must warn again after recovering and failing anew."""
    aggregator = RuntimeMetricsAggregator(cache={}, worker_id_getter=lambda: 1)
    broken = {"broken": _family("broken", "stateset", [_sample("broken", 1.0)])}
    working = {
        "broken": _family("broken", "gauge", [_sample("broken", 1.0)]),
    }

    def warnings():
        return [r for r in caplog.records if r.levelno == logging.WARNING]

    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            _aggregate(broken, aggregator)
        assert len(warnings()) == 1

        _aggregate(working, aggregator)  # recovers, clearing the reported state
        _aggregate(broken, aggregator)
        assert len(warnings()) == 2


def test_unsupported_family_does_not_drop_the_endpoint():
    """One family the exporter cannot build must not discard the metrics
    collected from the rest of the endpoint."""
    aggregator = RuntimeMetricsAggregator(cache={}, worker_id_getter=lambda: 1)
    unified, raw = {}, {}

    metrics = {
        "broken": _family("broken", "stateset", [_sample("broken", 1.0)]),
        "vllm:num_requests_running": _family(
            "vllm:num_requests_running",
            "gauge",
            [_sample("vllm:num_requests_running", 3.0)],
        ),
    }

    aggregator._process_endpoint_metrics(
        metrics,
        {"worker_id": "1", "model_name": "m"},
        "vLLM",
        "0.11.0",
        unified,
        raw,
        aggregator._get_metrics_config(),
    )

    assert "broken" not in raw
    assert [s.value for s in raw["vllm:num_requests_running"].samples] == [3.0]
    assert "gpustack:num_requests_running" in unified


# ---------------------------------------------------------------------------
# Cache lifecycle: metrics of a model instance that is gone must stop being
# exported. The worker cache is shared with ``MetricExporter``, which re-exports
# whatever is in it, so a stale entry keeps showing up as a live instance.
# ---------------------------------------------------------------------------

_RUNTIME_ENDPOINT = "10.0.0.1:40034"
_WAITING_METRIC = "gpustack:num_requests_waiting"
_SOURCE_WAITING_METRIC = "sglang:num_queue_reqs"


class _FakeModelInstance:
    """Only the attributes used by ``aggregate()``/``_build_base_labels()``."""

    def __init__(self, instance_id: int = 999, model_id: int = 26):
        self.id = instance_id
        self.name = f"test-instance-{instance_id}"
        self.model_id = model_id
        self.role = None
        self.worker_id = 77
        self.worker_name = "test-worker"
        self.worker_ip = "10.0.0.1"
        self.ports = [40034]
        # Non-None keeps aggregate() from probing the backend over HTTP.
        self.api_detected_backend_version = "0.6.18"


class _FakeModel:
    id = 26
    name = "test-model"


class _ExporterProbe:
    """Minimal ``self`` to call the real ``MetricExporter.collect_runtime_metrics``."""

    def __init__(self, cache):
        self._cache = cache


def _gauge(name, value):
    family = GaugeMetricFamily(name, "test")
    family.add_metric([], value)
    return family


def _active_endpoints(instance, model):
    return (
        {_RUNTIME_ENDPOINT},
        {_RUNTIME_ENDPOINT: instance},
        {instance.id: model},
    )


def _cache_aggregator(cache, endpoints_provider):
    """Aggregator wired to stubbed endpoints/metrics so tests stay offline."""
    aggregator = RuntimeMetricsAggregator(
        cache=cache, worker_id_getter=lambda: 77, clientset=None
    )
    # Pre-seed the builtin metrics config to avoid the online config fetch.
    aggregator._metrics_config_cache["config"] = get_builtin_metrics_config()
    aggregator._find_active_model_endpoints = (
        lambda worker_id, metrics_config: endpoints_provider()
    )
    aggregator._metrics_client.fetch_metrics_from_endpoints = lambda endpoints: {
        endpoint: {_SOURCE_WAITING_METRIC: _gauge(_SOURCE_WAITING_METRIC, 10.0)}
        for endpoint in endpoints
    }
    return aggregator


def test_aggregate_fills_cache_for_an_active_instance():
    """An active instance populates the shared cache and gets exported."""
    cache = {}
    instance, model = _FakeModelInstance(), _FakeModel()
    aggregator = _cache_aggregator(cache, lambda: _active_endpoints(instance, model))

    with patch(
        "gpustack.worker.runtime_metrics_aggregator.get_backend",
        return_value="SGLang",
    ):
        aggregator.aggregate()

    assert _WAITING_METRIC in cache["unified"]
    exported = list(MetricExporter.collect_runtime_metrics(_ExporterProbe(cache)))
    assert [family.name for family in exported] == [_WAITING_METRIC]


def test_aggregate_clears_cache_when_the_last_instance_is_removed():
    """The last instance of a worker disappearing must drop its cached metrics:
    otherwise the exporter keeps re-exporting the frozen values of a model
    instance that no longer exists."""
    cache = {}
    instance, model = _FakeModelInstance(), _FakeModel()
    state = {"has_instance": True}

    def endpoints_provider():
        if state["has_instance"]:
            return _active_endpoints(instance, model)
        # The instance is gone: this worker has no endpoint of its own anymore.
        return set(), {}, {}

    aggregator = _cache_aggregator(cache, endpoints_provider)

    with patch(
        "gpustack.worker.runtime_metrics_aggregator.get_backend",
        return_value="SGLang",
    ):
        aggregator.aggregate()
        assert _WAITING_METRIC in cache["unified"]

        state["has_instance"] = False
        aggregator.aggregate()

    assert cache["unified"] == {}
    assert cache["raw"] == {}
    assert list(MetricExporter.collect_runtime_metrics(_ExporterProbe(cache))) == []
