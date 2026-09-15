from types import SimpleNamespace

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
