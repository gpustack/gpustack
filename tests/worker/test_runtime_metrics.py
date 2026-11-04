import logging
from types import SimpleNamespace
from unittest.mock import patch
from unittest.mock import Mock

import pytest
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


@pytest.mark.parametrize(
    "type_suffix,content_type,eof",
    [
        ("_total", PROMETHEUS_CONTENT_TYPE, ""),
        ("", OPENMETRICS_CONTENT_TYPE, "# EOF\n"),
    ],
)
def test_mooncake_receive_failures_reach_the_unified_counter(
    type_suffix, content_type, eof
):
    text = (
        "# HELP vllm:mooncake_num_failed_recvs Failed PD KV receives.\n"
        f"# TYPE vllm:mooncake_num_failed_recvs{type_suffix} counter\n"
        'vllm:mooncake_num_failed_recvs_total{model_name="m",engine="0"} 7.0\n' + eof
    )
    unified, raw = _aggregate(_families(text, content_type))
    family = unified["gpustack:pd_kv_receive_failed"]
    assert family.type == "counter"
    assert [(sample.name, sample.value) for sample in family.samples] == [
        ("gpustack:pd_kv_receive_failed_total", 7.0)
    ]
    assert family.samples[0].labels == {
        "worker_id": "1",
        "model_name": "m",
        "model_instance_id": "1",
        "engine": "0",
    }
    assert [
        sample.value for sample in raw["vllm:mooncake_num_failed_recvs"].samples
    ] == [7.0]


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
        # Read by _build_base_labels for the disaggregation role label; a
        # missing attribute would skip the whole endpoint's metrics.
        self.role = None


class _FakeModel:
    id = 26
    name = "test-model"
    backend = 'SGLang'


class _ExporterProbe:
    """Minimal ``self`` to call the real ``MetricExporter.collect_runtime_metrics``."""

    def __init__(self, cache):
        self._cache = cache

    def collect(self):
        yield from MetricExporter.collect_runtime_metrics(self)


def _gauge(name, value):
    family = GaugeMetricFamily(name, "test")
    family.add_metric([], value)
    return family


def _active_endpoints(instance, model):
    from gpustack.utils.model_instance_workers import ModelInstanceMetricsSource
    from gpustack.worker.runtime_metrics_aggregator import _RuntimeMetricsEndpoint

    return (
        {_RUNTIME_ENDPOINT},
        {
            _RUNTIME_ENDPOINT: _RuntimeMetricsEndpoint(
                instance,
                ModelInstanceMetricsSource(
                    instance.id,
                    instance.model_id,
                    instance.worker_id,
                    0,
                    '',
                    'legacy',
                    instance.ports[0],
                    instance.ports[0],
                ),
                instance.worker_ip,
                instance.worker_name,
            )
        },
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
    assert {family.name for family in exported} == {
        _WAITING_METRIC,
        'gpustack:runtime_metrics_scrape_status',
    }


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


@pytest.mark.parametrize('value', [0.0, 10.0])
def test_runtime_cache_and_both_exporters_expire_when_aggregation_stops(
    monkeypatch, value
):
    from prometheus_client import CollectorRegistry, generate_latest
    from gpustack.worker.exporter import RawCollector
    import gpustack.worker.runtime_metrics_aggregator as module

    clock = [100.0]
    monkeypatch.setattr(module, 'monotonic', lambda: clock[0], raising=False)
    cache = {}
    instance, model = _FakeModelInstance(), _FakeModel()
    aggregator = _cache_aggregator(cache, lambda: _active_endpoints(instance, model))
    aggregator._metrics_client.fetch_metrics_from_endpoints = lambda endpoints: {
        endpoint: {_SOURCE_WAITING_METRIC: _gauge(_SOURCE_WAITING_METRIC, value)}
        for endpoint in endpoints
    }
    with patch(f'{module.__name__}.get_backend', return_value='SGLang'):
        aggregator.aggregate()

    raw_registry = CollectorRegistry()
    raw_registry.register(RawCollector(cache))
    unified_registry = CollectorRegistry()
    unified_registry.register(_ExporterProbe(cache))
    assert [
        sample.value
        for name, family in cache['unified'].items()
        if name == _WAITING_METRIC
        for sample in family.samples
    ] == [value]
    assert list(MetricExporter.collect_runtime_metrics(_ExporterProbe(cache)))
    assert _SOURCE_WAITING_METRIC.encode() in generate_latest(raw_registry)
    assert _WAITING_METRIC.encode() in generate_latest(unified_registry)
    clock[0] += 29
    assert list(cache['unified'].values())
    assert generate_latest(raw_registry)
    clock[0] += 1
    assert list(cache['unified'].values()) == []
    assert list(cache['raw'].values()) == []
    assert len(cache['unified']) == 0
    assert _WAITING_METRIC not in cache['unified']
    with pytest.raises(KeyError):
        cache['unified'][_WAITING_METRIC]
    assert list(MetricExporter.collect_runtime_metrics(_ExporterProbe(cache))) == []
    assert generate_latest(raw_registry) == b''
    assert generate_latest(unified_registry) == b''


def test_failed_aggregation_attempt_does_not_refresh_successful_cache(monkeypatch):
    import gpustack.worker.runtime_metrics_aggregator as module
    from gpustack.worker.exporter import RawCollector

    clock = [100.0]
    monkeypatch.setattr(module, 'monotonic', lambda: clock[0], raising=False)
    cache = {}
    instance, model = _FakeModelInstance(), _FakeModel()
    state = {'fail': False}

    def endpoints():
        if state['fail']:
            raise RuntimeError('listing unavailable')
        return _active_endpoints(instance, model)

    aggregator = _cache_aggregator(cache, endpoints)
    with patch(f'{module.__name__}.get_backend', return_value='SGLang'):
        aggregator.aggregate()
    state['fail'] = True
    clock[0] += 29
    with pytest.raises(RuntimeError, match='listing unavailable'):
        aggregator.aggregate()
    clock[0] += 2
    assert list(MetricExporter.collect_runtime_metrics(_ExporterProbe(cache))) == []
    assert list(RawCollector(cache).collect()) == []


@pytest.mark.parametrize('result', [None, {}])
def test_failed_or_empty_runtime_scrape_drops_old_samples_without_zero(
    monkeypatch, result
):
    from gpustack.worker.exporter import RawCollector
    import gpustack.worker.runtime_metrics_aggregator as module

    clock = [100.0]
    monkeypatch.setattr(module, 'monotonic', lambda: clock[0])
    cache = {}
    instance, model = _FakeModelInstance(), _FakeModel()
    aggregator = _cache_aggregator(cache, lambda: _active_endpoints(instance, model))
    with patch(f'{module.__name__}.get_backend', return_value='SGLang'):
        aggregator.aggregate()
    assert list(cache['raw'].values())
    aggregator._metrics_client.fetch_metrics_from_endpoints = lambda endpoints: {
        endpoint: result for endpoint in endpoints
    }
    clock[0] += 1
    aggregator.aggregate()
    assert set(cache['unified']) == {'gpustack:runtime_metrics_scrape_status'}
    assert set(cache['raw']) == {'gpustack:runtime_metrics_scrape_status'}
    assert [
        f.name for f in MetricExporter.collect_runtime_metrics(_ExporterProbe(cache))
    ] == ['gpustack:runtime_metrics_scrape_status']
    assert [f.name for f in RawCollector(cache).collect()] == [
        'gpustack:runtime_metrics_scrape_status'
    ]


def _managed_instance(workloads=None, **changes):
    from gpustack.schemas.models import DistributedServers, ModelInstanceWorkload

    values = dict(
        id=999,
        model_id=26,
        name='managed',
        worker_id=1,
        worker_name='primary',
        worker_ip='10.0.0.1',
        cluster_id=4,
        restart_count=3,
        state='running',
        draining_since=None,
        ports=None,
        port=None,
        named_ports=None,
        backend='vLLM',
        backend_version='0.19.0',
        api_detected_backend_version='0.19.0',
        role=None,
        distributed_servers=DistributedServers(
            workloads=(
                [ModelInstanceWorkload.model_validate(w) for w in workloads]
                if workloads is not None
                else None
            )
        ),
    )
    values.update(changes)
    return SimpleNamespace(**values)


def _rank_task(worker_id, rank, *, node_rank=0, state='running'):
    from tests.utils.model import new_model_instance_workload

    return new_model_instance_workload(
        worker_id,
        rank,
        node_rank,
        state=state,
        name=f'rank-{rank}-node-{node_rank}',
        dp_rank_placements=[
            dict(
                dp_rank=rank,
                node_rank=node_rank,
                gpu_indexes=[rank],
                **(
                    {'api_port': 40000 + rank, 'ready': state == 'running'}
                    if node_rank == 0
                    else {}
                ),
            )
        ],
    )


def _component_task(kind, *, worker_id=1, state='running', attempt=0):
    return dict(
        name=f'{kind}-{worker_id}',
        worker_id=worker_id,
        state=state,
        component=dict(
            kind=kind,
            api_port=40000 if kind == 'sglang_engine' else 41000,
            ready=state == 'running',
            **(
                {'dp_size': 4}
                if kind == 'sglang_engine'
                else {'metrics_port': 42000, 'attempt': attempt}
            ),
        ),
    )


def _source_aggregator(instance, *, worker_id=2, model=None):
    from tests.utils.model import make_model

    model = model or make_model(backend='vLLM')
    model.id = 26
    worker = SimpleNamespace(
        id=worker_id,
        name=f'worker-{worker_id}',
        ip=f'10.0.0.{worker_id}',
        cluster_id=4,
        me=True,
    )
    client = SimpleNamespace(
        workers=SimpleNamespace(get=Mock(return_value=worker)),
        model_instances=SimpleNamespace(
            list=Mock(side_effect=lambda **kwargs: SimpleNamespace(items=[instance]))
        ),
        models=SimpleNamespace(list=Mock(return_value=SimpleNamespace(items=[model]))),
    )
    aggregator = RuntimeMetricsAggregator(
        cache={}, worker_id_getter=lambda: worker_id, clientset=client
    )
    aggregator._metrics_config_cache['config'] = get_builtin_metrics_config()
    return aggregator, client


def test_managed_metrics_discovery_uses_cluster_all_rows_and_real_local_owner():
    instance = _managed_instance(
        [
            _rank_task(1, 0),
            _rank_task(2, 1),
            _rank_task(2, 0, node_rank=1),
        ]
    )
    aggregator, client = _source_aggregator(instance)
    endpoints, sources, _ = aggregator._find_active_model_endpoints(
        2, get_builtin_metrics_config()
    )
    assert endpoints == {'10.0.0.2:40001'}
    assert sources['10.0.0.2:40001'].source.dp_rank == 1
    client.model_instances.list.assert_called_once_with(
        params={'cluster_id': 4}, use_cache=True
    )
    client.models.list.assert_called_once_with(use_cache=True)


@pytest.mark.parametrize('workloads, expected', [(None, [40020]), ([], [])])
def test_metrics_source_manifest_empty_is_not_legacy(workloads, expected):
    from gpustack.utils.model_instance_workers import model_instance_metrics_sources

    instance = _managed_instance(workloads, port=40020, ports=[40020])
    assert [
        s.metrics_port for s in model_instance_metrics_sources(instance)
    ] == expected


def test_sglang_shared_engine_survives_starting_smg_and_is_not_logical_dp_sources():
    from gpustack.utils.model_instance_workers import model_instance_metrics_sources

    instance = _managed_instance(
        [
            _component_task('sglang_engine'),
            _component_task('sglang_smg', state='pending', attempt=2),
        ],
        state='starting',
    )
    sources = model_instance_metrics_sources(instance)
    assert [(s.kind, s.metrics_port, s.dp_rank) for s in sources] == [
        ('sglang_engine', 40000, None)
    ]
    instance.distributed_servers.workloads[1].state = 'running'
    instance.distributed_servers.workloads[1].component.ready = True
    sources = model_instance_metrics_sources(instance)
    assert [(s.kind, s.metrics_port, s.attempt) for s in sources] == [
        ('sglang_engine', 40000, None),
        ('sglang_smg', 42000, 2),
    ]


def test_native_heterogeneous_labels_and_platform_collisions_are_preserved():
    families = {
        'vllm:num_requests_running': _family(
            'vllm:num_requests_running',
            'gauge',
            [
                _sample('vllm:num_requests_running', 0, {'worker_id': 'native'}),
                _sample('vllm:num_requests_running', 3, {'priority': 'high'}),
            ],
        )
    }
    unified, raw = _aggregate(families)
    samples = raw['vllm:num_requests_running'].samples
    assert samples[0].labels['worker_id'] == '1'
    assert samples[0].labels['native_worker_id'] == 'native'
    assert samples[1].labels['priority'] == 'high'
    assert [s.value for s in unified['gpustack:num_requests_running'].samples] == [0, 3]


def test_native_platform_collision_does_not_swallow_existing_native_label():
    families = {
        'vllm:num_requests_running': _family(
            'vllm:num_requests_running',
            'gauge',
            [
                _sample(
                    'vllm:num_requests_running',
                    3,
                    {'worker_id': 'native', 'native_worker_id': 'already-native'},
                ),
            ],
        )
    }
    _, raw = _aggregate(families)
    assert not raw.get('vllm:num_requests_running', SimpleNamespace(samples=[])).samples


def test_global_engine_owner_filter_keeps_equal_values_and_unknown_raw_scope():
    instance = _managed_instance([_rank_task(2, 0), _rank_task(2, 1)])
    aggregator, _ = _source_aggregator(instance)
    metrics = _families(
        '# TYPE vllm:prompt_tokens_total counter\n'
        'vllm:prompt_tokens_total{engine="0"} 5\n'
        'vllm:prompt_tokens_total{engine="1"} 5\n'
        '# TYPE vllm:num_requests_running gauge\n'
        'vllm:num_requests_running 0\n',
        PROMETHEUS_CONTENT_TYPE,
    )
    aggregator._metrics_client.fetch_metrics_from_endpoints = lambda endpoints: {
        ep: metrics for ep in endpoints
    }
    aggregator.aggregate()
    samples = aggregator._cache['unified']['gpustack:prompt_tokens'].samples
    assert sorted((s.labels['engine'], s.value) for s in samples) == [
        ('0', 5),
        ('1', 5),
    ]
    assert all(s.labels['worker_id'] == '2' for s in samples)
    assert all(s.labels['restart_count'] == '3' for s in samples)
    assert 'gpustack:num_requests_running' not in aggregator._cache['unified']
    unknown = aggregator._cache['raw']['vllm:num_requests_running'].samples
    assert all(s.labels['exposition_scope'] == 'unknown' for s in unknown)
    assert all('endpoint_dp_rank' not in s.labels for s in unknown)


@pytest.mark.parametrize('change', ['epoch', 'owner', 'attempt', 'removed'])
def test_late_scrape_is_discarded_after_fresh_source_identity_changes(change):
    instance = _managed_instance(
        [_component_task('sglang_engine'), _component_task('sglang_smg', attempt=2)],
        backend='SGLang',
    )
    from tests.utils.model import make_model

    aggregator, client = _source_aggregator(
        instance, worker_id=1, model=make_model(backend='SGLang')
    )
    fresh = _managed_instance(
        [_component_task('sglang_engine'), _component_task('sglang_smg', attempt=2)],
        backend='SGLang',
    )
    if change == 'epoch':
        fresh.restart_count += 1
    elif change == 'owner':
        fresh.distributed_servers.workloads[1].worker_id = 2
    elif change == 'attempt':
        fresh.distributed_servers.workloads[1].component.attempt += 1
    client.model_instances.list.side_effect = [
        SimpleNamespace(items=[instance]),
        SimpleNamespace(items=[] if change == 'removed' else [fresh]),
    ]
    metrics = _families(
        '# TYPE smg_router_requests gauge\nsmg_router_requests 5\n', None
    )
    aggregator._metrics_client.fetch_metrics_from_endpoints = lambda endpoints: {
        ep: metrics for ep in endpoints if ep.endswith(':42000')
    }
    aggregator.aggregate()
    assert not aggregator._cache['raw'].get('smg_router_requests')
    assert client.model_instances.list.call_count == 2


@pytest.mark.parametrize(
    'body, status_code, status',
    [
        ('# TYPE x gauge\nx 0\n', 200, 'success'),
        ('# HELP x empty family\n# TYPE x gauge\n', 200, 'empty'),
        ('not a valid exposition\n', 200, 'parse_error'),
        ('', 503, 'error'),
    ],
)
def test_scrape_outcomes_distinguish_zero_empty_parse_and_http_error(
    monkeypatch, body, status_code, status
):
    from gpustack.worker.runtime_metrics_client import Client, Config

    monkeypatch.setattr(
        'gpustack.worker.runtime_metrics_client.requests.get',
        lambda *a, **kw: SimpleNamespace(
            text=body, status_code=status_code, headers={}
        ),
    )
    result = Client(Config(max_retries=0)).fetch_metrics_from_endpoint('127.0.0.1:1')
    assert result.status == status
    assert bool(result) == (status == 'success')
    if result:
        assert result['x'].samples[0].value == 0


def test_sglang_native_ranks_shared_source_and_smg_raw_are_separate():
    from tests.utils.model import make_model

    instance = _managed_instance(
        [_component_task('sglang_engine'), _component_task('sglang_smg', attempt=2)],
        backend='SGLang',
        api_detected_backend_version=None,
    )
    aggregator, client = _source_aggregator(
        instance, worker_id=1, model=make_model(backend='SGLang')
    )
    engine = _families(
        '# TYPE sglang:num_queue_reqs gauge\n'
        'sglang:num_queue_reqs{dp_rank="0",tp_rank="1",pp_rank="2",'
        'moe_ep_rank="3",engine_type="decode",priority="high"} 0 1700000000000\n'
        'sglang:num_queue_reqs{attn_dp_rank="1"} 5\n',
        None,
    )
    smg = _families(
        '# TYPE smg_router_requests_total counter\nsmg_router_requests_total 7\n', None
    )
    aggregator._metrics_client.fetch_metrics_from_endpoints = lambda endpoints: {
        ep: smg if ep.endswith(':42000') else engine for ep in endpoints
    }
    probe = Mock(return_value='0.5.22')
    aggregator._metrics_client.fetch_runtime_version_from_endpoint = probe
    aggregator.aggregate()
    probe.assert_called_once_with('10.0.0.1:40000', 'SGLang')
    assert not hasattr(client.model_instances, 'update')
    samples = aggregator._cache['unified']['gpustack:num_requests_waiting'].samples
    assert len(samples) == 2
    assert samples[0].labels['dp_rank'] == '0'
    assert samples[0].labels['tp_rank'] == '1'
    assert samples[0].labels['pp_rank'] == '2'
    assert samples[0].labels['moe_ep_rank'] == '3'
    assert samples[0].labels['engine_type'] == 'decode'
    assert samples[0].labels['priority'] == 'high'
    assert samples[0].timestamp == 1700000000
    assert samples[1].labels['attn_dp_rank'] == '1'
    assert all('endpoint_dp_rank' not in s.labels for s in samples)
    assert all(s.labels['exposition_scope'] == 'shared_engine' for s in samples)
    raw_smg = aggregator._cache['raw']['smg_router_requests'].samples
    assert [
        (s.value, s.labels['source_kind'], s.labels['component_attempt'])
        for s in raw_smg
    ] == [(7, 'sglang_smg', '2')]
    assert not any('pd_router' in name for name in aggregator._cache['unified'])


def test_smg_only_success_does_not_probe_an_engine_version():
    from tests.utils.model import make_model

    instance = _managed_instance(
        [_component_task('sglang_engine'), _component_task('sglang_smg')],
        backend='SGLang',
        api_detected_backend_version=None,
    )
    aggregator, _ = _source_aggregator(
        instance, worker_id=1, model=make_model(backend='SGLang')
    )
    smg = _families('# TYPE smg_router_requests gauge\nsmg_router_requests 1\n', None)
    aggregator._metrics_client.fetch_metrics_from_endpoints = lambda endpoints: {
        ep: smg if ep.endswith(':42000') else None for ep in endpoints
    }
    aggregator._metrics_client.fetch_runtime_version_from_endpoint = Mock()
    aggregator.aggregate()
    aggregator._metrics_client.fetch_runtime_version_from_endpoint.assert_not_called()


def test_managed_version_probe_is_once_per_epoch_batch_and_never_persisted():
    instance = _managed_instance(
        [_rank_task(2, 0), _rank_task(2, 1)], api_detected_backend_version=None
    )
    aggregator, client = _source_aggregator(instance)
    payload = _families(PROMETHEUS_TEXT, None)
    aggregator._metrics_client.fetch_metrics_from_endpoints = lambda endpoints: {
        ep: payload for ep in endpoints
    }
    probe = Mock(return_value='0.19.0')
    aggregator._metrics_client.fetch_runtime_version_from_endpoint = probe
    aggregator.aggregate()
    assert probe.call_count == 1
    assert not hasattr(client.model_instances, 'update')


@pytest.mark.parametrize('fallback', [None, '0.18.0'])
def test_managed_version_probe_retries_failure_and_caches_success(fallback):
    instance = _managed_instance(
        [_rank_task(2, 0)],
        api_detected_backend_version=None,
        backend_version=fallback,
    )
    aggregator, _ = _source_aggregator(instance)
    aggregator._metrics_client.fetch_metrics_from_endpoints = lambda endpoints: {
        ep: _families(PROMETHEUS_TEXT, None) for ep in endpoints
    }
    probe = Mock(side_effect=[None, '0.19.0'])
    aggregator._metrics_client.fetch_runtime_version_from_endpoint = probe
    aggregator.aggregate()
    aggregator.aggregate()
    aggregator.aggregate()
    assert probe.call_count == 2
    assert list(aggregator._runtime_version_cache.values()) == ['0.19.0']


def test_conflicting_socket_ownership_is_rejected_instead_of_overwritten():
    instance = _managed_instance([_rank_task(2, 0)])
    other = _managed_instance([_rank_task(2, 0)], id=1000)
    aggregator, client = _source_aggregator(instance)
    client.model_instances.list.side_effect = lambda **kw: SimpleNamespace(
        items=[instance, other]
    )
    endpoints, sources, _ = aggregator._find_active_model_endpoints(
        2, get_builtin_metrics_config()
    )
    assert not endpoints
    assert not sources


@pytest.mark.parametrize(
    'state, ready, draining',
    [
        ('stopped', False, None),
        ('failed', False, None),
        ('pending', False, None),
        ('running', False, None),
        ('running', True, 'draining'),
    ],
)
def test_inactive_or_draining_managed_sources_are_not_scraped(state, ready, draining):
    from gpustack.utils.model_instance_workers import model_instance_metrics_sources

    task = _rank_task(2, 0, state=state)
    task['dp_rank_placements'][0]['ready'] = ready
    instance = _managed_instance([task], draining_since=draining)
    assert not model_instance_metrics_sources(instance)


@pytest.mark.parametrize('status', ['success', 'error', 'parse_error', 'empty'])
def test_source_status_is_bounded_and_business_failure_never_becomes_zero(status):
    from gpustack.worker.runtime_metrics_client import MetricsScrapeResult

    instance = _managed_instance([_rank_task(2, 0)])
    aggregator, _ = _source_aggregator(instance)
    payload = _families(PROMETHEUS_TEXT, None) if status == 'success' else {}
    aggregator._metrics_client.fetch_metrics_from_endpoints = lambda endpoints: {
        ep: MetricsScrapeResult(payload, status=status) for ep in endpoints
    }
    aggregator.aggregate()
    sample = aggregator._cache['raw']['gpustack:runtime_metrics_scrape_status'].samples[
        0
    ]
    assert sample.labels['status'] == status
    assert sample.value == 1
    assert ('gpustack:prompt_tokens' in aggregator._cache['unified']) == (
        status == 'success'
    )
    assert aggregator._cache['unified'][
        'gpustack:runtime_metrics_scrape_status'
    ].samples == [sample]


def test_legacy_pd_metrics_port_and_role_are_unchanged():
    from gpustack.utils.model_instance_workers import model_instance_metrics_sources

    instance = _managed_instance(
        None,
        port=40020,
        ports=[40020, 40030],
        named_ports={'prometheus': {'base': 40030}},
        role='prefill',
    )
    source = model_instance_metrics_sources(instance)[0]
    assert source.api_port == 40020
    assert source.metrics_port == 40030
    assert source.kind == 'legacy'
    assert source.dp_rank is None


@pytest.mark.parametrize(
    'field, replacement',
    [('cluster_id', 5), ('system_principal_id', 8), ('ip', '10.0.0.20')],
)
def test_worker_rebind_or_address_change_discards_inflight_samples(field, replacement):
    instance = _managed_instance([_rank_task(2, 0)])
    aggregator, client = _source_aggregator(instance)
    worker = client.workers.get.return_value
    worker.system_principal_id = 7
    fresh_worker = SimpleNamespace(**vars(worker))
    setattr(fresh_worker, field, replacement)
    client.workers.get.side_effect = [worker, fresh_worker]
    aggregator._metrics_client.fetch_metrics_from_endpoints = lambda endpoints: {
        ep: _families(PROMETHEUS_TEXT, None) for ep in endpoints
    }
    aggregator.aggregate()
    assert not aggregator._cache['unified']
    assert not aggregator._cache['raw']


@pytest.mark.parametrize('me', [False, None, 'missing'])
def test_source_discovery_refuses_worker_no_longer_bound_to_current_principal(me):
    instance = _managed_instance([_rank_task(2, 0)])
    aggregator, client = _source_aggregator(instance)
    if me == 'missing':
        del client.workers.get.return_value.me
    else:
        client.workers.get.return_value.me = me
    with pytest.raises(ValueError, match='current cluster-bound Worker'):
        aggregator._find_active_model_endpoints(2, get_builtin_metrics_config())
    client.model_instances.list.assert_not_called()


def test_legacy_version_update_rechecks_generation_and_rejects_managed_replacement():
    instance = _managed_instance(None, port=40000, ports=[40000])
    aggregator, client = _source_aggregator(instance, worker_id=1)
    update = Mock()
    client.model_instances.update = update
    for current in (
        _managed_instance(None, port=40000, ports=[40000], restart_count=4),
        _managed_instance([_rank_task(1, 0)]),
    ):
        client.model_instances.get = Mock(return_value=current)
        aggregator._update_model_instance(
            instance.id,
            expected_instance=instance,
            api_detected_backend_version='0.19.0',
        )
        client.model_instances.get.assert_called_once_with(
            id=instance.id, use_cache=False
        )
    update.assert_not_called()


def test_unknown_scope_samples_from_two_apis_of_one_task_keep_source_identity():
    task = _rank_task(2, 0)
    task['dp_rank_placements'] += _rank_task(2, 1)['dp_rank_placements']
    instance = _managed_instance([task])
    aggregator, _ = _source_aggregator(instance)
    payload = _families('# TYPE process_count gauge\nprocess_count 1\n', None)
    aggregator._metrics_client.fetch_metrics_from_endpoints = lambda endpoints: {
        ep: payload for ep in endpoints
    }
    aggregator.aggregate()
    samples = aggregator._cache['raw']['process_count'].samples
    identities = {tuple(sorted(s.labels.items())) for s in samples}
    assert len(identities) == len(samples) == 2
    assert all('endpoint_dp_rank' not in s.labels for s in samples)
    assert all('gpu_indexes' not in s.labels for s in samples)


@pytest.mark.parametrize('failed_rank', [None, 0])
def test_shared_three_rank_payload_exports_only_native_owner_without_duplicates(
    failed_rank,
):
    from prometheus_client import CollectorRegistry, generate_latest
    from gpustack.worker.exporter import RawCollector

    task = _rank_task(2, 0)
    for rank in (1, 2):
        task['dp_rank_placements'] += _rank_task(2, rank)['dp_rank_placements']
    instance = _managed_instance([task])
    aggregator, _ = _source_aggregator(instance)
    payload = _families(
        '# TYPE vllm:prompt_tokens_total counter\n'
        'vllm:prompt_tokens_total{engine="0"} 5\n'
        'vllm:prompt_tokens_total{engine="1"} 5\n'
        'vllm:prompt_tokens_total{engine="2"} 5\n'
        '# TYPE process_count gauge\nprocess_count 0\n',
        None,
    )
    failed_endpoint = None if failed_rank is None else f'10.0.0.2:{40000 + failed_rank}'
    aggregator._metrics_client.fetch_metrics_from_endpoints = lambda endpoints: {
        ep: None if ep == failed_endpoint else payload for ep in endpoints
    }
    aggregator.aggregate()
    expected_ranks = {'0', '1', '2'} - (
        {str(failed_rank)} if failed_rank is not None else set()
    )
    raw = aggregator._cache['raw']['vllm:prompt_tokens'].samples
    assert len(raw) == len(expected_ranks)
    assert {s.labels['engine'] for s in raw} == expected_ranks
    unified = aggregator._cache['unified']['gpustack:prompt_tokens'].samples
    assert {s.labels['engine'] for s in unified} == expected_ranks
    for collector in (
        RawCollector(aggregator._cache),
        _ExporterProbe(aggregator._cache),
    ):
        registry = CollectorRegistry()
        registry.register(collector)
        rendered = generate_latest(registry).decode()
        samples = [s for family in parse_metrics_text(rendered) for s in family.samples]
        identities = [(s.name, tuple(sorted(s.labels.items()))) for s in samples]
        assert len(identities) == len(set(identities))


@pytest.mark.parametrize('target_type', ['counter', 'gauge'])
def test_counter_mapping_total_suffix_uses_canonical_prometheus_family_name(
    target_type,
):
    from prometheus_client import CollectorRegistry, generate_latest
    from gpustack.worker.exporter import RawCollector

    aggregator = RuntimeMetricsAggregator(cache={}, worker_id_getter=lambda: 1)
    config = {
        'runtime_mapping': {
            'custom': {'*': {'requests_total': 'gpustack:requests_total'}}
        },
        'gpustack_metrics': {
            'gpustack:requests_total': {'type': target_type, 'description': 'requests'}
        },
    }
    unified, raw = {}, {}
    aggregator._process_endpoint_metrics(
        {
            'requests_total': _family(
                'requests_total', 'counter', [_sample('requests_total', 4)]
            )
        },
        {'worker_id': '1'},
        'custom',
        None,
        unified,
        raw,
        config,
    )
    for key, metrics, expected in (
        ('raw', raw, 'requests_total'),
        ('unified', unified, 'gpustack:requests_total'),
    ):
        registry = CollectorRegistry()
        collector = (
            RawCollector({'raw': metrics})
            if key == 'raw'
            else _ExporterProbe({'unified': metrics})
        )
        registry.register(collector)
        rendered = generate_latest(registry).decode()
        samples = [s for family in parse_metrics_text(rendered) for s in family.samples]
        assert [(s.name, s.value) for s in samples] == [(expected, 4)]


def test_source_inventory_keeps_registered_pending_and_failed_components():
    from gpustack.utils.model_instance_workers import (
        model_instance_metrics_source_inventory,
        model_instance_metrics_sources,
    )

    instance = _managed_instance(
        [
            _component_task('sglang_engine', state='failed'),
            _component_task('sglang_smg', state='pending', attempt=3),
        ],
        state='error',
    )
    inventory = model_instance_metrics_source_inventory(instance)
    assert [(s.kind, s.workload_state, s.ready) for s in inventory] == [
        ('sglang_engine', 'failed', False),
        ('sglang_smg', 'pending', False),
    ]
    assert inventory[0].dp_size == 4
    assert inventory[1].attempt == 3
    assert not model_instance_metrics_sources(instance)


def test_default_metrics_include_exact_smg_domain_and_scrape_status():
    from prometheus_client import CollectorRegistry, generate_latest
    from tests.utils.model import make_model

    instance = _managed_instance(
        [
            _component_task('sglang_engine'),
            _component_task('sglang_smg', attempt=2),
        ],
        backend='SGLang',
    )
    aggregator, _ = _source_aggregator(
        instance, worker_id=1, model=make_model(backend='SGLang')
    )
    payload = _families(
        '# TYPE smg_router_requests_total counter\n'
        'smg_router_requests_total{router_type="http",backend_type="regular",'
        'connection_mode="http",model="native",endpoint="chat",streaming="true"} 7\n'
        '# TYPE smg_worker_selection_total counter\n'
        'smg_worker_selection_total{worker_type="regular",connection_mode="http",'
        'model="native",policy="round_robin"} 9\n'
        '# TYPE smg_worker_requests_active gauge\n'
        'smg_worker_requests_active{worker="opaque-engine@2"} 0\n'
        '# TYPE sglang:num_running_reqs gauge\nsglang:num_running_reqs 99\n',
        None,
    )
    aggregator._metrics_client.fetch_metrics_from_endpoints = lambda endpoints: {
        ep: payload if ep.endswith(':42000') else None for ep in endpoints
    }
    aggregator.aggregate()
    registry = CollectorRegistry()
    exporter = MetricExporter(
        cfg=SimpleNamespace(worker_metrics_port=10151),
        collector=None,
        worker_name_getter=lambda: 'worker-1',
        worker_ip_getter=lambda: '10.0.0.1',
        worker_id_getter=lambda: 1,
        cache=aggregator._cache,
    )
    exporter.collect_worker_metrics = lambda: iter([])
    registry.register(exporter)
    samples = [
        s
        for f in parse_metrics_text(generate_latest(registry).decode())
        for s in f.samples
    ]
    values = {s.name: s for s in samples if not s.name.endswith('scrape_status')}
    assert {k: v.value for k, v in values.items()} == {
        'gpustack:smg_router_requests_total': 7,
        'gpustack:smg_worker_selections_total': 9,
        'gpustack:smg_worker_requests_active': 0,
    }
    assert (
        values['gpustack:smg_worker_requests_active'].labels['worker']
        == 'opaque-engine@2'
    )
    assert all(v.labels['component_attempt'] == '2' for v in values.values())
    assert {
        s.labels['status'] for s in samples if s.name.endswith('scrape_status')
    } == {'success', 'error'}


def test_legacy_pd_mapping_and_engine_metrics_keep_their_domain():
    payload = _families(
        '# TYPE smg_router_requests_total counter\nsmg_router_requests_total 3\n'
        '# TYPE smg_worker_selection_total counter\nsmg_worker_selection_total 4\n'
        '# TYPE smg_worker_requests_active gauge\nsmg_worker_requests_active 0\n'
        '# TYPE sglang:num_queue_reqs gauge\nsglang:num_queue_reqs 2\n',
        None,
    )
    aggregator = RuntimeMetricsAggregator(cache={}, worker_id_getter=lambda: 1)
    unified, raw = {}, {}
    aggregator._process_endpoint_metrics(
        payload,
        {'source_kind': 'legacy', 'role': 'router'},
        'SGLang',
        '0.5.22',
        unified,
        raw,
        get_builtin_metrics_config(),
    )
    assert set(unified) == {
        'gpustack:pd_router_requests',
        'gpustack:pd_router_worker_selections',
        'gpustack:pd_router_worker_requests_active',
        'gpustack:num_requests_waiting',
    }
    assert all(
        s.labels['role'] == 'router' for f in unified.values() for s in f.samples
    )
    assert {s.name for f in unified.values() for s in f.samples} == {
        'gpustack:pd_router_requests_total',
        'gpustack:pd_router_worker_selections_total',
        'gpustack:pd_router_worker_requests_active',
        'gpustack:num_requests_waiting',
    }


def test_managed_version_probe_cache_is_fenced_by_generation():
    instance = _managed_instance([_rank_task(2, 0)], api_detected_backend_version=None)
    aggregator, _ = _source_aggregator(instance)
    aggregator._metrics_client.fetch_metrics_from_endpoints = lambda endpoints: {
        ep: _families(PROMETHEUS_TEXT, None) for ep in endpoints
    }
    probe = Mock(return_value='0.30.0')
    aggregator._metrics_client.fetch_runtime_version_from_endpoint = probe
    aggregator.aggregate()
    aggregator.aggregate()
    assert probe.call_count == 1
    instance.restart_count += 1
    aggregator.aggregate()
    assert probe.call_count == 2


def test_scrape_batch_deadline_preserves_completed_endpoints_and_bounds_threads():
    import threading
    import time
    from gpustack.worker.runtime_metrics_client import (
        Client,
        Config,
        MetricsScrapeResult,
    )

    client = Client(Config(batch_timeout=0.05))
    release = threading.Event()
    calls = []

    def fetch(endpoint):
        calls.append(endpoint)
        if endpoint == 'hung':
            release.wait(2)
        return MetricsScrapeResult({'samples': 1}, status='success')

    client.fetch_metrics_from_endpoint = fetch
    started = time.monotonic()
    try:
        result = client.fetch_metrics_from_endpoints(['good', 'hung'], max_workers=2)
        assert time.monotonic() - started < 1
        assert result['good'].status == 'success'
        assert result['hung'].status == 'error'
        pool = client._executor
        client.fetch_metrics_from_endpoints(['good', 'hung'], max_workers=2)
        assert client._executor is pool
        assert calls.count('hung') == 1
    finally:
        release.set()
        client._executor.shutdown(wait=True)


def test_discovery_reads_full_watch_snapshots_without_list_http():
    from gpustack.client.generated_model_client import ModelClient
    from gpustack.client.generated_model_instance_client import ModelInstanceClient

    from tests.utils.model import new_model_instance
    from gpustack.schemas.models import ModelInstancePublic, ModelPublic, SourceEnum
    from datetime import datetime, timezone

    instance = new_model_instance(1, 'mi', 26, worker_id=2)
    instance.source = SourceEnum.HUGGING_FACE
    instance.created_at = instance.updated_at = datetime.now(timezone.utc)
    instance.huggingface_repo_id = 'org/model'
    instance.cluster_id = 4
    instance = ModelInstancePublic.model_validate(instance.model_dump())
    aggregator, client = _source_aggregator(instance)
    http = Mock()
    models = ModelClient(http)
    instances = ModelInstanceClient(http)
    models._watch_started = instances._watch_started = True
    model = client.models.list().items[0]
    model.created_at = model.updated_at = datetime.now(timezone.utc)
    models._cache = {26: ModelPublic.model_validate(model.model_dump())}
    instances._cache = {
        id_: instance.model_copy(update={'id': id_}) for id_ in range(150)
    }
    client.models, client.model_instances = models, instances
    rows, model_rows = aggregator._list_worker_models(2)
    assert len(rows.items) == 150
    assert len(model_rows.items) == 1
    http.get_httpx_client.assert_not_called()


def test_late_batch_results_cannot_be_relabelled_as_a_new_generation():
    import threading
    from gpustack.worker.runtime_metrics_client import (
        Client,
        Config,
        MetricsScrapeResult,
    )

    client = Client(Config(batch_timeout=0.02))
    release = threading.Event()
    calls = []

    def fetch(endpoint):
        calls.append(endpoint)
        generation = len(calls)
        if generation == 1:
            release.wait(2)
        return MetricsScrapeResult({'generation': generation}, status='success')

    client.fetch_metrics_from_endpoint = fetch
    try:
        assert client.fetch_metrics_from_endpoints(['api'])['api'].status == 'error'
        old = next(iter(client._inflight.values()))
        assert client.fetch_metrics_from_endpoints(['api'])['api'].status == 'error'
        release.set()
        old.result(timeout=1)
        assert client.fetch_metrics_from_endpoints(['api'])['api']['generation'] == 2
    finally:
        release.set()
        client._executor.shutdown(wait=True)


def _hybrid_tasks():
    return [
        dict(
            name=f'hybrid-{worker_id}',
            worker_id=worker_id,
            state='running',
            component=dict(
                kind='vllm_hybrid',
                api_port=40000 if worker_id in (1, 3) else None,
                dp_size=1 if worker_id in (1, 3) else None,
                ready=True if worker_id in (1, 3) else None,
            ),
        )
        for worker_id in (1, 2, 3, 4)
    ]


@pytest.mark.parametrize('worker_id,expected', [(1, 1), (2, 0), (3, 1), (4, 0)])
def test_hybrid_scrapes_each_shared_api_once_and_preserves_all_native_engines(
    worker_id, expected
):
    instance = _managed_instance(_hybrid_tasks())
    aggregator, _ = _source_aggregator(instance, worker_id=worker_id)
    metrics = _families(
        '# TYPE vllm:prompt_tokens_total counter\n'
        'vllm:prompt_tokens_total{engine="0"} 5\n'
        'vllm:prompt_tokens_total{engine="1"} 7\n',
        PROMETHEUS_CONTENT_TYPE,
    )
    calls = []

    def fetch(endpoints):
        calls.extend(endpoints)
        return {ep: metrics for ep in endpoints}

    aggregator._metrics_client.fetch_metrics_from_endpoints = fetch
    aggregator.aggregate()
    assert calls == ([f'10.0.0.{worker_id}:40000'] if expected else [])
    samples = aggregator._cache['unified'].get('gpustack:prompt_tokens')
    if not expected:
        assert samples is None
        return
    assert [(s.labels['engine'], s.value) for s in samples.samples] == [
        ('0', 5),
        ('1', 7),
    ]
    assert all(s.labels['source_kind'] == 'vllm_hybrid' for s in samples.samples)
    assert all('endpoint_dp_rank' not in s.labels for s in samples.samples)


@pytest.mark.parametrize('change', ['epoch', 'owner', 'removed', 'unready'])
def test_hybrid_late_metrics_cannot_relabel_replaced_shared_api(change):
    instance = _managed_instance(_hybrid_tasks())
    fresh = _managed_instance(_hybrid_tasks())
    if change == 'epoch':
        fresh.restart_count += 1
    elif change == 'owner':
        fresh.distributed_servers.workloads[0].worker_id = 9
    elif change == 'unready':
        fresh.distributed_servers.workloads[0].component.ready = False
    aggregator, client = _source_aggregator(instance, worker_id=1)
    client.model_instances.list.side_effect = [
        SimpleNamespace(items=[instance]),
        SimpleNamespace(items=[] if change == 'removed' else [fresh]),
    ]
    metrics = _families(
        '# TYPE vllm:num_requests_running gauge\nvllm:num_requests_running 2\n', None
    )
    aggregator._metrics_client.fetch_metrics_from_endpoints = lambda endpoints: {
        ep: metrics for ep in endpoints
    }
    aggregator.aggregate()
    assert not aggregator._cache['raw'].get('vllm:num_requests_running')
