import json
import math
from pathlib import Path

import pytest
import yaml
from prometheus_client import CollectorRegistry, generate_latest

from gpustack.utils.metrics import (
    get_builtin_metrics_config,
    get_builtin_metrics_config_file_path,
)
from gpustack.worker.runtime_metrics_aggregator import (
    RuntimeMetricsAggregator,
    _parse_mapping_entry,
)
from gpustack.worker.runtime_metrics_client import parse_metrics_text

FIXTURES = Path(__file__).resolve().parents[1] / 'fixtures' / 'runtime_metrics'
BASE_LABELS = {
    'worker_id': '1',
    'worker_name': 'fixture-worker',
    'model_id': '7',
    'model_name': 'fixture-model',
    'model_instance_id': '11',
}
PROMETHEUS_CONTENT_TYPE = 'text/plain; version=0.0.4; charset=utf-8'
OPENMETRICS_CONTENT_TYPE = 'application/openmetrics-text; version=1.0.0; charset=utf-8'
AUDITED_CASES = json.loads((FIXTURES / 'audited_mappings.json').read_text())['cases']
AUDITED_METRICS = [
    (case['id'], metric) for case in AUDITED_CASES for metric in case['metrics']
]


def _normalize(
    text,
    runtime,
    runtime_version,
    content_type=PROMETHEUS_CONTENT_TYPE,
    base_labels=None,
):
    families = {
        family.name: family for family in parse_metrics_text(text, content_type)
    }
    aggregator = RuntimeMetricsAggregator(cache={}, worker_id_getter=lambda: 1)
    unified, raw = {}, {}
    aggregator._process_endpoint_metrics(
        families,
        {**BASE_LABELS, **(base_labels or {})},
        runtime,
        runtime_version,
        unified,
        raw,
        get_builtin_metrics_config(),
    )
    return unified, raw


def _renamed_samples(source, source_name, target_name):
    return [
        (
            sample.name.replace(source_name, target_name, 1),
            {**sample.labels, **BASE_LABELS},
            sample.value,
        )
        for sample in source.samples
    ]


def _assert_exported_samples_are_unique(families):
    class Collector:
        def collect(self):
            yield from families.values()

    registry = CollectorRegistry()
    registry.register(Collector())
    rendered = generate_latest(registry).decode()
    samples = [
        sample
        for family in parse_metrics_text(rendered, PROMETHEUS_CONTENT_TYPE)
        for sample in family.samples
    ]
    identities = [
        (sample.name, tuple(sorted(sample.labels.items()))) for sample in samples
    ]
    assert len(identities) == len(set(identities))


def _assert_unique_yaml_mapping_keys(node):
    if isinstance(node, yaml.MappingNode):
        names = [key.value for key, _ in node.value]
        assert len(names) == len(
            set(names)
        ), f'Duplicate YAML mapping keys near line {node.start_mark.line + 1}: {names}'
        for _, value in node.value:
            _assert_unique_yaml_mapping_keys(value)
    elif isinstance(node, yaml.SequenceNode):
        for value in node.value:
            _assert_unique_yaml_mapping_keys(value)


def test_builtin_metrics_yaml_has_no_silent_duplicate_keys():
    text = Path(get_builtin_metrics_config_file_path()).read_text()

    _assert_unique_yaml_mapping_keys(yaml.compose(text))


def test_builtin_runtime_mapping_targets_are_declared_metric_families():
    config = get_builtin_metrics_config()
    for runtime, versions in config['runtime_mapping'].items():
        for version, mapping in versions.items():
            for source, entry in mapping.items():
                if entry is None:
                    continue
                resolved = _parse_mapping_entry(entry)
                assert resolved is not None, (runtime, version, source, entry)
                target, _ = resolved
                assert target in config['gpustack_metrics'], (runtime, version, source)
                assert config['gpustack_metrics'][target]['type'].lower() in {
                    'counter',
                    'gauge',
                    'histogram',
                    'summary',
                    'info',
                    'unknown',
                    'untyped',
                }


@pytest.fixture(scope='module')
def audited_normalized_metrics():
    return {
        case['id']: _normalize(
            (FIXTURES / case['fixture']).read_text(),
            case['runtime'],
            case['version'],
            base_labels=case['base_labels'],
        )
        for case in AUDITED_CASES
    }


@pytest.mark.parametrize(
    'case_id,metric',
    AUDITED_METRICS,
    ids=[f'{case_id}-{metric["raw"]}' for case_id, metric in AUDITED_METRICS],
)
def test_audited_engine_samples_keep_type_labels_and_units(
    case_id, metric, audited_normalized_metrics
):
    unified, raw = audited_normalized_metrics[case_id]
    source, target = metric['raw'], metric['canonical']
    assert source in raw
    assert target in unified, (source, target, metric['source'])
    assert raw[source].type == metric['type'].lower()
    assert unified[target].type == metric['type'].lower()
    assert len(raw[source].samples) == len(metric['samples'])
    assert len(unified[target].samples) == len(metric['samples'])

    for native, normalized, expected in zip(
        raw[source].samples, unified[target].samples, metric['samples']
    ):
        suffix = expected['suffix']
        assert native.name == source + suffix
        assert normalized.name == target + suffix
        assert native.value == pytest.approx(expected['value'])
        for label, value in expected['labels'].items():
            assert native.labels[label] == value
        for label, value in BASE_LABELS.items():
            assert native.labels[label] == value
        expected_labels = dict(native.labels)
        actual_labels = dict(normalized.labels)
        if suffix == '_bucket' and metric['scale'] != 1:
            boundary = expected_labels.pop('le')
            actual_boundary = actual_labels.pop('le')
            assert float(actual_boundary) == pytest.approx(
                float(boundary) * metric['scale']
            )
        assert actual_labels == expected_labels
        expected_value = expected['value']
        if suffix not in ('_bucket', '_count'):
            expected_value *= metric['scale']
        assert normalized.value == pytest.approx(expected_value)


@pytest.mark.parametrize('case_id', [case['id'] for case in AUDITED_CASES])
def test_audited_fixture_exports_have_no_duplicate_series(
    case_id, audited_normalized_metrics
):
    unified, raw = audited_normalized_metrics[case_id]

    _assert_exported_samples_are_unique(unified)
    _assert_exported_samples_are_unique(raw)


def test_vllm_iteration_tokens_and_request_preemptions_keep_histogram_semantics(
    audited_normalized_metrics,
):
    unified, _ = audited_normalized_metrics['vllm_v0_33_0rc3']

    for name in ('gpustack:iteration_tokens', 'gpustack:request_num_preemptions'):
        assert unified[name].type == 'histogram'
        assert [
            sample.value
            for sample in unified[name].samples
            if sample.name.endswith('_count')
        ] == [4]
    cumulative = unified['gpustack:request_preemptions']
    assert cumulative.type == 'counter'
    assert cumulative.samples[0].name == 'gpustack:request_preemptions_total'


def test_vllm_cached_total_and_by_source_are_not_folded_together(
    audited_normalized_metrics,
):
    unified, _ = audited_normalized_metrics['vllm_v0_33_0rc3']
    aggregate = unified['gpustack:prompt_tokens_cached']
    breakdown = unified['gpustack:prompt_tokens_cached_by_source']

    assert aggregate.type == breakdown.type == 'counter'
    assert len(aggregate.samples) == len(breakdown.samples) == 1
    assert 'source' not in aggregate.samples[0].labels
    assert breakdown.samples[0].labels['source'] == 'host'


def test_vllm_offload_bytes_and_cpu_allocation_chunks_are_separate_histograms(
    audited_normalized_metrics,
):
    unified, _ = audited_normalized_metrics['vllm_v0_33_0rc3']
    byte_sizes = unified['gpustack:kv_offload_load_bytes_per_operation']
    chunk_sizes = unified['gpustack:kv_offload_cpu_allocation_chunks']

    assert byte_sizes.type == chunk_sizes.type == 'histogram'
    assert [
        sample.value for sample in byte_sizes.samples if sample.name.endswith('_sum')
    ] == [10240]
    assert [
        sample.value for sample in chunk_sizes.samples if sample.name.endswith('_sum')
    ] == [12]


def test_sglang_window_cache_and_swa_projection_keep_distinct_families(
    audited_normalized_metrics,
):
    unified, _ = audited_normalized_metrics['sglang_main_1be97a0']
    window = unified['gpustack:window_kv_cache_usage_ratio']
    projection = unified['gpustack:swa_kv_cache_usage_ratio']

    assert window.type == projection.type == 'gauge'
    assert len(window.samples) == len(projection.samples) == 1
    assert window.samples[0].labels['window'] == 'window_4096'
    assert 'window' not in projection.samples[0].labels
    assert window.samples[0].value == projection.samples[0].value == 0.5


@pytest.mark.parametrize(
    'content_type,trailer',
    [
        (PROMETHEUS_CONTENT_TYPE, ''),
        (OPENMETRICS_CONTENT_TYPE, '# EOF\n'),
    ],
)
def test_sglang_request_tpot_and_itl_remain_distinct_histograms(
    content_type, trailer, caplog
):
    text = (FIXTURES / 'sglang_v0_5_23_timing.prom').read_text() + trailer
    unified, raw = _normalize(text, 'SGLang', '0.5.23', content_type)

    assert not any(
        'Failed to parse OpenMetrics' in record.message for record in caplog.records
    )

    tpot_name = 'gpustack:time_per_output_token_seconds'
    itl_name = 'gpustack:inter_token_latency_seconds'
    assert tpot_name in unified
    assert itl_name in unified
    assert unified[tpot_name].type == 'histogram'
    assert unified[itl_name].type == 'histogram'

    for source_name, target_name in (
        ('sglang:request_time_per_output_token_seconds', tpot_name),
        ('sglang:inter_token_latency_seconds', itl_name),
    ):
        assert [
            (sample.name, sample.labels, sample.value)
            for sample in unified[target_name].samples
        ] == _renamed_samples(raw[source_name], source_name, target_name)

    assert {sample.labels['is_streaming'] for sample in unified[tpot_name].samples} == {
        'true',
        'false',
    }
    assert all(
        'is_streaming' not in sample.labels for sample in unified[itl_name].samples
    )
    assert {
        sample.labels['is_streaming']: sample.value
        for sample in unified[tpot_name].samples
        if sample.name.endswith('_count')
    } == {'true': 3, 'false': 2}
    assert [
        sample.value
        for sample in unified[itl_name].samples
        if sample.name.endswith('_count')
    ] == [22]
    _assert_exported_samples_are_unique(unified)
    _assert_exported_samples_are_unique(raw)


def test_sglang_cached_token_sources_and_native_model_identity_survive_normalization():
    text = (
        '# TYPE sglang:cached_tokens_total counter\n'
        'sglang:cached_tokens_total{model_name="engine-model",cache_source="device"} 10\n'
        'sglang:cached_tokens_total{model_name="engine-model",cache_source="host"} 20\n'
        'sglang:cached_tokens_total{model_name="engine-model",cache_source="storage"} 30\n'
    )
    unified, raw = _normalize(
        text, 'SGLang', '0.5.23', base_labels={'source_kind': 'sglang_engine'}
    )

    for family in (
        raw['sglang:cached_tokens'],
        unified['gpustack:prompt_tokens_cached'],
    ):
        assert family.type == 'counter'
        assert len(family.samples) == 3
        assert {
            sample.labels['cache_source']: sample.value for sample in family.samples
        } == {
            'device': 10,
            'host': 20,
            'storage': 30,
        }
        assert all(
            sample.labels['model_name'] == 'fixture-model' for sample in family.samples
        )
        assert all(
            sample.labels['native_model_name'] == 'engine-model'
            for sample in family.samples
        )
    _assert_exported_samples_are_unique(unified)
    _assert_exported_samples_are_unique(raw)


def test_sglang_prompt_and_generation_counters_keep_streaming_dimensions():
    text = (
        '# TYPE sglang:prompt_tokens_total counter\n'
        'sglang:prompt_tokens_total{model_name="fixture-model",is_streaming="true"} 11\n'
        'sglang:prompt_tokens_total{model_name="fixture-model",is_streaming="false"} 19\n'
        '# TYPE sglang:generation_tokens_total counter\n'
        'sglang:generation_tokens_total{model_name="fixture-model",is_streaming="true"} 7\n'
        'sglang:generation_tokens_total{model_name="fixture-model",is_streaming="false"} 13\n'
    )
    unified, raw = _normalize(text, 'SGLang', '0.5.23')

    for source, target, expected in (
        ('sglang:prompt_tokens', 'gpustack:prompt_tokens', {'true': 11, 'false': 19}),
        (
            'sglang:generation_tokens',
            'gpustack:generation_tokens',
            {'true': 7, 'false': 13},
        ),
    ):
        for family in (raw[source], unified[target]):
            assert family.type == 'counter'
            assert len(family.samples) == 2
            assert {
                sample.labels['is_streaming']: sample.value for sample in family.samples
            } == expected
    _assert_exported_samples_are_unique(unified)
    _assert_exported_samples_are_unique(raw)


@pytest.mark.parametrize('runtime_version', ['0.5.7', '0.5.12', '0.5.23'])
def test_sglang_current_retractions_and_cumulative_preemptions_stay_separate(
    runtime_version,
):
    current_name = 'gpustack:num_requests_retracted'
    total_name = 'gpustack:request_preemptions'
    for current, cumulative in ((4, 7), (1, 9)):
        text = (
            '# TYPE sglang:num_retracted_reqs gauge\n'
            f'sglang:num_retracted_reqs{{model_name="fixture-model"}} {current}\n'
            '# TYPE sglang:num_retracted_requests_total counter\n'
            'sglang:num_retracted_requests_total'
            f'{{model_name="fixture-model"}} {cumulative}\n'
        )
        unified, raw = _normalize(text, 'SGLang', runtime_version)

        assert unified[current_name].type == 'gauge'
        assert [(s.name, s.value) for s in unified[current_name].samples] == [
            (current_name, current)
        ]
        assert unified[total_name].type == 'counter'
        assert [(s.name, s.value) for s in unified[total_name].samples] == [
            (f'{total_name}_total', cumulative)
        ]
        assert raw['sglang:num_retracted_reqs'].type == 'gauge'
        assert raw['sglang:num_retracted_requests'].type == 'counter'


def test_sglang_retraction_gauge_does_not_create_an_absent_counter():
    text = (
        '# TYPE sglang:num_retracted_reqs gauge\n'
        'sglang:num_retracted_reqs{model_name="fixture-model"} 4\n'
    )
    unified, _ = _normalize(text, 'SGLang', '0.5.6')

    assert unified['gpustack:num_requests_retracted'].type == 'gauge'
    assert 'gpustack:request_preemptions' not in unified


def test_sglang_v0_5_2_cumulative_and_interval_gauges_keep_separate_families():
    text = (
        '# TYPE sglang:total_retracted_reqs gauge\n'
        'sglang:total_retracted_reqs{model_name="fixture-model"} 17\n'
        '# TYPE sglang:num_retracted_reqs gauge\n'
        'sglang:num_retracted_reqs{model_name="fixture-model"} 2\n'
    )
    unified, _ = _normalize(text, 'SGLang', '0.5.2')

    snapshot = unified['gpustack:retracted_requests_total_snapshot']
    assert snapshot.type == 'gauge'
    assert [(sample.name, sample.value) for sample in snapshot.samples] == [
        ('gpustack:retracted_requests_total_snapshot', 17)
    ]
    interval = unified['gpustack:num_requests_retracted']
    assert interval.type == 'gauge'
    assert [sample.value for sample in interval.samples] == [2]
    assert 'gpustack:request_preemptions' not in unified
    _assert_exported_samples_are_unique(unified)


@pytest.mark.parametrize('runtime_version', ['0.5.3', '0.5.23', None])
def test_sglang_cumulative_gauge_snapshot_is_raw_only_outside_verified_versions(
    runtime_version,
):
    source = 'sglang:total_retracted_reqs'
    text = f'# TYPE {source} gauge\n{source}{{model_name="fixture-model"}} 17\n'
    unified, raw = _normalize(text, 'SGLang', runtime_version)

    assert unified == {}
    assert raw[source].type == 'gauge'


@pytest.mark.parametrize(
    'source_name,target_name,counter_name',
    [
        (
            'sglang:spec_num_draft_tokens',
            'gpustack:spec_decode_configured_draft_tokens',
            'gpustack:spec_decode_draft_tokens',
        ),
        (
            'sglang:spec_num_steps',
            'gpustack:spec_decode_configured_steps',
            'gpustack:spec_decode_drafts',
        ),
    ],
)
@pytest.mark.parametrize('configured_value', [5, 3])
def test_sglang_speculative_configuration_is_not_a_cumulative_denominator(
    source_name, target_name, counter_name, configured_value
):
    text = (
        f'# TYPE {source_name} gauge\n'
        f'{source_name}{{model_name="fixture-model"}} {configured_value}\n'
    )
    unified, _ = _normalize(text, 'SGLang', '0.5.23')

    assert unified[target_name].type == 'gauge'
    assert [(sample.name, sample.value) for sample in unified[target_name].samples] == [
        (target_name, configured_value)
    ]
    assert counter_name not in unified


def _histogram_text(name):
    return (
        f'# TYPE {name} histogram\n'
        f'{name}_bucket{{model_name="fixture-model",le="0.01"}} 1\n'
        f'{name}_bucket{{model_name="fixture-model",le="0.1"}} 2\n'
        f'{name}_bucket{{model_name="fixture-model",le="+Inf"}} 2\n'
        f'{name}_sum{{model_name="fixture-model"}} 0.05\n'
        f'{name}_count{{model_name="fixture-model"}} 2\n'
    )


def test_vllm_legacy_per_token_histogram_is_itl_before_the_explicit_name():
    source_name = 'vllm:time_per_output_token_seconds'
    target_name = 'gpustack:inter_token_latency_seconds'
    unified, raw = _normalize(_histogram_text(source_name), 'vLLM', '0.10.1')

    assert 'gpustack:time_per_output_token_seconds' not in unified
    assert unified[target_name].type == 'histogram'
    assert [
        (sample.name, sample.labels, sample.value)
        for sample in unified[target_name].samples
    ] == _renamed_samples(raw[source_name], source_name, target_name)


@pytest.mark.parametrize('runtime_version', ['0.10.2', '0.11.0', '0.32.0'])
def test_vllm_explicit_itl_is_not_duplicated_by_its_legacy_alias(runtime_version):
    legacy_name = 'vllm:time_per_output_token_seconds'
    explicit_name = 'vllm:inter_token_latency_seconds'
    tpot_name = 'vllm:request_time_per_output_token_seconds'
    text = ''.join(
        _histogram_text(name) for name in (legacy_name, explicit_name, tpot_name)
    )
    unified, raw = _normalize(text, 'vLLM', runtime_version)

    assert legacy_name in raw
    assert set(unified) == {
        'gpustack:inter_token_latency_seconds',
        'gpustack:time_per_output_token_seconds',
    }
    for source_name, target_name in (
        (explicit_name, 'gpustack:inter_token_latency_seconds'),
        (tpot_name, 'gpustack:time_per_output_token_seconds'),
    ):
        samples = unified[target_name].samples
        assert [
            (sample.name, sample.labels, sample.value) for sample in samples
        ] == _renamed_samples(raw[source_name], source_name, target_name)
        identities = [
            (sample.name, tuple(sorted(sample.labels.items()))) for sample in samples
        ]
        assert len(identities) == len(set(identities))
    _assert_exported_samples_are_unique(unified)
    _assert_exported_samples_are_unique(raw)


@pytest.mark.parametrize('runtime_version', ['0.4.4', '0.4.4.post1', '0.4.4.post3'])
def test_sglang_legacy_per_token_histogram_keeps_its_e2e_semantics(runtime_version):
    source_name = 'sglang:time_per_output_token_seconds'
    target_name = 'gpustack:e2e_time_per_output_token_seconds'
    unified, raw = _normalize(_histogram_text(source_name), 'SGLang', runtime_version)

    assert set(unified) == {target_name}
    assert unified[target_name].type == 'histogram'
    assert [
        (sample.name, sample.labels, sample.value)
        for sample in unified[target_name].samples
    ] == _renamed_samples(raw[source_name], source_name, target_name)


@pytest.mark.parametrize('runtime_version', ['0.4.4.post4', '0.4.5', '0.5.23', None])
def test_sglang_legacy_per_token_name_is_raw_only_outside_verified_versions(
    runtime_version,
):
    source_name = 'sglang:time_per_output_token_seconds'
    unified, raw = _normalize(_histogram_text(source_name), 'SGLang', runtime_version)

    assert unified == {}
    assert source_name in raw


@pytest.mark.parametrize(
    'runtime,runtime_version,source_name,target_name,family_type',
    [
        (
            'vLLM',
            '0.32.0',
            'vllm:num_requests_running',
            'gpustack:num_requests_running',
            'gauge',
        ),
        (
            'SGLang',
            '0.5.23',
            'sglang:num_running_reqs',
            'gpustack:num_requests_running',
            'gauge',
        ),
        ('vLLM', '0.32.0', 'vllm:prompt_tokens', 'gpustack:prompt_tokens', 'counter'),
        (
            'SGLang',
            '0.5.23',
            'sglang:prompt_tokens',
            'gpustack:prompt_tokens',
            'counter',
        ),
        (
            'vLLM',
            '0.32.0',
            'vllm:request_prompt_tokens',
            'gpustack:request_prompt_tokens',
            'histogram',
        ),
        (
            'SGLang',
            '0.5.23',
            'sglang:prompt_tokens_histogram',
            'gpustack:request_prompt_tokens',
            'histogram',
        ),
    ],
)
def test_current_engine_exposition_keeps_metric_type_and_native_values(
    runtime, runtime_version, source_name, target_name, family_type
):
    if family_type == 'histogram':
        text = _histogram_text(source_name)
    else:
        exposed_name = source_name + ('_total' if family_type == 'counter' else '')
        text = (
            f'# TYPE {exposed_name} {family_type}\n'
            f'{exposed_name}{{model_name="fixture-model"}} 7\n'
        )
    unified, raw = _normalize(text, runtime, runtime_version)

    assert set(unified) == {target_name}
    assert unified[target_name].type == family_type
    assert [
        (sample.name, sample.labels, sample.value)
        for sample in unified[target_name].samples
    ] == _renamed_samples(raw[source_name], source_name, target_name)


@pytest.mark.parametrize(
    'runtime,runtime_version,source_name',
    [
        ('vLLM', '0.32.0', 'vllm:num_requests_running'),
        ('SGLang', '0.5.23', 'sglang:num_running_reqs'),
    ],
)
@pytest.mark.parametrize('value', ['0', 'NaN', '+Inf', '-Inf'])
def test_normalization_preserves_zero_and_non_finite_samples_without_fabrication(
    runtime, runtime_version, source_name, value
):
    text = (
        f'# TYPE {source_name} gauge\n'
        f'{source_name}{{model_name="fixture-model"}} {value}\n'
    )
    unified, raw = _normalize(text, runtime, runtime_version)
    for family in (raw[source_name], unified['gpustack:num_requests_running']):
        assert len(family.samples) == 1
        actual = family.samples[0].value
        if value == 'NaN':
            assert math.isnan(actual)
        else:
            assert actual == float(value)


@pytest.mark.parametrize(
    'runtime,runtime_version', [('vLLM', '0.32.0'), ('SGLang', '0.5.23')]
)
def test_absent_runtime_samples_do_not_become_zero_metrics(runtime, runtime_version):
    unified, raw = _normalize('', runtime, runtime_version)

    assert unified == {}
    assert raw == {}
