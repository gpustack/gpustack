"""Query declarations drive expressions without changing authorized API semantics."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from gpustack.monitoring import config
from gpustack.monitoring.catalog import model_metric_definitions, HARDWARE_KEYS
from gpustack.monitoring.queries import (
    MetricQueryScope,
    build_gauge_query,
    build_counter_query,
    build_counter_observations,
    build_ratio_query,
    build_histogram_query,
    build_histogram_components,
)


def _document():
    return deepcopy(config.load_query_config().model_dump(mode='json'))


@pytest.mark.parametrize('view', ['aggregate', 'instances'])
def test_packaged_expressions_preserve_all_migration_queries(view):
    fixture = (
        Path(__file__).parents[1] / 'fixtures/monitoring_queries/query_parity.json'
    )
    expected = next(
        row['expressions']
        for row in json.loads(fixture.read_text())
        if row['view'] == view
    )
    scope = MetricQueryScope(1, 9, (12, 13), ('prefill', 'decode'), view)
    actual = {
        'gauge': build_gauge_query('gpustack:num_requests_running', scope),
        'gauge_max': build_gauge_query(
            'gpustack:pd_router_worker_health',
            scope,
            aggregate='max',
            group_by=('role', 'model_instance_id', 'worker'),
        ),
        'counter': build_counter_query('gpustack:generation_tokens_total', scope, 90),
        'counter_summary': build_counter_query(
            'gpustack:generation_tokens_total', scope, 3600, summary=True
        ),
        'counter_observations': build_counter_observations(
            'gpustack:prefix_cache_hits_total', scope, 3600, summary=True
        ),
        'ratio': build_ratio_query(
            'gpustack:prefix_cache_hits_total',
            'gpustack:prefix_cache_queries_total',
            scope,
            90,
        ),
        **{
            f'histogram_{s}': build_histogram_query(
                'gpustack:time_to_first_token_seconds', scope, s, 90
            )
            for s in ('avg', 'p50', 'p95', 'p99')
        },
        **{
            f'histogram_summary_{k}': v
            for k, v in build_histogram_components(
                'gpustack:time_to_first_token_seconds', scope, 3600
            ).items()
        },
    }
    assert actual == expected


def test_metric_profiles_and_chart_sources_are_resolved_from_yaml():
    definitions = {m.key: m for m in model_metric_definitions('sglang')}
    assert definitions['prefix_cache_hit_rate'].kind == 'gauge'
    assert definitions['prefix_cache_hit_rate'].query_profile == 'gauge'
    assert definitions['prefix_cache_hit_rate'].denominator is None
    assert definitions['pd_kv_transfer'].measurement_scope == 'bootstrap_handshake'
    assert definitions['pd_kv_transfer'].section == 'pd'
    assert 'filesystem_usage' not in HARDWARE_KEYS
    declaration = config.get_query_config()
    assert declaration.charts['model.ttft'].metrics == ('ttft',)
    assert all(m.kind == 'gauge' for m in declaration.hardware_metrics.values())
    assert declaration.metrics['ttft'].query_profile == 'histogram'
    assert (
        declaration.profiles['histogram']
        .expressions['p95']
        .startswith('histogram_quantile(0.95,')
    )


def test_changing_only_a_declared_expression_changes_the_query_and_keeps_scope(
    monkeypatch,
):
    document = _document()
    document['profiles']['counter']['expressions'][
        'rate'
    ] = 'irate(${family}${selector}[${window_seconds}s])'
    declaration = config.MonitoringQueries.model_validate(document)
    monkeypatch.setattr(config, 'get_query_config', lambda: declaration)
    query = build_counter_query(
        'gpustack:generation_tokens_total',
        MetricQueryScope(1, 9, (12,), ('decode',)),
        90,
    )
    assert 'irate(gpustack:generation_tokens_total{' in query
    assert 'model_instance_id=~"12"' in query
    assert 'gpustack:runtime_metrics_scrape_status' in query
    assert 'gpustack:model_instance_source_info' in query


def test_metric_can_select_a_specialized_profile_without_changing_other_metrics(
    monkeypatch,
):
    document = _document()
    document['profiles']['custom_histogram'] = deepcopy(
        document['profiles']['histogram']
    )
    document['profiles']['custom_histogram']['expressions'][
        'p95'
    ] = 'clamp_min(histogram_quantile(0.95, ${buckets}), 0)'
    document['metrics']['ttft']['query_profile'] = 'custom_histogram'
    declaration = config.MonitoringQueries.model_validate(document)
    monkeypatch.setattr(config, 'get_query_config', lambda: declaration)
    definitions = {m.key: m for m in model_metric_definitions('vllm')}
    scope = MetricQueryScope(1, 9, (12,), ('',))
    ttft = build_histogram_query(
        definitions['ttft'].family,
        scope,
        'p95',
        90,
        profile=definitions['ttft'].query_profile,
    )
    tpot = build_histogram_query(
        definitions['tpot'].family,
        scope,
        'p95',
        90,
        profile=definitions['tpot'].query_profile,
    )
    assert ttft.startswith('clamp_min(')
    assert tpot.startswith('histogram_quantile(')


@pytest.mark.parametrize(
    'seconds,step,window',
    [
        (300, 15, 75),
        (3600, 30, 90),
        (21600, 180, 240),
        (86400, 720, 780),
        (604800, 5040, 5100),
    ],
)
def test_declared_default_windows_match_the_api(seconds, step, window):
    policy = config.get_query_config().time_windows.llm
    assert policy.step(seconds) == step
    assert policy.rate_window(step) == window


@pytest.mark.parametrize(
    'section,key,value',
    [
        ('expressions', 'aggregate', '${expression}'),
        ('expressions', 'engine_guard', '${expression} ${selector}'),
        ('expressions', 'divide', '${numerator} / ${unknown}'),
        ('expressions', 'pair', '${left.__class__} ${right}'),
    ],
)
def test_invalid_or_missing_template_variables_are_rejected(section, key, value):
    document = _document()
    document[section][key] = value
    with pytest.raises(ValidationError):
        config.MonitoringQueries.model_validate(document)


@pytest.mark.parametrize(
    'mutation',
    [
        'version',
        'bool_version',
        'missing_profile',
        'kind_mismatch',
        'hardware_missing_kind',
        'hardware_invalid_kind',
        'hardware_profile_mismatch',
        'unknown_chart_metric',
        'chart_api',
        'filesystem_api',
        'scope_removed',
        'bad_backend_field',
        'bool_step',
    ],
)
def test_invalid_declarations_fail_closed(mutation):
    document = _document()
    if mutation == 'version':
        document['schema_version'] = 2
    elif mutation == 'bool_version':
        document['schema_version'] = True
    elif mutation == 'missing_profile':
        document['metrics']['ttft']['query_profile'] = 'missing'
    elif mutation == 'kind_mismatch':
        document['metrics']['ttft']['query_profile'] = 'gauge'
    elif mutation == 'hardware_missing_kind':
        del document['hardware_metrics']['gpu_utilization']['kind']
    elif mutation == 'hardware_invalid_kind':
        document['hardware_metrics']['gpu_utilization']['kind'] = 'counter'
    elif mutation == 'hardware_profile_mismatch':
        document['hardware_metrics']['gpu_utilization']['query_profile'] = 'counter'
    elif mutation == 'unknown_chart_metric':
        document['charts']['model.ttft']['metrics'] = ['unregistered']
    elif mutation == 'chart_api':
        document['charts']['model.ttft']['apis'] = ['worker_resource']
    elif mutation == 'filesystem_api':
        document['charts']['worker.filesystem']['apis'] = ['gpu_resource']
    elif mutation == 'scope_removed':
        document['profiles']['counter']['expressions'][
            'rate'
        ] = 'rate(${family}[${window_seconds}s])'
    elif mutation == 'bad_backend_field':
        document['metrics']['ttft']['backend_overrides'] = {
            'sglang': {'unexpected': True}
        }
    elif mutation == 'bool_step':
        document['time_windows']['llm']['min_step_seconds'] = True
    with pytest.raises(ValidationError):
        config.MonitoringQueries.model_validate(document)


def test_duplicate_yaml_keys_are_rejected(tmp_path):
    path = tmp_path / 'queries.yaml'
    path.write_text('schema_version: 1\nschema_version: 1\n')
    with pytest.raises(ValueError, match='Duplicate query configuration key'):
        config.load_query_config(path)


def test_default_profile_used_by_source_identity_checks_cannot_disappear():
    document = _document()
    document['profiles']['renamed_hardware'] = document['profiles'].pop(
        'hardware_gauge'
    )
    for metric in document['hardware_metrics'].values():
        metric['query_profile'] = 'renamed_hardware'
    with pytest.raises(ValidationError, match='hardware_gauge'):
        config.MonitoringQueries.model_validate(document)


@pytest.mark.asyncio
async def test_model_collector_consumes_metric_profile_and_time_policy(monkeypatch):
    from tests.monitoring.test_models import (
        _collector,
        _histogram_data,
        _instance,
        _model,
        _prometheus,
    )

    document = _document()
    document['time_windows']['llm']['target_points'] = 60
    document['profiles']['custom_histogram'] = deepcopy(
        document['profiles']['histogram']
    )
    document['profiles']['custom_histogram']['expressions'][
        'p95'
    ] = 'clamp_min(histogram_quantile(0.95, ${buckets}), 0)'
    document['metrics']['ttft']['query_profile'] = 'custom_histogram'
    declaration = config.MonitoringQueries.model_validate(document)
    monkeypatch.setattr(config, 'get_query_config', lambda: declaration)
    module = _collector()
    requests = _prometheus(monkeypatch, module, _histogram_data)
    response = await module.collect_model_metrics(
        _model(), [_instance()], window_seconds=3600, client=object()
    )
    assert (response.step, response.rate_window_seconds) == (60, 120)
    assert response.metrics['ttft'].status == 'available'
    assert any(
        query.startswith('clamp_min(') and '[120s]' in query
        for kind, query in requests
        if kind == 'range'
    )
    assert any(
        'increase(' in query and '[3600s]' in query
        for kind, query in requests
        if kind == 'instant'
    )


@pytest.mark.asyncio
async def test_direct_resource_collector_consumes_hardware_expression_and_window(
    monkeypatch,
):
    from gpustack.monitoring.resources import (
        collect_resource_metrics,
        inventory_devices,
    )
    from tests.monitoring.test_hardware import _worker, _environment

    document = _document()
    document['time_windows']['hardware']['target_points'] = 120
    document['time_windows']['hardware']['freshness_seconds'] = 90
    declaration = config.MonitoringQueries.model_validate(document)
    monkeypatch.setattr(config, 'get_query_config', lambda: declaration)
    worker = _worker()
    _, _, calls = _environment(monkeypatch, [worker])
    response = await collect_resource_metrics(
        worker, inventory_devices(worker), window_seconds=3600, client=object()
    )
    assert response.step == 30
    assert calls and all('<= 90)' in expression for expression in calls)
