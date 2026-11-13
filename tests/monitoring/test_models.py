"""The model monitoring DTO and its finite, source-aware observations."""

import importlib
import importlib.util
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError


def _schemas():
    name = 'gpustack.schemas.model_metrics'
    assert importlib.util.find_spec(name) is not None, 'Monitoring DTO is missing'
    return importlib.import_module(name)


def test_metric_preserves_real_zero_and_missing_points():
    schema = _schemas()
    series = schema.MetricSeries(
        name='Decode',
        statistic='current',
        dimensions=schema.MetricDimensions(role='decode', model_instance_id=4),
        points=[(100, 0), (130, None)],
    )
    assert series.model_dump(mode='json')['points'] == [[100.0, 0.0], [130.0, None]]
    assert series.model_dump(mode='json')['dimensions'] == {
        'role': 'decode',
        'model_instance_id': 4,
    }


def test_metric_rejects_nonfinite_numbers_and_unapproved_dimensions():
    schema = _schemas()
    with pytest.raises(ValidationError):
        schema.MetricSeries(name='bad', statistic='avg', points=[(100, float('inf'))])
    with pytest.raises(ValidationError):
        schema.MetricSummary(statistic='p95', value=float('nan'))
    with pytest.raises(ValidationError):
        schema.MetricDimensions(worker_ip='secret.internal')


def test_histogram_contract_has_four_summary_statistics_and_no_minmax():
    schema = _schemas()
    metric = schema.ModelMetric(
        kind='histogram',
        unit='seconds',
        measurement_scope='engine_local',
        status='no_observations',
        reason='zero_observations',
        supported_statistics=['avg', 'p50', 'p95', 'p99'],
        summaries=[
            schema.MetricSummary(statistic=s, value=None, sample_count=0)
            for s in ('avg', 'p50', 'p95', 'p99')
        ],
    )
    assert [s.statistic for s in metric.summaries] == ['avg', 'p50', 'p95', 'p99']
    with pytest.raises(ValidationError):
        schema.MetricSummary(statistic='max', value=1)


def _collector():
    name = 'gpustack.monitoring.models'
    assert importlib.util.find_spec(name) is not None, 'Model collector is missing'
    return importlib.import_module(name)


@pytest.mark.asyncio
@pytest.mark.parametrize('owned', [True, False])
async def test_session_lifetime_covers_query_preparation_errors(monkeypatch, owned):
    module = _collector()
    monkeypatch.setattr(module, '_prometheus_url', lambda: 'http://metrics.invalid')
    client = SimpleNamespace(close=AsyncMock())
    factory = Mock(return_value=client)
    monkeypatch.setattr(module.aiohttp, 'ClientSession', factory)
    roles = module._roles
    prepared = []

    def prepare(definition, *args):
        prepared.append(definition.key)
        if len(prepared) == 2:
            raise ValueError('Invalid query preparation')
        return roles(definition, *args)

    monkeypatch.setattr(module, '_roles', prepare)
    with pytest.raises(ValueError, match='Invalid query preparation'):
        await module.collect_model_metrics(
            _model(),
            [_instance()],
            window_seconds=3600,
            **({} if owned else {'client': client}),
        )
    assert len(prepared) == 2
    assert client.close.await_count == int(owned)
    assert factory.call_count == int(owned)


def _model(backend='vLLM', pd=False):
    return SimpleNamespace(
        id=9,
        name='deployment',
        cluster_id=1,
        backend=backend,
        roles=(
            [SimpleNamespace(name=r) for r in ('prefill', 'decode', 'router')]
            if pd
            else []
        ),
        disaggregation=(
            SimpleNamespace(mode=SimpleNamespace(value='vllm-nixl')) if pd else None
        ),
    )


def _instance(id=12, role=None):
    return SimpleNamespace(
        id=id,
        model_id=9,
        name=f'instance-{id}',
        role=role,
        worker_id=2,
        worker_ip='10.0.0.2',
        port=8000 + id,
        ports=[8000 + id],
        distributed_servers=None,
    )


def _entry(value, id=12, role='', restart='0', **labels):
    metric = {
        'role': role,
        'model_instance_id': str(id),
        'restart_count': restart,
        'cluster_id': '1',
        'model_id': '9',
        'source_kind': 'legacy',
        'source_name': 'legacy',
        'worker_id': '2',
        **labels,
    }
    return {'metric': metric, 'value': [5000, str(value)]}


def _prometheus(monkeypatch, module, instant, ranges=None):
    requests = []
    monkeypatch.setattr(module, '_prometheus_url', lambda: 'http://metrics.invalid')
    monkeypatch.setattr(module.time, 'time', lambda: 5000.0)

    async def query_instant(client, url, query, at):
        requests.append(('instant', query))
        return instant(query)

    async def query_range(client, url, query, start, end, step):
        requests.append(('range', query))
        return ranges(query) if ranges else []

    monkeypatch.setattr(module, '_query_instant', query_instant)
    monkeypatch.setattr(module, '_query_range', query_range)
    return requests


def _histogram_data(query, counts=10, other_bucket=None):
    if 'gpustack:time_to_first_token_seconds_count{' in query:
        return [_entry(counts)]
    if 'gpustack:time_to_first_token_seconds_sum{' in query:
        return [_entry(1)]
    if 'gpustack:time_to_first_token_seconds_bucket{' in query:
        return [
            _entry(v, le=le)
            for le, v in [('0.1', counts / 2), ('0.2', counts), ('+Inf', counts)]
        ]
    return []


@pytest.mark.asyncio
async def test_histogram_summaries_always_cover_four_full_window_statistics(
    monkeypatch,
):
    module = _collector()
    requests = _prometheus(monkeypatch, module, _histogram_data)
    result = await module.collect_model_metrics(
        _model(),
        [_instance()],
        window_seconds=3600,
        statistics=('avg',),
        section='llm',
        client=object(),
    )
    metric = result.metrics['ttft']
    assert metric.status == 'available'
    assert metric.reason is None
    assert {s.statistic for s in metric.summaries} == {'avg', 'p50', 'p95', 'p99'}
    assert next(s for s in metric.summaries if s.statistic == 'avg').value == 0.1
    assert all(s.sample_count == 10 for s in metric.summaries)
    assert any('[3600s]' in q and 'increase(' in q for _, q in requests)
    assert any('[90s]' in q for kind, q in requests if kind == 'range')
    assert result.history_scope.startswith('current_instances')


@pytest.mark.asyncio
async def test_custom_interval_freezes_every_query_and_summary_window(monkeypatch):
    module = _collector()
    _prometheus(monkeypatch, module, _histogram_data)
    instant_calls, range_calls = [], []

    async def instant(client, url, query, at):
        instant_calls.append((query, at))
        return _histogram_data(query)

    async def ranged(client, url, query, start, end, step):
        range_calls.append((start, end, step))
        if 'time_to_first_token_seconds_' not in query:
            return []
        return [
            {'metric': _entry(1)['metric'], 'values': [[start, '0.1'], [end, '0.2']]}
        ]

    monkeypatch.setattr(module, '_query_instant', instant)
    monkeypatch.setattr(module, '_query_range', ranged)
    result = await module.collect_model_metrics(
        _model(),
        [_instance()],
        window_seconds=3600,
        start=1000,
        end=2000,
        client=object(),
    )
    assert (result.start, result.end) == (1000, 2000)
    assert all(at == 2000 for _, at in instant_calls)
    assert all((start, end) == (1000, 2000) for start, end, _ in range_calls)
    assert any(
        'increase(' in query and '[1000s]' in query for query, _ in instant_calls
    )
    assert not any('[3600s]' in query for query, _ in instant_calls)
    assert [series.statistic for series in result.metrics['ttft'].series] == [
        'avg',
        'p50',
        'p95',
        'p99',
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize('view', ['aggregate', 'instances'])
@pytest.mark.parametrize(
    'statistics',
    [('avg', 'p50', 'p95', 'p99'), ('avg',), ('p50', 'p99')],
)
async def test_selected_latency_trends_share_a_window_and_keep_full_summaries(
    monkeypatch, view, statistics
):
    module = _collector()

    def ranges(query):
        if 'gpustack:time_to_first_token_seconds_' not in query:
            return []
        return [
            {
                'metric': _entry(1)['metric'],
                'values': [[4910, '0.1'], [5000, '0.2']],
            }
        ]

    requests = _prometheus(monkeypatch, module, _histogram_data, ranges)
    result = await module.collect_model_metrics(
        _model(),
        [_instance()],
        window_seconds=3600,
        statistics=statistics,
        view=view,
        client=object(),
    )
    metric = result.metrics['ttft']
    assert [series.statistic for series in metric.series] == list(statistics)
    assert all(series.points == [(4910, 0.1), (5000, 0.2)] for series in metric.series)
    assert all(summary.sample_count == 10 for summary in metric.summaries)
    assert [summary.statistic for summary in metric.summaries] == [
        'avg',
        'p50',
        'p95',
        'p99',
    ]
    queries = [
        query
        for kind, query in requests
        if kind == 'range' and 'time_to_first_token_seconds_' in query
    ]
    assert len(queries) == len(statistics)
    assert all('[90s]' in query for query in queries)
    for statistic, quantile in (('p50', '0.5'), ('p95', '0.95'), ('p99', '0.99')):
        assert any(f'histogram_quantile({quantile},' in query for query in queries) == (
            statistic in statistics
        )


@pytest.mark.asyncio
async def test_valid_histogram_history_with_idle_points_does_not_keep_no_samples_reason(
    monkeypatch,
):
    module = _collector()

    def ranges(query):
        if 'time_to_first_token_seconds' not in query:
            return []
        return [
            {
                'metric': _entry(1)['metric'],
                'values': [[3500, '0.15'], [5000, 'NaN']],
            }
        ]

    _prometheus(monkeypatch, module, _histogram_data, ranges)
    result = await module.collect_model_metrics(
        _model(), [_instance()], window_seconds=3600, client=object()
    )
    metric = result.metrics['ttft']
    assert metric.status == 'available'
    assert metric.reason is None
    assert metric.summaries[0].sample_count == 10
    assert any(value is not None for line in metric.series for _, value in line.points)
    assert all(line.points[-1][1] is None for line in metric.series)


@pytest.mark.asyncio
async def test_incompatible_source_buckets_keep_average_and_hide_aggregate_quantiles(
    monkeypatch,
):
    module = _collector()

    def instant(query):
        data = _histogram_data(query)
        if 'gpustack:time_to_first_token_seconds_' not in query:
            return data
        if '_bucket{' in query:
            return data + [
                _entry(v, id=13, le=le)
                for le, v in [('0.1', 5), ('0.3', 10), ('+Inf', 10)]
            ]
        return data + [_entry(10 if '_count{' in query else 1, id=13)]

    requests = _prometheus(monkeypatch, module, instant)
    result = await module.collect_model_metrics(
        _model(), [_instance(), _instance(13)], window_seconds=3600, client=object()
    )
    metric = result.metrics['ttft']
    assert metric.status == 'available'
    assert metric.supported_statistics == ['avg']
    assert metric.reason == 'incompatible_histogram_buckets'
    assert next(s for s in metric.summaries if s.statistic == 'avg').value == 0.1
    assert all(s.value is None for s in metric.summaries if s.statistic != 'avg')
    assert not any(
        'histogram_quantile' in q and 'time_to_first_token' in q
        for kind, q in requests
        if kind == 'range'
    )


@pytest.mark.asyncio
async def test_histogram_zero_observations_are_null_and_partial_coverage_is_reported(
    monkeypatch,
):
    module = _collector()
    _prometheus(monkeypatch, module, lambda q: _histogram_data(q, counts=0))
    result = await module.collect_model_metrics(
        _model(), [_instance(), _instance(13)], window_seconds=3600, client=object()
    )
    metric = result.metrics['ttft']
    assert metric.status == 'no_observations'
    assert metric.reason == 'zero_observations'
    assert all(s.value is None and s.sample_count == 0 for s in metric.summaries)
    assert metric.coverage.expected_instances == 2
    assert metric.coverage.observed_instances == 1
    assert metric.coverage.partial


@pytest.mark.asyncio
async def test_gauge_zero_is_available_and_nonfinite_points_remain_gaps(monkeypatch):
    module = _collector()

    def instant(query):
        return [_entry(0)] if 'gpustack:num_requests_running{' in query else []

    def ranges(query):
        return [
            {
                'metric': {'worker_ip': 'secret', 'model_instance_id': '12'},
                'values': [[4900, '0'], [4930, 'NaN'], [4960, '+Inf']],
            }
        ]

    _prometheus(monkeypatch, module, instant, ranges)
    result = await module.collect_model_metrics(
        _model(), [_instance()], window_seconds=3600, client=object()
    )
    metric = result.metrics['running_requests']
    assert metric.status == 'available'
    assert metric.summaries[0].value == 0
    assert metric.series[0].points == [(4900.0, 0.0), (4930.0, None), (4960.0, None)]
    assert 'secret' not in metric.model_dump_json()


@pytest.mark.asyncio
async def test_one_metric_failure_does_not_expose_promql_or_blank_other_metrics(
    monkeypatch,
):
    module = _collector()

    def instant(query):
        if 'gpustack:time_to_first_token_seconds_' in query:
            raise ValueError('http://private.invalid/query?token=secret PromQL dump')
        return [_entry(0)] if 'gpustack:num_requests_running{' in query else []

    _prometheus(monkeypatch, module, instant)
    result = await module.collect_model_metrics(
        _model(), [_instance()], window_seconds=3600, client=object()
    )
    assert result.available
    assert result.metrics['ttft'].status == 'query_error'
    assert result.metrics['ttft'].reason == 'query_failed'
    assert result.metrics['running_requests'].status == 'available'
    assert 'private.invalid' not in result.model_dump_json()
    assert 'secret' not in result.model_dump_json()


@pytest.mark.asyncio
async def test_pd_role_applicability_is_not_replaced_by_another_stage(monkeypatch):
    module = _collector()
    requests = _prometheus(monkeypatch, module, lambda q: [])
    result = await module.collect_model_metrics(
        _model('SGLang', pd=True),
        [_instance(role='prefill')],
        window_seconds=3600,
        role='prefill',
        client=object(),
    )
    assert result.metrics['ttft'].status == 'not_applicable'
    assert result.metrics['tpot'].status == 'not_applicable'
    assert result.metrics['prefill'].status == 'unsupported'
    assert result.metrics['prefill'].supported_statistics == []
    assert not any('time_to_first_token_seconds' in q for _, q in requests)
    assert not any('request_stage_latency_seconds' in q for _, q in requests)


@pytest.mark.asyncio
async def test_missing_prometheus_is_a_controlled_no_data_reason(monkeypatch):
    module = _collector()
    monkeypatch.setattr(module, '_prometheus_url', lambda: None)
    result = await module.collect_model_metrics(
        _model(), [_instance()], window_seconds=3600, client=object()
    )
    assert not result.available and result.reason == 'prometheus_unavailable'
    assert result.metrics['ttft'].status == 'no_data'


@pytest.mark.asyncio
async def test_histogram_summary_deduplicates_jobs_but_adds_streaming_cohorts(
    monkeypatch,
):
    module = _collector()

    def instant(query):
        if 'gpustack:time_to_first_token_seconds_' not in query:
            return []
        rows = []
        for job in ('direct', 'duplicate'):
            for streaming, count in [('true', 3), ('false', 2)]:
                if '_count{' in query:
                    rows.append(_entry(count, job=job, is_streaming=streaming))
                elif '_sum{' in query:
                    rows.append(_entry(count * 0.4, job=job, is_streaming=streaming))
                else:
                    rows.extend(
                        _entry(value, le=le, job=job, is_streaming=streaming)
                        for le, value in [('0.1', 0), ('0.5', count), ('+Inf', count)]
                    )
        return rows

    _prometheus(monkeypatch, module, instant)
    result = await module.collect_model_metrics(
        _model(), [_instance()], window_seconds=3600, view='instances', client=object()
    )
    summaries = result.metrics['ttft'].summaries
    assert all(s.sample_count == 5 for s in summaries)
    assert next(s for s in summaries if s.statistic == 'avg').value == 0.4
    assert all(s.dimensions.model_instance_id is None for s in summaries)


@pytest.mark.asyncio
async def test_missing_histogram_sum_does_not_produce_false_zero_average(monkeypatch):
    module = _collector()
    _prometheus(
        monkeypatch,
        module,
        lambda q: (
            []
            if 'gpustack:time_to_first_token_seconds_sum{' in q
            else _histogram_data(q)
        ),
    )
    result = await module.collect_model_metrics(
        _model(), [_instance()], window_seconds=3600, client=object()
    )
    assert (
        next(s for s in result.metrics['ttft'].summaries if s.statistic == 'avg').value
        is None
    )


@pytest.mark.asyncio
async def test_cache_lookup_denominator_without_hits_is_unknown_not_zero(monkeypatch):
    module = _collector()

    def instant(query):
        return [_entry(1)] if 'gpustack:prefix_cache_queries_total{' in query else []

    _prometheus(monkeypatch, module, instant)
    result = await module.collect_model_metrics(
        _model(), [_instance()], window_seconds=3600, client=object()
    )
    metric = result.metrics['prefix_cache_hit_rate']
    assert metric.status == 'no_data'
    assert all(s.value is None for s in metric.summaries)


@pytest.mark.asyncio
async def test_empty_histogram_retains_four_explicit_missing_summary_slots(monkeypatch):
    module = _collector()
    _prometheus(monkeypatch, module, lambda q: [])
    result = await module.collect_model_metrics(
        _model(), [_instance()], window_seconds=3600, client=object()
    )
    summaries = result.metrics['ttft'].summaries
    assert [s.statistic for s in summaries] == ['avg', 'p50', 'p95', 'p99']
    assert all(s.value is None and s.sample_count is None for s in summaries)


@pytest.mark.asyncio
async def test_router_health_resolves_trusted_dp_peer_and_never_returns_unknown_url(
    monkeypatch,
):
    module = _collector()
    router = _instance(14, role='router')
    upstream = _instance(12, role='prefill')

    def instant(query):
        if 'gpustack:pd_router_worker_health{' in query:
            return [
                _entry(0, id=14, role='router', worker='http://10.0.0.2:8012@0'),
                _entry(1, id=14, role='router', worker='http://private.other:99'),
            ]
        return []

    _prometheus(monkeypatch, module, instant)
    result = await module.collect_model_metrics(
        _model('SGLang', pd=True),
        [router],
        upstream_instances=[router, upstream],
        window_seconds=3600,
        section='pd',
        role='router',
        client=object(),
    )
    metric = result.metrics['router_health']
    assert metric.status == 'available'
    assert metric.summaries[0].value == 0
    assert metric.summaries[0].dimensions.upstream_instance_id == 12
    assert 'private.other' not in result.model_dump_json()
    assert 'http://' not in result.model_dump_json()


@pytest.mark.asyncio
async def test_malformed_prometheus_matrix_remains_a_local_metric_error(monkeypatch):
    module = _collector()
    _prometheus(monkeypatch, module, lambda q: [_entry(0)], lambda q: ['invalid-row'])
    result = await module.collect_model_metrics(
        _model(), [_instance()], window_seconds=3600, client=object()
    )
    assert result.metrics['running_requests'].status == 'query_error'
    assert result.metrics['running_requests'].reason == 'query_failed'


@pytest.mark.asyncio
async def test_deadline_marks_incomplete_metric_and_keeps_completed_results(
    monkeypatch,
):
    module = _collector()

    def instant(query):
        if 'gpustack:num_requests_running{' in query:
            return [_entry(0)]
        return _histogram_data(query)

    _prometheus(monkeypatch, module, instant)
    monkeypatch.setattr(module, '_COLLECT_DEADLINE_SECONDS', 0.01)

    async def ranges(client, url, query, start, end, step):
        if 'gpustack:time_to_first_token_seconds' in query:
            await asyncio.sleep(10)
        return []

    monkeypatch.setattr(module, '_query_range', ranges)
    result = await module.collect_model_metrics(
        _model(), [_instance()], window_seconds=3600, client=object()
    )
    assert result.available
    assert result.metrics['running_requests'].status == 'available'
    assert result.metrics['ttft'].status == 'query_error'
    assert result.metrics['ttft'].reason == 'query_failed'


@pytest.mark.asyncio
async def test_selected_window_summary_includes_observed_old_restart_epochs(
    monkeypatch,
):
    module = _collector()

    def instant(query):
        first = _histogram_data(query, counts=5)
        if 'gpustack:time_to_first_token_seconds_' not in query:
            return first
        second = [
            {**row, 'metric': {**row['metric'], 'restart_count': '1'}} for row in first
        ]
        return first + second

    _prometheus(monkeypatch, module, instant)
    instance = _instance()
    instance.restart_count = 2
    result = await module.collect_model_metrics(
        _model(), [instance], window_seconds=3600, client=object()
    )
    summaries = result.metrics['ttft'].summaries
    assert all(s.sample_count == 10 for s in summaries)
    assert next(s for s in summaries if s.statistic == 'avg').value == 0.2


@pytest.mark.asyncio
async def test_kv_current_summaries_keep_each_instance_and_native_rank(monkeypatch):
    module = _collector()

    def instant(query):
        if 'gpustack:kv_cache_usage_ratio{' in query:
            return [
                _entry(0.2, engine='0'),
                _entry(0.8, engine='1'),
                _entry(0.6, id=13, engine='2'),
            ]
        return []

    _prometheus(monkeypatch, module, instant)
    result = await module.collect_model_metrics(
        _model(), [_instance(), _instance(13)], window_seconds=3600, client=object()
    )
    metric = result.metrics['kv_usage']
    assert sorted(s.value for s in metric.summaries) == [20, 60, 80]
    highest = max(metric.summaries, key=lambda s: s.value)
    assert highest.dimensions.model_instance_id == 12
    assert highest.dimensions.model_instance_name == 'instance-12'
    assert highest.dimensions.dp_rank == '1'


@pytest.mark.asyncio
async def test_named_nonpd_roles_keep_their_local_source_and_do_not_become_decode(
    monkeypatch,
):
    module = _collector()
    model = _model()
    model.roles = [SimpleNamespace(name='encoder')]
    instance = _instance(role='encoder')

    def instant(query):
        return (
            [_entry(2, role='encoder')]
            if 'gpustack:num_requests_running{' in query
            else []
        )

    _prometheus(monkeypatch, module, instant)
    result = await module.collect_model_metrics(
        model, [instance], window_seconds=3600, client=object()
    )
    assert result.metrics['running_requests'].status == 'available'
    assert result.metrics['running_requests'].summaries[0].dimensions.role == 'encoder'
    assert result.metrics['running_requests'].summaries[0].value == 2


@pytest.mark.asyncio
async def test_unrecognized_backend_without_instances_returns_readable_missing_state(
    monkeypatch,
):
    module = _collector()
    requests = _prometheus(monkeypatch, module, lambda q: [])
    result = await module.collect_model_metrics(
        _model('non-llm-backend'), [], window_seconds=3600, client=object()
    )
    assert result.metrics['ttft'].status == 'no_data'
    assert result.metrics['ttft'].reason == 'no_samples'
    assert requests == []


@pytest.mark.asyncio
async def test_mindie_uses_its_mapped_cache_ratio_gauge_without_counter_fallback(
    monkeypatch,
):
    module = _collector()

    def instant(query):
        return [_entry(0.25)] if 'gpustack:prefix_cache_hit_rate{' in query else []

    _prometheus(monkeypatch, module, instant)
    result = await module.collect_model_metrics(
        _model('MindIE'), [_instance()], window_seconds=3600, client=object()
    )
    metric = result.metrics['prefix_cache_hit_rate']
    assert metric.kind == 'gauge' and metric.unit == 'percent'
    assert metric.summaries[0].value == 25


@pytest.mark.asyncio
async def test_incompatible_prefill_buckets_do_not_hide_decode_quantiles(monkeypatch):
    module = _collector()

    def instant(query):
        if 'gpustack:time_to_first_token_seconds_' not in query:
            return []
        if '_count{' in query or '_sum{' in query:
            value = 10 if '_count{' in query else 1
            return [
                _entry(value, id=id, role=role)
                for id, role in [(12, 'prefill'), (13, 'prefill'), (14, 'decode')]
            ]
        return [
            _entry(value, id=id, role=role, le=le)
            for id, role, upper in [
                (12, 'prefill', '0.2'),
                (13, 'prefill', '0.3'),
                (14, 'decode', '0.2'),
            ]
            for le, value in [('0.1', 5), (upper, 10), ('+Inf', 10)]
        ]

    requests = _prometheus(monkeypatch, module, instant)
    result = await module.collect_model_metrics(
        _model(pd=True),
        [_instance(12, 'prefill'), _instance(13, 'prefill'), _instance(14, 'decode')],
        window_seconds=3600,
        client=object(),
    )
    metric = result.metrics['ttft']
    p95 = {s.dimensions.role: s.value for s in metric.summaries if s.statistic == 'p95'}
    assert p95['prefill'] is None
    assert p95['decode'] == pytest.approx(0.19)
    assert 'p95' in metric.supported_statistics
    quantiles = [
        q
        for kind, q in requests
        if kind == 'range' and 'time_to_first_token' in q and 'histogram_quantile' in q
    ]
    assert quantiles and all('role=~"decode"' in q for q in quantiles)


@pytest.mark.asyncio
@pytest.mark.parametrize('partial_id', [12, 13])
@pytest.mark.parametrize('missing_side', ['hits', 'queries'])
async def test_partial_cache_hit_cohort_is_not_aggregated_as_zero(
    monkeypatch, partial_id, missing_side
):
    module = _collector()

    def instant(query):
        if 'gpustack:prefix_cache_hits_total{' in query:
            return [_entry(5, engine='0')] + (
                [_entry(5, id=partial_id, engine='1')]
                if missing_side == 'queries'
                else []
            )
        if 'gpustack:prefix_cache_queries_total{' in query:
            return [_entry(10, engine='0')] + (
                [_entry(10, id=partial_id, engine='1')]
                if missing_side == 'hits'
                else []
            )
        return []

    _prometheus(monkeypatch, module, instant)
    result = await module.collect_model_metrics(
        _model(), [_instance(), _instance(13)], window_seconds=3600, client=object()
    )
    metric = result.metrics['prefix_cache_hit_rate']
    assert metric.summaries[0].value is None
    assert metric.reason == 'partial_source_observations'
    assert metric.coverage.partial


@pytest.mark.asyncio
async def test_partial_ratio_population_does_not_hide_complete_pd_role(monkeypatch):
    module = _collector()

    def instant(query):
        if 'gpustack:prefix_cache_hits_total{' in query:
            return [
                _entry(5, role='prefill', engine='0'),
                _entry(5, role='prefill', engine='1'),
                _entry(2, id=13, role='decode', engine='0'),
            ]
        if 'gpustack:prefix_cache_queries_total{' in query:
            return [
                _entry(10, role='prefill', engine='0'),
                _entry(10, id=13, role='decode', engine='0'),
            ]
        return []

    _prometheus(monkeypatch, module, instant)
    result = await module.collect_model_metrics(
        _model(pd=True),
        [_instance(12, 'prefill'), _instance(13, 'decode')],
        window_seconds=3600,
        client=object(),
    )
    metric = result.metrics['prefix_cache_hit_rate']
    summaries = {s.dimensions.role: s.value for s in metric.summaries}
    assert summaries == {'prefill': None, 'decode': 20}
    assert metric.status == 'available'
    assert metric.reason == 'partial_source_observations'
    assert metric.coverage.observed_instances == 1 and metric.coverage.partial


@pytest.mark.asyncio
async def test_instance_quantile_trends_survive_incompatible_deployment_summary(
    monkeypatch,
):
    module = _collector()

    def instant(query):
        first = _histogram_data(query)
        if 'gpustack:time_to_first_token_seconds_' not in query:
            return []
        if '_bucket{' in query:
            return first + [
                _entry(value, id=13, le=le)
                for le, value in [('0.1', 5), ('0.3', 10), ('+Inf', 10)]
            ]
        return first + [_entry(10 if '_count{' in query else 1, id=13)]

    requests = _prometheus(monkeypatch, module, instant)
    result = await module.collect_model_metrics(
        _model(),
        [_instance(), _instance(13)],
        window_seconds=3600,
        view='instances',
        client=object(),
    )
    metric = result.metrics['ttft']
    assert next(s for s in metric.summaries if s.statistic == 'p95').value is None
    assert 'p95' in metric.supported_statistics
    assert any(
        'histogram_quantile(0.95,' in q and 'time_to_first_token' in q
        for kind, q in requests
        if kind == 'range'
    )
