"""Gateway averages retain their own request population and real missing states."""

import pytest

from gpustack.monitoring.catalog import model_metric_definitions


def definition(key='ingress_ttft'):
    return next(d for d in model_metric_definitions('vllm') if d.key == key)


def entry(value, consumer='one', **labels):
    return {'metric': {'ai_consumer': consumer, **labels}, 'value': [5000, str(value)]}


@pytest.mark.asyncio
@pytest.mark.parametrize('key,expected', [('ingress_ttft', 0.05), ('ingress_e2e', 0.1)])
async def test_streaming_ttft_and_all_request_latency_use_their_own_counts(
    key, expected
):
    from gpustack.monitoring.ingress import collect_ingress_metric

    async def read(query, *, ranged=False):
        if ranged:
            return []
        if 'llm_stream_duration_count' in query:
            value = 2
        elif 'llm_duration_count' in query:
            value = 10
        else:
            value = 100 if 'first_token_duration' in query else 1000
        return [entry(value)]

    metric = await collect_ingress_metric(
        definition(key),
        model_id=9,
        instance_ids=(12,),
        read=read,
        window_seconds=3600,
        rate_window_seconds=90,
        statistics=('avg',),
    )
    assert metric.summaries[0].value == expected


@pytest.mark.asyncio
async def test_ingress_uses_weighted_native_sum_count_and_only_exposes_average():
    from gpustack.monitoring.ingress import collect_ingress_metric

    queries = []

    async def read(query, *, ranged=False):
        queries.append(query)
        count = 'stream_duration_count' in query
        if ranged:
            return [
                {'metric': {'ai_consumer': c}, 'values': [[4990, str(v)]]}
                for c, v in [('one', 1 if count else 100), ('two', 3 if count else 900)]
            ]
        return [entry(1 if count else 100), entry(3 if count else 900, 'two')]

    metric = await collect_ingress_metric(
        definition(),
        model_id=9,
        instance_ids=(12, 13),
        read=read,
        window_seconds=3600,
        rate_window_seconds=90,
        statistics=('avg', 'p99'),
    )
    assert metric.status == 'available'
    assert metric.supported_statistics == ['avg']
    assert metric.summaries[0].value == pytest.approx(0.25)
    assert metric.summaries[0].sample_count == 4
    assert metric.series[0].statistic == 'avg'
    assert metric.series[0].points == [(4990, 0.25)]
    assert metric.series[0].dimensions.role is None
    assert all('ai_cluster=~' in q and 'model-9-(12|13)' in q for q in queries)
    assert all(
        'model_instance_id=' not in q and 'gpustack:time_to_first' not in q
        for q in queries
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'count,expected,status', [(0, None, 'no_observations'), (2, 0, 'available')]
)
async def test_zero_requests_is_missing_but_zero_latency_is_observed(
    count, expected, status
):
    from gpustack.monitoring.ingress import collect_ingress_metric

    async def read(query, *, ranged=False):
        if ranged:
            return []
        return [entry(count if 'stream_duration_count' in query else 0)]

    metric = await collect_ingress_metric(
        definition(),
        model_id=9,
        instance_ids=(12,),
        read=read,
        window_seconds=3600,
        rate_window_seconds=90,
        statistics=('avg',),
    )
    assert metric.status == status
    assert metric.summaries[0].value == expected


@pytest.mark.asyncio
async def test_native_gateway_errors_are_not_classified_as_unsupported():
    from gpustack.monitoring.ingress import collect_ingress_metric

    async def read(query, *, ranged=False):
        raise ValueError('query failed')

    with pytest.raises(ValueError):
        await collect_ingress_metric(
            definition(),
            model_id=9,
            instance_ids=(12,),
            read=read,
            window_seconds=3600,
            rate_window_seconds=90,
            statistics=('avg',),
        )


@pytest.mark.parametrize('model_id,ids', [(0, (12,)), (9, ()), (9, ('12.*',))])
def test_gateway_selector_cannot_expand_an_authorized_population(model_id, ids):
    from gpustack.monitoring.ingress import gateway_selector

    with pytest.raises(ValueError):
        gateway_selector(model_id, ids)


@pytest.mark.asyncio
async def test_duplicate_gateway_scrapes_do_not_double_count_and_conflicts_fail():
    from gpustack.monitoring.ingress import collect_ingress_metric

    conflict = False

    async def read(query, *, ranged=False):
        if not ranged:
            return [entry(2 if 'stream_duration_count' in query else 100)]
        value = 2 if 'stream_duration_count' in query else 100
        return [
            {
                'metric': {'job': job, 'ai_consumer': 'one'},
                'values': [[4990, str(value + (conflict and job == 'b'))]],
            }
            for job in ('a', 'b')
        ]

    kwargs = dict(
        model_id=9,
        instance_ids=(12,),
        read=read,
        window_seconds=3600,
        rate_window_seconds=90,
        statistics=('avg',),
    )
    metric = await collect_ingress_metric(definition(), **kwargs)
    assert metric.series[0].points == [(4990, 0.05)]
    conflict = True
    with pytest.raises(ValueError, match='Conflicting'):
        await collect_ingress_metric(definition(), **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize('pd', [False, True])
@pytest.mark.parametrize(
    'context,status',
    [
        ('deployment', 'available'),
        ('gateway', 'available'),
        ('direct', 'not_applicable'),
        ('unknown', 'no_data'),
        ('filtered', 'not_applicable'),
    ],
)
async def test_model_collection_preserves_request_path_and_native_failure_states(
    monkeypatch, pd, context, status
):
    from types import SimpleNamespace
    from gpustack.monitoring import models

    monkeypatch.setattr(models, '_prometheus_url', lambda: 'http://metrics.invalid')
    monkeypatch.setattr(models.time, 'time', lambda: 5000)
    queries = []

    async def instant(client, url, query, at):
        if 'route_upstream_model_consumer_metric_llm_' not in query:
            return []
        queries.append(query)
        return [entry(2 if '_count{' in query else 100)]

    async def ranged(*args):
        return []

    monkeypatch.setattr(models, '_query_instant', instant)
    monkeypatch.setattr(models, '_query_range', ranged)
    model = SimpleNamespace(
        id=9,
        cluster_id=1,
        backend='vllm',
        roles=[],
        disaggregation=SimpleNamespace(mode='vllm-nixl') if pd else None,
    )
    instance = SimpleNamespace(
        id=12,
        name='member',
        role='router' if pd else '',
        worker_id=3,
        worker_ip='10.0.0.1',
        port=8000,
        ports=[8000],
        distributed_servers=None,
    )
    result = await models.collect_model_metrics(
        model, [instance], window_seconds=3600, client=object(), ingress_context=context
    )
    metric = result.metrics['ingress_ttft']
    assert metric.status == status
    if status == 'available':
        assert metric.summaries[0].value == pytest.approx(0.05)
        assert metric.supported_statistics == ['avg']
        assert queries
    else:
        assert queries == []

    if status == 'available':

        async def failed(*args):
            raise ValueError('query failed')

        monkeypatch.setattr(models, '_query_instant', failed)
        result = await models.collect_model_metrics(
            model,
            [instance],
            window_seconds=3600,
            client=object(),
            ingress_context=context,
        )
        assert result.metrics['ingress_ttft'].status == 'query_error'
