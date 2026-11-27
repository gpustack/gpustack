"""The model monitoring DTO and its finite, source-aware observations."""

import importlib
import importlib.util
import asyncio
import re
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


def _collection(module):
    return module._Collection(
        _model(),
        [_instance()],
        object(),
        'http://metrics.invalid',
        1400,
        5000,
        15,
        90,
        'aggregate',
        ('avg',),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', [asyncio.TimeoutError, asyncio.CancelledError])
async def test_failed_coalesced_read_can_retry(monkeypatch, failure):
    module = _collector()
    collection = _collection(module)
    native = AsyncMock(side_effect=[failure(), []])
    monkeypatch.setattr(collection, 'read_native', native)

    with pytest.raises(failure):
        await collection.read('same query')
    assert await collection.read('same query') == []
    assert native.await_count == 2


@pytest.mark.asyncio
async def test_cancelled_waiter_keeps_shared_read_and_success_cache(monkeypatch):
    module = _collector()
    collection = _collection(module)
    started, release = asyncio.Event(), asyncio.Event()

    async def native(*args, **kwargs):
        started.set()
        await release.wait()
        return []

    native = AsyncMock(side_effect=native)
    monkeypatch.setattr(collection, 'read_native', native)
    first = asyncio.create_task(collection.read('same query'))
    await started.wait()
    second = asyncio.create_task(collection.read('same query'))
    await asyncio.sleep(0)
    first.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await first
        release.set()
        assert await second == []
        assert await collection.read('same query') == []
        assert native.await_count == 1
    finally:
        release.set()
        await asyncio.gather(first, second, return_exceptions=True)


@pytest.mark.asyncio
async def test_last_cancelled_waiter_drains_read_before_retry(monkeypatch):
    module = _collector()
    collection = _collection(module)
    started, finalized = asyncio.Event(), asyncio.Event()

    async def native(*args, **kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            finalized.set()

    monkeypatch.setattr(collection, 'read_native', native)
    waiter = asyncio.create_task(collection.read('same query'))
    await started.wait()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert finalized.is_set()
    replacement = AsyncMock(return_value=[])
    monkeypatch.setattr(collection, 'read_native', replacement)
    assert await collection.read('same query') == []
    replacement.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize('owned', [True, False])
@pytest.mark.parametrize('outcome', ['failure', 'unexpected', 'cancelled', 'deadline'])
async def test_collection_drains_reads_before_releasing_client(
    monkeypatch,
    owned,
    outcome,
):
    module = _collector()
    monkeypatch.setattr(module, '_prometheus_url', lambda: 'http://metrics.invalid')
    definition = next(
        item
        for item in module.model_metric_definitions('vllm')
        if item.key == 'running_requests'
    )
    monkeypatch.setattr(module, 'model_metric_definitions', lambda _: [definition])
    started, finalized = asyncio.Event(), asyncio.Event()
    native_tasks, read_tasks = set(), set()
    read = module._Collection.read

    async def track_read(self, *args, **kwargs):
        read_tasks.add(asyncio.current_task())
        return await read(self, *args, **kwargs)

    monkeypatch.setattr(module._Collection, 'read', track_read)

    async def instant(*args):
        if outcome not in ('failure', 'unexpected'):
            return []
        await started.wait()
        if outcome == 'unexpected':
            raise RuntimeError('Unexpected query failure')
        raise asyncio.TimeoutError('Summary query failed')

    async def ranged(*args):
        native_tasks.add(asyncio.current_task())
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            finalized.set()

    closed_after_drain = []

    async def close():
        closed_after_drain.append(
            finalized.is_set()
            and all(task.done() for task in native_tasks | read_tasks)
        )

    client = SimpleNamespace(close=AsyncMock(side_effect=close))
    monkeypatch.setattr(module.aiohttp, 'ClientSession', lambda: client)
    monkeypatch.setattr(module, '_query_instant', instant)
    monkeypatch.setattr(module, '_query_range', ranged)
    if outcome == 'deadline':
        monkeypatch.setattr(module, '_COLLECT_DEADLINE_SECONDS', 0.01)
    collector = asyncio.create_task(
        module.collect_model_metrics(
            _model(),
            [_instance()],
            window_seconds=3600,
            **({} if owned else {'client': client}),
        )
    )
    try:
        await started.wait()
        if outcome == 'cancelled':
            collector.cancel()
            with pytest.raises(asyncio.CancelledError):
                await collector
        elif outcome == 'unexpected':
            with pytest.raises(RuntimeError, match='Unexpected query failure'):
                await collector
        else:
            result = await collector
            assert result.metrics['running_requests'].status == 'query_error'
        assert finalized.is_set()
        assert all(task.done() for task in native_tasks | read_tasks)
        assert closed_after_drain == ([True] if owned else [])
        assert client.close.await_count == int(owned)
    finally:
        for task in native_tasks | read_tasks | {collector}:
            if not task.done():
                task.cancel()
        await asyncio.gather(
            *native_tasks, *read_tasks, collector, return_exceptions=True
        )


@pytest.mark.asyncio
@pytest.mark.parametrize('owned', [True, False])
@pytest.mark.parametrize('cancellations', [1, 3])
@pytest.mark.parametrize('initial', ['failure', 'cancelled', 'native_cancelled'])
async def test_cancellation_during_cleanup_waits_for_every_owned_task(
    monkeypatch,
    owned,
    cancellations,
    initial,
):
    module = _collector()
    definitions = [
        definition
        for definition in module.model_metric_definitions('vllm')
        if definition.key in ('running_requests', 'waiting_requests')
    ]
    monkeypatch.setattr(module, 'model_metric_definitions', lambda _: definitions)
    monkeypatch.setattr(module, '_prometheus_url', lambda: 'http://metrics.invalid')
    started = {name: asyncio.Event() for name in ('running', 'waiting')}
    cleaning, release_native = asyncio.Event(), asyncio.Event()
    client_closing, release_client = asyncio.Event(), asyncio.Event()
    native_tasks, finalized = {}, set()
    closed_after_drain = []

    async def instant(client, url, query, at):
        if 'num_requests_running' in query:
            await started['running'].wait()
            await started['waiting'].wait()
            if initial == 'failure':
                raise RuntimeError('Unexpected query failure')
            if initial == 'native_cancelled':
                raise asyncio.CancelledError
        return []

    async def ranged(client, url, query, *args):
        name = 'running' if 'num_requests_running' in query else 'waiting'
        native_tasks[name] = asyncio.current_task()
        started[name].set()
        try:
            await asyncio.Event().wait()
        finally:
            if name == 'waiting':
                cleaning.set()
            await release_native.wait()
            finalized.add(name)

    async def close_client():
        closed_after_drain.append(
            finalized == {'running', 'waiting'}
            and all(task.done() for task in native_tasks.values())
        )
        client_closing.set()
        await release_client.wait()

    client = SimpleNamespace(close=AsyncMock(side_effect=close_client))
    monkeypatch.setattr(module.aiohttp, 'ClientSession', lambda: client)
    monkeypatch.setattr(module, '_query_instant', instant)
    monkeypatch.setattr(module, '_query_range', ranged)
    existing_tasks = asyncio.all_tasks()
    collector = asyncio.create_task(
        module.collect_model_metrics(
            _model(),
            [_instance()],
            window_seconds=3600,
            **({} if owned else {'client': client}),
        )
    )
    try:
        if initial == 'cancelled':
            await started['running'].wait()
            await started['waiting'].wait()
            collector.cancel()
        await asyncio.wait_for(cleaning.wait(), 1)
        for _ in range(cancellations):
            collector.cancel()
            await asyncio.sleep(0)
        assert not collector.done()
        assert client.close.await_count == 0
        release_native.set()
        if owned:
            await asyncio.wait_for(client_closing.wait(), 1)
            collector.cancel()
            await asyncio.sleep(0)
            assert not collector.done()
            release_client.set()
        with pytest.raises(asyncio.CancelledError):
            await collector
        assert finalized == {'running', 'waiting'}
        assert client.close.await_count == int(owned)
        assert closed_after_drain == ([True] if owned else [])
        assert not asyncio.all_tasks() - existing_tasks
    finally:
        release_native.set()
        release_client.set()
        leftovers = asyncio.all_tasks() - existing_tasks
        for task in leftovers:
            task.cancel()
        await asyncio.gather(collector, *leftovers, return_exceptions=True)


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
        rows = ranges(query) if ranges else []
        if 'histogram_upper_bound' in query and not any(
            'histogram_upper_bound' in row.get('metric', {}) for row in rows
        ):
            # These fixtures describe exact quantiles without overflow at the edge.
            bound = re.search(r'"histogram_upper_bound", "([^"]+)"', query).group(1)
            return [
                {
                    **row,
                    'metric': {**row['metric'], 'histogram_upper_bound': bound},
                    'values': [[at, '1'] for at, _ in row['values']],
                }
                for row in rows
            ]
        return rows

    monkeypatch.setattr(module, '_query_instant', query_instant)
    monkeypatch.setattr(module, '_query_range', query_range)
    return requests


def _histogram_data(query, counts=10, bounds=('0.1', '0.2')):
    if 'gpustack:time_to_first_token_seconds_count{' in query:
        return [_entry(counts)]
    if 'gpustack:time_to_first_token_seconds_sum{' in query:
        return [_entry(1)]
    if 'gpustack:time_to_first_token_seconds_bucket{' in query:
        return [
            _entry(v, le=le)
            for le, v in [
                (bounds[0], counts / 2),
                (bounds[1], counts),
                ('+Inf', counts),
            ]
        ]
    return []


def _histogram_range_component(part, labels, *values):
    return {
        'metric': {**labels, 'histogram_component': part},
        'values': list(values),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize('view', ['aggregate', 'instances'])
@pytest.mark.parametrize(
    ('zero_bucket', 'all_zero'),
    [(None, False), (0, False), (10, True), (None, True)],
    ids=['absent', 'empty', 'full', 'vllm-exact-zero'],
)
async def test_first_bucket_tpot_keeps_average_without_fabricated_percentiles(
    monkeypatch, view, zero_bucket, all_zero
):
    module = _collector()
    prefix = 'gpustack:time_per_output_token_seconds'
    average = 0.0 if all_zero else 0.0032
    later_average = 0.0 if all_zero else 0.008
    bounds = (
        '0.01',
        '0.025',
        '0.05',
        '0.075',
        '0.1',
        '0.15',
        '0.2',
        '0.3',
        '0.4',
        '0.5',
        '0.75',
        '1.0',
        '2.5',
        '5.0',
        '7.5',
        '10.0',
        '20.0',
        '40.0',
        '80.0',
        '+Inf',
    )
    buckets = [_entry(10, le=bound) for bound in bounds]
    if zero_bucket is not None:
        buckets.insert(0, _entry(zero_bucket, le='0'))

    def instant(query):
        if prefix + '_count{' in query:
            return [_entry(10)]
        if prefix + '_sum{' in query:
            return [_entry(average * 10)]
        if prefix + '_bucket{' in query:
            return buckets
        return []

    def ranges(query):
        if prefix + '_' not in query:
            return []
        labels = _entry(0)['metric']
        if 'histogram_component' in query:
            later_buckets = {'0.01': '5'} if not all_zero else {}
            return [
                _histogram_range_component('count', labels, (4910, '10'), (5000, '10')),
                _histogram_range_component(
                    'sum', labels, (4910, average * 10), (5000, later_average * 10)
                ),
            ] + [
                _histogram_range_component(
                    'buckets',
                    row['metric'],
                    (4910, row['value'][1]),
                    (5000, later_buckets.get(row['metric']['le'], row['value'][1])),
                )
                for row in buckets
            ]
        for quantile, interpolated, resolved in (
            ('0.5', 0.005, 0.01),
            ('0.95', 0.0095, 0.0235),
            ('0.99', 0.0099, 0.0247),
        ):
            if f'histogram_quantile({quantile},' in query:
                if zero_bucket == 10:
                    interpolated = resolved = 0
                elif all_zero:
                    resolved = interpolated
                return [
                    {
                        'metric': labels,
                        'values': [[4910, str(interpolated)], [5000, str(resolved)]],
                    }
                ]
        return [{'metric': labels, 'values': [[4910, average], [5000, later_average]]}]

    _prometheus(monkeypatch, module, instant, ranges)
    result = await module.collect_model_metrics(
        _model(),
        [_instance()],
        start=1400,
        end=5000,
        window_seconds=3600,
        view=view,
        statistics=('avg', 'p50', 'p95', 'p99'),
        section='llm',
        client=object(),
    )

    metric = result.metrics['tpot']
    summaries = {s.statistic: s.value for s in metric.summaries}
    series = {line.statistic: line.points for line in metric.series}
    if all_zero:
        assert metric.reason is None
        assert summaries == dict.fromkeys(('avg', 'p50', 'p95', 'p99'), 0)
        assert series == dict.fromkeys(summaries, [(4910, 0), (5000, 0)])
        return

    assert metric.reason == 'histogram_first_bucket_unresolved'
    assert summaries == {
        'avg': 0.0032,
        'p50': None,
        'p95': None,
        'p99': None,
    }
    assert series == {
        'avg': [(4910, 0.0032), (5000, 0.008)],
        'p50': [(4910, None), (5000, 0.01)],
        'p95': [(4910, None), (5000, 0.0235)],
        'p99': [(4910, None), (5000, 0.0247)],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize('view', ['aggregate', 'instances'])
async def test_first_bucket_trend_uses_each_points_active_layout(monkeypatch, view):
    module = _collector()
    prefix = 'gpustack:time_to_first_token_seconds'
    labels = _entry(0)['metric']
    old_labels = _entry(0, restart='1')['metric']

    def instant(query):
        if prefix + '_count{' in query:
            return [_entry(100)]
        if prefix + '_sum{' in query:
            return [_entry(1)]
        if prefix + '_bucket{' in query:
            return [_entry(100, le=bound) for bound in ('0.01', '0.02', '+Inf')]
        return []

    def ranges(query):
        if prefix + '_' not in query:
            return []
        if 'histogram_quantile(' in query:
            return [
                {
                    'metric': labels,
                    'values': [[4910, '0.005'], [4925, '0.0025'], [5000, '0.005']],
                }
            ]
        if 'histogram_component' in query:
            return (
                [
                    _histogram_range_component(
                        'count', old_labels, (4910, '100'), (4925, '100')
                    ),
                    _histogram_range_component('count', labels, (5000, '100')),
                ]
                + [
                    _histogram_range_component(
                        'buckets',
                        {**old_labels, 'le': bound},
                        (4910, old),
                        (4925, '100'),
                    )
                    for bound, old in [
                        ('0.005', '50'),
                        ('0.01', '100'),
                        ('+Inf', '100'),
                    ]
                ]
                + [
                    _histogram_range_component(
                        'buckets', {**labels, 'le': bound}, (5000, '100')
                    )
                    for bound in ('0.01', '0.02', '+Inf')
                ]
            )
        return []

    _prometheus(monkeypatch, module, instant, ranges)
    result = await module.collect_model_metrics(
        _model(),
        [_instance()],
        start=1400,
        end=5000,
        window_seconds=3600,
        view=view,
        statistics=('p50',),
        section='llm',
        client=object(),
    )

    assert result.metrics['ttft'].series[0].points == [
        (4910, 0.005),
        (4925, None),
        (5000, None),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize('view', ['aggregate', 'instances'])
@pytest.mark.parametrize(
    ('bucket_state', 'reason'),
    [('missing', 'histogram_bounds_unavailable'), ('failed', 'query_failed')],
)
async def test_first_bucket_trend_reports_incomplete_buckets(
    monkeypatch, view, bucket_state, reason
):
    module = _collector()

    def instant(query):
        if 'time_to_first_token_seconds_sum{' in query:
            return [_entry(0.1)]
        return _histogram_data(query, bounds=('0.01', '0.02'))

    def ranges(query):
        if 'gpustack:time_to_first_token_seconds_' not in query:
            return []
        labels = _entry(0)['metric']
        old_labels = _entry(0, restart='1')['metric']
        if 'histogram_quantile(' in query:
            return [
                {
                    'metric': labels,
                    'values': [[4910, '0.0025'], [4925, '0.0025'], [5000, '0.01']],
                }
            ]
        if 'histogram_component' in query:
            if bucket_state == 'failed':
                raise OSError('Bucket source unavailable')
            return (
                [
                    _histogram_range_component(
                        'count', old_labels, (4910, '10'), (4925, '10')
                    ),
                    _histogram_range_component('count', labels, (5000, '10')),
                ]
                + [
                    _histogram_range_component(
                        'buckets', {**old_labels, 'le': bound}, (4910, '10')
                    )
                    for bound in ('0.005', '0.01', '+Inf')
                ]
                + [
                    _histogram_range_component(
                        'buckets', {**labels, 'le': bound}, (5000, later)
                    )
                    for bound, later in [('0.01', '5'), ('0.02', '10'), ('+Inf', '10')]
                ]
            )
        return [
            {
                'metric': labels,
                'values': [[4910, '0.003'], [4925, '0.004'], [5000, '0.01']],
            }
        ]

    _prometheus(monkeypatch, module, instant, ranges)
    result = await module.collect_model_metrics(
        _model(),
        [_instance()],
        start=1400,
        end=5000,
        window_seconds=3600,
        view=view,
        statistics=('avg', 'p50'),
        section='llm',
        client=object(),
    )

    metric = result.metrics['ttft']
    assert metric.status == 'available'
    assert metric.reason == reason
    series = {line.statistic: line.points for line in metric.series}
    assert series['avg'] == [(4910, 0.003), (4925, 0.004), (5000, 0.01)]
    verified = None if bucket_state == 'missing' else 0.0025
    assert series['p50'] == [(4910, verified), (4925, 0.0025), (5000, 0.01)]


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
    # One extra CDF verifies all quantiles reaching the shared finite edge.
    assert len(queries) == len(statistics) + (0 if statistics == ('avg',) else 1)
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
        labels = _entry(1)['metric']
        if 'histogram_component' in query:
            return [
                _histogram_range_component('count', labels, (3500, '10'), (5000, '0')),
            ] + [
                _histogram_range_component(
                    'buckets', {**labels, 'le': bound}, (3500, count), (5000, '0')
                )
                for bound, count in [('0.1', '5'), ('0.2', '10'), ('+Inf', '10')]
            ]
        return [
            {
                'metric': labels,
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
async def test_short_model_interval_does_not_enable_benchmark_trend_context(
    monkeypatch,
):
    from gpustack.monitoring.time_range import MonitoringInterval

    module = _collector()
    requests = _prometheus(
        monkeypatch,
        module,
        lambda query: _histogram_data(query) if '[75s]' in query else [],
    )
    result = await module.collect_model_metrics(
        _model(),
        [_instance()],
        window_seconds=20,
        interval=MonitoringInterval(1000.25, 1019.5954151, 20),
        client=object(),
    )

    assert not result.metrics['ttft'].series
    assert all(summary.value is None for summary in result.metrics['ttft'].summaries)
    assert not any(kind == 'instant' and '[75s]' in query for kind, query in requests)


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
@pytest.mark.parametrize('sample', [{}, {'value': [5000]}, {'value': '12'}])
async def test_malformed_gateway_instant_remains_a_local_metric_error(
    monkeypatch, sample
):
    module = _collector()

    def instant(query):
        labels = {'ai_consumer': 'consumer'}
        if 'llm_first_token_duration' in query or 'llm_stream_duration_count' in query:
            return [{'metric': labels, **sample}]
        if 'llm_service_duration' in query:
            return [{'metric': labels, 'value': [5000, '100']}]
        if 'llm_duration_count' in query:
            return [{'metric': labels, 'value': [5000, '2']}]
        return []

    _prometheus(monkeypatch, module, instant)
    result = await module.collect_model_metrics(
        _model(), [_instance()], window_seconds=3600, client=object()
    )

    assert result.metrics['ingress_ttft'].status == 'query_error'
    assert result.metrics['ingress_ttft'].reason == 'query_failed'
    assert result.metrics['ingress_e2e'].status == 'available'
    assert result.metrics['ingress_e2e'].summaries[0].value == pytest.approx(0.05)


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
    monkeypatch.setattr(module, '_COLLECT_DEADLINE_SECONDS', 0.05)

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


@pytest.mark.parametrize(
    'key,expected',
    [
        ('prefill', 'prefill'),
        ('decode', 'decode'),
        ('tpot', 'decode'),
        ('itl', 'decode'),
        ('input_tokens', 'prefill'),
        ('output_tokens', 'decode'),
        ('router_requests', 'router'),
        ('router_request_duration', 'router'),
    ],
)
def test_pd_stage_and_router_metrics_have_one_native_owner(key, expected):
    module = _collector()
    definition = next(
        d for d in module.model_metric_definitions('vllm') if d.key == key
    )
    assert module._roles(definition, _model(pd=True), 'vllm', 'all') == (
        (expected,),
        None,
        None,
    )
    other = 'prefill' if expected != 'prefill' else 'decode'
    assert module._roles(definition, _model(pd=True), 'vllm', other) == (
        (),
        'not_applicable',
        'role_not_applicable',
    )


@pytest.mark.parametrize(
    'mode,backend,key,owner,status',
    [
        ('vllm-nixl', 'vllm', 'pd_kv_transfer', 'decode', None),
        ('vllm-nixl', 'vllm', 'pd_kv_transfer_post', 'decode', None),
        ('vllm-nixl', 'vllm', 'pd_kv_transfer_bytes', 'decode', None),
        ('vllm-nixl', 'vllm', 'pd_kv_transfer_speed', None, 'unsupported'),
        ('vllm-mooncake', 'vllm', 'pd_kv_transfer', 'prefill', None),
        ('vllm-mooncake', 'vllm', 'pd_kv_transfer_bytes', 'prefill', None),
        ('vllm-mooncake', 'vllm', 'pd_kv_transfer_post', None, 'unsupported'),
        ('vllm-ascend-mooncake', 'vllm', 'pd_kv_transfer', None, 'unsupported'),
        ('vllm-ascend-mooncake', 'vllm', 'pd_kv_transfer_bytes', None, 'unsupported'),
        ('sglang-mooncake', 'sglang', 'pd_kv_transfer', 'decode', None),
        ('sglang-mooncake', 'sglang', 'pd_kv_allocation', 'decode', None),
        ('sglang-mooncake', 'sglang', 'pd_kv_transfer_bytes', 'prefill', None),
        ('sglang-mooncake', 'sglang', 'pd_kv_transfer_speed', 'prefill', None),
        ('sglang-nixl', 'sglang', 'pd_kv_transfer_post', None, 'unsupported'),
        ('sglang-nixl', 'sglang', 'pd_prefill_bootstrap_queue', 'prefill', None),
        ('sglang-nixl', 'sglang', 'pd_prefill_inflight_queue', 'prefill', None),
        ('sglang-nixl', 'sglang', 'pd_decode_prealloc_queue', 'decode', None),
        ('sglang-nixl', 'sglang', 'pd_decode_transfer_queue', 'decode', None),
    ],
)
def test_pd_connector_ownership_uses_the_recorded_recipe(
    mode, backend, key, owner, status
):
    module = _collector()
    model = _model(backend, pd=True)
    model.disaggregation.mode.value = mode
    definition = next(
        d for d in module.model_metric_definitions(backend) if d.key == key
    )
    roles, actual_status, reason = module._roles(definition, model, backend, 'all')
    assert roles == (() if owner is None else (owner,))
    assert actual_status == status
    assert reason == ('metric_not_provided' if status == 'unsupported' else None)


@pytest.mark.asyncio
async def test_pd_missing_ingress_queries_native_gateway_without_engine_fallback(
    monkeypatch,
):
    module = _collector()
    requests = _prometheus(monkeypatch, module, lambda query: [])
    response = await module.collect_model_metrics(
        _model(pd=True),
        [_instance(12, 'prefill'), _instance(13, 'decode'), _instance(14, 'router')],
        window_seconds=3600,
        client=object(),
    )
    for key in ('ingress_ttft', 'ingress_e2e'):
        metric = response.metrics[key]
        assert metric.status == 'no_data'
        assert metric.reason == 'no_samples'
        assert metric.measurement_scope == 'service_ingress'
        assert metric.series == [] and metric.summaries[0].value is None
        assert metric.supported_statistics == ['avg']
    assert not any('gpustack:service_' in query for _, query in requests)
    assert any(
        'route_upstream_model_consumer_metric_llm_' in query for _, query in requests
    )
    for key, family, role, identity in (
        ('prefill', 'request_prefill_time_seconds', 'prefill', 12),
        ('decode', 'request_decode_time_seconds', 'decode', 13),
        ('router_requests', 'pd_router_requests_total', 'router', 14),
    ):
        selected = [q for _, q in requests if 'gpustack:' + family in q]
        assert selected, key
        assert all(f'role=~"{role}"' in q for q in selected)
        assert all(f'model_instance_id=~"{identity}"' in q for q in selected)


@pytest.mark.asyncio
async def test_sglang_optional_sender_histograms_stay_missing_without_samples(
    monkeypatch,
):
    module = _collector()
    requests = _prometheus(monkeypatch, module, lambda query: [])
    model = _model('SGLang', pd=True)
    model.disaggregation.mode.value = 'sglang-mooncake'
    response = await module.collect_model_metrics(
        model,
        [_instance(12, 'prefill'), _instance(13, 'decode'), _instance(14, 'router')],
        window_seconds=3600,
        client=object(),
        section='pd',
    )
    for key, family in (
        ('pd_kv_transfer_bytes', 'pd_kv_transfer_bytes'),
        ('pd_kv_transfer_speed', 'pd_kv_transfer_bandwidth_bytes_per_second'),
    ):
        metric = response.metrics[key]
        assert metric.status == 'no_data'
        assert all(summary.value is None for summary in metric.summaries)
        assert not metric.series
        queries = [q for _, q in requests if 'gpustack:' + family in q]
        assert queries and all('role=~"prefill"' in query for query in queries)
    from gpustack.schemas.benchmark_monitoring import BenchmarkModelMetrics

    historical = BenchmarkModelMetrics.model_validate(response.model_dump())
    assert (
        historical.metrics['pd_kv_transfer'].measurement_scope == 'bootstrap_handshake'
    )
    assert historical.metrics['pd_kv_transfer_post'].status == 'unsupported'
    assert (
        historical.metrics['pd_kv_allocation'].measurement_scope
        == 'receive_allocation_duration'
    )


@pytest.mark.parametrize('backend', ['vllm', 'sglang'])
def test_pd_only_metrics_do_not_query_ordinary_deployments(backend):
    module = _collector()
    for definition in module.model_metric_definitions(backend):
        if (
            definition.section == 'pd'
            and definition.measurement_scope != 'service_ingress'
        ):
            assert module._roles(definition, _model(backend), backend, 'all') == (
                (),
                'not_applicable',
                'role_not_applicable',
            )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'backend,mode', [('vllm', 'vllm-nixl'), ('sglang', 'sglang-mooncake')]
)
async def test_complete_pd_query_fanout_remains_bounded(monkeypatch, backend, mode):
    import re

    module = _collector()
    model = _model(backend, pd=True)
    model.disaggregation.mode.value = mode
    active, peak, requests = 0, 0, []

    async def read(query, ranged=False):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        requests.append(query)
        try:
            # Yield across all metric requests to exercise the shared semaphore.
            await asyncio.sleep(0)
            if 'route_upstream_model_consumer_metric_llm_' in query:
                return []
            roles = re.search(r'role=~"([^"]*)"', query)[1].split('|')
            identities = {'prefill': 12, 'decode': 13, 'router': 14}
            if ranged:
                return [
                    {
                        **_entry(1, identities[role], role),
                        'values': [[4500, '1'], [5000, '1']],
                    }
                    for role in roles
                ]
            part = re.search(r'gpustack:[a-z0-9_]+_(count|sum|bucket)\{', query)
            if part and part[1] == 'bucket':
                return [
                    _entry(value, identities[role], role, le=bound)
                    for role in roles
                    for bound, value in [('0.1', 5), ('0.2', 10), ('+Inf', 10)]
                ]
            return [
                _entry(10 if part and part[1] == 'count' else 1, identities[role], role)
                for role in roles
            ]
        finally:
            active -= 1

    async def instant(client, url, query, at):
        return await read(query)

    async def ranged(client, url, query, start, end, step):
        return await read(query, ranged=True)

    monkeypatch.setattr(module, '_prometheus_url', lambda: 'http://metrics.invalid')
    monkeypatch.setattr(module, '_query_instant', instant)
    monkeypatch.setattr(module, '_query_range', ranged)
    response = await module.collect_model_metrics(
        model,
        [_instance(12, 'prefill'), _instance(13, 'decode'), _instance(14, 'router')],
        window_seconds=3600,
        client=object(),
    )
    assert 1 < peak <= 8
    assert len(requests) <= 128
    assert not any(
        metric.status == 'query_error' for metric in response.metrics.values()
    )
    assert response.metrics['pd_kv_transfer'].status == 'available'
    assert response.metrics['router_request_duration'].status == 'available'


def _censored_transfer_source(monkeypatch, cdf_state='available', *, short=False):
    module = _collector()
    prefix = 'gpustack:pd_kv_transfer_seconds'
    requests = []
    times = (4990, 4999) if short else (4910, 5000)

    def entry(value, **labels):
        return _entry(value, role='decode', **labels)

    async def instant(client, url, query, at):
        empty = short and '[15s]' in query
        if prefix + '_count{' in query:
            return [entry(0 if empty else 100)]
        if prefix + '_sum{' in query:
            return [entry(0 if empty else 1400)]
        if prefix + '_bucket{' in query:
            return [
                entry(0 if empty else value, le=bound)
                for bound, value in [('1.0', 0), ('5.0', 2), ('+Inf', 100)]
            ]
        return []

    async def ranged(client, url, query, start, end, step):
        if prefix + '_' not in query:
            return []
        requests.append(query)
        labels = entry(0)['metric']
        if 'histogram_upper_bound' in query:
            if cdf_state == 'failed':
                raise OSError('CDF source unavailable')
            if cdf_state == 'missing':
                return []
            return [
                {
                    'metric': {**labels, 'histogram_upper_bound': '5.0'},
                    'values': [[times[0], '0.02'], [times[1], '0.5']],
                },
                _histogram_range_component(
                    'count', labels, (times[0], '100'), (times[1], '100')
                ),
            ] + [
                _histogram_range_component(
                    'buckets',
                    {**labels, 'le': bound},
                    (times[0], first),
                    (times[1], last),
                )
                for bound, first, last in [
                    ('1.0', '0', '0'),
                    ('5.0', '2', '50'),
                    ('+Inf', '100', '100'),
                ]
            ]
        value = '5' if 'histogram_quantile(' in query else '14'
        return [{'metric': labels, 'values': [[times[0], value], [times[1], value]]}]

    monkeypatch.setattr(module, '_prometheus_url', lambda: 'http://metrics.invalid')
    monkeypatch.setattr(module, '_query_instant', instant)
    monkeypatch.setattr(module, '_query_range', ranged)
    return module, requests


@pytest.mark.asyncio
@pytest.mark.parametrize('view', ['aggregate', 'instances'])
@pytest.mark.parametrize('cdf_state', ['available', 'missing', 'failed'])
async def test_transfer_overflow_retains_average_and_only_emits_proven_lower_bounds(
    monkeypatch, view, cdf_state
):
    module, requests = _censored_transfer_source(monkeypatch, cdf_state)
    result = await module.collect_model_metrics(
        _model(pd=True),
        [_instance(12, 'decode'), _instance(13, 'prefill')],
        start=1400,
        end=5000,
        window_seconds=3600,
        view=view,
        client=object(),
        section='pd',
    )
    metric = result.metrics['pd_kv_transfer']
    assert metric.status == 'available'
    avg = next(line for line in metric.series if line.statistic == 'avg')
    assert avg.points == [(4910, 14), (5000, 14)]
    assert avg.lower_bounds is None
    assert next(s for s in metric.summaries if s.statistic == 'avg').value == 14
    for summary in metric.summaries:
        if summary.statistic != 'avg':
            assert summary.value is None and summary.lower_bound == 5
    assert sum('histogram_upper_bound' in query for query in requests) == 1
    if cdf_state == 'available':
        p50 = next(line for line in metric.series if line.statistic == 'p50')
        assert p50.points == [(4910, None), (5000, 5)]
        assert p50.lower_bounds == [(4910, 5)]
        p99 = next(line for line in metric.series if line.statistic == 'p99')
        assert p99.points == [(4910, None), (5000, None)]
        assert p99.lower_bounds == [(4910, 5), (5000, 5)]
        assert metric.reason == 'histogram_quantile_out_of_range'
    else:
        for line in metric.series:
            if line.statistic != 'avg':
                assert all(value is None for _, value in line.points)
                assert line.lower_bounds is None
        assert metric.reason == (
            'query_failed' if cdf_state == 'failed' else 'histogram_bounds_unavailable'
        )


@pytest.mark.asyncio
async def test_instance_bounds_keep_schemas_separate_from_incompatible_summaries(
    monkeypatch,
):
    module = _collector()
    prefix = 'gpustack:time_to_first_token_seconds'
    calls = []

    async def instant(client, url, query, at):
        if prefix + '_count{' in query:
            return [_entry(100, id=identity) for identity in (12, 13)]
        if prefix + '_sum{' in query:
            return [_entry(1400, id=identity) for identity in (12, 13)]
        if prefix + '_bucket{' in query:
            return [
                _entry(value, id=identity, le=bound)
                for identity, ceiling in [(12, '5.0'), (13, '10.0')]
                for bound, value in [('1.0', 0), (ceiling, 2), ('+Inf', 100)]
            ]
        return []

    async def ranged(client, url, query, start, end, step):
        if prefix + '_' not in query:
            return []
        if 'histogram_upper_bound' in query:
            calls.append(query)
            return [
                {
                    'metric': {
                        **_entry(0, id=identity)['metric'],
                        'histogram_upper_bound': str(bound),
                    },
                    'values': [[5000, '0.02']],
                }
                for identity, bound in [(12, 5), (13, 10)]
            ]
        return [
            {
                'metric': _entry(0, id=identity)['metric'],
                'values': [
                    [5000, str(bound if 'histogram_quantile(' in query else 14)]
                ],
            }
            for identity, bound in [(12, 5), (13, 10)]
        ]

    monkeypatch.setattr(module, '_prometheus_url', lambda: 'http://metrics.invalid')
    monkeypatch.setattr(module, '_query_instant', instant)
    monkeypatch.setattr(module, '_query_range', ranged)
    result = await module.collect_model_metrics(
        _model(),
        [_instance(12), _instance(13)],
        start=1400,
        end=5000,
        window_seconds=3600,
        view='instances',
        statistics=('p99',),
        client=object(),
    )
    metric = result.metrics['ttft']
    assert metric.status == 'available'
    assert metric.reason == 'incompatible_histogram_buckets'
    assert all(
        s.value is None and s.lower_bound is None
        for s in metric.summaries
        if s.statistic != 'avg'
    )
    assert len(calls) == 1
    assert calls[0].count('histogram_upper_bound') == 2
    assert {
        line.dimensions.model_instance_id: line.lower_bounds for line in metric.series
    } == {12: [(5000, 5)], 13: [(5000, 10)]}


@pytest.mark.asyncio
async def test_average_only_trends_do_not_query_cdf_but_keep_summary_limits(
    monkeypatch,
):
    module, requests = _censored_transfer_source(monkeypatch)
    result = await module.collect_model_metrics(
        _model(pd=True),
        [_instance(12, 'decode')],
        window_seconds=3600,
        start=1400,
        end=5000,
        statistics=('avg',),
        client=object(),
        section='pd',
    )
    metric = result.metrics['pd_kv_transfer']
    assert [line.statistic for line in metric.series] == ['avg']
    assert not any('histogram_upper_bound' in query for query in requests)
    assert next(s for s in metric.summaries if s.statistic == 'p99').lower_bound == 5


@pytest.mark.asyncio
async def test_short_phase_limits_do_not_borrow_rolling_observations_for_summaries(
    monkeypatch,
):
    from gpustack.monitoring.time_range import MonitoringInterval

    module, requests = _censored_transfer_source(monkeypatch, short=True)
    result = await module.collect_model_metrics(
        _model(pd=True),
        [_instance(12, 'decode')],
        window_seconds=3600,
        interval=MonitoringInterval(4985, 5000, 15),
        statistics=('p99',),
        allow_trend_lookback=True,
        client=object(),
        section='pd',
    )
    metric = result.metrics['pd_kv_transfer']
    assert all(
        s.value is None and s.lower_bound is None and s.sample_count == 0
        for s in metric.summaries
    )
    assert len(metric.series) == 1
    assert metric.series[0].points == [(4990, None), (4999, None)]
    assert metric.series[0].lower_bounds == [(4990, 5), (4999, 5)]
    assert metric.reason == 'histogram_quantile_out_of_range'
    assert metric.status == 'stale'
    assert sum('histogram_upper_bound' in query for query in requests) == 1


@pytest.mark.asyncio
async def test_cdf_deadline_never_leaves_unverified_ceiling_values(monkeypatch):
    module, _ = _censored_transfer_source(monkeypatch)
    original = module._query_range

    async def delayed(client, url, query, start, end, step):
        if 'histogram_upper_bound' in query:
            await asyncio.sleep(10)
        return await original(client, url, query, start, end, step)

    monkeypatch.setattr(module, '_query_range', delayed)
    monkeypatch.setattr(module, '_COLLECT_DEADLINE_SECONDS', 0.05)
    result = await module.collect_model_metrics(
        _model(pd=True),
        [_instance(12, 'decode')],
        window_seconds=3600,
        start=1400,
        end=5000,
        client=object(),
        section='pd',
    )
    metric = result.metrics['pd_kv_transfer']
    assert metric.status == 'query_error' and metric.reason == 'query_failed'
    assert next(s for s in metric.summaries if s.statistic == 'avg').value == 14
    assert next(s for s in metric.summaries if s.statistic == 'p99').lower_bound == 5
    assert all(
        value is None
        for line in metric.series
        if line.statistic != 'avg'
        for _, value in line.points
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'case,valid',
    [
        ('same-cohort-alias', False),
        ('conflicting-alias', False),
        ('distinct-cohort-alias', True),
        ('same-le-scrape-copy', True),
    ],
)
async def test_numeric_bucket_aliases_fail_closed_only_within_one_native_cohort(
    monkeypatch, case, valid
):
    module = _collector()
    prefix = 'gpustack:time_to_first_token_seconds'
    requests = []

    def records(part):
        profiles = [('a', '5.0')]
        if case == 'distinct-cohort-alias':
            profiles.append(('b', '5'))
        if part == 'bucket':
            rows = [
                _entry(value, engine=engine, le=label)
                for engine, ceiling in profiles
                for label, value in [('1.0', 0), (ceiling, 2), ('+Inf', 100)]
            ]
            if case in ('same-cohort-alias', 'conflicting-alias'):
                rows.append(
                    _entry(3 if case == 'conflicting-alias' else 2, engine='a', le='5')
                )
        else:
            rows = [
                _entry(1400 if part == 'sum' else 100, engine=engine)
                for engine, _ in profiles
            ]
        if case == 'same-le-scrape-copy':
            rows += [
                {**row, 'metric': {**row['metric'], 'job': 'second-scraper'}}
                for row in rows
            ]
        return rows

    async def instant(client, url, query, at):
        for part in ('count', 'sum', 'bucket'):
            if prefix + '_' + part + '{' in query:
                return records(part)
        return []

    async def ranged(client, url, query, start, end, step):
        if prefix + '_' not in query:
            return []
        requests.append(query)
        if 'histogram_component' in query:
            return [
                _histogram_range_component(part, row['metric'], (5000, row['value'][1]))
                for part, native in [('count', 'count'), ('buckets', 'bucket')]
                for row in module.deduplicate_instant(records(native))
            ]
        return [{'metric': _entry(0)['metric'], 'values': [[5000, '0.25']]}]

    monkeypatch.setattr(module, '_prometheus_url', lambda: 'http://metrics.invalid')
    monkeypatch.setattr(module, '_query_instant', instant)
    monkeypatch.setattr(module, '_query_range', ranged)
    result = await module.collect_model_metrics(
        _model(),
        [_instance()],
        window_seconds=3600,
        start=1400,
        end=5000,
        client=object(),
    )
    metric = result.metrics['ttft']
    assert next(s for s in metric.summaries if s.statistic == 'avg').value == 14
    assert any('histogram_quantile(' in q for q in requests) == valid
    if valid:
        assert (
            next(s for s in metric.summaries if s.statistic == 'p99').lower_bound == 5
        )
        assert metric.reason == 'histogram_quantile_out_of_range'
    else:
        assert metric.reason == 'incompatible_histogram_buckets'
        assert metric.supported_statistics == ['avg']
        assert all(
            s.value is None and s.lower_bound is None
            for s in metric.summaries
            if s.statistic != 'avg'
        )
