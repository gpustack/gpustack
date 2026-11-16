"""Benchmark history uses measured time and immutable, authorized associations."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


def _ctx(owner=4, kind=None):
    from gpustack.api.tenant import TenantContext
    from gpustack.schemas.principals import OrgRole, PrincipalType

    return TenantContext(
        user=SimpleNamespace(kind=kind or PrincipalType.USER),
        is_platform_admin=False,
        current_principal_id=owner,
        org_role=OrgRole.OWNER,
    )


def _recorded():
    from gpustack.schemas.benchmark_monitoring import BenchmarkRunSnapshot

    return BenchmarkRunSnapshot(
        captured_at=900,
        confirmed_at=901,
        model_id=9,
        cluster_id=1,
        owner_principal_id=4,
        backend='vLLM',
        instances=[
            dict(
                id=12,
                name='historical-instance',
                worker_id=3,
                worker_ids=[3],
                gpu_indexes=[0],
                restart_count=0,
                pid=400,
                allocations={3: [0]},
                source_identities=[
                    dict(
                        model_instance_id='12',
                        worker_id='3',
                        restart_count='0',
                        source_kind='legacy',
                        source_name='legacy',
                        endpoint_dp_rank='',
                        component_attempt='',
                    )
                ],
            )
        ],
        workers=[
            dict(
                id=3,
                name='secret-worker',
                worker_uuid='secret-uuid',
                owner_principal_id=4,
            )
        ],
        devices=[
            dict(
                id='device3',
                worker_id=3,
                name='secret-gpu',
                index=0,
                uuid='gpu-uuid',
                model_instance_ids=[12],
            )
        ],
    )


def _history():
    from gpustack.schemas.benchmark_monitoring import BenchmarkMonitoringContext
    from gpustack.schemas.model_metrics import ModelMonitoringCapabilities

    module = _module()
    snapshot = _recorded()
    stages, window = module.benchmark_windows([_point()])
    return module.BenchmarkHistory(
        context=BenchmarkMonitoringContext(
            benchmark_id=7,
            model_id=9,
            available=True,
            run_window=window,
            stages=stages,
            resource_status='available',
            capabilities=ModelMonitoringCapabilities(model_id=9),
            workers=[dict(id=3, name='secret-worker')],
        ),
        snapshot=snapshot,
        model=SimpleNamespace(
            id=9, cluster_id=1, backend='vLLM', roles=[], disaggregation=None
        ),
        instances=snapshot.instances,
        workers={w.id: w for w in snapshot.workers},
        devices=snapshot.devices,
    )


def _module():
    from gpustack.monitoring import benchmarks

    return benchmarks


def _point(sequence=0, start=1000.25, end=1040.75):
    return SimpleNamespace(
        input_tokens=128,
        sequence=sequence,
        rate=4,
        raw_metrics={
            'scheduler_metrics': {
                'request_start_time': start - 2,
                'measure_start_time': start,
                'measure_end_time': end,
                'request_end_time': end + 3,
            }
        },
    )


def test_phase_windows_retain_fractional_short_measurements():
    stages, window = _module().benchmark_windows([_point()])
    assert stages[0].id == '128:0'
    assert stages[0].start == 1000.25
    assert stages[0].end == 1040.75
    assert window.start == 998.25 and window.end == 1043.75


def test_navigation_metadata_preserves_execution_number_and_observed_search_pattern():
    points = [_point(i, 1000 + i * 50, 1040 + i * 50) for i in range(4)]
    for point, rate in zip(points, [256, 512, 384, 281]):
        point.rate = rate
    stages, _ = _module().benchmark_windows(list(reversed(points)), mode='auto_tune')
    assert [s.rate for s in stages] == [256, 512, 384, 281]
    assert [s.measurement_number for s in stages] == [1, 2, 3, 4]
    assert [s.search_phase for s in stages] == ['expand', 'expand', 'refine', 'refine']
    assert {s.search_phase_source for s in stages} == {'load_history'}
    assert stages[-1].id == '128:3'
    assert stages[-1].start == 1150


@pytest.mark.parametrize('mode,expected', [('stages', 'manual'), ('single', 'single')])
def test_manual_and_single_navigation_never_infer_adaptive_search(mode, expected):
    points = [_point(0), _point(1, 1050, 1090)]
    points[0].rate, points[1].rate = 512, 256
    stages, _ = _module().benchmark_windows(points, mode=mode)
    assert {s.search_phase for s in stages} == {expected}
    assert {s.search_phase_source for s in stages} == {'configuration'}


def test_navigation_cohorts_and_saturation_probe_keep_independent_identities():
    first, second, probe = _point(0), _point(0, 1100, 1140), _point(2, 900, 910)
    first.rate, second.rate, probe.rate = 512, 4, None
    second.input_tokens = 256
    stages, _ = _module().benchmark_windows([first, second, probe], mode='auto_tune')
    assert [s.id for s in stages] == ['128:0', '128:2', '256:0']
    assert [s.search_phase for s in stages] == ['expand', 'probe', 'expand']
    assert stages[1].measurement_number is None
    assert stages[2].measurement_number == 1


@pytest.mark.parametrize('timing', [None, {}, {'measure_start_time': float('nan')}])
def test_missing_timing_never_infers_phase_time(timing):
    point = _point()
    point.raw_metrics = {'scheduler_metrics': timing}
    stages, window = _module().benchmark_windows([point])
    assert stages[0].status == 'unavailable'
    assert stages[0].start is None and window is None


def test_stage_statistics_use_half_open_window_and_keep_zero():
    from gpustack.schemas.model_metrics import MetricSeries

    series = MetricSeries(
        name='gpu',
        statistic='current',
        points=[(9, 999), (10, 0), (20, 100), (30, None), (40, 200)],
    )
    stats = _module().sample_statistics(series, 10, 40, step=10, threshold=90)
    assert stats.average == 50
    assert stats.peak == 100 and stats.peak_at == 20
    assert stats.last == 100 and stats.last_at == 20
    assert stats.threshold_fraction == 0.5
    assert stats.observed_samples == 2 and stats.expected_samples == 3
    assert stats.coverage == pytest.approx(2 / 3)


def test_exact_interval_does_not_change_public_model_minimum():
    from gpustack.monitoring.time_range import monitoring_interval

    module = _module()
    interval = module.exact_interval(1000.25, 1040.75)
    assert interval.start == 1000.25 and interval.end == 1040.75
    with pytest.raises(ValueError):
        monitoring_interval(0, 1000, 1041)


@pytest.mark.parametrize(
    'start,end,message',
    [
        (1000, 1000.0009, 'Benchmark range must span 1 millisecond to 7 days'),
        (True, 1001, 'Benchmark range must span 1 millisecond to 7 days'),
        (float('nan'), 1001, 'Benchmark range must span 1 millisecond to 7 days'),
        (1000, 1001, 'Benchmark range cannot end in the future'),
    ],
)
def test_exact_interval_preserves_benchmark_validation_messages(
    monkeypatch, start, end, message
):
    module = _module()
    monkeypatch.setattr(module.time, 'time', lambda: 1000)
    with pytest.raises(ValueError, match=message):
        module.exact_interval(start, end)


def test_fractional_summary_duration_is_rendered_as_milliseconds():
    from gpustack.monitoring.queries import (
        MetricQueryScope,
        build_histogram_components,
    )

    scope = MetricQueryScope(1, 9, (12,), ('',))
    query = build_histogram_components('gpustack:latency_seconds', scope, 40.5)
    assert '[40500ms]' in query['sum']
    assert '[40.5s]' not in query['sum']


@pytest.mark.asyncio
async def test_unknown_history_does_not_query_current_model(monkeypatch):
    from gpustack.schemas.benchmark import Benchmark
    from gpustack.schemas.principals import OrgRole, PrincipalType
    from gpustack.api.tenant import TenantContext

    module = _module()
    benchmark = Benchmark(id=7, name='legacy', model_id=9, owner_principal_id=4)
    monkeypatch.setattr(
        module.Benchmark, 'one_by_id', AsyncMock(return_value=benchmark)
    )
    monkeypatch.setattr(
        module.BenchmarkResult, 'all_by_fields', AsyncMock(return_value=[_point()])
    )
    lookup = AsyncMock()
    monkeypatch.setattr(module.Model, 'one_by_id', lookup)
    ctx = TenantContext(
        user=SimpleNamespace(kind=PrincipalType.USER),
        is_platform_admin=False,
        current_principal_id=4,
        org_role=OrgRole.OWNER,
    )
    history = await module.load_benchmark_history(None, ctx, 7)
    assert history.context.reason == 'historical_association_unavailable'
    assert history.context.run_window.start == 998.25
    lookup.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('supplied_id,owner', [(7, 4), (8, 4), (7, 8)])
async def test_preloaded_benchmark_reuses_only_the_requested_visible_row(
    monkeypatch, supplied_id, owner
):
    from gpustack.api.exceptions import NotFoundException
    from gpustack.schemas.benchmark import Benchmark

    module = _module()
    requested = Benchmark(id=7, name='run', model_id=9, owner_principal_id=4)
    supplied = Benchmark(
        id=supplied_id, name='other', model_id=99, owner_principal_id=owner
    )
    lookup = AsyncMock(return_value=requested)
    results = AsyncMock(return_value=[])
    monkeypatch.setattr(module.Benchmark, 'one_by_id', lookup)
    monkeypatch.setattr(module.BenchmarkResult, 'all_by_fields', results)
    if owner != 4:
        with pytest.raises(NotFoundException):
            await module.load_benchmark_history(None, _ctx(), 7, benchmark=supplied)
        lookup.assert_not_awaited()
        results.assert_not_awaited()
        return

    history = await module.load_benchmark_history(None, _ctx(), 7, benchmark=supplied)
    assert history.context.model_id == (99 if supplied_id == 7 else 9)
    assert lookup.await_count == (0 if supplied_id == 7 else 1)


@pytest.mark.asyncio
async def test_visible_benchmark_does_not_expose_hidden_hardware(monkeypatch):
    from gpustack.schemas.benchmark import Benchmark, BenchmarkSnapshot

    module = _module()
    benchmark = Benchmark(
        id=7,
        name='b',
        model_id=9,
        owner_principal_id=4,
        snapshot=BenchmarkSnapshot(monitoring=_recorded()),
    )
    monkeypatch.setattr(
        module.Benchmark, 'one_by_id', AsyncMock(return_value=benchmark)
    )
    monkeypatch.setattr(
        module.BenchmarkResult, 'all_by_fields', AsyncMock(return_value=[_point()])
    )
    monkeypatch.setattr(
        module.Model,
        'one_by_id',
        AsyncMock(return_value=SimpleNamespace(owner_principal_id=4)),
    )
    instance = SimpleNamespace(id=12, owner_principal_id=4, model_id=9)
    worker = SimpleNamespace(id=3, owner_principal_id=8, worker_uuid='secret-uuid')
    session = SimpleNamespace(
        exec=AsyncMock(
            side_effect=[
                SimpleNamespace(all=lambda: [instance]),
                SimpleNamespace(all=lambda: [worker]),
            ]
        )
    )
    monkeypatch.setattr(module, 'collect_storage_policy', AsyncMock(return_value={}))
    history = await module.load_benchmark_history(session, _ctx(), 7)
    instance_query, worker_query = [
        call.args[0].compile().params for call in session.exec.await_args_list
    ]
    assert instance_query == {'id_1': [12], 'owner_principal_id_1': 4}
    assert worker_query == {'id_1': [3], 'owner_principal_id_1': 4}
    data = history.context.model_dump_json()
    assert history.context.available
    assert history.context.resource_status == 'restricted'
    assert history.context.devices == [] and history.context.workers == []
    assert history.context.instances[0].worker_id is None
    assert history.context.instances[0].worker_ids == []
    assert history.context.instances[0].gpu_indexes == []
    assert all(
        s not in data
        for s in ('secret-', 'source_identities', 'worker_ip', 'pid', 'allocations')
    )


def test_internal_run_snapshot_is_persisted_but_not_public():
    from datetime import datetime
    from gpustack.schemas.benchmark import (
        Benchmark,
        BenchmarkFullPublic,
        BenchmarkPublic,
        BenchmarkSnapshot,
    )

    benchmark = Benchmark(
        id=7,
        name='b',
        snapshot=BenchmarkSnapshot(monitoring=_recorded()),
        cluster_id=1,
        model_instance_name='mi',
        created_at=datetime(2026, 1, 1),
        updated_at=datetime(2026, 1, 1),
    )
    assert benchmark.model_dump()['snapshot']['monitoring']['workers']
    for public in (BenchmarkFullPublic, BenchmarkPublic):
        data = public.model_validate(benchmark, from_attributes=True).model_dump_json()
        assert 'monitoring' not in data and 'secret-worker' not in data


@pytest.mark.asyncio
@pytest.mark.parametrize('phase_id', [None, '128:0'])
async def test_hidden_per_source_observation_is_restricted_without_source_count(
    monkeypatch, phase_id
):
    from gpustack.schemas.model_metrics import (
        MetricSeries,
        MetricDimensions,
        ModelMetric,
        ModelMetrics,
    )

    module, history = _module(), _history()
    history.workers = {}
    response = ModelMetrics(
        model_id=9,
        start=1000.25,
        end=1040.75,
        step=15,
        rate_window_seconds=75,
        last_refreshed_at=1040.75,
        history_scope='current',
        metrics={
            'kv_usage': ModelMetric(
                kind='gauge',
                unit='percent',
                measurement_scope='source',
                status='available',
                series=[
                    MetricSeries(
                        name='secret-source',
                        statistic='current',
                        dimensions=MetricDimensions(
                            worker_id=3, source_name='secret-source'
                        ),
                        points=[(1010, 0)],
                    )
                ],
            )
        },
    )
    monkeypatch.setattr(
        module, 'collect_model_metrics', AsyncMock(return_value=response)
    )
    result = await module.collect_benchmark_model_metrics(history, phase_id=phase_id)
    assert result.metrics['kv_usage'].status == 'restricted'
    assert result.metrics['kv_usage'].series == []
    assert result.metrics['kv_usage'].summaries == []
    assert result.metrics['kv_usage'].coverage is None
    assert 'secret-source' not in result.model_dump_json()


def test_short_phase_masks_rolling_points_and_retains_real_summary():
    from gpustack.schemas.model_metrics import MetricSeries, MetricSummary, ModelMetric
    from gpustack.schemas.benchmark_monitoring import BenchmarkModelMetrics

    history = _history()
    result = BenchmarkModelMetrics(
        model_id=9,
        start=1000.25,
        end=1040.75,
        step=15,
        rate_window_seconds=75,
        last_refreshed_at=1040.75,
        history_scope='benchmark_snapshot',
        metrics={
            'ttft': ModelMetric(
                kind='histogram',
                unit='seconds',
                measurement_scope='engine_local',
                status='available',
                summaries=[MetricSummary(statistic='p99', value=0.8)],
                series=[
                    MetricSeries(
                        name='decode',
                        statistic='p99',
                        points=[(1010, 9), (1040.75, 99)],
                    )
                ],
            )
        },
    )
    _module()._project_samples(result, history, '128:0')
    assert result.metrics['ttft'].series[0].points == [(1010, None)]
    assert result.metrics['ttft'].summaries[0].value == 0.8
    assert result.metrics['ttft'].reason == 'insufficient_phase_rate_window'


@pytest.mark.asyncio
async def test_generation_changed_never_queries_reallocated_resources(monkeypatch):
    module, history = _module(), _history()
    history.context.resource_status = 'unavailable'
    history.context.resource_reason = 'historical_association_changed'
    monkeypatch.setattr(
        module.prometheus, 'prometheus_url', lambda: 'http://metrics.invalid'
    )
    collect = AsyncMock()
    monkeypatch.setattr(module.hardware, '_collect_all_history', collect)
    response = await module.collect_benchmark_resource_metrics(
        history, phase_id='128:0'
    )
    assert response.reason == 'historical_association_changed'
    assert not response.available
    collect.assert_not_awaited()


@pytest.mark.asyncio
async def test_stage_end_resource_value_excludes_next_stage_start(monkeypatch):
    from gpustack.schemas.model_metrics import (
        MetricSeries,
        MetricSummary,
        ModelMetric,
        MetricDimensions,
    )

    module, history = _module(), _history()
    monkeypatch.setattr(
        module.prometheus, 'prometheus_url', lambda: 'http://metrics.invalid'
    )

    async def collect(devices, workers, response, client, url, **kwargs):
        dimensions = MetricDimensions(worker_id=3, device_id='device3')
        response.metrics['gpu_utilization'] = ModelMetric(
            kind='gauge',
            unit='percent',
            measurement_scope='physical_gpu',
            status='available',
            series=[
                MetricSeries(
                    name='GPU',
                    statistic='current',
                    dimensions=dimensions,
                    points=[(1000.25, 0), (1015.25, 60), (1040.75, 100)],
                )
            ],
            summaries=[
                MetricSummary(statistic='current', dimensions=dimensions, value=100)
            ],
        )

    monkeypatch.setattr(module.hardware, '_collect_all_history', collect)
    response = await module.collect_benchmark_resource_metrics(
        history, phase_id='128:0', client=object()
    )
    assert response.metrics['gpu_utilization'].summaries[0].value == 60
    stats = response.sample_statistics['gpu_utilization'][0]
    assert stats.average == 30 and stats.peak == 60 and stats.last == 60


@pytest.mark.asyncio
async def test_next_phase_only_sample_does_not_make_selected_phase_available(
    monkeypatch,
):
    from gpustack.schemas.model_metrics import ModelMetric, MetricSeries, MetricSummary

    module, history = _module(), _history()
    monkeypatch.setattr(
        module.prometheus, 'prometheus_url', lambda: 'http://metrics.invalid'
    )

    async def collect(devices, workers, response, client, url, **kwargs):
        response.metrics['gpu_utilization'] = ModelMetric(
            kind='gauge',
            unit='percent',
            measurement_scope='physical_gpu',
            status='available',
            series=[
                MetricSeries(name='GPU', statistic='current', points=[(1040.75, 100)])
            ],
            summaries=[MetricSummary(statistic='current', value=100)],
        )

    monkeypatch.setattr(module.hardware, '_collect_all_history', collect)
    response = await module.collect_benchmark_resource_metrics(
        history, phase_id='128:0', client=object()
    )
    assert not response.available
    assert response.metrics['gpu_utilization'].status == 'no_data'
    assert response.reason == 'no_samples'


@pytest.mark.asyncio
async def test_run_verification_batches_resources_and_records_missing_generations(
    monkeypatch,
):
    from gpustack.schemas.benchmark import Benchmark, BenchmarkSnapshot

    module, recorded = _module(), _recorded()
    member = recorded.instances[0]
    recorded.instances.extend(
        [
            member.model_copy(update={'id': 13}),
            member.model_copy(update={'id': 14}),
        ]
    )
    current = [
        SimpleNamespace(id=id, pid=400, worker_id=3, restart_count=0) for id in (12, 13)
    ]
    worker = SimpleNamespace(id=3, worker_uuid='secret-uuid', owner_principal_id=4)
    session = SimpleNamespace(
        exec=AsyncMock(
            side_effect=[
                SimpleNamespace(all=lambda: current),
                SimpleNamespace(all=lambda: []),
                SimpleNamespace(all=lambda: [worker]),
            ]
        )
    )
    monkeypatch.setattr(module, '_ASSOCIATION_BATCH_SIZE', 2, raising=False)
    monkeypatch.setattr(
        module, '_source_identities', lambda _: member.source_identities
    )
    monkeypatch.setattr(module.hardware, '_allocated_gpus', lambda _: {3: {0}})
    benchmark = Benchmark(name='b', snapshot=BenchmarkSnapshot(monitoring=recorded))

    result = await module.verify_benchmark_run(benchmark, session)

    assert [event.model_instance_id for event in result.monitoring.events] == [14]
    assert [call.args[0].compile().params for call in session.exec.await_args_list] == [
        {'id_1': [12, 13]},
        {'id_1': [14]},
        {'id_1': [3]},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['attempt', 'worker_uuid'])
async def test_run_verification_records_source_or_worker_generation_change(
    monkeypatch, change
):
    from gpustack.schemas.benchmark import Benchmark, BenchmarkSnapshot

    module, recorded = _module(), _recorded()
    current = SimpleNamespace(id=12, pid=400, worker_id=3, restart_count=0)
    identities = recorded.instances[0].source_identities
    if change == 'attempt':
        identities = [{**identities[0], 'component_attempt': '1'}]
    monkeypatch.setattr(module, '_source_identities', lambda instance: identities)
    monkeypatch.setattr(module.hardware, '_allocated_gpus', lambda instance: {3: {0}})
    monkeypatch.setattr(
        module.ModelInstance, 'all_by_fields', AsyncMock(return_value=[current])
    )
    monkeypatch.setattr(
        module.Worker,
        'all_by_fields',
        AsyncMock(
            return_value=[
                SimpleNamespace(
                    id=3,
                    worker_uuid=(
                        'other-uuid' if change == 'worker_uuid' else 'secret-uuid'
                    ),
                    owner_principal_id=4,
                )
            ]
        ),
    )
    benchmark = Benchmark(name='b', snapshot=BenchmarkSnapshot(monitoring=recorded))
    result = await module.verify_benchmark_run(benchmark, None, confirm=True)
    assert result.monitoring.events[-1].kind == 'generation_changed'
    assert result.monitoring.events[-1].model_instance_id == 12
    assert result.monitoring.confirmed_at == 901


def test_scope_guards_freeze_all_source_worker_and_attempt_identities():
    from gpustack.monitoring.queries import MetricQueryScope, build_histogram_components

    first = _recorded().instances[0].source_identities[0]
    second = {
        **first,
        'worker_id': '8',
        'source_kind': 'sglang_smg',
        'source_name': 'router-task',
        'component_attempt': '2',
    }
    scope = MetricQueryScope(1, 9, (12,), ('',), source_identities=(first, second))
    query = build_histogram_components('gpustack:latency_seconds', scope, 40)['sum']
    assert 'worker_id="3"' in query and 'worker_id="8"' in query
    assert 'component_attempt="2"' in query and 'source_name="router-task"' in query
    assert ')[40s]' not in query


def test_benchmark_gauges_reject_samples_stored_before_phase():
    from gpustack.monitoring.queries import MetricQueryScope, build_gauge_query
    from gpustack.monitoring.hardware import _fresh

    scope = MetricQueryScope(1, 9, (12,), ('',), sample_start=1000.25)
    query = build_gauge_query('gpustack:running_requests', scope)
    assert 'timestamp(gpustack:running_requests' in query
    assert '>= 1000.25' in query
    assert '>= 1000.25' in _fresh('gpustack:gpu_utilization{}', not_before=1000.25)


@pytest.mark.asyncio
@pytest.mark.parametrize('phase_id', [None, '128:0'])
async def test_worker_epoch_change_does_not_mix_engine_history(monkeypatch, phase_id):
    module, history = _module(), _history()
    history.context.available = False
    history.context.reason = 'historical_association_changed'
    collector = AsyncMock()
    monkeypatch.setattr(module, 'collect_model_metrics', collector)
    result = await module.collect_benchmark_model_metrics(history, phase_id=phase_id)
    assert result.reason == 'historical_association_changed'
    collector.assert_not_awaited()


@pytest.mark.asyncio
async def test_valid_phase_can_query_when_another_phase_has_no_timing(monkeypatch):
    module, history = _module(), _history()
    unknown = _point(sequence=1)
    unknown.raw_metrics = {}
    stages, window = module.benchmark_windows([_point(), unknown])
    history.context.stages, history.context.run_window = stages, window
    history.context.available = False
    history.context.reason = 'run_timing_unavailable'
    monkeypatch.setattr(module.prometheus, 'prometheus_url', lambda: None)
    result = await module.collect_benchmark_resource_metrics(history, phase_id='128:0')
    assert result.start == 1000.25 and result.end == 1040.75
    assert result.reason == 'prometheus_unavailable'


@pytest.mark.asyncio
async def test_capture_preserves_allocations_when_inventory_identity_is_ambiguous(
    monkeypatch,
):
    from gpustack.schemas.benchmark import Benchmark, BenchmarkSnapshot
    from gpustack.schemas.models import Model, ModelInstance
    from tests.monitoring.test_hardware import _worker

    module = _module()
    worker = _worker()
    device = worker.status.gpu_devices[0]
    worker.status.gpu_devices = [device, device.model_copy(deep=True)]
    member = ModelInstance(
        id=12,
        model_id=9,
        name='instance',
        worker_id=worker.id,
        pid=400,
        ports=[8000],
        port=8000,
        restart_count=0,
        gpu_indexes=[device.index],
    )
    model = Model(id=9, name='m', cluster_id=1, owner_principal_id=4, backend='vllm')
    snapshot = await module.capture_benchmark_run(
        BenchmarkSnapshot(),
        model,
        [member],
        None,
        workers={worker.id: worker},
    )
    assert len(snapshot.monitoring.devices) == 1
    assert snapshot.monitoring.devices[0].uuid is None
    assert snapshot.monitoring.instances[0].allocations == {worker.id: [device.index]}
    monkeypatch.setattr(
        module.ModelInstance, 'all_by_fields', AsyncMock(return_value=[member])
    )
    monkeypatch.setattr(
        module.Worker, 'all_by_fields', AsyncMock(return_value=[worker])
    )
    verified = await module.verify_benchmark_run(
        Benchmark(name='b', snapshot=snapshot), None, confirm=True
    )
    assert verified.monitoring.confirmed_at is not None
    assert [e.kind for e in verified.monitoring.events] == ['association_captured']


def _short_slo_history():
    history = _history()
    points = [
        _point(start=1000.25, end=1011.1880345),
        _point(sequence=1, start=1013.1090295, end=1019.5954151),
    ]
    for point in points:
        timing = point.raw_metrics['scheduler_metrics']
        timing['request_start_time'] = timing['measure_start_time']
        timing['request_end_time'] = timing['measure_end_time']
    history.context.stages, history.context.run_window = _module().benchmark_windows(
        points
    )
    return history


@pytest.mark.asyncio
@pytest.mark.parametrize('phase_id', [None, '128:0'])
@pytest.mark.parametrize('summary_count', [None, 0])
async def test_short_slo_run_keeps_engine_context_separate_from_window_summaries(
    monkeypatch, phase_id, summary_count
):
    from gpustack.monitoring import models
    from tests.monitoring.test_models import _entry, _histogram_data, _prometheus

    module, history = _module(), _short_slo_history()
    interval = module._selected_interval(history, phase_id, None, None)

    def instant(query):
        histogram = query.replace(
            'gpustack:time_per_output_token_seconds',
            'gpustack:time_to_first_token_seconds',
        )
        if '[75s]' in query:
            return _histogram_data(histogram)
        if summary_count is not None:
            return _histogram_data(histogram, counts=summary_count)
        return []

    def ranges(query):
        if not any(
            family in query
            for family in (
                'time_to_first_token_seconds',
                'time_per_output_token_seconds',
                'prompt_tokens_total',
            )
        ):
            return []
        return [
            {
                'metric': _entry(1)['metric'],
                'values': [
                    [interval.start, '0'],
                    [interval.start + 15, '0.15'],
                    [interval.end, '99'],
                ],
            }
        ]

    calls = _prometheus(monkeypatch, models, instant, ranges)
    result = await module.collect_benchmark_model_metrics(
        history, phase_id=phase_id, client=object()
    )

    assert result.start == interval.start and result.end == interval.end
    assert result.rate_window_seconds == 75
    for key in ('ttft', 'tpot', 'input_tokens'):
        metric = result.metrics[key]
        values = [value for line in metric.series for _, value in line.points]
        if phase_id is None:
            assert any(value == 0 for value in values)
            assert any(value == 0.15 for value in values)
            assert metric.reason != 'insufficient_phase_rate_window'
        else:
            assert all(value is None for value in values)
            assert metric.reason in (
                'insufficient_phase_rate_window',
                'no_samples',
                'zero_observations',
            )
        assert all(value != 99 for value in values)
        assert all(summary.value is None for summary in metric.summaries)
    for key in ('ttft', 'tpot'):
        summaries = result.metrics[key].summaries
        assert len(summaries) == 4
        assert all(summary.sample_count == summary_count for summary in summaries)
    context_queries = [q for kind, q in calls if kind == 'instant' and '[75s]' in q]
    assert bool(context_queries) is (phase_id is None)
    for _, query in calls:
        assert 'worker_id="3"' in query
        assert 'restart_count="0"' in query
        assert 'source_kind="legacy"' in query
        assert 'source_name="legacy"' in query
        assert 'component_attempt=""' in query


@pytest.mark.asyncio
async def test_short_run_context_rejects_incompatible_histogram_quantiles(monkeypatch):
    from gpustack.monitoring import models
    from tests.monitoring.test_models import _entry, _prometheus

    history = _short_slo_history()

    def instant(query):
        if '[75s]' not in query or 'time_to_first_token_seconds' not in query:
            return []
        if '_bucket{' in query:
            return [
                _entry(value, engine=engine, le=bound)
                for engine, middle in [('0', '0.2'), ('1', '0.3')]
                for bound, value in [('0.1', 5), (middle, 10), ('+Inf', 10)]
            ]
        return [_entry(10 if '_count{' in query else 1, engine=e) for e in ('0', '1')]

    def ranges(query):
        if 'time_to_first_token_seconds' not in query:
            return []
        return [{'metric': _entry(1)['metric'], 'values': [[1015.25, '0']]}]

    calls = _prometheus(monkeypatch, models, instant, ranges)
    result = await _module().collect_benchmark_model_metrics(history, client=object())
    metric = result.metrics['ttft']
    assert metric.supported_statistics == ['avg']
    assert metric.reason == 'incompatible_histogram_buckets'
    assert [line.statistic for line in metric.series] == ['avg']
    assert metric.series[0].points == [(1015.25, 0)]
    assert all(summary.value is None for summary in metric.summaries)
    assert not any(
        kind == 'range'
        and 'time_to_first_token_seconds' in query
        and 'histogram_quantile' in query
        for kind, query in calls
    )


@pytest.mark.asyncio
async def test_short_slo_rolling_context_does_not_invent_zero_latency(monkeypatch):
    from gpustack.monitoring import models
    from tests.monitoring.test_models import _histogram_data, _prometheus

    calls = _prometheus(
        monkeypatch, models, lambda query: _histogram_data(query, counts=0)
    )
    result = await _module().collect_benchmark_model_metrics(
        _short_slo_history(), client=object()
    )
    metric = result.metrics['ttft']
    assert metric.status == 'no_observations'
    assert metric.reason == 'zero_observations'
    assert all(summary.value is None for summary in metric.summaries)
    assert all(summary.sample_count == 0 for summary in metric.summaries)
    assert not any(
        kind == 'range'
        and 'time_to_first_token_seconds' in query
        and 'histogram_quantile' in query
        for kind, query in calls
    )


@pytest.mark.parametrize(
    'start,end,expected_start,expected_end,samples',
    [
        (1794590261.206044, 1794590276.1269433, 1794590261.207, 1794590276.126, 15),
        (1000.000044, 1003.000054, 1000.001, 1003.0, 3),
        (1000.25, 1003.25, 1000.25, 1003.249, 3),
        (1000.250001, 1000.251001, 1000.251, 1000.251, 1),
    ],
)
def test_benchmark_evaluation_grid_stays_inside_exact_runner_bounds(
    start, end, expected_start, expected_end, samples
):
    from gpustack.schemas.model_metrics import MetricSeries

    module = _module()
    grid = module.benchmark_query_interval(module.exact_interval(start, end))
    assert grid.start == expected_start and grid.end == expected_end
    assert start <= grid.start <= grid.end < end
    stats = module.sample_statistics(
        MetricSeries(name='GPU', statistic='current'), start, end, step=1
    )
    assert stats.expected_samples == samples


@pytest.mark.asyncio
@pytest.mark.parametrize('span', [14.920899391174316, 3.0])
@pytest.mark.parametrize('raw_samples', [True, False])
async def test_short_benchmark_queries_inner_millisecond_grid_and_preserves_missing_data(
    monkeypatch, span, raw_samples
):
    import math
    from gpustack.monitoring import models
    from tests.monitoring.test_models import _entry, _histogram_data, _prometheus

    module, history = _module(), _history()
    start = 1794590261.206044
    point = _point(start=start, end=start + span)
    timing = point.raw_metrics['scheduler_metrics']
    timing['request_start_time'], timing['request_end_time'] = start, start + span
    history.context.stages, history.context.run_window = module.benchmark_windows(
        [point]
    )
    calls = []

    def instant(query):
        return _histogram_data(query) if raw_samples and '[75s]' in query else []

    _prometheus(monkeypatch, models, instant)
    monkeypatch.setattr(models.time, 'time', lambda: start + 100)

    async def ranges(client, url, query, query_start, query_end, step):
        calls.append((query, query_start, query_end, step))
        assert start <= query_start <= query_end < start + span
        assert step == 1
        if not raw_samples:
            return []
        if not any(
            f in query
            for f in (
                'time_to_first_token_seconds',
                'prompt_tokens_total',
                'num_requests_running',
            )
        ):
            return []
        # Prometheus stores query evaluation timestamps at millisecond precision.
        ticks = [
            math.floor((query_start + n * step) * 1000) / 1000
            for n in range(math.floor((query_end - query_start) / step) + 1)
        ]
        if 'num_requests_running' in query:
            assert f'>= {start}' in query
            ticks = [tick for tick in ticks if tick >= start + 1.5]
        return [
            {'metric': _entry(1)['metric'], 'values': [[tick, '0'] for tick in ticks]}
        ]

    monkeypatch.setattr(models, '_query_range', ranges)
    result = await module.collect_benchmark_model_metrics(history, client=object())
    assert result.start == start and result.end == start + span
    assert result.step == 1 and result.rate_window_seconds == 75
    assert result.available is raw_samples
    for key in ('ttft', 'input_tokens', 'running_requests'):
        values = [
            value for line in result.metrics[key].series for _, value in line.points
        ]
        if raw_samples:
            assert values and all(value == 0 for value in values)
        else:
            assert not values
    assert calls


@pytest.mark.asyncio
@pytest.mark.parametrize('phase_id', [None, '128:0'])
async def test_pd_additive_metrics_keep_recorded_owners_phase_bounds_and_priority_totals(
    monkeypatch, phase_id
):
    import re

    from prometheus_client.parser import text_string_to_metric_families

    from gpustack.monitoring import models
    from gpustack.schemas.benchmark_monitoring import BenchmarkRunMember

    history, module = _history(), _module()
    history.model.backend = 'SGLang'
    history.model.disaggregation = SimpleNamespace(mode='sglang-mooncake')
    history.instances = [
        BenchmarkRunMember(
            **{
                **history.snapshot.instances[0].model_dump(),
                'id': identity,
                'name': f'{role}-{identity}',
                'role': role,
                'source_identities': [
                    {
                        **history.snapshot.instances[0].source_identities[0],
                        'model_instance_id': str(identity),
                    }
                ],
            }
        )
        for identity, role in [(12, 'prefill'), (13, 'decode'), (14, 'router')]
    ]
    # This fixed native payload contains a total plus a partition, not three
    # independent populations. Its declared total is seven, not fourteen.
    native = '''# TYPE sglang:num_prefill_bootstrap_queue_reqs gauge
sglang:num_prefill_bootstrap_queue_reqs{priority=""} 7
sglang:num_prefill_bootstrap_queue_reqs{priority="0"} 5
sglang:num_prefill_bootstrap_queue_reqs{priority="1"} 2
'''
    samples = next(text_string_to_metric_families(native)).samples
    family = 'gpustack:pd_prefill_bootstrap_queue_requests'
    requests = []
    labels = {
        **history.instances[0].source_identities[0],
        'model_id': '9',
        'cluster_id': '1',
        'role': 'prefill',
    }

    def total(query):
        selector = re.search(re.escape(family) + r'\{([^}]*)\}', query)
        assert selector is not None
        return sum(
            sample.value
            for sample in samples
            if 'priority=""' not in selector[1]
            or sample.labels.get('priority', '') == ''
        )

    async def instant(client, url, query, at):
        requests.append(('instant', query, at))
        if family not in query:
            return []
        return [{'metric': labels, 'value': [at, str(total(query))]}]

    async def ranged(client, url, query, start, end, step):
        requests.append(('range', query, (start, end, step)))
        if family not in query:
            return []
        return [
            {
                'metric': labels,
                'values': [[start, str(total(query))], [end, str(total(query))]],
            }
        ]

    monkeypatch.setattr(models, '_prometheus_url', lambda: 'http://metrics.invalid')
    monkeypatch.setattr(models, '_query_instant', instant)
    monkeypatch.setattr(models, '_query_range', ranged)
    result = await module.collect_benchmark_model_metrics(
        history, phase_id=phase_id, section='all', client=object()
    )
    assert {
        'ingress_ttft',
        'ingress_e2e',
        'pd_kv_transfer_post',
        'pd_kv_allocation',
        'pd_kv_transfer_bytes',
        'pd_kv_transfer_speed',
        'pd_prefill_bootstrap_queue',
        'pd_prefill_inflight_queue',
        'pd_decode_prealloc_queue',
        'pd_decode_transfer_queue',
        'router_requests',
        'router_request_duration',
    }.issubset(result.metrics)
    metric = result.metrics['pd_prefill_bootstrap_queue']
    assert metric.summaries[0].value == 7
    assert metric.summaries[0].dimensions.role == 'prefill'
    assert result.sample_statistics['pd_prefill_bootstrap_queue'][0].peak == 7
    assert result.metrics['ingress_ttft'].status == 'not_applicable'
    assert result.metrics['ingress_ttft'].reason == 'gateway_bypassed'
    assert result.metrics['pd_kv_transfer'].measurement_scope == 'bootstrap_handshake'
    assert result.metrics['pd_kv_transfer_post'].status == 'unsupported'
    expected = (1000.25, 1040.75) if phase_id else (998.25, 1043.75)
    assert (result.start, result.end) == expected
    for kind, query, bounds in requests:
        if kind == 'range':
            assert bounds[0] >= expected[0] and bounds[1] < expected[1]
        if family in query:
            assert 'role=~"prefill"' in query
            assert 'model_instance_id=~"12"' in query
            assert 'component_attempt=""' in query
            assert 'worker_id="3"' in query
    assert not any('gpustack:service_' in query for _, query, _ in requests)


@pytest.mark.asyncio
@pytest.mark.parametrize('phase_id', [None, '128:0'])
@pytest.mark.parametrize('raw_samples', [True, False])
async def test_short_resource_grid_counts_evaluations_and_never_borrows_old_gauge_samples(
    monkeypatch, phase_id, raw_samples
):
    import math

    module, history = _module(), _history()
    start, end = 1000.000044, 1003.000054
    point = _point(start=start, end=end)
    timing = point.raw_metrics['scheduler_metrics']
    timing['request_start_time'], timing['request_end_time'] = start, end
    history.context.stages, history.context.run_window = module.benchmark_windows(
        [point]
    )
    monkeypatch.setattr(module.prometheus, 'prometheus_url', lambda: 'http://unused')
    identity = AsyncMock()
    instant = AsyncMock(return_value=[])
    monkeypatch.setattr(module.hardware, '_identity_presence', identity)
    monkeypatch.setattr(module.prometheus, 'query_instant', instant)
    calls = []

    async def ranges(client, url, query, query_start, query_end, step):
        calls.append((query_start, query_end, step))
        assert (query_start, query_end, step) == (1000.001, 1003.0, 1)
        assert f'>= {start}' in query
        if not raw_samples or 'worker_gpu_utilization' not in query:
            return []
        # Only an actual scrape inside the interval may supply a gauge value.
        ticks = [
            query_start + n * step
            for n in range(math.floor((query_end - query_start) / step) + 1)
        ]
        return [
            {
                'metric': {
                    'worker_id': '3',
                    'worker_uuid': 'secret-uuid',
                    'owner_principal_id': '4',
                    'gpu_uuid': 'gpu-uuid',
                    'gpu_chip_index': '0',
                },
                'values': [[tick, '0'] for tick in ticks if tick >= start + 1.5],
            }
        ]

    monkeypatch.setattr(module.prometheus, 'query_range', ranges)
    result = await module.collect_benchmark_resource_metrics(
        history, phase_id=phase_id, client=object()
    )
    assert identity.await_args.args[3] == 1003.0
    assert instant.await_count > 0
    assert all(call.args[3] == 1003.0 for call in instant.await_args_list)
    assert result.start == start and result.end == end and result.step == 1
    assert result.available is raw_samples
    metric = result.metrics['gpu_utilization']
    assert metric.summaries[0].value == (0 if raw_samples else None)
    stats = result.sample_statistics['gpu_utilization'][0]
    assert stats.expected_samples == 3
    assert stats.observed_samples == int(raw_samples)
    assert stats.coverage == pytest.approx(1 / 3 if raw_samples else 0)
    assert calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'mode,gateway,expected',
    [
        ('model_instance', None, 'not_applicable'),
        ('route', None, 'no_data'),
        ('route', 'http://gateway.invalid', 'available'),
    ],
)
async def test_benchmark_ingress_requires_its_recorded_gateway_path(
    monkeypatch, mode, gateway, expected
):
    from gpustack.monitoring import models

    module = _module()
    history = _history()
    history.target_mode, history.gateway_url = mode, gateway
    queries = []
    monkeypatch.setattr(models, '_prometheus_url', lambda: 'http://metrics.invalid')

    async def instant(client, url, query, at):
        if 'route_upstream_model_consumer_metric_llm_' not in query:
            return []
        queries.append(query)
        value = 2 if '_count{' in query else 100
        return [{'metric': {'ai_consumer': 'one'}, 'value': [at, str(value)]}]

    async def ranged(*args):
        return []

    monkeypatch.setattr(models, '_query_instant', instant)
    monkeypatch.setattr(models, '_query_range', ranged)
    result = await module.collect_benchmark_model_metrics(history, client=object())
    metric = result.metrics['ingress_ttft']
    assert metric.status == expected
    if expected == 'available':
        assert metric.summaries[0].value == pytest.approx(0.05)
        assert metric.supported_statistics == ['avg']
        assert queries and all('model-9-(12)' in query for query in queries)
    else:
        assert not queries
