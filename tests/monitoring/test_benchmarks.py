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
    monkeypatch.setattr(
        module.ModelInstance,
        'one_by_id',
        AsyncMock(return_value=SimpleNamespace(owner_principal_id=4, model_id=9)),
    )
    monkeypatch.setattr(
        module.Worker,
        'one_by_id',
        AsyncMock(
            return_value=SimpleNamespace(
                owner_principal_id=8, worker_uuid='secret-uuid'
            )
        ),
    )
    monkeypatch.setattr(module, 'collect_storage_policy', AsyncMock(return_value={}))
    history = await module.load_benchmark_history(None, _ctx(), 7)
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
async def test_hidden_per_source_observation_is_restricted_without_source_count(
    monkeypatch,
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
    result = await module.collect_benchmark_model_metrics(history, phase_id='128:0')
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
        module.ModelInstance, 'one_by_id', AsyncMock(return_value=current)
    )
    monkeypatch.setattr(
        module.Worker,
        'one_by_id',
        AsyncMock(
            return_value=SimpleNamespace(
                worker_uuid='other-uuid' if change == 'worker_uuid' else 'secret-uuid',
                owner_principal_id=4,
            )
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
async def test_worker_epoch_change_does_not_mix_engine_history(monkeypatch):
    module, history = _module(), _history()
    history.context.available = False
    history.context.reason = 'historical_association_changed'
    collector = AsyncMock()
    monkeypatch.setattr(module, 'collect_model_metrics', collector)
    result = await module.collect_benchmark_model_metrics(history, phase_id='128:0')
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
        module.ModelInstance, 'one_by_id', AsyncMock(return_value=member)
    )
    monkeypatch.setattr(module.Worker, 'one_by_id', AsyncMock(return_value=worker))
    verified = await module.verify_benchmark_run(
        Benchmark(name='b', snapshot=snapshot), None, confirm=True
    )
    assert verified.monitoring.confirmed_at is not None
    assert [e.kind for e in verified.monitoring.events] == ['association_captured']
