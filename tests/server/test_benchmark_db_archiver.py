import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gpustack.schemas.benchmark_monitoring import BenchmarkModelMetrics
from gpustack.server import benchmark_monitoring_archiver as worker


@pytest.mark.asyncio
@pytest.mark.parametrize('failing_operation', ('claim', 'fail', 'release'))
async def test_archive_job_database_failure_does_not_cancel_sibling(
    monkeypatch, caplog, failing_operation
):
    job = worker.BenchmarkMonitoringArchiver(SimpleNamespace())
    first = SimpleNamespace(benchmark_id=1)
    second = SimpleNamespace(benchmark_id=2)

    async def claim(benchmark_id, *args, **kwargs):
        if benchmark_id == 1 and failing_operation == 'claim':
            raise OSError('database unavailable')
        return first if benchmark_id == 1 else second

    async def capture(lease):
        if lease is first and failing_operation == 'fail':
            raise RuntimeError('capture failed')

    async def release(lease):
        if lease is first and failing_operation == 'release':
            raise OSError('database unavailable')

    capture_mock = AsyncMock(side_effect=capture)
    release_mock = AsyncMock(side_effect=release)
    monkeypatch.setattr(job.store, 'claim', claim)
    monkeypatch.setattr(job, '_capture', capture_mock)
    monkeypatch.setattr(job.store, 'fail', AsyncMock(side_effect=OSError))
    monkeypatch.setattr(job.store, 'release', release_mock)
    async with asyncio.TaskGroup() as tasks:
        tasks.create_task(job._execute(1))
        tasks.create_task(job._execute(2))

    capture_mock.assert_any_await(second)
    release_mock.assert_any_await(second)
    if failing_operation != 'claim':
        release_mock.assert_any_await(first)
    assert 'OSError' in caplog.text


def histogram(start=1000, end=1015):
    return BenchmarkModelMetrics(
        model_id=9,
        start=start,
        end=end,
        step=5,
        rate_window_seconds=75,
        last_refreshed_at=end,
        history_scope='benchmark_snapshot',
        metrics={
            'ttft': dict(
                kind='histogram',
                unit='ms',
                measurement_scope='engine',
                status='available',
                summaries=[dict(statistic='p99', value=67)],
                series=[
                    dict(
                        name='P99',
                        statistic='p99',
                        points=[(1000, 0), (1005, None), (1010, 30)],
                        lower_bounds=[(1005, 500)],
                    )
                ],
            )
        },
    )


def test_merge_capture_preserves_exact_and_censored_observations():
    previous, current = histogram(), histogram()
    metric = current.metrics['ttft']
    metric.series[0].points = [(1000, None), (1005, None), (1010, 31)]
    metric.series[0].lower_bounds = [(1005, 499)]
    metric.summaries = []
    result = worker._merge_scope(current, previous, 1000, 1015)
    line = result.metrics['ttft'].series[0]
    assert line.points == [(1000, 0), (1005, None), (1010, 31)]
    assert line.lower_bounds == [(1005, 500)]
    assert result.metrics['ttft'].summaries[0].value == 67


def test_merge_keeps_range_observations_but_not_summary_from_old_window():
    previous, current = histogram(), histogram(end=1020)
    current.metrics['ttft'].series[0].points.append((1015, 123))
    current.metrics['ttft'].summaries = []
    result = worker._merge_scope(current, previous, 1000, 1020)
    assert len(result.metrics['ttft'].series[0].points) == 4
    assert result.metrics['ttft'].summaries == []


@pytest.mark.asyncio
async def test_run_tail_uses_frozen_grid_and_native_summary(monkeypatch):
    job = worker.BenchmarkMonitoringArchiver(SimpleNamespace())
    key = 'run/runtime/all/aggregate'
    previous = histogram()
    tail = histogram(start=1015, end=1025)
    tail.metrics['ttft'].series[0].points = [(1015, 42), (1020, 44)]
    tail.metrics['ttft'].series[0].lower_bounds = None
    native = histogram(end=1025)
    native.metrics['ttft'].series = []
    native.metrics['ttft'].summaries[0].value = 79
    collect = AsyncMock(side_effect=[{key: tail}, {key: native}])
    monkeypatch.setattr(job, '_collect_scope', collect)
    history = SimpleNamespace(context=SimpleNamespace(events=[]))
    plan = dict(step=5, rate_window=75)
    scopes, errors = await job._capture_run(
        history,
        plan,
        {key: previous.model_dump(mode='json')},
        start=1000,
        end=1025,
        cursor=1015,
        client=None,
    )
    assert collect.await_args_list[0].kwargs['start'] == 1015
    assert collect.await_args_list[0].args[3]['anchor'] == 1000
    assert collect.await_args_list[1].kwargs['summary_only'] is True
    value = scopes[key]
    assert not errors
    assert value.metrics['ttft'].summaries[0].value == 79
    assert value.metrics['ttft'].series[0].points == [
        (1000, 0),
        (1005, None),
        (1010, 30),
        (1015, 42),
        (1020, 44),
    ]


def test_clean_capture_with_sparse_samples_is_ready_with_partial_coverage():
    response = histogram()
    response.metrics['ttft'].series[0].points = [(1000, 0), (1005, None)]
    response.metrics['ttft'].series[0].lower_bounds = None
    state, coverage, complete = worker._archive_outcome(
        {'run/runtime/all/aggregate': response}, set(), True
    )
    assert (state, coverage, complete) == ('ready', 'partial', True)


def test_failed_capture_can_be_sealed_with_valid_saved_values():
    state, coverage, complete = worker._archive_outcome(
        {'run/runtime/all/aggregate': histogram()}, {'run/runtime/all/aggregate'}, True
    )
    assert (state, coverage, complete) == ('partial', 'partial', False)


def test_archive_preserves_observations_before_the_first_generation_change():
    response = histogram()
    history = SimpleNamespace(
        context=SimpleNamespace(
            events=[
                SimpleNamespace(kind='generation_changed', at=1012),
                SimpleNamespace(kind='generation_changed', at=1007),
            ]
        )
    )
    archived = worker.BenchmarkMonitoringArchiver._finish_scope(
        'run/runtime/all/aggregate', response, history
    )
    metric = archived.metrics['ttft']
    assert archived.available
    assert metric.reason == 'historical_association_changed'
    assert metric.series[0].points == [(1000, 0), (1005, None), (1010, None)]
    assert metric.series[0].lower_bounds == [(1005, 500)]
    assert metric.summaries[0].value is None


@pytest.mark.asyncio
async def test_repair_can_rebuild_corrupt_bundle_without_publishing_empty(monkeypatch):
    job = worker.BenchmarkMonitoringArchiver(SimpleNamespace())
    lease = SimpleNamespace(
        repair_requested=True, finalize_requested_at=900, benchmark_id=1
    )
    record = SimpleNamespace(
        snapshot={},
        payload=b'corrupt',
        payload_sha256='bad',
        uncompressed_bytes=100,
        expires_at=None,
    )
    monkeypatch.setattr(job, '_load_capture', AsyncMock(return_value=(record, 1000)))
    bundle = {'context': {}, 'scopes': {}}
    prepare = AsyncMock(
        side_effect=[
            (bundle, None, 'no_data', 'none', True),
            (bundle, 950, 'ready', 'complete', True),
        ]
    )
    monkeypatch.setattr(job, '_prepare', prepare)
    publish = AsyncMock(return_value=True)
    monkeypatch.setattr(job.store, 'publish', publish)
    with pytest.raises(worker.BenchmarkPayloadError, match='archive_payload_corrupt'):
        await job._capture(lease)
    publish.assert_not_awaited()
    await job._capture(lease)
    assert prepare.await_args.args[1] == {}
    assert publish.await_args.kwargs['sealed'] is True
    assert publish.await_args.kwargs['capture_complete'] is True


@pytest.mark.asyncio
async def test_terminal_capture_waits_for_late_samples_then_clamps_to_measured_window(
    monkeypatch,
):
    job = worker.BenchmarkMonitoringArchiver(SimpleNamespace())
    lease = SimpleNamespace(repair_requested=False, finalize_requested_at=1069)
    context = SimpleNamespace(
        run_window=SimpleNamespace(start=1000, end=1040), events=[], stages=[]
    )
    record = SimpleNamespace(
        snapshot={
            'monitoring': {'confirmed_at': 990},
            'state': 'completed',
        },
        plan={'step': 5, 'rate_window': 75},
        payload=None,
        expires_at=None,
        captured_until=1050,
    )
    monkeypatch.setattr(job, '_load_capture', AsyncMock(return_value=(record, 1070)))
    release = AsyncMock()
    monkeypatch.setattr(job.store, 'release', release)
    publish = AsyncMock(return_value=True)
    monkeypatch.setattr(job.store, 'publish', publish)
    await job._capture(lease)
    release.assert_awaited_once_with(lease, retry_seconds=29)
    publish.assert_not_awaited()

    job._load_capture.return_value = (record, 1100)
    previous = {'context': {'run_window': {'start': 1000, 'end': 1050}}, 'scopes': {}}
    capture_run = AsyncMock(return_value=({}, set()))
    monkeypatch.setattr(job, '_capture_run', capture_run)
    _, _, until = await job._capture_run_window(
        SimpleNamespace(context=context),
        record.plan,
        record,
        previous,
        1100,
        True,
        False,
        None,
    )
    assert until == 1040
    assert capture_run.await_args.kwargs['cursor'] == 1000


def test_evaluation_range_excludes_end_and_skips_tail_without_grid_ticks():
    assert worker._evaluation_range(1011.25, 1014.9, 1000.25, 5) is None
    assert worker._evaluation_range(1011.25, 1015.25, 1000.25, 5) is None
    interval = worker._evaluation_range(1011.25, 1015.2501, 1000.25, 5)
    assert interval.start == interval.end == 1015.25


def test_restricted_metric_does_not_restore_old_values():
    previous, response = histogram(), histogram()
    current = response.metrics['ttft']
    current.status = 'restricted'
    current.series = []
    current.summaries = []
    output = worker._merge_scope(response, previous, 1000, 1015).metrics['ttft']
    assert output.status == 'restricted'
    assert not output.series and not output.summaries


def test_max_scopes_accommodates_max_benchmark_result_points():
    from gpustack.routes.benchmarks import MAX_BENCHMARK_RESULT_POINTS

    assert worker._MAX_SCOPES >= 9 * (1 + MAX_BENCHMARK_RESULT_POINTS)
    # Check that 227 stages (and up to 500) fit cleanly within _MAX_SCOPES
    assert 9 * (1 + 227) <= worker._MAX_SCOPES
    assert 9 * (1 + 500) <= worker._MAX_SCOPES


@pytest.mark.asyncio
async def test_archive_scope_budget_exceeded_fails_without_infinite_retry(monkeypatch):
    job = worker.BenchmarkMonitoringArchiver(SimpleNamespace())
    lease = SimpleNamespace(
        benchmark_id=1,
        run_id='run-1',
        worker_id='w-1',
        fencing_token=1,
        write_seq=0,
        snapshot_seq=0,
        repair_requested=False,
        finalize_requested_at=1000,
    )

    async def mock_capture_scope_failure(_lease):
        raise worker.BenchmarkPayloadError('archive_scope_budget_exceeded')

    fail_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(job.store, 'claim', AsyncMock(return_value=lease))
    monkeypatch.setattr(job, '_capture', mock_capture_scope_failure)
    monkeypatch.setattr(job.store, 'fail', fail_mock)

    await job._execute(1)

    fail_mock.assert_awaited_once()
    assert fail_mock.await_args.args[1] == 'archive_scope_budget_exceeded'
    assert fail_mock.await_args.kwargs.get('retry_seconds') is None


@pytest.mark.asyncio
async def test_archive_scope_budget_exceeded_rejects_repair(monkeypatch):
    store = worker.BenchmarkPayloadStore()
    record = SimpleNamespace(
        benchmark_id=1,
        run_id='run-1',
        sealed=True,
        last_error='archive_scope_budget_exceeded',
        repair_requested=False,
    )

    async def mock_locked_payload(session, benchmark_id):
        return record

    async def mock_database_now(session):
        return 1000.0

    monkeypatch.setattr('gpustack.monitoring.benchmark_payload.locked_payload', mock_locked_payload)
    monkeypatch.setattr('gpustack.monitoring.benchmark_payload.database_now', mock_database_now)
    monkeypatch.setattr('gpustack.monitoring.benchmark_payload._available', lambda rec, now: True)

    session = SimpleNamespace()
    eligible = await store.request_repair(session, 1, 'run-1')
    assert eligible is False
