"""Completed-count sampling keeps planned cancellations out of failure decisions."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpustack.worker.benchmark import analysis
from gpustack.worker.benchmark.request_stats import success_rate, unplanned_incomplete
from gpustack.worker.benchmark_manager import BenchmarkManager, CollectedResults
from gpustack.worker.schemas.benchmark_runner import (
    GenerativeBenchmarksReport,
    GenerativeRequestStats,
)


def _incomplete_sample(index, start, end, *, error='Request was cancelled'):
    return {
        'request_id': f'cancelled-{index}',
        'info': {
            'status': 'cancelled',
            'error': error,
            'timings': {
                'resolve_start': start,
                'request_start': start,
                'resolve_end': end,
            },
        },
        'input_metrics': {},
        'output_metrics': {},
    }


@pytest.fixture
def report_data():
    path = Path(__file__).parents[1] / 'fixtures/benchmark/runner_min_requests.json'
    raw = json.loads(path.read_text())
    point = raw['benchmarks'][0]
    stop = point['end_time'] + 0.001
    point['scheduler_state']['end_processing_time'] = stop
    count = point['metrics']['request_totals']['incomplete']
    point['requests_truncated']['incomplete'] = [
        _incomplete_sample(i, point['start_time'], stop + 0.001) for i in range(count)
    ]
    return raw


def test_completed_count_metadata_preserves_original_counts(report_data):
    report = GenerativeBenchmarksReport.model_validate(report_data)
    row = report.to_results()[0]
    original = report_data['benchmarks'][0]['metrics']['request_totals']
    assert row['request_total'] == original['total']
    assert row['request_successful'] == original['successful']
    assert row['request_incomplete'] == original['incomplete']
    assert row['request_errored'] == 0
    assert row['raw_metrics']['termination'] == {
        'reason': 'min_requests',
        'requested': 50,
        'processed': 50,
        'cancelled_at_boundary': original['incomplete'],
    }


def test_mixed_incomplete_requests_still_affect_success_rate(report_data):
    point = report_data['benchmarks'][0]
    samples = point['requests_truncated']['incomplete']
    # A request cancelled before the stop and a request with an actual failure
    # still count, even when the point later reaches its sample target.
    samples[0]['info']['timings']['resolve_end'] = point['end_time']
    samples[1]['info']['traceback'] = 'Connection reset'
    # A cancelled request can already have a completed transport response.
    samples[2]['info']['timings']['request_end'] = point['end_time']
    # A request can start after the metrics window closes and still be
    # cancelled by the sampling stop before the run ends.
    stop = point['scheduler_state']['end_processing_time']
    samples.append(
        _incomplete_sample('outside-window', point['end_time'] + 0.0005, stop + 0.001)
    )

    report = GenerativeBenchmarksReport.model_validate(report_data)
    row = report.to_results()[0]
    assert row['raw_metrics']['termination']['cancelled_at_boundary'] == 8
    assert report.to_metrics().raw_metrics['termination']['cancelled_at_boundary'] == 8
    assert unplanned_incomplete(row) == 3
    assert success_rate(row) == pytest.approx(54 / 57)

    manager = object.__new__(BenchmarkManager)
    errors, incomplete = manager._load_request_samples(report)
    assert errors == []
    assert [sample.request_id for sample in incomplete] == [
        'cancelled-0',
        'cancelled-1',
        'cancelled-2',
    ]


@pytest.mark.parametrize('companion', ['matching', 'different', 'unavailable'])
def test_truncated_summary_uses_only_available_boundary_evidence(
    report_data, tmp_path, companion
):
    point = report_data['benchmarks'][0]
    totals = point['metrics']['request_totals']
    totals.update(incomplete=26, total=80)
    stop = point['scheduler_state']['end_processing_time']
    samples = [
        _incomplete_sample(i, point['start_time'], stop + 0.001) for i in range(25)
    ]
    samples.insert(
        0, _incomplete_sample('failure', point['start_time'], point['end_time'])
    )
    point['requests_truncated']['incomplete'] = samples[:20]
    summary_path = tmp_path / '12__p0.json'
    summary_path.write_text(json.dumps(report_data))
    if companion != 'unavailable':
        point['requests'] = {'incomplete': samples}
        if companion == 'different':
            point['scheduler_state']['end_processing_time'] += 10
        (tmp_path / '12__p0.full.json').write_text(json.dumps(report_data))

    report = GenerativeBenchmarksReport.load_file(summary_path)
    row = report.to_results()[0]
    expected = 25 if companion == 'matching' else 19
    assert row['raw_metrics']['termination']['cancelled_at_boundary'] == expected
    assert unplanned_incomplete(row) == 26 - expected
    assert len(row['raw_metrics']['requests_truncated']['incomplete']) == 20


def test_healthy_completed_count_point_meets_slo_and_has_no_failure_warning(
    report_data,
):
    report = GenerativeBenchmarksReport.model_validate(report_data)
    rows = report.to_results()
    benchmark = SimpleNamespace(
        auto_tune=True, slo_avg_ttft_ms=10000.0, max_points=18, upper_bound=16
    )
    assert analysis.success_ok(rows[0])
    primary, _ = analysis.select_result_points(benchmark, rows)
    assert primary['rate'] == 16
    best = analysis.compute_best_points(benchmark, rows)
    validity = analysis.compute_validity(benchmark, rows, best)
    codes = {warning['code'] for warning in validity['warnings']}
    assert 'point_high_error' not in codes
    assert 'slo_never_met' not in codes


@pytest.mark.parametrize('reason', ['max_duration', 'max_errors', 'requests_exhausted'])
def test_other_stop_reasons_keep_incomplete_requests_in_success_rate(
    report_data, reason
):
    report_data['benchmarks'][0]['scheduler_state']['end_processing_constraints'] = {
        reason: {
            'request_processing': 'stop_local',
            'metadata': {'processed_requests': 50},
        }
    }
    row = GenerativeBenchmarksReport.model_validate(report_data).to_results()[0]
    assert not analysis.success_ok(row)
    assert 'cancelled_at_boundary' not in row['raw_metrics']['termination']
    assert row['raw_metrics']['termination']['requested'] == 50


def test_unconfirmed_completed_target_does_not_exempt_cancellations(report_data):
    action = report_data['benchmarks'][0]['scheduler_state'][
        'end_processing_constraints'
    ]['min_requests']
    action['metadata']['processed_exceeded'] = False
    row = GenerativeBenchmarksReport.model_validate(report_data).to_results()[0]
    assert not analysis.success_ok(row)


def test_parent_termination_works_for_a_point_without_metrics(report_data):
    report = GenerativeBenchmarksReport.model_validate(report_data)
    report.benchmarks.append(report.benchmarks[0].model_copy(update={'metrics': None}))

    raw = report.to_metrics().raw_metrics
    assert raw['termination']['cancelled_at_boundary'] > 0
    assert raw['benchmarks'][1]['termination'] == {
        'reason': 'min_requests',
        'requested': 50,
        'processed': 50,
    }


def test_created_count_reports_retain_their_failure_accounting(report_data):
    state = report_data['benchmarks'][0]['scheduler_state']
    for key in (
        'scheduler_constraints',
        'end_queuing_constraints',
        'end_processing_constraints',
    ):
        actions = state[key]
        action = actions.pop('min_requests')
        action['metadata']['max_requests'] = action['metadata'].pop('min_requests')
        actions['max_requests'] = action
    row = GenerativeBenchmarksReport.model_validate(report_data).to_results()[0]
    assert row['raw_metrics']['termination']['reason'] == 'max_requests'
    assert row['raw_metrics']['termination']['requested'] == 50
    assert 'cancelled_at_boundary' not in row['raw_metrics']['termination']
    assert not analysis.success_ok(row)


@pytest.mark.parametrize(
    'processed_exceeded,has_stop_time,keeps_boundary_sample',
    [(True, True, False), (False, True, True), (True, False, True)],
)
def test_only_boundary_cancellations_are_omitted_from_failure_samples(
    report_data, processed_exceeded, has_stop_time, keeps_boundary_sample
):
    report = GenerativeBenchmarksReport.model_validate(report_data)
    point = report.benchmarks[0]
    point.scheduler_state['end_processing_constraints']['min_requests']['metadata'][
        'processed_exceeded'
    ] = processed_exceeded
    stop_time = 1794538109.196
    if has_stop_time:
        point.scheduler_state['end_processing_time'] = stop_time
    else:
        point.scheduler_state.pop('end_processing_time', None)

    def sample(name, status, resolve_end, error):
        return GenerativeRequestStats.model_validate(
            {
                'request_id': name,
                'info': {
                    'status': status,
                    'timings': {'resolve_end': resolve_end},
                    'error': error,
                },
                'input_metrics': {},
                'output_metrics': {},
            }
        )

    point.requests_truncated.errored = [
        sample('errored', 'errored', stop_time + 1, 'HTTP 429')
    ]
    point.requests_truncated.incomplete = [
        sample('boundary', 'cancelled', stop_time + 1, 'Request was cancelled'),
        sample('before', 'cancelled', stop_time - 1, 'Request was cancelled'),
        sample('unknown', 'cancelled', None, 'Request was cancelled'),
        sample('failure', 'cancelled', stop_time + 1, 'Upstream disconnected'),
    ]

    manager = object.__new__(BenchmarkManager)
    errors, incomplete = manager._load_request_samples(report)
    assert [sample.request_id for sample in errors] == ['errored']
    expected = ['before', 'unknown', 'failure']
    if keeps_boundary_sample:
        expected.insert(0, 'boundary')
    assert [sample.request_id for sample in incomplete] == expected


def test_server_errors_still_fail_the_success_floor(report_data):
    totals = report_data['benchmarks'][0]['metrics']['request_totals']
    totals['successful'] -= 4
    totals['errored'] += 4
    row = GenerativeBenchmarksReport.model_validate(report_data).to_results()[0]
    assert not analysis.success_ok(row)


@pytest.mark.parametrize('cancelled', [-1, True, 11.5, 1000])
def test_invalid_exclusion_metadata_does_not_improve_success_rate(cancelled):
    row = {
        'request_total': 65,
        'request_successful': 54,
        'request_errored': 0,
        'request_incomplete': 11,
        'raw_metrics': {
            'termination': {
                'reason': 'min_requests',
                'cancelled_at_boundary': cancelled,
            }
        },
    }
    assert not analysis.success_ok(row)


def test_parent_metrics_retain_representative_termination(report_data):
    report = GenerativeBenchmarksReport.model_validate(report_data)
    metrics = report.to_metrics()
    termination = metrics.raw_metrics['termination']
    assert termination['reason'] == 'min_requests'
    assert termination['cancelled_at_boundary'] == metrics.request_incomplete
    assert 'scheduler_state' not in metrics.raw_metrics['benchmarks'][0]
    assert metrics.raw_metrics['benchmarks'][0]['termination'] == termination


@pytest.mark.parametrize('with_results', [True, False])
@pytest.mark.parametrize('errored', [0, 4])
def test_completion_messages_exclude_only_planned_cancellations(
    report_data, with_results, errored
):
    totals = report_data['benchmarks'][0]['metrics']['request_totals']
    totals['successful'] -= errored
    totals['errored'] += errored
    report = GenerativeBenchmarksReport.model_validate(report_data)
    metrics = report.to_metrics()
    rows = report.to_results() if with_results else []
    manager = object.__new__(BenchmarkManager)
    collected = CollectedResults(
        results=rows, metrics=metrics, report=report, loaded=1, skipped=0
    )
    messages = []
    uploaded = []
    manager._collect_results = lambda benchmark: collected
    manager._load_request_samples = lambda *args, **kwargs: ([], [])
    manager._log_request_failures_if_any = lambda **kwargs: None
    manager._post_metrics_and_results = lambda benchmark, m, results: uploaded.append(m)
    manager._write_best_points_and_validity = lambda *args, **kwargs: None
    manager._retry_sync = lambda operation, **kwargs: operation()
    manager._update_benchmark_state_sync = (
        lambda benchmark_id, **kwargs: messages.append(kwargs['state_message'])
    )

    manager._sync_benchmark_metrics(SimpleNamespace(id=1, name='healthy'))

    assert uploaded[0].request_incomplete == totals['incomplete']
    if errored:
        assert len(messages) == 1
        assert 'errored=4, incomplete=0' in messages[0]
    else:
        assert messages == []
