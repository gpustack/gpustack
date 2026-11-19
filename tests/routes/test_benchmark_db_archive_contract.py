from unittest.mock import AsyncMock

from fastapi.testclient import TestClient
import pytest

from gpustack.routes import benchmarks as routes
from gpustack.schemas.benchmark import BenchmarkStateEnum
from tests.routes.test_benchmark_monitoring import api as monitoring_api

api = monitoring_api


@pytest.mark.parametrize(
    'suffix', ('monitoring', 'metrics', 'resource-metrics', 'phase-observations')
)
def test_live_observation_reports_current_archive_state(api, monkeypatch, suffix):
    api.benchmark.run_id = 'test-run'
    api.benchmark.state = BenchmarkStateEnum.RUNNING
    status = AsyncMock(
        return_value={
            'state': 'failed',
            'run_id': 'test-run',
            'content_version': '3',
            'captured_until': 1030,
            'data_coverage': 'partial',
            'expires_at': 2000,
            'retryable': True,
            'last_error': 'capture_failed',
        }
    )
    monkeypatch.setattr(routes, 'archive_status', status)
    with TestClient(api.app) as client:
        result = client.get(f'/v2/benchmarks/7/{suffix}?run_id=test-run')
    assert result.status_code == 200
    body = result.json()
    observations = (
        list(body['phases']['128:0'].values())
        if suffix == 'phase-observations'
        else [body]
    )
    for observation in observations:
        storage = observation['observation_storage']
        assert storage['source'] == 'live'
        assert storage['content_version'] is None
        assert storage['archive_state'] == 'failed'
        assert storage['last_error'] == 'capture_failed'
        assert storage['captured_until'] == 1030
        assert storage['data_coverage'] == 'partial'
        assert storage['expires_at'] == 2000
        assert storage['retryable'] is True
    status.assert_awaited_once_with(None, api.ctx, api.benchmark, run_id='test-run')
    api.lookup.assert_awaited_once_with(None, 7)


def test_archive_content_mismatch_does_not_fall_back_to_live(api, monkeypatch):
    api.benchmark.run_id = 'test-run'
    api.benchmark.state = BenchmarkStateEnum.RUNNING
    monkeypatch.setattr(routes, 'read_archive_metrics', AsyncMock(return_value=None))
    with TestClient(api.app) as client:
        result = client.get('/v2/benchmarks/7/metrics?content_version=42')
    assert result.status_code == 409
    assert result.json()['details']['code'] == 'benchmark_archive_content_changed'
    api.model_lookup.assert_not_awaited()


def test_finished_benchmark_cannot_restart_in_place(api):
    api.benchmark.run_id = 'test-run'
    api.benchmark.state = BenchmarkStateEnum.COMPLETED
    with TestClient(api.app) as client:
        result = client.patch(
            '/v2/benchmarks/7/state', json={'state': 'pending', 'run_id': 'test-run'}
        )
    assert result.status_code == 409
    assert result.json()['details']['code'] == 'benchmark_already_executed'
