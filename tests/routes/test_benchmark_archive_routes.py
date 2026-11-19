"""Archived observations remain available without their live deployment."""

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from gpustack.routes import benchmarks as routes
from gpustack.schemas.benchmark_monitoring import (
    BenchmarkModelMetrics,
    BenchmarkMonitoringContext,
    BenchmarkResourceMetrics,
)
from gpustack.schemas.model_metrics import ModelMonitoringCapabilities
from gpustack.schemas.benchmark import BenchmarkStateEnum
from tests.routes.test_benchmark_monitoring import api as monitoring_api

api = monitoring_api


def _context():
    return BenchmarkMonitoringContext(
        benchmark_id=7,
        model_id=9,
        available=True,
        capabilities=ModelMonitoringCapabilities(model_id=9),
    )


@pytest.mark.parametrize(
    'path,reader,response',
    [
        ('/monitoring', 'read_archive_context', _context()),
        (
            '/metrics',
            'read_archive_metrics',
            BenchmarkModelMetrics(
                model_id=9,
                start=1000,
                end=1010,
                step=1,
                rate_window_seconds=75,
                last_refreshed_at=1011,
                history_scope='benchmark_snapshot',
                available=True,
            ),
        ),
        (
            '/resource-metrics',
            'read_archive_resources',
            BenchmarkResourceMetrics(
                model_id=9,
                start=1000,
                end=1010,
                step=1,
                last_refreshed_at=1011,
                history_scope='benchmark_snapshot',
                available=True,
            ),
        ),
    ],
)
def test_archive_is_read_before_current_model_lookup(
    api, monkeypatch, path, reader, response
):
    api.benchmark.state = BenchmarkStateEnum.COMPLETED
    api.model_lookup.return_value = None
    read = AsyncMock(return_value=response)
    monkeypatch.setattr(routes, reader, read)
    with TestClient(api.app) as client:
        result = client.get('/v2/benchmarks/7' + path)
    assert result.status_code == 200
    assert result.json()['available'] is True
    api.model_lookup.assert_not_awaited()
    read.assert_awaited_once()


def test_foreign_archive_is_rejected_before_storage(api, monkeypatch):
    api.benchmark.owner_principal_id = 99
    read = AsyncMock(return_value=_context())
    monkeypatch.setattr(routes, 'read_archive_context', read)
    with TestClient(api.app) as client:
        result = client.get('/v2/benchmarks/7/monitoring')
    assert result.status_code == 404
    read.assert_not_awaited()


def test_unavailable_content_version_never_falls_back_to_live(api, monkeypatch):
    monkeypatch.setattr(routes, 'read_archive_context', AsyncMock(return_value=None))
    with TestClient(api.app) as client:
        result = client.get('/v2/benchmarks/7/monitoring?content_version=4')
    assert result.status_code == 409
    api.model_lookup.assert_not_awaited()


def test_old_execution_cannot_overwrite_replacement_results(api):
    api.benchmark.run_id = 'new-run'
    with TestClient(api.app) as client:
        result = client.post('/v2/benchmarks/7/results?run_id=old-run', json=[])
    assert result.status_code == 409


def test_missing_execution_token_cannot_overwrite_identified_results(api):
    api.benchmark.run_id = 'new-run'
    with TestClient(api.app) as client:
        result = client.post('/v2/benchmarks/7/results', json=[])
    assert result.status_code == 409


def test_live_context_exposes_failed_job_even_before_first_publication(
    api, monkeypatch
):
    api.benchmark.run_id = 'new-run'
    api.benchmark.state = BenchmarkStateEnum.COMPLETED
    monkeypatch.setattr(routes, 'read_archive_context', AsyncMock(return_value=None))
    monkeypatch.setattr(
        routes,
        'archive_status',
        AsyncMock(
            return_value={
                'state': 'failed',
                'run_id': 'new-run',
                'content_version': None,
                'last_error': 'storage_capacity_exceeded',
            }
        ),
    )
    with TestClient(api.app) as client:
        result = client.get('/v2/benchmarks/7/monitoring')
    assert result.status_code == 200
    assert result.json()['observation_storage']['archive_state'] == 'failed'
    assert (
        result.json()['observation_storage']['last_error']
        == 'storage_capacity_exceeded'
    )
