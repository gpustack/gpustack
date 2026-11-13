"""Benchmark history endpoints enforce caller and independent resource visibility."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from gpustack.api.auth import get_current_user
from gpustack.api.exceptions import register_handlers
from gpustack.api.tenant import get_tenant_context
from gpustack.monitoring import benchmarks as monitoring
from gpustack.monitoring import models as engine
from gpustack.routes.routes import management_router
from gpustack.schemas.benchmark import Benchmark, BenchmarkSnapshot
from gpustack.schemas.principals import OrgRole, PrincipalType
from gpustack.server.deps import get_session
from tests.monitoring.test_benchmarks import _ctx, _point, _recorded

PATHS = ['/monitoring', '/metrics', '/resource-metrics']


@pytest.fixture
def api(monkeypatch):
    ctx = _ctx()
    ctx.user.id = 22
    ctx.user.is_admin = False
    ctx.user.username = 'reader'
    ctx.user.worker, ctx.user.cluster = None, None
    benchmark = Benchmark(
        id=7,
        name='benchmark',
        model_id=9,
        owner_principal_id=4,
        snapshot=BenchmarkSnapshot(monitoring=_recorded()),
    )
    lookup = AsyncMock(return_value=benchmark)
    model_lookup = AsyncMock(return_value=SimpleNamespace(owner_principal_id=4))
    worker = SimpleNamespace(owner_principal_id=4, worker_uuid='secret-uuid')
    monkeypatch.setattr(monitoring.Benchmark, 'one_by_id', lookup)
    monkeypatch.setattr(
        monitoring.BenchmarkResult, 'all_by_fields', AsyncMock(return_value=[_point()])
    )
    monkeypatch.setattr(monitoring.Model, 'one_by_id', model_lookup)
    monkeypatch.setattr(
        monitoring.ModelInstance,
        'one_by_id',
        AsyncMock(return_value=SimpleNamespace(owner_principal_id=4, model_id=9)),
    )
    monkeypatch.setattr(monitoring.Worker, 'one_by_id', AsyncMock(return_value=worker))
    policy = AsyncMock(return_value={})
    monkeypatch.setattr(monitoring, 'collect_storage_policy', policy)
    monkeypatch.setattr(monitoring.prometheus, 'prometheus_url', lambda: None)
    monkeypatch.setattr(engine, '_prometheus_url', lambda: None)
    app = FastAPI()
    register_handlers(app)
    app.dependency_overrides[get_current_user] = lambda: ctx.user
    app.dependency_overrides[get_tenant_context] = lambda: ctx
    app.dependency_overrides[get_session] = lambda: None
    app.include_router(management_router)
    return SimpleNamespace(
        app=app,
        ctx=ctx,
        benchmark=benchmark,
        lookup=lookup,
        model_lookup=model_lookup,
        worker=worker,
        policy=policy,
    )


@pytest.mark.parametrize('suffix', PATHS)
def test_foreign_benchmark_is_rejected_before_resource_or_policy_lookup(api, suffix):
    api.benchmark.owner_principal_id = 99
    with TestClient(api.app) as client:
        response = client.get('/v2/benchmarks/7' + suffix)
    assert response.status_code == 404
    api.model_lookup.assert_not_awaited()
    api.policy.assert_not_awaited()


@pytest.mark.parametrize('suffix', PATHS)
@pytest.mark.parametrize('principal', ['system', 'member'])
def test_monitoring_requires_user_org_owner_before_loading_data(api, suffix, principal):
    if principal == 'system':
        api.ctx.user.kind = PrincipalType.SYSTEM
    else:
        api.ctx.org_role = OrgRole.MEMBER
    with TestClient(api.app) as client:
        response = client.get('/v2/benchmarks/7' + suffix)
    assert response.status_code == 403
    api.lookup.assert_not_awaited()


def test_context_returns_exact_short_phase_and_sanitized_associations(api):
    with TestClient(api.app) as client:
        response = client.get('/v2/benchmarks/7/monitoring')
    assert response.status_code == 200
    body = response.json()
    assert body['stages'][0]['start'] == 1000.25
    assert body['stages'][0]['end'] == 1040.75
    assert body['run_window'] == {'start': 998.25, 'end': 1043.75}
    assert body['instances'][0]['worker_ids'] == [3]
    assert body['devices'][0]['model_instance_ids'] == [12]
    assert (
        'source_identities' not in response.text and 'secret-uuid' not in response.text
    )


@pytest.mark.parametrize('suffix', ['/metrics', '/resource-metrics'])
def test_phase_queries_allow_exact_short_range(api, suffix):
    with TestClient(api.app) as client:
        response = client.get('/v2/benchmarks/7' + suffix, params={'phase_id': '128:0'})
    assert response.status_code == 200
    body = response.json()
    assert body['start'] == 1000.25 and body['end'] == 1040.75
    assert body['history_scope'] == 'benchmark_snapshot'
    assert body['reason'] == 'prometheus_unavailable'


@pytest.mark.parametrize('suffix', ['/metrics', '/resource-metrics'])
@pytest.mark.parametrize(
    'params',
    [
        {'phase_id': 'unknown'},
        {'start': 1001},
        {'start': 990, 'end': 1040},
        {'phase_id': '128:0', 'start': 1001, 'end': 1041},
    ],
)
def test_queries_reject_unknown_or_out_of_bounds_windows(api, suffix, params):
    with TestClient(api.app) as client:
        response = client.get('/v2/benchmarks/7' + suffix, params=params)
    assert response.status_code in (400, 404)


def test_hidden_resources_return_no_names_or_counts(api):
    api.worker.owner_principal_id = 99
    with TestClient(api.app) as client:
        response = client.get('/v2/benchmarks/7/resource-metrics')
    assert response.status_code == 200
    body = response.json()
    assert body['reason'] == 'resource_not_visible'
    assert body['allocated_gpu_count'] == body['allocated_worker_count'] == 0
    assert body['devices'] == [] and body['workers'] == []
    assert 'secret-worker' not in response.text and 'secret-gpu' not in response.text
