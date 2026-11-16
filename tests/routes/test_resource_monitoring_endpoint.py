from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlparse

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from gpustack.api.auth import get_current_user
from gpustack.api.exceptions import register_handlers
from gpustack.api.tenant import get_tenant_context
from gpustack.routes import resource_monitoring as routes
from gpustack.routes.routes import management_router
from gpustack.schemas.clusters import Cluster
from gpustack.schemas.principals import OrgRole, PrincipalType
from gpustack.schemas.resource_metrics import ResourceMetrics
from gpustack.server.deps import get_session
from tests.monitoring.test_hardware import _context, _worker

PATHS = [
    f'/v2/{resource}/{identifier}/{suffix}'
    for resource, identifier in [('workers', '7'), ('gpu-devices', 'gpu-7')]
    for suffix in (
        'detail',
        'metrics',
        'monitoring-capabilities',
        'monitoring-dashboard',
    )
]


@pytest.fixture
def resource_api(monkeypatch):
    worker, ctx = _worker(), _context()
    worker.status.gpu_devices[0].core.total = 80
    for name, value in dict(
        hostname='worker',
        ip='192.0.2.7',
        ifname='eth0',
        port=10150,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    ).items():
        setattr(worker, name, value)
    for name, value in dict(
        is_admin=False, username='reader', worker=None, cluster=None
    ).items():
        setattr(ctx.user, name, value)
    row = SimpleNamespace(
        id='gpu-7',
        worker_id=7,
        index=8,
        uuid='GPU-private',
        device_chip_index=2,
        owner_principal_id=17,
    )
    lookup = AsyncMock(return_value=worker)
    monkeypatch.setattr(routes.Worker, 'one_by_id', lookup)
    monkeypatch.setattr(routes.GPUDevice, 'one_by_id', AsyncMock(return_value=row))
    monkeypatch.setattr(
        routes.Cluster,
        'one_by_id',
        AsyncMock(return_value=Cluster(id=1, name='cluster', owner_principal_id=17)),
    )
    allocated = AsyncMock(return_value=(0, {}))
    monkeypatch.setattr(routes.worker_routes, '_lookup_allocated', allocated)
    metrics = AsyncMock(
        return_value=ResourceMetrics(
            worker_id=7, start=1000, end=1600, step=15, last_refreshed_at=1600
        )
    )
    policy = AsyncMock(return_value={})
    monkeypatch.setattr(routes, 'collect_resource_metrics', metrics)
    monkeypatch.setattr(routes, 'collect_storage_policy', policy)
    monkeypatch.setattr(
        routes,
        'get_global_config',
        lambda: SimpleNamespace(
            grafana_url='https://grafana.example.test',
            grafana_worker_dashboard_uid='worker',
            get_grafana_url=lambda: 'https://grafana.example.test',
        ),
    )
    session = SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(all=lambda: []))
    )
    app = FastAPI()
    register_handlers(app)
    app.dependency_overrides[get_current_user] = lambda: ctx.user
    app.dependency_overrides[get_tenant_context] = lambda: ctx
    app.dependency_overrides[get_session] = lambda: session
    app.include_router(management_router)
    return SimpleNamespace(
        app=app,
        worker=worker,
        ctx=ctx,
        row=row,
        lookup=lookup,
        metrics=metrics,
        policy=policy,
        allocated=allocated,
        session=session,
    )


@pytest.mark.parametrize('path', PATHS)
def test_resource_http_routes_reject_foreign_worker_before_collection(
    resource_api, path
):
    api = resource_api
    api.worker.owner_principal_id = 99
    with TestClient(api.app) as client:
        response = client.get(path, follow_redirects=False)
    assert response.status_code == 404
    api.metrics.assert_not_awaited()
    api.policy.assert_not_awaited()
    api.allocated.assert_not_awaited()
    api.session.exec.assert_not_awaited()


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize('principal', ['system', 'unresolved'])
def test_resource_http_routes_reject_invalid_principals_before_lookup(
    resource_api, path, principal
):
    api = resource_api
    if principal == 'system':
        api.ctx.user.kind = PrincipalType.SYSTEM
    else:
        api.ctx.current_principal_id = None
    with TestClient(api.app) as client:
        response = client.get(path, follow_redirects=False)
    assert response.status_code == 403
    api.lookup.assert_not_awaited()
    api.metrics.assert_not_awaited()
    api.policy.assert_not_awaited()


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize('role', [OrgRole.OWNER, OrgRole.MEMBER])
def test_org_user_can_read_visible_hardware_routes(resource_api, path, role):
    resource_api.ctx.org_role = role
    with TestClient(resource_api.app) as client:
        response = client.get(path, follow_redirects=False)
    assert response.status_code == (
        302 if path.endswith('monitoring-dashboard') else 200
    )


def test_member_detail_does_not_enumerate_deployments_or_grant_edit(resource_api):
    api = resource_api
    api.ctx.org_role = OrgRole.MEMBER
    with TestClient(api.app) as client:
        response = client.get('/v2/workers/7/detail')
    assert response.status_code == 200
    assert response.json()['instances'] == []
    assert not response.json()['can_edit']
    api.session.exec.assert_not_awaited()


@pytest.mark.parametrize(
    'suffix', ['detail', 'metrics', 'monitoring-capabilities', 'monitoring-dashboard']
)
def test_gpu_http_routes_reject_replaced_inventory_identity(resource_api, suffix):
    api = resource_api
    api.row.uuid = 'old-device'
    with TestClient(api.app) as client:
        response = client.get(
            f'/v2/gpu-devices/gpu-7/{suffix}',
            params={'start': 1730000000.25, 'end': 1730000040.75},
            follow_redirects=False,
        )
    assert response.status_code == 404
    api.metrics.assert_not_awaited()
    api.policy.assert_not_awaited()
    api.allocated.assert_not_awaited()


@pytest.mark.parametrize('resource', ['workers/7', 'gpu-devices/gpu-7'])
@pytest.mark.parametrize('suffix', ['metrics', 'monitoring-dashboard'])
@pytest.mark.parametrize(
    'start,end,seconds,start_ms,end_ms',
    [
        (1730000000.123, 1730000000.124, 1, '1730000000123', '1730000000124'),
        (1730000000.25, 1730000040.75, 41, '1730000000250', '1730000040750'),
    ],
)
def test_resource_http_ranges_keep_precise_phase_boundaries(
    resource_api, resource, suffix, start, end, seconds, start_ms, end_ms
):
    api = resource_api
    with TestClient(api.app) as client:
        response = client.get(
            f'/v2/{resource}/{suffix}',
            params={'window': 'invalid', 'start': start, 'end': end},
            follow_redirects=False,
        )
    if suffix == 'metrics':
        assert response.status_code == 200
        kwargs = api.metrics.await_args.kwargs
        assert (kwargs['start'], kwargs['end'], kwargs['window_seconds']) == (
            start,
            end,
            seconds,
        )
        if resource.startswith('gpu-devices/'):
            from gpustack.monitoring.resources import inventory_devices

            assert kwargs['device_id'] == inventory_devices(api.worker)[0].public.id
    else:
        assert response.status_code == 302
        query = parse_qs(urlparse(response.headers['location']).query)
        assert query['from'] == [start_ms]
        assert query['to'] == [end_ms]
        assert query['var-worker_name'] == [api.worker.name]
        if resource.startswith('gpu-devices/'):
            assert query['var-gpu_index'] == ['0']


@pytest.mark.parametrize('resource', ['workers/7', 'gpu-devices/gpu-7'])
@pytest.mark.parametrize('suffix', ['metrics', 'monitoring-dashboard'])
@pytest.mark.parametrize(
    'params',
    [
        {'start': 1000},
        {'start': 1000, 'end': 1000.0009},
        {'start': 0, 'end': 604800.001},
        {'start': 'nan', 'end': 2000},
        {'start': 1000, 'end': 'inf'},
        {'start': 'true', 'end': 1001},
        {'start': 9999999999, 'end': 10000000000},
    ],
)
def test_resource_http_invalid_ranges_fail_before_metrics_or_redirect(
    resource_api, resource, suffix, params
):
    api = resource_api
    with TestClient(api.app) as client:
        response = client.get(
            f'/v2/{resource}/{suffix}', params=params, follow_redirects=False
        )
    assert response.status_code == 422
    assert 'location' not in response.headers
    api.metrics.assert_not_awaited()


@pytest.mark.parametrize('resource', ['workers/7', 'gpu-devices/gpu-7'])
def test_resource_capability_minimum_matches_second_precision_custom_range(
    resource_api, resource
):
    with TestClient(resource_api.app) as client:
        capabilities = client.get(f'/v2/{resource}/monitoring-capabilities')
        response = client.get(
            f'/v2/{resource}/metrics', params={'start': 1000, 'end': 1001}
        )
    assert capabilities.status_code == response.status_code == 200
    assert capabilities.json()['min_query_seconds'] == 1
    assert capabilities.json()['max_query_seconds'] == 604800
    assert capabilities.json()['supports_custom_range']
    kwargs = resource_api.metrics.await_args.kwargs
    assert (kwargs['start'], kwargs['end'], kwargs['window_seconds']) == (1000, 1001, 1)
