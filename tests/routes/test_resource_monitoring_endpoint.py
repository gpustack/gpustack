from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

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
def test_owner_can_read_resource_http_routes(resource_api, path):
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
        response = client.get(f'/v2/gpu-devices/gpu-7/{suffix}', follow_redirects=False)
    assert response.status_code == 404
    api.metrics.assert_not_awaited()
    api.policy.assert_not_awaited()
    api.allocated.assert_not_awaited()
