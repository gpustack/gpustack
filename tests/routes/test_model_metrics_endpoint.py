"""Monitoring follows Model authorization and never accepts arbitrary populations."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from gpustack.api.auth import get_current_user
from gpustack.api.exceptions import register_handlers
from gpustack.api.tenant import TenantContext, get_tenant_context
from gpustack.routes import models as route
from gpustack.routes.routes import SYSTEM_ROUTES, management_router
from gpustack.schemas.model_metrics import (
    ModelMetrics,
    ModelResourceMetrics,
    ModelMonitoringCapabilities,
)
from gpustack.schemas.principals import OrgRole, PrincipalType
from gpustack.server.db import get_session


@pytest.fixture
def monitoring_api(monkeypatch):
    user = SimpleNamespace(
        id=7,
        name='reader',
        kind=PrincipalType.USER,
        is_admin=False,
        worker=None,
        cluster=None,
    )
    context = TenantContext(
        user=user,
        is_platform_admin=False,
        current_principal_id=42,
        org_role=OrgRole.OWNER,
    )
    instance = SimpleNamespace(
        id=12,
        model_id=9,
        name='instance',
        role=None,
        owner_principal_id=42,
        deleted_at=None,
    )
    model = SimpleNamespace(
        id=9,
        name='model',
        owner_principal_id=42,
        cluster_id=1,
        instances=[instance],
        deleted_at=None,
    )
    runtime = AsyncMock(
        return_value=ModelMetrics(
            model_id=9,
            start=100,
            end=200,
            step=15,
            rate_window_seconds=75,
            last_refreshed_at=200,
            history_scope='current_instances_with_observed_source_history',
        )
    )
    hardware = AsyncMock(
        return_value=ModelResourceMetrics(
            model_id=9,
            start=100,
            end=200,
            step=15,
            last_refreshed_at=200,
            history_scope='owner_qualified_current_associated_resources',
        )
    )
    monkeypatch.setattr(route.Model, 'one_by_id', AsyncMock(return_value=model))
    monkeypatch.setattr(route, 'collect_model_metrics', runtime, raising=False)
    monkeypatch.setattr(
        route, 'collect_model_resource_metrics', hardware, raising=False
    )
    app = FastAPI()
    register_handlers(app)
    app.dependency_overrides[get_current_user] = lambda: user
    app.dependency_overrides[get_tenant_context] = lambda: context
    app.dependency_overrides[get_session] = lambda: None
    app.include_router(management_router)
    return SimpleNamespace(
        app=app,
        ctx=context,
        user=user,
        model=model,
        instance=instance,
        runtime=runtime,
        hardware=hardware,
    )


def test_owner_gets_runtime_contract_and_server_validated_options(monitoring_api):
    api = monitoring_api
    with TestClient(api.app) as client:
        response = client.get(
            '/v2/models/9/metrics?instance_ids=12,12&statistic=avg'
            '&view=instances&section=llm&window=30m'
        )
    assert response.status_code == 200
    assert response.json()['source'] == 'engine'
    assert api.runtime.await_args.kwargs['window_seconds'] == 1800
    assert api.runtime.await_args.kwargs['view'] == 'instances'
    assert api.runtime.await_args.kwargs['statistic'] == 'avg'
    assert [i.id for i in api.runtime.await_args.args[1]] == [12]


@pytest.mark.parametrize('suffix', ['metrics', 'resource-metrics'])
def test_custom_range_delegates_exact_authorized_bounds(monitoring_api, suffix):
    api = monitoring_api
    with TestClient(api.app) as client:
        response = client.get(
            f'/v2/models/9/{suffix}?start=1000&end=2000&instance_ids=12'
        )
    assert response.status_code == 200
    collector = api.runtime if suffix == 'metrics' else api.hardware
    assert collector.await_args.kwargs['start'] == 1000
    assert collector.await_args.kwargs['end'] == 2000
    assert collector.await_args.kwargs['window_seconds'] == 1000
    assert [i.id for i in collector.await_args.args[1]] == [12]


@pytest.mark.parametrize('suffix', ['metrics', 'resource-metrics'])
@pytest.mark.parametrize(
    'query',
    [
        'start=1000',
        'end=2000',
        'start=1000&end=1299',
        'start=2000&end=1000',
        'start=nan&end=2000',
        'start=1000&end=inf',
        'start=1000.5&end=2000',
        'start=0&end=604801',
        'start=999999999900&end=999999999999',
    ],
)
def test_invalid_custom_bounds_never_reach_collectors(monitoring_api, suffix, query):
    api = monitoring_api
    with TestClient(api.app) as client:
        response = client.get(f'/v2/models/9/{suffix}?{query}')
    assert response.status_code in (400, 422)
    api.runtime.assert_not_awaited()
    api.hardware.assert_not_awaited()


def test_capabilities_are_guarded_and_never_expose_global_storage_contents(
    monitoring_api, monkeypatch
):
    api = monitoring_api
    capabilities = AsyncMock(
        return_value=ModelMonitoringCapabilities(
            model_id=9,
            retention_seconds=1296000,
            retention_known=True,
            retention_source='prometheus_runtimeinfo',
        )
    )
    monkeypatch.setattr(route, 'collect_monitoring_capabilities', capabilities)
    with TestClient(api.app) as client:
        response = client.get('/v2/models/9/monitoring-capabilities')
    assert response.status_code == 200
    body = response.json()
    assert body['model_id'] == 9 and body['supports_custom_range']
    assert body['max_query_seconds'] == 604800
    assert body['retention_seconds'] == 1296000
    assert not any(key in body for key in ('hostname', 'minTime', 'flags', 'metrics'))
    capabilities.reset_mock()
    api.model.owner_principal_id = 43
    with TestClient(api.app) as client:
        response = client.get('/v2/models/9/monitoring-capabilities')
    assert response.status_code == 404
    capabilities.assert_not_awaited()


@pytest.mark.parametrize('view', ['aggregate', 'instances'])
def test_all_statistics_keeps_authorized_population_and_overrides_single_statistic(
    monitoring_api, view
):
    api = monitoring_api
    with TestClient(api.app) as client:
        response = client.get(
            f'/v2/models/9/metrics?instance_ids=12&all_statistics=true'
            f'&statistic=p95&view={view}'
        )
    assert response.status_code == 200
    assert api.runtime.await_args.kwargs['statistic'] == 'all'
    assert api.runtime.await_args.kwargs['view'] == view
    assert [i.id for i in api.runtime.await_args.args[1]] == [12]


def test_resource_endpoint_delegates_the_authorized_population(monitoring_api):
    api = monitoring_api
    with TestClient(api.app) as client:
        response = client.get(
            '/v2/models/9/resource-metrics?instance_ids=12&device_id=opaque'
        )
    assert response.status_code == 200
    assert response.json()['source'] == 'hardware'
    assert api.hardware.await_args.kwargs['device_id'] == 'opaque'
    assert api.hardware.await_args.args[0] is api.model
    assert api.hardware.await_args.args[2] is api.ctx


def test_resource_endpoint_delegates_all_devices_flag_with_authorized_population(
    monitoring_api,
):
    api = monitoring_api
    with TestClient(api.app) as client:
        response = client.get(
            '/v2/models/9/resource-metrics?all_devices=true&instance_ids=12'
        )
    assert response.status_code == 200
    assert api.hardware.await_args.kwargs['all_devices'] is True
    assert api.hardware.await_args.kwargs['device_id'] is None
    assert [instance.id for instance in api.hardware.await_args.args[1]] == [12]


@pytest.mark.parametrize(
    'suffix', ['metrics', 'resource-metrics', 'monitoring-capabilities']
)
@pytest.mark.parametrize('identity', ['member', 'personal', 'system'])
def test_monitoring_does_not_grant_member_personal_or_system_access(
    monitoring_api,
    suffix,
    identity,
):
    api = monitoring_api
    if identity == 'member':
        api.ctx.org_role = OrgRole.MEMBER
    elif identity == 'personal':
        api.ctx.org_role = None
        api.ctx.current_is_personal_scope = True
    else:
        api.user.kind = PrincipalType.SYSTEM
    with TestClient(api.app) as client:
        response = client.get(f'/v2/models/9/{suffix}')
    assert response.status_code == 403
    api.runtime.assert_not_awaited()
    api.hardware.assert_not_awaited()


@pytest.mark.parametrize('suffix', ['metrics', 'resource-metrics'])
def test_foreign_model_and_foreign_instance_filters_reject_entire_request(
    monitoring_api,
    suffix,
):
    api = monitoring_api
    with TestClient(api.app) as client:
        response = client.get(f'/v2/models/9/{suffix}?instance_ids=12,999')
    assert response.status_code == 404
    api.runtime.assert_not_awaited()
    api.hardware.assert_not_awaited()
    api.model.owner_principal_id = 43
    with TestClient(api.app) as client:
        response = client.get(f'/v2/models/9/{suffix}')
    assert response.status_code == 404


@pytest.mark.parametrize(
    'query',
    [
        'window=1m',
        'window=8d',
        'instance_ids=',
        'instance_ids=12,',
        'instance_ids=0',
        'instance_ids=12%22%7D',
        'instance_ids=' + ','.join(str(i) for i in range(1, 34)),
        'view=everything',
        'role=admin',
        'statistic=max',
        'section=raw',
    ],
)
def test_invalid_filters_never_reach_prometheus(monitoring_api, query):
    api = monitoring_api
    with TestClient(api.app) as client:
        response = client.get('/v2/models/9/metrics?' + query)
    assert response.status_code in (400, 422)
    api.runtime.assert_not_awaited()


def test_platform_admin_act_as_org_still_cannot_read_another_org(monitoring_api):
    api = monitoring_api
    api.user.is_admin = True
    api.ctx.is_platform_admin = True
    api.model.owner_principal_id = 43
    with TestClient(api.app) as client:
        response = client.get('/v2/models/9/metrics')
    assert response.status_code == 404
    api.ctx.current_principal_id = None
    with TestClient(api.app) as client:
        response = client.get('/v2/models/9/metrics')
    assert response.status_code == 200


def test_monitoring_endpoints_are_not_worker_runtime_allowlist_entries(monitoring_api):
    with TestClient(monitoring_api.app) as client:
        response = client.get('/v2/models/9/metrics')
    assert response.status_code == 200
    assert ('GET', route.get_model_metrics) not in SYSTEM_ROUTES
    assert ('GET', route.get_model_resource_metrics) not in SYSTEM_ROUTES
    assert ('GET', route.get_model_monitoring_capabilities) not in SYSTEM_ROUTES
