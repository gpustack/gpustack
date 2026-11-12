"""Native monitoring links preserve only supported, authorized Grafana scopes."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlsplit

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from gpustack.api.auth import get_current_user
from gpustack.api.exceptions import register_handlers
from gpustack.api.tenant import TenantContext, get_tenant_context
from gpustack.routes import models as route
from gpustack.routes.routes import management_router
from gpustack.schemas.models import DisaggregationSpec, PDModeEnum
from gpustack.schemas.principals import OrgRole, PrincipalType
from gpustack.server.db import get_session


@pytest.fixture
def dashboard_api(monkeypatch):
    user = SimpleNamespace(id=7, kind=PrincipalType.USER, is_admin=False)
    context = TenantContext(
        user=user,
        current_principal_id=42,
        org_role=OrgRole.OWNER,
        is_platform_admin=False,
    )
    instances = [
        SimpleNamespace(
            id=id,
            name=f'instance-{id}',
            model_id=9,
            owner_principal_id=42,
            deleted_at=None,
            role=role,
        )
        for id, role in [(12, 'prefill'), (13, 'decode'), (14, 'router')]
    ]
    model = SimpleNamespace(
        id=9,
        name='model',
        cluster_id=1,
        owner_principal_id=42,
        instances=instances,
        deleted_at=None,
        disaggregation=None,
    )
    monkeypatch.setattr(route.Model, 'one_by_id', AsyncMock(return_value=model))
    monkeypatch.setattr(
        route.Cluster,
        'one_by_id',
        AsyncMock(return_value=SimpleNamespace(name='cluster')),
    )
    monkeypatch.setattr(
        route,
        'get_global_config',
        lambda: SimpleNamespace(
            grafana_model_dashboard_uid='model',
            grafana_pd_dashboard_uid='pd',
            get_grafana_url=lambda: 'http://grafana',
        ),
    )
    monkeypatch.setattr(route, 'resolve_grafana_base_url', lambda *_: 'http://grafana')
    monkeypatch.setattr('gpustack.server.pd_mode_catalog.get_pd_mode', lambda *_: None)
    app = FastAPI()
    register_handlers(app)
    app.dependency_overrides[get_current_user] = lambda: user
    app.dependency_overrides[get_tenant_context] = lambda: context
    app.dependency_overrides[get_session] = lambda: None
    app.include_router(management_router)
    return SimpleNamespace(
        app=app, model=model, instances=instances, ctx=context, user=user
    )


def request_dashboard(api, query):
    with TestClient(api.app) as client:
        return client.get('/v2/models/9/dashboard?' + query, follow_redirects=False)


def test_plain_dashboard_keeps_window_and_multi_instance_fences(dashboard_api):
    response = request_dashboard(dashboard_api, 'window=30m&instance_ids=13,12')
    assert response.status_code == 302
    values = parse_qs(urlsplit(response.headers['location']).query)
    assert values['from'] == ['now-1800s']
    assert values['to'] == ['now']
    assert values['var-scope_model_id'] == ['"9"']
    assert values['var-scope_cluster_id'] == ['"1"']
    assert json.loads(values['var-scope_model_instance_id'][0]) == '^(12|13)$'
    assert values['var-selected_model_instance_id'] == ['$__all']
    assert 'var-role' not in values


def test_custom_dashboard_range_uses_exact_milliseconds(dashboard_api):
    response = request_dashboard(dashboard_api, 'start=1000&end=2000&instance_ids=12')
    assert response.status_code == 302
    values = parse_qs(urlsplit(response.headers['location']).query)
    assert values['from'] == ['1000000']
    assert values['to'] == ['2000000']
    assert json.loads(values['var-scope_model_instance_id'][0]) == '^(12)$'


def test_pd_link_uses_selected_instances_roles_without_inventing_instance_filter(
    dashboard_api,
):
    dashboard_api.model.disaggregation = DisaggregationSpec(mode=PDModeEnum.VLLM_NIXL)
    response = request_dashboard(dashboard_api, 'window=6h&instance_ids=12,13')
    assert response.status_code == 302
    values = parse_qs(urlsplit(response.headers['location']).query)
    assert values['from'] == ['now-21600s']
    assert values['var-role'] == ['prefill', 'decode']
    assert values['var-counted_role'] == ['decode']
    assert not any('instance' in key for key in values)


def test_pd_link_preserves_explicit_router_role(dashboard_api):
    dashboard_api.model.disaggregation = DisaggregationSpec(mode=PDModeEnum.VLLM_NIXL)
    response = request_dashboard(dashboard_api, 'window=24h&role=router')
    assert response.status_code == 302
    values = parse_qs(urlsplit(response.headers['location']).query)
    assert values['var-role'] == ['router']


def test_pd_selection_without_a_supported_role_is_not_broadened_to_all(dashboard_api):
    dashboard_api.model.disaggregation = DisaggregationSpec(mode=PDModeEnum.VLLM_NIXL)
    dashboard_api.instances[0].role = None
    assert (
        request_dashboard(dashboard_api, 'window=1h&instance_ids=12').status_code == 400
    )


@pytest.mark.parametrize(
    'query, status',
    [
        ('window=1m', 400),
        ('window=8d', 400),
        ('window=1h&instance_ids=12,999', 404),
        ('window=1h&instance_ids=12%22%7D', 400),
        ('window=1h&role=arbitrary', 422),
        ('window=1h&role=prefill', 400),
    ],
)
def test_invalid_dashboard_scopes_are_rejected(dashboard_api, query, status):
    assert request_dashboard(dashboard_api, query).status_code == status


@pytest.mark.parametrize(
    'identity', ['foreign_model', 'foreign_instance', 'member', 'system']
)
def test_scoped_link_cannot_expand_model_monitoring_visibility(dashboard_api, identity):
    if identity == 'foreign_model':
        dashboard_api.model.owner_principal_id = 43
    elif identity == 'foreign_instance':
        dashboard_api.instances[0].owner_principal_id = 43
    elif identity == 'member':
        dashboard_api.ctx.org_role = OrgRole.MEMBER
    else:
        dashboard_api.user.kind = PrincipalType.SYSTEM
    expected = 404 if identity.startswith('foreign') else 403
    assert (
        request_dashboard(dashboard_api, 'window=1h&instance_ids=12').status_code
        == expected
    )


def test_client_cannot_override_dashboard_destination_or_promql_scope(dashboard_api):
    response = request_dashboard(
        dashboard_api, 'window=1h&url=https://evil.test&var-scope_model_id=.*'
    )
    assert response.status_code == 302
    assert response.headers['location'].startswith('http://grafana/d/')
    values = parse_qs(urlsplit(response.headers['location']).query)
    assert values['var-scope_model_id'] == ['"9"']
