"""Dashboard model activity respects both tenant and cluster boundaries."""

from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.dialects import mysql, postgresql
from sqlalchemy.sql import visitors
from sqlalchemy.sql.elements import BinaryExpression
from sqlmodel import col, select

from gpustack.api.exceptions import ForbiddenException, NotFoundException
from gpustack.api.tenant import TenantContext
from gpustack.routes import dashboard as route
from gpustack.schemas.model_usage import ModelUsage
from gpustack.schemas.models import Model
from gpustack.schemas.principals import OrgRole, PrincipalType


def _ctx(owner_id=10, role=OrgRole.OWNER, admin=False, grants=(7,)):
    return TenantContext(
        user=SimpleNamespace(kind=PrincipalType.USER),
        is_platform_admin=admin,
        current_principal_id=owner_id,
        org_role=role,
        accessible_cluster_ids=set(grants),
    )


def _has_predicate(expression, predicate):
    return any(
        isinstance(node, BinaryExpression) and node.compare(predicate)
        for node in visitors.iterate(expression)
    )


def _assert_usage_scope(statement, cluster_id, owner_id):
    if cluster_id is not None:
        assert _has_predicate(
            statement.whereclause,
            col(ModelUsage.model_id).in_(
                select(Model.id).where(Model.cluster_id == cluster_id)
            ),
        )
    else:
        assert 'cluster_id' not in str(statement.whereclause)
    if owner_id is not None:
        assert _has_predicate(
            statement.whereclause, ModelUsage.owner_principal_id == owner_id
        )
    else:
        assert 'owner_principal_id' not in str(statement.whereclause)


@pytest.fixture
def dashboard_data(monkeypatch):
    cluster = SimpleNamespace(id=7, owner_principal_id=20)
    monkeypatch.setattr(route.Cluster, 'one_by_id', AsyncMock(return_value=cluster))
    resources = AsyncMock(
        return_value=dict(
            cluster_count=None,
            worker_count=2,
            gpu_count=4,
            model_count=1,
            model_instance_count=1,
        )
    )
    load = AsyncMock(
        return_value=dict(
            current=dict(cpu=10, ram=20, gpu=30, vram=40),
            history=dict(cpu=[], ram=[], gpu=[], vram=[]),
        )
    )
    monkeypatch.setattr(route, 'get_resource_counts', resources)
    monkeypatch.setattr(route, 'get_system_load', load)
    monkeypatch.setattr(
        route.ModelInstance, 'all_by_fields', AsyncMock(return_value=[])
    )
    monkeypatch.setattr(route, '_get_maas_active_models', AsyncMock(return_value=[]))
    rows = [
        [
            SimpleNamespace(
                date=date.today(),
                total_prompt_tokens=120,
                total_completion_tokens=30,
                total_requests=2,
            )
        ],
        [
            SimpleNamespace(
                user_id=100,
                username='tenant-user',
                total_prompt_tokens=120,
                total_completion_tokens=30,
            )
        ],
        [
            SimpleNamespace(
                id=71,
                name='tenant-model',
                categories=None,
                instance_count=1,
                total_token_count=150,
            )
        ],
    ]
    session = SimpleNamespace(
        exec=AsyncMock(side_effect=[MagicMock(all=lambda r=r: r) for r in rows])
    )
    return SimpleNamespace(
        cluster=cluster, session=session, resources=resources, load=load
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('cluster_id', [None, 7])
@pytest.mark.parametrize(
    'ctx, owner_id',
    [
        (_ctx(), 10),
        (_ctx(owner_id=10, role=None, admin=True), 10),
        (_ctx(owner_id=None, role=None, admin=True), None),
    ],
    ids=['org-owner', 'admin-in-org', 'admin-all'],
)
async def test_dashboard_activity_uses_caller_scope(
    dashboard_data, ctx, owner_id, cluster_id
):
    data = dashboard_data
    result = await route.dashboard(data.session, ctx, cluster_id)

    assert result.model_usage.top_users[0].username == 'tenant-user'
    assert result.model_usage.prompt_token_history[0].value == 120
    assert result.active_models[0].name == 'tenant-model'
    assert result.active_models[0].token_count == 150
    statements = [call.args[0] for call in data.session.exec.await_args_list]
    assert len(statements) == 3
    for statement in statements[:2]:
        _assert_usage_scope(statement, cluster_id, owner_id)
    if owner_id is not None:
        assert _has_predicate(
            statements[2].whereclause, Model.owner_principal_id == owner_id
        )
        assert _has_predicate(statements[2], ModelUsage.owner_principal_id == owner_id)
    if cluster_id is not None:
        assert _has_predicate(statements[2].whereclause, Model.cluster_id == cluster_id)
    infrastructure_owner = owner_id if cluster_id is None else None
    data.resources.assert_awaited_once_with(
        data.session, cluster_id, infrastructure_owner
    )
    data.load.assert_awaited_once_with(data.session, cluster_id, infrastructure_owner)


@pytest.mark.asyncio
@pytest.mark.parametrize('cluster_owner', [None, 10, 20])
@pytest.mark.parametrize(
    'ctx',
    [_ctx(role=OrgRole.MEMBER), _ctx(role=None), _ctx(owner_id=None, role=None)],
    ids=['org-member', 'personal', 'no-tenant'],
)
async def test_cluster_visibility_only_grants_infrastructure(
    dashboard_data, ctx, cluster_owner
):
    data = dashboard_data
    data.cluster.owner_principal_id = cluster_owner
    result = await route.dashboard(data.session, ctx, cluster_id=7)

    assert result.resource_counts.worker_count == 2
    assert result.resource_counts.gpu_count == 4
    assert result.system_load.current.gpu == 30
    assert result.model_usage.model_dump() == dict(
        api_request_history=[],
        completion_token_history=[],
        prompt_token_history=[],
        top_users=[],
    )
    assert result.active_models == []
    data.session.exec.assert_not_awaited()
    data.resources.assert_awaited_once_with(data.session, 7, None)
    data.load.assert_awaited_once_with(data.session, 7, None)


@pytest.mark.asyncio
@pytest.mark.parametrize('role', [None, OrgRole.MEMBER])
async def test_aggregate_dashboard_requires_usage_permission(dashboard_data, role):
    data = dashboard_data
    with pytest.raises(ForbiddenException):
        await route.dashboard(data.session, _ctx(role=role), cluster_id=None)
    data.session.exec.assert_not_awaited()
    data.resources.assert_not_awaited()
    data.load.assert_not_awaited()


@pytest.mark.asyncio
async def test_invisible_cluster_is_rejected_before_reading_activity(dashboard_data):
    data = dashboard_data
    with pytest.raises(NotFoundException):
        await route.dashboard(data.session, _ctx(grants=()), cluster_id=7)
    data.session.exec.assert_not_awaited()
    data.resources.assert_not_awaited()
    data.load.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('dialect', [postgresql.dialect(), mysql.dialect()])
@pytest.mark.parametrize('model_ids', [None, [], [99]])
async def test_usage_filters_cannot_widen_cluster_scope(dialect, model_ids):
    session = SimpleNamespace(exec=AsyncMock(return_value=MagicMock(all=lambda: [])))
    await route.get_model_usage_stats(
        session,
        cluster_id=7,
        owner_principal_id=10,
        model_ids=model_ids,
        provider_model_names={9: ['provider-model']},
    )
    statement = session.exec.await_args.args[0]
    _assert_usage_scope(statement, 7, 10)
    sql = str(
        statement.compile(dialect=dialect, compile_kwargs={'literal_binds': True})
    )
    assert 'models.cluster_id = 7' in sql
    assert 'model_usages.owner_principal_id = 10' in sql


@pytest.mark.asyncio
async def test_empty_cluster_usage_does_not_fall_back_to_platform_totals():
    session = SimpleNamespace(exec=AsyncMock(return_value=MagicMock(all=lambda: [])))
    result = await route.get_model_usage_summary(session, cluster_id=7)
    assert result.top_users == []
    assert result.prompt_token_history == []
    for call in session.exec.await_args_list:
        _assert_usage_scope(call.args[0], 7, None)
