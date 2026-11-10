"""Model create and sparse update validate Router definitions in the owning Org."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from gpustack.api.exceptions import register_handlers
from gpustack.routes import models
from gpustack.schemas.inference_backend import InferenceBackend
from gpustack.schemas.models import Model, RoleSpec, DisaggregationSpec
from gpustack.schemas.principals import PrincipalType
from gpustack.server.deps import get_session, get_tenant_context


@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['create', 'update'])
@pytest.mark.parametrize('valid_org_command', [True, False])
async def test_pd_model_routes_validate_the_owning_org_backend(
    monkeypatch, operation, valid_org_command
):
    payload = {
        'name': 'pd',
        'source': 'local_path',
        'local_path': '/model',
        'backend': 'Custom',
        'image_name': 'parent:engine',
        'run_command': 'engine',
        'roles': [
            {'name': 'prefill'},
            {'name': 'decode'},
            {'name': 'router', 'backend': 'router-custom'},
        ],
        'disaggregation': {'mode': 'custom'},
        'enable_model_route': False,
    }
    stored = Model(
        id=7,
        owner_principal_id=42,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        **{
            k: v
            for k, v in payload.items()
            if k not in {'enable_model_route', 'roles', 'disaggregation'}
        },
        roles=[RoleSpec.model_validate(role) for role in payload['roles']],
        disaggregation=DisaggregationSpec.model_validate(payload['disaggregation']),
    )
    platform = InferenceBackend(
        backend_name='router-custom', default_run_command='router {{PLATFORM_ONLY}}'
    )
    owner = InferenceBackend(
        backend_name='router-custom',
        owner_principal_id=42,
        default_run_command=(
            'router --port {{port}}'
            if valid_org_command
            else 'router {{ORG_UNDECLARED}}'
        ),
    )
    lookup = AsyncMock(
        side_effect=lambda session, fields: (
            owner if fields['owner_principal_id'] == 42 else platform
        )
    )
    monkeypatch.setattr(InferenceBackend, 'one_by_fields', lookup)
    session = MagicMock(commit=AsyncMock(), rollback=AsyncMock())
    ctx = SimpleNamespace(
        current_principal_id=42, user=SimpleNamespace(id=99, kind=PrincipalType.SYSTEM)
    )
    monkeypatch.setattr(
        models, '_resolve_target_org', AsyncMock(return_value=(42, None))
    )
    monkeypatch.setattr(models, 'assert_cluster_belongs_to_org', AsyncMock())
    monkeypatch.setattr(models, '_assert_model_name_available', AsyncMock())
    monkeypatch.setattr(models, 'validate_shared_kv_cache', AsyncMock())
    monkeypatch.setattr(models, 'revoke_model_access_cache', AsyncMock())
    persist = AsyncMock(return_value=stored)
    monkeypatch.setattr(models, '_persist_model_create', persist)
    monkeypatch.setattr(models, 'lock_model', AsyncMock(return_value=stored))
    save = AsyncMock()
    monkeypatch.setattr(
        models, 'ModelService', lambda session: SimpleNamespace(update=save)
    )
    monkeypatch.setattr(
        models.ModelRoute, 'one_by_fields', AsyncMock(return_value=None)
    )
    app = FastAPI()
    app.include_router(models.router, prefix='/v2/models')
    register_handlers(app)
    app.dependency_overrides[get_session] = lambda: session
    app.dependency_overrides[get_tenant_context] = lambda: ctx
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url='http://test'
    ) as client:
        if operation == 'create':
            response = await client.post('/v2/models', json=payload)
        else:
            response = await client.put(
                '/v2/models/7',
                json={
                    'name': 'pd',
                    'source': 'local_path',
                    'local_path': '/model',
                    'description': 'edited',
                },
            )
    assert response.status_code == (200 if valid_org_command else 400), response.text
    assert [call.args[1]['owner_principal_id'] for call in lookup.call_args_list] == [
        None,
        42,
    ]
    if valid_org_command:
        assert response.json()['owner_principal_id'] == 42
        (persist if operation == 'create' else save).assert_awaited_once()
        session.commit.assert_awaited_once()
    else:
        assert 'ORG_UNDECLARED' in response.text
        assert 'PLATFORM_ONLY' not in response.text
        persist.assert_not_awaited()
        save.assert_not_awaited()
        session.commit.assert_not_awaited()
