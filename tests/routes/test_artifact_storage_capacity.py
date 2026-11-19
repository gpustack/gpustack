from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gpustack.api.exceptions import register_handlers
from gpustack.api.tenant import get_tenant_context
from gpustack.artifacts import capacity
from gpustack.routes import artifact_storage
from gpustack.schemas.artifact_storage import ArtifactStorageCapacity


def usage(maximum=0):
    return {
        'max_bytes': maximum,
        'unlimited': maximum == 0,
        'used_bytes': 2**34,
        'remaining_bytes': None if not maximum else 0,
        'usage_percent': None if not maximum else 100,
        'state': 'unlimited' if not maximum else 'full',
        'warning_threshold_percent': 80,
    }


def application(admin):
    app = FastAPI()
    register_handlers(app)
    app.dependency_overrides[get_tenant_context] = lambda: SimpleNamespace(
        is_platform_admin=admin
    )
    app.include_router(artifact_storage.router)
    return app


@pytest.mark.parametrize('admin', [False, True])
@pytest.mark.parametrize('method', ['get', 'put'])
def test_capacity_read_and_update_require_platform_administrator(
    monkeypatch, admin, method
):
    manager = SimpleNamespace(
        usage=AsyncMock(return_value=usage()),
        set_limit=AsyncMock(return_value=usage()),
    )
    monkeypatch.setattr(artifact_storage, 'CapacityManager', lambda: manager)
    with TestClient(application(admin)) as client:
        response = (
            client.get('/capacity')
            if method == 'get'
            else client.put('/capacity', json={'max_bytes': 0})
        )
    assert response.status_code == (200 if admin else 403)
    operation = manager.usage if method == 'get' else manager.set_limit
    assert operation.await_count == int(admin)
    if admin:
        assert response.json()['used_bytes'] == 2**34
        assert response.json()['unlimited'] is True


@pytest.mark.parametrize('value', [-1, 1.5, True, '100', 9007199254740992])
def test_capacity_update_rejects_invalid_limits_before_side_effects(monkeypatch, value):
    manager = SimpleNamespace(set_limit=AsyncMock())
    monkeypatch.setattr(artifact_storage, 'CapacityManager', lambda: manager)
    with TestClient(application(True)) as client:
        response = client.put('/capacity', json={'max_bytes': value})
    assert response.status_code == 422
    assert manager.set_limit.await_count == 0


def test_first_capacity_get_initializes_and_keeps_the_configured_limit(monkeypatch):
    row = ArtifactStorageCapacity(id=1, used_bytes=40)
    session = SimpleNamespace(
        exec=AsyncMock(return_value=Mock(one=lambda: row)),
        add=Mock(),
        commit=AsyncMock(),
    )

    @asynccontextmanager
    async def sessions():
        yield session

    config = SimpleNamespace(benchmark_archive_max_total_bytes=100)
    monkeypatch.setattr(capacity, 'async_session', sessions)
    monkeypatch.setattr(capacity, 'get_global_config', lambda: config)

    with TestClient(application(True)) as client:
        first = client.get('/capacity')
        config.benchmark_archive_max_total_bytes = 200
        second = client.get('/capacity')
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert first.json()['max_bytes'] == 100
    assert first.json()['remaining_bytes'] == 60
    assert row.initialized
    session.commit.assert_awaited_once()


@pytest.mark.parametrize('method', ['get', 'put'])
def test_database_errors_are_safe_and_explicit(monkeypatch, caplog, method):
    manager = SimpleNamespace(
        usage=AsyncMock(side_effect=RuntimeError('postgresql://secret@private')),
        set_limit=AsyncMock(side_effect=RuntimeError('database-secret')),
    )
    monkeypatch.setattr(artifact_storage, 'CapacityManager', lambda: manager)
    with TestClient(application(True)) as client:
        response = (
            client.get('/capacity')
            if method == 'get'
            else client.put('/capacity', json={'max_bytes': 0})
        )
    assert response.status_code == 503
    assert 'capacity_unavailable' in response.text
    assert 'secret' not in response.text
    assert 'private' not in response.text
    records = [
        record for record in caplog.records if record.name == artifact_storage.__name__
    ]
    assert len(records) == 1
    operation = 'read' if method == 'get' else 'update'
    assert records[0].levelname == 'ERROR'
    assert operation in records[0].getMessage()
    assert 'RuntimeError' in records[0].getMessage()
    assert 'secret' not in caplog.text
    assert 'private' not in caplog.text
