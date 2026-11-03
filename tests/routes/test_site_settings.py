from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, HTTPException, Request, Response
from types import SimpleNamespace
from gpustack.api.exceptions import ServiceUnavailableException, register_handlers
from fastapi.testclient import TestClient
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from gpustack.api.auth import get_current_user
from gpustack.routes import site_settings
from gpustack.schemas.site_settings import SiteSettings, SiteSettingsUpdate
from gpustack.schemas.users import User
from gpustack.schemas.api_keys import PermissionScope
from gpustack.server.db import get_session


def fake_session(row=None):
    session = MagicMock()
    session.get = AsyncMock(return_value=row)
    session.exec = AsyncMock(return_value=MagicMock())
    session.exec.return_value.first.return_value = row
    session.exec.return_value.one.return_value = row
    for method in ['flush', 'commit', 'rollback']:
        setattr(session, method, AsyncMock())

    @asynccontextmanager
    async def begin_nested():
        yield

    session.begin_nested = begin_nested
    return session


@pytest.fixture
def client():
    # Exercise the production mount and auth tiers, including management scopes.
    from gpustack.routes.routes import api_router

    session = fake_session()
    app = FastAPI()
    register_handlers(app)
    app.include_router(api_router)
    app.dependency_overrides[get_session] = lambda: session
    with TestClient(app) as client:
        yield client, app, session


@pytest.mark.parametrize(
    'role,status', [('anonymous', 401), ('user', 403), ('admin', 200)]
)
def test_only_admin_can_read_and_update(client, role, status):
    http, app, session = client

    async def identity():
        if role == 'anonymous':
            raise HTTPException(status_code=401)
        return User(id=1, name=role, is_admin=role == 'admin')

    app.dependency_overrides[get_current_user] = identity
    assert http.get('/v2/site-settings').status_code == status
    result = http.put(
        '/v2/site-settings', json={'site_name': 'Lab', 'primary_color': '#123456'}
    )
    assert result.status_code == status
    if role != 'admin':
        session.commit.assert_not_awaited()


def test_public_read_does_not_authenticate_or_write(client):
    http, app, session = client

    async def forbidden_auth():
        raise AssertionError('Public branding must not authenticate')

    app.dependency_overrides[get_current_user] = forbidden_auth
    response = http.get('/v2/site-settings/public')
    assert response.status_code == 200
    assert response.json()['site_name'] == 'GPUStack'
    assert 'id' not in response.json()
    assert response.headers['cache-control'] == 'no-cache'
    session.add.assert_not_called()
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_and_noop_timestamps():
    row = SiteSettings()
    session = fake_session(row)
    values = SiteSettingsUpdate(site_name='Lab', primary_color='#123456')
    result = await site_settings.update_site_settings(values, session, Response())
    assert result.site_name == 'Lab'
    assert result.updated_at is not None
    again = await site_settings.update_site_settings(values, session, Response())
    assert again.updated_at == result.updated_at
    session.commit.assert_awaited()


@pytest.mark.asyncio
async def test_concurrent_first_save_uses_existing_singleton():
    session = fake_session()
    row = SiteSettings()
    session.exec.return_value.one.return_value = row
    session.flush.side_effect = [IntegrityError('insert', {}, Exception()), None]
    values = SiteSettingsUpdate(site_name='Lab', primary_color='#123456')
    result = await site_settings.update_site_settings(values, session, Response())
    assert result.site_name == 'Lab'
    assert session.exec.await_count == 1
    session.get.assert_awaited_once_with(SiteSettings, 1)
    calls = [call[0] for call in session.mock_calls]
    assert calls.index('flush') < calls.index('exec')
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_failed_save_rolls_back():
    session = fake_session(SiteSettings())
    session.commit.side_effect = SQLAlchemyError('database unavailable')
    with pytest.raises(ServiceUnavailableException) as error:
        await site_settings.update_site_settings(
            SiteSettingsUpdate(site_name='Lab', primary_color='#123456'),
            session,
            Response(),
        )
    assert error.value.status_code == 503
    session.rollback.assert_awaited_once()


@pytest.mark.parametrize('path', ['/v2/site-settings', '/v2/site-settings/public'])
def test_reads_return_persisted_branding_without_internal_fields(client, path):
    http, app, session = client
    app.dependency_overrides[get_current_user] = lambda: User(
        id=1, name='admin', is_admin=True
    )
    session.get.return_value = SiteSettings(
        site_name='Research Lab', footer_url='/about'
    )
    result = http.get(path)
    assert result.status_code == 200
    assert result.json()['site_name'] == 'Research Lab'
    assert result.json()['footer_url'] == '/about'
    assert set(result.json()) == {
        'site_name',
        'primary_color',
        'logo',
        'logo_dark',
        'mini_logo',
        'mini_logo_dark',
        'favicon',
        'footer_name',
        'footer_url',
        'updated_at',
    }
    assert result.headers['cache-control'] == (
        'no-cache' if path.endswith('/public') else 'no-store'
    )
    session.commit.assert_not_awaited()


@pytest.mark.parametrize('path', ['/v2/site-settings', '/v2/site-settings/public'])
def test_read_failure_returns_503_without_database_details(client, path):
    http, app, session = client
    app.dependency_overrides[get_current_user] = lambda: User(
        id=1, name='admin', is_admin=True
    )
    session.get.side_effect = SQLAlchemyError('private connection details')
    result = http.get(path)
    assert result.status_code == 503
    assert 'temporarily unavailable' in result.text
    assert 'private connection details' not in result.text
    session.commit.assert_not_awaited()


@pytest.mark.parametrize(
    'payload',
    [
        {},
        {'site_name': 'Lab'},
        {'site_name': ' ', 'primary_color': '#123456'},
        {'site_name': 'Lab', 'primary_color': '#123456', 'id': 2},
        {
            'site_name': 'Lab',
            'primary_color': '#123456',
            'footer_url': 'javascript:alert(1)',
        },
        {
            'site_name': 'Lab',
            'primary_color': '#123456',
            'logo': 'data:image/svg+xml;base64,PHN2Zz4=',
        },
    ],
)
def test_invalid_updates_do_not_write(client, payload):
    http, app, session = client
    app.dependency_overrides[get_current_user] = lambda: User(
        id=1, name='admin', is_admin=True
    )
    assert http.put('/v2/site-settings', json=payload).status_code == 422
    session.add.assert_not_called()
    session.exec.assert_not_awaited()
    session.commit.assert_not_awaited()


def test_put_replaces_optional_fields_and_public_read_observes_saved_values(client):
    http, app, session = client
    app.dependency_overrides[get_current_user] = lambda: User(
        id=1, name='admin', is_admin=True
    )
    row = SiteSettings(
        site_name='Old', logo='old-logo', footer_name='Old Co.', footer_url='/old'
    )
    session.exec.return_value.first.return_value = row
    session.exec.return_value.one.return_value = row
    session.get.return_value = row
    saved = http.put(
        '/v2/site-settings', json={'site_name': 'Lab', 'primary_color': '#123456'}
    )
    assert saved.status_code == 200
    assert saved.headers['cache-control'] == 'no-store'
    assert saved.json()['updated_at'] is not None
    assert saved.json()['logo'] is None
    assert saved.json()['footer_name'] is None
    assert saved.json()['footer_url'] is None
    session.commit.assert_awaited_once()
    assert http.get('/v2/site-settings/public').json() == saved.json()
    again = http.put(
        '/v2/site-settings', json={'site_name': 'Lab', 'primary_color': '#123456'}
    )
    assert again.json()['updated_at'] == saved.json()['updated_at']


@pytest.mark.asyncio
async def test_first_save_of_defaults_creates_only_the_singleton():
    session = fake_session()
    response = await site_settings.update_site_settings(
        SiteSettingsUpdate(site_name='GPUStack', primary_color='#007BFF'),
        session,
        Response(),
    )
    row = session.add.call_args.args[0]
    assert row.id == 1
    assert response.updated_at is None
    session.add.assert_called_once()
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize('phase', ['get', 'exec', 'flush'])
async def test_database_errors_before_commit_roll_back(phase):
    session = fake_session(SiteSettings())
    getattr(session, phase).side_effect = SQLAlchemyError('database unavailable')
    with pytest.raises(ServiceUnavailableException):
        await site_settings.update_site_settings(
            SiteSettingsUpdate(site_name='Lab', primary_color='#123456'),
            session,
            Response(),
        )
    session.rollback.assert_awaited_once()
    session.commit.assert_not_awaited()


@pytest.mark.parametrize(
    'scope,status',
    [('inference', 403), ('management', 200), (PermissionScope.ALL, 200)],
)
def test_administrator_api_keys_require_management_scope(client, scope, status):
    http, app, session = client

    async def identity(request: Request):
        request.state.api_key = SimpleNamespace(scope=[scope])
        return User(id=1, name='admin', is_admin=True)

    app.dependency_overrides[get_current_user] = identity
    assert http.get('/v2/site-settings').status_code == status
    assert (
        http.put(
            '/v2/site-settings', json={'site_name': 'Lab', 'primary_color': '#123456'}
        ).status_code
        == status
    )
    assert http.get('/v2/site-settings/public').status_code == 200


@pytest.mark.parametrize('persisted', [False, True])
@pytest.mark.parametrize('validator', ['weak', 'strong', 'list', 'wildcard'])
def test_public_settings_revalidation_avoids_resending_assets(
    client, persisted, validator
):
    http, app, session = client
    if persisted:
        session.get.return_value = SiteSettings(site_name='Lab', logo='large-asset')
    initial = http.get('/v2/site-settings/public')
    etag = initial.headers['etag']
    headers = {
        'weak': etag,
        'strong': etag[2:],
        'list': f'"unrelated", {etag}',
        'wildcard': '*',
    }
    cached = http.get(
        '/v2/site-settings/public', headers={'If-None-Match': headers[validator]}
    )
    assert cached.status_code == 304
    assert cached.content == b''
    assert cached.headers['etag'] == etag
    assert cached.headers['cache-control'] == 'no-cache'
    session.commit.assert_not_awaited()


def test_public_etag_tracks_content_including_images_and_other_replica_changes(client):
    http, app, session = client
    initial = http.get('/v2/site-settings/public')
    # A different replica can populate the row after an initial default read.
    row = SiteSettings()
    session.get.return_value = row
    cached = http.get(
        '/v2/site-settings/public', headers={'If-None-Match': initial.headers['etag']}
    )
    assert cached.status_code == 304
    row.logo = 'changed-asset'
    changed = http.get(
        '/v2/site-settings/public', headers={'If-None-Match': initial.headers['etag']}
    )
    assert changed.status_code == 200
    assert changed.json()['logo'] == 'changed-asset'
    assert changed.headers['etag'] != initial.headers['etag']


@pytest.mark.asyncio
async def test_update_refreshes_values_after_acquiring_the_existing_row_lock():
    stale = SiteSettings()
    latest = SiteSettings(site_name='Concurrent writer')
    session = fake_session(stale)
    session.exec.return_value.one.return_value = latest
    result = await site_settings.update_site_settings(
        SiteSettingsUpdate(site_name='GPUStack', primary_color='#007BFF'),
        session,
        Response(),
    )
    assert result.site_name == 'GPUStack'
    assert result.updated_at is not None
    statement = session.exec.call_args.args[0]
    assert statement.get_execution_options()['populate_existing'] is True
    from sqlalchemy.dialects import mysql, postgresql

    for dialect in [mysql.dialect(), postgresql.dialect()]:
        assert 'FOR UPDATE' in str(statement.compile(dialect=dialect))
    session.add.assert_not_called()
