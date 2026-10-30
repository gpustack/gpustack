"""Global backends are shared; organization overrides stay within read scope."""

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from sqlalchemy.orm.evaluator import _EvaluatorCompiler

from gpustack.api.exceptions import register_handlers
from gpustack.api.tenant import TenantContext, get_tenant_context
from gpustack.mixins import active_record
from gpustack.routes import inference_backend as route
from gpustack.schemas.common import ListParams
from gpustack.schemas.inference_backend import InferenceBackend
from gpustack.schemas.models import BackendEnum, BackendSourceEnum
from gpustack.schemas.principals import PrincipalType
from gpustack.server.bus import Event, EventType
from gpustack.server.db import get_session

SCOPES = [
    "org-a",
    "org-b",
    "cluster",
    "cluster-no-owner",
    "admin",
    "admin-org",
    "server",
]


def _context(scope):
    return TenantContext(
        user=SimpleNamespace(
            kind=(
                PrincipalType.SYSTEM
                if scope.startswith("cluster") or scope == "server"
                else PrincipalType.USER
            )
        ),
        is_platform_admin=scope.startswith("admin"),
        current_principal_id=(
            10 if scope in ("org-a", "admin-org") else 20 if scope == "org-b" else None
        ),
        org_role=None,
        scoped_cluster_id=1 if scope.startswith("cluster") else None,
        scoped_cluster_owner_id=10 if scope == "cluster" else None,
    )


def _visible_ids(scope):
    if scope in ("admin", "server"):
        return [1, 2, 3]
    if scope == "cluster-no-owner":
        return [1]
    return [1, 3] if scope == "org-b" else [1, 2]


@pytest.fixture
def backends(monkeypatch):
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [
        InferenceBackend(
            id=row_id,
            backend_name=name,
            owner_principal_id=owner,
            backend_source=BackendSourceEnum.CUSTOM,
            default_run_command=f"run-{name}",
            created_at=timestamp,
            updated_at=timestamp,
        )
        for row_id, name, owner in (
            (1, "global-custom", None),
            (2, "org-a-custom", 10),
            (3, "org-b-custom", 20),
        )
    ]
    session = AsyncMock()

    @asynccontextmanager
    async def open_session():
        yield session

    async def filtered_rows(session, fields, extra_conditions):
        matches = _EvaluatorCompiler(InferenceBackend).process(*extra_conditions)
        return [row for row in rows if matches(row) is True]

    async def by_fields(session, fields):
        return next(
            (
                row
                for row in rows
                if all(getattr(row, k) == v for k, v in fields.items())
            ),
            None,
        )

    monkeypatch.setattr(route, "async_session", open_session)
    monkeypatch.setattr(route.Worker, "all", AsyncMock(return_value=[]))
    monkeypatch.setattr(route.RunnerOverrideEntry, "all", AsyncMock(return_value=[]))
    monkeypatch.setattr(InferenceBackend, "all", AsyncMock(return_value=rows))
    monkeypatch.setattr(
        InferenceBackend, "all_by_fields", AsyncMock(side_effect=filtered_rows)
    )
    monkeypatch.setattr(
        InferenceBackend, "one_by_fields", AsyncMock(side_effect=by_fields)
    )
    monkeypatch.setattr(
        InferenceBackend,
        "one_by_id",
        AsyncMock(side_effect=lambda session, id: rows[id - 1]),
    )
    app = FastAPI()
    register_handlers(app)
    app.include_router(route.router, prefix="/v2/inference-backends")
    app.dependency_overrides[get_session] = lambda: session
    return SimpleNamespace(app=app, rows=rows)


@pytest.mark.parametrize("scope", SCOPES)
@pytest.mark.parametrize("path", ["", "/all", "/list"])
def test_backend_lists_preserve_global_and_org_visibility(backends, scope, path):
    backends.app.dependency_overrides[get_tenant_context] = lambda: _context(scope)
    with TestClient(backends.app) as client:
        response = client.get(f"/v2/inference-backends{path}")
    assert response.status_code == 200
    items = response.json() if path == "/all" else response.json()["items"]
    expected_ids = (
        [1] if path == "/list" and scope in ("admin", "server") else _visible_ids(scope)
    )
    expected_names = {backends.rows[i - 1].backend_name for i in expected_ids}
    if path == "/list":
        expected_names.add(BackendEnum.CUSTOM.value)
    assert {item["backend_name"] for item in items} == expected_names


@pytest.mark.parametrize("scope", SCOPES)
@pytest.mark.parametrize("row_id", [1, 2, 3])
def test_backend_detail_matches_list_visibility(backends, scope, row_id):
    backends.app.dependency_overrides[get_tenant_context] = lambda: _context(scope)
    with TestClient(backends.app) as client:
        response = client.get(f"/v2/inference-backends/{row_id}")
    if row_id in _visible_ids(scope):
        assert response.status_code == 200
        assert (
            response.json()["default_run_command"]
            == backends.rows[row_id - 1].default_run_command
        )
    else:
        assert response.status_code == 404
        assert "run-org" not in response.text


@pytest.mark.parametrize("scope", SCOPES)
@pytest.mark.parametrize("has_override", [False, True])
def test_backend_name_prefers_own_override_then_global(backends, scope, has_override):
    backends.rows[0].backend_name = "shared-custom"
    for row in backends.rows[1:]:
        if has_override:
            row.backend_name = "shared-custom"
    backends.app.dependency_overrides[get_tenant_context] = lambda: _context(scope)
    with TestClient(backends.app) as client:
        response = client.get("/v2/inference-backends/backend_name/shared-custom")
    assert response.status_code == 200
    expected_id = 1
    if has_override and scope in ("org-a", "cluster", "admin-org"):
        expected_id = 2
    elif has_override and scope == "org-b":
        expected_id = 3
    assert response.json()["id"] == expected_id


@pytest.mark.parametrize("scope", ["org-a", "cluster", "admin-org", "cluster-no-owner"])
def test_backend_name_does_not_resolve_another_org(backends, scope):
    backends.app.dependency_overrides[get_tenant_context] = lambda: _context(scope)
    with TestClient(backends.app) as client:
        response = client.get("/v2/inference-backends/backend_name/org-b-custom")
    assert response.status_code == 400
    assert "run-org-b" not in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", SCOPES)
async def test_backend_watch_scopes_replay_and_live_events(
    backends, monkeypatch, scope
):
    cached_all = AsyncMock(return_value=backends.rows)
    monkeypatch.setattr(InferenceBackend, "cached_all", cached_all)
    events = [
        Event(type=kind, data=row)
        for kind in (EventType.CREATED, EventType.UPDATED, EventType.DELETED)
        for row in backends.rows
    ]
    events.append(Event(type=EventType.HEARTBEAT, data=None))
    subscriber = SimpleNamespace(
        receive=AsyncMock(side_effect=[*events, asyncio.CancelledError()])
    )
    bus = MagicMock()
    bus.subscribe.return_value = subscriber
    monkeypatch.setattr(active_record, "event_bus", bus)
    response = await route.get_inference_backends(
        ctx=_context(scope), params=ListParams(watch=True)
    )
    frames = []
    with pytest.raises(asyncio.CancelledError):
        async for frame in response.body_iterator:
            frames.append(frame)
    assert frames[-1] == "\n\n"
    payloads = [json.loads(frame) for frame in frames[:-1]]
    assert [(event["type"], event["data"]["id"]) for event in payloads] == [
        (kind.value, row_id)
        for kind in (
            EventType.CREATED,
            EventType.CREATED,
            EventType.UPDATED,
            EventType.DELETED,
        )
        for row_id in _visible_ids(scope)
    ]
    cached_all.assert_awaited_once()
    bus.unsubscribe.assert_called_once_with("inferencebackend", subscriber)


@pytest.mark.parametrize("scope", ["org-a", "cluster", "admin-org", "cluster-no-owner"])
def test_backend_events_without_ownership_are_not_treated_as_global(scope):
    assert not route._is_backend_visible(_context(scope), SimpleNamespace(id=99))
