"""Provider and route management reject cluster and worker credentials."""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from gpustack.api.auth import get_current_user
from gpustack.api.exceptions import register_handlers
from gpustack.api.tenant import TenantContext, get_tenant_context
from gpustack.routes import benchmarks, model_provider, model_routes
from gpustack.routes.routes import management_router
from gpustack.schemas.principals import OrgRole, PrincipalType
from gpustack.server.db import get_session

MANAGEMENT_ROUTERS = [
    (model_provider.router, "/v2/model-providers"),
    (model_routes.router, "/v2/model-routes"),
    (model_routes.target_router, "/v2/model-route-targets"),
]
MANAGEMENT_REQUESTS = [
    (method, prefix + route.path.replace("{id}", "2"))
    for router, prefix in MANAGEMENT_ROUTERS
    for route in router.routes
    for method in sorted(route.methods)
] + [("GET", prefix + "?watch=true") for _, prefix in MANAGEMENT_ROUTERS]


def _context(kind):
    user = SimpleNamespace(
        id=None if kind == "server" else 7,
        name="system/worker/test",
        kind=(
            PrincipalType.SYSTEM
            if kind in ("cluster", "worker", "server")
            else PrincipalType.USER
        ),
        is_admin=kind == "admin",
        worker=SimpleNamespace(id=11, cluster_id=1) if kind == "worker" else None,
        cluster=SimpleNamespace(id=1) if kind == "cluster" else None,
    )
    return TenantContext(
        user=user,
        is_platform_admin=kind == "admin",
        current_principal_id=10 if kind in ("owner", "member") else None,
        org_role=(
            OrgRole.OWNER
            if kind == "owner"
            else OrgRole.MEMBER if kind == "member" else None
        ),
        scoped_cluster_id=1 if kind in ("cluster", "worker") else None,
        scoped_cluster_owner_id=10 if kind in ("cluster", "worker") else None,
    )


@pytest.fixture(scope="module")
def management_app():
    app = FastAPI()
    register_handlers(app)
    # Exercise the production management gate and route-level role checks.
    app.include_router(management_router)
    return app


@pytest.fixture
def management_api(management_app, monkeypatch):
    session = AsyncMock()
    lookup = AsyncMock(return_value=None)
    for model in (
        model_provider.ModelProvider,
        model_routes.ModelRoute,
        model_routes.ModelRouteTarget,
        benchmarks.Benchmark,
    ):
        monkeypatch.setattr(model, "one_by_id", lookup)

    @asynccontextmanager
    async def open_session():
        yield session

    monkeypatch.setattr(model_provider, "async_session", open_session)
    monkeypatch.setattr(model_routes, "async_session", open_session)
    management_app.dependency_overrides[get_session] = lambda: session
    yield SimpleNamespace(app=management_app, lookup=lookup, session=session)
    management_app.dependency_overrides.clear()


def _authenticate(api, kind):
    ctx = _context(kind)
    api.app.dependency_overrides[get_current_user] = lambda: ctx.user
    api.app.dependency_overrides[get_tenant_context] = lambda: ctx


@pytest.mark.parametrize("kind", ["cluster", "worker"])
@pytest.mark.parametrize("method,path", MANAGEMENT_REQUESTS)
def test_service_accounts_cannot_reach_management_handlers(
    management_api, kind, method, path
):
    _authenticate(management_api, kind)
    with TestClient(management_api.app) as client:
        response = client.request(method, path, json={})
    assert response.status_code == 403
    management_api.lookup.assert_not_awaited()
    management_api.session.exec.assert_not_awaited()
    management_api.session.commit.assert_not_awaited()


@pytest.mark.parametrize("kind", ["owner", "admin", "server", "member"])
@pytest.mark.parametrize("prefix", [prefix for _, prefix in MANAGEMENT_ROUTERS])
def test_management_permissions_preserve_user_roles_and_platform_token(
    management_api, kind, prefix
):
    _authenticate(management_api, kind)
    with TestClient(management_api.app) as client:
        response = client.delete(f"{prefix}/2")
    if kind == "member":
        assert response.status_code == 403
        management_api.lookup.assert_not_awaited()
    else:
        # The allowed caller reaches the resource lookup and its missing-row check.
        assert response.status_code == 404
        management_api.lookup.assert_awaited_once()


@pytest.mark.parametrize("kind", ["worker", "cluster", "server"])
@pytest.mark.parametrize(
    "method,suffix,payload",
    [("PATCH", "state", {}), ("POST", "metrics", {}), ("POST", "results", [])],
)
def test_runtime_callbacks_still_admit_service_accounts(
    management_api, kind, method, suffix, payload
):
    _authenticate(management_api, kind)
    with TestClient(management_api.app) as client:
        response = client.request(method, f"/v2/benchmarks/2/{suffix}", json=payload)
    assert response.status_code == 404
    management_api.lookup.assert_awaited_once()
