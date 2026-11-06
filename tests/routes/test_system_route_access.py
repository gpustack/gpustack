"""Management APIs admit service accounts only on explicit runtime endpoints."""

import re
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import APIRouter, Depends, FastAPI, Request, Response
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
import pytest

from gpustack.api.auth import get_current_user, require_system_route_access
from gpustack.api.exceptions import register_handlers
from gpustack.api.tenant import TenantContext, get_tenant_context
from gpustack.routes import cache_providers, model_instances
from gpustack.routes.routes import (
    CLUSTER_SYSTEM_ROUTES,
    SYSTEM_ROUTES,
    management_router,
    worker_client_router,
)
from gpustack.schemas.api_keys import PermissionScope
from gpustack.schemas.cache_providers import CacheProvider
from gpustack.schemas.principals import PrincipalType
from gpustack.server.db import get_session
from gpustack.worker.cache_provider_manager import CacheProviderManager


def _principal(kind):
    return SimpleNamespace(
        id=None if kind == "server" else 7,
        name="system/worker/test",
        kind=PrincipalType.USER if kind in ("user", "admin") else PrincipalType.SYSTEM,
        is_admin=kind == "admin",
        worker=SimpleNamespace(id=2, cluster_id=1) if kind == "worker" else None,
        cluster=SimpleNamespace(id=1) if kind == "cluster" else None,
    )


@pytest.mark.parametrize(
    "kind", ["worker", "cluster", "unlinked", "user", "admin", "server"]
)
@pytest.mark.parametrize(
    "method,path,allowed",
    [
        ("GET", "/runtime", True),
        ("POST", "/runtime", False),
        ("GET", "/new?allow_system=true", False),
        ("GET", "/overlap", False),
    ],
)
def test_gate_uses_method_and_dispatched_endpoint(kind, method, path, allowed):
    async def runtime():
        return {"ok": True}

    async def management():
        return {"ok": True}

    allowed_routes = {("GET", runtime)}
    gate = require_system_route_access(allowed_routes)
    router = APIRouter(dependencies=[Depends(gate)])
    router.add_api_route("/runtime", runtime, methods=["GET", "POST"])
    # Adding a route or mutating the source set cannot expand the gate's policy.
    allowed_routes.add(("GET", management))
    router.add_api_route("/new", management, methods=["GET"])
    # A later allowed route cannot authorize the first matching handler.
    router.add_api_route("/overlap", management, methods=["GET"])
    router.add_api_route("/overlap", runtime, methods=["GET"])
    app = FastAPI()
    register_handlers(app)
    app.include_router(router)
    app.dependency_overrides[get_current_user] = lambda: _principal(kind)
    with TestClient(app) as client:
        response = client.request(method, path)
    expected = 200 if allowed or kind in ("user", "admin", "server") else 403
    assert response.status_code == expected


# Deduplicate paths in production registration order, just as routing does.
PRODUCTION_ROUTES = {}
for route in management_router.routes:
    if isinstance(route, APIRoute):
        for method in sorted(route.methods):
            PRODUCTION_ROUTES.setdefault((method, route.path), route.endpoint)
DENIED_REQUESTS = [
    (kind, method, re.sub(r"\{[^}]+\}", "2", path))
    for kind in ("worker", "cluster")
    for (method, path), endpoint in PRODUCTION_ROUTES.items()
    if (method, endpoint)
    not in (
        SYSTEM_ROUTES | CLUSTER_SYSTEM_ROUTES if kind == "cluster" else SYSTEM_ROUTES
    )
]
RUNTIME_REQUESTS = [
    (method, re.sub(r"\{[^}]+\}", "2", path))
    for (method, path), endpoint in PRODUCTION_ROUTES.items()
    if (method, endpoint) in SYSTEM_ROUTES
]


@pytest.fixture(scope="module")
def management_app():
    app = FastAPI()
    register_handlers(app)
    app.include_router(management_router)
    return app


@pytest.fixture
def management_api(management_app, monkeypatch):
    user = _principal("worker")
    ctx = TenantContext(
        user=user,
        is_platform_admin=False,
        current_principal_id=None,
        org_role=None,
        scoped_cluster_id=1,
        scoped_cluster_owner_id=10,
    )
    handler = AsyncMock(return_value=Response(status_code=204))
    session = AsyncMock()
    for route in management_app.routes:
        if isinstance(route, APIRoute):
            # Preserve dependency solving and dispatch; isolate handler side effects.
            monkeypatch.setattr(route.dependant, "call", handler)
    management_app.dependency_overrides[get_current_user] = lambda: user
    management_app.dependency_overrides[get_tenant_context] = lambda: ctx
    management_app.dependency_overrides[get_session] = lambda: session
    yield SimpleNamespace(
        app=management_app, handler=handler, session=session, user=user
    )
    management_app.dependency_overrides.clear()


@pytest.mark.parametrize("kind,method,path", DENIED_REQUESTS)
def test_all_non_runtime_management_routes_deny_system(
    management_api, kind, method, path
):
    management_api.app.dependency_overrides[get_current_user] = lambda: _principal(kind)
    with TestClient(management_api.app) as client:
        response = client.request(method, path, json={})
    assert response.status_code == 403, response.text
    if method != "HEAD":
        assert (
            "Service accounts cannot access this management endpoint" in response.text
        )
    management_api.handler.assert_not_awaited()
    management_api.session.exec.assert_not_awaited()


@pytest.mark.parametrize("method,path", RUNTIME_REQUESTS)
def test_runtime_routes_pass_the_gate_with_production_dependencies(
    management_api, method, path
):
    with TestClient(management_api.app) as client:
        response = client.request(method, path, json={})
    # Required payload fields still undergo the endpoint's schema validation.
    assert response.status_code in (204, 422), response.text
    if response.status_code == 204:
        management_api.handler.assert_awaited_once()


def test_runtime_allowlist_preserves_api_key_scope_checks(management_api):
    def inference_key(request: Request):
        request.state.api_key = SimpleNamespace(scope=[PermissionScope.INFERENCE])
        return management_api.user

    management_api.app.dependency_overrides[get_current_user] = inference_key
    with TestClient(management_api.app) as client:
        response = client.get("/v2/metrics/config")
    assert response.status_code == 403
    assert "API key does not have permission" in response.text
    management_api.handler.assert_not_awaited()


def test_runtime_policy_entries_resolve_and_worker_router_has_no_management_routes():
    registered = {
        (method, endpoint) for (method, _), endpoint in PRODUCTION_ROUTES.items()
    }
    assert SYSTEM_ROUTES | CLUSTER_SYSTEM_ROUTES <= registered
    assert ("POST", model_instances.create_model_instance) in SYSTEM_ROUTES
    for route in worker_client_router.routes:
        assert all(
            (method, route.endpoint) in SYSTEM_ROUTES for method in route.methods
        )


# Contract from worker call sites, independent of SYSTEM_ROUTES. Keep additions
# tied to a worker operation so a missing allowlist entry cannot hide its test.
WORKER_REQUESTS = [
    # worker.py, worker_manager.py and WorkerStatusClient: registration and liveness.
    ("POST", "/workers"),
    ("GET", "/workers"),
    ("GET", "/workers/2"),
    ("POST", "/worker-status"),
    ("POST", "/worker-heartbeat"),
    # serve_manager.py, backends/base.py and workload_cleaner.py.
    ("GET", "/models"),
    ("GET", "/models?watch=true"),
    ("GET", "/models/2"),
    ("PUT", "/models/2"),
    ("GET", "/model-instances"),
    ("GET", "/model-instances?watch=true"),
    ("GET", "/model-instances/2"),
    ("PUT", "/model-instances/2"),
    ("DELETE", "/model-instances/2"),
    # serve_manager.py through DistributedPortClient.
    ("GET", "/model-instances/2/port-reservations"),
    ("POST", "/model-instances/2/port-reservations"),
    ("DELETE", "/model-instances/2/port-reservations"),
    # model_file_manager.py.
    ("GET", "/model-files?watch=true"),
    ("GET", "/model-files/2"),
    ("PUT", "/model-files/2"),
    # cache_service_manager.py and cache_provider_manager.py.
    ("GET", "/cache-providers?page=0"),
    ("GET", "/cache-services/2"),
    ("GET", "/cache-service-instances"),
    ("GET", "/cache-service-instances?watch=true"),
    ("GET", "/cache-service-instances/2"),
    ("PUT", "/cache-service-instances/2"),
    # benchmark_manager.py and benchmark/runner.py.
    ("GET", "/benchmarks"),
    ("GET", "/benchmarks?watch=true"),
    ("GET", "/benchmarks/2"),
    ("PATCH", "/benchmarks/2/state"),
    ("POST", "/benchmarks/2/metrics"),
    ("POST", "/benchmarks/2/results"),
    # inference_backend_manager.py, backends/base.py, runtime_metrics_aggregator.py.
    ("GET", "/inference-backends/all"),
    ("GET", "/inference-backends?watch=true"),
    ("GET", "/runner-override-entries"),
    ("GET", "/metrics/config"),
]


@pytest.mark.parametrize("method,path", WORKER_REQUESTS)
def test_worker_call_contract_passes_production_auth(management_api, method, path):
    with TestClient(management_api.app) as client:
        response = client.request(method, f"/v2{path}", json={})
    assert response.status_code in (204, 422), response.text
    if response.status_code == 204:
        management_api.handler.assert_awaited_once()


@pytest.mark.parametrize("kind", ["worker", "cluster", "unlinked", "user", "server"])
@pytest.mark.parametrize("path", ["manifests", "registration-token"])
def test_cluster_secret_routes_require_their_intended_identity(
    management_api, kind, path
):
    management_api.app.dependency_overrides[get_current_user] = lambda: _principal(kind)
    with TestClient(management_api.app) as client:
        response = client.get(f"/v2/clusters/1/{path}")
    allowed = kind in ("user", "server") or (kind == "cluster" and path == "manifests")
    assert response.status_code == (204 if allowed else 403), response.text
    if not allowed:
        management_api.handler.assert_not_awaited()


def test_worker_catalog_sync_reaches_the_real_catalog_handler(monkeypatch):
    app = FastAPI()
    register_handlers(app)
    app.include_router(management_router)
    app.dependency_overrides[get_current_user] = lambda: _principal("worker")
    app.dependency_overrides[get_session] = lambda: AsyncMock()
    provider = CacheProvider.model_validate(
        {
            "name": "demo",
            "display_name": "Demo",
            "topology": "per_node",
            "default_version": "v1",
            "default_image": "demo:v1",
            "versions": {"v1": {}},
            "default_run_command": "demo --port {{port}}",
        }
    )
    lookup = AsyncMock(return_value=[provider])
    monkeypatch.setattr(cache_providers, "get_cache_providers", lookup)
    with TestClient(app, base_url="http://testserver/v2/") as client:
        clientset = SimpleNamespace(
            http_client=SimpleNamespace(get_httpx_client=lambda: client)
        )
        manager = CacheProviderManager(lambda: clientset)
        manager.sync()
        assert manager.loaded
        assert manager.lookup("demo") == (provider, True)
    lookup.assert_awaited_once()
