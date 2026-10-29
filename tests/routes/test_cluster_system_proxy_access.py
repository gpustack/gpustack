"""Controller cluster credentials reach Kubernetes only through their own proxy."""

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from kubernetes_asyncio.client.exceptions import ApiException

from gpustack.api.auth import get_current_user
from gpustack.api.exceptions import register_handlers
from gpustack.gpu_instances.cluster_apis import ClusterOps
from gpustack.k8s.pod_security import NAMESPACE_LABELS
from gpustack.routes import clusters
from gpustack.routes.routes import management_router
from gpustack.schemas.clusters import ClusterProvider
from gpustack.schemas.principals import PrincipalType


@pytest.fixture(scope="module")
def proxy_app():
    app = FastAPI()
    register_handlers(app)
    app.include_router(management_router)
    app.state.http_client = app.state.http_client_no_proxy = None
    return app


@pytest.fixture
def proxy_api(proxy_app, monkeypatch):
    scope = SimpleNamespace(id=1, owner_principal_id=10)
    principals = {
        kind: SimpleNamespace(
            id=7,
            name=f"system/{kind}/test",
            kind=PrincipalType.SYSTEM,
            is_admin=False,
            cluster=scope if kind == "cluster" else None,
            worker=(
                SimpleNamespace(id=3, cluster_id=1, owner_principal_id=10)
                if kind == "worker"
                else None
            ),
        )
        for kind in ("cluster", "worker")
    }

    def authenticate(request: Request):
        return principals[request.headers["Authorization"].removeprefix("Bearer ")]

    proxy_app.dependency_overrides[get_current_user] = authenticate
    session = MagicMock()

    @asynccontextmanager
    async def open_session():
        yield session

    monkeypatch.setattr(clusters, "async_session", open_session)
    target = SimpleNamespace(
        id=1,
        name="target",
        owner_principal_id=10,
        deleted_at=None,
        provider=ClusterProvider.Kubernetes,
    )
    lookup = AsyncMock(return_value=target)
    workers = AsyncMock(return_value=[SimpleNamespace(id=3)])
    monkeypatch.setattr(clusters.Cluster, "one_by_id", lookup)
    monkeypatch.setattr(clusters.Worker, "all_by_fields", workers)
    forwarded = MagicMock()

    async def forward(**kwargs):
        forwarded(**kwargs)
        if kwargs["path"].startswith("cluster-proxy/api/v1/namespaces"):
            payload = {
                "apiVersion": "v1",
                "kind": "Namespace",
                "metadata": {"name": "gpustack-owner", "labels": NAMESPACE_LABELS},
            }
        elif kwargs["method"] == "GET":
            payload = {"items": [{"metadata": {"name": "gpu"}}]}
        else:
            payload = json.loads(kwargs["data"] or b"{}")
        status = 201 if kwargs["method"] == "POST" else 200
        yield json.dumps(payload), {"content-type": "application/json"}, status

    monkeypatch.setattr(clusters, "stream_to_worker", forward)
    yield SimpleNamespace(
        app=proxy_app,
        target=target,
        lookup=lookup,
        workers=workers,
        forwarded=forwarded,
    )
    proxy_app.dependency_overrides.clear()


@pytest.mark.parametrize("kind", ["cluster", "worker"])
@pytest.mark.parametrize("method", ["GET", "POST", "PUT", "PATCH", "DELETE"])
@pytest.mark.parametrize("target_id,owner_id", [(1, 10), (2, 10), (2, 20)])
@pytest.mark.parametrize(
    "path",
    [
        "apis/worker.gpustack.ai/v1/instancetypes",
        "apis/worker.gpustack.ai/v1/namespaces/gpustack-tenant-b/instances",
    ],
)
def test_controller_proxy_requires_cluster_identity_and_matching_cluster(
    proxy_api, kind, method, target_id, owner_id, path
):
    api = proxy_api
    api.target.id, api.target.owner_principal_id = target_id, owner_id
    with TestClient(api.app) as client:
        response = client.request(
            method,
            f"/v2/clusters/{target_id}/proxy/{path}",
            headers={"Authorization": f"Bearer {kind}", "X-Organization-Id": "20"},
            json={"metadata": {"name": "demo"}},
            params=(
                {"watch": "true", "resourceVersion": "123"} if method == "GET" else None
            ),
        )
    if kind == "worker":
        assert response.status_code == 403
        api.lookup.assert_not_awaited()
    elif target_id != 1:
        assert response.status_code == 404
    else:
        assert response.status_code == (201 if method == "POST" else 200), response.text
        api.forwarded.assert_called_once()
        call = api.forwarded.call_args.kwargs
        assert (call["method"], call["path"]) == (method, f"cluster-proxy/{path}")
        if method == "GET":
            assert call["params"] == {"watch": "true", "resourceVersion": "123"}
        else:
            assert json.loads(call["data"]) == {"metadata": {"name": "demo"}}
        return
    api.workers.assert_not_awaited()
    api.forwarded.assert_not_called()


@pytest.mark.asyncio
async def test_cluster_ops_controller_calls_pass_production_proxy_auth(
    proxy_api, monkeypatch
):
    """Exercise the real Kubernetes client paths and registration-token header."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=proxy_api.app)
    ) as http:
        async with ClusterOps(
            server_api_port=1,
            cluster_id=1,
            cluster_registration_token="cluster",
            cluster_owner_principal_identifier="owner",
        ) as ops:

            async def request(
                method, url, query_params=None, headers=None, body=None, **kwargs
            ):
                response = await http.request(
                    method, url, params=query_params, headers=headers, json=body
                )
                result = SimpleNamespace(
                    status=response.status_code,
                    reason=response.reason_phrase,
                    data=response.content,
                    getheader=response.headers.get,
                    getheaders=lambda: response.headers,
                )
                if response.is_error:
                    raise ApiException(http_resp=result)
                return result

            monkeypatch.setattr(ops.api_client, "request", request)
            assert await ops.list_instance_types() == {
                "items": [{"metadata": {"name": "gpu"}}]
            }
            created = await ops.create_instance(
                {"metadata": {"name": "demo"}, "spec": {}}, ignore_existed=False
            )
            assert created["metadata"]["namespace"] == ops.org_namespace
            assert (await ops.stop_instance("demo"))["spec"] == {"stop": True}
            assert await ops.delete_instance("demo")

    calls = {
        (c.kwargs["method"], c.kwargs["path"])
        for c in proxy_api.forwarded.call_args_list
    }
    root = "cluster-proxy/apis/worker.gpustack.ai/v1"
    assert ("GET", f"{root}/instancetypes") in calls
    assert ("POST", f"{root}/namespaces/gpustack-owner/instances") in calls
    assert ("PATCH", f"{root}/namespaces/gpustack-owner/instances/demo") in calls
    assert ("DELETE", f"{root}/namespaces/gpustack-owner/instances/demo") in calls
