"""G4 (gpustack#5947): a cluster without a reachable worker is an expected
condition, not a server fault.

The gpustack-operator polls this proxy for worker API-service readiness, so
logging each 503 at ERROR turns a normal wait into a log flood. The 503 itself —
status, reason and message — must not change.
"""

import contextlib
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI

from gpustack.api.exceptions import (
    HTTPException,
    ForbiddenException,
    InternalServerErrorException,
    ServiceUnavailableException,
    register_handlers,
)
from gpustack.api.tenant import TenantContext
from gpustack.routes import clusters as clusters_route
from gpustack.schemas.clusters import ClusterProvider
from gpustack.schemas.principals import PrincipalType

CLUSTER_ID = 1
PROXY_PATH = "apis/apiregistration.k8s.io/v1/apiservices/v1.worker.gpustack.ai"
NO_WORKERS_MESSAGE = f"No reachable workers in cluster default(id: {CLUSTER_ID})"
EXCEPTIONS_LOGGER = "gpustack.api.exceptions"


def _admin_ctx() -> TenantContext:
    user = MagicMock()
    user.kind = PrincipalType.USER
    return TenantContext(
        user=user,
        is_platform_admin=True,
        current_principal_id=1,
        org_role=None,
    )


def _empty_cluster(monkeypatch):
    """A Kubernetes cluster the caller may manage, with no READY worker."""
    cluster = SimpleNamespace(
        id=CLUSTER_ID,
        name="default",
        deleted_at=None,
        owner_principal_id=1,
        provider=ClusterProvider.Kubernetes,
    )

    @contextlib.asynccontextmanager
    async def _session():
        yield MagicMock()

    monkeypatch.setattr(clusters_route, "async_session", _session)
    monkeypatch.setattr(
        clusters_route.Cluster, "one_by_id", AsyncMock(return_value=cluster)
    )
    monkeypatch.setattr(
        clusters_route.Worker, "all_by_fields", AsyncMock(return_value=[])
    )


def _handler():
    app = FastAPI()
    register_handlers(app)
    return app.exception_handlers[HTTPException]


def _request():
    request = MagicMock()
    request.url.path = f"/v2/clusters/{CLUSTER_ID}/proxy/{PROXY_PATH}"
    request.method = "GET"
    return request


@pytest.mark.asyncio
async def test_no_reachable_workers_is_answered_without_an_error_record(
    monkeypatch, caplog
):
    """The whole path: the route raises, the registered handler answers, nothing at ERROR."""
    _empty_cluster(monkeypatch)

    with pytest.raises(ServiceUnavailableException) as excinfo:
        await clusters_route.cluster_apiserver_proxy(
            request=MagicMock(),
            ctx=_admin_ctx(),
            id=CLUSTER_ID,
            path=PROXY_PATH,
        )

    with caplog.at_level(logging.DEBUG, logger=EXCEPTIONS_LOGGER):
        response = await _handler()(_request(), excinfo.value)

    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []
    assert response.status_code == 503
    assert json.loads(response.body) == {
        "code": 503,
        "reason": "ServiceUnavailable",
        "message": NO_WORKERS_MESSAGE,
    }


@pytest.mark.asyncio
async def test_handler_still_logs_an_unmarked_server_error(caplog):
    """The quiet path is opt-in: a plain 5xx keeps its ERROR record."""
    handler = _handler()

    with caplog.at_level(logging.DEBUG, logger=EXCEPTIONS_LOGGER):
        await handler(_request(), InternalServerErrorException("boom"))

    assert [r for r in caplog.records if r.levelno >= logging.ERROR] != []


@pytest.mark.parametrize(
    "raw",
    [
        # Literal dot segment, as it arrives when the server port is reached
        # directly (the gateway normalizes this form before it reaches us).
        "apis/worker.gpustack.ai/v1/namespaces/own/../kube-system/secrets",
        # Single-encoded dot segment after the gateway normalized plain ones:
        # one decode round in _canonicalize_proxy_path yields "..".
        "apis/worker.gpustack.ai/v1/namespaces/own/%2e%2e/kube-system/secrets",
        # Double-encoded dot segment, which the gateway forwards verbatim;
        # Starlette has already decoded it once, so the tail arrives as the
        # single-encoded form above. Spelled out anyway to pin the behavior.
        "apis/worker.gpustack.ai/v1/namespaces/own/%252e%252e/kube-system/secrets",
        # Trailing escapes that survive a decode round, and backslashes.
        "apis/worker.gpustack.ai/v1%00/namespaces/own/secrets",
        "apis\\worker.gpustack.ai\\v1\\namespaces\\own\\secrets",
        # Decoded URL delimiters: '#' or '?' reaching the canonicalizer after
        # Starlette's own decode round (wire form %23 / %3f, or the literal
        # characters for a direct caller) authorizes the suffix as a
        # namespace segment while the forwarding URL library treats it as a
        # fragment or query, turning a namespaced read into an all-namespace
        # list. The double-encoded forms are included as the wire shape a
        # gateway passes through verbatim; after Starlette's decode round
        # they arrive here as the single-encoded forms above.
        "apis/worker.gpustack.ai/v1/instances%23/namespaces/gpustack-user-6",
        "apis/worker.gpustack.ai/v1/instances%3fnamespaces=gpustack-user-6",
        "apis/worker.gpustack.ai/v1/instances#/namespaces/gpustack-user-6",
        "apis/worker.gpustack.ai/v1/instances?namespaces=gpustack-user-6",
        "apis/worker.gpustack.ai/v1/instances%2523/namespaces/gpustack-user-6",
    ],
)
@pytest.mark.asyncio
async def test_traversal_shaped_tails_are_rejected_before_any_lookup(raw):
    """The authorized path and the forwarded path must be the same plain
    Kubernetes API path: a tail that still looks like traversal or leftover
    encoding after one decode round is refused before any cluster lookup, so
    no caller class (manager included) can pass the scope check as one path
    and reach the API server as another."""
    with pytest.raises(ForbiddenException):
        await clusters_route.cluster_apiserver_proxy(
            request=MagicMock(),
            ctx=_admin_ctx(),
            id=CLUSTER_ID,
            path=raw,
        )


@pytest.mark.parametrize(
    "raw,canonical",
    [
        (
            "apis/worker.gpustack.ai/v1/namespaces/own/gpuinstances",
            "apis/worker.gpustack.ai/v1/namespaces/own/gpuinstances",
        ),
        # Empty segments collapse; leading slashes are not significant.
        (
            "/apis//worker.gpustack.ai/v1/namespaces/own",
            "apis/worker.gpustack.ai/v1/namespaces/own",
        ),
        # A percent-escape that decodes to an ordinary character is fine.
        (
            "apis/worker.gpustack.ai/v1/namespaces/ow%6e/secrets",
            "apis/worker.gpustack.ai/v1/namespaces/own/secrets",
        ),
    ],
)
def test_plain_tails_canonicalize_to_the_forwarded_form(raw, canonical):
    assert clusters_route._canonicalize_proxy_path(raw) == canonical
