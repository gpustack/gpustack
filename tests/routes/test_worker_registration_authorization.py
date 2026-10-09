from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from gpustack.api.auth import get_current_user
from gpustack.api.exceptions import register_handlers
from gpustack.routes import workers
from gpustack.schemas.clusters import Cluster, ClusterProvider
from gpustack.schemas.principals import PrincipalType
from gpustack.schemas.workers import Worker, WorkerCreate, WorkerStatus


def _principal(kind):
    return SimpleNamespace(
        id=None if kind == "server" else 10,
        name="system/worker/audit",
        kind=PrincipalType.USER if kind in ("admin", "user") else PrincipalType.SYSTEM,
        is_admin=kind == "admin",
        worker=SimpleNamespace(cluster_id=100) if kind == "worker" else None,
        cluster=SimpleNamespace(id=100) if kind == "cluster" else None,
    )


def _payload(cluster_id, existing):
    return WorkerCreate(
        name="test-worker",
        cluster_id=cluster_id,
        hostname="test-host",
        ip="192.0.2.10",
        ifname="eth0",
        port=10150,
        worker_uuid="",
        status=WorkerStatus.get_default_status(),
        labels={"gpustack.existence-check": "true"} if existing else {},
    ).model_dump(mode="json")


@pytest.fixture
def registration(monkeypatch):
    session = AsyncMock()

    @asynccontextmanager
    async def session_context():
        yield session

    def persisted_worker(_session, **kwargs):
        worker = kwargs["new_worker"]
        worker.id = 20
        worker.created_at = worker.updated_at = datetime.now(timezone.utc)
        return worker

    mocks = SimpleNamespace(
        open_session=MagicMock(side_effect=session_context),
        list_workers=AsyncMock(return_value=[]),
        get_worker=AsyncMock(),
        get_cluster=AsyncMock(),
        get_principal=AsyncMock(return_value=None),
        persist=AsyncMock(side_effect=persisted_worker),
    )
    monkeypatch.setattr(workers, "async_session", mocks.open_session)
    monkeypatch.setattr(Worker, "all_by_fields", mocks.list_workers)
    monkeypatch.setattr(Worker, "one_by_id", mocks.get_worker)
    monkeypatch.setattr(Cluster, "one_by_id", mocks.get_cluster)
    monkeypatch.setattr(
        workers, "_resolve_existing_worker_principal", mocks.get_principal
    )
    monkeypatch.setattr(workers, "_persist_worker_registration", mocks.persist)
    app = FastAPI()
    register_handlers(app)
    app.include_router(workers.router, prefix="/v2/workers")
    mocks.app = app
    return mocks


@pytest.mark.parametrize("principal_kind", ["worker", "cluster", "unlinked", "user"])
@pytest.mark.parametrize("existing", [False, True], ids=["new", "reregister"])
def test_registration_denies_other_cluster_before_database_access(
    registration, principal_kind, existing
):
    registration.app.dependency_overrides[get_current_user] = lambda: _principal(
        principal_kind
    )
    with TestClient(registration.app) as client:
        response = client.post("/v2/workers", json=_payload(200, existing))

    assert response.status_code == 403
    assert "token" not in response.json()
    registration.open_session.assert_not_called()
    registration.list_workers.assert_not_awaited()
    registration.persist.assert_not_awaited()


@pytest.mark.parametrize(
    "principal_kind,requested_cluster_id,target_cluster_id",
    [
        ("worker", 100, 100),
        ("worker", None, 100),
        ("cluster", 100, 100),
        ("cluster", None, 100),
        ("admin", 200, 200),
        ("server", 200, 200),
    ],
)
@pytest.mark.parametrize("existing", [False, True], ids=["new", "reregister"])
def test_registration_allows_authorized_cluster(
    registration, principal_kind, requested_cluster_id, target_cluster_id, existing
):
    registration.app.dependency_overrides[get_current_user] = lambda: _principal(
        principal_kind
    )
    cluster = Cluster(
        id=target_cluster_id, name="test-cluster", provider=ClusterProvider.Docker
    )
    registration.get_cluster.return_value = cluster
    if existing:
        worker = Worker.model_validate(
            {
                **_payload(target_cluster_id, True),
                "id": 20,
                "token": "synthetic-worker-token",
                "system_principal_id": 30,
            }
        )
        registration.list_workers.return_value = [worker]
        registration.get_worker.return_value = worker
        registration.get_principal.return_value = SimpleNamespace(api_keys=[object()])

    with TestClient(registration.app) as client:
        response = client.post(
            "/v2/workers", json=_payload(requested_cluster_id, existing)
        )

    assert response.status_code == 200, response.text
    assert response.json()["cluster_id"] == target_cluster_id
    registration.persist.assert_awaited_once()
    new_worker = registration.persist.await_args.kwargs["new_worker"]
    assert new_worker.cluster_id == target_cluster_id
    if existing:
        assert response.json()["token"] == "synthetic-worker-token"


@pytest.mark.parametrize("principal_kind", ["admin", "server", "unlinked"])
def test_registration_requires_target_cluster(registration, principal_kind):
    registration.app.dependency_overrides[get_current_user] = lambda: _principal(
        principal_kind
    )
    with TestClient(registration.app) as client:
        response = client.post("/v2/workers", json=_payload(None, False))

    assert response.status_code == 403
    registration.open_session.assert_not_called()
