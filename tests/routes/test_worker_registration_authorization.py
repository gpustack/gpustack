from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from gpustack.api.auth import get_current_user
from gpustack.api.exceptions import InvalidException, register_handlers
from gpustack.routes import workers
from gpustack.schemas.clusters import Cluster, ClusterProvider, ClusterStateEnum
from gpustack.schemas.principals import PrincipalType
from gpustack.schemas.workers import Worker, WorkerCreate, WorkerStatus
from gpustack.server import services


def _principal(kind, worker_row_id=20):
    return SimpleNamespace(
        id=None if kind == "server" else 10,
        name="system/worker/audit",
        kind=PrincipalType.USER if kind in ("admin", "user") else PrincipalType.SYSTEM,
        is_admin=kind == "admin",
        # id 20 links the principal to its own worker row (the one the
        # reregistration fixtures below match by name).
        worker=(
            SimpleNamespace(cluster_id=100, id=worker_row_id)
            if kind == "worker"
            else None
        ),
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


def test_registration_cannot_reregister_another_workers_record(registration):
    # Cluster scope alone does not authorize a worker credential to adopt
    # a record it can name (by uuid, or by name plus the self-reported
    # existence-check label): the matched record must be its own.
    registration.app.dependency_overrides[get_current_user] = lambda: _principal(
        "worker", worker_row_id=999
    )
    cluster = Cluster(id=100, name="test-cluster", provider=ClusterProvider.Docker)
    registration.get_cluster.return_value = cluster
    worker = Worker.model_validate(
        {
            **_payload(100, True),
            "id": 20,
            "token": "synthetic-worker-token",
            "system_principal_id": 30,
        }
    )
    registration.list_workers.return_value = [worker]
    registration.get_worker.return_value = worker
    registration.get_principal.return_value = SimpleNamespace(api_keys=[object()])

    with TestClient(registration.app) as client:
        response = client.post("/v2/workers", json=_payload(100, True))

    assert response.status_code == 403
    registration.persist.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cluster_id,new_owner,allowed",
    [(100, 99, True), (100, 777, False), (200, 99, False)],
)
async def test_reregistration_reconciles_only_authorized_cluster_owner(
    monkeypatch, cluster_id, new_owner, allowed
):
    existing = Worker(id=20, name="test-worker", cluster_id=100, owner_principal_id=8)
    incoming = Worker(
        id=20, name="test-worker", cluster_id=100, owner_principal_id=new_owner
    )
    cluster = Cluster(
        id=cluster_id,
        name="test-cluster",
        owner_principal_id=99,
        state=ClusterStateEnum.READY,
    )
    session = SimpleNamespace(commit=AsyncMock())
    save = AsyncMock()
    monkeypatch.setattr(Worker, "save", save)
    monkeypatch.setattr(Worker, "_publish_event_after_commit", MagicMock())
    monkeypatch.setattr(services, "delete_cache_by_key", AsyncMock())

    async def register():
        return await workers._persist_worker_registration(
            session,
            existing_worker=existing,
            new_worker=incoming,
            new_token="unused-test-token",
            to_create_principal=None,
            existing_principal=None,
            to_create_apikey=None,
            all_workers=[existing],
            cluster=cluster,
        )

    if allowed:
        result = await register()
        assert result.owner_principal_id == cluster.owner_principal_id
        save.assert_awaited_once_with(session, auto_commit=False)
        session.commit.assert_awaited_once()
        with pytest.raises(InvalidException):
            await existing.update(session, {"owner_principal_id": 777})
    else:
        with pytest.raises(InvalidException):
            await register()
        assert existing.owner_principal_id == 8
        save.assert_not_awaited()
        session.commit.assert_not_awaited()
