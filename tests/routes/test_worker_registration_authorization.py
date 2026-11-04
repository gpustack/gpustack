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
        session=session,
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
    monkeypatch.setattr(
        workers.ModelInstance, "all_by_field", AsyncMock(return_value=[])
    )
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
@pytest.mark.parametrize("version", ["2.3.1", "2.3.0"])
def test_registration_allows_authorized_cluster(
    registration,
    principal_kind,
    requested_cluster_id,
    target_cluster_id,
    existing,
    version,
    monkeypatch,
):
    monkeypatch.setattr(
        "gpustack.extension.resolve_version_info", lambda: ("2.3.1-ent1", "test")
    )
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

    payload = _payload(requested_cluster_id, existing)
    payload["worker_version"] = version
    with TestClient(registration.app) as client:
        response = client.post("/v2/workers", json=payload)

    assert response.status_code == 200, response.text
    assert response.json()["cluster_id"] == target_cluster_id
    assert response.json()["worker_version"] == version
    registration.persist.assert_awaited_once()
    new_worker = registration.persist.await_args.kwargs["new_worker"]
    assert new_worker.cluster_id == target_cluster_id
    new_worker.heartbeat_time = datetime.now(timezone.utc)
    new_worker.state_message = None
    new_worker.compute_state()
    assert new_worker.state == "ready"
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


def test_reregistration_rechecks_cluster_after_worker_lock(registration):
    api = registration
    api.app.dependency_overrides[get_current_user] = lambda: _principal('worker')
    discovered = Worker.model_validate(
        {
            **_payload(100, True),
            'id': 20,
            'token': 'synthetic-token',
            'system_principal_id': 30,
        }
    )
    current = discovered.model_copy(deep=True)
    current.cluster_id = 200
    api.list_workers.return_value = [discovered]
    api.get_worker.return_value = current
    with TestClient(api.app) as client:
        response = client.post('/v2/workers', json=_payload(100, True))
    assert response.status_code == 403
    assert 'Worker no longer belongs to this cluster' in response.text
    api.get_cluster.assert_not_awaited()
    api.persist.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'binding', ['main', 'subordinate', 'workload', 'legacy_router', 'none']
)
async def test_reregistration_locks_only_complete_related_model_cohorts(
    monkeypatch, binding
):
    from gpustack.schemas.models import (
        DistributedServers,
        ModelInstanceSubordinateWorker,
    )
    from tests.utils.model import new_model_instance, new_model_instance_workload

    worker = Worker(id=20, name='target', cluster_id=100)
    target = new_model_instance(
        1, 'target', 5, worker_id=20 if binding in ('main', 'legacy_router') else 7
    )
    target.distributed_servers = DistributedServers(
        workloads=[] if binding != 'legacy_router' else None
    )
    if binding == 'subordinate':
        target.distributed_servers.subordinate_workers = [
            ModelInstanceSubordinateWorker(worker_id=20)
        ]
    elif binding == 'workload':
        target.distributed_servers = DistributedServers(
            workloads=[new_model_instance_workload(worker_id=20)]
        )
    peer = new_model_instance(2, 'peer', 5, worker_id=8)
    peer.distributed_servers = DistributedServers(workloads=[])
    unrelated = new_model_instance(3, 'unrelated', 9, worker_id=99)
    second = new_model_instance(4, 'second-model', 1, worker_id=20)
    late = new_model_instance(5, 'late-peer', 5, worker_id=8)
    discovery = [unrelated] if binding == 'none' else [peer, unrelated, target, second]
    calls = []

    async def query(session, field, value, **kwargs):
        calls.append((field, value, kwargs))
        assert field == 'cluster_id' and value == 100 and kwargs == {'refresh': True}
        return discovery

    monkeypatch.setattr(workers.ModelInstance, 'all_by_field', query)
    session = MagicMock()
    result = MagicMock()
    result.all.return_value = [target, peer, second, late]
    session.exec = AsyncMock(return_value=result)
    rows = await workers._registration_instances(session, worker)
    assert rows == ([] if binding == 'none' else [target, peer, second, late])
    assert len(calls) == 1
    if binding == 'none':
        session.exec.assert_not_awaited()
    else:
        session.exec.assert_awaited_once()
        statement = session.exec.call_args.args[0]
        assert statement.compile().params == {'model_id_1': [1, 5]}
        assert str(statement).endswith('ORDER BY model_instances.id FOR UPDATE')
        assert statement.get_execution_options() == {
            'populate_existing': True,
            'autoflush': False,
        }


def test_registration_closes_discovery_before_locking_worker(registration, monkeypatch):
    api = registration
    api.app.dependency_overrides[get_current_user] = lambda: _principal('worker')
    worker = Worker.model_validate(
        {
            **_payload(100, True),
            'id': 20,
            'token': 'synthetic-token',
            'system_principal_id': 30,
        }
    )
    api.list_workers.return_value = [worker]
    api.get_cluster.return_value = Cluster(
        id=100, name='cluster', provider=ClusterProvider.Docker
    )
    api.get_principal.return_value = SimpleNamespace(api_keys=[object()])
    sessions, closed = [], []

    @asynccontextmanager
    async def session_context():
        assert len(sessions) == len(closed)
        session = AsyncMock()
        sessions.append(session)
        try:
            yield session
        finally:
            closed.append(session)

    async def lock(session, identity, **kwargs):
        assert len(closed) == 1 and session is sessions[1]
        assert kwargs == {'for_update': True, 'refresh': True}
        return worker

    api.get_worker.side_effect = lock
    monkeypatch.setattr(workers, 'async_session', session_context)
    with TestClient(api.app) as client:
        response = client.post('/v2/workers', json=_payload(100, True))
    assert response.status_code == 200, response.text
    assert len(sessions) == len(closed) == 2
    assert api.list_workers.await_args.args[0] is sessions[0]
    assert api.persist.await_args.args[0] is sessions[1]


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


@pytest.mark.asyncio
@pytest.mark.parametrize('version', [None, 'unknown', '2.3.0', '2.3.1'])
async def test_version_mismatch_reregistration_preserves_identity_and_assignments(
    monkeypatch, version
):
    monkeypatch.setattr(
        'gpustack.extension.resolve_version_info', lambda: ('2.3.1-ent1', 'test')
    )
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlmodel.ext.asyncio.session import AsyncSession
    from gpustack.schemas.models import (
        Model,
        ModelInstance,
        DistributedServers,
        SourceEnum,
        BackendEnum,
    )
    from gpustack.schemas.workers import WorkerStateEnum
    from gpustack.server import worker_status_buffer as buffer
    from gpustack.mixins import active_record
    from gpustack.server import services
    from unittest.mock import patch

    engine = create_async_engine('sqlite+aiosqlite://')
    async with engine.begin() as conn:
        for table in (
            Cluster.__table__,
            Worker.__table__,
            Model.__table__,
            ModelInstance.__table__,
        ):
            await conn.run_sync(table.create)

    @asynccontextmanager
    async def session_context():
        async with AsyncSession(engine, expire_on_commit=False) as session:
            yield session

    monkeypatch.setattr(workers, 'async_session', session_context)
    monkeypatch.setattr(buffer, 'async_session', session_context)
    monkeypatch.setattr(workers, '_build_worker_config_dict', lambda cluster: {})
    monkeypatch.setattr(
        workers,
        '_resolve_existing_worker_principal',
        AsyncMock(return_value=SimpleNamespace(api_keys=[object()])),
    )
    monkeypatch.setattr(services, 'delete_cache_by_key', AsyncMock())
    monkeypatch.setattr(active_record, 'delete_cache_by_key', AsyncMock())
    publish = AsyncMock()
    monkeypatch.setattr(active_record.event_bus, 'publish', publish)
    try:
        async with AsyncSession(engine, expire_on_commit=False) as session:
            cluster = Cluster(
                id=100,
                name='cluster',
                provider=ClusterProvider.Docker,
                owner_principal_id=1,
            )
            worker = Worker.model_validate(
                {
                    **_payload(100, True),
                    'id': 20,
                    'token': 'existing-token',
                    'system_principal_id': 30,
                    'worker_uuid': 'registered-uuid',
                    'state': WorkerStateEnum.READY,
                    'heartbeat_time': datetime.now(timezone.utc),
                    'owner_principal_id': 1,
                }
            )
            worker.worker_version = '2.3.1'
            model = Model(
                id=1,
                name='model',
                source=SourceEnum.HUGGING_FACE,
                huggingface_repo_id='org/model',
                cluster_id=100,
                backend=BackendEnum.VLLM,
                owner_principal_id=1,
            )
            instance = ModelInstance(
                id=1,
                name='instance',
                model_id=1,
                model_name='model',
                source=SourceEnum.HUGGING_FACE,
                huggingface_repo_id='org/model',
                worker_id=20,
                cluster_id=100,
                owner_principal_id=1,
                distributed_servers=DistributedServers(workloads=[]),
                port=40000,
                ports=[40000],
                restart_count=0,
            )
            session.add_all([cluster, worker, model, instance])
            await session.commit()
        old_payload = WorkerCreate.model_validate(_payload(100, True))
        old_payload.ip, old_payload.port = '192.0.2.99', 12345
        old_payload.worker_version = version
        from httpx import AsyncClient, ASGITransport

        app = FastAPI()
        register_handlers(app)
        app.include_router(workers.router, prefix='/v2/workers')
        app.dependency_overrides[get_current_user] = lambda: _principal('worker')
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url='http://test'
        ) as client:
            accepted = await client.post(
                '/v2/workers', json=old_payload.model_dump(mode='json')
            )
        assert accepted.status_code == 200, accepted.text
        assert accepted.json()['state'] == 'ready'
        async with AsyncSession(engine) as session:
            stored = await Worker.one_by_id(session, 20, refresh=True)
            assert stored.worker_version == version
            assert stored.state == WorkerStateEnum.READY
            assert stored.token == 'existing-token' and stored.system_principal_id == 30
            assert stored.ip == '192.0.2.99' and stored.port == 12345
            assert stored.worker_uuid == 'registered-uuid'
            untouched = await ModelInstance.one_by_id(session, 1, refresh=True)
            assert (
                untouched.distributed_servers.workloads == []
                and untouched.ports == [40000]
            )
        buffer.worker_status_flush_buffer[20] = {
            'status': WorkerStatus(),
            'state_message': None,
        }
        await buffer.flush_worker_status()
        async with AsyncSession(engine) as session:
            stored = await Worker.one_by_id(session, 20, refresh=True)
            assert stored.worker_version == version
            assert stored.state == WorkerStateEnum.READY
        compatible = WorkerCreate.model_validate(_payload(100, True))
        compatible.worker_version = '2.3.1'
        from gpustack.api.exceptions import ForbiddenException

        with pytest.raises(ForbiddenException):
            await workers.create_worker(
                _principal('worker', worker_row_id=99), compatible
            )
        import asyncio
        from gpustack.api.exceptions import InternalServerErrorException

        await asyncio.sleep(0)
        publish.reset_mock()
        with patch.object(
            AsyncSession, 'commit', AsyncMock(side_effect=RuntimeError('commit failed'))
        ):
            with pytest.raises(InternalServerErrorException):
                await workers.create_worker(_principal('worker'), compatible)
        await asyncio.sleep(0)
        publish.assert_not_awaited()
        async with AsyncSession(engine) as session:
            unchanged = await Worker.one_by_id(session, 20, refresh=True)
            assert unchanged.worker_version == version
            assert unchanged.token == 'existing-token'
        response = await workers.create_worker(_principal('worker'), compatible)
        assert response.worker_version == '2.3.1'
        assert response.state == WorkerStateEnum.READY
        buffer.worker_status_flush_buffer[20] = {'status': None, 'state_message': None}
        await buffer.flush_worker_status()
        buffer.heartbeat_flush_buffer.add(20)
        await buffer.flush_heartbeats()
        async with AsyncSession(engine) as session:
            admitted = await Worker.one_by_id(session, 20, refresh=True)
            assert admitted.worker_version == '2.3.1'
            assert admitted.state == WorkerStateEnum.READY
        response = await workers.create_worker(_principal('worker'), old_payload)
        assert response.worker_version == version
        assert response.state == WorkerStateEnum.READY
        assert response.token == 'existing-token'
        assert not any(
            call.args[1].type.value == 'DELETED' for call in publish.await_args_list
        )
    finally:
        buffer.worker_status_flush_buffer.clear()
        await engine.dispose()


@pytest.mark.parametrize(
    'cohort', ['live_router', 'stopped_router', 'unassigned_pending']
)
def test_registration_repair_binding_obeys_only_assigned_managed_cohort(
    registration, monkeypatch, cohort
):
    from gpustack.schemas.models import DistributedServers, ModelInstanceLegacyStop
    from tests.utils.model import new_model_instance
    from uuid import uuid4

    registration.app.dependency_overrides[get_current_user] = lambda: _principal(
        'worker'
    )
    registration.get_cluster.return_value = Cluster(
        id=100, name='cluster', provider=ClusterProvider.Docker
    )
    worker = Worker.model_validate(
        {**_payload(100, True), 'id': 20, 'token': 'old', 'system_principal_id': None}
    )
    registration.list_workers.return_value = [worker]
    registration.get_worker.return_value = worker
    registration.get_principal.return_value = None
    engine = new_model_instance(1, 'engine', model_id=1, worker_id=7)
    engine.distributed_servers = DistributedServers(workloads=[])
    router = new_model_instance(2, 'router', model_id=1, worker_id=20)
    if cohort == 'unassigned_pending':
        engine.worker_id = None
    elif cohort == 'stopped_router':
        router.distributed_servers = DistributedServers(
            legacy_stop=ModelInstanceLegacyStop(
                token=uuid4(),
                restart_count=0,
                worker_ids=[20],
                worker_principal_ids={20: 30},
                confirmed_worker_ids=[20],
            )
        )
    monkeypatch.setattr(
        workers.ModelInstance,
        'all_by_field',
        AsyncMock(return_value=[engine, router]),
    )
    registration.session.exec.return_value = MagicMock()
    registration.session.exec.return_value.all.return_value = [engine, router]
    payload = WorkerCreate.model_validate(_payload(100, True))
    payload.worker_version = '2.3.1'
    with TestClient(registration.app) as client:
        response = client.post('/v2/workers', json=payload.model_dump(mode='json'))
    assert response.status_code == (
        409 if cohort == 'live_router' else 200
    ), response.text
    if cohort == 'live_router':
        registration.persist.assert_not_awaited()
        assert worker.token == 'old' and worker.system_principal_id is None
    else:
        registration.persist.assert_awaited_once()
        assert registration.persist.await_args.kwargs['to_create_principal'] is not None
