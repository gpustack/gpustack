from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from sqlalchemy.dialects import mysql, postgresql

from gpustack.api.exceptions import register_handlers
from gpustack.api.tenant import TenantContext, get_tenant_context
from gpustack.routes import workers
from gpustack.schemas.principals import OrgRole, PrincipalType
from gpustack.schemas.workers import Worker, WorkerUpdate
from gpustack.server.db import get_session
from gpustack.server.services import WorkerService


def _worker(id, labels=None, owner=10, cluster_id=1):
    return Worker(
        id=id,
        name=f"worker-{id}",
        hostname=f"host-{id}",
        ip=f"192.0.2.{id}",
        port=10150,
        cluster_id=cluster_id,
        owner_principal_id=owner,
        labels=labels or {},
    )


@pytest.fixture
def batch(monkeypatch):
    rows = [
        _worker(1, {"env": "old", "topology.gpustack.ai/rack": "r1"}),
        _worker(2, {"team": "inference"}, cluster_id=2),
    ]
    ctx = TenantContext(
        user=SimpleNamespace(kind=PrincipalType.USER),
        is_platform_admin=False,
        current_principal_id=10,
        org_role=OrgRole.OWNER,
    )
    session = AsyncMock()
    result = MagicMock()
    result.all.side_effect = lambda: rows
    session.exec.return_value = result
    save = AsyncMock()
    monkeypatch.setattr(WorkerService, "batch_update", save)

    app = FastAPI()
    register_handlers(app)
    app.include_router(workers.router, prefix="/v2/workers")
    app.dependency_overrides[get_tenant_context] = lambda: ctx
    app.dependency_overrides[get_session] = lambda: session
    with TestClient(app) as client:
        yield SimpleNamespace(
            client=client, rows=rows, ctx=ctx, session=session, save=save
        )


def _post(batch, ids=None, labels=None):
    return batch.client.post(
        "/v2/workers/labels",
        json={
            "worker_ids": [1, 2] if ids is None else ids,
            "labels": {"env": "prod"} if labels is None else labels,
        },
    )


def test_batch_merges_each_workers_labels_in_one_save(batch):
    response = _post(batch)
    assert response.status_code == 204, response.text
    assert response.content == b""
    assert batch.rows[0].labels == {
        "env": "prod",
        "topology.gpustack.ai/rack": "r1",
    }
    assert batch.rows[1].labels == {"team": "inference", "env": "prod"}
    batch.save.assert_awaited_once_with(batch.rows)


@pytest.mark.parametrize("dialect", [postgresql.dialect(), mysql.dialect()])
def test_batch_locks_fresh_live_rows_in_id_order_with_tenant_scope(batch, dialect):
    assert _post(batch).status_code == 204
    statement = batch.session.exec.await_args.args[0]
    sql = str(
        statement.compile(dialect=dialect, compile_kwargs={"literal_binds": True})
    )
    assert "workers.id IN (1, 2)" in sql
    assert "workers.owner_principal_id = 10" in sql
    assert "workers.deleted_at IS NULL" in sql
    assert "ORDER BY workers.id" in sql
    assert "FOR UPDATE" in sql
    assert statement.get_execution_options()["populate_existing"] is True


def test_duplicate_targets_are_updated_once(batch):
    assert _post(batch, ids=[2, 1, 1]).status_code == 204
    batch.save.assert_awaited_once_with(batch.rows)


@pytest.mark.parametrize("labels", [{}, {"env": "old"}])
def test_unchanged_labels_do_not_write(batch, labels):
    batch.rows[:] = batch.rows[:1]
    original = dict(batch.rows[0].labels)
    assert _post(batch, ids=[1], labels=labels).status_code == 204
    assert batch.rows[0].labels == original
    batch.save.assert_not_awaited()


@pytest.mark.parametrize("missing_id", [1, 2])
def test_missing_or_filtered_target_refuses_entire_batch(batch, missing_id):
    batch.rows[:] = [row for row in batch.rows if row.id != missing_id]
    original = [dict(row.labels) for row in batch.rows]
    assert _post(batch).status_code == 404
    assert [row.labels for row in batch.rows] == original
    batch.save.assert_not_awaited()


def test_foreign_target_refuses_entire_batch_before_mutation(batch):
    batch.rows[1].owner_principal_id = 20
    original = [dict(row.labels) for row in batch.rows]
    assert _post(batch).status_code == 404
    assert [row.labels for row in batch.rows] == original
    batch.save.assert_not_awaited()


@pytest.mark.parametrize("personal", [False, True])
def test_org_member_and_personal_scope_cannot_write_labels(batch, personal):
    batch.ctx.org_role = None if personal else OrgRole.MEMBER
    batch.ctx.current_is_personal_scope = personal
    assert _post(batch).status_code == 403
    batch.save.assert_not_awaited()


def test_platform_admin_can_update_across_owners(batch):
    batch.ctx.is_platform_admin = True
    batch.ctx.current_principal_id = None
    batch.rows[1].owner_principal_id = 20
    assert _post(batch).status_code == 204
    batch.save.assert_awaited_once()


def test_admin_acting_as_org_is_still_scoped(batch):
    batch.ctx.is_platform_admin = True
    batch.rows[1].owner_principal_id = 20
    assert _post(batch).status_code == 404
    batch.save.assert_not_awaited()


def test_cluster_system_principal_cannot_write_another_cluster(batch):
    batch.ctx.user.kind = PrincipalType.SYSTEM
    batch.ctx.scoped_cluster_id = 1
    assert _post(batch).status_code == 404
    batch.save.assert_not_awaited()
    statement = batch.session.exec.await_args.args[0]
    sql = str(statement.compile(compile_kwargs={"literal_binds": True}))
    assert "workers.cluster_id = 1" in sql


@pytest.mark.parametrize(
    "payload",
    [
        {"worker_ids": [], "labels": {"env": "prod"}},
        {"worker_ids": [0], "labels": {"env": "prod"}},
        {"worker_ids": [1], "labels": {"env": None}},
        {"worker_ids": [1]},
    ],
)
def test_invalid_request_is_rejected_before_database_access(batch, payload):
    response = batch.client.post("/v2/workers/labels", json=payload)
    assert response.status_code == 422
    batch.session.exec.assert_not_awaited()
    batch.save.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("labels", [{}, {"env": "prod"}])
async def test_single_worker_edit_still_replaces_labels(batch, monkeypatch, labels):
    worker = batch.rows[0]
    monkeypatch.setattr(Worker, "one_by_id", AsyncMock(return_value=worker))

    async def update(_self, row, patch):
        row.labels = patch["labels"]

    monkeypatch.setattr(WorkerService, "update", update)
    await workers.update_worker(
        ctx=batch.ctx,
        session=batch.session,
        id=worker.id,
        worker_in=WorkerUpdate(name=worker.name, labels=labels),
    )
    assert worker.labels == labels
