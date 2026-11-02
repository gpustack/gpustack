"""Deleting a worker must free the tunnel claims its client id still holds.

A worker's tunnel registers the worker's address in the in-memory CIDR
registry, and that claim is only released when the websocket reports a
disconnect. A half-open tunnel may never report one, so the worker-delete
path releases the claims explicitly; otherwise a re-registered worker on the
same address is rejected as a foreign claim until the server restarts.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from gpustack.routes import workers as workers_route
from gpustack.schemas.workers import Worker


def _session():
    session = MagicMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    return session


def _request(handler):
    request = MagicMock()
    request.app.state.message_server_handler = handler
    return request


def _patch_route(monkeypatch, worker, delete):
    monkeypatch.setattr(Worker, "one_by_id", AsyncMock(return_value=worker))
    monkeypatch.setattr(workers_route, "assert_resource_visible", lambda *a, **k: None)
    monkeypatch.setattr(
        workers_route, "assert_org_owned_writable", lambda *a, **k: None
    )
    monkeypatch.setattr(
        workers_route, "WorkerService", lambda session: MagicMock(delete=delete)
    )


def _worker(worker_uuid):
    worker = MagicMock(spec=Worker)
    worker.deleted_at = None
    worker.external_id = None
    worker.worker_uuid = worker_uuid
    worker.state = None
    worker.system_principal = None
    return worker


@pytest.mark.asyncio
async def test_delete_worker_releases_its_tunnel_claims(monkeypatch):
    worker_uuid = str(uuid.uuid4())
    worker = _worker(worker_uuid)
    delete = AsyncMock()
    _patch_route(monkeypatch, worker, delete)

    handler = MagicMock()
    handler.release_client = AsyncMock(return_value=True)

    await workers_route.delete_worker(
        request=_request(handler), ctx=MagicMock(), session=_session(), id=1
    )

    delete.assert_awaited_once_with(worker, soft=False)
    handler.release_client.assert_awaited_once_with(uuid.UUID(worker_uuid))


@pytest.mark.asyncio
async def test_delete_worker_ignores_a_release_failure(monkeypatch):
    worker = _worker(str(uuid.uuid4()))
    delete = AsyncMock()
    _patch_route(monkeypatch, worker, delete)

    handler = MagicMock()
    handler.release_client = AsyncMock(side_effect=RuntimeError("registry gone"))

    # The row is already deleted, so a release failure must not fail the
    # request: a leftover claim is a degraded state, not a failed delete.
    await workers_route.delete_worker(
        request=_request(handler), ctx=MagicMock(), session=_session(), id=1
    )

    delete.assert_awaited_once()
