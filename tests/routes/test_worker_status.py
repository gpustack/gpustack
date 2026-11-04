"""Invariants of the worker status reporting endpoint.

The status report is operational state, not identity: a worker may
describe its machine but may never rewrite the identity the server
assigned it.
"""

from types import SimpleNamespace

import pytest

from gpustack.routes import workers as workers_route
from gpustack.server.worker_status_buffer import (
    worker_status_flush_buffer,
    worker_status_flush_buffer_lock,
)
from gpustack.schemas.workers import WorkerStatus, WorkerStatusStored


def _user(worker_id: int = 1) -> SimpleNamespace:
    return SimpleNamespace(worker=SimpleNamespace(id=worker_id))


def _status(**overrides) -> WorkerStatusStored:
    fields = {
        "hostname": "worker-1",
        "ip": "10.0.0.9",
        "ifname": "eth0",
        "port": 10150,
        "status": WorkerStatus.get_default_status(),
        "worker_uuid": "attacker-uuid",
    }
    fields.update(overrides)
    return WorkerStatusStored(**fields)


@pytest.mark.asyncio
async def test_status_report_cannot_rewrite_worker_uuid():
    # A worker echoing (or forging) a worker_uuid in its status report
    # must not have the value written to its worker row: worker_uuid is
    # the route-ownership key in the proxy plane, so a rewrite would let
    # one worker impersonate another's registrations.
    status = _status(worker_uuid="victim-uuid")

    await workers_route.create_worker_status(_user(worker_id=7), status)

    async with worker_status_flush_buffer_lock:
        buffered = worker_status_flush_buffer.pop(7)
    assert "worker_uuid" not in buffered


@pytest.mark.asyncio
async def test_status_report_cannot_rewrite_registered_worker_version():
    status = WorkerStatusStored.model_validate(
        {**_status().model_dump(), 'worker_version': '2.3.1'}
    )
    await workers_route.create_worker_status(_user(worker_id=7), status)
    async with worker_status_flush_buffer_lock:
        buffered = worker_status_flush_buffer.pop(7)
    assert 'worker_version' not in buffered


@pytest.mark.asyncio
async def test_status_report_buffers_operational_fields():
    status = _status(ip="10.0.0.9", hostname="worker-1")

    await workers_route.create_worker_status(_user(worker_id=7), status)

    async with worker_status_flush_buffer_lock:
        buffered = worker_status_flush_buffer.pop(7)
    assert buffered["ip"] == "10.0.0.9"
    assert buffered["hostname"] == "worker-1"
    assert "heartbeat_time" in buffered


@pytest.mark.asyncio
@pytest.mark.parametrize('empty', [True, False])
async def test_flush_keeps_registered_version_instead_of_late_packet(
    empty, monkeypatch
):
    from contextlib import asynccontextmanager
    from datetime import datetime, timezone
    from unittest.mock import AsyncMock
    from gpustack.server import worker_status_buffer as buffer
    from gpustack.schemas.workers import (
        Worker,
        WorkerStateEnum,
    )
    from gpustack.server.services import WorkerService

    fresh = Worker(
        name='worker',
        hostname='worker',
        ip='10.0.0.1',
        id=7,
        status=WorkerStatus(),
        worker_version='2.3.0',
        state=WorkerStateEnum.NOT_READY,
        heartbeat_time=datetime.now(timezone.utc),
    )
    stale = fresh.model_copy(deep=True)
    stale.worker_version = '2.3.1'
    monkeypatch.setattr(
        'gpustack.extension.resolve_version_info', lambda: ('2.3.1', 'test')
    )

    @asynccontextmanager
    async def session():
        yield SimpleNamespace()

    packet = (
        None
        if empty
        else WorkerStatus.model_validate({'orchestration': {'protocol_version': 1}})
    )
    buffer.worker_status_flush_buffer[7] = {'status': packet, 'state_message': None}
    locked = AsyncMock(return_value=[fresh])
    monkeypatch.setattr(buffer, 'async_session', session)
    monkeypatch.setattr(Worker, 'all_by_fields', AsyncMock(return_value=[stale]))
    monkeypatch.setattr(Worker, 'all_by_ids', locked)
    persist = AsyncMock()
    monkeypatch.setattr(WorkerService, 'batch_update', persist)
    await buffer.flush_worker_status()
    locked.assert_awaited_once()
    assert locked.await_args.kwargs['for_update'] is True
    assert persist.await_args.args[0] == [fresh]
    assert fresh.worker_version == '2.3.0'
    assert fresh.state == WorkerStateEnum.READY
