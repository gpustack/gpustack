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
async def test_status_report_buffers_operational_fields():
    status = _status(ip="10.0.0.9", hostname="worker-1")

    await workers_route.create_worker_status(_user(worker_id=7), status)

    async with worker_status_flush_buffer_lock:
        buffered = worker_status_flush_buffer.pop(7)
    assert buffered["ip"] == "10.0.0.9"
    assert buffered["hostname"] == "worker-1"
    assert "heartbeat_time" in buffered
