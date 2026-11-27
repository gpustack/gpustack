from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gpustack import envs
from gpustack.schemas.config import ModelInstanceProxyModeEnum
from gpustack.schemas.models import ModelInstanceStateEnum
from gpustack.schemas.workers import (
    Maintenance,
    Worker,
    WorkerCreate,
    WorkerStateEnum,
    WorkerUpdate,
)
from gpustack.server.services import WorkerService
from gpustack.server.worker_syncer import WorkerSyncer
from gpustack.utils import network
from gpustack.utils.model_instance_workers import (
    inference_proxy_unavailable,
    model_instance_endpoints,
)
from tests.fixtures.workers.fixtures import linux_nvidia_1_4090_24gx1
from tests.utils.mock import mock_async_session
from tests.utils.model import new_model_instance


@pytest.fixture
def proxy_liveness(monkeypatch):
    clock = SimpleNamespace(now=datetime.now(timezone.utc))

    class ClockDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock.now

    monkeypatch.setattr(network, 'datetime', ClockDateTime)
    worker = linux_nvidia_1_4090_24gx1()
    worker.id = 1
    worker.ifname = 'eth0'
    worker.port = 10150
    worker.worker_uuid = 'worker-1'
    worker.created_at = worker.updated_at = clock.now
    worker.state = WorkerStateEnum.MAINTENANCE
    worker.maintenance = Maintenance(enabled=True)
    worker.proxy_mode = ModelInstanceProxyModeEnum.WORKER
    worker.unreachable = False
    worker.heartbeat_time = clock.now
    rows = [worker]
    writes = []
    cm = mock_async_session()
    monkeypatch.setattr('gpustack.server.worker_syncer.async_session', lambda: cm)
    monkeypatch.setattr(
        Worker,
        'all',
        AsyncMock(
            side_effect=lambda session: [row.model_copy(deep=True) for row in rows]
        ),
    )
    monkeypatch.setattr(
        Worker,
        'all_by_ids',
        AsyncMock(
            side_effect=lambda session, ids, **kwargs: [
                row for row in rows if row.id in ids
            ]
        ),
    )

    async def save(changed):
        if changed:
            writes.append([row.model_dump() for row in changed])

    monkeypatch.setattr(WorkerService, 'batch_update', AsyncMock(side_effect=save))
    syncer = WorkerSyncer(lambda: MagicMock(), lambda: MagicMock())

    async def probe(row):
        row.unreachable = False

    probe = AsyncMock(side_effect=probe)
    monkeypatch.setattr(syncer, '_set_worker_unreachable', probe)
    monkeypatch.setattr(envs, 'WORKER_UNREACHABLE_CHECK_MODE', 'disabled')
    return SimpleNamespace(
        worker=worker, rows=rows, writes=writes, clock=clock, syncer=syncer, probe=probe
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['disabled', 'auto'])
async def test_maintenance_worker_heartbeat_expiry_notifies_without_another_worker_event(
    proxy_liveness, monkeypatch, mode
):
    ctx = proxy_liveness
    monkeypatch.setattr(envs, 'WORKER_UNREACHABLE_CHECK_MODE', mode)
    if mode == 'auto':
        for identity in range(2, 52):
            extra = ctx.worker.model_copy(deep=True)
            extra.id = identity
            extra.proxy_mode = ModelInstanceProxyModeEnum.DIRECT
            ctx.rows.append(extra)
    await ctx.syncer._sync_workers_states()
    assert ctx.writes == []

    ctx.clock.now += timedelta(seconds=envs.WORKER_HEARTBEAT_GRACE_PERIOD + 1)
    await ctx.syncer._sync_workers_states()

    assert len(ctx.writes) == 1
    assert ctx.worker.state == WorkerStateEnum.MAINTENANCE
    assert not ctx.worker.unreachable
    assert inference_proxy_unavailable(ctx.worker)
    assert inference_proxy_unavailable(Worker.model_validate(ctx.worker.model_dump()))
    await ctx.syncer._sync_workers_states()
    assert len(ctx.writes) == 1
    ctx.probe.assert_not_awaited()

    ctx.worker.heartbeat_time = ctx.clock.now
    await ctx.syncer._sync_workers_states()
    assert len(ctx.writes) == 2
    assert not inference_proxy_unavailable(ctx.worker)
    assert not ctx.worker.unreachable


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'change', ['direct', 'tunnel', 'leave_maintenance', 'enable_probe']
)
async def test_maintenance_heartbeat_evidence_is_scoped_and_cleared(
    proxy_liveness, monkeypatch, change
):
    ctx = proxy_liveness
    ctx.clock.now += timedelta(seconds=envs.WORKER_HEARTBEAT_GRACE_PERIOD + 1)
    await ctx.syncer._sync_workers_states()
    assert inference_proxy_unavailable(ctx.worker)
    if change == 'direct':
        ctx.worker.proxy_mode = ModelInstanceProxyModeEnum.DIRECT
    elif change == 'tunnel':
        ctx.worker.proxy_mode = ModelInstanceProxyModeEnum.TUNNEL
        ctx.worker.proxy_address = 'http://127.0.0.1:30079'
    elif change == 'leave_maintenance':
        ctx.worker.maintenance.enabled = False
        ctx.worker.compute_state()
    else:
        monkeypatch.setattr(envs, 'WORKER_UNREACHABLE_CHECK_MODE', 'enabled')

    await ctx.syncer._sync_workers_states()

    assert ctx.worker.maintenance.worker_proxy_heartbeat_lost is None
    assert not ctx.worker.unreachable
    assert not inference_proxy_unavailable(ctx.worker)


@pytest.mark.asyncio
async def test_new_heartbeat_wins_over_an_unlocked_expired_snapshot(
    proxy_liveness, monkeypatch
):
    ctx = proxy_liveness
    ctx.clock.now += timedelta(seconds=envs.WORKER_HEARTBEAT_GRACE_PERIOD + 1)

    async def current_rows(session, ids, **kwargs):
        ctx.worker.heartbeat_time = ctx.clock.now
        return [ctx.worker] if ctx.worker.id in ids else []

    monkeypatch.setattr(Worker, 'all_by_ids', current_rows)
    await ctx.syncer._sync_workers_states()

    assert ctx.writes == []
    assert not inference_proxy_unavailable(ctx.worker)


@pytest.mark.asyncio
async def test_restart_retains_and_then_retires_only_derived_liveness(proxy_liveness):
    ctx = proxy_liveness
    ctx.clock.now += timedelta(seconds=envs.WORKER_HEARTBEAT_GRACE_PERIOD + 1)
    await ctx.syncer._sync_workers_states()
    assert inference_proxy_unavailable(ctx.worker)
    restarted = WorkerSyncer(lambda: MagicMock(), lambda: MagicMock())
    await restarted._sync_workers_states()
    assert len(ctx.writes) == 1
    ctx.worker.heartbeat_time = ctx.clock.now
    await restarted._sync_workers_states()
    assert not inference_proxy_unavailable(ctx.worker)
    assert not ctx.worker.unreachable


@pytest.mark.asyncio
async def test_paused_probes_do_not_overwrite_a_real_probe_failure(proxy_liveness):
    ctx = proxy_liveness
    ctx.worker.unreachable = True
    await ctx.syncer._sync_workers_states()
    assert ctx.worker.unreachable
    assert inference_proxy_unavailable(ctx.worker)


def test_legacy_endpoint_without_distributed_servers():
    instance = new_model_instance(1, 'legacy', 1, worker_id=1)
    instance.state = ModelInstanceStateEnum.RUNNING
    instance.port = 40000
    instance.distributed_servers = None
    assert [endpoint.worker_id for endpoint in model_instance_endpoints(instance)] == [
        1
    ]


@pytest.mark.parametrize(
    'existing_flag, incoming_flag', [(None, True), (True, None), (True, False)]
)
def test_worker_registration_preserves_only_server_liveness(
    proxy_liveness, existing_flag, incoming_flag
):
    from gpustack.routes.workers import update_worker_data

    worker = proxy_liveness.worker
    worker.maintenance.worker_proxy_heartbeat_lost = existing_flag
    if existing_flag:
        worker.heartbeat_time -= timedelta(days=1)
    incoming = WorkerCreate.model_validate(worker.model_dump())
    incoming.maintenance.worker_proxy_heartbeat_lost = incoming_flag

    registered = update_worker_data(incoming, worker)

    assert registered.maintenance.worker_proxy_heartbeat_lost is existing_flag
    created = update_worker_data(incoming)
    assert created.maintenance.worker_proxy_heartbeat_lost is None


@pytest.mark.parametrize(
    'proxy_mode',
    [
        ModelInstanceProxyModeEnum.WORKER,
        ModelInstanceProxyModeEnum.DIRECT,
        ModelInstanceProxyModeEnum.TUNNEL,
    ],
)
def test_reregistration_retires_derived_loss_after_fresh_heartbeat_or_mode_change(
    proxy_liveness, proxy_mode
):
    from gpustack.routes.workers import update_worker_data

    worker = proxy_liveness.worker
    worker.maintenance.worker_proxy_heartbeat_lost = True
    incoming = WorkerCreate.model_validate(worker.model_dump())
    incoming.proxy_mode = proxy_mode

    registered = update_worker_data(incoming, worker)

    assert registered.maintenance.worker_proxy_heartbeat_lost is None
    assert not registered.unreachable


@pytest.mark.asyncio
@pytest.mark.parametrize('enabled', [False, True])
async def test_worker_update_cannot_inject_or_erase_server_liveness(
    proxy_liveness, monkeypatch, enabled
):
    from gpustack.routes import workers as routes

    worker = proxy_liveness.worker
    worker.maintenance.worker_proxy_heartbeat_lost = True
    worker.heartbeat_time -= timedelta(days=1)
    body = WorkerUpdate(
        name=worker.name,
        maintenance=Maintenance(
            enabled=enabled, worker_proxy_heartbeat_lost=not enabled
        ),
    )
    lookup = AsyncMock(return_value=worker)
    monkeypatch.setattr(Worker, 'one_by_id', lookup)
    monkeypatch.setattr(routes, 'assert_resource_visible', lambda *a, **kw: None)
    monkeypatch.setattr(routes, 'assert_org_owned_writable', lambda *a, **kw: None)
    saved = AsyncMock()
    monkeypatch.setattr(WorkerService, 'update', saved)

    session = MagicMock()
    await routes.update_worker(MagicMock(), session, worker.id, body)

    lookup.assert_awaited_once_with(session, worker.id, for_update=True, refresh=True)
    expected = True if enabled else None
    assert worker.maintenance.worker_proxy_heartbeat_lost is expected
    stored = Maintenance.model_validate(saved.await_args.args[1]['maintenance'])
    assert stored.worker_proxy_heartbeat_lost is expected
