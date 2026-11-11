from types import SimpleNamespace
from unittest.mock import AsyncMock
import asyncio
import json
import re
from datetime import datetime, timezone

import pytest

from gpustack.api.exceptions import InvalidException
from gpustack.api.tenant import TenantContext
from gpustack.schemas.models import Model, ModelInstance, ModelInstanceStateEnum
from gpustack.schemas.models import DistributedServers, ModelInstanceSubordinateWorker
from gpustack.schemas.models import DPRankPlacement, ModelInstanceWorkload
from gpustack.schemas.principals import OrgRole, PrincipalType
from gpustack.schemas.workers import (
    GPUCoreInfo,
    GPUDeviceStatus,
    MemoryInfo,
    Worker,
    WorkerStatus,
)


def _device(index=8, uuid='GPU-private', chip=2):
    return GPUDeviceStatus(
        index=index,
        uuid=uuid,
        device_chip_index=chip,
        name='GPU',
        core=GPUCoreInfo(utilization_rate=35),
        memory=MemoryInfo(total=1000, used=400),
        power_used=90,
        temperature=50,
    )


def _worker(owner=17, devices=None):
    return Worker(
        id=7,
        name='visible-worker',
        cluster_id=1,
        worker_uuid='server-worker',
        owner_principal_id=owner,
        status=WorkerStatus(
            gpu_devices=devices if devices is not None else [_device()]
        ),
    )


def _instance(id=11, gpu_indexes=None):
    return ModelInstance(
        id=id,
        name=f'mi-{id}',
        model_id=5,
        owner_principal_id=17,
        worker_id=7,
        ports=[8000],
        gpu_indexes=gpu_indexes if gpu_indexes is not None else [8],
    )


def _context(admin=False):
    return TenantContext(
        user=SimpleNamespace(kind=PrincipalType.USER, id=1),
        is_platform_admin=admin,
        current_principal_id=None if admin else 17,
        org_role=OrgRole.OWNER,
        accessible_cluster_ids={1},
    )


def _environment(monkeypatch, workers, *, query=None):
    from gpustack.server import model_hardware_metrics as hardware

    session = SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(all=lambda: workers))
    )
    calls = []

    async def fake_range(client, url, expr, start, end, step):
        calls.append(expr)
        if query:
            return query(expr, start, end)
        if 'identity_info' in expr:
            return [
                {
                    'metric': {
                        'worker_id': str(worker.id),
                        'worker_uuid': worker.worker_uuid,
                        'owner_principal_id': str(worker.owner_principal_id or ''),
                        'gpu_uuid': device.uuid,
                        'gpu_chip_index': str(device.device_chip_index),
                        'gpu_index': str(device.index),
                    },
                    'values': [[end, '1']],
                }
                for worker in workers
                for device in worker.status.gpu_devices
            ]
        labels = {
            key: json.loads(value)
            for key, value in re.findall(r'(\w+)=("(?:[^"\\]|\\.)*")', expr)
        }
        return [{'metric': labels, 'values': [[start, '0'], [end, '35']]}]

    async def fake_instant(client, url, expr, at):
        rows = await fake_range(client, url, expr, at, at, 1)
        return [dict(metric=r['metric'], value=r['values'][-1]) for r in rows]

    monkeypatch.setattr(
        hardware.prometheus_query, 'prometheus_url', lambda: 'http://prom'
    )
    monkeypatch.setattr(hardware.prometheus_query, 'query_range', fake_range)
    monkeypatch.setattr(hardware.prometheus_query, 'query_instant', fake_instant)
    return hardware, session, calls


async def _collect(hardware, session, instances=None, ctx=None, **kwargs):
    return await hardware.collect_model_resource_metrics(
        Model(id=5, name='model', owner_principal_id=17),
        instances or [_instance()],
        ctx or _context(),
        session,
        window_seconds=600,
        client=object(),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_sparse_placement_uses_stable_owner_qualified_identity(monkeypatch):
    hardware, session, calls = _environment(monkeypatch, [_worker()])
    result = await _collect(hardware, session)
    assert result.available
    assert result.allocated_gpu_count == result.allocated_worker_count == 1
    assert result.devices[0].index == 8
    assert result.devices[0].chip_index == 2
    assert result.devices[0].current['gpu_memory_used'] == 400
    assert result.devices[0].history_available
    assert 'GPU-private' not in result.model_dump_json()
    assert all('owner_principal_id="17"' in expr for expr in calls)
    assert all('worker_uuid="server-worker"' in expr for expr in calls)
    assert not any('worker_node_' in expr or 'gpu_index=' in expr for expr in calls)
    gpu = result.metrics['gpu_utilization']
    assert gpu.unit == 'percent'
    assert gpu.series[0].dimensions.device_id == result.selected_device_id
    assert gpu.series[0].points[0][1] == 0


@pytest.mark.asyncio
async def test_model_access_and_cluster_grant_do_not_grant_worker_metrics(monkeypatch):
    hardware, session, calls = _environment(monkeypatch, [_worker(owner=99)])
    result = await _collect(hardware, session)
    assert result.allocated_gpu_count == 1
    assert result.devices == []
    assert result.reason == 'resource_not_visible'
    assert result.metrics['gpu_memory_used'].status == 'restricted'
    assert calls == []


@pytest.mark.asyncio
async def test_invalid_or_foreign_device_is_rejected_before_prometheus(monkeypatch):
    hardware, session, calls = _environment(monkeypatch, [_worker()])
    with pytest.raises(InvalidException):
        await _collect(hardware, session, device_id='another-device')
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'devices', [[_device(uuid='')], [_device(), _device(uuid='GPU-other')]]
)
async def test_unknown_identity_keeps_current_allocation_without_history(
    monkeypatch, devices
):
    hardware, session, calls = _environment(monkeypatch, [_worker(devices=devices)])
    result = await _collect(hardware, session)
    assert not result.devices[0].history_available
    assert result.devices[0].reason == 'device_identity_unresolved'
    assert not any('worker_gpu_' in expr for expr in calls)


@pytest.mark.asyncio
async def test_duplicate_sources_do_not_duplicate_a_physical_device(monkeypatch):
    hardware, session, _ = _environment(monkeypatch, [_worker()])
    result = await _collect(hardware, session, [_instance(11), _instance(12)])
    assert len(result.devices) == result.allocated_gpu_count == 1
    assert result.devices[0].model_instance_ids == [11, 12]


@pytest.mark.asyncio
async def test_partition_parent_sensors_are_not_queried_or_returned(monkeypatch):
    hardware, session, calls = _environment(
        monkeypatch, [_worker(devices=[_device(uuid='MIG-partition')])]
    )
    result = await _collect(hardware, session)
    assert result.devices[0].scope == 'partition'
    for key in ('gpu_temperature', 'gpu_power', 'gpu_utilization'):
        assert result.metrics[key].status == 'unsupported'
        assert result.devices[0].current[key] is None
    assert not any(
        'worker_gpu_power' in expr or 'worker_gpu_temperature' in expr for expr in calls
    )


@pytest.mark.asyncio
async def test_empty_stable_history_never_falls_back_to_ordinal_series(monkeypatch):
    hardware, session, calls = _environment(
        monkeypatch, [_worker()], query=lambda *args: []
    )
    result = await _collect(hardware, session)
    assert result.metrics['gpu_memory_used'].status == 'no_data'
    assert result.metrics['gpu_memory_used'].reason == 'legacy_history_unavailable'
    assert not result.devices[0].history_available
    assert not any('worker_node_' in expr for expr in calls)


@pytest.mark.asyncio
async def test_nonfinite_and_foreign_owner_values_do_not_escape(monkeypatch):
    def query(expr, start, end):
        labels = {
            'worker_id': '7',
            'worker_uuid': 'server-worker',
            'owner_principal_id': '17',
        }
        if 'worker_gpu_' in expr:
            labels.update(gpu_uuid='GPU-private', gpu_chip_index='2')
        return [
            {'metric': labels, 'values': [[start, 'NaN'], [end, 'Inf']]},
            {'metric': {**labels, 'owner_principal_id': '99'}, 'values': [[end, '80']]},
        ]

    hardware, session, _ = _environment(monkeypatch, [_worker()], query=query)
    result = await _collect(hardware, session)
    assert not result.available
    assert 'GPU-private' not in result.model_dump_json()
    assert all(m.status == 'no_data' for m in result.metrics.values())


@pytest.mark.asyncio
async def test_unowned_worker_is_only_queried_with_empty_owner_snapshot(monkeypatch):
    hardware, session, calls = _environment(monkeypatch, [_worker(owner=None)])
    result = await _collect(hardware, session, ctx=_context(admin=True))
    assert result.available
    assert calls
    assert all('owner_principal_id=""' in expr for expr in calls)


@pytest.mark.asyncio
async def test_selected_owner_changes_opaque_identity_and_query_scope(monkeypatch):
    worker = _worker()
    hardware, session, calls = _environment(monkeypatch, [worker])
    first = await _collect(hardware, session, ctx=_context(admin=True))
    worker.owner_principal_id = 29
    calls.clear()
    second = await _collect(hardware, session, ctx=_context(admin=True))
    assert first.selected_device_id != second.selected_device_id
    assert all('owner_principal_id="29"' in expr for expr in calls)


@pytest.mark.asyncio
async def test_one_sensor_query_failure_keeps_other_resources_and_sanitizes_errors(
    monkeypatch,
):
    hardware, session, _ = _environment(monkeypatch, [_worker()])
    original = hardware.prometheus_query.query_range

    async def failing(client, url, expr, start, end, step):
        if 'worker_gpu_power_watts' in expr:
            raise ValueError('secret http://private-host/metrics')
        return await original(client, url, expr, start, end, step)

    monkeypatch.setattr(hardware.prometheus_query, 'query_range', failing)
    result = await _collect(hardware, session)
    assert result.available
    assert result.metrics['gpu_power'].status == 'query_error'
    assert result.metrics['gpu_memory_used'].status == 'available'
    assert 'private-host' not in result.model_dump_json()


@pytest.mark.asyncio
async def test_identity_probe_and_numeric_queries_share_one_deadline(monkeypatch):
    hardware, session, _ = _environment(monkeypatch, [_worker()])
    ranges = AsyncMock(return_value=[])

    async def never_finishes(*args):
        await asyncio.Event().wait()

    monkeypatch.setattr(hardware, 'COLLECTION_TIMEOUT_SECONDS', 0.01)
    monkeypatch.setattr(hardware.prometheus_query, 'query_instant', never_finishes)
    monkeypatch.setattr(hardware.prometheus_query, 'query_range', ranges)
    result = await _collect(hardware, session)
    assert not result.available
    assert result.devices[0].current['gpu_memory_used'] == 400
    assert ranges.await_count == 0


@pytest.mark.asyncio
async def test_proven_stable_sensor_without_samples_is_not_called_legacy(monkeypatch):
    hardware, session, _ = _environment(monkeypatch, [_worker()])
    original = hardware.prometheus_query.query_instant

    async def instant(client, url, expr, at):
        return await original(client, url, expr, at) if 'identity_info' in expr else []

    monkeypatch.setattr(hardware.prometheus_query, 'query_instant', instant)
    monkeypatch.setattr(
        hardware.prometheus_query, 'query_range', AsyncMock(return_value=[])
    )
    result = await _collect(hardware, session)
    assert result.metrics['gpu_memory_used'].reason == 'no_samples'
    assert not result.devices[0].history_available


@pytest.mark.asyncio
async def test_all_cards_return_current_without_per_card_history_queries(monkeypatch):
    hardware, session, calls = _environment(
        monkeypatch,
        [_worker(devices=[_device(), _device(index=3, uuid='GPU-second')])],
    )
    result = await _collect(hardware, session, [_instance(gpu_indexes=[8, 3])])
    assert [d.index for d in result.devices] == [3, 8]
    assert all(d.current['gpu_memory_used'] == 400 for d in result.devices)
    assert sum('identity_info' in expr for expr in calls) == 1
    numeric = [
        expr for expr in calls if 'worker_gpu_' in expr and 'identity_info' not in expr
    ]
    assert len(numeric) == 10
    assert result.devices[0].history_available
    assert all('gpu_uuid="GPU-second"' in expr for expr in numeric)


@pytest.mark.asyncio
async def test_subordinate_worker_placements_are_counted_and_authorized(monkeypatch):
    worker = _worker()
    other = _worker(owner=99, devices=[_device(index=4, uuid='GPU-other')])
    other.id = 9
    instance = _instance()
    instance.distributed_servers = DistributedServers(
        subordinate_workers=[
            ModelInstanceSubordinateWorker(worker_id=9, gpu_indexes=[4])
        ]
    )
    hardware, session, calls = _environment(monkeypatch, [worker, other])
    result = await _collect(hardware, session, [instance])
    assert result.allocated_gpu_count == result.allocated_worker_count == 2
    assert [d.worker_id for d in result.devices] == [7]
    assert all('worker_id="9"' not in expr for expr in calls)


@pytest.mark.asyncio
async def test_headless_workload_allocation_does_not_require_a_metrics_source(
    monkeypatch,
):
    from gpustack.utils.model_instance_workers import (
        model_instance_metrics_source_inventory,
    )

    instance = _instance()
    instance.ports = None
    instance.distributed_servers = DistributedServers(
        orchestration_mode='managed',
        subordinate_workers=[
            ModelInstanceSubordinateWorker(worker_id=9, gpu_indexes=[])
        ],
        workloads=[
            ModelInstanceWorkload(
                name='remote-rank',
                worker_id=9,
                state='pending',
                dp_rank_placements=[
                    DPRankPlacement(dp_rank=0, node_rank=1, gpu_indexes=[4])
                ],
            )
        ],
    )
    assert model_instance_metrics_source_inventory(instance) == []
    hardware, session, _ = _environment(monkeypatch, [_worker()])
    result = await _collect(hardware, session, [instance])
    assert result.allocated_gpu_count == result.allocated_worker_count == 2
    assert [(device.worker_id, device.index) for device in result.devices] == [(7, 8)]


@pytest.mark.asyncio
async def test_org_member_can_read_owned_hardware(monkeypatch):
    hardware, session, _ = _environment(monkeypatch, [_worker()])
    ctx = _context()
    ctx.org_role = OrgRole.MEMBER
    result = await _collect(hardware, session, ctx=ctx)
    assert result.available


@pytest.mark.asyncio
async def test_historical_samples_without_current_sample_are_marked_stale(monkeypatch):
    hardware, session, _ = _environment(monkeypatch, [_worker()])
    original = hardware.prometheus_query.query_instant

    async def instant(client, url, expr, at):
        return await original(client, url, expr, at) if 'identity_info' in expr else []

    monkeypatch.setattr(hardware.prometheus_query, 'query_instant', instant)
    result = await _collect(hardware, session)
    assert result.metrics['gpu_memory_used'].status == 'stale'
    assert result.metrics['gpu_memory_used'].summaries[0].value is None
    assert result.metrics['gpu_memory_used'].series[0].points


@pytest.mark.asyncio
async def test_unavailable_prometheus_keeps_current_without_claiming_legacy(
    monkeypatch,
):
    hardware, session, calls = _environment(monkeypatch, [_worker()])
    clock = iter((1000, 1100))
    monkeypatch.setattr(hardware, 'time', SimpleNamespace(time=lambda: next(clock)))
    monkeypatch.setattr(hardware.prometheus_query, 'prometheus_url', lambda: None)
    result = await _collect(hardware, session)
    assert result.last_refreshed_at == 1100
    assert result.reason == 'prometheus_unavailable'
    assert result.devices[0].current['gpu_memory_used'] == 400
    assert result.devices[0].reason == 'prometheus_unavailable'
    assert calls == []


@pytest.mark.asyncio
async def test_offline_info_does_not_hide_proven_owner_qualified_history(monkeypatch):
    hardware, session, _ = _environment(monkeypatch, [_worker()])
    monkeypatch.setattr(
        hardware.prometheus_query, 'query_instant', AsyncMock(return_value=[])
    )
    result = await _collect(hardware, session)
    assert result.metrics['gpu_memory_used'].status == 'stale'
    assert result.devices[0].history_available
    assert result.devices[0].reason is None


@pytest.mark.asyncio
async def test_sensor_failure_cancels_its_unfinished_peer_query(monkeypatch):
    hardware, session, _ = _environment(monkeypatch, [_worker()])
    original_range = hardware.prometheus_query.query_range
    original_instant = hardware.prometheus_query.query_instant
    cancelled = asyncio.Event()
    unfinished = []

    async def range_query(client, url, expr, start, end, step):
        if 'worker_gpu_power_watts' in expr:
            await asyncio.sleep(0)
            raise ValueError('query failed')
        return await original_range(client, url, expr, start, end, step)

    async def instant(client, url, expr, at):
        if 'worker_gpu_power_watts' in expr:
            unfinished.append(asyncio.current_task())
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        return await original_instant(client, url, expr, at)

    monkeypatch.setattr(hardware.prometheus_query, 'query_range', range_query)
    monkeypatch.setattr(hardware.prometheus_query, 'query_instant', instant)
    try:
        result = await _collect(hardware, session)
        assert result.metrics['gpu_power'].status == 'query_error'
        assert cancelled.is_set()
    finally:
        for task in unfinished:
            task.cancel()
        await asyncio.gather(*unfinished, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('aware', [True, False])
async def test_last_reported_time_only_uses_known_timezone(monkeypatch, aware):
    worker = _worker()
    stamp = datetime.fromtimestamp(1700000000, timezone.utc)
    worker.heartbeat_time = stamp if aware else stamp.replace(tzinfo=None)
    hardware, session, _ = _environment(monkeypatch, [worker])
    result = await _collect(hardware, session)
    assert result.devices[0].last_reported_at == (1700000000 if aware else None)


@pytest.mark.asyncio
@pytest.mark.parametrize('managed', [False, True])
async def test_scheduled_allocation_is_visible_before_ports_exist(monkeypatch, managed):
    from gpustack.utils.model_instance_workers import (
        model_instance_metrics_source_inventory,
    )

    instance = _instance()
    instance.state = ModelInstanceStateEnum.SCHEDULED
    instance.ports = None
    instance.distributed_servers = DistributedServers(
        orchestration_mode='managed' if managed else 'legacy',
        workloads=[] if managed else None,
        subordinate_workers=[
            ModelInstanceSubordinateWorker(worker_id=9, gpu_indexes=[4])
        ],
    )
    assert model_instance_metrics_source_inventory(instance) == []
    other = _worker(owner=99, devices=[_device(index=4, uuid='GPU-other')])
    other.id = 9
    hardware, session, calls = _environment(monkeypatch, [_worker(), other])
    result = await _collect(hardware, session, [instance])
    assert result.allocated_gpu_count == result.allocated_worker_count == 2
    assert [(device.worker_id, device.index) for device in result.devices] == [(7, 8)]
    assert result.devices[0].current['gpu_memory_used'] == 400
    assert result.available
    assert all('worker_id="9"' not in expr for expr in calls)
