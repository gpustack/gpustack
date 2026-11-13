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
    from gpustack.monitoring import hardware

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
@pytest.mark.parametrize('all_devices', [False, True])
async def test_custom_hardware_interval_keeps_every_device_on_the_same_bounds(
    monkeypatch, all_devices
):
    bounds = []

    def query(expr, start, end):
        bounds.append((start, end))
        labels = {
            key: json.loads(value)
            for key, value in re.findall(r'(\w+)=("(?:[^"\\]|\\.)*")', expr)
        }
        if 'identity_info' in expr:
            labels.update(gpu_uuid='GPU-private', gpu_chip_index='2', gpu_index='8')
        return [{'metric': labels, 'values': [[start, '1'], [end, '1']]}]

    hardware, session, _ = _environment(monkeypatch, [_worker()], query=query)
    monkeypatch.setattr(hardware.time, 'time', lambda: 5000)
    result = await _collect(
        hardware, session, start=1000, end=2000, all_devices=all_devices
    )
    assert (result.start, result.end) == (1000, 2000)
    assert all(pair in ((1000, 2000), (2000, 2000)) for pair in bounds)
    assert any(pair == (1000, 2000) for pair in bounds)
    assert result.metrics['gpu_utilization'].series[0].points == [(1000, 1), (2000, 1)]


@pytest.mark.asyncio
@pytest.mark.parametrize('all_devices', [False, True])
async def test_model_access_and_cluster_grant_do_not_grant_worker_metrics(
    monkeypatch, all_devices
):
    hardware, session, calls = _environment(monkeypatch, [_worker(owner=99)])
    result = await _collect(hardware, session, all_devices=all_devices)
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


def _batch_query(workers):
    def query(expr, start, end):
        gpu = 'worker_gpu_' in expr
        identity = 'identity_info' in expr
        rows = []
        for worker in workers:
            base = {
                'worker_id': str(worker.id),
                'worker_uuid': worker.worker_uuid,
                'owner_principal_id': str(worker.owner_principal_id or ''),
            }
            for device in worker.status.gpu_devices if gpu else [None]:
                labels = dict(base)
                if device is not None:
                    labels.update(
                        gpu_uuid=device.uuid,
                        gpu_chip_index=str(device.device_chip_index),
                    )
                if identity:
                    labels['gpu_index'] = str(device.index)
                value = (
                    1
                    if identity
                    else (0 if device and device.index == 9 else worker.id)
                )
                rows.append(
                    {
                        'metric': labels,
                        'values': [[start, str(value)], [end, str(value)]],
                    }
                )
                rows.append(
                    {
                        'metric': {**labels, 'owner_principal_id': 'foreign'},
                        'values': [[end, '99']],
                    }
                )
        return rows

    return query


@pytest.mark.asyncio
async def test_all_devices_query_all_authorized_gpus_and_unique_workers_in_fixed_batches(
    monkeypatch,
):
    first = _worker(
        devices=[_device(chip=0), _device(index=9, uuid='GPU-second', chip=0)]
    )
    other = _worker(devices=[_device(uuid='GPU-third', chip=0)])
    other.id, other.name, other.worker_uuid = 8, 'second-worker', 'second-uuid'
    router = _worker(devices=[])
    router.id, router.name, router.worker_uuid = 9, 'router-worker', 'router-uuid'
    workers = [first, other, router]
    second_instance = _instance(13)
    second_instance.worker_id = 8
    router_instance = _instance(14, gpu_indexes=[])
    router_instance.worker_id = 9
    hardware, session, calls = _environment(
        monkeypatch, workers, query=_batch_query(workers)
    )
    result = await _collect(
        hardware,
        session,
        [
            _instance(gpu_indexes=[8, 9]),
            _instance(12),
            second_instance,
            router_instance,
        ],
        all_devices=True,
    )
    assert result.all_devices and result.selected_device_id is None
    assert result.allocated_gpu_count == result.allocated_worker_count == 3
    assert [(worker.id, worker.name) for worker in result.workers] == [
        (7, 'visible-worker'),
        (8, 'second-worker'),
        (9, 'router-worker'),
    ]
    assert len(calls) == 17
    assert sum('identity_info' in expr for expr in calls) == 1
    for key, metric in result.metrics.items():
        assert metric.status == 'available' and metric.reason is None
        assert len(metric.series) == 3
        assert len(metric.summaries) == 3
        if key.startswith('gpu_'):
            assert {line.dimensions.device_id for line in metric.series} == {
                device.id for device in result.devices
            }
        else:
            assert [line.dimensions.worker_id for line in metric.series] == [7, 8, 9]
    assert [s.value for s in result.metrics['gpu_utilization'].summaries] == [7, 0, 8]
    assert all(
        device.history_available and device.reason is None for device in result.devices
    )
    assert 'GPU-private' not in result.model_dump_json()
    assert all('owner_principal_id="17"' in expr for expr in calls)


@pytest.mark.asyncio
async def test_all_devices_keeps_missing_and_partition_members_without_claiming_zero(
    monkeypatch,
):
    worker = _worker(
        devices=[
            _device(chip=0),
            _device(index=9, uuid='', chip=0),
            _device(index=10, uuid='MIG-partition', chip=0),
        ]
    )
    hardware, session, calls = _environment(
        monkeypatch, [worker], query=_batch_query([worker])
    )
    result = await _collect(
        hardware, session, [_instance(gpu_indexes=[8, 9, 10])], all_devices=True
    )
    metric = result.metrics['gpu_utilization']
    assert metric.status == 'available' and metric.reason == 'partial_metric_data'
    assert [summary.value for summary in metric.summaries] == [7, None, None]
    assert metric.series[1].points == metric.series[2].points == []
    assert result.devices[1].reason == 'device_identity_unresolved'
    assert not result.devices[1].history_available
    assert result.devices[0].history_available and result.devices[2].history_available
    sensors = [expr for expr in calls if 'worker_gpu_utilization_percent' in expr]
    assert all(
        'MIG-partition' not in expr and 'gpu_uuid=""' not in expr for expr in sensors
    )
    assert result.metrics['gpu_memory_used'].summaries[2].value == 7


@pytest.mark.asyncio
async def test_all_devices_preserves_worker_visibility_and_cpu_only_history(
    monkeypatch,
):
    denied = _worker(owner=99)
    denied.id, denied.name = 8, 'denied-worker'
    cpu = _worker(devices=[])
    cpu.id, cpu.name, cpu.worker_uuid = 9, 'cpu-worker', 'cpu-uuid'
    denied_instance = _instance(12)
    denied_instance.worker_id = 8
    cpu_instance = _instance(13, gpu_indexes=[])
    cpu_instance.worker_id = 9
    workers = [_worker(), denied, cpu]
    hardware, session, calls = _environment(
        monkeypatch, workers, query=_batch_query(workers)
    )
    result = await _collect(
        hardware,
        session,
        [_instance(), denied_instance, cpu_instance],
        all_devices=True,
    )
    assert [worker.id for worker in result.workers] == [7, 9]
    assert [
        summary.dimensions.worker_id for summary in result.metrics['host_cpu'].summaries
    ] == [7, 9]
    assert len(result.devices) == 1
    assert all(
        'worker_id="8"' not in expr and 'owner_principal_id="99"' not in expr
        for expr in calls
    )
    assert 'denied-worker' not in result.model_dump_json()


@pytest.mark.asyncio
async def test_all_devices_rejects_conflicting_selection_and_excess_targets_before_queries(
    monkeypatch,
):
    hardware, session, calls = _environment(monkeypatch, [_worker()])
    with pytest.raises(InvalidException):
        await _collect(hardware, session, all_devices=True, device_id='opaque')
    monkeypatch.setattr(hardware, 'MAX_RESOURCE_TARGETS', 0)
    with pytest.raises(InvalidException):
        await _collect(hardware, session, all_devices=True)
    assert calls == []


@pytest.mark.asyncio
async def test_all_devices_failure_keeps_other_families_and_cancels_peer_query(
    monkeypatch,
):
    hardware, session, _ = _environment(monkeypatch, [_worker()])
    original = hardware.prometheus_query.query_range
    original_instant = hardware.prometheus_query.query_instant
    cancelled = asyncio.Event()

    async def ranged(client, url, expr, start, end, step):
        if 'worker_gpu_power_watts' in expr:
            await asyncio.sleep(0)
            raise ValueError('private http://internal/')
        return await original(client, url, expr, start, end, step)

    async def instant(client, url, expr, at):
        if 'worker_gpu_power_watts' in expr:
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        return await original_instant(client, url, expr, at)

    monkeypatch.setattr(hardware.prometheus_query, 'query_range', ranged)
    monkeypatch.setattr(hardware.prometheus_query, 'query_instant', instant)
    result = await _collect(hardware, session, all_devices=True)
    assert cancelled.is_set()
    assert result.metrics['gpu_power'].status == 'query_error'
    assert result.metrics['gpu_power'].series[0].points == []
    assert result.metrics['gpu_power'].summaries[0].value is None
    assert result.metrics['gpu_memory_used'].status == 'available'
    assert result.devices[0].history_available
    assert 'internal' not in result.model_dump_json()


@pytest.mark.asyncio
async def test_all_devices_deadline_keeps_inventory_and_empty_authorized_series(
    monkeypatch,
):
    hardware, session, _ = _environment(monkeypatch, [_worker()])

    async def never_finishes(*args):
        await asyncio.Event().wait()

    monkeypatch.setattr(hardware, 'COLLECTION_TIMEOUT_SECONDS', 0.01)
    monkeypatch.setattr(hardware.prometheus_query, 'query_instant', never_finishes)
    result = await _collect(hardware, session, all_devices=True)
    assert not result.available
    assert not result.devices[0].history_available
    assert result.devices[0].current['gpu_memory_used'] == 400
    metric = result.metrics['gpu_memory_used']
    assert len(metric.series) == 1 and metric.series[0].points == []
    assert metric.series[0].dimensions.device_id == result.devices[0].id
    assert metric.status == 'query_error'


@pytest.mark.asyncio
@pytest.mark.parametrize('count', [1, 32, 128])
async def test_all_device_query_count_does_not_grow_with_device_count(
    monkeypatch, count
):
    worker = _worker(
        devices=[
            _device(index=index, uuid=f'GPU-{index}', chip=0) for index in range(count)
        ]
    )
    hardware, session, calls = _environment(
        monkeypatch, [worker], query=_batch_query([worker])
    )
    result = await _collect(
        hardware, session, [_instance(gpu_indexes=list(range(count)))], all_devices=True
    )
    assert len(calls) == 17
    assert len(result.metrics['gpu_memory_used'].series) == count
    assert len(result.metrics['host_cpu'].series) == 1
    assert all(device.history_available for device in result.devices)


@pytest.mark.asyncio
async def test_all_devices_without_prometheus_keeps_named_gaps_and_real_snapshots(
    monkeypatch,
):
    hardware, session, calls = _environment(monkeypatch, [_worker()])
    monkeypatch.setattr(hardware.prometheus_query, 'prometheus_url', lambda: None)
    result = await _collect(hardware, session, all_devices=True)
    assert result.reason == 'prometheus_unavailable'
    assert result.devices[0].current['gpu_memory_used'] == 400
    metric = result.metrics['gpu_memory_used']
    assert metric.reason == 'prometheus_unavailable' and metric.status == 'query_error'
    assert len(metric.series) == 1 and metric.series[0].points == []
    assert metric.series[0].dimensions.device_id == result.devices[0].id
    assert metric.summaries[0].value is None
    assert calls == []
