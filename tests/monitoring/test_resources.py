from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.dialects import mysql, postgresql

from gpustack.api.exceptions import InvalidException, NotFoundException
from gpustack.schemas.workers import MountPoint, WorkerStatus
from tests.monitoring.test_hardware import _worker, _context, _instance


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'seconds,fixed,step',
    [
        (15, True, 1),
        (273, True, 2),
        (3600, True, 15),
        (604800, True, 2520),
        (300, False, 15),
    ],
)
async def test_fixed_hardware_inspection_is_detailed_without_changing_relative_grids(
    monkeypatch, seconds, fixed, step
):
    from gpustack.monitoring import resources

    end = 1700000000
    monkeypatch.setattr(resources.time, 'time', lambda: end)
    monkeypatch.setattr(
        resources.hardware.prometheus_query, 'prometheus_url', lambda: 'http://prom'
    )
    collect = AsyncMock()
    monkeypatch.setattr(resources.hardware, '_collect_all_history', collect)
    response = await resources.collect_resource_metrics(
        _worker(),
        [],
        window_seconds=seconds,
        start=end - seconds if fixed else None,
        end=end if fixed else None,
        client=object(),
    )
    assert response.step == step
    assert collect.await_args.args[2].step == step
    assert seconds / step <= 240


@pytest.mark.asyncio
async def test_short_hardware_inspection_retains_a_brief_cpu_peak(monkeypatch):
    import math
    from gpustack.monitoring import resources

    worker = _worker()
    labels = resources.hardware._worker_identity(worker)
    monkeypatch.setattr(
        resources.hardware.prometheus_query, 'prometheus_url', lambda: 'http://prom'
    )

    def value(at):
        return '80' if 1006 <= at < 1010 else '20'

    async def ranged(client, url, query, start, end, step):
        if 'worker_host_cpu_utilization_percent' not in query:
            return []
        return [
            {
                'metric': labels,
                'values': [
                    [at, value(at)]
                    for at in (
                        start + i * step
                        for i in range(math.floor((end - start) / step) + 1)
                    )
                ],
            }
        ]

    async def instant(client, url, query, at):
        return (
            [{'metric': labels, 'value': [at, value(at)]}]
            if 'worker_host_cpu_utilization_percent' in query
            else []
        )

    monkeypatch.setattr(resources.hardware.prometheus_query, 'query_range', ranged)
    monkeypatch.setattr(resources.hardware.prometheus_query, 'query_instant', instant)
    result = await resources.collect_resource_metrics(
        worker, [], window_seconds=20, start=1000, end=1020, client=object()
    )
    points = result.metrics['host_cpu'].series[0].points
    assert max(value for _, value in points if value is not None) == 80
    assert all(1000 <= at < 1020 for at, _ in points)


@pytest.mark.asyncio
async def test_direct_worker_denies_foreign_owner_before_prometheus(monkeypatch):
    from gpustack.routes import resource_monitoring as routes

    query = AsyncMock()
    monkeypatch.setattr(routes, 'collect_resource_metrics', query)
    monkeypatch.setattr(routes.Worker, 'one_by_id', AsyncMock(return_value=_worker(99)))
    with pytest.raises(NotFoundException):
        await routes.get_worker_metrics(
            SimpleNamespace(), SimpleNamespace(), _context(), 7
        )
    query.assert_not_called()


def test_associations_include_distributed_gpu_and_exclude_other_worker():
    from gpustack.monitoring.resources import instance_gpu_indexes
    from gpustack.schemas.models import (
        DistributedServers,
        ModelInstanceSubordinateWorker,
    )

    instance = _instance(gpu_indexes=[1, 2])
    instance.distributed_servers = DistributedServers(
        subordinate_workers=[
            ModelInstanceSubordinateWorker(worker_id=8, gpu_indexes=[4])
        ]
    )
    assert instance_gpu_indexes(instance, 7) == {1, 2}
    assert instance_gpu_indexes(instance, 8) == {4}
    assert instance_gpu_indexes(instance, 9) == set()


@pytest.mark.asyncio
@pytest.mark.parametrize('selected_gpu', [False, True])
async def test_association_placements_include_all_visible_workers_and_gpus(
    selected_gpu,
):
    from gpustack.schemas.models import (
        DistributedServers,
        Model,
        ModelInstanceSubordinateWorker,
    )
    from gpustack.monitoring.resources import inventory_devices, resource_associations
    from tests.monitoring.test_hardware import _device

    worker = _worker(devices=[_device(1), _device(2)])
    instance = _instance(gpu_indexes=[2, 1, 1])
    instance.distributed_servers = DistributedServers(
        subordinate_workers=[
            ModelInstanceSubordinateWorker(worker_id=8, gpu_indexes=[3, 0]),
            ModelInstanceSubordinateWorker(worker_id=9, gpu_indexes=[4]),
            ModelInstanceSubordinateWorker(worker_id=10, gpu_indexes=[5]),
            ModelInstanceSubordinateWorker(worker_id=11, gpu_indexes=[6]),
        ]
    )
    peer = _worker()
    peer.id, peer.name, peer.ip = 8, 'peer-worker', '192.0.2.8'
    foreign = _worker(owner=99)
    foreign.id, foreign.name = 9, 'private-worker'
    removed = _worker()
    removed.id = 10
    from datetime import datetime, timezone

    removed.deleted_at = datetime.now(timezone.utc)
    unrelated = _worker()
    unrelated.id = 12
    model = Model(id=5, name='visible', owner_principal_id=17)
    session = SimpleNamespace(
        exec=AsyncMock(
            side_effect=[
                SimpleNamespace(all=lambda: [(instance, model)]),
                SimpleNamespace(all=lambda: [peer, foreign, removed, unrelated]),
            ]
        )
    )
    devices = inventory_devices(worker)
    result = await resource_associations(
        worker, devices, _context(), session, devices[0] if selected_gpu else None
    )
    association = result[0]
    assert association.spans_workers
    assert not association.placements_complete
    assert [p.worker_id for p in association.placements] == [7, 8]
    assert [p.gpu_indexes for p in association.placements] == [[1, 2], [0, 3]]
    assert [p.is_main for p in association.placements] == [True, False]
    assert association.placements[1].worker_name == 'peer-worker'
    assert association.placements[1].worker_ip == '192.0.2.8'
    assert len(association.gpu_device_ids) == 2
    assert 'private-worker' not in association.model_dump_json()
    assert session.exec.await_count == 2


@pytest.mark.asyncio
async def test_local_multi_gpu_placement_does_not_query_unrelated_workers():
    from gpustack.schemas.models import Model
    from gpustack.monitoring.resources import inventory_devices, resource_associations

    worker = _worker()
    instance = _instance(gpu_indexes=[8, 9])
    model = Model(id=5, name='visible', owner_principal_id=17)
    session = SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(all=lambda: [(instance, model)]))
    )
    result = await resource_associations(
        worker, inventory_devices(worker), _context(), session
    )
    assert result[0].placements_complete
    assert result[0].placements[0].gpu_indexes == [8, 9]
    assert not result[0].spans_workers
    session.exec.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize('dialect', [postgresql.dialect(), mysql.dialect()])
async def test_subordinate_worker_view_keeps_main_worker_and_cpu_placement(dialect):
    from gpustack.schemas.models import (
        DistributedServers,
        Model,
        ModelInstanceSubordinateWorker,
    )
    from gpustack.monitoring.resources import inventory_devices, resource_associations

    worker = _worker()
    instance = _instance(gpu_indexes=[])
    instance.worker_id = 8
    instance.cluster_id = worker.cluster_id
    instance.distributed_servers = DistributedServers(
        subordinate_workers=[
            ModelInstanceSubordinateWorker(worker_id=7, gpu_indexes=[8])
        ]
    )
    main = _worker()
    main.id, main.name = 8, 'main-worker'
    model = Model(id=5, name='visible', owner_principal_id=17)
    session = SimpleNamespace(
        exec=AsyncMock(
            side_effect=[
                SimpleNamespace(all=lambda: [(instance, model)]),
                SimpleNamespace(all=lambda: [main]),
            ]
        )
    )
    result = await resource_associations(
        worker, inventory_devices(worker), _context(), session
    )
    statement = session.exec.await_args_list[0].args[0]
    sql = str(
        statement.compile(dialect=dialect, compile_kwargs={'literal_binds': True})
    )
    assert f'model_instances.cluster_id = {worker.cluster_id}' in sql
    assert 'model_instances.owner_principal_id = 17' in sql
    assert 'models.owner_principal_id = 17' in sql
    assert result[0].placements_complete
    assert [p.worker_id for p in result[0].placements] == [8, 7]
    assert [p.gpu_indexes for p in result[0].placements] == [[], [8]]
    assert result[0].placements[0].is_main


@pytest.mark.asyncio
async def test_filesystem_queries_use_owner_uuid_and_mountpoint(monkeypatch):
    from gpustack.monitoring import resources

    worker = _worker()
    worker.status = WorkerStatus(
        filesystem=[
            MountPoint(
                name='ext4',
                mount_point='/data',
                mount_from='/dev/sda',
                total=100,
                used=0,
                free=100,
                available=100,
            )
        ]
    )
    calls = []

    async def query(client, url, expr, *args):
        calls.append(expr)
        return []

    monkeypatch.setattr(resources.hardware.prometheus_query, 'query_range', query)
    monkeypatch.setattr(resources.hardware.prometheus_query, 'query_instant', query)
    monkeypatch.setattr(
        resources.hardware.prometheus_query, 'prometheus_url', lambda: 'http://prom'
    )
    response = await resources.collect_resource_metrics(
        worker, [], window_seconds=300, client=object()
    )
    calls = [expr for expr in calls if 'worker_filesystem_' in expr]
    assert len(calls) == 2
    assert all(
        'worker_uuid="server-worker"' in expr
        and 'owner_principal_id="17"' in expr
        and 'mountpoint="/data"' in expr
        for expr in calls
    )
    assert response.metrics['filesystem_usage'].series[0].points == []
    assert response.metrics['filesystem_usage'].summaries[0].value is None


@pytest.mark.asyncio
async def test_cpu_only_worker_has_host_history_without_gpu(monkeypatch):
    from gpustack.monitoring import resources

    worker = _worker(devices=[])
    monkeypatch.setattr(
        resources.hardware.prometheus_query, 'prometheus_url', lambda: None
    )
    result = await resources.collect_resource_metrics(worker, [], window_seconds=3600)
    assert result.worker_id == 7
    assert result.devices == []
    assert result.metrics['gpu_utilization'].status == 'not_applicable'
    assert result.metrics['host_cpu'].status == 'query_error'
    assert 'model_id' not in result.model_dump()


@pytest.mark.asyncio
async def test_gpu_metrics_reject_inventory_uuid_replacement_before_query(monkeypatch):
    from gpustack.routes import resource_monitoring as routes

    gpu = SimpleNamespace(
        worker_id=7,
        index=8,
        uuid='GPU-replaced',
        device_chip_index=2,
        owner_principal_id=17,
    )
    monkeypatch.setattr(routes.GPUDevice, 'one_by_id', AsyncMock(return_value=gpu))
    monkeypatch.setattr(routes.Worker, 'one_by_id', AsyncMock(return_value=_worker()))
    query = AsyncMock()
    monkeypatch.setattr(routes, 'collect_resource_metrics', query)
    with pytest.raises(NotFoundException):
        await routes.get_gpu_metrics(
            SimpleNamespace(), SimpleNamespace(), _context(), 'view-id'
        )
    query.assert_not_called()


@pytest.mark.asyncio
async def test_gpu_metrics_require_visible_parent_worker(monkeypatch):
    from gpustack.routes import resource_monitoring as routes

    gpu = SimpleNamespace(
        worker_id=7,
        index=8,
        uuid='GPU-private',
        device_chip_index=2,
        owner_principal_id=17,
    )
    monkeypatch.setattr(routes.GPUDevice, 'one_by_id', AsyncMock(return_value=gpu))
    monkeypatch.setattr(
        routes.Worker, 'one_by_id', AsyncMock(return_value=_worker(owner=99))
    )
    query = AsyncMock()
    monkeypatch.setattr(routes, 'collect_resource_metrics', query)
    with pytest.raises(NotFoundException):
        await routes.get_gpu_metrics(
            SimpleNamespace(), SimpleNamespace(), _context(), 'view-id'
        )
    query.assert_not_called()


@pytest.mark.asyncio
async def test_visible_worker_does_not_expose_foreign_models_or_instances():
    from gpustack.schemas.models import Model
    from gpustack.monitoring.resources import (
        inventory_devices,
        resource_associations,
    )

    worker = _worker()
    devices = inventory_devices(worker)
    visible = _instance()
    foreign_instance = _instance(id=12)
    foreign_instance.owner_principal_id = 99
    visible_model = Model(id=5, name='visible', owner_principal_id=17)
    foreign_model = Model(id=6, name='private', owner_principal_id=99)
    rows = [
        (visible, visible_model),
        (foreign_instance, visible_model),
        (_instance(id=13), foreign_model),
    ]
    session = SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(all=lambda: rows))
    )
    result = await resource_associations(worker, devices, _context(), session)
    assert [item.id for item in result] == [visible.id]
    assert devices[0].public.model_instance_ids == [visible.id]


@pytest.mark.asyncio
async def test_read_only_resource_role_cannot_enumerate_deployments():
    from gpustack.schemas.principals import OrgRole
    from gpustack.monitoring.resources import (
        inventory_devices,
        resource_associations,
    )

    ctx = _context()
    ctx.org_role = OrgRole.MEMBER
    session = SimpleNamespace(exec=AsyncMock())
    worker = _worker()
    result = await resource_associations(
        worker, inventory_devices(worker), ctx, session
    )
    assert result == []
    session.exec.assert_not_called()


@pytest.mark.asyncio
async def test_unbound_or_removed_opaque_gpu_is_not_replaced_with_first_gpu():
    from gpustack.monitoring.resources import (
        collect_resource_metrics,
        inventory_devices,
    )

    worker = _worker()
    with pytest.raises(NotFoundException):
        await collect_resource_metrics(
            worker,
            inventory_devices(worker),
            device_id='unrelated',
            window_seconds=3600,
        )


@pytest.mark.asyncio
async def test_gpu_history_contains_only_the_selected_current_identity(monkeypatch):
    from tests.monitoring.test_hardware import _device, _environment
    from gpustack.monitoring.resources import (
        collect_resource_metrics,
        inventory_devices,
    )

    worker = _worker(devices=[_device(1, 'GPU-a', 0), _device(2, 'GPU-b', 0)])
    _, _, calls = _environment(monkeypatch, [worker])
    devices = inventory_devices(worker)
    result = await collect_resource_metrics(
        worker,
        devices,
        device_id=devices[1].public.id,
        window_seconds=3600,
        client=object(),
    )
    assert len(result.devices) == 1
    assert result.devices[0].index == 2
    assert all('GPU-a' not in expr for expr in calls if 'identity_info' not in expr)
    for key, metric in result.metrics.items():
        if key.startswith('gpu_'):
            assert all(
                series.dimensions.device_id == devices[1].public.id
                for series in metric.series
            )


@pytest.mark.asyncio
@pytest.mark.parametrize('selected_gpu', [False, True])
@pytest.mark.parametrize(
    'start,end,query_start,query_end',
    [
        (1730000000.123, 1730000000.124, 1730000000.123, 1730000000.123),
        (1730000000.25, 1730000040.75, 1730000000.25, 1730000040.749),
        (1000.250001, 1000.251001, 1000.251, 1000.251),
    ],
)
async def test_resource_phase_queries_keep_exact_bounds_and_owner_identity(
    monkeypatch, selected_gpu, start, end, query_start, query_end
):
    from tests.monitoring.test_hardware import _device, _environment
    from gpustack.monitoring.resources import (
        collect_resource_metrics,
        inventory_devices,
    )

    worker = _worker(devices=[_device(1, 'GPU-a', 0), _device(2, 'GPU-b', 0)])
    hardware, _, calls = _environment(monkeypatch, [worker])
    range_query = AsyncMock(wraps=hardware.prometheus_query.query_range)
    instant_query = AsyncMock(wraps=hardware.prometheus_query.query_instant)
    monkeypatch.setattr(hardware.prometheus_query, 'query_range', range_query)
    monkeypatch.setattr(hardware.prometheus_query, 'query_instant', instant_query)
    devices = inventory_devices(worker)
    device_id = devices[1].public.id if selected_gpu else None
    result = await collect_resource_metrics(
        worker,
        devices,
        device_id=device_id,
        window_seconds=3600,
        start=start,
        end=end,
        client=object(),
    )
    assert (result.start, result.end) == (start, end)
    assert result.selected_device_id == device_id
    assert range_query.await_count > 0 and instant_query.await_count > 0
    assert all(
        call.args[3:5] == (query_start, query_end)
        for call in range_query.await_args_list
    )
    assert all(call.args[3] == query_end for call in instant_query.await_args_list)
    assert all('owner_principal_id="17"' in expr for expr in calls)
    assert all('worker_uuid="server-worker"' in expr for expr in calls)
    assert all(f'>= {start}' in expr for expr in calls if 'identity_info' not in expr)
    if selected_gpu:
        assert [device.index for device in result.devices] == [2]
        assert all('GPU-a' not in expr for expr in calls if 'identity_info' not in expr)
        assert all(
            series.dimensions.device_id == device_id
            for key, metric in result.metrics.items()
            if key.startswith('gpu_')
            for series in metric.series
        )


@pytest.mark.asyncio
@pytest.mark.parametrize('fixed_range', [False, True])
@pytest.mark.parametrize('observed', [False, True])
@pytest.mark.parametrize('selected_gpu', [False, True])
@pytest.mark.parametrize(
    'start,end', [(1730000000.123, 1730000000.124), (1000.25, 1040.75)]
)
async def test_fixed_phase_excludes_boundary_and_old_samples_without_requiring_new_metadata(
    monkeypatch, fixed_range, observed, selected_gpu, start, end
):
    import math
    from gpustack.monitoring import resources

    worker = _worker()
    labels = {
        'worker_id': '7',
        'worker_uuid': 'server-worker',
        'owner_principal_id': '17',
        'gpu_uuid': 'GPU-private',
        'gpu_chip_index': '2',
        'gpu_index': '8',
    }
    samples = [(start - 0.001, 75), (end, 99)]
    if observed:
        samples.append((start, 0))
    calls = []

    def rows(expr, ticks, field):
        calls.append((expr, ticks))
        if 'identity_info' in expr:
            # Stable metadata can be older than the selected phase.
            assert '>= ' not in expr
            return [{'metric': labels, 'value': [ticks[-1], '1']}]
        if 'worker_gpu_utilization' not in expr:
            return []
        guarded = f'>= {start}' in expr
        values = []
        for tick in ticks:
            eligible = [(t, v) for t, v in samples if t <= tick]
            if eligible:
                timestamp, value = max(eligible)
                if not guarded or timestamp >= start:
                    values.append([tick, str(value)])
        result = (
            [{'metric': labels, field: values if field == 'values' else values[-1]}]
            if values
            else []
        )
        # Unexpected rows cannot substitute another tenant, Worker or GPU identity.
        for key, value in (
            ('owner_principal_id', '99'),
            ('worker_uuid', 'old-worker'),
            ('gpu_uuid', 'replaced-gpu'),
        ):
            foreign = [[tick, '88'] for tick in ticks]
            result.append(
                {
                    'metric': {**labels, key: value},
                    field: foreign if field == 'values' else foreign[-1],
                }
            )
        return result

    async def ranged(client, url, expr, query_start, query_end, step):
        ticks = [
            query_start + n * step
            for n in range(math.floor((query_end - query_start) / step) + 1)
        ]
        return rows(expr, ticks, 'values')

    async def instant(client, url, expr, at):
        return rows(expr, [at], 'value')

    monkeypatch.setattr(resources.time, 'time', lambda: end)
    monkeypatch.setattr(
        resources.hardware.prometheus_query, 'prometheus_url', lambda: 'http://prom'
    )
    monkeypatch.setattr(resources.hardware.prometheus_query, 'query_range', ranged)
    monkeypatch.setattr(resources.hardware.prometheus_query, 'query_instant', instant)
    devices = resources.inventory_devices(worker)
    result = await resources.collect_resource_metrics(
        worker,
        devices,
        device_id=devices[0].public.id if selected_gpu else None,
        window_seconds=300,
        start=start if fixed_range else None,
        end=end if fixed_range else None,
        client=object(),
    )
    assert (result.start, result.end) == (
        start if fixed_range else end - 300,
        end,
    )
    metric = result.metrics['gpu_utilization']
    assert metric.summaries[0].value == (
        (0 if observed else None) if fixed_range else 99
    )
    assert metric.status == ('available' if observed or not fixed_range else 'no_data')
    assert result.devices[0].history_available is (observed or not fixed_range)
    if fixed_range:
        assert all(start <= tick < end for _, ticks in calls for tick in ticks)
        assert all(
            f'>= {start}' in expr for expr, _ in calls if 'identity_info' not in expr
        )
        assert all(
            value in (0, None) for series in metric.series for _, value in series.points
        )
    else:
        assert any(end in ticks for _, ticks in calls)
        assert all('>= ' not in expr for expr, _ in calls)


@pytest.mark.asyncio
async def test_fixed_phase_never_replaces_a_removed_opaque_gpu(monkeypatch):
    from gpustack.monitoring import resources
    from tests.monitoring.test_hardware import _device

    worker = _worker(devices=[_device(1, 'GPU-old', 0)])
    old_device_id = resources.inventory_devices(worker)[0].public.id
    worker.status.gpu_devices = [_device(1, 'GPU-new', 0)]
    query = AsyncMock()
    monkeypatch.setattr(resources.hardware.prometheus_query, 'query_range', query)
    monkeypatch.setattr(resources.hardware.prometheus_query, 'query_instant', query)
    with pytest.raises(NotFoundException):
        await resources.collect_resource_metrics(
            worker,
            resources.inventory_devices(worker),
            device_id=old_device_id,
            window_seconds=3600,
            start=1730000000.25,
            end=1730000040.75,
            client=object(),
        )
    query.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('start,end', [(1000, None), (1000, 1000.0009), (0, 604801)])
async def test_invalid_resource_fixed_range_is_not_replaced_by_relative_window(
    monkeypatch, start, end
):
    from gpustack.monitoring import resources

    query = AsyncMock()
    monkeypatch.setattr(resources.hardware, '_collect_all_history', query)
    worker = _worker()
    with pytest.raises(InvalidException):
        await resources.collect_resource_metrics(
            worker,
            resources.inventory_devices(worker),
            window_seconds=3600,
            start=start,
            end=end,
            client=object(),
        )
    query.assert_not_awaited()


@pytest.mark.asyncio
async def test_resource_capabilities_authorize_before_storage_policy(monkeypatch):
    from gpustack.routes import resource_monitoring as routes

    monkeypatch.setattr(routes.Worker, 'one_by_id', AsyncMock(return_value=_worker(99)))
    policy = AsyncMock()
    monkeypatch.setattr(routes, 'collect_storage_policy', policy)
    with pytest.raises(NotFoundException):
        await routes.get_worker_capabilities(
            SimpleNamespace(), SimpleNamespace(), _context(), 7
        )
    policy.assert_not_called()


def test_owner_qualified_filesystem_export_retains_zero_and_drops_duplicate_mounts():
    from gpustack.worker.exporter import _stable_filesystems
    from gpustack.schemas.workers import MountPoint

    mount = MountPoint(name='ext4', mount_point='/data', total=100, used=0)
    families = list(
        _stable_filesystems(
            [mount],
            ['worker_id', 'worker_uuid', 'owner_principal_id'],
            ['7', 'uuid-7', '17'],
        )
    )
    utilization = next(f for f in families if f.name.endswith('utilization_percent'))
    assert utilization.samples[0].value == 0
    assert utilization.samples[0].labels['owner_principal_id'] == '17'
    assert not list(_stable_filesystems([mount, mount], [], []))


@pytest.mark.asyncio
async def test_grafana_redirect_preserves_time_and_legacy_inventory_slot(monkeypatch):
    from urllib.parse import parse_qs, urlparse
    from gpustack.routes import resource_monitoring as routes
    from gpustack.monitoring.resources import inventory_devices

    worker = _worker()
    monkeypatch.setattr(routes.Worker, 'one_by_id', AsyncMock(return_value=worker))
    monkeypatch.setattr(
        routes.Cluster,
        'one_by_id',
        AsyncMock(return_value=SimpleNamespace(name='Visible cluster')),
    )
    monkeypatch.setattr(
        routes,
        'get_global_config',
        lambda: SimpleNamespace(
            get_grafana_url=lambda: 'https://grafana.example',
            grafana_worker_dashboard_uid='workers',
        ),
    )
    monkeypatch.setattr(
        routes, 'resolve_grafana_base_url', lambda *args: 'https://grafana.example'
    )
    result = await routes.get_worker_monitoring_dashboard(
        SimpleNamespace(),
        SimpleNamespace(),
        _context(),
        7,
        start=1730000000,
        end=1730003600,
        device_id=inventory_devices(worker)[0].public.id,
    )
    query = parse_qs(urlparse(result.headers['location']).query)
    assert query['from'] == ['1730000000000']
    assert query['to'] == ['1730003600000']
    assert query['var-worker_name'] == [worker.name]
    # This inventory's logical GPU 8 / chip 2 is exported in node slot 0 / chip 0.
    assert query['var-gpu_index'] == ['0']
    assert query['var-gpu_chip_index'] == ['0']


@pytest.mark.asyncio
@pytest.mark.parametrize('id', [-1, 0, 2**63])
async def test_invalid_worker_identifiers_do_not_reach_database(monkeypatch, id):
    from gpustack.api.exceptions import InvalidException
    from gpustack.routes import resource_monitoring as routes

    lookup = AsyncMock()
    monkeypatch.setattr(routes.Worker, 'one_by_id', lookup)
    with pytest.raises(InvalidException):
        await routes.get_worker_metrics(
            SimpleNamespace(), SimpleNamespace(), _context(), id
        )
    lookup.assert_not_called()


def test_worker_edit_hint_respects_global_and_member_read_only_access():
    from gpustack.schemas.principals import OrgRole
    from gpustack.routes.resource_monitoring import can_edit_worker

    ctx = _context()
    assert can_edit_worker(ctx, _worker())
    ctx.org_role = OrgRole.MEMBER
    assert not can_edit_worker(ctx, _worker())
    assert not can_edit_worker(_context(), _worker(owner=None))
    assert can_edit_worker(_context(admin=True), _worker(owner=None))


@pytest.mark.asyncio
async def test_dashboard_redirect_does_not_reveal_an_unreadable_cluster(monkeypatch):
    from gpustack.routes import resource_monitoring as routes

    monkeypatch.setattr(routes.Worker, 'one_by_id', AsyncMock(return_value=_worker()))
    monkeypatch.setattr(
        routes.Cluster,
        'one_by_id',
        AsyncMock(
            return_value=SimpleNamespace(name='Private cluster', owner_principal_id=99)
        ),
    )
    monkeypatch.setattr(
        routes,
        'get_global_config',
        lambda: SimpleNamespace(
            get_grafana_url=lambda: 'https://grafana.example',
            grafana_worker_dashboard_uid='workers',
        ),
    )
    with pytest.raises(NotFoundException):
        await routes.get_worker_monitoring_dashboard(
            SimpleNamespace(), SimpleNamespace(), _context(), 7
        )


@pytest.mark.asyncio
async def test_legacy_empty_inventory_fields_allow_current_detail_without_fabricating_identity(
    monkeypatch,
):
    from gpustack.routes import resource_monitoring as routes
    from tests.monitoring.test_hardware import _device

    worker = _worker(devices=[_device(0, '', 0)])
    gpu = SimpleNamespace(
        worker_id=7, index=0, uuid=None, device_chip_index=None, owner_principal_id=17
    )
    monkeypatch.setattr(routes.GPUDevice, 'one_by_id', AsyncMock(return_value=gpu))
    monkeypatch.setattr(routes.Worker, 'one_by_id', AsyncMock(return_value=worker))
    _, devices, selected = await routes._resolve(
        SimpleNamespace(), _context(), 'legacy-view', True
    )
    assert selected == devices[0].public.id
    assert devices[0].identity is None
    assert not devices[0].public.history_available
