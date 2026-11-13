from types import SimpleNamespace
from unittest.mock import Mock
from datetime import datetime, timezone

import pytest

from gpustack.schemas.workers import (
    GPUDeviceStatus,
    GPUCoreInfo,
    MemoryInfo,
    WorkerPublic,
)
from gpustack.worker.exporter import MetricExporter
from gpustack.worker import exporter as exporter_module


def _gpu(index, uuid, chip=0):
    return GPUDeviceStatus(
        index=index,
        device_index=index,
        device_chip_index=chip,
        uuid=uuid,
        name='GPU',
        core=GPUCoreInfo(total=80, utilization_rate=35),
        memory=MemoryInfo(total=1000, used=400),
        temperature=50,
        power_used=100,
    )


def _collect(
    devices, *, trusted=None, registered_uuid='server-worker', server_failure=None
):
    status = SimpleNamespace(
        os=None,
        kernel=None,
        uptime=None,
        cpu=SimpleNamespace(total=8, utilization_rate=20),
        memory=MemoryInfo(total=10000, used=2000),
        gpu_devices=devices,
        filesystem=None,
    )
    server_worker = trusted or SimpleNamespace(
        id=7, me=True, worker_uuid='server-worker', owner_principal_id=17
    )
    response = Mock()
    server_payload = WorkerPublic.model_validate(
        {
            **vars(server_worker),
            'name': 'server-worker',
            'hostname': 'server-host',
            'ip': '10.0.0.1',
            'ifname': 'eth0',
            'port': 10150,
            'status': None,
            'created_at': datetime.now(timezone.utc),
            'updated_at': datetime.now(timezone.utc),
        }
    ).model_dump(mode='json')
    assert 'worker_uuid' not in server_payload
    response.json.return_value = server_payload
    http = SimpleNamespace(get=Mock(return_value=response, side_effect=server_failure))
    client = SimpleNamespace(
        http_client=SimpleNamespace(get_httpx_client=lambda: http),
        model_instances=SimpleNamespace(list=lambda: SimpleNamespace(items=[])),
    )
    exporter = MetricExporter(
        cfg=SimpleNamespace(worker_metrics_port=10151),
        collector=SimpleNamespace(
            timed_collect=lambda **kwargs: SimpleNamespace(
                status=status, worker_uuid='untrusted-worker', owner_principal_id=999
            )
        ),
        worker_name_getter=lambda: 'renameable',
        worker_ip_getter=lambda: '10.0.0.1',
        worker_id_getter=lambda: 7,
        worker_uuid_getter=lambda: registered_uuid,
        clientset_getter=lambda: client,
    )
    exporter.refresh_worker_identity()
    metrics = list(exporter.collect_worker_metrics())
    samples = {m.name: m.samples for m in metrics}
    return samples, client


def _identity_exporter():
    response = Mock()
    response.json.return_value = {'id': 7, 'me': True, 'owner_principal_id': 17}
    http = Mock()
    http.get.return_value = response
    client = SimpleNamespace(http_client=SimpleNamespace(get_httpx_client=lambda: http))
    exporter = MetricExporter(
        cfg=SimpleNamespace(worker_metrics_port=10151),
        collector=None,
        worker_name_getter=lambda: 'worker',
        worker_ip_getter=lambda: '192.0.2.7',
        worker_id_getter=lambda: 7,
        worker_uuid_getter=lambda: 'registered-uuid',
        clientset_getter=lambda: client,
    )
    status = SimpleNamespace(
        cpu=SimpleNamespace(utilization_rate=20),
        memory=MemoryInfo(total=10000, used=2000),
        gpu_devices=[_gpu(0, 'GPU-a')],
        filesystem=[],
    )
    return exporter, http, status


def test_stable_collection_only_reads_a_background_refreshed_snapshot():
    exporter, http, status = _identity_exporter()
    assert list(exporter._collect_stable_hardware(status)) == []
    http.get.assert_not_called()
    exporter.refresh_worker_identity()
    http.get.assert_called_once_with('/workers/7', timeout=2.0)
    http.get.side_effect = AssertionError('Network I/O inside collection')
    for _ in range(3):
        assert list(exporter._collect_stable_hardware(status))
    assert http.get.call_count == 1


def test_expired_or_changed_registration_does_not_publish_cached_owner(
    monkeypatch, caplog
):
    exporter, http, status = _identity_exporter()
    now = [100.0]
    monkeypatch.setattr(exporter_module.time, 'monotonic', lambda: now[0])
    exporter.refresh_worker_identity()
    now[0] += exporter_module.HARDWARE_IDENTITY_TTL_SECONDS
    assert list(exporter._collect_stable_hardware(status)) == []
    assert 'identity became unavailable' in caplog.text
    exporter.refresh_worker_identity()
    exporter._worker_uuid_getter = lambda: 'another-registration'
    assert list(exporter._collect_stable_hardware(status)) == []
    assert http.get.call_count == 2


def test_background_failure_revokes_snapshot_and_logs_one_transition(caplog):
    exporter, http, status = _identity_exporter()
    exporter.refresh_worker_identity()
    http.get.side_effect = TimeoutError('unreachable')
    exporter.refresh_worker_identity()
    assert list(exporter._collect_stable_hardware(status)) == []
    exporter.refresh_worker_identity()
    warnings = [r for r in caplog.records if r.levelname == 'WARNING']
    assert len(warnings) == 1
    http.get.side_effect = None
    http.get.return_value.json.return_value['owner_principal_id'] = 99
    exporter.refresh_worker_identity()
    samples = [s for m in exporter._collect_stable_hardware(status) for s in m.samples]
    assert samples and all(s.labels['owner_principal_id'] == '99' for s in samples)


def test_stable_gpu_identity_uses_server_owner_and_real_chip():
    samples, client = _collect([_gpu(8, 'GPU-unique', chip=2)])
    values = samples.get('gpustack:worker_gpu_utilization_percent', [])
    assert len(values) == 1
    assert values[0].labels == {
        'worker_id': '7',
        'worker_uuid': 'server-worker',
        'owner_principal_id': '17',
        'gpu_uuid': 'GPU-unique',
        'gpu_chip_index': '2',
    }
    assert values[0].value == 35
    client.http_client.get_httpx_client().get.assert_called_once_with(
        '/workers/7', timeout=2.0
    )
    legacy = samples['gpustack:worker_node_gpu_utilization_rate'][0]
    assert legacy.labels['gpu_index'] == '0'
    assert legacy.labels['gpu_chip_index'] == '0'


def test_reordering_devices_preserves_stable_numeric_identity():
    a, b = _gpu(3, 'GPU-a'), _gpu(8, 'GPU-b')
    before, _ = _collect([a, b])
    after, _ = _collect([b, a])
    key = 'gpustack:worker_gpu_memory_used_bytes'
    assert len(before.get(key, [])) == 2
    assert {tuple(sorted(s.labels.items())) for s in before[key]} == {
        tuple(sorted(s.labels.items())) for s in after[key]
    }


def test_stable_info_proves_reported_index_without_changing_numeric_identity():
    samples, _ = _collect([_gpu(8, 'GPU-private', chip=2)])
    info = samples.get('gpustack:worker_gpu_identity', [])
    assert len(info) == 1
    assert info[0].labels['gpu_index'] == '8'
    assert info[0].labels['gpu_chip_index'] == '2'
    assert info[0].labels['owner_principal_id'] == '17'


@pytest.mark.parametrize(
    'devices',
    [
        [_gpu(3, '')],
        [_gpu(3, 'GPU-same'), _gpu(8, 'GPU-same')],
        [_gpu(3, 'GPU-a'), _gpu(3, 'GPU-b')],
        [_gpu(None, 'GPU-a')],
    ],
)
def test_ambiguous_devices_do_not_create_stable_history(devices):
    samples, _ = _collect(devices)
    assert not samples.get('gpustack:worker_gpu_memory_used_bytes', [])
    assert samples.get('gpustack:worker_host_memory_used_bytes', [])


def test_partition_does_not_export_parent_power_temperature_or_utilization():
    samples, _ = _collect([_gpu(4, 'MIG-partition')])
    assert samples.get('gpustack:worker_gpu_memory_used_bytes', [])
    for name in ('power_watts', 'temperature_celsius', 'utilization_percent'):
        assert not samples.get('gpustack:worker_gpu_' + name, [])


@pytest.mark.parametrize(
    'worker',
    [
        SimpleNamespace(id=8, me=True, worker_uuid='w', owner_principal_id=17),
        SimpleNamespace(id=7, me=False, worker_uuid='w', owner_principal_id=17),
    ],
)
def test_unverified_server_identity_keeps_only_legacy_metrics(worker):
    samples, _ = _collect([_gpu(3, 'GPU-a')], trusted=worker)
    assert not samples.get('gpustack:worker_host_memory_used_bytes', [])
    assert samples['gpustack:worker_node_gram_used_bytes']


def test_unowned_server_snapshot_is_empty_without_inventing_an_owner():
    samples, _ = _collect(
        [_gpu(3, 'GPU-a')],
        trusted=SimpleNamespace(
            id=7, me=True, worker_uuid='server-worker', owner_principal_id=None
        ),
    )
    values = samples.get('gpustack:worker_host_cpu_utilization_percent', [])
    assert len(values) == 1
    assert values[0].labels['owner_principal_id'] == ''


def test_nonfinite_sensor_values_do_not_enter_stable_families():
    device = _gpu(8, 'GPU-a')
    device.core.utilization_rate = float('nan')
    device.power_used = float('inf')
    samples, _ = _collect([device])
    assert not samples.get('gpustack:worker_gpu_utilization_percent')
    assert not samples.get('gpustack:worker_gpu_power_watts')
    assert samples['gpustack:worker_gpu_memory_used_bytes']


@pytest.mark.parametrize('registered_uuid', ['', None])
def test_missing_registered_uuid_does_not_use_collector_report(registered_uuid):
    samples, _ = _collect([_gpu(3, 'GPU-a')], registered_uuid=registered_uuid)
    assert not samples.get('gpustack:worker_host_memory_used_bytes', [])
    assert samples['gpustack:worker_node_gram_used_bytes']


def test_registration_uuid_survives_public_serialization_via_worker_getter(tmp_path):
    from gpustack.schemas.workers import WorkerRegistrationPublic
    from gpustack.utils.uuid import set_legacy_uuid
    from gpustack.worker.worker import Worker as WorkerProcess

    registered = WorkerRegistrationPublic.model_validate(
        {
            'id': 7,
            'name': 'w',
            'hostname': 'h',
            'ip': '10.0.0.1',
            'ifname': 'eth0',
            'port': 10150,
            'status': None,
            'created_at': datetime.now(timezone.utc),
            'updated_at': datetime.now(timezone.utc),
            'token': 'test-token',
            'worker_uuid': 'registration-uuid',
        }
    )
    assert registered.model_dump(mode='json')['worker_uuid'] == 'registration-uuid'
    set_legacy_uuid(str(tmp_path), registered.worker_uuid)
    process = WorkerProcess.__new__(WorkerProcess)
    process._worker_uuid = None
    process._config = SimpleNamespace(data_dir=str(tmp_path))
    samples, _ = _collect([_gpu(8, 'GPU-a')], registered_uuid=process.worker_uuid())
    assert (
        samples['gpustack:worker_gpu_memory_used_bytes'][0].labels['worker_uuid']
        == 'registration-uuid'
    )


def test_owner_snapshot_failure_does_not_use_stale_or_self_reported_identity():
    samples, client = _collect([_gpu(8, 'GPU-a')], server_failure=TimeoutError())
    assert not samples.get('gpustack:worker_host_memory_used_bytes', [])
    assert not samples.get('gpustack:worker_gpu_identity', [])
    assert samples['gpustack:worker_node_gram_used_bytes']
    assert client.http_client.get_httpx_client().get.call_args.kwargs['timeout'] == 2.0
