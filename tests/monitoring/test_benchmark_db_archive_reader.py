"""Read one saved package with its authorization and publication metadata."""

from types import SimpleNamespace

import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.exceptions import ConflictException, ForbiddenException
from gpustack.monitoring import archive
from gpustack.monitoring.benchmark_payload import encode_payload
from gpustack.schemas.benchmark_monitoring_payload import BenchmarkMonitoringPayload
from tests.monitoring.test_benchmarks import _ctx, _history


@pytest.mark.asyncio
async def test_replay_uses_published_permission_and_size_after_source_changes():
    history = _history()
    context = history.context.model_dump(mode='json')
    context['instances'] = [
        dict(id=id, name=f'instance-{id}', worker_id=3, worker_ids=[3], gpu_indexes=[0])
        for id in (12, 13)
    ]
    package = encode_payload({'context': context, 'scopes': {}})
    benchmark = SimpleNamespace(
        id=7, run_id='replay', owner_principal_id=4, cluster_id=1
    )
    engine = create_async_engine('sqlite+aiosqlite:///:memory:')
    try:
        async with engine.begin() as conn:
            await conn.run_sync(BenchmarkMonitoringPayload.__table__.create)
        async with AsyncSession(engine, expire_on_commit=False) as session:
            session.add(
                BenchmarkMonitoringPayload(
                    benchmark_id=7,
                    run_id='replay',
                    owner_principal_id=4,
                    state='ready',
                    sealed=True,
                    write_seq=3,
                    payload=package.data,
                    payload_sha256=package.sha256,
                    compressed_bytes=package.compressed_bytes,
                    uncompressed_bytes=package.uncompressed_bytes,
                    hardware_visible=False,
                    snapshot=history.snapshot.model_dump(mode='json'),
                )
            )
            await session.commit()
        async with AsyncSession(engine) as session:
            response = await archive.read_archive_context(session, _ctx(), benchmark)
        assert response.observation_storage.content_version == '3'
        assert response.observation_storage.compressed_bytes is None
        assert response.resource_status == 'restricted'
        assert response.workers == [] and response.devices == []
        assert all(
            m.worker_id is None and m.worker_ids == [] and m.gpu_indexes == []
            for m in response.instances
        )
        async with AsyncSession(engine) as session:
            with pytest.raises(ForbiddenException):
                await archive.read_archive_resources(session, _ctx(), benchmark)
            with pytest.raises(ConflictException) as err:
                await archive.read_archive_metrics(
                    session, _ctx(), benchmark, content_version='2'
                )
            assert err.value.details['code'] == 'benchmark_archive_content_changed'
        async with AsyncSession(engine) as session:
            current = await session.get(BenchmarkMonitoringPayload, 7)
            current.hardware_visible = True
            await session.commit()
            response = await archive.read_archive_context(session, _ctx(), benchmark)
            assert response.workers[0].name == 'secret-worker'
            assert (
                response.observation_storage.compressed_bytes
                == package.compressed_bytes
            )
            phase = await archive.read_archive_phase_observations(
                session, _ctx(), benchmark, phase_ids=['128:0']
            )
            assert phase['content_version'] == '3'
            assert (
                phase['phases']['128:0']['runtime'].reason
                == 'archive_scope_unavailable'
            )
    finally:
        await engine.dispose()


@pytest.mark.parametrize(
    'prometheus_url,source_interval,request_step,expected',
    [
        (None, None, 5, 5),
        (None, 1, 1, 5),
        ('https://prometheus.example', 30, 5, 30),
        ('https://prometheus.example', 10, 30, 30),
    ],
)
def test_capture_enrollment_respects_source_resolution(
    monkeypatch, prometheus_url, source_interval, request_step, expected
):
    from unittest.mock import Mock

    from gpustack.schemas.benchmark import Benchmark, BenchmarkSnapshot

    monkeypatch.setattr(
        archive,
        'get_global_config',
        lambda: SimpleNamespace(
            prometheus_url=prometheus_url,
            benchmark_archive_sample_step_seconds=request_step,
            benchmark_archive_source_interval_seconds=source_interval,
        ),
    )
    benchmark = Benchmark(
        id=7,
        name='run',
        model_id=9,
        run_id='new-run',
        owner_principal_id=4,
        snapshot=BenchmarkSnapshot(monitoring=_history().snapshot),
    )
    session = Mock()
    archive.enroll_archive(session, benchmark)
    row = session.add.call_args.args[0]
    assert row.benchmark_id == 7 and row.run_id == 'new-run'
    assert row.plan['step'] == expected
    assert row.plan['rate_window'] == max(75, 4 * (source_interval or 5))
    assert row.state == 'pending'


@pytest.mark.asyncio
async def test_worker_updates_cannot_backfill_preexisting_benchmark(monkeypatch):
    from unittest.mock import AsyncMock, Mock

    session = Mock()
    lock = AsyncMock(return_value=None)
    monkeypatch.setattr(archive, 'locked_payload', lock)
    await archive.update_archive_snapshot(session, SimpleNamespace(id=7, run_id='old'))
    lock.assert_awaited_once_with(session, 7)
    session.add.assert_not_called()
