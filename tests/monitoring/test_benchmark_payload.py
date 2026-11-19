"""A stale or rejected writer cannot change saved observations or quota."""

import gzip
from hashlib import sha256

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import undefer
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.monitoring import benchmark_payload as storage
from gpustack.schemas.artifact_storage import ArtifactStorageCapacity as Capacity
from gpustack.schemas.benchmark_monitoring_payload import (
    BenchmarkMonitoringPayload as Payload,
)


def test_codec_checks_bytes_and_preserves_null_and_zero():
    values = {'metrics': [None, 0, 42], 'context': {'name': '模型'}}
    bundle = storage.encode_payload(values)
    assert bundle.sha256 == sha256(bundle.data).hexdigest()
    assert (
        storage.decode_payload(
            bundle.data,
            sha256=bundle.sha256,
            uncompressed_bytes=bundle.uncompressed_bytes,
        )
        == values
    )
    with pytest.raises(storage.BenchmarkPayloadError, match='archive_payload_corrupt'):
        storage.decode_payload(
            bundle.data, sha256='0' * 64, uncompressed_bytes=bundle.uncompressed_bytes
        )


def test_decode_checks_length_before_unbounded_decompression():
    data = gzip.compress(b' ' * 4096 + b'{}')
    with pytest.raises(storage.BenchmarkPayloadError, match='archive_payload_corrupt'):
        storage.decode_payload(
            data, sha256=sha256(data).hexdigest(), uncompressed_bytes=32, max_bytes=100
        )
    with pytest.raises(
        storage.BenchmarkPayloadError, match='archive_payload_too_large'
    ):
        storage.encode_payload({'data': 'x' * 512}, max_bytes=100)


@pytest_asyncio.fixture
async def database(monkeypatch):
    engine = create_async_engine('sqlite+aiosqlite:///:memory:')
    async with engine.begin() as connection:
        await connection.run_sync(Payload.__table__.create)
        await connection.run_sync(Capacity.__table__.create)

    def sessions():
        return AsyncSession(engine, expire_on_commit=False)

    clock = [100.0]

    async def now(session):
        return clock[0]

    monkeypatch.setattr(storage, 'database_now', now)
    async with sessions() as session:
        session.add(Capacity(id=1, initialized=True, max_bytes=0))
        session.add(Payload(benchmark_id=1, run_id='run', owner_principal_id=3))
        await session.commit()
    yield storage.BenchmarkPayloadStore(session_factory=sessions), sessions, clock
    await engine.dispose()


async def save(store, lease, *, data=None, **kwargs):
    return await store.publish(
        lease,
        storage.encode_payload(data or {'value': 1}),
        captured_until=kwargs.pop('captured_until', 20),
        state='ready' if kwargs.get('sealed') else 'capturing',
        data_coverage='complete',
        capture_complete=kwargs.get('sealed', False),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_expired_holder_cannot_publish_before_or_after_takeover(database):
    store, sessions, clock = database
    first = await store.claim(1, 'first', ttl=10)
    assert await store.claim(1, 'second') is None
    clock[0] += 10
    assert not await store.renew(first)
    assert not await save(store, first)
    second = await store.claim(1, 'second')
    assert second.token > first.token
    assert not await store.fail(first, 'old_error')
    assert not await save(store, first)
    assert await save(store, second)
    async with sessions() as session:
        row = await session.get(Payload, 1)
        capacity = await session.get(Capacity, 1)
        assert row.captured_until == 20
        assert row.last_error is None
        assert capacity.used_bytes == row.compressed_bytes
        assert 'payload' not in row.__dict__


@pytest.mark.asyncio
async def test_publish_quota_failure_keeps_previous_content_and_counters(database):
    store, sessions, _ = database
    assert await save(store, await store.claim(1, 'first'))
    async with sessions() as session:
        row = await session.get(Payload, 1)
        capacity = await session.get(Capacity, 1)
        original_hash = row.payload_sha256
        original_size = capacity.used_bytes
        capacity.max_bytes = original_size
        await session.commit()
    next_lease = await store.claim(1, 'second')
    assert not await save(store, next_lease, captured_until=19)
    with pytest.raises(
        storage.BenchmarkPayloadError, match='storage_capacity_exceeded'
    ):
        await save(store, next_lease, data={'values': list(range(256))})
    async with sessions() as session:
        row = (
            await session.exec(select(Payload).options(undefer(Payload.payload)))
        ).one()
        assert row.payload_sha256 == original_hash
        assert (
            row.compressed_bytes
            == original_size
            == (await session.get(Capacity, 1)).used_bytes
        )
    assert await save(store, next_lease, data={'value': 1})
    async with sessions() as session:
        assert await store.delete(session, 1)
        await session.commit()
    async with sessions() as session:
        assert await session.get(Payload, 1) is None
        assert (await session.get(Capacity, 1)).used_bytes == 0
    assert not await save(store, next_lease)


@pytest.mark.asyncio
async def test_finalize_revokes_old_snapshot_and_repair_failure_keeps_sealed_bundle(
    database,
):
    store, sessions, clock = database
    old = await store.claim(1, 'first')
    async with sessions() as session:
        row = await storage.locked_payload(session, 1)
        row.snapshot_seq += 1
        row.finalize_requested_at = 100
        row.lease_owner, row.lease_until = None, 0
        await session.commit()
    assert not await save(store, old)
    final = await store.claim(1, 'second')
    assert not await save(store, final)
    assert await save(store, final, sealed=True)
    assert await store.claim(1, 'first') is None
    async with sessions() as session:
        row = await session.get(Payload, 1)
        saved_hash = row.payload_sha256
        assert await store.request_repair(session, 1, 'run')
        await session.commit()
    repair = await store.claim(1, 'first')
    assert repair.repair_requested
    assert await store.fail(repair, 'query_error')
    async with sessions() as session:
        row = await session.get(Payload, 1)
        assert row.sealed and row.state == 'ready'
        assert row.payload_sha256 == saved_hash
        assert row.last_error == 'query_error'
        assert await store.request_repair(session, 1, 'run')
        await session.commit()
    assert await store.claim(1, 'second') is None
    clock[0] += 60
    assert (await store.claim(1, 'second')).repair_requested


@pytest.mark.asyncio
async def test_repeated_sealed_repair_waits_for_capture_interval(database):
    store, sessions, clock = database
    interval = 60
    async with sessions() as session:
        row = await storage.locked_payload(session, 1)
        row.finalize_requested_at = clock[0]
        await session.commit()
    final = await store.claim(1, 'archiver')
    assert await save(store, final, sealed=True, next_capture_seconds=interval)

    async def request():
        async with sessions() as session:
            assert await store.request_repair(session, 1, 'run')
            await session.commit()

    await request()
    first = await store.claim(1, 'archiver')
    assert first.repair_requested
    assert await store.publish(
        first,
        storage.encode_payload({'value': 2}),
        captured_until=20,
        state='partial',
        data_coverage='partial',
        capture_complete=False,
        sealed=True,
        next_capture_seconds=interval,
    )

    await request()
    clock[0] += interval - 1
    assert await store.claim(1, 'archiver') is None
    await request()
    clock[0] += 1
    assert (await store.claim(1, 'archiver')).repair_requested


@pytest.mark.asyncio
async def test_capture_permissions_change_only_when_package_is_published(database):
    store, sessions, _ = database

    async def set_workers(owners):
        async with sessions() as session:
            row = await storage.locked_payload(session, 1)
            row.snapshot = {
                'monitoring': {
                    'owner_principal_id': 3,
                    'workers': [{'owner_principal_id': owner} for owner in owners],
                }
            }
            row.snapshot_seq += 1
            await session.commit()
            return row.hardware_visible

    assert not await set_workers([3, 4])
    assert await save(store, await store.claim(1, 'first'))
    assert not await set_workers([3, 3])
    assert await save(store, await store.claim(1, 'second'))
    assert await set_workers([3, 4])


@pytest.mark.asyncio
async def test_expiry_releases_only_committed_bytes_and_fences_repairs(database):
    store, sessions, clock = database
    async with sessions() as session:
        row = await storage.locked_payload(session, 1)
        row.finalize_requested_at = clock[0]
        await session.commit()
    final = await store.claim(1, 'first')
    assert await save(store, final, sealed=True, expires_at=clock[0] + 10)
    async with sessions() as session:
        assert await store.request_repair(session, 1, 'run')
        await session.commit()
    repair = await store.claim(1, 'second')
    assert not await store.expire(1)
    clock[0] += 10
    assert not await save(store, repair, sealed=True)
    assert await store.expire(1)
    assert not await store.expire(1)
    assert await store.claim(1, 'third') is None
    async with sessions() as session:
        row = await session.get(Payload, 1)
        assert row.state == 'expired'
        assert row.payload_sha256 is None and row.compressed_bytes == 0
        assert (await session.get(Capacity, 1)).used_bytes == 0
