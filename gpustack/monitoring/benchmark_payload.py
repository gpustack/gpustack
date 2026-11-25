"""Bounded monitoring bundles and database transactions for Benchmark archiving."""

from dataclasses import dataclass
import gzip
import hashlib
from io import BytesIO
import json
import re
import zlib

from sqlalchemy import text, update
from sqlmodel import select

from gpustack.artifacts import capacity as capacity_policy
from gpustack.schemas.artifact_storage import MAX_CAPACITY_BYTES
from gpustack.schemas.benchmark_monitoring_payload import BenchmarkMonitoringPayload
from gpustack.server.db import async_session

DEFAULT_MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
ARCHIVE_FORMAT_VERSION = 1
_TERMINAL_RETRY_AT = 32503680000.0  # Year 3000 sentinel for disabled retries (MySQL DOUBLE compatible)


class BenchmarkPayloadError(RuntimeError):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class EncodedPayload:
    data: bytes
    sha256: str
    uncompressed_bytes: int

    @property
    def compressed_bytes(self) -> int:
        return len(self.data)


def encode_payload(
    value: dict, *, max_bytes: int = DEFAULT_MAX_ARCHIVE_BYTES
) -> EncodedPayload:
    """Encode one JSON object using gzip and enforce both size limits."""
    if not isinstance(value, dict):
        raise BenchmarkPayloadError('archive_payload_corrupt')
    output = BytesIO()
    size = 0
    encoder = json.JSONEncoder(
        ensure_ascii=False, allow_nan=False, separators=(',', ':')
    )
    try:
        with gzip.GzipFile(fileobj=output, mode='wb', mtime=0) as compressed:
            for token in encoder.iterencode(value):
                chunk = token.encode('utf-8')
                size += len(chunk)
                if size > max_bytes:
                    raise BenchmarkPayloadError('archive_payload_too_large')
                compressed.write(chunk)
    except (ValueError, TypeError, RecursionError) as exc:
        raise BenchmarkPayloadError('archive_payload_corrupt') from exc
    data = output.getvalue()
    if len(data) > max_bytes:
        raise BenchmarkPayloadError('archive_payload_too_large')
    return EncodedPayload(data, hashlib.sha256(data).hexdigest(), size)


def _reject_constant(value):
    raise ValueError('Invalid JSON constant')


def decode_payload(
    data: bytes,
    *,
    sha256: str,
    uncompressed_bytes: int,
    max_bytes: int = DEFAULT_MAX_ARCHIVE_BYTES,
) -> dict:
    """Check stored bytes, then inflate at most the declared size plus one byte."""
    if len(data) > max_bytes or uncompressed_bytes > max_bytes:
        raise BenchmarkPayloadError('archive_payload_too_large')
    if uncompressed_bytes <= 0 or hashlib.sha256(data).hexdigest() != sha256:
        raise BenchmarkPayloadError('archive_payload_corrupt')
    try:
        inflater = zlib.decompressobj(16 + zlib.MAX_WBITS)
        raw = inflater.decompress(data, uncompressed_bytes + 1)
        if not inflater.eof or inflater.unused_data or len(raw) != uncompressed_bytes:
            raise BenchmarkPayloadError('archive_payload_corrupt')
        result = json.loads(raw, parse_constant=_reject_constant)
    except (zlib.error, ValueError, RecursionError) as exc:
        raise BenchmarkPayloadError('archive_payload_corrupt') from exc
    if not isinstance(result, dict):
        raise BenchmarkPayloadError('archive_payload_corrupt')
    return result


@dataclass(frozen=True)
class BenchmarkArchiveLease:
    benchmark_id: int
    run_id: str
    owner: str
    token: int
    write_seq: int
    snapshot_seq: int
    repair_requested: bool
    finalize_requested_at: float | None


async def database_now(session) -> float:
    """Read the database wall clock, including time spent waiting for locks."""
    dialect = session.get_bind().dialect.name
    if dialect == 'postgresql':
        query = "EXTRACT(EPOCH FROM clock_timestamp())"
    elif dialect in ('mysql', 'mariadb'):
        query = 'UNIX_TIMESTAMP(CURRENT_TIMESTAMP(6))'
    elif dialect == 'sqlite':
        query = "(julianday('now') - 2440587.5) * 86400"
    else:
        raise RuntimeError('Unsupported database for Benchmark archive leases')
    return float((await session.exec(select(text(query)))).one())


async def locked_payload(session, benchmark_id: int):
    """Lock a row and reload its metadata without retrieving compressed bytes."""
    if session.get_bind().dialect.name == 'sqlite':
        # SQLite serializes writers at the database level rather than locking rows.
        await session.exec(
            update(BenchmarkMonitoringPayload)
            .where(BenchmarkMonitoringPayload.benchmark_id == benchmark_id)
            .values(benchmark_id=BenchmarkMonitoringPayload.benchmark_id)
            .execution_options(synchronize_session=False)
        )
    result = await session.exec(
        select(BenchmarkMonitoringPayload)
        .where(BenchmarkMonitoringPayload.benchmark_id == benchmark_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    return result.one_or_none()


def _available(record, now):
    return (
        record is not None
        and record.state != 'expired'
        and (record.expires_at is None or record.expires_at > now)
    )


def _same_lease(record, lease, now):
    return (
        _available(record, now)
        and record.run_id == lease.run_id
        and record.lease_owner == lease.owner
        and record.fencing_token == lease.token
        and record.lease_until > now
    )


def _same_capture(record, lease):
    return (
        record.snapshot_seq == lease.snapshot_seq
        and record.write_seq == lease.write_seq
        and record.repair_requested == lease.repair_requested
        and record.finalize_requested_at == lease.finalize_requested_at
    )


def _publish_allowed(record, lease, captured_until, sealed):
    terminal = lease.finalize_requested_at is not None or lease.repair_requested
    # The terminal measurement can end before the provisional live window.
    first_final = (
        sealed and not record.sealed and lease.finalize_requested_at is not None
    )
    return (
        _same_capture(record, lease)
        and sealed == terminal
        and (not record.sealed or lease.repair_requested)
        and (
            first_final
            or record.captured_until is None
            or captured_until is not None
            and captured_until >= record.captured_until
        )
    )


def _change_capacity(capacity, delta):
    usage = capacity.used_bytes + delta
    if usage < 0:
        raise BenchmarkPayloadError('storage_capacity_accounting_mismatch')
    if delta > 0 and (
        usage > MAX_CAPACITY_BYTES or capacity.max_bytes and usage > capacity.max_bytes
    ):
        raise BenchmarkPayloadError('storage_capacity_exceeded')
    capacity.used_bytes = usage


def _hardware_visible(record):
    snapshot = record.snapshot.get('monitoring') or {}
    owner = snapshot.get('owner_principal_id')
    return (
        owner is not None
        and owner == record.owner_principal_id
        and all(
            worker.get('owner_principal_id') == owner
            for worker in snapshot.get('workers', [])
        )
    )


class BenchmarkPayloadStore:
    """Serialize each capture and account for the actual stored gzip bytes."""

    def __init__(self, config=None, session_factory=None):
        self.sessions = session_factory or async_session
        self.max_bytes = getattr(
            config, 'benchmark_archive_max_run_bytes', DEFAULT_MAX_ARCHIVE_BYTES
        )

    async def claim(
        self, benchmark_id: int, worker_id: str, *, ttl: float = 30
    ) -> BenchmarkArchiveLease | None:
        """Grant a new identity to the next writer of a due task."""
        async with self.sessions() as session, session.begin():
            record = await locked_payload(session, benchmark_id)
            now = await database_now(session)
            if (
                not _available(record, now)
                or record.lease_until > now
                or record.retry_at > now
                or record.sealed
                and not record.repair_requested
            ):
                return None
            record.lease_owner = worker_id
            record.lease_until = now + ttl
            record.fencing_token += 1
            return BenchmarkArchiveLease(
                benchmark_id,
                record.run_id,
                worker_id,
                record.fencing_token,
                record.write_seq,
                record.snapshot_seq,
                record.repair_requested,
                record.finalize_requested_at,
            )

    async def renew(self, lease, *, ttl: float = 30) -> bool:
        """Extend a live lease. A late heartbeat cannot revive it."""
        async with self.sessions() as session, session.begin():
            record = await locked_payload(session, lease.benchmark_id)
            now = await database_now(session)
            if not _same_lease(record, lease, now):
                return False
            record.lease_until = now + ttl
            return True

    async def release(self, lease, *, retry_seconds: float = 0) -> bool:
        async with self.sessions() as session, session.begin():
            record = await locked_payload(session, lease.benchmark_id)
            now = await database_now(session)
            if not _same_lease(record, lease, now):
                return False
            record.lease_owner, record.lease_until = None, 0
            record.retry_at = now + retry_seconds
            return True

    async def fail(
        self, lease, reason: str, *, retry_seconds: float | None = 60
    ) -> bool:
        """Record a writer failure without replacing data or unsealing an archive."""
        async with self.sessions() as session, session.begin():
            record = await locked_payload(session, lease.benchmark_id)
            now = await database_now(session)
            if not _same_lease(record, lease, now) or not _same_capture(record, lease):
                return False
            code = (
                reason if re.fullmatch('[a-z0-9_]{1,128}', reason) else 'capture_failed'
            )
            record.last_error = code
            record.lease_owner, record.lease_until = None, 0
            if retry_seconds is None:
                record.retry_at = _TERMINAL_RETRY_AT
                if lease.finalize_requested_at is not None:
                    record.sealed = True
            else:
                record.retry_at = now + retry_seconds
            return True

    async def publish(
        self,
        lease,
        encoded: EncodedPayload,
        *,
        captured_until: float | None,
        state: str,
        data_coverage: str,
        capture_complete: bool,
        sealed: bool = False,
        expires_at: float | None = None,
        next_capture_seconds: float = 0,
    ) -> bool:
        """Replace the bundle and charge its size difference in the same transaction."""
        if max(encoded.compressed_bytes, encoded.uncompressed_bytes) > self.max_bytes:
            raise BenchmarkPayloadError('archive_payload_too_large')
        async with self.sessions() as session, session.begin():
            capacity = await capacity_policy.lock_capacity(session)
            record = await locked_payload(session, lease.benchmark_id)
            now = await database_now(session)
            if not _same_lease(record, lease, now) or not _publish_allowed(
                record, lease, captured_until, sealed
            ):
                return False
            _change_capacity(
                capacity, encoded.compressed_bytes - record.compressed_bytes
            )
            record.payload = encoded.data
            record.format_version = ARCHIVE_FORMAT_VERSION
            record.payload_sha256 = encoded.sha256
            record.compressed_bytes = encoded.compressed_bytes
            record.uncompressed_bytes = encoded.uncompressed_bytes
            record.captured_until = captured_until
            record.state = state
            record.data_coverage = data_coverage
            record.capture_complete = capture_complete
            record.hardware_visible = _hardware_visible(record)
            record.sealed = sealed
            if expires_at is not None:
                record.expires_at = expires_at
            record.repair_requested = False
            record.last_error = None
            record.write_seq += 1
            # First finalization leaves an immediate repair available. After a
            # repair, space further captures by the same interval as live work.
            record.retry_at = (
                now + next_capture_seconds
                if not sealed or lease.repair_requested
                else 0
            )
            record.lease_owner, record.lease_until = None, 0
            return True

    async def request_repair(self, session, benchmark_id: int, run_id: str) -> bool:
        """Schedule a repair in the caller's transaction, keeping its old data live."""
        record = await locked_payload(session, benchmark_id)
        now = await database_now(session)
        if not _available(record, now) or record.run_id != run_id:
            return False
        if record.last_error == 'archive_scope_budget_exceeded':
            return False
        if record.repair_requested:
            return True
        if not record.sealed and not record.last_error:
            return False
        record.repair_requested = record.sealed
        if not record.sealed:
            record.retry_at = 0
        record.last_error = None
        record.lease_owner, record.lease_until = None, 0
        record.fencing_token += 1
        return True

    async def delete(self, session, benchmark_id: int) -> bool:
        """Remove the archive and its quota charge in the caller's transaction."""
        capacity = await capacity_policy.lock_capacity(session)
        record = await locked_payload(session, benchmark_id)
        if record is None:
            return False
        _change_capacity(capacity, -record.compressed_bytes)
        await session.delete(record)
        return True

    async def expire(self, benchmark_id: int) -> bool:
        """Release the bytes of a due record while keeping its expired status."""
        async with self.sessions() as session, session.begin():
            capacity = await capacity_policy.lock_capacity(session)
            record = await locked_payload(session, benchmark_id)
            now = await database_now(session)
            if (
                record is None
                or record.state == 'expired'
                or record.expires_at is None
                or record.expires_at > now
            ):
                return False
            _change_capacity(capacity, -record.compressed_bytes)
            record.payload = record.payload_sha256 = None
            record.compressed_bytes = record.uncompressed_bytes = 0
            record.state, record.data_coverage = 'expired', 'none'
            record.sealed, record.capture_complete = True, False
            record.hardware_visible = False
            record.captured_until, record.last_error = None, None
            record.snapshot, record.plan = {}, {}
            record.repair_requested = False
            record.lease_owner, record.lease_until = None, 0
            record.fencing_token += 1
            record.write_seq += 1
            return True
