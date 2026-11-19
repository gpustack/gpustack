"""A Benchmark's saved monitoring data and the state required to capture it."""

from typing import Any, Dict, Optional

from sqlalchemy import BigInteger, Column, Double, Index, JSON, LargeBinary
from sqlalchemy.dialects.mysql import LONGBLOB
from sqlalchemy.orm import declared_attr, deferred
from sqlmodel import Field, SQLModel


class BenchmarkMonitoringPayload(SQLModel, table=True):
    __tablename__ = 'benchmark_monitoring_payloads'
    __table_args__ = (
        Index('ix_benchmark_monitoring_payloads_due', 'sealed', 'retry_at'),
        Index('ix_benchmark_monitoring_payloads_expiry', 'expires_at'),
    )

    # Deletion updates the shared quota, so the business row cannot cascade.
    benchmark_id: int = Field(
        primary_key=True, sa_column_kwargs={'autoincrement': False}
    )
    run_id: str = Field(max_length=64)
    owner_principal_id: Optional[int] = None
    snapshot: Dict[str, Any] = Field(default_factory=dict, sa_type=JSON)
    plan: Dict[str, Any] = Field(default_factory=dict, sa_type=JSON)
    snapshot_seq: int = Field(default=0, sa_type=BigInteger)
    state: str = Field(default='pending', max_length=32)
    sealed: bool = False
    capture_complete: bool = False
    data_coverage: str = Field(default='none', max_length=16)
    hardware_visible: bool = False
    captured_until: Optional[float] = Field(default=None, sa_type=Double)
    expires_at: Optional[float] = Field(default=None, sa_type=Double)
    format_version: int = 1
    payload: Optional[bytes] = Field(
        default=None, sa_column=Column(LargeBinary().with_variant(LONGBLOB(), 'mysql'))
    )
    payload_sha256: Optional[str] = Field(default=None, max_length=64)
    compressed_bytes: int = Field(default=0, sa_type=BigInteger)
    uncompressed_bytes: int = Field(default=0, sa_type=BigInteger)
    lease_owner: Optional[str] = Field(default=None, max_length=36)
    lease_until: float = Field(default=0, sa_type=Double)
    fencing_token: int = Field(default=0, sa_type=BigInteger)
    write_seq: int = Field(default=0, sa_type=BigInteger)
    finalize_requested_at: Optional[float] = Field(default=None, sa_type=Double)
    repair_requested: bool = False
    retry_at: float = Field(default=0, sa_type=Double)
    last_error: Optional[str] = Field(default=None, max_length=128)

    @declared_attr.directive
    def __mapper_args__(cls):
        # The status endpoint and heartbeat can read metadata without the BLOB.
        return {
            'properties': {'payload': deferred(cls.__table__.c.payload, raiseload=True)}
        }
