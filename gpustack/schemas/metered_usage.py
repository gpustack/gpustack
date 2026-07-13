"""Unified hourly rollup of time-based resource consumption.

``metered_usage`` is the single aggregate table behind the unified resource
metering framework. Every time-based resource (GPU instance, CPU instance,
persistent volume — and future ones) accumulates here as

    one row per (meter_key, resource_id, bucket_start)

where ``bucket_start`` is a UTC hour bucket and ``quantity`` is stored in the
meter's canonical integer unit. Coarser granularities (day/week/month) are
derived at query time via ``date_trunc``. Adding a new time-based resource
means registering a ``MeterDef`` + writing a collector — no schema change.

Design notes
------------
* **Tokens are NOT here.** LLM token usage stays in ``model_usages`` (a daily
  rollup); the summary layer unions the two sources.
* **Quantities only, no cost.** This is a pure metering table — it stores
  ``quantity`` in the meter's canonical unit and never any price/cost.
* **GPU per-card / CPU whole-machine.** Instances meter on the single
  ``instance.uptime`` meter (wall-clock metered seconds). GPU rows key on
  ``sku = gpu_type`` and yield GPU-Hours (``quantity * sku_count / 3600`` where
  ``sku_count`` = card count); CPU rows key on ``sku = cpu flavor`` and yield
  Instance-Hours (``quantity / 3600``, ``sku_count`` = 1). CPU / memory are not
  metered separately.
* **Natural key = resource identity.** The unique constraint is
  ``(meter_key, resource_id, bucket_start)`` — all non-null — so the
  collector's idempotent upsert works even for resources with no owner
  (created directly via K8s, ``owner_principal_id`` NULL).
* **Hot/cold archival.** Rows older than retention move to
  ``metered_usage_archive`` (see server.usage archiver); the dashboard reads
  only the hot table, audit reads the archive directly via DB.

Query-side conversions
----------------------
    instance_hours = SUM(quantity) / 3600                  (meter=instance.uptime)
    gpu_hours      = SUM(quantity * sku_count) / 3600       (meter=instance.uptime, GPU only)
    gb_days        = SUM(quantity) / 1024 / 86400           (meter=storage.capacity)
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, ClassVar, Dict, Optional, Tuple

from pydantic import ConfigDict
from sqlalchemy import (
    BigInteger,
    Column,
    Integer,
    String,
    UniqueConstraint,
    update,
)
from sqlmodel import Field, SQLModel

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import JSON, UTCDateTime


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# resource_type — the category of metered object.
RESOURCE_TYPE_GPU_INSTANCE = "gpu_instance"
RESOURCE_TYPE_CPU_INSTANCE = "cpu_instance"
RESOURCE_TYPE_PERSISTENT_VOLUME = "persistent_volume"

# meter_key — what is being measured.
METER_INSTANCE_UPTIME = "instance.uptime"
METER_STORAGE_CAPACITY = "storage.capacity"

# unit — canonical integer unit of ``quantity``.
UNIT_SECONDS = "seconds"
UNIT_MIB_SECONDS = "mib_seconds"


# ---------------------------------------------------------------------------
# Meter registry (code constants, not a DB table)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MeterDef:
    """Definition of a meter: its canonical unit + which resource types emit it.

    Adding a new time-based resource = register a ``MeterDef`` + write a
    collector; no schema change.
    """

    key: str
    resource_types: Tuple[str, ...]
    unit: str


METERS: Dict[str, MeterDef] = {
    METER_INSTANCE_UPTIME: MeterDef(
        METER_INSTANCE_UPTIME,
        (RESOURCE_TYPE_GPU_INSTANCE, RESOURCE_TYPE_CPU_INSTANCE),
        UNIT_SECONDS,
    ),
    METER_STORAGE_CAPACITY: MeterDef(
        METER_STORAGE_CAPACITY,
        (RESOURCE_TYPE_PERSISTENT_VOLUME,),
        UNIT_MIB_SECONDS,
    ),
}


# ---------------------------------------------------------------------------
# Hot table
# ---------------------------------------------------------------------------


class MeteredUsage(SQLModel, BaseModelMixin, table=True):
    """Hourly rollup row for one (resource, meter, hour)."""

    __tablename__: ClassVar[str] = "metered_usage"
    __table_args__ = (
        # Natural key = resource identity. All three columns are non-null, so
        # the collector's upsert is robust even when owner_principal_id / sku
        # are NULL.
        UniqueConstraint(
            "meter_key",
            "resource_id",
            "bucket_start",
            name="uq_metered_usage",
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)

    # —— Tenant subjects + creator (real columns, queryable) ——
    # Three-principal model, mirroring model_usages:
    #   owner_principal_id    — the PROVIDER of the consumed resource. For
    #                           instances/PV that's the CLUSTER owner (clusters
    #                           are shareable across orgs).
    #   consumer_principal_id — the CONSUMER: the tenant that created the resource
    #                           on the (possibly shared) cluster. == owner when
    #                           run on one's own cluster; differs when consuming
    #                           a shared cluster. NULL = global (admin "All").
    #   creator_id            — the actual user (actor) who created it.
    # FK-less on purpose: this is an audit row, so deleting a principal
    # or the cluster must NOT null out the attribution (a SET NULL would erase
    # "who this usage belongs to"). ids stay as reported; the ``*_name`` snapshots keep
    # it human-readable after the parent is gone (UI marks such rows
    # ``(Deleted)``).
    owner_principal_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    owner_name: Optional[str] = Field(default=None)
    consumer_principal_id: Optional[int] = Field(
        default=None, sa_column=Column(Integer)
    )
    consumer_name: Optional[str] = Field(default=None)
    # Consumer principal kind (``org`` / ``user`` / ``group``) snapshot, so the
    # Organization breakdown can tag personal (USER) consumers. NULL pre-upgrade.
    consumer_principal_kind: Optional[str] = Field(default=None)
    creator_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    creator_name: Optional[str] = Field(default=None)
    cluster_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    cluster_name: Optional[str] = Field(default=None)

    # —— Meter ——
    meter_key: str = Field(sa_column=Column(String(64), nullable=False))
    resource_type: str = Field(sa_column=Column(String(32), nullable=False))
    # FK-less: resource_id is polymorphic (instance / volume id spaces) and
    # must survive parent deletion. resource_name travels as a snapshot.
    resource_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    resource_name: str = Field(default=..., max_length=255)
    resource_display_name: Optional[str] = Field(default=None, max_length=255)

    # —— Grouping dimensions ——
    # ``sku`` is the "Type" breakdown dimension: GPU = gpu_type (card model);
    # CPU = cpu flavor; storage = storage type.
    sku: Optional[str] = Field(default=None, max_length=128)
    # Count of sku units — a real column so per-unit metrics are plain SQL.
    # GPU instance = card count (drives GPU-Hours = SUM(quantity * sku_count));
    # CPU instance = 1 (whole machine); storage = 1.
    sku_count: int = Field(default=1)
    # Display-only metadata. Never grouped / filtered on.
    dimensions: Optional[Dict[str, Any]] = Field(
        default=None, sa_column=Column(JSON(), nullable=True)
    )

    # —— Quantity + time (canonical integer) ——
    # UTC hour bucket (truncated to the hour). Coarser granularities are
    # derived via date_trunc at query time.
    bucket_start: datetime = Field(sa_column=Column(UTCDateTime(), nullable=False))
    quantity: int = Field(
        default=0, sa_column=Column(BigInteger, nullable=False, default=0)
    )
    unit: str = Field(sa_column=Column(String(32), nullable=False))

    # Incremental-settlement cursor: the high-water "already accumulated up
    # to" instant for this row. Lets the periodic tick add only the new delta
    # without double counting, and lets a restart resume.
    settled_until: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime(), nullable=True)
    )

    # Bucket finalization. NULL = open (may still accumulate); non-NULL = sealed
    # and immutable. A bucket is sealed once its hour has fully elapsed plus a
    # grace window for late events (see ``seal_due``), after which no segment can
    # land in it — so a sealed row can be trusted as final (it won't change).
    sealed_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime(), nullable=True)
    )

    model_config = ConfigDict(protected_namespaces=())

    @classmethod
    async def seal_due(
        cls,
        session,
        meter_key: str,
        now: datetime,
        grace_seconds: int,
    ) -> None:
        """Seal every still-open bucket whose hour has fully elapsed.

        A bucket ``[t, t+1h)`` is finalized once ``now >= t + 1h + grace``: the
        hour is over and the grace window has absorbed any late-arriving event,
        so nothing more can land in it. Run on the collector tick *after*
        settling, so a still-running resource's current hour is written before
        it becomes eligible. Idempotent — only NULL ``sealed_at`` rows of this
        meter are touched.
        """
        cutoff = now - timedelta(hours=1, seconds=grace_seconds)
        await session.exec(
            update(cls)
            .where(
                cls.meter_key == meter_key,
                cls.sealed_at.is_(None),
                cls.bucket_start <= cutoff,
            )
            .values(sealed_at=now)
        )
        await session.commit()


# ---------------------------------------------------------------------------
# Cold archive — identical column layout for bulk INSERT ... SELECT.
# ---------------------------------------------------------------------------


class MeteredUsageArchive(SQLModel, BaseModelMixin, table=True):
    """Cold-storage archive for ``metered_usage``.

    Same column layout as the hot table; ``id`` is a plain primary key without
    autoincrement (rows reuse the source id). The owner/creator/cluster id
    columns are FK-less here — audit rows must outlive parent deletes.
    """

    __tablename__: ClassVar[str] = "metered_usage_archive"

    id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, primary_key=True, autoincrement=False),
    )
    owner_principal_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    owner_name: Optional[str] = Field(default=None)
    consumer_principal_id: Optional[int] = Field(
        default=None, sa_column=Column(Integer)
    )
    consumer_name: Optional[str] = Field(default=None)
    # Consumer principal kind (``org`` / ``user`` / ``group``) snapshot, so the
    # Organization breakdown can tag personal (USER) consumers. NULL pre-upgrade.
    consumer_principal_kind: Optional[str] = Field(default=None)
    creator_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    creator_name: Optional[str] = Field(default=None)
    cluster_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    cluster_name: Optional[str] = Field(default=None)
    meter_key: str = Field(sa_column=Column(String(64), nullable=False))
    resource_type: str = Field(sa_column=Column(String(32), nullable=False))
    resource_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    resource_name: str = Field(default=..., max_length=255)
    resource_display_name: Optional[str] = Field(default=None, max_length=255)
    sku: Optional[str] = Field(default=None, max_length=128)
    sku_count: int = Field(default=1)
    dimensions: Optional[Dict[str, Any]] = Field(
        default=None, sa_column=Column(JSON(), nullable=True)
    )
    bucket_start: datetime = Field(sa_column=Column(UTCDateTime(), nullable=False))
    quantity: int = Field(
        default=0, sa_column=Column(BigInteger, nullable=False, default=0)
    )
    unit: str = Field(sa_column=Column(String(32), nullable=False))
    settled_until: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime(), nullable=True)
    )
    sealed_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime(), nullable=True)
    )

    model_config = ConfigDict(protected_namespaces=())
