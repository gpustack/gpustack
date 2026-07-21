"""Event-level audit log for resource lifecycle changes.

``resource_events`` is the event source of truth — every metered phase
transition / lifecycle event for an instance or persistent volume is
appended here. Downstream consumers:

* ``ResourceUsageCollector`` / ``StorageUsageCollector`` derive their
  "where am I in the state machine" from the events log instead of a
  separate checkpoint table.
* The UI "Resource Events" page reads from here directly.

Same FK-less audit-survival contract as ``model_usage_details``: ids stay
as reported when the underlying entity is deleted; mutable display fields
travel as ``*_name`` snapshots alongside.

**No price columns** — this is a pure metering / audit log; it records what
happened and when, never any cost.
"""

from datetime import datetime
from typing import Any, ClassVar, Dict, Optional

from pydantic import ConfigDict
from sqlalchemy import Column, Integer, String
from sqlmodel import Field, SQLModel

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import JSON, UTCDateTime


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# resource_type
RESOURCE_TYPE_GPU_INSTANCE = "gpu_instance"
RESOURCE_TYPE_CPU_INSTANCE = "cpu_instance"
RESOURCE_TYPE_PERSISTENT_VOLUME = "persistent_volume"

# event_type
# Lifecycle (also drive the collectors' state machine):
EVENT_TYPE_CREATED = "created"
EVENT_TYPE_DELETED = "deleted"
EVENT_TYPE_PHASE_TO_METERED = "phase_to_metered"
EVENT_TYPE_PHASE_LEFT_METERED = "phase_left_metered"
# Audit-only (no collector effect):
EVENT_TYPE_UPDATED = "updated"
EVENT_TYPE_ATTACHED = "attached"
EVENT_TYPE_DETACHED = "detached"


# ---------------------------------------------------------------------------
# Hot table
# ---------------------------------------------------------------------------


class ResourceEvent(SQLModel, BaseModelMixin, table=True):
    """Per-event resource lifecycle row.

    ``id`` columns (``creator_id`` / ``cluster_id`` / ``resource_id``) are
    plain integers without FKs — audit rows must outlive the entities they
    describe, so ``SET NULL`` would lose information the downstream collector
    and the resource-events UI both need.
    """

    __tablename__: ClassVar[str] = "resource_events"

    id: Optional[int] = Field(default=None, primary_key=True)

    # Wall-clock anchor for the event. Distinct from ``created_at`` so the
    # archiver / resource-events query sort on event semantics even after rows
    # are migrated to the archive table.
    occurred_at: datetime = Field(sa_column=Column(UTCDateTime(), nullable=False))

    # Tenant scope + creator snapshots (FK-less, see class docstring).
    # owner = provider (cluster owner); consumer = the tenant that created the
    # resource on the (possibly shared) cluster; creator = the user. Mirrors
    # metered_usage / model_usages.
    owner_principal_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    owner_name: Optional[str] = Field(default=None)
    consumer_principal_id: Optional[int] = Field(
        default=None, sa_column=Column(Integer)
    )
    consumer_name: Optional[str] = Field(default=None)
    # Consumer principal kind (``org`` / ``user`` / ``group``) snapshot.
    consumer_principal_kind: Optional[str] = Field(default=None)
    creator_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    creator_name: Optional[str] = Field(default=None)

    cluster_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    cluster_name: Optional[str] = Field(default=None)

    # Resource (polymorphic). See ``RESOURCE_TYPE_*`` constants.
    resource_type: str = Field(sa_column=Column(String(32), nullable=False))
    resource_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    resource_name: str = Field(default=..., max_length=255)

    # Event. See ``EVENT_TYPE_*`` constants.
    event_type: str = Field(sa_column=Column(String(64), nullable=False))
    event_message: Optional[str] = Field(default=None, max_length=1024)

    # ``status.phase`` at event time — denormalized so the collector can
    # answer "what phase was the resource in at T?" without re-reading
    # ``spec_snapshot``.
    phase: Optional[str] = Field(default=None, max_length=64)

    # Full resource snapshot at event time (GPUInstanceSpec / PVSpec / …).
    # JSON-encoded; schema-evolution-tolerant.
    spec_snapshot: Optional[Dict[str, Any]] = Field(
        default=None, sa_column=Column(JSON(), nullable=True)
    )

    model_config = ConfigDict(protected_namespaces=())


# ---------------------------------------------------------------------------
# Cold archive — identical column layout for bulk INSERT ... SELECT.
# ---------------------------------------------------------------------------


class ResourceEventArchive(SQLModel, BaseModelMixin, table=True):
    """Cold-storage archive for ``resource_events``.

    Same column layout as the hot table; ``id`` is a plain primary key
    without autoincrement — rows are archived from ``resource_events`` and reuse
    the source id.
    """

    __tablename__: ClassVar[str] = "resource_events_archive"

    id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, primary_key=True, autoincrement=False),
    )
    occurred_at: datetime = Field(sa_column=Column(UTCDateTime(), nullable=False))
    owner_principal_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    owner_name: Optional[str] = Field(default=None)
    consumer_principal_id: Optional[int] = Field(
        default=None, sa_column=Column(Integer)
    )
    consumer_name: Optional[str] = Field(default=None)
    # Consumer principal kind (``org`` / ``user`` / ``group``) snapshot.
    consumer_principal_kind: Optional[str] = Field(default=None)
    creator_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    creator_name: Optional[str] = Field(default=None)
    cluster_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    cluster_name: Optional[str] = Field(default=None)
    resource_type: str = Field(sa_column=Column(String(32), nullable=False))
    resource_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    resource_name: str = Field(default=..., max_length=255)
    event_type: str = Field(sa_column=Column(String(64), nullable=False))
    event_message: Optional[str] = Field(default=None, max_length=1024)
    phase: Optional[str] = Field(default=None, max_length=64)
    spec_snapshot: Optional[Dict[str, Any]] = Field(
        default=None, sa_column=Column(JSON(), nullable=True)
    )

    model_config = ConfigDict(protected_namespaces=())
