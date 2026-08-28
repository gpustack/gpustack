import hashlib
import json
from datetime import datetime
from typing import ClassVar, Optional, List

from pydantic import ConfigDict, BaseModel
from sqlalchemy import UniqueConstraint, Column, Index, Integer, ForeignKey
from sqlalchemy.ext.hybrid import hybrid_property
from sqlmodel import SQLModel, Field

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import (
    pydantic_camel_case_generator,
    pydantic_column_type,
    ItemList,
    ListParams,
    PaginatedList,
)


class GPUInstanceTypeUnitResources(BaseModel):
    """
    Represents the unit resources of a GPU instance type.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    cpu: Optional[str] = None
    """
    The per-device unit CPU resources of the GPU instance type, ends with "m".
    """

    ram: Optional[str] = None
    """
    The per-device RAM resources of the GPU instance type, ends with "Mi".
    """


class GPUInstanceTypeCPUCache(BaseModel):
    """
    Represents the cache information of the CPU of a GPU instance type.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    l1i: Optional[str] = None
    """
    The L1 instruction cache size in bytes of the CPU, e.g. "64".
    """

    l1d: Optional[str] = None
    """
    The L1 data cache size in bytes of the CPU, e.g. "64".
    """

    l2: Optional[str] = None
    """
    The L2 cache size in bytes of the CPU, e.g. "256", "512".
    """

    l3: Optional[str] = None
    """
    The L3 cache size in bytes of the CPU, e.g. "8192", "16384".
    """


class GPUInstanceTypeCPU(BaseModel):
    """
    Represents the CPU resource information of a GPU instance type.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    physical_cores: Optional[str] = None
    """
    The number of physical cores of the CPU, e.g. "4", "8".
    """

    threads_per_physical_core: Optional[str] = None
    """
    The number of threads per physical core of the CPU, e.g. "2", "4".
    """

    logical_cores: Optional[str] = None
    """
    The number of logical cores of the CPU, e.g. "8", "16".
    """

    stepping: Optional[str] = None
    """
    The stepping of the CPU, e.g. "0", "1".
    """

    clock_speed: Optional[str] = None
    """
    The speed in Hz of the CPU, e.g. "2000".
    """

    max_clock_speed: Optional[str] = None
    """
    The maximum speed in Hz of the CPU, e.g. "3000".
    """

    cache_line: Optional[str] = None
    """
    The cache line size in bytes of the CPU, e.g. "64", "128".
    """

    cache: Optional[GPUInstanceTypeCPUCache] = None
    """
    The cache information of the CPU.
    """


class GPUInstanceTypeAcceleratorCPU(GPUInstanceTypeCPU):
    """
    Represents the CPU information of the accelerator of a GPU instance type.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    manufacturer: Optional[str] = None
    """
    The name of the CPU manufacturer, e.g. "amd", "intel".
    """

    product: Optional[str] = None
    """
    The name of the CPU product.
    """

    family: Optional[str] = None
    """
    The family of the CPU.
    """


class GPUInstanceTypeAcceleratorSlicedLogicalDetail(BaseModel):
    """
    Represents the aggregated logical (software) slicing capability of an
    accelerator group.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    cores_percentage_overcommit: bool = False
    """
    Whether each slice may claim up to 100% of the device compute (time-sharing /
    weighted sharing); false means compute is partitioned.
    """

    count: Optional[int] = None
    """
    The maximum number of soft slices the group can host, summed across cards.
    """


class GPUInstanceTypeAcceleratorSlicedPhysicalDetailProfile(BaseModel):
    """
    Represents one physical (hardware) slicing profile aggregated across an
    accelerator group's cards, e.g. an NVIDIA MIG profile.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    name: Optional[str] = None
    """
    The profile identifier, e.g. "1g.5gb".
    """

    count: Optional[int] = None
    """
    The number of instances of this profile, summed by name across the group.
    """

    memory_mib: Optional[int] = None
    """
    The memory size of this profile in MiB, e.g. 5120 for "1g.5gb".
    """


class GPUInstanceTypeAcceleratorSlicedPhysicalDetail(BaseModel):
    """
    Represents the aggregated physical (hardware) slicing capability of an
    accelerator group.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    profiles: Optional[List[GPUInstanceTypeAcceleratorSlicedPhysicalDetailProfile]] = (
        None
    )
    """
    The group's physical slicing profiles, summed by name.
    """

    count: Optional[int] = None
    """
    The group's physical-slice ceiling, summed across cards.
    """


class GPUInstanceTypeAcceleratorSlicedDetail(BaseModel):
    """
    Represents the group-level slicing capability of an accelerator, aggregated
    from its per-card slicing status.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    logical: Optional[GPUInstanceTypeAcceleratorSlicedLogicalDetail] = None
    """
    The aggregated logical (software) slicing capability.
    """

    physical: Optional[GPUInstanceTypeAcceleratorSlicedPhysicalDetail] = None
    """
    The aggregated physical (hardware) slicing capability.
    """


class GPUInstanceTypeAcceleratorDetail(BaseModel):
    """
    Represents the observed accelerator information of a GPU instance type.

    Mirrors the status-side Go InstanceTypeAcceleratorDetail: it carries the
    aggregated slicing capability (slicedDetail), which replaces the spec-side
    sliceable capability flag.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    memory: Optional[str] = None
    """
    The VRAM size of the accelerator, e.g. "65535Mi".
    """

    cores: Optional[str] = None
    """
    The number of cores of the accelerator, e.g. "128", "256".
    """

    compute_capability: Optional[str] = None
    """
    The compute capability of the accelerator, e.g. "8.0", "7.0".
    """

    sliced_detail: Optional[GPUInstanceTypeAcceleratorSlicedDetail] = None
    """
    The pool's aggregated slicing capability for this accelerator group.
    """

    cpu: Optional[GPUInstanceTypeAcceleratorCPU] = None
    """
    The CPU information of the accelerator.
    """


class GPUInstanceTypeDetail(GPUInstanceTypeCPU, GPUInstanceTypeAcceleratorDetail):
    """
    Represents the observed hardware descriptor of a GPU instance type.

    Mirrors the status-side Go InstanceTypeDetail: the device identity fields
    plus the inlined CPU details and inlined accelerator detail, producing the
    flat JSON the gateway emits under ``status.detail``.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    manufacturer: Optional[str] = None
    """
    The name of the GPU instance type manufacturer, e.g. "nvidia", "generic".
    """

    product: Optional[str] = None
    """
    The name of the GPU instance type product.
    """

    family: Optional[str] = None
    """
    The family of the GPU instance type.
    """


class GPUInstanceTypeSpec(BaseModel):
    """
    Represents the specification of a GPU instance type.

    Holds only the definitional fields an admin sets; observed hardware (CPU /
    accelerator / manufacturer detail) lives on ``GPUInstanceTypeStatus.detail``.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    display_name: Optional[str] = None
    """
    The human-friendly display name of the GPU instance type, e.g. "A10G Pool".
    """

    accelerator_group: Optional[str] = None
    """
    The accelerator group (the acceleratable key) of an accelerated pool,
    e.g. "nvidia-a10g"; empty for a generic pool.
    """

    general_group: Optional[str] = None
    """
    The general (CPU) group of the pool: the real CPU key when
    instance-type-aware-cpu-manufacturer is on, or the "generic" sentinel for a
    collapsed (unaware) generic pool; empty for an accelerated pool when
    awareness is off.
    """

    acceleratable: bool = False
    """
    Indicates whether the pool represents accelerated hardware; a generic
    (CPU-only) pool is false. It delimits generic from accelerated flavors.
    """

    os: Optional[str] = None
    """
    The operating system of the GPU instance type, e.g. "linux", "windows".
    """

    arch: Optional[str] = None
    """
    The architecture of the GPU instance type, e.g. "amd64", "arm64".
    """

    unit_resources: Optional[GPUInstanceTypeUnitResources] = None
    """
    The unit resources of the GPU instance type, which represents the resources of one GPU card.
    """

    local_storage: Optional[str] = None
    """
    The ephemeral local storage of the GPU instance type, e.g. "100Gi".
    """


class GPUInstanceTypeResource(BaseModel):
    """
    Represents the resource information of a GPU instance type.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    once_max_request: str
    """
    The maximum resource that can be requested once, e.g. "4".
    """

    remaining: str
    """
    The remaining resource that can be requested, e.g. "16".
    """

    capacity: str
    """
    The total capacity of the resource, e.g. "20".
    """


class GPUInstanceTypeAcceleratorProfileCount(BaseModel):
    """
    Pairs a physical-slice profile name with a count of instances — allocated
    (bound) or remaining (still buildable), per the field carrying it.

    It is deliberately not the capability catalog's
    ``GPUInstanceTypeAcceleratorSlicedPhysicalDetailProfile``, which carries
    ``memory_mib``: a ledger entry never does, and the operator keeps the two
    types independently evolvable for that reason.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    name: str
    """
    The profile identifier, e.g. "1g.10gb". Required, mirroring the operator's
    ``json:"name"`` with no omitempty: the ledger is keyed by profile name, so a
    nameless entry carries nothing a caller could select on.
    """

    count: Optional[int] = None
    """
    The number of instances of this profile. Absent means zero rather than
    unknown: the operator omits the field at zero, so an entry naming only a
    profile is that profile at zero.
    """


class GPUInstanceTypePartitionedResource(GPUInstanceTypeResource):
    """
    Represents the hardware-partitionable resource of a GPU instance type: the
    scalar view every resource shares, plus the pool's per-profile ledger.

    The per-profile lists answer "which partition profiles can I still get".
    Neither alternative does: the scalar ``remaining`` is a best case over a
    card's profiles rather than a total (the profiles on one card compete for the
    same physical slices, so summing them would multiply-count the same
    hardware), and ``detail.sliced_detail`` is the static capability catalog,
    which by design does not move as instances are carved and released.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    allocated_profiles: Optional[List[GPUInstanceTypeAcceleratorProfileCount]] = None
    """
    How many instances of each profile the pool's partitioned cards currently
    hold, summed by profile name. A profile holding nothing is absent rather
    than listed at zero — unlike remaining_profiles, where zero carries meaning.
    """

    remaining_profiles: Optional[List[GPUInstanceTypeAcceleratorProfileCount]] = None
    """
    How many more instances of each profile the pool can still host, summed by
    profile name. Every profile the pool offers gets an entry, even at zero, so
    "offered but currently full" stays distinguishable from "not offered".
    """


class GPUInstanceTypeStatus(BaseModel):
    """
    Represents the status of a GPU instance type.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
        # The table-backed read validates a ``GPUInstanceType`` row into
        # ``GPUInstanceTypePublic``, and that row's ``status`` is a
        # ``GPUInstanceTypeStatusPublic`` INSTANCE rather than a dict — so this
        # annotation has to be able to read it attribute by attribute.
        from_attributes=True,
    )

    detail: Optional[GPUInstanceTypeDetail] = None
    """
    The observed hardware descriptor of the GPU instance type, computed by the
    operator from the matched flavor's notes and the pool's device ledger.
    """

    phase: Optional[str] = None
    """
    The phase of the GPU instance type, e.g. "Active", "Inactive".
    """

    phase_message: Optional[str] = None
    """
    Phase message is the message of the phase.
    """

    accelerator: Optional[GPUInstanceTypeResource] = None
    """
    The allocatable-as-exclusive accelerator resource of the candidate, e.g. "1", "4".
    """

    accelerator_shared: Optional[GPUInstanceTypeResource] = None
    """
    The shareable accelerator resource of the candidate, e.g. "10", "40".
    """

    accelerator_sliced: Optional[GPUInstanceTypeResource] = None
    """
    The sliceable accelerator resource of the candidate, e.g. "100", "400".
    """

    accelerator_partitioned: Optional[GPUInstanceTypePartitionedResource] = None
    """
    The hardware-partitionable view: the partition instances the pool's
    partitioned cards can still host, summed over those cards, plus the pool's
    per-profile ledger. It is disjoint from the three views above — a card in a
    partitioning mode can serve no other kind of claim — so a pool with no
    partitioned card reports zero here.
    """

    cpu: Optional[GPUInstanceTypeResource] = None
    """
    The CPU once max request resource of the candidate, e.g. "4", "8".
    """


class GPUInstanceTypeStatusPublic(BaseModel):
    """
    Represents the persisted status of a GPU instance type.

    The server-side projection records the observed hardware descriptor — the
    operator backfills ``status.detail`` asynchronously (via a MODIFIED event,
    not the initial ADDED) — plus the phase pair that drives the list's phase
    badge and its activate / deactivate actions.

    The resource ledger (``accelerator*``, ``cpu``, the partitioned profile
    ledger) is deliberately NOT persisted, and a new field joins this subset only
    after its change frequency is weighed: every row write publishes on the
    internal bus, which the list route puts one SSE stream per open page on the
    other end of, so a volatile field turns cluster-wide workload churn into a
    fan-out of events. The ledger is recomputed on every workload movement,
    whereas the three fields here move on real state transitions. Current
    remaining capacity is read live from the gateway instead, via the aggregated
    route.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    detail: Optional[GPUInstanceTypeDetail] = None
    """
    The observed hardware descriptor of the GPU instance type, mirrored from
    the operator's ``status.detail``.
    """

    phase: Optional[str] = None
    """
    The phase of the GPU instance type, e.g. "Active", "Draining", "Inactive".
    Mirrored from the operator's ``status.phase``, which upstream derives in
    ``apistatus.GetSummaryOfClusterQueue`` from the backing ClusterQueue's stop
    policy and conditions.
    """

    phase_message: Optional[str] = None
    """
    Phase message is the message of the phase. It comes out of the same upstream
    ``apistatus.GetSummaryOfClusterQueue`` call as ``phase``, so the two always
    move together and are projected together.
    """


class GPUInstanceType(SQLModel, BaseModelMixin, table=True):
    """
    Server-side projection of a cluster's ``worker.gpustack.ai/v1`` InstanceType.

    Populated exclusively by ``GPUInstanceTypeController`` from each cluster's
    InstanceType watch (never by tenant input); it backs instance-type validation
    and the snapshot stamped onto a ``GPUInstance`` at create/update time.
    """

    # ``display_name`` below is a SQLAlchemy hybrid, not a field. Pydantic
    # rejects an unannotated class attribute outright, and annotating it would
    # make it a field with a column; this is the documented escape hatch.
    model_config = ConfigDict(ignored_types=(hybrid_property,))

    __tablename__ = "gpu_instance_types"
    __table_args__ = (
        # ``snapshot`` encodes (cluster_id, name, spec), so it is the row's
        # global identity: enforcing its uniqueness de-duplicates identical
        # types and backs the controller's query-first upsert / revive.
        UniqueConstraint("snapshot", name="uq_gpu_instance_type_snapshot"),
        # Covers this table's one hot read, the fleet-wide list:
        # ``WHERE deleted_at IS NULL AND cluster_id IN (...) ORDER BY
        # created_at DESC`` plus its COUNT, on every page load and every watch
        # open. The same prefix serves the controller's and the model deploy
        # validation's ``(cluster_id, deleted_at)`` lookups. Sibling tables
        # carry the two-column form of this for the identical pattern
        # (``idx_clusters_deleted_at_created_at`` and friends); ``cluster_id``
        # sits in the middle because every query on this table names it.
        Index(
            "idx_gpu_instance_types_deleted_at_cluster_id_created_at",
            "deleted_at",
            "cluster_id",
            "created_at",
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)

    cluster_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("clusters.id", ondelete="CASCADE"),
            nullable=False,
        ),
    )
    """
    Reference to the cluster this instance type belongs to. Cluster-scoped, not
    tenant-scoped: the table mirrors the cluster catalog, so there is no owner
    principal.
    """

    name: str
    """
    Name of the instance type within its cluster (the CR's ``metadata.name``).
    """

    spec: GPUInstanceTypeSpec = Field(
        sa_type=pydantic_column_type(GPUInstanceTypeSpec),
    )
    """
    Specification mirrored from the operator InstanceType.
    """

    status: Optional[GPUInstanceTypeStatusPublic] = Field(
        default=None,
        sa_type=pydantic_column_type(GPUInstanceTypeStatusPublic),
    )
    """
    Status mirrored from the operator InstanceType — only the named subset is
    persisted (see ``GPUInstanceTypeStatusPublic``). ``None`` until the operator
    backfills it.
    """

    snapshot: str
    """
    Stable identity hash (``sha1:<hexdigest>``) over ``(cluster_id, name, spec)``
    with the mutable ``display_name`` excluded, unique per row. See
    ``compute_snapshot``.

    It doubles as ``metered_usage.sku`` — the metering / pricing reference key —
    so it must stay byte-stable for the lifetime of a type (guaranteed: the
    operator webhook freezes the whole spec, the CR name is immutable, and
    ``display_name`` is excluded here).
    """

    definition_snapshot: Optional[str] = Field(default=None, index=True)
    """
    Cluster-independent definition hash (``sha1:<hexdigest>``) over
    ``(name, spec)`` — the same value on every cluster that offers this exact
    definition. NOT unique: N clusters offering one definition are N rows
    sharing it. See ``compute_definition_snapshot``.
    """

    derived_from_node: bool = Field(default=False)
    """
    Whether the GPUStack Operator derived this type from a node's resource
    flavors, mirrored from ``metadata.labels[DERIVED_FROM_NODE_LABEL]``. It
    cannot be derived from anything else the row holds — the cluster's
    derived-from-node setting is a fleet switch, not a row provenance, and a
    derived spec is byte-identical in shape to a hand-authored one.

    Deliberately outside both snapshots: identity is ``(cluster_id, name, spec)``
    and ``snapshot`` doubles as ``metered_usage.sku``, so a provenance marker
    must not be able to move it.
    """

    @hybrid_property
    def display_name(self) -> Optional[str]:
        """The mutable label, reachable the way every other GPU Service model
        exposes it — as ``display_name`` on the model.

        It is stored inside the ``spec`` JSON column rather than as a column of
        its own, deliberately: it is the one mutable spec field and the one
        excluded from the identity ``snapshot``. That storage decision should
        not leak into every caller, though. The list route narrows and orders on
        the display name exactly as its siblings do — ``fuzzy_fields`` resolves
        the key with ``getattr(cls, key)`` for SQL and ``getattr(event.data,
        key)`` for the watch stream, and both land here.

        Not a column and not a field: it adds no migration, and ``model_dump``
        does not carry it (the wire form stays ``spec.displayName``).
        """
        return self.spec.display_name if self.spec else None

    @display_name.expression
    def display_name(cls):  # noqa: N805 - SQLAlchemy hybrid expression form
        """The SQL twin, via SQLAlchemy's generic JSON indexing.

        Each dialect compiles its own extraction — ``spec ->> 'displayName'`` on
        PostgreSQL, ``JSON_UNQUOTE(JSON_EXTRACT(...))`` on MySQL — so there is no
        raw SQL and no dialect branching here. A row without a display name is
        SQL NULL, which is what lets a search fall through to the ``name`` arm
        and a sort coalesce back to ``name``.
        """
        return cls.spec["displayName"].as_string()

    def is_deleted(self) -> bool:
        """Whether the type row is soft-deleted (``deleted_at`` set)."""
        return self.deleted_at is not None

    def _identity_spec(self) -> dict:
        """The definitional spec that both snapshots hash over.

        ``exclude_none`` keeps identity stable across additive schema evolution:
        an unset optional field must not enter the payload, so introducing a new
        optional definitional field later does not churn the snapshot of
        existing types the operator never set it on. ``display_name`` is dropped
        because it is the one mutable spec field.
        """
        spec = self.spec.model_dump(mode="json", exclude_none=True)
        spec.pop("display_name", None)
        return spec

    @staticmethod
    def _digest(payload: dict) -> str:
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return f"sha1:{hashlib.sha1(blob.encode('utf-8')).hexdigest()}"

    def compute_snapshot(self) -> str:
        """Return this type's stable identity snapshot as ``sha1:<hexdigest>``.

        Identity is the cluster-scoped name plus the definitional spec — the
        spec now holds only definitional fields (observed hardware lives on
        ``status.detail``) — with the mutable ``display_name`` dropped. So two
        definitions that differ only by display name share a snapshot, while a
        change to a definitional field (e.g. ``unit_resources``) diverges it.
        """
        return self._digest(
            {
                "cluster_id": self.cluster_id,
                "name": self.name,
                "spec": self._identity_spec(),
            }
        )

    def compute_definition_snapshot(self) -> str:
        """Return the cluster-independent definition snapshot.

        Identical to :meth:`compute_snapshot` minus ``cluster_id``, so the same
        definition rolled out to N clusters yields N ``snapshot`` values but ONE
        ``definition_snapshot``. Metering carries both: ``snapshot`` is the
        pricing key (per-cluster pricing stays expressible), while
        ``definition_snapshot`` backs cross-cluster aggregation, bulk pricing,
        and a future switch of pricing granularity without migrating history.

        ``name`` is still part of it — a later change to the operator's derived
        naming gives newly-created types a new value (existing rows never
        change, the projection is keyed by ``snapshot``).
        """
        return self._digest({"name": self.name, "spec": self._identity_spec()})


class GPUInstanceTypeBase(BaseModel):
    """
    Base model for GPU instance type, containing common fields.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    name: str
    """
    Name of the GPU instance type. In a per-cluster view this is the object's
    name within its cluster; in the aggregated view it is the aggregated name.
    """

    spec: GPUInstanceTypeSpec
    """
    Specification of the GPU instance type.
    """

    status: Optional[GPUInstanceTypeStatus] = None
    """
    Status of the GPU instance type. ``None`` on a table-backed read until the
    operator first reports one; the live write responses always carry it (an
    unreconciled CR maps to ``{}``, not to ``None``).
    """


class GPUInstanceTypeSpecUpdate(BaseModel):
    """
    Represents the editable specification of an existing GPU instance type.

    Only the display name is editable; every other field is fixed once the GPU
    instance type exists, so all of them are intentionally absent here.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    display_name: Optional[str] = None
    """
    The human-friendly display name of the GPU instance type, e.g. "A10G Pool".
    """


class GPUInstanceTypeCreate(BaseModel):
    """
    Represents the data required to create a new GPU instance type.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    name: str
    """
    Created name of the GPU instance type.
    Must be unique in the scope of the owning principal.
    """

    spec: GPUInstanceTypeSpec
    """
    Specification for the GPU instance type.
    """


class GPUInstanceTypeUpdate(BaseModel):
    """
    Represents the data required to update an existing GPU instance type.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    name: str
    """
    Name of the GPU instance type to update. It identifies the target and is
    itself immutable.
    """

    spec: GPUInstanceTypeSpecUpdate
    """
    Editable specification for the GPU instance type.
    """


DERIVED_FROM_NODE_LABEL = "schedule.gpustack.ai/derived-from-node"
"""
Label the GPUStack Operator stamps on the GPU instance types it derives from a
node's resource flavors, as opposed to the ones an admin created by hand.
"""


class GPUInstanceTypePublic(GPUInstanceTypeBase):
    """
    Represents the public view of a GPU instance type,
    containing only fields that are safe to expose to clients.

    Served from two sources: the ``gpu_instance_types`` record table (the list
    and its watch stream, which carry the row's identity and timestamps) and a
    cluster's live CR (the write routes, which carry neither). Every field the
    table alone supplies is therefore optional.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
        # The watch path validates a ``GPUInstanceType`` ORM row straight into
        # this model — ``ActiveRecordMixin._convert_to_public_class`` resolves
        # "<model>Public" by name and calls ``model_validate`` on the row — so it
        # has to read attributes, not only a dict.
        from_attributes=True,
    )

    id: Optional[int] = None
    """
    Identifier of the record-table row. Absent on the write routes' responses,
    which return the cluster's live CR rather than a row.
    """

    cluster_id: Optional[int] = None
    """
    Reference to the cluster this GPU instance type belongs to — the fleet-wide
    list's only cluster reference, matching ``WorkerPublic`` (the name is
    resolved client-side). Absent on the write routes' responses, which are
    already cluster-scoped by their request.
    """

    created_at: Optional[datetime] = None
    """
    When the projection first recorded this GPU instance type. Absent on the
    write routes' responses.
    """

    updated_at: Optional[datetime] = None
    """
    When the projection last refreshed this GPU instance type. Absent on the
    write routes' responses.
    """

    derived_from_node: bool = False
    """
    Whether the GPUStack Operator derived this GPU instance type from a node's
    resource flavors (see :data:`DERIVED_FROM_NODE_LABEL`). It records
    provenance, not immutability: upstream authors a derived type create-only and
    never updates it, so an admin's edits to it are preserved — but while the
    cluster's ``instance-type-derived-from-node`` setting is on, a DELETE is
    undone by the next flavor reconcile, so clients should present a delete as
    one that will not stick.
    """


class GPUInstanceTypeListParams(ListParams):
    # ``status.phase`` is deliberately not sortable: it lives inside the
    # ``status`` JSON column rather than in a real column, and a JSON-path sort
    # over a nullable status buys nothing — fuzzy name search is this list's
    # discovery affordance.
    sortable_fields: ClassVar[List[str]] = [
        "name",
        "cluster_id",
        "created_at",
        "updated_at",
    ]


GPUInstanceTypesPublic = PaginatedList[GPUInstanceTypePublic]


class GPUAggregatedInstanceTypeOnceMaxRequestCandidate(BaseModel):
    """
    Represents the candidate GPU instance type for once max request accelerator tier.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    cluster: str
    """
    The cluster where the GPU instance type is available, e.g. "cluster-1".
    """

    name: str
    """
    The name of the GPU instance type, e.g. "nvidia-a100-40gb-sxm4".
    """

    phase: Optional[str] = None
    """
    The phase of the GPU instance type, e.g. "Active", "Draining", "Inactive".
    """

    accelerator: Optional[GPUInstanceTypeResource] = None
    """
    The allocatable-as-exclusive accelerator resource of the candidate, e.g. "1", "4".
    """

    accelerator_shared: Optional[GPUInstanceTypeResource] = None
    """
    The shareable accelerator resource of the candidate, e.g. "10", "40".
    """

    accelerator_sliced: Optional[GPUInstanceTypeResource] = None
    """
    The sliceable accelerator resource of the candidate, e.g. "100", "400".
    """

    accelerator_partitioned: Optional[GPUInstanceTypePartitionedResource] = None
    """
    The hardware-partitionable accelerator resource of the candidate,
    e.g. "7", "14", plus the per-profile partition ledger the tier and item sum
    by profile name.
    """

    cpu: Optional[GPUInstanceTypeResource] = None
    """
    The CPU once max request resource of the candidate, e.g. "4", "8".
    """

    accelerator_sliced_detail: Optional[GPUInstanceTypeAcceleratorSlicedDetail] = None
    """
    The candidate's observed slicing capability, taken from the cluster instance
    type's status detail.
    """


class GPUAggregatedInstanceTypeOverviewResource(BaseModel):
    """
    Represents the overview resources of a GPU instance type that can be requested.

    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    accelerator: Optional[str] = None
    """
    The allocatable-as-exclusive accelerator resource, e.g. "1", "4".
    """

    accelerator_shared: Optional[str] = None
    """
    The shareable accelerator resource, e.g. "10", "40".
    """

    accelerator_sliced: Optional[str] = None
    """
    The sliceable accelerator resource, e.g. "100", "400".
    """

    accelerator_partitioned: Optional[List[GPUInstanceTypeAcceleratorProfileCount]] = (
        None
    )
    """
    The hardware-partitionable accelerator resource, expressed per partition
    profile: in once_max_request the winning member's obtainable profiles, each
    capped at one — a partition request is a single instance on a single card; in
    remaining the sum of every Active member's obtainable profiles, by name.

    This dimension has no honest scalar, which is why it alone is a list: the
    profiles of one card compete for the same physical slices, so a total over
    them is not a capacity and a best case over them is not a total. A profile the
    fleet offers but cannot currently build stays listed at zero, so "offered but
    full" stays distinguishable from "not offered" — which detail.sliced_detail
    cannot answer either, as it records only which profiles are offered at all. A
    member publishing no per-profile ledger contributes nothing here, even when
    its own partitioned view counts instances.
    """

    cpu: Optional[str] = None
    """
    The CPU resource, e.g. "4", "8".
    """


class GPUAggregatedInstanceTypeOnceMaxRequestTier(BaseModel):
    """
    Represents the accelerator tier for selecting GPU instance types based on once max request.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    once_max_request: GPUAggregatedInstanceTypeOverviewResource
    """
    The once max request overview resources of this accelerator tier.
    """

    remaining: Optional[GPUAggregatedInstanceTypeOverviewResource] = None
    """
    The total remaining requestable resources of this tier.
    Each dimension is the sum across all candidates in the tier, so it is an
    aggregate total and may not be achievable in a single allocation.
    """

    candidates: Optional[List[GPUAggregatedInstanceTypeOnceMaxRequestCandidate]] = None
    """
    Candidate GPU instance types for this once max request tier.
    """

    accelerator_sliced_detail: Optional[GPUInstanceTypeAcceleratorSlicedDetail] = None
    """
    The tier's aggregated slicing capability: the sum of its candidates' slicing
    capability (profile counts summed by name).
    """


class GPUAggregatedInstanceTypeStatus(BaseModel):
    """
    Represents the status of an aggregated GPU instance type.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    detail: Optional[GPUInstanceTypeDetail] = None
    """
    The observed hardware descriptor of the aggregated GPU instance type — the
    hardware shared by all candidates, with the fleet-wide slicing capability
    folded into its accelerator sliced detail.
    """

    once_max_request: GPUAggregatedInstanceTypeOverviewResource
    """
    The once max request overview resources of the GPU instance type.
    """

    remaining: GPUAggregatedInstanceTypeOverviewResource
    """
    The total remaining requestable resources of the GPU instance type.
    Each dimension is the sum across all tiers, so it is an aggregate total and
    may not be achievable in a single allocation.
    """

    tiers: Optional[List[GPUAggregatedInstanceTypeOnceMaxRequestTier]] = None
    """
    The tiers for selecting GPU instance types.
    If the spec.acceleratable is true, the dimension is accelerator, and the once max request tiers are grouped by accelerator resource.
    If the spec.acceleratable is false, the dimension is cpu, and the once max request tiers are grouped by cpu resource.
    """


class GPUAggregatedInstanceTypeBase(GPUInstanceTypeBase):
    """
    Base model for GPU instance type, containing common fields.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    status: GPUAggregatedInstanceTypeStatus
    """
    Status of the GPU instance type.
    """


class GPUAggregatedInstanceTypePublic(GPUAggregatedInstanceTypeBase):
    """
    Public representation of a GPU instance type,
    containing only fields that are safe to expose to clients.
    """

    pass


GPUAggregatedInstanceTypesPublic = ItemList[GPUAggregatedInstanceTypePublic]
