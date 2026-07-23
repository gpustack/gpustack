import hashlib
import json
from typing import Optional, List

from pydantic import ConfigDict, BaseModel
from sqlalchemy import UniqueConstraint, Column, Integer, ForeignKey
from sqlmodel import SQLModel, Field

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import (
    pydantic_camel_case_generator,
    pydantic_column_type,
    ItemList,
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


class GPUInstanceTypeStatus(BaseModel):
    """
    Represents the status of a GPU instance type.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
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

    cpu: Optional[GPUInstanceTypeResource] = None
    """
    The CPU once max request resource of the candidate, e.g. "4", "8".
    """


class GPUInstanceType(SQLModel, BaseModelMixin, table=True):
    """
    Server-side projection of a cluster's ``worker.gpustack.ai/v1`` InstanceType.

    Populated exclusively by ``GPUInstanceTypeController`` from the operator watch
    stream (never by tenant input); it backs instance-type validation and the
    snapshot stamped onto a ``GPUInstance`` at create/update time.
    """

    __tablename__ = "gpu_instance_types"
    __table_args__ = (
        # ``snapshot`` encodes (cluster_id, name, spec), so it is the row's
        # global identity: enforcing its uniqueness de-duplicates identical
        # types and backs the controller's query-first upsert / revive.
        UniqueConstraint("snapshot", name="uq_gpu_instance_type_snapshot"),
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

    snapshot: str
    """
    Stable identity hash (``sha1:<hexdigest>``) over ``(cluster_id, name, spec)``
    with the mutable ``display_name`` excluded, unique per row. See
    ``compute_snapshot``.
    """

    def is_deleted(self) -> bool:
        """Whether the type row is soft-deleted (``deleted_at`` set)."""
        return self.deleted_at is not None

    def compute_snapshot(self) -> str:
        """Return this type's stable identity snapshot as ``sha1:<hexdigest>``.

        Identity is the cluster-scoped name plus the definitional spec — the
        spec now holds only definitional fields (observed hardware lives on
        ``status.detail``) — with the mutable ``display_name`` dropped. So two
        definitions that differ only by display name share a snapshot, while a
        change to a definitional field (e.g. ``unit_resources``) diverges it.
        """
        # ``exclude_none`` keeps identity stable across additive schema
        # evolution: an unset optional field must not enter the payload, so
        # introducing a new optional definitional field later does not churn the
        # snapshot of existing types the operator never set it on.
        spec = self.spec.model_dump(mode="json", exclude_none=True)
        spec.pop("display_name", None)
        payload = json.dumps(
            {"cluster_id": self.cluster_id, "name": self.name, "spec": spec},
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
        return f"sha1:{digest}"


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

    status: GPUInstanceTypeStatus
    """
    Status of the GPU instance type.
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


class GPUInstanceTypePublic(GPUInstanceTypeBase):
    """
    Represents the public view of a GPU instance type,
    containing only fields that are safe to expose to clients.
    """

    pass


GPUInstanceTypesPublic = ItemList[GPUInstanceTypePublic]


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
