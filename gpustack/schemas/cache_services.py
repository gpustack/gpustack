import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from sqlalchemy import JSON, Column, ForeignKey, Integer, String, UniqueConstraint
from sqlmodel import Field, SQLModel, Text

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import (
    ItemList,
    PaginatedList,
    UTCDateTime,
    pydantic_column_type,
)
from gpustack.schemas.principals import _platform_principal_id

CACHE_SERVICE_WORKLOAD_TYPE = "cache-service"


def cache_service_instance_workload_name(
    cache_service_id: int, instance_id: int
) -> str:
    """Name of the workload a managed cache service instance runs as on
    its worker."""
    return f"cache-svc-{cache_service_id}-i{instance_id}"


class CacheServiceStateEnum(str, Enum):
    r"""
    Enum for Cache Service / Cache Service Instance State

    Transitions (managed instances):

       |- Server -|- - - - - - Worker - - - - - -|
       |          |                              |
    PENDING ---> ---> STARTING ---> RUNNING / ERROR
                                       ^
                                       |(health probe)
                                       v
                                  UNREACHABLE

    A managed service's own state is an aggregate the server controller
    computes over its instances' states. External services skip STARTING:
    the health checker flips them between RUNNING and UNREACHABLE.
    """

    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"
    UNREACHABLE = "unreachable"

    def __str__(self):
        return self.value


class CacheServiceEndpoint(BaseModel):
    """Where an engine attaches, resolved from the instance the platform
    picked for it: never stored, carried from resolution to injection."""

    host: Optional[str] = None
    port: Optional[int] = None
    url: Optional[str] = None
    """Alternative to host+port for providers addressed by URL."""

    params: Dict[str, Any] = {}
    """Facts about the resolved placement the provider's injection
    templates read, in the platform's own vocabulary (e.g. locality)."""


class CacheServiceL2Storage(BaseModel):
    """The L2 storage backend the cache server spills
    KV cache to when its in-memory (L1) capacity is exceeded."""

    backend: str
    """Key into the provider's declared l2_backends."""

    params: Dict[str, Any] = {}
    """Backend field name -> value, per the provider's field declarations."""

    adapter_flag_enabled: Optional[bool] = None
    """Optional per-backend switch. When the provider marks its adapter flag
    optional, ``None`` uses the catalog default; false tells the UI to hide
    the backend fields and does not emit the adapter flag."""


class CacheServiceConfig(BaseModel):
    image: Optional[str] = None
    """Container image for the custom provider version; ignored otherwise."""

    env: Optional[Dict[str, str]] = None
    """Extra environment variables, set on every component's container.
    Unlike a command-line flag, an env var is namespaced by the program
    that reads it, so one meant for a single role is inert in the others
    — and the vars worth setting here (a log level, a proxy) are usually
    meant for all of them anyway."""

    parameters: Optional[Dict[str, List[str]]] = None
    """Extra command-line flags appended to a component's command, keyed
    by the component they belong to ("" for the one process a provider
    without components runs). Flags belong to a role rather than to the
    service: each component runs its own binary, whose parser rejects
    another's flags. User-specified flags override template defaults."""

    fields: Optional[Dict[str, Any]] = None
    """Values for the provider's declared fields, keyed by field
    name;
    they fill the fields' {{name}} template placeholders (free-form
    parameters still override any flag the templates produce)."""

    l2_storages: Optional[List[CacheServiceL2Storage]] = None
    """Managed mode only: ordered L2 storage backends forming a cascade.
    The first entry is the preferred read tier; all entries receive writes."""

    management_url: Optional[str] = None
    """Link to the cache engine's own management UI, rendered beside the
    service name in the list and detail views. Display-only: the
    platform never calls it."""


@dataclass
class CacheServiceDeploymentMetadata:
    name: str
    labels: Dict[str, str]


class CacheServiceBase(SQLModel):
    name: str = Field(index=True)
    provider_name: str
    provider_version: Optional[str] = None
    cluster_id: int = Field(foreign_key="clusters.id", nullable=False)
    worker_id: Optional[int] = None
    """Replicas topology only: pins one instance to an explicitly chosen
    worker; empty leaves placement to the controller.
    Per-node providers derive their placement from the cluster's workers
    instead."""

    worker_selector: Optional[Dict[str, str]] = Field(
        sa_column=Column(JSON), default=None
    )
    """Labels a cluster worker must ALL match for the service to place an
    instance on it: per_node components run on the matching workers,
    replicas components pick theirs from the same set. None or empty
    means every worker of the cluster."""

    config: Optional[CacheServiceConfig] = Field(
        sa_column=Column(pydantic_column_type(CacheServiceConfig)), default=None
    )
    state: CacheServiceStateEnum = Field(
        default=CacheServiceStateEnum.PENDING,
        sa_column=Column(String(length=64), nullable=False),
    )
    state_message: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    healthy: Optional[bool] = None
    last_check_at: Optional[datetime] = Field(
        sa_column=Column(UTCDateTime), default=None
    )
    restart_on_error: Optional[bool] = True
    """Automatically restart (with backoff) when a cache
    server instance exits; False parks the instance in ERROR for manual
    handling. Applies to all of the service's instances."""


class CacheService(CacheServiceBase, BaseModelMixin, table=True):
    __tablename__ = "cache_services"
    __table_args__ = (
        UniqueConstraint(
            "owner_principal_id", "name", name="uix_cache_services_name_per_owner"
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    owner_principal_id: int = Field(
        default_factory=_platform_principal_id,
        sa_column=Column(
            Integer, ForeignKey("principals.id", ondelete="CASCADE"), nullable=False
        ),
    )


class CacheServiceCreate(CacheServiceBase):
    pass


class CacheServiceUpdate(CacheServiceBase):
    pass


class CacheServicePublic(CacheServiceBase):
    id: int
    owner_principal_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime


CacheServicesPublic = PaginatedList[CacheServicePublic]


class CacheServiceInstanceBase(SQLModel):
    """One cache server container of a managed cache service. The parent
    provider's topology dictates the desired rows: a replicas component
    gets its configured count placed across the cluster's matching
    workers, one per worker — what a component's instances share on a
    node they would collide over — so a cluster with fewer workers runs
    what fits; a per_node component gets one instance per non-deleted
    matching worker. Rows on NOT_READY workers are kept —
    the worker restarts its container when it comes back.
    """

    component: str = Field(
        default="", sa_column=Column(String(length=64), nullable=False)
    )
    """Which of the provider's declared components this instance runs
    (e.g. "master" / "store"). Empty for single-component
    providers."""

    component_addresses: Optional[Dict[str, str]] = Field(
        sa_column=Column(JSON), default=None
    )
    """Dependency addresses resolved by the controller at creation
    (component name -> host:port), rendered into this instance's
    templates as {{component.<name>.address}}. Stamped rather than
    looked up on the worker: the server knows the dependency's placement,
    and a stamp that stops matching the current address marks the
    instance for recreation (the address is baked into the running
    process's config)."""

    name: str = Field(index=True)
    """Display identity: the parent service's name (as of instance
    creation) plus a short random suffix, mirroring model instance
    naming."""

    cache_service_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey("cache_services.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        )
    )
    worker_id: int
    cluster_id: int
    """Denormalized from the parent service so cluster-bound service
    accounts' reads (list conditions, watch filter) scope without a join."""

    ports: Optional[Dict[str, int]] = Field(sa_column=Column(JSON), default=None)
    """Every port allocated for this instance on its worker (port name ->
    port), one per name its component declares, rendered into its
    templates as {{ports.<name>}}. Recorded so a restart keeps the ports
    it already published to peers."""

    port: Optional[int] = None
    """The port this instance is addressed by — its component's
    address_port, denormalized out of ``ports``. A reader that has only
    the row (the instance list, a watch event) builds the instance's
    address from it without resolving the provider's catalog."""

    state: CacheServiceStateEnum = Field(
        default=CacheServiceStateEnum.PENDING,
        sa_column=Column(String(length=64), nullable=False),
    )
    state_message: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    healthy: Optional[bool] = None
    last_check_at: Optional[datetime] = Field(
        sa_column=Column(UTCDateTime), default=None
    )
    restart_count: Optional[int] = 0
    last_restart_time: Optional[datetime] = Field(
        sa_column=Column(UTCDateTime), default=None
    )

    spec_digest: Optional[str] = None
    """Digest of the container-shaping part of the parent service's spec
    (provider_version + config) as of instance creation. The controller
    reconciles the instance *set*, not the spec — a spec edit leaves
    running containers untouched by design (recovery is
    delete-to-recreate) — so this digest is how that drift is made
    visible instead of silent: the aggregate flags the service when any
    instance was created from an older spec. None on rows predating the
    field (unknown, never flagged)."""

    def get_deployment_metadata(self) -> CacheServiceDeploymentMetadata:
        return CacheServiceDeploymentMetadata(
            name=cache_service_instance_workload_name(self.cache_service_id, self.id),
            labels={
                "type": CACHE_SERVICE_WORKLOAD_TYPE,
                "cache-service-id": str(self.cache_service_id),
                "cache-service-instance-id": str(self.id),
            },
        )


class CacheServiceInstance(CacheServiceInstanceBase, BaseModelMixin, table=True):
    __tablename__ = "cache_service_instances"
    __table_args__ = (
        # A component places one instance per worker, so this is the row's
        # identity — and the database saying so is what keeps two
        # reconcile passes racing over the same missing row from creating
        # it twice.
        UniqueConstraint(
            "cache_service_id",
            "component",
            "worker_id",
            name="uix_cache_service_instances_component_per_worker",
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)


class CacheServiceInstanceCreate(CacheServiceInstanceBase):
    pass


class CacheServiceInstanceUpdate(CacheServiceInstanceBase):
    pass


class CacheServiceInstancePublic(CacheServiceInstanceBase):
    id: int
    created_at: datetime
    updated_at: datetime


CacheServiceInstancesPublic = PaginatedList[CacheServiceInstancePublic]


class CacheServiceModelSummary(BaseModel):
    """A model deployment attached to a cache service, as listed on the
    service detail page. Deliberately lightweight: no instance join."""

    id: int
    name: str
    replicas: int
    ready_replicas: int
    backend: Optional[str] = None


CacheServiceModelsPublic = ItemList[CacheServiceModelSummary]


class CacheServiceMetricSeries(BaseModel):
    """One chartable series of a semantic metric: filtered identifying
    labels (worker/instance) plus [timestamp, value] points; value is
    None where the sample is non-finite (chart gap)."""

    labels: Dict[str, str] = {}
    points: List[List[Optional[float]]] = []


class CacheServiceMetricChart(BaseModel):
    """One semantic metric, charted at two granularities: the
    service-level aggregate (ratios weighted by actual traffic — the
    default view, readable at any fleet size) and the per-instance
    breakdown behind a toggle."""

    aggregate: List[CacheServiceMetricSeries] = []
    instances: List[CacheServiceMetricSeries] = []


class CacheServiceAttachedMetrics(BaseModel):
    """External-cache hit accounting of one attached engine instance
    over the requested window. The row set is database-enumerated; the
    numbers come from the engine's own counters (vLLM's
    external_prefix_cache_*), so an instance whose engine exports none
    keeps its row with empty values."""

    model_id: Optional[int] = None
    model_name: Optional[str] = None
    model_instance_name: Optional[str] = None
    worker_name: Optional[str] = None
    hit_tokens: Optional[float] = None
    queried_tokens: Optional[float] = None
    hit_rate: Optional[float] = None


class ModelCacheMetricsPublic(BaseModel):
    """The same hit accounting read from the deployment's side: every
    instance of one model, so the model view can tell what the shared
    cache is doing for it without reading the cache service's telemetry
    (which is the service owner's to see).

    ``available=False`` carries why no numbers can be read at all (the
    deployment uses no shared cache, observability is off, Prometheus is
    unreachable); rows stay database-enumerated, so an instance whose
    engine exports no counters is present with empty values.
    """

    available: bool = False
    reason: Optional[str] = None
    window: Optional[int] = None
    instances: List[CacheServiceAttachedMetrics] = []


class CacheServiceMetricsPublic(BaseModel):
    """Semantic metric series for one cache service, translated from the
    provider's declared mappings and queried from the built-in
    Prometheus. available=False carries why charts cannot render (no
    declaration / observability disabled / Prometheus unreachable)."""

    available: bool = False
    reason: Optional[str] = None
    start: Optional[float] = None
    end: Optional[float] = None
    step: Optional[int] = None
    mappings: Dict[str, CacheServiceMetricChart] = {}
    throughput: Dict[str, CacheServiceMetricChart] = {}
    attached: List[CacheServiceAttachedMetrics] = []


def cache_service_spec_digest(service: "CacheService") -> str:
    """Digest of the spec fields that shape a running cache container:
    provider_version and config (image, capacity, parameters, env,
    fields, L2). worker_id/worker_selector are excluded — the controller
    reconciles placement live, so they cannot drift."""
    payload = {
        "provider_version": service.provider_version,
        "config": service.config.model_dump() if service.config else None,
    }
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


class CacheConfigSnapshot(BaseModel):
    """
    Resolved shared-cache connection info denormalized onto a ModelInstance
    at creation time, so the worker can inject engine config without a
    server round-trip.
    """

    cache_service_id: int
    cache_service_name: Optional[str] = None
    provider_name: Optional[str] = None
    provider_version: Optional[str] = None
    endpoint: Optional[CacheServiceEndpoint] = None
    chunk_size: Optional[int] = None
    env: Dict[str, str] = {}
    args: List[str] = []
    files: Dict[str, str] = {}
    """Connector config files keyed by container path, written by the
    serving script before the engine starts."""

    injected: bool = False
    """False means the instance starts without the shared cache (degraded)."""

    reason: Optional[str] = None
    """Human-readable reason when injected is False."""

    endpoint_live: Optional[bool] = None
    """Whether the endpoint the engine started with is still served
    (None = never evaluated, treated as live). The snapshot records the
    engine's actual startup config and never mutates while it runs; this
    is the one field that tracks the present — the controller re-resolves
    on cache-instance changes and flips it when the recorded endpoint
    stops (or resumes) being attachable, so "attached" indicators do not
    report a cache that is gone."""
