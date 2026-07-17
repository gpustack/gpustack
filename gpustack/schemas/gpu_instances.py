from enum import Enum
from typing import Optional, ClassVar, List, Literal

from pydantic import (
    ConfigDict,
    BaseModel,
    AliasChoices,
    Field as PField,
)
from sqlalchemy import UniqueConstraint, Column, Integer, ForeignKey
from sqlmodel import SQLModel, Field

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import (
    pydantic_camel_case_generator,
    pydantic_column_type,
    ListParams,
    PublicFields,
    PaginatedList,
)
from gpustack.schemas.gpu_instance_persistent_volumes import (
    GPUInstancePersistentVolumeSpec,
)


class GPUInstancePort(BaseModel):
    """
    Represents a port mapping for GPU instances.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    port: int
    """
    The port number inside the container to expose.
    """

    protocol: Literal["TCP", "UDP", "SCTP"] = "TCP"
    """
    The protocol for the port.
    Defaults to "TCP".
    """

    name: Optional[str] = None
    """
    Name of the port mapping.
    """


class GPUInstanceEnvVar(BaseModel):
    """
    Represents an environment variable for GPU instances.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    name: str
    """
    Name of the environment variable.
    """

    value: str
    """
    Value of the environment variable.
    """


class GPUInstanceResources(BaseModel):
    """
    Represents the resource requirements for a GPU instance.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    cpu: Optional[str] = None
    """
    CPU resource request/limit for the GPU instance,
    e.g., "1" for 1 CPU.
    """

    ram: Optional[str] = None
    """
    RAM resource request/limit for the GPU instance,
    e.g., "2Gi" for 2 gigabyte of memory.
    """

    local_storage: Optional[str] = None
    """
    Local storage resource request/limit for the GPU instance,
    e.g., "15Gi" for 15 gigabytes of local storage.
    """

    accelerator: Optional[str] = None
    """
    Accelerator resource request/limit for the GPU instance,
    e.g., "1" for 1 GPU.
    """


class GPUInstanceImagePullSecretReference(BaseModel):
    """
    Represents a reference to a Kubernetes Secret for pulling container images.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    name: str
    """
    Name of the GPUStack Operator InstanceImagePullSecret to use for pulling container images.
    """


class GPUInstanceEphemeralVolume(BaseModel):
    """
    Represents an ephemeral volume specification for a GPU instance.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    capacity: str = "15Gi"
    """
    Capacity of the ephemeral volume.
    """


class GPUInstancePersistentVolumeReference(BaseModel):
    """
    Represents a reference to a GPU instance persistent volume for a GPU instance.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    name: str
    """
    Name of the GPUInstancePersistentVolume to use.
    """


class GPUInstancePersistentVolumeTemplate(GPUInstancePersistentVolumeReference):
    """
    Represents a template for creating a new GPU instance persistent volume when creating a GPU instance.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    spec: GPUInstancePersistentVolumeSpec
    """
    Specification for the GPU instance persistent volume to create.
    """

    release_with_instance: bool = True
    """
    Whether to release the GPU instance persistent volume when the GPU instance is deleted.
    Defaults to True.
    """


class GPUInstanceVolume(BaseModel):
    """
    Represents a volume specification for a GPU instance.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    ephemeral: Optional[GPUInstanceEphemeralVolume] = None
    """
    Optional specification for an ephemeral volume to use for the GPU instance.
    """

    persistent: Optional[GPUInstancePersistentVolumeReference] = None
    """
    Optional reference to a GPU instance persistent volume to use for the GPU instance.
    """

    persistent_template: Optional[GPUInstancePersistentVolumeTemplate] = None
    """
    Optional template for creating a new GPU instance persistent volume to use for the GPU instance.
    """


class GPUInstanceSSHPublicKeyReference(BaseModel):
    """
    Represents a reference to a GPU instance SSH public key for a GPU instance.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    name: str
    """
    Name of the GPU instance SSH public key to use for the GPU instance.
    """


class GPUInstanceIP(BaseModel):
    """
    Represents the host IP address information for a GPU instance.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    ip: str
    """
    The IP address of the host where the GPU instance is running.
    """


class GPUInstanceServicePort(GPUInstancePort):
    """
    Represents the service port information for a GPU instance.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    node_port: Optional[int] = None
    """
    Optional node port number if the GPU instance is exposed via a NodePort service.
    """


class GPUInstanceSpec(BaseModel):
    """
    Represents the specification for creating a GPU instance.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    type_: str
    """
    Type of the GPU instance.
    Used the name of GPUAggregatedInstanceTypeOnceMaxRequestCandidate.
    """

    image: str
    """
    Container image of the GPU instance to use.
    """

    image_pull_policy: Literal["Always", "IfNotPresent", "Never"] = "IfNotPresent"
    """
    Container image pull policy for the GPU instance.
    Defaults to "IfNotPresent".
    """

    command: Optional[List[str]] = None
    """
    Command to run the GPU instance.
    If not specified, the default command from the image.
    """

    ports: Optional[List[GPUInstancePort]] = None
    """
    List of port mappings for the GPU instance.
    """

    env: Optional[List[GPUInstanceEnvVar]] = None
    """
    List of environment variables for the GPU instance.
    """

    resources: Optional[GPUInstanceResources] = None
    """
    Resource requirements for the GPU instance,
    including CPU, RAM, local storage, and optional accelerator.
    """

    volume_mount: str = "/workspace"
    """
    The path inside the container where the GPU instance's volume will be mounted.
    Defaults to "/workspace".
    """

    image_pull_secret: Optional[GPUInstanceImagePullSecretReference] = None
    """
    Optional reference to a GPUStack Operator InstanceImagePullSecret for pulling container images.
    """

    volume: Optional[GPUInstanceVolume] = None
    """
    Volume specification for the GPU instance,
    which can include an ephemeral volume,
    a reference to an existing GPU instance persistent volume,
    or a template for creating a new GPU instance persistent volume.
    """

    ssh_public_keys: Optional[List[GPUInstanceSSHPublicKeyReference]] = None
    """
    Optional list of references to GPU instance SSH public keys to use for the GPU instance.
    """


class GPUInstanceSpecUpdate(BaseModel):
    """
    Represents the specification for updating a GPU instance.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    ssh_public_keys: Optional[List[GPUInstanceSSHPublicKeyReference]] = None
    """
    Optional list of references to GPU instance SSH public keys to update for the GPU instance.
    """


class GPUInstanceAcceleratorAllocation(BaseModel):
    """
    Represents the allocation status of an accelerator device for a GPU instance.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    id: str
    """
    Unique identifier is the universally unique identifier for this device.
    """

    index: int
    """
    Index is the logic number of the device, starting from 0.
    """

    mode: int
    """
    Mode is the allocation mode of the device.
    """

    allocated: Optional[int] = None
    """
    Allocated is the allocated units of the device.
    """


class GPUInstanceDevicesAllocationGroup(BaseModel):
    """
    Represents a group of allocated accelerator devices for a GPU instance, grouped by manufacturer and type.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    id: str
    """
    Unique identifier is the universally unique identifier for this device group.
    """

    manufacturer: str
    """
    Name of the manufacturer, e.g., "nvidia", "amd", etc.
    """

    accelerators: Optional[List[GPUInstanceAcceleratorAllocation]] = None
    """
    List of the allocated accelerator devices in this group.
    """


class GPUInstancePhase(str, Enum):
    """Canonical phase strings written to :class:`GPUInstanceStatus.phase`.

    Defined here (not on the controller) so the route layer can reference
    them without importing controllers, which would create a circular
    dependency.

    A ``str`` subclass enum (``StrEnum`` is 3.11+ only; this repo supports
    3.10). Members compare equal to their string value and support ``str``
    methods, and ``GPUInstanceStatus.phase`` (typed ``Optional[str]``) stores
    the coerced plain-string value, so serialization stays ``"Ready"`` etc.
    """

    # GPUStack-specific phases:

    CREATE_FAILED = "CreateFailed"
    SSH_KEY_CREATE_FAILED = "SSHPublicKeyCreateFailed"
    PV_TYPE_CREATE_FAILED = "PersistentVolumeTypeCreateFailed"
    PV_CREATE_FAILED = "PersistentVolumeCreateFailed"
    DELETING = "Deleting"
    STOPPING = "Stopping"
    STOPPED = "Stopped"
    STARTING = "Starting"
    UNKNOWN = "Unknown"

    # Kubernetes-specific phases:

    INITIALIZE_FAILED = "InitializeFailed"
    NOT_READY = "NotReady"
    READY = "Ready"


# The GPUStack-defined failure phases, enumerated for user-action gating / UX
# (e.g. which lifecycle actions are allowed from a given phase). This is the
# narrow, known set — distinct from ``GPUInstance.is_failed()``, which uses the
# broad ``endswith("Failed")`` because the worker side may report failure
# phases outside this enum.
FAILED_PHASES = frozenset(
    {
        GPUInstancePhase.CREATE_FAILED,
        GPUInstancePhase.SSH_KEY_CREATE_FAILED,
        GPUInstancePhase.PV_TYPE_CREATE_FAILED,
        GPUInstancePhase.PV_CREATE_FAILED,
        GPUInstancePhase.INITIALIZE_FAILED,
    }
)


# Phases where the instance is mid-transition between settled states. User
# lifecycle actions (e.g. /stop) are gated off these until it settles.
TRANSITIONING_PHASES = frozenset(
    {
        GPUInstancePhase.DELETING,
        GPUInstancePhase.STOPPING,
        GPUInstancePhase.STARTING,
        GPUInstancePhase.NOT_READY,
    }
)


# Phases where execution is intentionally halted but resumable (e.g. /start
# from Stopped).
INTERRUPTED_PHASES = frozenset({GPUInstancePhase.STOPPED})


# Label stamped on the worker CR's ``metadata.labels`` carrying the GPUStack
# ``GPUInstance.id``. The downstream watcher resolves a CR back to its row by
# reading this label first (falling back to namespace parsing).
KUBERES_INSTANCE_ID_LABEL = "gpustack.ai/instance-id"


class GPUInstanceStatus(BaseModel):
    """
    Represents the status of a GPU instance, including any relevant state information.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    namespace: str = ""
    """
    The Kubernetes namespace where the GPU instance is running.
    """

    phase: Optional[str] = None
    """
    The current phase of the GPU instance, such as "Pending", "Running", or "Failed".
    """

    phase_message: Optional[str] = None
    """
    Optional message providing additional details about the current phase of the GPU instance.
    """

    node_name: Optional[str] = None
    """
    The name of the Kubernetes node where the GPU instance is running.
    """

    access_addresses: Optional[List[str]] = None
    """
    Optional list of addresses (e.g., IPs or hostnames) that can be used
    to access the GPU instance.
    """

    host_ips: Optional[List[GPUInstanceIP]] = PField(
        default=None,
        validation_alias=AliasChoices("hostIPs", "host_ips"),
        serialization_alias="hostIPs",
    )
    """
    Optional list of host IP addresses where the GPU instance is running.
    """

    pod_ips: Optional[List[GPUInstanceIP]] = PField(
        default=None,
        validation_alias=AliasChoices("podIPs", "pod_ips"),
        serialization_alias="podIPs",
    )
    """
    Optional list of IP addresses where the GPU instance's pod is running.
    """

    ports: Optional[List[GPUInstanceServicePort]] = None
    """
    Optional list of port expose from the GPU instance.
    """

    allocations: Optional[List[GPUInstanceDevicesAllocationGroup]] = None
    """
    Optional list of allocated accelerator devices for the GPU instance, grouped by manufacturer and type.
    """


class GPUInstanceBase(SQLModel):
    """
    Base model for GPU instances, containing common fields.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    # For tenant scope.
    # Every object belongs to one Org. The route layer fills this with
    # ctx.current_principal_id (or platform_principal_id() for admin).
    owner_principal_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("principals.id", ondelete="CASCADE"),
            nullable=False,
        ),
    )

    display_name: Optional[str] = Field(
        nullable=True,
        default=None,
        max_length=63,
    )
    """
    Display name of the GPU instance, for easier identification by users.
    """

    description: Optional[str] = Field(
        nullable=True,
        default=None,
        max_length=1024,
    )
    """
    Description of the GPU instance.
    """


class GPUInstance(GPUInstanceBase, BaseModelMixin, table=True):
    """
    Represents a GPU instance.
    """

    __tablename__ = 'gpu_instances'
    __table_args__ = (
        # Enforce unique constraint on (owner_principal_id, name) to ensure
        # each principal can only have one key with a given name.
        # This allows different principals to have keys with the same name,
        # but prevents duplicates for the same principal.
        UniqueConstraint(
            'owner_principal_id',
            'name',
            name='uq_gpu_instance_name_per_principal',
        ),
    )
    id: Optional[int] = Field(default=None, primary_key=True)

    # Record the cluster where the GPU instance is running for auditing and management purposes.
    cluster_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("clusters.id", ondelete="CASCADE"),
            nullable=False,
        ),
    )
    """
    Reference to the cluster where the GPU instance is running.
    """

    # Record the creator of the GPU instance for auditing and ownership purposes.
    creator_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("principals.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    """
    Reference to the principal who created the GPU instance.
    """

    # Mirror of ``spec.volume.persistent[_template].name`` as a real FK
    # column. The route resolves the user-facing name to this id at create
    # time; ephemeral-volume instances leave this NULL.
    persistent_volume_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey(
                "gpu_instance_persistent_volumes.id",
                ondelete="SET NULL",
            ),
            nullable=True,
        ),
    )

    name: str = Field(
        max_length=63,
    )
    """
    Name of the GPU instance.
    Must be unique in the scope of the owning principal.
    """

    spec: GPUInstanceSpec = Field(
        sa_type=pydantic_column_type(GPUInstanceSpec),
    )
    """
    Specification for the GPU instance, including container image, resources, volumes, etc.
    """

    status: Optional[GPUInstanceStatus] = Field(
        sa_type=pydantic_column_type(GPUInstanceStatus),
        default=None,
    )
    """
    Status of the GPU instance, including phase, host IPs, ports, etc.
    """

    # -- phase predicates -------------------------------------------------- #
    # Single source of truth for phase semantics, so the controller and routes
    # stop re-deriving them with ad-hoc ``== PHASE_*`` / ``endswith`` checks.

    def _phase(self) -> Optional[str]:
        return (self.status or GPUInstanceStatus()).phase

    def is_creating(self) -> bool:
        """Brand-new instance whose phase has not been set yet (pre-create)."""
        return self._phase() is None

    def is_ready(self) -> bool:
        """Ready *and* fully populated.

        Phase is ``Ready`` and every status field the spec implies is present:
        access addresses and ports when the spec exposes ports, and device
        allocations when it requests an accelerator. A Ready-but-incomplete row
        still needs reconciling, so phase ``Ready`` alone is not enough.
        """
        if self._phase() != GPUInstancePhase.READY:
            return False
        fields: List[str] = []
        if self.spec.ports:
            fields.extend(["access_addresses", "ports"])
        if self.spec.resources and self.spec.resources.accelerator:
            fields.append("allocations")
        return all(getattr(self.status, field) for field in fields)

    def is_starting(self) -> bool:
        return self._phase() == GPUInstancePhase.STARTING

    def is_stopping(self) -> bool:
        return self._phase() == GPUInstancePhase.STOPPING

    def is_stopped(self) -> bool:
        return self._phase() == GPUInstancePhase.STOPPED

    def is_deleting(self) -> bool:
        return self._phase() == GPUInstancePhase.DELETING

    def is_failed(self) -> bool:
        """Any failure phase, including worker-side ones outside this enum.

        The worker CR may report failure phases GPUStack does not define, so
        this uses the broad ``endswith("Failed")`` rather than membership in
        :data:`FAILED_PHASES` (which enumerates only the GPUStack-defined
        failure phases for user-action gating).
        """
        phase = self._phase()
        return phase is not None and phase.endswith("Failed")

    def is_transitioning(self) -> bool:
        """Whether the instance is mid-transition and still needs re-polling.

        Settled states (ready, stopped, failed) are done; everything else —
        pre-create (``phase is None``), starting, stopping, deleting, unknown,
        not-ready, and Ready-but-incomplete — is still reconciling.
        """
        return not (self.is_ready() or self.is_stopped() or self.is_failed())

    # -- downstream CR (de)serialization ----------------------------------- #

    def convert_to_kuberes(self) -> dict:
        """Full ``worker.gpustack.ai/v1`` Instance CR body: ``metadata`` + ``spec``.

        The ops layer (:class:`ClusterOps`) fills the cluster-plumbing bits the
        schema can't know — ``apiVersion`` / ``kind`` / ``metadata.namespace``.

        Field names follow the Go CRD's camelCase via the ``alias_generator``
        on :class:`GPUInstanceSpec`, so ``model_dump(by_alias=True)`` produces
        the on-wire keys directly. Spec transforms (identical to the former
        ``cluster_apis_util.spec_instance``):

        1. ``displayName`` / ``description`` live on the row in Python but on
           ``InstanceSpec`` in Go, so they are hoisted into the spec dict.
        2. ``volume.persistentTemplate`` collapses into ``volume.persistent``
           (name only; ``spec`` / ``releaseWithInstance`` drive server-side
           provisioning and are not part of the CRD).
        3. ``sshPublicKeys`` (list) collapses into ``sshPublicKey`` (singular
           ``LocalObjectReference`` named after the instance).
        """
        spec = self.spec.model_dump(by_alias=True, exclude_none=True)

        if self.display_name is not None:
            spec["displayName"] = self.display_name
        if self.description is not None:
            spec["description"] = self.description

        volume = spec.get("volume")
        if volume is not None:
            tmpl = volume.pop("persistentTemplate", None)
            if tmpl is not None:
                volume["persistent"] = {"name": tmpl["name"]}

        spec.pop("sshPublicKeys", None)
        spec["sshPublicKey"] = {"name": self.name}

        return {
            "metadata": {
                "name": self.name,
                "labels": {KUBERES_INSTANCE_ID_LABEL: str(self.id)},
            },
            "spec": spec,
        }

    def merge_from_kuberes(self, downstream: dict) -> GPUInstanceStatus:
        """Map a downstream worker CR dict into a :class:`GPUInstanceStatus`.

        Pure merge — **no** concurrency guards (DELETING sticky,
        ``session.refresh``) and no mutation of ``self``; those stay in the
        controller's write path (``_write_status``).

        The k8s ``namespace`` is taken from the CR's ``metadata.namespace``
        (authoritative), so callers don't stamp it separately.
        """
        payload = dict(downstream.get("status") or {})
        namespace = (downstream.get("metadata") or {}).get("namespace")
        if namespace is not None:
            payload["namespace"] = namespace
        return GPUInstanceStatus.model_validate(payload)


class GPUInstanceUpdate(GPUInstanceBase):
    """
    Represents the data that can be updated for a GPU instance.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    spec: Optional[GPUInstanceSpecUpdate] = None
    """
    Optional updated specification for the GPU instance.
    """


class GPUInstanceCreate(GPUInstanceUpdate):
    """
    Represents the data required to create a GPU instance.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    name: str
    """
    Created name of the GPU instance.
    Must be unique in the scope of the owning principal.
    """

    spec: GPUInstanceSpec
    """
    Specification for the GPU instance, including container image, resources, volumes, etc.
    """

    cluster_id: Optional[int] = None
    """
    Reference to the cluster where the GPU instance should be created.
    """


class GPUInstancePublic(GPUInstanceCreate, PublicFields):
    """
    Represents the public view of a GPU instance,
    containing only fields that are safe to expose to clients.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    creator_id: Optional[int] = None
    """
    Reference to the principal who created the GPU instance.
    """

    status: Optional[GPUInstanceStatus] = None
    """
    Status of the GPU instance, including phase, host IPs, ports, etc.
    """


class GPUInstanceListParams(ListParams):
    sortable_fields: ClassVar[List[str]] = [
        "id",
        "name",
        "cluster_id",
        "created_at",
        "updated_at",
    ]


GPUInstancesPublic = PaginatedList[GPUInstancePublic]
