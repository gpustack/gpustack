from datetime import datetime, timezone
from enum import Enum
from typing import ClassVar, Dict, Optional, Any
from pydantic import ConfigDict, BaseModel
from sqlmodel import (
    Field,
    SQLModel,
    JSON,
    Column,
    Text,
    Relationship,
    Integer,
    ForeignKey,
)
from sqlalchemy import String
from gpustack import envs
from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import (
    ListParams,
    PaginatedList,
    UTCDateTime,
    pydantic_column_type,
)
from typing import List
from sqlalchemy.orm import declarative_base

from gpustack.utils.network import is_offline
from .clusters import ClusterProvider, Cluster, WorkerPool
from gpustack.schemas.config import (
    PredefinedConfigNoDefaults,
    ModelInstanceProxyModeEnum,
)

Base = declarative_base()


class UtilizationInfo(BaseModel):
    total: int = Field(default=None)
    utilization_rate: Optional[float] = Field(default=None)  # rang from 0 to 100


class MemoryInfo(UtilizationInfo):
    is_unified_memory: bool = Field(default=False)
    used: Optional[int] = Field(default=None)
    allocated: Optional[int] = Field(default=None)


class CPUInfo(UtilizationInfo):
    pass


class GPUCoreInfo(UtilizationInfo):
    pass


class GPUNetworkInfo(BaseModel):
    status: str = Field(default="down")  # Network status (up/down)
    inet: str = Field(default="")  # IPv4 address
    netmask: str = Field(default="")  # Subnet mask
    mac: str = Field(default="")  # MAC address
    gateway: str = Field(default="")  # Default gateway
    iface: Optional[str] = Field(default=None)  # Network interface name
    mtu: Optional[int] = Field(default=None)  # Maximum Transmission Unit


class SwapInfo(UtilizationInfo):
    used: Optional[int] = Field(default=None)
    pass


class GPUDeviceInfo(BaseModel):
    vendor: Optional[str] = Field(default="")
    """
    Manufacturer of the GPU device, e.g. nvidia, amd, ascend, etc.
    """
    type: Optional[str] = Field(default="")
    """
    Device runtime backend type, e.g. cuda, rocm, cann, etc.
    """
    index: Optional[int] = Field(default=None)
    """
    GPU index, which is the logic ID of the GPU chip,
    which is a human-readable index and counted from 0 generally.
    It might be recognized as the GPU device ID in some cases, when there is no more than one GPU chip on the same card.
    """
    device_index: Optional[int] = Field(default=0)
    """
    GPU device index, which is the index of the onboard GPU device.
    In Linux, it can be retrieved under the /dev/ path.
    For example, /dev/nvidia0 (the first Nvidia card), /dev/davinci2(the third Ascend card), etc.
    """
    device_chip_index: Optional[int] = Field(default=0)
    """
    GPU device chip index, which is the index of the GPU chip on the card.
    It works with `device_index` to identify a GPU chip uniquely.
    For example, the first chip on the first card is 0, and the second chip on the first card is 1.
    """
    arch_family: Optional[str] = Field(default=None)
    """
    Architecture family of the GPU device.
    """
    name: str = Field(default="")
    """
    GPU name, e.g. NVIDIA A100-SXM4-40GB, NVIDIA RTX 3090, AMD MI100, Ascend 310P, etc.
    """
    uuid: Optional[str] = Field(default="")
    """
    UUID is a unique identifier assigned to each GPU device.
    """
    driver_version: Optional[str] = Field(default=None)
    """
    Driver version of the GPU device, e.g. for NVIDIA GPUs.
    """
    runtime_version: Optional[str] = Field(default=None)
    """
    Runtime version of the GPU device, e.g. CUDA version for NVIDIA GPUs.
    """
    compute_capability: Optional[str] = Field(default=None)
    """
    Compute compatibility version of the GPU device, e.g. for NVIDIA GPUs.
    """
    core: Optional[GPUCoreInfo] = Field(sa_column=Column(JSON), default=None)
    """
    Core information of the GPU device.
    """
    memory: Optional[MemoryInfo] = Field(sa_column=Column(JSON), default=None)
    """
    Memory information of the GPU device.
    """
    temperature: Optional[float] = Field(default=None)
    """
    Temperature of the GPU device in Celsius.
    """
    network: Optional[GPUNetworkInfo] = Field(sa_column=Column(JSON), default=None)
    """
    Network information of the GPU device, mainly for Ascend devices.
    """


GPUDevicesInfo = List[GPUDeviceInfo]


class MountPoint(BaseModel):
    name: str = Field(default="")
    mount_point: str = Field(default="")
    mount_from: str = Field(default="")
    total: int = Field(default=None)  # in bytes
    used: Optional[int] = Field(default=None)
    free: Optional[int] = Field(default=None)
    available: Optional[int] = Field(default=None)


FileSystemInfo = List[MountPoint]


class OperatingSystemInfo(BaseModel):
    name: str = Field(default="")
    version: str = Field(default="")


class KernelInfo(BaseModel):
    name: str = Field(default="")
    release: str = Field(default="")
    version: str = Field(default="")
    architecture: str = Field(default="")


class UptimeInfo(BaseModel):
    uptime: float = Field(default=None)  # in seconds
    boot_time: str = Field(default="")


class SystemReserved(BaseModel):
    ram: Optional[int] = Field(default=None)
    vram: Optional[int] = Field(default=None)


class RPCServer(BaseModel):
    pid: Optional[int] = None
    port: Optional[int] = None
    gpu_index: Optional[int] = None


class WorkerStateEnum(str, Enum):
    r"""
    Enum for Worker State

    State Transition Diagram:

    Phase 1: Provisioning Controller          |  Phase 2: Healthcheck Controller
    ------------------------------------------|------------------------------------
    PENDING > PROVISIONING > INITIALIZING > READY < -----|----------->  UNREACHABLE
                |             |         |      ^         |       (Worker Endpoint Unreachable)
                |             |         |      |         |
                |-------------|---------|------|         └----------->   NOT_READY
                \_____________________________/|                 (Worker Stop Posting Status)
                                   ERROR       | (Provisioning failed)       ^
                                     |         |        |                    |
                                     v         |        v                    |
                                 DELETING  <---┘ (provisioning end)          |
                                               |                             |
                                               |                             |
    Phase 3: Upgrade and Maintain              |                             |
    -------------------------------------------|-----------------------------|-----
                                               v                             |
                                           MAINTENANCE <---------------------┘
                                           (Back to Ready/Not Ready after maintenance completed)
    """

    NOT_READY = "not_ready"
    READY = "ready"
    UNREACHABLE = "unreachable"
    PENDING = "pending"
    PROVISIONING = "provisioning"
    INITIALIZING = "initializing"
    DELETING = "deleting"
    ERROR = "error"
    MAINTENANCE = "maintenance"

    @property
    def is_provisioning(self) -> bool:
        return self in [
            WorkerStateEnum.PENDING,
            WorkerStateEnum.PROVISIONING,
            WorkerStateEnum.INITIALIZING,
            WorkerStateEnum.DELETING,
            WorkerStateEnum.ERROR,
        ]


class SystemInfo(BaseModel):
    cpu: Optional[CPUInfo] = Field(sa_column=Column(JSON), default=None)
    memory: Optional[MemoryInfo] = Field(sa_column=Column(JSON), default=None)
    swap: Optional[SwapInfo] = Field(sa_column=Column(JSON), default=None)
    filesystem: Optional[FileSystemInfo] = Field(sa_column=Column(JSON), default=None)
    os: Optional[OperatingSystemInfo] = Field(sa_column=Column(JSON), default=None)
    kernel: Optional[KernelInfo] = Field(sa_column=Column(JSON), default=None)
    uptime: Optional[UptimeInfo] = Field(sa_column=Column(JSON), default=None)


class Maintenance(BaseModel):
    enabled: bool = False
    message: Optional[str] = None


class WorkerStatus(SystemInfo):
    """
    rpc_servers: Deprecated
    """

    gpu_devices: Optional[GPUDevicesInfo] = Field(sa_column=Column(JSON), default=None)
    rpc_servers: Optional[Dict[int, RPCServer]] = Field(
        sa_column=Column(JSON), default=None
    )

    model_config = ConfigDict(from_attributes=True)

    @classmethod
    def get_default_status(cls) -> 'WorkerStatus':
        return WorkerStatus(
            cpu=CPUInfo(total=0),
            memory=MemoryInfo(total=0, is_unified_memory=False),
            swap=SwapInfo(total=0),
            filesystem=[],
            os=OperatingSystemInfo(name="", version=""),
            kernel=KernelInfo(name="", release="", version="", architecture=""),
            uptime=UptimeInfo(uptime=0, boot_time=""),
            gpu_devices=[],
        )


class WorkerStatusStored(BaseModel):
    advertise_address: Optional[str] = None
    hostname: str
    ip: str
    ifname: str
    port: int
    metrics_port: Optional[int] = None

    system_reserved: Optional[SystemReserved] = Field(
        sa_column=Column(pydantic_column_type(SystemReserved))
    )
    state_message: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    status: Optional[WorkerStatus] = Field(
        sa_column=Column(pydantic_column_type(WorkerStatus))
    )

    worker_uuid: str = Field(sa_column=Column(String(255), nullable=False))
    machine_id: Optional[str] = Field(
        default=None
    )  # The machine ID of the worker, used for identifying the worker in the cluster

    proxy_mode: Optional[ModelInstanceProxyModeEnum] = Field(
        default=ModelInstanceProxyModeEnum.WORKER,
    )


class WorkerStatusPublic(WorkerStatusStored):
    gateway_endpoint: Optional[str] = None


class WorkerUpdate(SQLModel):
    """
    WorkerUpdate: updatable fields for Worker
    """

    name: str = Field(index=True, unique=True)
    labels: Dict[str, str] = Field(sa_column=Column(JSON), default={})
    maintenance: Optional[Maintenance] = Field(
        default=None,
        sa_column=Column(pydantic_column_type(Maintenance), default=None),
    )


class WorkerCreate(WorkerStatusStored, WorkerUpdate):
    cluster_id: Optional[int] = Field(
        sa_column=Column(Integer, ForeignKey("clusters.id"), nullable=False),
        default=None,
    )
    external_id: Optional[str] = Field(
        default=None, sa_column=Column(String(255), nullable=True)
    )


class WorkerBase(WorkerCreate):
    state: WorkerStateEnum = WorkerStateEnum.NOT_READY
    heartbeat_time: Optional[datetime] = Field(
        sa_column=Column(UTCDateTime), default=None
    )
    unreachable: bool = False

    def compute_state(self):
        if self.maintenance and self.maintenance.enabled:
            self.state = WorkerStateEnum.MAINTENANCE
            self.state_message = self.maintenance.message
            return

        if self.state.is_provisioning:
            return
        if self.state == WorkerStateEnum.NOT_READY and self.state_message is not None:
            return

        is_not_ready_flag, last_heartbeat_str = is_offline(
            self.heartbeat_time,
            envs.WORKER_HEARTBEAT_GRACE_PERIOD,
            datetime.now(timezone.utc),
        )

        if is_not_ready_flag:
            reschedule_minutes = envs.MODEL_INSTANCE_RESCHEDULE_GRACE_PERIOD / 60
            self.state = WorkerStateEnum.NOT_READY
            self.state_message = (
                f"Heartbeat lost (last heartbeat: {last_heartbeat_str}). "
                f"If the worker remains unresponsive for more than {reschedule_minutes:.1f} minutes, "
                "the instances on this worker will be rescheduled automatically. "
                "If this downtime is planned maintenance, please enable maintenance mode. "
                "Otherwise, please <a href='https://docs.gpustack.ai/latest/troubleshooting/#view-gpustack-logs'>check the worker logs</a>."
            )
            return

        if self.unreachable:
            address = self.advertise_address or self.ip
            healthz_url = f"http://{address}:{self.port}/healthz"
            msg = (
                "Server cannot access the "
                f"worker's health check endpoint at {healthz_url}. "
                "Please verify the port requirements in the "
                "<a href='https://docs.gpustack.ai/latest/installation/requirements/#port-requirements'>documentation</a>"
            )
            self.state = WorkerStateEnum.UNREACHABLE
            self.state_message = msg
            return

        self.state = WorkerStateEnum.READY
        self.state_message = None

    provider: ClusterProvider = Field(default=ClusterProvider.Docker)
    worker_pool_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("worker_pools.id"), nullable=True),
    )  # The worker pool this worker belongs to

    # Not setting foreign key to manage lifecycle
    ssh_key_id: Optional[int] = Field(
        default=None, sa_column=Column(Integer, nullable=True)
    )
    provider_config: Optional[Dict[str, Any]] = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )


class Worker(WorkerBase, BaseModelMixin, table=True):
    __tablename__ = 'workers'
    id: Optional[int] = Field(default=None, primary_key=True)

    cluster: Cluster = Relationship(
        back_populates="cluster_workers", sa_relationship_kwargs={"lazy": "selectin"}
    )
    worker_pool: Optional[WorkerPool] = Relationship(
        back_populates="pool_workers", sa_relationship_kwargs={"lazy": "selectin"}
    )

    # This field should be replaced by x509 credential if mTLS is supported.
    token: Optional[str] = Field(default=None, nullable=True)

    @property
    def provision_progress(self) -> Optional[str]:
        """
        The provisioning progress should have following steps:
        1. create_ssh_key
        2. create_instance with created ssh_key
        3. wait_for_started
        4. wait_for_public_ip
        5. [optional] create_volumes_and_attach
        """
        if self.state == WorkerStateEnum.INITIALIZING:
            return "5/5"
        if (
            self.state != WorkerStateEnum.PROVISIONING
            and self.state != WorkerStateEnum.PENDING
        ):
            return None
        format = "{}/{}"
        total = 5
        current = sum(
            [
                self.state == WorkerStateEnum.PROVISIONING,
                self.ssh_key_id is not None,
                self.external_id is not None,
                self.ip is not None and self.ip != "",
                "volume_ids" in (self.provider_config or {}),
            ]
        )
        return format.format(current, total)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if super().__eq__(other) and isinstance(other, Worker):
            return self.id == other.id
        return False


class WorkerListParams(ListParams):
    sortable_fields: ClassVar[List[str]] = [
        "name",
        "state",
        "ip",
        "status.cpu.utilization_rate",
        "status.memory.utilization_rate",
        "gpus",  # gpu count, the same naming pattern as in Clusters
        "created_at",
        "updated_at",
    ]


class WorkerPublic(
    WorkerBase,
):
    id: int
    created_at: datetime
    updated_at: datetime
    me: Optional[bool] = None  # Indicates if the worker is the current worker
    provision_progress: Optional[str] = None  # Indicates the provisioning progress

    worker_uuid: Optional[str] = Field(default=None, exclude=True)
    machine_id: Optional[str] = Field(default=None, exclude=True)


class WorkerRegistrationPublic(WorkerPublic):
    token: str
    worker_config: Optional["PredefinedConfigNoDefaults"] = None


WorkersPublic = PaginatedList[WorkerPublic]
