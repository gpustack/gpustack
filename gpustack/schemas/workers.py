from datetime import datetime
from enum import Enum
from typing import Dict, Optional
from pydantic import ConfigDict, BaseModel
from sqlmodel import Field, SQLModel, JSON, Column
from sqlalchemy import DDL, event

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import PaginatedList, pydantic_column_type
from typing import List
from sqlalchemy.orm import declarative_base
from gpustack.schemas.stmt import (
    worker_after_create_view_stmt,
    worker_after_drop_view_stmt,
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


class SwapInfo(UtilizationInfo):
    used: Optional[int] = Field(default=None)
    pass


class GPUDeviceInfo(BaseModel):
    name: str = Field(default="")
    uuid: Optional[str] = Field(default="")
    vendor: Optional[str] = Field(default="")
    index: Optional[int] = Field(default=None)
    core: Optional[GPUCoreInfo] = Field(sa_column=Column(JSON), default=None)
    memory: Optional[MemoryInfo] = Field(sa_column=Column(JSON), default=None)
    temperature: Optional[float] = Field(default=None)  # in celsius


GPUDevicesInfo = List[GPUDeviceInfo]


class VendorEnum(str, Enum):
    NVIDIA = "NVIDIA"
    AMD = "AMD"
    Intel = "Intel"
    MTHREADS = "MTHREADS"
    Apple = "Apple"


class MountPoint(BaseModel):
    name: str = Field(default="")
    mount_point: str = Field(default="")
    mount_from: str = Field(default="")
    total: int = Field(default=None)  # in bytes
    used: int = Field(default=None)
    free: int = Field(default=None)
    available: int = Field(default=None)


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
    memory: int = Field(default=None)
    gpu_memory: int = Field(default=None)


class WorkerStateEnum(str, Enum):
    NOT_READY = "not_ready"
    READY = "ready"


class WorkerStatus(BaseModel):
    cpu: Optional[CPUInfo] = Field(sa_column=Column(JSON), default=None)
    memory: Optional[MemoryInfo] = Field(sa_column=Column(JSON), default=None)
    gpu_devices: Optional[GPUDevicesInfo] = Field(sa_column=Column(JSON), default=None)
    swap: Optional[SwapInfo] = Field(sa_column=Column(JSON), default=None)
    filesystem: Optional[FileSystemInfo] = Field(sa_column=Column(JSON), default=None)
    os: Optional[OperatingSystemInfo] = Field(sa_column=Column(JSON), default=None)
    kernel: Optional[KernelInfo] = Field(sa_column=Column(JSON), default=None)
    uptime: Optional[UptimeInfo] = Field(sa_column=Column(JSON), default=None)

    model_config = ConfigDict(from_attributes=True)


class WorkerBase(SQLModel):
    name: str = Field(index=True, unique=True)
    hostname: str
    ip: str
    labels: Dict[str, str] = Field(sa_column=Column(JSON), default={})

    system_reserved: Optional[SystemReserved] = Field(
        sa_column=Column(pydantic_column_type(SystemReserved))
    )
    state: WorkerStateEnum = WorkerStateEnum.NOT_READY
    state_message: Optional[str] = None
    status: Optional[WorkerStatus] = Field(
        sa_column=Column(pydantic_column_type(WorkerStatus))
    )


class Worker(WorkerBase, BaseModelMixin, table=True):
    __tablename__ = 'workers'
    id: Optional[int] = Field(default=None, primary_key=True)


class WorkerCreate(WorkerBase):
    pass


class WorkerUpdate(WorkerBase):
    pass


class WorkerPublic(
    WorkerBase,
):
    id: int
    created_at: datetime
    updated_at: datetime


WorkersPublic = PaginatedList[WorkerPublic]


# add event listener
event.listen(Worker.metadata, "after_create", DDL(worker_after_drop_view_stmt))
event.listen(Worker.metadata, "after_create", DDL(worker_after_create_view_stmt))
