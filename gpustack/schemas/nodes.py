from datetime import datetime
from typing import Dict
from pydantic import field_validator, BaseModel
from sqlmodel import Field, SQLModel, JSON, Column

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import PaginatedList
from typing import Any, List


class MemoryInfo(BaseModel):
    total: float = Field(default=-1)  # in bytes
    used: float = Field(default=-1)
    allocated: float = Field(default=-1)


class SwapInfo(BaseModel):
    total: float = Field(default=-1)  # in bytes
    used: float = Field(default=-1)


class CPUInfo(BaseModel):
    total: float = Field(default=-1)  # cores
    allocated: float = Field(default=-1)  # cores
    utilization_rate: float = Field(default=-1)


class GPUDevice(BaseModel):
    uuid: str = Field(default="")
    name: str = Field(default="")
    vendor: str = Field(default="")
    index: int = Field(default=-1)
    core_total: int = Field(default=-1)
    core_allocated: int = Field(default=-1)
    core_utilization_rate: float = Field(default=-1)
    memory_total: float = Field(default=-1)  # in bytes
    memory_allocated: float = Field(default=-1)
    memory_used: float = Field(default=-1)
    temperature: float = Field(default=-1)  # in celsius


GPUInfo = List[GPUDevice]


class MountPoint(BaseModel):
    name: str = Field(default="")
    mount_point: str = Field(default="")
    mount_from: str = Field(default="")
    total: float = Field(default=-1)  # in bytes
    used: float = Field(default=-1)
    free: float = Field(default=-1)
    available: float = Field(default=-1)


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
    uptime: float = Field(default=-1)  # in seconds
    boot_time: str = Field(default="")


class NodeStatus(BaseModel):
    cpu: CPUInfo | None = Field(sa_column=Column(JSON), default=None)
    memory: MemoryInfo | None = Field(sa_column=Column(JSON), default=None)
    gpu: GPUInfo | None = Field(sa_column=Column(JSON), default=None)
    swap: SwapInfo | None = Field(sa_column=Column(JSON), default=None)
    filesystem: FileSystemInfo | None = Field(sa_column=Column(JSON), default=None)
    os: OperatingSystemInfo | None = Field(sa_column=Column(JSON), default=None)
    kernel: KernelInfo | None = Field(sa_column=Column(JSON), default=None)
    uptime: UptimeInfo | None = Field(sa_column=Column(JSON), default=None)
    state: str | None = Field(default="unknown")


class NodeBase(SQLModel):
    name: str = Field(index=True, unique=True)
    hostname: str
    address: str
    labels: Dict[str, str] = Field(sa_column=Column(JSON), default={})
    status: NodeStatus | None = Field(
        sa_column=Column(JSON))

    # Workaround for https://github.com/tiangolo/sqlmodel/issues/63
    # It generates a warning.
    # TODO Find a better way.
    @field_validator("status")
    def validate_node_status(cls, val: NodeStatus):
        if val is None:
            empty_dict: Dict[str, Any] = {}
            return empty_dict
        return val.model_dump()


class Node(NodeBase, BaseModelMixin, table=True):
    id: int | None = Field(default=None, primary_key=True)


class NodeCreate(NodeBase):
    pass


class NodeUpdate(NodeBase):
    pass


class NodePublic(
    NodeBase,
):
    id: int
    created_at: datetime
    updated_at: datetime


NodesPublic = PaginatedList[NodePublic]
