from typing import ClassVar, List, Optional
from pydantic import BaseModel, ConfigDict
from sqlmodel import SQLModel, Column, Field

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import ListParams, PaginatedList, pydantic_column_type
from gpustack.schemas.workers import (
    GPUCoreInfo,
    GPUDeviceStatus,
    GPUNetworkInfo,
    MemoryInfo,
)

# GPUDevice is a view created from the `workers` table.


class GPUDeviceBase(GPUDeviceStatus, BaseModel):
    # Each row of this view is projected out of the workers.status JSON, and
    # the driver hands those columns back as plain dicts. Declaring them with
    # pydantic_column_type validates each row once at the load boundary, so
    # every reader of the view holds the declared models and serializes typed
    # values instead of dicts.
    core: Optional[GPUCoreInfo] = Field(
        default=None, sa_column=Column(pydantic_column_type(GPUCoreInfo))
    )
    memory: Optional[MemoryInfo] = Field(
        default=None, sa_column=Column(pydantic_column_type(MemoryInfo))
    )
    network: Optional[GPUNetworkInfo] = Field(
        default=None, sa_column=Column(pydantic_column_type(GPUNetworkInfo))
    )
    id: str
    worker_id: int
    worker_name: str
    worker_ip: str
    worker_ifname: str
    cluster_id: int
    # NULL = belongs to a worker on a global cluster.
    owner_principal_id: Optional[int] = None


class GPUDevice(GPUDeviceBase, SQLModel, BaseModelMixin, table=True):
    __tablename__ = "gpu_devices_view"
    __mapper_args__ = {'primary_key': ["id"]}


class GPUDeviceListParams(ListParams):
    sortable_fields: ClassVar[List[str]] = [
        "name",
        "index",
        "cluster_id",
        "worker_name",
        "vendor",
        "temperature",
        "core.utilization_rate",
        "memory.utilization_rate",
        "created_at",
        "updated_at",
    ]


class GPUDevicePublic(GPUDeviceBase):
    # Unlike WorkerPublic (SQLModel-based, from_attributes=True), this class
    # is plain pydantic — without from_attributes the streaming path's
    # _convert_to_public_class(GPUDevice) raises ValidationError and kills
    # the watch stream on the first replayed event.
    model_config = ConfigDict(from_attributes=True)


GPUDevicesPublic = PaginatedList[GPUDevicePublic]
