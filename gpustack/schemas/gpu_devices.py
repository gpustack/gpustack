from typing import ClassVar, List
from pydantic import BaseModel
from sqlmodel import SQLModel

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import ListParams, PaginatedList
from gpustack.schemas.workers import GPUDeviceInfo

# GPUDevice is a view created from the `workers` table.


class GPUDeviceBase(GPUDeviceInfo, BaseModel):
    id: str
    worker_id: int
    worker_name: str
    worker_ip: str
    worker_ifname: str
    cluster_id: int


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
    pass


GPUDevicesPublic = PaginatedList[GPUDevicePublic]
