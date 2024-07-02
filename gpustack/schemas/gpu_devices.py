from pydantic import BaseModel
from sqlmodel import SQLModel

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import PaginatedList
from gpustack.schemas.workers import GPUDeviceInfo

# GPUDevice is a view created from the `workers` table.


class GPUDeviceBase(GPUDeviceInfo, BaseModel):
    id: str
    worker_id: int
    worker_name: str
    worker_ip: str


class GPUDevice(GPUDeviceBase, SQLModel, BaseModelMixin, table=True):
    __tablename__ = "gpu_devices_view"
    __mapper_args__ = {'primary_key': ["id"]}


class GPUDevicePublic(GPUDeviceBase):
    pass


GPUDevicesPublic = PaginatedList[GPUDevicePublic]
