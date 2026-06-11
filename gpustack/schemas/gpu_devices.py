from typing import ClassVar, List, Optional
from pydantic import BaseModel, ConfigDict
from sqlmodel import SQLModel

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import ListParams, PaginatedList
from gpustack.schemas.workers import GPUDeviceStatus

# GPUDevice is a view created from the `workers` table.


class GPUDeviceBase(GPUDeviceStatus, BaseModel):
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
