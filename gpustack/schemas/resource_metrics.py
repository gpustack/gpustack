"""Native hardware details, independent of deployment permissions."""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, FiniteFloat

from gpustack.schemas.model_metrics import ModelMetric, ResourceDevice
from gpustack.schemas.workers import WorkerPublic


class ResourcePlacement(BaseModel):
    worker_id: int
    worker_name: str
    worker_ip: Optional[str] = None
    gpu_indexes: List[int] = Field(default_factory=list)
    is_main: bool = False


class ResourceAssociation(BaseModel):
    id: int
    name: str
    model_id: int
    model_name: str
    state: str
    state_message: Optional[str] = None
    role: Optional[str] = None
    backend: Optional[str] = None
    backend_version: Optional[str] = None
    gpu_device_ids: List[str] = Field(default_factory=list)
    spans_workers: bool = False
    placements: List[ResourcePlacement] = Field(default_factory=list)
    placements_complete: bool = False


class ResourceDetail(BaseModel):
    worker: WorkerPublic
    can_edit: bool = False
    cluster_name: Optional[str] = None
    selected_device_id: Optional[str] = None
    devices: List[ResourceDevice] = Field(default_factory=list)
    instances: List[ResourceAssociation] = Field(default_factory=list)


class ResourceMetrics(BaseModel):
    worker_id: int
    selected_device_id: Optional[str] = None
    source: Literal['hardware'] = 'hardware'
    available: bool = False
    reason: Optional[str] = None
    start: FiniteFloat
    end: FiniteFloat
    step: int
    last_refreshed_at: FiniteFloat
    history_scope: str = 'owner_qualified_current_resource'
    devices: List[ResourceDevice] = Field(default_factory=list)
    metrics: Dict[str, ModelMetric] = Field(default_factory=dict)
