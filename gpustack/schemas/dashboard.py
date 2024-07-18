from typing import List, Optional
from pydantic import BaseModel, ConfigDict

from gpustack.schemas.workers import UtilizationInfo


class TimeSeriesData(BaseModel):
    timestamp: int
    value: float


class GPUUtilizationInfo(UtilizationInfo):
    worker_name: str
    index: int


class WorkerUtilizationInfo(UtilizationInfo):
    worker_name: str


class MaxMinUtilizationInfo:
    # worker with the max utilization rate
    max_cpu: Optional[WorkerUtilizationInfo] = None
    max_memory: Optional[WorkerUtilizationInfo] = None

    # gpu with the max utilization rate
    max_gpu: Optional[GPUUtilizationInfo] = None
    max_gpu_memory: Optional[GPUUtilizationInfo] = None

    # worker with the min utilization rate
    min_cpu: Optional[WorkerUtilizationInfo] = None
    min_memory: Optional[WorkerUtilizationInfo] = None

    # gpu with the min utilization rate
    min_gpu: Optional[GPUUtilizationInfo] = None
    min_gpu_memory: Optional[GPUUtilizationInfo] = None


class CurrentSystemLoad(BaseModel, MaxMinUtilizationInfo):
    cpu: UtilizationInfo
    memory: UtilizationInfo
    gpu: UtilizationInfo
    gpu_memory: UtilizationInfo


class HistorySystemLoad(BaseModel):
    cpu: List[TimeSeriesData]
    memory: List[TimeSeriesData]
    gpu: List[TimeSeriesData]
    gpu_memory: List[TimeSeriesData]


class SystemLoadSummary(BaseModel):
    current: CurrentSystemLoad
    history: HistorySystemLoad


class ModelUsageUserSummary(BaseModel):
    user_id: int
    username: str
    prompt_token_count: int
    completion_token_count: int


class ModelUsageSummary(BaseModel):
    api_request_history: List[TimeSeriesData]
    completion_token_history: List[TimeSeriesData]
    prompt_token_history: List[TimeSeriesData]
    top_users: List[ModelUsageUserSummary]


class ResourceClaim(BaseModel):
    memory: int  # in bytes
    gpu_memory: int  # in bytes


class ModelSummary(BaseModel):
    id: int
    name: str
    resource_claim: ResourceClaim
    instance_count: int
    token_count: int


class ResourceCounts(BaseModel):
    worker_count: int
    gpu_count: int
    model_count: int
    model_instance_count: int

    model_config = ConfigDict(protected_namespaces=())


class SystemSummary(BaseModel):
    resource_counts: ResourceCounts
    system_load: SystemLoadSummary
    model_usage: ModelUsageSummary
    active_models: List[ModelSummary]

    model_config = ConfigDict(protected_namespaces=())
