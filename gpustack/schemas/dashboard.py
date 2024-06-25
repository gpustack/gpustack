from typing import List
from pydantic import BaseModel, ConfigDict

from gpustack.schemas.workers import UtilizationInfo


class TimeSeriesData(BaseModel):
    timestamp: int
    value: float


class CurrentSystemLoad(BaseModel):
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


class ModelSummary(BaseModel):
    id: int
    name: str
    gpu_utilization: float
    gpu_memory_utilization: float
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
