from typing import List, Optional
from pydantic import BaseModel, ConfigDict


class TimeSeriesData(BaseModel):
    timestamp: int
    value: float


class CurrentSystemLoad(BaseModel):
    cpu: float
    ram: float
    gpu: float
    vram: float


class HistorySystemLoad(BaseModel):
    cpu: List[TimeSeriesData]
    ram: List[TimeSeriesData]
    gpu: List[TimeSeriesData]
    vram: List[TimeSeriesData]


class SystemLoadSummary(BaseModel):
    current: CurrentSystemLoad
    history: HistorySystemLoad


class ModelUsageUserSummary(BaseModel):
    user_id: int
    username: str
    prompt_token_count: int
    completion_token_count: int


class ModelUsageStats(BaseModel):
    api_request_history: List[TimeSeriesData]
    completion_token_history: List[TimeSeriesData]
    prompt_token_history: List[TimeSeriesData]


class ModelUsageSummary(ModelUsageStats):
    top_users: Optional[List[ModelUsageUserSummary]] = None


class ResourceClaim(BaseModel):
    ram: int  # in bytes
    vram: int  # in bytes


class ModelSummary(BaseModel):
    id: Optional[int] = None
    provider_id: Optional[int] = None
    name: str
    resource_claim: Optional[ResourceClaim] = None
    instance_count: int
    token_count: int
    categories: Optional[List[str]] = None


class ResourceCounts(BaseModel):
    worker_count: int
    gpu_count: int
    model_count: int
    model_instance_count: int
    cluster_count: Optional[int] = None

    model_config = ConfigDict(protected_namespaces=())


class SystemSummary(BaseModel):
    cluster_id: Optional[int] = None
    resource_counts: ResourceCounts
    system_load: SystemLoadSummary
    model_usage: ModelUsageSummary
    active_models: List[ModelSummary]

    model_config = ConfigDict(protected_namespaces=())
