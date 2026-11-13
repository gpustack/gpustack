"""Measured Benchmark intervals and immutable run monitoring associations."""

from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, FiniteFloat

from gpustack.schemas.model_metrics import (
    MetricDimensions,
    ModelMetrics,
    ModelMonitoringCapabilities,
    ModelResourceMetrics,
    ResourceDevice,
    ResourceWorker,
)
from gpustack.schemas.models import DisaggregationSpec, RoleSpec


class BenchmarkRunMember(BaseModel):
    id: int
    name: str
    role: Optional[str] = None
    worker_id: Optional[int] = None
    restart_count: Optional[int] = None
    pid: Optional[int] = None
    worker_ip: Optional[str] = None
    port: Optional[int] = None
    gpu_indexes: List[int] = Field(default_factory=list)
    worker_ids: List[int] = Field(default_factory=list)
    source_identities: List[Dict[str, str]] = Field(default_factory=list)
    allocations: Dict[int, List[int]] = Field(default_factory=dict)
    source_addresses: List[Tuple[str, int]] = Field(default_factory=list)


class BenchmarkRunWorker(BaseModel):
    id: int
    name: str
    worker_uuid: Optional[str] = None
    owner_principal_id: Optional[int] = None


class BenchmarkRunDevice(BaseModel):
    id: str
    worker_id: int
    name: str
    index: int
    uuid: Optional[str] = None
    chip_index: int = 0
    scope: Literal['physical_gpu', 'partition'] = 'physical_gpu'
    model_instance_ids: List[int] = Field(default_factory=list)


class BenchmarkRunEvent(BaseModel):
    at: FiniteFloat
    kind: Literal['association_captured', 'generation_changed']
    model_instance_id: Optional[int] = None


class BenchmarkRunSnapshot(BaseModel):
    captured_at: FiniteFloat
    confirmed_at: Optional[FiniteFloat] = None
    model_id: int
    cluster_id: int
    owner_principal_id: Optional[int] = None
    backend: str
    roles: List[RoleSpec] = Field(default_factory=list)
    disaggregation: Optional[DisaggregationSpec] = None
    instances: List[BenchmarkRunMember] = Field(default_factory=list)
    workers: List[BenchmarkRunWorker] = Field(default_factory=list)
    devices: List[BenchmarkRunDevice] = Field(default_factory=list)
    events: List[BenchmarkRunEvent] = Field(default_factory=list)


class BenchmarkMonitoringWindow(BaseModel):
    start: FiniteFloat
    end: FiniteFloat


class BenchmarkMonitoringStage(BaseModel):
    id: str
    sequence: int
    input_tokens: Optional[int] = None
    rate: Optional[FiniteFloat] = None
    start: Optional[FiniteFloat] = None
    end: Optional[FiniteFloat] = None
    traffic_start: Optional[FiniteFloat] = None
    traffic_end: Optional[FiniteFloat] = None
    status: Literal['available', 'unavailable'] = 'unavailable'
    reason: Optional[str] = 'phase_timing_unavailable'


class BenchmarkMonitoringMember(BaseModel):
    id: int
    name: str
    role: Optional[str] = None
    worker_id: Optional[int] = None
    worker_ids: List[int] = Field(default_factory=list)
    gpu_indexes: List[int] = Field(default_factory=list)


class BenchmarkMonitoringContext(BaseModel):
    benchmark_id: int
    model_id: Optional[int] = None
    history_scope: Literal['benchmark_snapshot'] = 'benchmark_snapshot'
    available: bool = False
    reason: Optional[str] = None
    run_window: Optional[BenchmarkMonitoringWindow] = None
    stages: List[BenchmarkMonitoringStage] = Field(default_factory=list)
    instances: List[BenchmarkMonitoringMember] = Field(default_factory=list)
    workers: List[ResourceWorker] = Field(default_factory=list)
    devices: List[ResourceDevice] = Field(default_factory=list)
    events: List[BenchmarkRunEvent] = Field(default_factory=list)
    resource_status: Literal['available', 'restricted', 'unavailable'] = 'unavailable'
    resource_reason: Optional[str] = None
    capabilities: ModelMonitoringCapabilities


class BenchmarkSampleStatistics(BaseModel):
    dimensions: MetricDimensions = Field(default_factory=MetricDimensions)
    average: Optional[FiniteFloat] = None
    peak: Optional[FiniteFloat] = None
    peak_at: Optional[FiniteFloat] = None
    last: Optional[FiniteFloat] = None
    last_at: Optional[FiniteFloat] = None
    threshold: Optional[FiniteFloat] = None
    threshold_fraction: Optional[FiniteFloat] = None
    observed_samples: int = 0
    expected_samples: int = 0
    coverage: float = 0


class BenchmarkModelMetrics(ModelMetrics):
    model_id: Optional[int] = None
    start: Optional[FiniteFloat] = None
    end: Optional[FiniteFloat] = None
    phase_id: Optional[str] = None
    sample_statistics: Dict[str, List[BenchmarkSampleStatistics]] = Field(
        default_factory=dict
    )


class BenchmarkResourceMetrics(ModelResourceMetrics):
    model_id: Optional[int] = None
    start: Optional[FiniteFloat] = None
    end: Optional[FiniteFloat] = None
    phase_id: Optional[str] = None
    sample_statistics: Dict[str, List[BenchmarkSampleStatistics]] = Field(
        default_factory=dict
    )
