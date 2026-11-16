"""Finite, authorized observations for native model monitoring."""

from typing import Dict, List, Literal, Optional, Tuple

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    model_serializer,
    model_validator,
)

MetricStatistic = Literal['avg', 'p50', 'p95', 'p99', 'rate', 'current']
MetricStatus = Literal[
    'available',
    'no_data',
    'no_observations',
    'unsupported',
    'not_applicable',
    'query_error',
    'restricted',
    'stale',
]
MetricView = Literal['aggregate', 'instances']
MetricRole = Literal['all', 'prefill', 'decode', 'router']
MetricSection = Literal['all', 'llm', 'pd']
TrendStatistic = Literal['avg', 'p50', 'p95', 'p99']


class MonitoringCapabilities(BaseModel):
    min_query_seconds: int = 300
    max_query_seconds: int = 604800
    supports_custom_range: bool = True
    retention_seconds: Optional[int] = None
    retention_size_based: bool = False
    retention_known: bool = False
    retention_source: Literal[
        'prometheus_runtimeinfo', 'prometheus_flags', 'unknown'
    ] = 'unknown'


class ModelMonitoringCapabilities(MonitoringCapabilities):
    model_id: int


class MetricDimensions(BaseModel):
    model_config = ConfigDict(extra='forbid')

    role: Optional[str] = None
    model_instance_id: Optional[int] = None
    model_instance_name: Optional[str] = None
    source_name: Optional[str] = None
    endpoint_dp_rank: Optional[str] = None
    dp_rank: Optional[str] = None
    worker_id: Optional[int] = None
    device_id: Optional[str] = None
    mount_point: Optional[str] = None
    upstream_instance_id: Optional[int] = None

    @model_serializer(mode='wrap')
    def serialize_dimensions(self, handler):
        return {key: value for key, value in handler(self).items() if value is not None}


class MetricSeries(BaseModel):
    name: str
    statistic: MetricStatistic
    dimensions: MetricDimensions = Field(default_factory=MetricDimensions)
    points: List[Tuple[FiniteFloat, Optional[FiniteFloat]]] = Field(
        default_factory=list
    )
    lower_bounds: Optional[List[Tuple[FiniteFloat, FiniteFloat]]] = None

    @model_validator(mode='after')
    def validate_bounds(self):
        if not self.lower_bounds:
            return self
        points = dict(self.points)
        for timestamp, bound in self.lower_bounds or []:
            if self.statistic not in ('p50', 'p95', 'p99') or bound < 0:
                raise ValueError('Only quantiles may carry nonnegative lower bounds')
            if timestamp not in points or points[timestamp] is not None:
                raise ValueError('A censored point cannot also be an exact observation')
        return self

    @model_serializer(mode='wrap')
    def serialize_bounds(self, handler):
        result = handler(self)
        if not self.lower_bounds:
            result.pop('lower_bounds', None)
        return result


class MetricSummary(BaseModel):
    statistic: MetricStatistic
    dimensions: MetricDimensions = Field(default_factory=MetricDimensions)
    value: Optional[FiniteFloat]
    sample_count: Optional[FiniteFloat] = None
    lower_bound: Optional[FiniteFloat] = None

    @model_validator(mode='after')
    def validate_bound(self):
        if self.lower_bound is not None and (
            self.value is not None
            or self.lower_bound < 0
            or self.statistic not in ('p50', 'p95', 'p99')
        ):
            raise ValueError('A quantile lower bound cannot be an exact observation')
        return self

    @model_serializer(mode='wrap')
    def serialize_bound(self, handler):
        result = handler(self)
        if self.lower_bound is None:
            result.pop('lower_bound', None)
        return result


class MetricCoverage(BaseModel):
    expected_instances: int = Field(ge=0)
    observed_instances: int = Field(ge=0)
    partial: bool


class ModelMetric(BaseModel):
    kind: Literal['histogram', 'counter', 'gauge']
    unit: str
    measurement_scope: str
    status: MetricStatus = 'no_data'
    reason: Optional[str] = None
    supported_statistics: List[MetricStatistic] = Field(default_factory=list)
    series: List[MetricSeries] = Field(default_factory=list)
    summaries: List[MetricSummary] = Field(default_factory=list)
    coverage: Optional[MetricCoverage] = None


class ModelMetrics(BaseModel):
    model_id: int
    source: Literal['engine'] = 'engine'
    available: bool = False
    reason: Optional[str] = None
    start: FiniteFloat
    end: FiniteFloat
    step: int
    rate_window_seconds: int
    last_refreshed_at: FiniteFloat
    history_scope: str
    metrics: Dict[str, ModelMetric] = Field(default_factory=dict)


class ResourceDevice(BaseModel):
    id: str
    worker_id: int
    worker_name: str
    name: str
    index: int
    chip_index: Optional[int] = None
    scope: Literal['physical_gpu', 'partition']
    model_instance_ids: List[int] = Field(default_factory=list)
    history_available: bool
    reason: Optional[str] = None
    current: Dict[str, Optional[FiniteFloat]] = Field(default_factory=dict)
    last_reported_at: Optional[FiniteFloat] = None


class ResourceWorker(BaseModel):
    id: int
    name: str


class ModelResourceMetrics(BaseModel):
    model_id: int
    source: Literal['hardware'] = 'hardware'
    available: bool = False
    reason: Optional[str] = None
    start: FiniteFloat
    end: FiniteFloat
    step: int
    last_refreshed_at: FiniteFloat
    history_scope: str
    allocated_gpu_count: int = 0
    allocated_worker_count: int = 0
    devices: List[ResourceDevice] = Field(default_factory=list)
    workers: List[ResourceWorker] = Field(default_factory=list)
    all_devices: bool = False
    selected_device_id: Optional[str] = None
    metrics: Dict[str, ModelMetric] = Field(default_factory=dict)
