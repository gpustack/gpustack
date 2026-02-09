from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, ClassVar, Dict, List, Optional
from pydantic import BaseModel
from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel, Text

from gpustack.schemas.common import (
    ListParams,
    PaginatedList,
    pydantic_column_type,
)
from gpustack.mixins import BaseModelMixin
from gpustack.schemas.models import (
    ComputedResourceClaim,
    ExtendedKVCacheConfig,
    SpeculativeConfig,
)

from gpustack.schemas.workers import GPUDeviceInfo, OperatingSystemInfo


DATASET_RANDOM = "Random"
DATASET_SHAREGPT = "ShareGPT"


class BenchmarkStateEnum(str, Enum):
    r"""
    Enum for Benchmark State

    Transitions:

       |- - Server - -|- - - - - - - Worker - - - - - - -|
       |              |                                  |
    PENDING ---> ---> ---> QUEUED ---> RUNNING ---> COMPLETED/STOPPED/ERROR
                              ^          ^
                              |          |
                              |----------|
                                         |
                                         |(Worker unreachable)
                                         v
                                     UNREACHABLE
    """

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"
    ERROR = "error"
    UNREACHABLE = "unreachable"

    def __str__(self):
        return self.value


class ModelInstanceRuntimeInfo(BaseModel):
    computed_resource_claim: Optional[ComputedResourceClaim]
    ports: Optional[List[int]]

    worker_id: Optional[int] = None
    worker_name: Optional[str] = None
    worker_ip: Optional[str] = None
    gpu_type: Optional[str] = None
    gpu_indexes: Optional[List[int]] = None
    gpu_ids: Optional[List[str]] = None


class ModelInstanceSnapshot(ModelInstanceRuntimeInfo):
    id: int
    name: str
    resolved_path: Optional[str] = None

    # resource info
    state: Optional[str] = None
    state_message: Optional[str] = None

    # backend info
    backend: Optional[str] = None
    backend_version: Optional[str] = None
    api_detected_backend_version: Optional[str] = None
    backend_parameters: Optional[List[str]] = Field(sa_type=JSON, default=None)
    image_name: Optional[str] = None
    run_command: Optional[str] = Field(sa_type=Text, default=None)
    env: Optional[Dict[str, str]] = Field(sa_type=JSON, default=None)

    # Extended KV Cache configuration. Currently maps to LMCache config in vLLM and SGLang.
    extended_kv_cache: Optional[ExtendedKVCacheConfig] = Field(
        sa_type=pydantic_column_type(ExtendedKVCacheConfig), default=None
    )

    speculative_config: Optional[SpeculativeConfig] = Field(
        sa_type=pydantic_column_type(SpeculativeConfig), default=None
    )

    # subordinate workers info
    subordinate_workers: Optional[List[ModelInstanceRuntimeInfo]] = None


class WorkerSnapshot(BaseModel):
    id: int
    name: str
    cpu_total: Optional[int] = None
    memory_total: Optional[int] = None
    os: Optional[OperatingSystemInfo] = None


class GPUSnapshot(GPUDeviceInfo):
    id: str
    worker_id: int
    worker_name: str
    memory_total: Optional[int] = None
    core_total: Optional[int] = None


@dataclass
class BenchmarkDeploymentMetadata:
    name: str
    labels: dict[str, str]


class BenchmarkBase(SQLModel):
    name: str = Field(index=True, unique=True)
    description: Optional[str] = Field(
        sa_type=Text,
        nullable=True,
        default=None,
    )

    profile: Optional[str] = Field(default="Custom")
    dataset_name: Optional[str] = Field(
        default=None
    )  # denormalized field for easier query
    dataset_input_tokens: Optional[int] = Field(default=None)
    dataset_output_tokens: Optional[int] = Field(default=None)
    dataset_seed: Optional[int] = Field(default=42)

    cluster_id: int = Field(default=None)
    model_id: Optional[int] = Field(default=None)
    model_name: Optional[str] = Field(
        default=None
    )  # denormalized field for easier query
    model_instance_name: str

    request_rate: int = Field(default=10)  # requests per second
    total_requests: Optional[int] = Field(
        default=None
    )  # total number of requests to send

    # Benchmark state fields
    state: BenchmarkStateEnum = Field(
        default=BenchmarkStateEnum.PENDING,
        index=True,
    )
    state_message: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    progress: Optional[float] = Field(default=None)
    worker_id: Optional[int] = Field(default=None)
    pid: Optional[int] = Field(default=None)

    def get_deployment_metadata(
        self,
    ) -> Optional[BenchmarkDeploymentMetadata]:
        """
        Get the deployment metadata for the benchmark.
        """

        return BenchmarkDeploymentMetadata(
            name=self.name,
            labels={
                "benchmark-name": self.name,
                "model-instance-name": self.model_instance_name or "",
                "type": "benchmark",
            },
        )


ModelInstanceSnapshots = Dict[str, ModelInstanceSnapshot]
WorkerSnapshots = Dict[str, WorkerSnapshot]
GPUSnapshots = Dict[str, GPUSnapshot]


class BenchmarkSnapshot(BaseModel):
    instances: Optional[ModelInstanceSnapshots] = None
    workers: Optional[WorkerSnapshots] = None
    gpus: Optional[GPUSnapshots] = None


class BenchmarkMetricsLite(SQLModel):
    requests_per_second_mean: Optional[float] = Field(
        default=None, description="Mean requests per second (unit: req/s)"
    )
    request_latency_mean: Optional[float] = Field(
        default=None, description="Mean request latency (unit: seconds)"
    )
    time_per_output_token_mean: Optional[float] = Field(
        default=None, description="Mean time per output token (unit: ms)"
    )
    inter_token_latency_mean: Optional[float] = Field(
        default=None, description="Mean inter-token latency (unit: ms)"
    )
    time_to_first_token_mean: Optional[float] = Field(
        default=None, description="Mean time to first token (unit: ms)"
    )
    tokens_per_second_mean: Optional[float] = Field(
        default=None, description="Mean tokens per second (unit: tok/s)"
    )
    output_tokens_per_second_mean: Optional[float] = Field(
        default=None, description="Mean output tokens per second (unit: tok/s)"
    )
    input_tokens_per_second_mean: Optional[float] = Field(
        default=None, description="Mean prompt tokens per second (unit: tok/s)"
    )
    request_concurrency_mean: Optional[float] = Field(
        default=None,
        description="Mean request concurrency (unit: number of concurrent requests)",
    )
    request_concurrency_max: Optional[float] = Field(
        default=None,
        description="Max request concurrency (unit: number of concurrent requests)",
    )
    request_total: Optional[int] = Field(
        default=None, description="Total number of requests made"
    )
    request_successful: Optional[int] = Field(
        default=None, description="Total number of successful requests"
    )
    request_errored: Optional[int] = Field(
        default=None, description="Total number of errored requests"
    )
    request_incomplete: Optional[int] = Field(
        default=None, description="Total number of incomplete requests"
    )


class BenchmarkMetrics(BenchmarkMetricsLite):
    raw_metrics: Optional[Dict[str, Any]] = Field(
        sa_column=Column(JSON), default=None
    )  # deferred loading of potentially large field


class BenchmarkWithSnapshots(BenchmarkBase):
    snapshot: Optional[BenchmarkSnapshot] = Field(
        default=None,
        sa_column=Column(pydantic_column_type(BenchmarkSnapshot)),
    )
    gpu_summary: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    gpu_vendor_summary: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )


class Benchmark(BenchmarkWithSnapshots, BenchmarkMetrics, BaseModelMixin, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

    __tablename__ = 'benchmarks'


class BenchmarkListParams(ListParams):
    sortable_fields: ClassVar[List[str]] = [
        "name",
        "dataset_name",
        "model_name",
        "state",
        "created_at",
        "updated_at",
        # metrics fields
        "requests_per_second_mean",
        "request_latency_mean",
        "time_per_output_token_mean",
        "inter_token_latency_mean",
        "time_to_first_token_mean",
        "tokens_per_second_mean",
        "output_tokens_per_second_mean",
        "input_tokens_per_second_mean",
        "request_concurrency_mean",
        "request_concurrency_max",
        "request_total",
        "request_successful",
        "request_errored",
        "request_incomplete",
    ]


class BenchmarkCreate(BenchmarkBase):
    pass


class BenchmarkUpdate(SQLModel):
    name: str = Field(index=True, unique=True)
    description: Optional[str] = Field(
        sa_type=Text,
        nullable=True,
        default=None,
    )


class BenchmarkStateUpdate(SQLModel):
    state: Optional[BenchmarkStateEnum] = None
    state_message: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    pid: Optional[int] = Field(default=None)
    progress: Optional[float] = None


class BenchmarkFullPublic(
    BenchmarkWithSnapshots,
    BenchmarkMetrics,
):
    id: int
    created_at: datetime
    updated_at: datetime

    gpu_summary: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    gpu_vendor_summary: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )


class BenchmarkPublic(
    BenchmarkWithSnapshots,
    BenchmarkMetricsLite,
):
    id: int
    created_at: datetime
    updated_at: datetime

    gpu_summary: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    gpu_vendor_summary: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )


BenchmarksPublic = PaginatedList[BenchmarkPublic]
