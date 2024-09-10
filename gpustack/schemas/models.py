from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, ConfigDict, model_validator
from sqlalchemy import JSON, Column
from sqlmodel import Field, Relationship, SQLModel

from gpustack.schemas.common import PaginatedList, pydantic_column_type
from gpustack.mixins import BaseModelMixin
from gpustack.schemas.workers import RPCServer

# Models


class SourceEnum(str, Enum):
    HUGGING_FACE = "huggingface"
    OLLAMA_LIBRARY = "ollama_library"


class PlacementStrategyEnum(str, Enum):
    SPREAD = "spread"
    BINPACK = "binpack"


class GPUSelector(BaseModel):
    worker_name: str
    gpu_index: int
    gpu_name: Optional[str] = None


class ModelSource(BaseModel):
    source: SourceEnum
    huggingface_repo_id: Optional[str] = None
    huggingface_filename: Optional[str] = None
    ollama_library_model_name: Optional[str] = None

    @model_validator(mode="after")
    def check_huggingface_fields(self):
        if self.source == SourceEnum.HUGGING_FACE:
            if not self.huggingface_repo_id or not self.huggingface_filename:
                raise ValueError(
                    "huggingface_repo_id and huggingface_filename must be provided "
                    "when source is 'huggingface'"
                )
        if self.source == SourceEnum.OLLAMA_LIBRARY:
            if not self.ollama_library_model_name:
                raise ValueError(
                    "ollama_library_model_name must be provided when source is 'ollama_library'"
                )
        return self


class ModelBase(SQLModel, ModelSource):
    name: str = Field(index=True, unique=True)
    description: Optional[str] = None

    replicas: int = Field(default=1, ge=0)
    ready_replicas: int = Field(default=0, ge=0)
    embedding_only: bool = False
    placement_strategy: PlacementStrategyEnum = PlacementStrategyEnum.SPREAD
    cpu_offloading: bool = False
    distributed_inference_across_workers: bool = False
    worker_selector: Optional[Dict[str, str]] = Field(
        sa_column=Column(JSON), default={}
    )
    gpu_selector: Optional[GPUSelector] = Field(
        sa_column=Column(pydantic_column_type(GPUSelector)), default=None
    )


class Model(ModelBase, BaseModelMixin, table=True):
    __tablename__ = 'models'
    id: Optional[int] = Field(default=None, primary_key=True)

    instances: list["ModelInstance"] = Relationship(
        sa_relationship_kwargs={"cascade": "delete", "lazy": "selectin"},
        back_populates="model",
    )


class ModelCreate(ModelBase):
    pass


class ModelUpdate(ModelBase):
    pass


class ModelPublic(
    ModelBase,
):
    id: int
    created_at: datetime
    updated_at: datetime


ModelsPublic = PaginatedList[ModelPublic]


# Model Instances


class ModelInstanceStateEnum(str, Enum):
    INITIALIZING = "initializing"
    PENDING = "pending"
    RUNNING = "running"
    SCHEDULED = "scheduled"
    ERROR = "error"
    DOWNLOADING = "downloading"
    ANALYZING = "analyzing"


class ComputedResourceClaim(BaseModel):
    is_unified_memory: Optional[bool] = False
    offload_layers: Optional[int] = None
    total_layers: Optional[int] = None
    ram: Optional[int] = Field(default=None)  # in bytes
    vram: Optional[Dict[int, int]] = Field(default=None)  # in bytes


class ModelInstanceRPCServer(RPCServer):
    worker_id: Optional[int] = None
    computed_resource_claim: Optional[ComputedResourceClaim] = Field(
        sa_column=Column(pydantic_column_type(ComputedResourceClaim)), default=None
    )


class DistributedServers(BaseModel):
    rpc_servers: Optional[List[ModelInstanceRPCServer]] = Field(
        sa_column=Column(JSON), default=[]
    )

    model_config = ConfigDict(from_attributes=True)


class ModelInstanceBase(SQLModel, ModelSource):
    name: str = Field(index=True, unique=True)
    worker_id: Optional[int] = None
    worker_name: Optional[str] = None
    worker_ip: Optional[str] = None
    pid: Optional[int] = None
    port: Optional[int] = None
    download_progress: Optional[float] = None
    state: ModelInstanceStateEnum = ModelInstanceStateEnum.PENDING
    state_message: Optional[str] = None
    computed_resource_claim: Optional[ComputedResourceClaim] = Field(
        sa_column=Column(pydantic_column_type(ComputedResourceClaim)), default=None
    )
    gpu_indexes: Optional[List[int]] = Field(sa_column=Column(JSON), default=[])

    model_id: int = Field(default=None, foreign_key="models.id")
    model_name: str

    distributed_servers: Optional[DistributedServers] = Field(
        sa_column=Column(pydantic_column_type(DistributedServers)), default=None
    )
    # The "model_id" field conflicts with the protected namespace "model_" in Pydantic.
    # Disable it given that it's not a real issue for this particular field.
    model_config = ConfigDict(protected_namespaces=())


class ModelInstance(ModelInstanceBase, BaseModelMixin, table=True):
    __tablename__ = 'model_instances'
    id: Optional[int] = Field(default=None, primary_key=True)

    model: Optional[Model] = Relationship(
        back_populates="instances",
        sa_relationship_kwargs={"lazy": "selectin"},
    )


class ModelInstanceCreate(ModelInstanceBase):
    pass


class ModelInstanceUpdate(ModelInstanceBase):
    pass


class ModelInstancePublic(
    ModelInstanceBase,
):
    id: int
    created_at: datetime
    updated_at: datetime


ModelInstancesPublic = PaginatedList[ModelInstancePublic]
