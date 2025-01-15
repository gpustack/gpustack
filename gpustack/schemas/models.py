from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, model_validator, Field as PydanticField
from sqlalchemy import JSON, Column
from sqlmodel import Field, Relationship, SQLModel

from gpustack.schemas.common import PaginatedList, pydantic_column_type
from gpustack.mixins import BaseModelMixin
from gpustack.schemas.workers import RPCServer
from gpustack.utils.command import find_parameter

# Models


class SourceEnum(str, Enum):
    HUGGING_FACE = "huggingface"
    OLLAMA_LIBRARY = "ollama_library"
    MODEL_SCOPE = "model_scope"
    LOCAL_PATH = "local_path"


class CategoryEnum(str, Enum):
    LLM = "llm"
    EMBEDDING = "embedding"
    IMAGE = "image"
    RERANKER = "reranker"
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech"


class PlacementStrategyEnum(str, Enum):
    SPREAD = "spread"
    BINPACK = "binpack"


class BackendEnum(str, Enum):
    LLAMA_BOX = "llama-box"
    VLLM = "vllm"
    VOX_BOX = "vox-box"


class GPUSelector(BaseModel):
    # format of each element: "worker_name:device:gpu_index", example: "worker1:cuda:0"
    gpu_ids: Optional[List[str]] = None


class ModelSource(BaseModel):
    source: SourceEnum
    huggingface_repo_id: Optional[str] = None
    huggingface_filename: Optional[str] = None
    ollama_library_model_name: Optional[str] = None
    model_scope_model_id: Optional[str] = None
    model_scope_file_path: Optional[str] = None
    local_path: Optional[str] = None

    @model_validator(mode="after")
    def check_huggingface_fields(self):
        if self.source == SourceEnum.HUGGING_FACE:
            if not self.huggingface_repo_id:
                raise ValueError(
                    "huggingface_repo_id must be provided "
                    "when source is 'huggingface'"
                )
        if self.source == SourceEnum.OLLAMA_LIBRARY:
            if not self.ollama_library_model_name:
                raise ValueError(
                    "ollama_library_model_name must be provided when source is 'ollama_library'"
                )

        if self.source == SourceEnum.MODEL_SCOPE:
            if not self.model_scope_model_id:
                raise ValueError(
                    "model_scope_model_id must be provided when source is 'model_scope'"
                )

        if self.source == SourceEnum.LOCAL_PATH:
            if not self.local_path:
                raise ValueError(
                    "local_path must be provided when source is 'local_path'"
                )
        return self

    model_config = ConfigDict(protected_namespaces=())


class ModelBase(SQLModel, ModelSource):
    name: str = Field(index=True, unique=True)
    description: Optional[str] = None
    meta: Optional[Dict[str, Any]] = Field(sa_column=Column(JSON), default={})

    replicas: int = Field(default=1, ge=0)
    ready_replicas: int = Field(default=0, ge=0)
    categories: List[str] = Field(sa_column=Column(JSON), default=[])
    embedding_only: Annotated[
        bool,
        PydanticField(default=False, deprecated="Deprecated, use categories instead"),
    ]
    image_only: Annotated[
        bool,
        PydanticField(default=False, deprecated="Deprecated, use categories instead"),
    ]
    reranker: Annotated[
        bool,
        PydanticField(default=False, deprecated="Deprecated, use categories instead"),
    ]
    speech_to_text: Annotated[
        bool,
        PydanticField(default=False, deprecated="Deprecated, use categories instead"),
    ]
    text_to_speech: Annotated[
        bool,
        PydanticField(default=False, deprecated="Deprecated, use categories instead"),
    ]
    placement_strategy: PlacementStrategyEnum = PlacementStrategyEnum.SPREAD
    cpu_offloading: bool = False
    distributed_inference_across_workers: bool = False
    worker_selector: Optional[Dict[str, str]] = Field(
        sa_column=Column(JSON), default={}
    )
    gpu_selector: Optional[GPUSelector] = Field(
        sa_column=Column(pydantic_column_type(GPUSelector)), default=None
    )

    backend: Optional[str] = None
    backend_version: Optional[str] = None
    backend_parameters: Optional[List[str]] = Field(sa_column=Column(JSON), default=[])

    @model_validator(mode="after")
    def validate(self):
        backend = get_backend(self)
        if backend == BackendEnum.LLAMA_BOX:
            if self.source == SourceEnum.HUGGING_FACE and not self.huggingface_filename:
                raise ValueError(
                    "huggingface_filename must be provided when source is 'huggingface'"
                )
        elif backend == BackendEnum.VLLM:
            if self.cpu_offloading:
                raise ValueError("CPU offloading is only supported for GGUF models")
            if self.distributed_inference_across_workers:
                raise ValueError(
                    "Distributed inference accross workers is only supported for GGUF models"
                )
        elif backend == BackendEnum.VOX_BOX:
            if self.distributed_inference_across_workers:
                raise ValueError(
                    "Distributed inference accross workers is only supported for GGUF models"
                )
        return self


class Model(ModelBase, BaseModelMixin, table=True):
    __tablename__ = 'models'
    id: Optional[int] = Field(default=None, primary_key=True)

    distributable: Optional[bool] = False
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
    STARTING = "starting"
    RUNNING = "running"
    SCHEDULED = "scheduled"
    ERROR = "error"
    DOWNLOADING = "downloading"
    ANALYZING = "analyzing"
    UNREACHABLE = "unreachable"


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

    # overwrite the hash to use in uniquequeue
    def __hash__(self):
        return self.id


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


def is_gguf_model(model: Model):
    """
    Check if the model is a GGUF model.
    Args:
        model: Model to check.
    """
    return (
        model.source == SourceEnum.OLLAMA_LIBRARY
        or (
            model.source == SourceEnum.HUGGING_FACE
            and model.huggingface_filename
            and model.huggingface_filename.endswith(".gguf")
        )
        or (
            model.source == SourceEnum.MODEL_SCOPE
            and model.model_scope_file_path
            and model.model_scope_file_path.endswith(".gguf")
        )
        or (
            model.source == SourceEnum.LOCAL_PATH
            and model.local_path
            and model.local_path.endswith(".gguf")
        )
        or (model.backend == BackendEnum.LLAMA_BOX)
    )


def is_audio_model(model: Model):
    """
    Check if the model is a STT or TTS model.
    Args:
        model: Model to check.
    """
    if model.backend == BackendEnum.VOX_BOX:
        return True

    if model.categories:
        return (
            'speech_to_text' in model.categories or 'text_to_speech' in model.categories
        )

    return False


def is_image_model(model: Model):
    """
    Check if the model is an image model.
    Args:
        model: Model to check.
    """
    return "image" in model.categories


def is_embedding_model(model: Model):
    """
    Check if the model is an embedding model.
    Args:
        model: Model to check.
    """
    return "embedding" in model.categories


def is_renaker_model(model: Model):
    """
    Check if the model is a reranker model.
    Args:
        model: Model to check.
    """
    return "reranker" in model.categories


def get_backend(model: Model) -> str:
    if model.backend:
        return model.backend

    if is_gguf_model(model):
        return BackendEnum.LLAMA_BOX

    if is_audio_model(model):
        return BackendEnum.VOX_BOX

    return BackendEnum.VLLM


def get_mmproj_filename(model: Model) -> Optional[str]:
    """
    Get the mmproj filename for the model. If the mmproj is not provided in the model's
    backend parameters, it will try to find the default mmproj file.
    """
    if get_backend(model) != BackendEnum.LLAMA_BOX:
        return None

    mmproj = find_parameter(model.backend_parameters, ["mmproj"])
    if mmproj and Path(mmproj).name == mmproj:
        return mmproj

    return "*mmproj*.gguf"
