from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, ConfigDict, model_validator
from sqlalchemy import Column
from sqlmodel import Field, Relationship, SQLModel

from gpustack.schemas.common import PaginatedList, pydantic_column_type
from gpustack.mixins import BaseModelMixin

# Models


class SourceEnum(str, Enum):
    huggingface = "huggingface"
    ollama_library = "ollama_library"
    s3 = "s3"


class ModelSource(BaseModel):
    source: SourceEnum
    huggingface_repo_id: Optional[str] = None
    huggingface_filename: Optional[str] = None
    ollama_library_model_name: Optional[str] = None
    s3_address: Optional[str] = None

    @model_validator(mode="after")
    def check_huggingface_fields(self):
        if self.source == SourceEnum.huggingface:
            if not self.huggingface_repo_id or not self.huggingface_filename:
                raise ValueError(
                    "huggingface_repo_id and huggingface_filename must be provided "
                    "when source is 'huggingface'"
                )
        if self.source == SourceEnum.ollama_library:
            if not self.ollama_library_model_name:
                raise ValueError(
                    "ollama_library_model_name must be provided when source is 'ollama_library'"
                )
        return self


class ModelBase(SQLModel, ModelSource):
    name: str = Field(index=True, unique=True)
    description: Optional[str] = None

    replicas: int = Field(default=1, ge=0)


class Model(ModelBase, BaseModelMixin, table=True):
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
    initializing = "Initializing"
    pending = "Pending"
    running = "Running"
    scheduled = "Scheduled"
    error = "Error"
    downloading = "Downloading"


class ComputedResourceClaim(BaseModel):
    is_unified_memory: Optional[bool] = False
    offload_layers: Optional[int] = None
    memory: Optional[int] = Field(default=None)  # in bytes
    gpu_memory: Optional[int] = Field(default=None)  # in bytes


class ModelInstanceBase(SQLModel, ModelSource):
    worker_id: Optional[int] = None
    worker_ip: Optional[str] = None
    pid: Optional[int] = None
    port: Optional[int] = None
    download_progress: Optional[float] = None
    state: ModelInstanceStateEnum = ModelInstanceStateEnum.pending
    state_message: Optional[str] = None
    computed_resource_claim: Optional[ComputedResourceClaim] = Field(
        sa_column=Column(pydantic_column_type(ComputedResourceClaim)), default=None
    )
    gpu_index: Optional[int] = None

    model_id: int = Field(default=None, foreign_key="model.id")
    model_name: str

    # The "model_id" field conflicts with the protected namespace "model_" in Pydantic.
    # Disable it given that it's not a real issue for this particular field.
    model_config = ConfigDict(protected_namespaces=())


class ModelInstance(ModelInstanceBase, BaseModelMixin, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

    model: Model | None = Relationship(
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
