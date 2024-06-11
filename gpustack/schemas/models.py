from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel
from sqlmodel import Field, Relationship, SQLModel

from gpustack.schemas.common import PaginatedList
from gpustack.mixins import BaseModelMixin


# Models


class SourceEnum(str, Enum):
    huggingface = "huggingface"
    s3 = "s3"


class ModelSource(BaseModel):
    source: SourceEnum
    huggingface_repo_id: Optional[str] = None
    huggingface_filename: Optional[str] = None
    s3_address: Optional[str] = None


class ModelBase(SQLModel, ModelSource):
    name: str = Field(index=True, unique=True)
    description: Optional[str] = None


class Model(ModelBase, BaseModelMixin, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

    instances: list["ModelInstance"] = Relationship(
        sa_relationship_kwargs={"cascade": "delete"}, back_populates="model"
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


class ModelInstanceBase(SQLModel, ModelSource):
    node_id: Optional[int] = None
    node_ip: Optional[str] = None
    pid: Optional[int] = None
    port: Optional[int] = None
    state: Optional[str] = None
    download_progress: Optional[float] = None

    model_id: int = Field(default=None, foreign_key="model.id")
    model_name: str

    class Config:
        # The "model_id" field conflicts with the protected namespace "model_" in Pydantic.
        # Disable it given that it's not a real issue for this particular field.
        protected_namespaces = ()


class ModelInstance(ModelInstanceBase, BaseModelMixin, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    # model_id: int

    model: Model | None = Relationship(back_populates="instances")


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
