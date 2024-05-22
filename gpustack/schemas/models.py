from datetime import datetime
from enum import Enum
from typing import Literal
from sqlmodel import Field, SQLModel

from gpustack.schemas.common import PaginatedList
from gpustack.mixins import BaseModelMixin


class SourceEnum(str, Enum):
    huggingface = "huggingface"
    s3 = "s3"


class ModelBase(SQLModel):
    name: str = Field(index=True, unique=True)
    description: str | None = None
    source: SourceEnum
    huggingface_model_id: str | None = None
    s3_address: str | None = None


class Model(ModelBase, BaseModelMixin, table=True):
    id: int | None = Field(default=None, primary_key=True)


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
