from datetime import datetime
from enum import Enum
from typing import Optional
from sqlmodel import Field, SQLModel

from gpustack.schemas.common import PaginatedList
from gpustack.mixins import BaseModelMixin


class SourceEnum(str, Enum):
    huggingface = "huggingface"
    s3 = "s3"


class ModelBase(SQLModel):
    name: str = Field(index=True, unique=True)
    description: Optional[str] = None
    source: SourceEnum
    huggingface_model_id: Optional[str] = None
    s3_address: Optional[str] = None


class Model(ModelBase, BaseModelMixin, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)


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
