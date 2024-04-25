from datetime import datetime
from typing import Literal
from sqlmodel import Field, SQLModel

from .common import PaginatedList
from ..mixins import BaseModelMixin


class ModelBase(SQLModel):
    name: str
    description: str | None = None
    source: str = Literal["huggingface", "s3"]
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
