from datetime import datetime
from typing import Optional
from sqlmodel import Field, SQLModel

from gpustack.schemas.common import PaginatedList
from gpustack.mixins import BaseModelMixin


class ModelInstanceBase(SQLModel):
    model_id: int
    node_id: Optional[int] = None
    node_ip: Optional[str] = None
    pid: Optional[int] = None
    port: Optional[int] = None
    state: Optional[str] = None

    class Config:
        # The "model_id" field conflicts with the protected namespace "model_" in Pydantic.
        # Disable it given that it's not a real issue for this particular field.
        protected_namespaces = ()


class ModelInstance(ModelInstanceBase, BaseModelMixin, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    model_id: int


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
