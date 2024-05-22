from datetime import datetime
from typing import Dict
from sqlmodel import Field, SQLModel, JSON, Column

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import PaginatedList, BaseModel


class ResourceSummary(BaseModel):
    capacity: Dict[str, float] = {}
    allocatable: Dict[str, float] = {}


from sqlalchemy.dialects.sqlite import JSON


class NodeBase(BaseModel, SQLModel):
    name: str = Field(index=True, unique=True)
    hostname: str
    address: str
    labels: Dict[str, str] = Field(sa_column=Column(JSON), default={})
    # resources: ResourceSummary | None = Field(sa_column=Column(JSON))
    state: str | None = None


class Node(NodeBase, BaseModelMixin, table=True):
    id: int | None = Field(default=None, primary_key=True)


class NodeCreate(NodeBase):
    pass


class NodeUpdate(NodeBase):
    pass


class NodePublic(
    NodeBase,
):
    id: int
    created_at: datetime
    updated_at: datetime


NodesPublic = PaginatedList[NodePublic]
