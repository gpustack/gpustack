from datetime import datetime
from pydantic import BaseModel
from sqlmodel import Field, SQLModel, JSON, Column

from gpustack.mixins import BaseModelMixin

from .common import PaginatedList


class ResourceSummary(BaseModel):
    capacity: dict[str, float] = {}
    allocatable: dict[str, float] = {}


class NodeBase(SQLModel):
    id: str
    name: str
    hostname: str
    address: str
    labels: dict[str, str] = Field(sa_column=Column(JSON), default={})
    resources: ResourceSummary = Field(sa_column=Column(JSON))
    state: str


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
