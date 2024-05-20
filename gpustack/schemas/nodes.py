from datetime import datetime
from pydantic import BaseModel
from sqlmodel import Field, SQLModel

from gpustack.mixins import BaseModelMixin

from .common import PaginatedList


class ResourceSummary(BaseModel):
    Capacity: dict[str, float] = {}
    allocatable: dict[str, float] = {}


class NodeBase(SQLModel):
    id: str
    name: str
    hostname: str
    address: str
    labels: dict[str, str] = {}
    resources: ResourceSummary
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
