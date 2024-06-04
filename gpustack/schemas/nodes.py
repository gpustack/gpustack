from datetime import datetime
from typing import Dict
from pydantic import field_validator, BaseModel
from sqlmodel import Field, SQLModel, JSON, Column

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import PaginatedList


class ResourceSummary(BaseModel):
    capacity: Dict[str, float] = {}
    allocatable: Dict[str, float] = {}


class NodeBase(SQLModel):
    name: str = Field(index=True, unique=True)
    hostname: str
    address: str
    labels: Dict[str, str] = Field(sa_column=Column(JSON), default={})
    resources: ResourceSummary | None = Field(sa_column=Column(JSON))
    state: str | None = None

    # Workaround for https://github.com/tiangolo/sqlmodel/issues/63
    # It generates a warning.
    # TODO Find a better way.
    @field_validator("resources")
    def validate_resources(cls, val: ResourceSummary):
        return val.model_dump()


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
