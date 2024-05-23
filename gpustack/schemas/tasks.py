from datetime import datetime
from typing import List
from sqlalchemy import Column
from sqlmodel import Field, SQLModel, JSON

from gpustack.schemas.common import PaginatedList
from gpustack.mixins import BaseModelMixin


class TaskBase(SQLModel):
    name: str
    method_path: str
    args: List = Field(sa_column=Column(JSON), default={})
    node_id: str
    pid: int | None = None


class Task(TaskBase, BaseModelMixin, table=True):
    id: int | None = Field(default=None, primary_key=True)


class TaskCreate(TaskBase):
    pass


class TaskUpdate(TaskBase):
    pass


class TaskPublic(
    TaskBase,
):
    id: int
    created_at: datetime
    updated_at: datetime


TasksPublic = PaginatedList[TaskPublic]
