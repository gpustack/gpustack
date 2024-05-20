from datetime import datetime
from typing import Literal
from sqlmodel import Field, SQLModel

from .common import PaginatedList
from ..mixins import BaseModelMixin


class TaskBase(SQLModel):
    name: str
    method_path: str
    args: list = []
    node_id: str
    pid: int | None = None
    state: str = Literal["Pending", "Running", "Completed", "Failed"]


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
