from datetime import datetime
from sqlmodel import Field, SQLModel

from .common import PaginatedList
from ..mixins import BaseModelMixin


class UserBase(SQLModel):
    username: str
    is_admin: bool = False
    full_name: str | None = None


class UserCreate(UserBase):
    password: str


class UserUpdate(UserBase):
    password: str | None = None


class UpdatePassword(SQLModel):
    current_password: str
    new_password: str


class User(UserBase, BaseModelMixin, table=True):
    id: int | None = Field(default=None, primary_key=True)
    hashed_password: str


class UserPublic(UserBase):
    id: int
    created_at: datetime
    updated_at: datetime


UsersPublic = PaginatedList[UserPublic]
