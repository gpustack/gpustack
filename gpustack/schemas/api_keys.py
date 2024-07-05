from datetime import datetime
from typing import Optional
from sqlmodel import Field, SQLModel

from .common import PaginatedList
from ..mixins import BaseModelMixin


class ApiKeyBase(SQLModel):
    name: str = Field(unique=True)
    description: Optional[str] = None


class ApiKey(ApiKeyBase, BaseModelMixin, table=True):
    __tablename__ = 'api_keys'
    id: Optional[int] = Field(default=None, primary_key=True)
    access_key: str = Field(unique=True, index=True)
    hashed_secret_key: str = Field(unique=True)
    user_id: int
    expires_at: datetime


class ApiKeyCreate(ApiKeyBase):
    expires_in: Optional[int] = None


class ApiKeyPublic(ApiKeyBase):
    id: int
    value: Optional[str] = None  # only available when creating
    created_at: datetime
    updated_at: datetime
    expires_at: datetime


ApiKeysPublic = PaginatedList[ApiKeyPublic]
