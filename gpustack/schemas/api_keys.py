from datetime import datetime
from typing import Optional
from sqlalchemy import Column, UniqueConstraint
from sqlmodel import Field, SQLModel

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import PaginatedList, UTCDateTime


class ApiKeyBase(SQLModel):
    name: str
    description: Optional[str] = None


class ApiKey(ApiKeyBase, BaseModelMixin, table=True):
    __tablename__ = 'api_keys'
    __table_args__ = (UniqueConstraint('name', 'user_id', name='uix_name_user_id'),)
    id: Optional[int] = Field(default=None, primary_key=True)
    access_key: str = Field(unique=True, index=True)
    hashed_secret_key: str = Field(unique=True)
    user_id: int
    expires_at: Optional[datetime] = Field(sa_column=Column(UTCDateTime), default=None)


class ApiKeyCreate(ApiKeyBase):
    expires_in: Optional[int] = None


class ApiKeyPublic(ApiKeyBase):
    id: int
    value: Optional[str] = None  # only available when creating
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime] = None


ApiKeysPublic = PaginatedList[ApiKeyPublic]
