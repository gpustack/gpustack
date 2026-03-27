from datetime import datetime
from typing import ClassVar, Optional, List, TYPE_CHECKING
from sqlalchemy import Column, UniqueConstraint
from sqlmodel import Field, SQLModel, Text, JSON, Relationship

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import ListParams, PaginatedList, UTCDateTime

if TYPE_CHECKING:
    from gpustack.schemas.users import User


class ApiKeyUpdate(SQLModel):
    allowed_model_names: Optional[List[str]] = Field(
        default=None,
        sa_column=Column(JSON, nullable=True),
    )
    description: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )


class ApiKeyBase(ApiKeyUpdate):
    name: str


class ApiKey(ApiKeyBase, BaseModelMixin, table=True):
    __tablename__ = 'api_keys'
    __table_args__ = (UniqueConstraint('user_id', 'name', name='uix_user_id_name'),)
    id: Optional[int] = Field(default=None, primary_key=True)
    access_key: str = Field(unique=True, index=True)
    hashed_secret_key: str = Field(unique=True)
    user_id: int = Field(foreign_key='users.id', nullable=False)
    expires_at: Optional[datetime] = Field(sa_column=Column(UTCDateTime), default=None)
    user: Optional["User"] = Relationship(
        back_populates="api_keys",
        sa_relationship_kwargs={"lazy": "noload"},
    )


class ApiKeyListParams(ListParams):
    sortable_fields: ClassVar[List[str]] = [
        "name",
        "expires_at",
        "created_at",
        "updated_at",
    ]


class ApiKeyCreate(ApiKeyBase):
    expires_in: Optional[int] = None


class ApiKeyPublic(ApiKeyBase):
    id: int
    value: Optional[str] = None  # only available when creating
    masked_value: Optional[str] = None  # partial characters for identification
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime] = None


ApiKeysPublic = PaginatedList[ApiKeyPublic]
