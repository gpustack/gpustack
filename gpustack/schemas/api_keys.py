from enum import Enum
from datetime import datetime
from typing import ClassVar, Optional, List, TYPE_CHECKING
from sqlalchemy import Column, UniqueConstraint
from sqlmodel import Field, SQLModel, Text, JSON, Relationship

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import ListParams, PaginatedList, UTCDateTime

if TYPE_CHECKING:
    from gpustack.schemas.users import User


class PermissionScope(str, Enum):
    """
    Permission scope for API key access control.

    Currently supports coarse-grained scopes. Future extensions may include:
    - management.readonly: Read-only API access (GET requests only)
    - management.write: Full API write access
    - inference.chat: Chat completion endpoints only
    - inference.embeddings: Embeddings endpoints only
    - inference.completions: Completions endpoints only
    """

    ALL = "*"
    MANAGEMENT = "management"
    INFERENCE = "inference"


class ApiKeyUpdate(SQLModel):
    allowed_model_names: Optional[List[str]] = Field(
        default=None,
        sa_column=Column(JSON, nullable=True),
    )
    description: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    scope: List[PermissionScope] = Field(
        default=[PermissionScope.ALL],
        sa_column=Column(JSON, nullable=False),
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
    is_custom: bool = Field(default=False, nullable=False)


class ApiKeyListParams(ListParams):
    sortable_fields: ClassVar[List[str]] = [
        "name",
        "expires_at",
        "created_at",
        "updated_at",
    ]


class ApiKeyCreate(ApiKeyBase):
    expires_in: Optional[int] = None
    custom: Optional[str] = None


class ApiKeyPublic(ApiKeyBase):
    id: int
    value: Optional[str] = None  # only available when creating
    masked_value: Optional[str] = None  # partial characters for identification
    is_custom: bool
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime] = None


ApiKeysPublic = PaginatedList[ApiKeyPublic]
