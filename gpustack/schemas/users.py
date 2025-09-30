from datetime import datetime
import re
from enum import Enum
from sqlalchemy import Enum as SQLEnum, Text

from typing import List, Optional, TYPE_CHECKING
from pydantic import field_validator
from sqlmodel import (
    Field,
    Relationship,
    Column,
    SQLModel,
    Integer,
    ForeignKey,
)
from .common import PaginatedList
from ..mixins import BaseModelMixin
from .clusters import Cluster
from .workers import Worker
from gpustack.schemas.links import ModelUserLink

if TYPE_CHECKING:
    from .api_keys import ApiKey
    from gpustack.schemas.models import Model


class UserRole(Enum):
    Worker = "Worker"
    Cluster = "Cluster"


class AuthProviderEnum(str, Enum):
    Local = "Local"
    OIDC = "OIDC"
    SAML = "SAML"


class UserBase(SQLModel):
    username: str
    is_admin: bool = False
    is_active: bool = True
    full_name: Optional[str] = None
    avatar_url: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    source: Optional[str] = Field(
        default=AuthProviderEnum.Local, sa_type=SQLEnum(AuthProviderEnum)
    )
    require_password_change: bool = Field(default=False)

    is_system: bool = False
    role: Optional[UserRole] = Field(
        default=None, description="Role of the user, e.g., worker or cluster"
    )
    cluster_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("clusters.id", ondelete="CASCADE")),
    )
    worker_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("workers.id", ondelete="CASCADE")),
    )


class UserCreate(UserBase):
    password: str

    @field_validator('password')
    def validate_password(cls, value):
        if not re.search(r'[A-Z]', value):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', value):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'[0-9]', value):
            raise ValueError('Password must contain at least one digit')
        if not re.search(r'[!@#$%^&*_+]', value):
            raise ValueError('Password must contain at least one special character')
        return value


class UserUpdate(UserBase):
    password: Optional[str] = None


class UpdatePassword(SQLModel):
    current_password: str
    new_password: str

    @field_validator('new_password')
    def validate_password(cls, value):
        if not re.search(r'[A-Z]', value):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', value):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'[0-9]', value):
            raise ValueError('Password must contain at least one digit')
        if not re.search(r'[!@#$%^&*_+]', value):
            raise ValueError('Password must contain at least one special character')
        return value


class User(UserBase, BaseModelMixin, table=True):
    __tablename__ = 'users'
    id: Optional[int] = Field(default=None, primary_key=True)
    hashed_password: str

    cluster: Optional[Cluster] = Relationship(
        back_populates="cluster_users", sa_relationship_kwargs={"lazy": "selectin"}
    )
    worker: Optional[Worker] = Relationship(sa_relationship_kwargs={"lazy": "selectin"})

    api_keys: List["ApiKey"] = Relationship(
        back_populates='user',
        sa_relationship_kwargs={"cascade": "delete", "lazy": "selectin"},
    )
    models: List["Model"] = Relationship(
        back_populates="users",
        link_model=ModelUserLink,
        sa_relationship_kwargs={"lazy": "selectin"},
    )


class UserActivationUpdate(SQLModel):
    is_active: bool


class UserPublic(UserBase):
    id: int
    created_at: datetime
    updated_at: datetime


UsersPublic = PaginatedList[UserPublic]
