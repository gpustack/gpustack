from datetime import datetime
import re
from enum import Enum
from sqlalchemy import Enum as SQLEnum, Text
from sqlalchemy.orm import selectinload
from sqlmodel.ext.asyncio.session import AsyncSession

from typing import ClassVar, List, Optional, TYPE_CHECKING
from pydantic import field_validator
from sqlmodel import (
    Field,
    Relationship,
    Column,
    SQLModel,
    Integer,
    ForeignKey,
)

from gpustack.schemas.common import ListParams
from .common import PaginatedList
from ..mixins import BaseModelMixin
from .clusters import Cluster
from .workers import Worker

if TYPE_CHECKING:
    from .api_keys import ApiKey
    from .principals import Principal


system_name_prefix = "system/cluster"
default_cluster_user_name = f"{system_name_prefix}-1"


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
    # 1:1 link to the user's Principal row. NOT NULL by construction —
    # every user has a principal, and that principal is the canonical
    # owner identity for resources the user creates in their personal
    # scope. RESTRICT prevents the principal from being deleted while
    # the user row still references it; ``users`` is supposed to be the
    # source of truth for user existence, so the principal goes away
    # only as part of user deletion.
    principal_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("principals.id", ondelete="RESTRICT"),
            nullable=False,
            unique=True,
        ),
    )


class UserCreate(UserBase):
    password: Optional[str] = None

    @field_validator('password')
    def validate_password(cls, value):
        if value is None or value == "":
            return value
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


class UserSelfUpdate(SQLModel):
    """Schema for users updating their own profile - excludes privileged fields"""

    full_name: Optional[str] = None
    avatar_url: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    password: Optional[str] = None

    @field_validator('password')
    def validate_password(cls, value):
        if value is None:
            return value
        if not re.search(r'[A-Z]', value):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', value):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'[0-9]', value):
            raise ValueError('Password must contain at least one digit')
        if not re.search(r'[!@#$%^&*_+]', value):
            raise ValueError('Password must contain at least one special character')
        return value


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

    cluster: Optional[Cluster] = Relationship(
        back_populates="cluster_users", sa_relationship_kwargs={"lazy": "noload"}
    )
    worker: Optional[Worker] = Relationship(sa_relationship_kwargs={"lazy": "noload"})

    # 1:1 link to the user's USER-principal. Setting ``user.principal``
    # (instead of ``user.principal_id``) at construction time lets
    # SQLAlchemy's unit of work insert the principal first and
    # auto-populate ``principal_id`` during a combined flush — the
    # standard idiom for satisfying a NOT NULL FK without a separate
    # round trip.
    principal: Optional["Principal"] = Relationship(
        sa_relationship_kwargs={"lazy": "noload"},
    )

    api_keys: List["ApiKey"] = Relationship(
        back_populates='user',
        sa_relationship_kwargs={"cascade": "delete", "lazy": "noload"},
    )


class UserActivationUpdate(SQLModel):
    is_active: bool


class UserListParams(ListParams):
    sortable_fields: ClassVar[List[str]] = [
        "username",
        "is_admin",
        "full_name",
        "source",
        "is_active",
        "created_at",
        "updated_at",
    ]


class UserPublic(UserBase):
    id: int
    created_at: datetime
    updated_at: datetime


class UserMePublic(UserPublic):
    """``GET /users/me`` response shape.

    Same as ``UserPublic`` plus ``require_password_change`` — that flag
    lives on the user's PASSWORD ``credentials`` row, not on the user
    row itself, so it's only surfaced on the endpoint that actually
    needs it (the frontend uses it to drive the first-login prompt).
    """

    require_password_change: bool = False


UsersPublic = PaginatedList[UserPublic]


def is_default_cluster_user(cluster_user: User) -> bool:
    return (
        cluster_user.is_system
        and cluster_user.cluster_id is not None
        and cluster_user.username == default_cluster_user_name
    )


async def get_default_cluster_user(session: AsyncSession) -> Optional[User]:
    return await User.one_by_field(
        session=session,
        field="username",
        value=default_cluster_user_name,
        options=[selectinload(User.cluster)],
    )
