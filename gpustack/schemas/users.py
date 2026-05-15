"""User — pydantic surface backed by the unified :class:`Principal`.

Before identity consolidation ``User`` was its own table; today
``User = Principal`` (the table-mapped class lives in
:mod:`gpustack.schemas.principals`). Code keeps using ``User`` for
clarity at call sites that are conceptually user-shaped (login,
api-key ownership, …), while every USER row physically lives in
``principals`` alongside ORG and GROUP rows.

Pydantic DTOs (``UserCreate``, ``UserUpdate``, ``UserPublic``, …) stay
here — they're API-surface shapes, not table mappings. They omit the
ORG / GROUP-only columns (``slug``, ``description``, ``kind``,
``parent_principal_id``) because those aren't part of the user-facing
contract.

Re-exports the enums (``AuthProviderEnum``, ``UserRole``) so existing
``from gpustack.schemas.users import AuthProviderEnum`` callers keep
working unchanged.
"""

import re
from datetime import datetime
from typing import ClassVar, List, Optional

from pydantic import field_validator
from sqlalchemy import Text
from sqlalchemy.orm import selectinload
from sqlmodel import Column, Field, SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.common import ListParams, PaginatedList
from gpustack.schemas.principals import (  # noqa: F401  re-exports
    AuthProviderEnum,
    Principal,
    PrincipalType,
    UserRole,
)


# ``User`` aliases the unified :class:`Principal` so that code which
# constructs / queries the user-shaped surface continues to work
# unchanged. Every USER row has ``kind == PrincipalType.USER``.
User = Principal


# System-actor naming conventions.
system_name_prefix = "system/cluster"
default_cluster_user_name = f"{system_name_prefix}-1"


# --------------------------------------------------------------------
# Pydantic DTOs (API-surface, not table mappings)
# --------------------------------------------------------------------


def _validate_password(value: str) -> str:
    if not re.search(r'[A-Z]', value):
        raise ValueError('Password must contain at least one uppercase letter')
    if not re.search(r'[a-z]', value):
        raise ValueError('Password must contain at least one lowercase letter')
    if not re.search(r'[0-9]', value):
        raise ValueError('Password must contain at least one digit')
    if not re.search(r'[!@#$%^&*_+]', value):
        raise ValueError('Password must contain at least one special character')
    return value


class UserBase(SQLModel):
    """User-facing fields on the public API. Subset of Principal.

    ``kind`` and the ORG/GROUP-only columns are intentionally omitted —
    they don't belong on the user-facing CRUD surface.
    """

    username: str
    is_admin: bool = False
    is_active: bool = True
    full_name: Optional[str] = None
    avatar_url: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    source: Optional[str] = Field(default=AuthProviderEnum.Local)
    require_password_change: bool = Field(default=False)

    is_system: bool = False
    role: Optional[UserRole] = Field(
        default=None, description="Role of the user, e.g., worker or cluster"
    )
    cluster_id: Optional[int] = None
    worker_id: Optional[int] = None


class UserCreate(UserBase):
    password: str

    @field_validator('password')
    def validate_password(cls, value):
        return _validate_password(value)


class UserUpdate(UserBase):
    password: Optional[str] = None


class UserSelfUpdate(SQLModel):
    """Schema for users updating their own profile — excludes
    privileged fields.
    """

    full_name: Optional[str] = None
    avatar_url: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    password: Optional[str] = None

    @field_validator('password')
    def validate_password(cls, value):
        if value is None:
            return value
        return _validate_password(value)


class UpdatePassword(SQLModel):
    current_password: str
    new_password: str

    @field_validator('new_password')
    def validate_password(cls, value):
        return _validate_password(value)


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


UsersPublic = PaginatedList[UserPublic]


def is_default_cluster_user(cluster_user: User) -> bool:
    return (
        cluster_user.is_system
        and cluster_user.cluster_id is not None
        and cluster_user.username == default_cluster_user_name
    )


async def get_default_cluster_user(session: AsyncSession) -> Optional[User]:
    # ``Principal.cluster`` is ``lazy='noload'`` (so generic reads don't
    # silently fan out per row). Bootstrap callers
    # (``Server._migrate_legacy_token`` / ``_ensure_registration_token``)
    # need the related Cluster row, so eager-load it here.
    return await User.one_by_field(
        session=session,
        field="username",
        value=default_cluster_user_name,
        options=[selectinload(User.cluster)],
    )
