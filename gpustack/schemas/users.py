"""User — pydantic surface backed by the unified :class:`Principal`.

Before identity consolidation ``User`` was its own table; today
``User = Principal`` (the table-mapped class lives in
:mod:`gpustack.schemas.principals`). Code keeps using ``User`` for
clarity at call sites that are conceptually user-shaped (login,
admin user CRUD, api-key ownership, …), while every USER row
physically lives in ``principals`` alongside ORG / GROUP / SYSTEM
rows. SYSTEM-context call sites should construct / query
``Principal`` directly to avoid mis-suggesting a human account.

Pydantic DTOs (``UserCreate``, ``UserUpdate``, ``UserPublic``, …) stay
here — they're API-surface shapes, not table mappings. They expose
``username`` and ``full_name`` JSON keys for backward compat, aliased
to the underlying ``name`` / ``display_name`` columns via Pydantic
``validation_alias`` + ``serialization_alias``. ORG / GROUP-only
columns (``description``, ``kind``, ``parent_principal_id``) aren't
on the user wire surface.

Re-exports ``AuthProviderEnum`` so existing
``from gpustack.schemas.users import AuthProviderEnum`` callers keep
working unchanged.
"""

import re
from datetime import datetime
from typing import ClassVar, List, Optional

from pydantic import (
    AliasChoices,
    ConfigDict,
    Field as PField,
    field_validator,
)
from sqlalchemy import Text
from sqlalchemy.orm import selectinload
from sqlmodel import Column, Field, SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.common import ListParams, PaginatedList
from gpustack.schemas.principals import (  # noqa: F401  re-exports
    AuthProviderEnum,
    Principal,
    PrincipalType,
)


# ``User`` aliases the unified :class:`Principal` so that code which
# constructs / queries the user-shaped surface continues to work
# unchanged. Every USER row has ``kind == PrincipalType.USER``.
User = Principal


# System-actor naming conventions. The on-disk ``name`` string stays
# ``system/cluster-1`` for the default cluster's SYSTEM principal —
# changing the value would orphan existing rows.
system_name_prefix = "system/cluster"
default_cluster_principal_name = f"{system_name_prefix}-1"


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

    The wire-level field names ``username`` / ``full_name`` are kept
    for backward compatibility with API clients (OAuth2 password flow,
    the bundled UI, the SDK). Internally they map onto the unified
    Principal columns ``name`` and ``display_name``; the
    ``validation_alias`` on each lets
    ``UserPublic.model_validate(principal)`` read directly off a
    ``Principal`` row without an explicit conversion step.
    """

    # ``populate_by_name=True`` so callers can still construct DTOs via
    # the wire field name (e.g. ``UserCreate(username='alice')``);
    # ``validation_alias`` makes ``model_validate(principal)`` pick up
    # ``principal.name`` / ``principal.display_name`` automatically.
    model_config = ConfigDict(populate_by_name=True)

    username: str = PField(
        validation_alias=AliasChoices('username', 'name'),
        serialization_alias='name',
    )
    is_admin: bool = False
    is_active: bool = True
    full_name: Optional[str] = PField(
        default=None,
        validation_alias=AliasChoices('full_name', 'display_name'),
        serialization_alias='display_name',
    )
    avatar_url: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    source: Optional[str] = Field(default=AuthProviderEnum.Local)
    require_password_change: bool = Field(default=False)


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

    Wire field name stays ``full_name``; persisted as ``display_name``
    on the Principal row (see :class:`UserBase` for the alias
    rationale).
    """

    model_config = ConfigDict(populate_by_name=True)

    full_name: Optional[str] = PField(
        default=None,
        validation_alias=AliasChoices('full_name', 'display_name'),
        serialization_alias='display_name',
    )
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
        "name",
        "is_admin",
        "display_name",
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


def is_default_cluster_principal(principal: Principal) -> bool:
    return (
        principal.kind == PrincipalType.SYSTEM
        and principal.name == default_cluster_principal_name
    )


async def get_default_cluster_principal(session: AsyncSession) -> Optional[Principal]:
    # ``Principal.cluster`` is ``lazy='noload'`` (so generic reads don't
    # silently fan out per row). Bootstrap callers
    # (``Server._migrate_legacy_token`` / ``_ensure_registration_token``)
    # need the related Cluster row, so eager-load it here.
    # Default cluster principal is SYSTEM kind. Scope by kind so a
    # GROUP that happens to share the same ``name`` (different
    # partition under the partitioned name uniqueness model) can't
    # shadow it.
    return await Principal.one_by_fields(
        session=session,
        fields={
            "kind": PrincipalType.SYSTEM,
            "name": default_cluster_principal_name,
        },
        options=[selectinload(Principal.cluster)],
    )
