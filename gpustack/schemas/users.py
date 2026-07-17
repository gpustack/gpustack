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
``username`` and ``full_name`` JSON keys for backward compat;
``validation_alias`` lets ``UserPublic.model_validate(principal)``
read directly off the underlying ``name`` / ``display_name`` columns
without an explicit conversion step. The wire→storage rename on
writes is handled in the route handlers (see ``routes/users.py``)
rather than via ``serialization_alias``, so FastAPI's default
``response_model_by_alias=True`` doesn't leak the storage column
names into responses. ORG / GROUP-only columns (``description``,
``kind``, ``parent_principal_id``) aren't on the user wire surface.

Re-exports ``AuthProviderEnum`` so existing
``from gpustack.schemas.users import AuthProviderEnum`` callers keep
working unchanged.
"""

import re
from datetime import datetime
from typing import ClassVar, List, Optional, Tuple

from pydantic import (
    AliasChoices,
    ConfigDict,
    Field as PField,
    computed_field,
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
    ``Principal`` row without an explicit conversion step. No
    ``serialization_alias`` is set — FastAPI's default
    ``response_model_by_alias=True`` would otherwise emit the storage
    column names on the wire. Route handlers map wire→storage
    explicitly when writing back to the Principal row.
    """

    # ``populate_by_name=True`` so callers can still construct DTOs via
    # the wire field name (e.g. ``UserCreate(username='alice')``);
    # ``validation_alias`` makes ``model_validate(principal)`` pick up
    # ``principal.name`` / ``principal.display_name`` automatically.
    model_config = ConfigDict(populate_by_name=True)

    username: str = PField(
        validation_alias=AliasChoices('username', 'name'),
    )
    is_admin: bool = False
    is_active: bool = True
    full_name: Optional[str] = PField(
        default=None,
        validation_alias=AliasChoices('full_name', 'display_name'),
    )
    avatar_url: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    source: Optional[str] = Field(default=AuthProviderEnum.Local.value)
    require_password_change: bool = Field(default=False)


class UserCreate(UserBase):
    # Optional because non-Local sources authenticate via the IdP and
    # carry no password row. The Local-source requires-password rule
    # lives in the route handler so it can reference ``source``.
    password: Optional[str] = None

    @field_validator('password')
    def validate_password(cls, value):
        if value is None:
            return value
        return _validate_password(value)


class UserUpdate(UserBase):
    password: Optional[str] = None
    # Overrides ``UserBase.source`` on two axes.
    #
    # Default ``None`` (vs. ``UserBase``'s ``"Local"``): an omitted
    # ``source`` must mean "leave as-is". Inheriting the ``Local``
    # default would silently flip every existing SSO user back to
    # Local the first time an admin saved an unrelated field via a
    # client that didn't send the key.
    #
    # Tighter type (``AuthProviderEnum`` vs. free-form ``str``):
    # ``_resolve_external_user`` needs an exact match on the next
    # login, so a garbage value like ``"banana"`` would lock the
    # user out with no error at write time. Pydantic rejects
    # anything outside the enum at 422.
    source: Optional[AuthProviderEnum] = None


class UserSelfUpdate(SQLModel):
    """Schema for users updating their own profile — excludes
    privileged fields.

    Wire field name stays ``full_name``; persisted as ``display_name``
    on the Principal row (see :class:`UserBase` for the alias
    rationale). The route handler maps wire→storage explicitly.
    """

    model_config = ConfigDict(populate_by_name=True)

    full_name: Optional[str] = PField(
        default=None,
        validation_alias=AliasChoices('full_name', 'display_name'),
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
        "username",
        "is_admin",
        "full_name",
        "source",
        "is_active",
        "created_at",
        "updated_at",
    ]

    # Wire-level field names that map to different DB column names.
    # Key = wire name (from API / UI), value = DB column name.
    _WIRE_TO_DB_SORT_MAP: ClassVar[dict[str, str]] = {
        "username": "name",
        "full_name": "display_name",
    }

    @computed_field
    @property
    def order_by(self) -> Optional[List[Tuple[str, str]]]:
        """Override to map wire-level sort field names to DB column names."""
        raw = super().order_by
        if raw is None:
            return None
        return [(self._WIRE_TO_DB_SORT_MAP.get(f, f), d) for f, d in raw]


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
