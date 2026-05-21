"""Pydantic surface for the Organization API.

Backed by ``principals`` rows where ``kind == ORG``. There is no
dedicated ``organizations`` table — the Organization concept is a
Pydantic-only DTO over ``Principal`` for the public API.
"""

import re
from datetime import datetime
from typing import ClassVar, List, Optional

from sqlmodel import Field, SQLModel

from gpustack.schemas.common import ListParams, PaginatedList
from gpustack.schemas.principals import (
    Principal,
    PrincipalType,
)


# ``PLATFORM_ORGANIZATION_ID`` used to be a module-level alias here.
# Removed because the platform principal id is no longer a stable
# compile-time constant after identity consolidation (the migration
# renumbers it above ``MAX(users.id)``). New callers should call
# ``platform_principal_id()`` directly (live runtime value, refreshed
# by :func:`init_platform_principal_id`).


# URL-safe identifier format — k8s-style ``metadata.name`` (lowercase,
# starts with letter, only letters/digits/hyphens, can't end with
# hyphen).
name_pattern = r'^[a-z](?:[a-z0-9\-]*[a-z0-9])?$'

# "Personal" is the UI label for the user-self namespace (USER
# principal rendered via ``OrganizationPublic.from_principal`` with
# ``is_personal=True``); "Global" is the UI label for admin-curated
# Platform rows (e.g. inference backends with owner_principal_id IS
# NULL). Backend never emits these strings as a real Org's
# ``display_name``, but letting an admin create a regular Org named
# "Personal" / "Global" would visually collide with those built-in
# UX slots. Match case-insensitively after trimming whitespace.
RESERVED_ORG_DISPLAY_NAMES = {"personal", "global", "system", "system-toolkit"}
RESERVED_ORG_NAMES = {"personal", "global", "system", "system-toolkit"}
# Legacy user-principal name pattern — keep humans from grabbing the
# slot of a user's auto-generated Personal namespace.
personal_name_pattern = re.compile(r'^user-\d+$')


def _check_reserved_display_name(display_name: str) -> None:
    """Raise ValueError if display_name is reserved for the system."""
    if not isinstance(display_name, str):
        raise ValueError("display_name must be a string")
    if display_name.strip().lower() in RESERVED_ORG_DISPLAY_NAMES:
        raise ValueError(
            f"'{display_name}' is a reserved organization name; please choose another"
        )


def _check_name_format(name: str) -> None:
    """Raise ValueError if the URL-safe ``name`` fails the formatting
    / reserved checks.
    """
    if not isinstance(name, str):
        raise ValueError("name must be a string")
    if not re.match(name_pattern, name):
        raise ValueError(
            "name must be lowercase, start with a letter, only contain "
            "letters, numbers, and hyphens, and not end with a hyphen"
        )
    if name.lower() in RESERVED_ORG_NAMES or personal_name_pattern.match(name):
        raise ValueError(f"'{name}' is a reserved name; please choose another")


def validate_org_input(
    *, display_name: Optional[str], name: Optional[str] = None
) -> None:
    """Validate user-supplied Org create/update payloads."""
    if display_name is not None:
        _check_reserved_display_name(display_name)
    if name is not None:
        _check_name_format(name)


class OrganizationUpdate(SQLModel):
    display_name: Optional[str] = Field(default=None, nullable=True)
    description: Optional[str] = Field(default=None, nullable=True)


class OrganizationCreate(OrganizationUpdate):
    name: str = Field(nullable=False)


class OrganizationListParams(ListParams):
    sortable_fields: ClassVar[List[str]] = [
        "display_name",
        "name",
        "created_at",
        "updated_at",
    ]


class OrganizationPublic(SQLModel):
    id: int
    name: Optional[str] = None
    display_name: Optional[str] = None
    description: Optional[str] = None
    # ``is_personal`` is no longer a stored flag — a row is "personal"
    # iff it's a USER principal (rendered through this DTO when listing
    # me/orgs etc.). The Org listing endpoint filters to ORG kind, so
    # this defaults to False there.
    is_personal: bool = False
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_principal(cls, p: Principal) -> "OrganizationPublic":
        """Render a Principal row as the Organization shape.

        ``name`` and ``display_name`` are passed through verbatim from
        the principal row; the client decides the rendered label. For
        USER principals (the Personal slot), ``is_personal=True`` flags
        the row so the UI can substitute its localized "Personal"
        label instead of the user's own ``display_name``.
        """
        is_personal = p.kind == PrincipalType.USER
        return cls(
            id=p.id,
            name=p.name,
            display_name=p.display_name,
            description=p.description,
            is_personal=is_personal,
            created_at=p.created_at,
            updated_at=p.updated_at,
        )


OrganizationsPublic = PaginatedList[OrganizationPublic]


class OrganizationMembershipPublic(SQLModel):
    """A member of an Org — either a User or a Group.

    USER and GROUP principals are peer-level in the new identity
    model, so the membership API treats them uniformly: identity
    fields (``principal_id``, ``principal_kind``, ``principal_name``,
    ``principal_display_name``, ``principal_description``) come off
    the ``principals`` row. ``principal_name`` is the stable
    identifier; ``principal_display_name`` is the optional human
    label. The UI is expected to render ``display_name || name``;
    both are sent so the UI can show them side-by-side or use
    ``name`` to disambiguate equal ``display_name`` values.
    """

    principal_id: int
    principal_kind: str
    principal_name: Optional[str] = None
    principal_display_name: Optional[str] = None
    principal_description: Optional[str] = None
    organization_id: int
    role: Optional[str] = None
    created_at: datetime
