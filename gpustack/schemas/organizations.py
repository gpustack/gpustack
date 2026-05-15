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


slug_pattern = r'^[a-z](?:[a-z0-9\-]*[a-z0-9])?$'

# "Personal" is the conceptual user-self namespace (no longer a separate
# Org row); "Global" is the UI label for admin-curated Platform rows
# (e.g. inference backends with owner_principal_id IS NULL). Letting users
# create regular Orgs with these names would collide with built-in UX
# slots. Match case-insensitively after trimming whitespace.
RESERVED_ORG_NAMES = {"personal", "global", "system", "system-toolkit"}
RESERVED_ORG_SLUGS = {"personal", "global", "system", "system-toolkit"}
# User-principal slug pattern — keep humans from grabbing the slot of a
# user's auto-generated Personal namespace.
personal_slug_pattern = re.compile(r'^user-\d+$')


def _check_reserved_name(name: str) -> None:
    """Raise ValueError if name is reserved for the system."""
    if not isinstance(name, str):
        raise ValueError("name must be a string")
    if name.strip().lower() in RESERVED_ORG_NAMES:
        raise ValueError(
            f"'{name}' is a reserved organization name; please choose another"
        )


def _check_slug_format(slug: str) -> None:
    """Raise ValueError if slug fails the formatting / reserved checks."""
    if not isinstance(slug, str):
        raise ValueError("slug must be a string")
    if not re.match(slug_pattern, slug):
        raise ValueError(
            "slug must be lowercase, start with a letter, only contain "
            "letters, numbers, and hyphens, and not end with a hyphen"
        )
    if slug.lower() in RESERVED_ORG_SLUGS or personal_slug_pattern.match(slug):
        raise ValueError(f"'{slug}' is a reserved slug; please choose another")


def validate_org_input(*, name: Optional[str], slug: Optional[str] = None) -> None:
    """Validate user-supplied Org create/update payloads."""
    if name is not None:
        _check_reserved_name(name)
    if slug is not None:
        _check_slug_format(slug)


class OrganizationUpdate(SQLModel):
    name: str = Field(nullable=False)
    description: Optional[str] = Field(default=None, nullable=True)


class OrganizationCreate(OrganizationUpdate):
    slug: str = Field(nullable=False)


class OrganizationListParams(ListParams):
    sortable_fields: ClassVar[List[str]] = [
        "name",
        "slug",
        "created_at",
        "updated_at",
    ]


class OrganizationPublic(SQLModel):
    id: int
    name: str
    slug: Optional[str] = None
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
        """Render a Principal row as the legacy Organization shape.

        For USER principals, surface ``name="Personal"`` so the
        OrgSwitcher renders the canonical label instead of the user's
        username (which is what's stored on the principal row for
        URL-prefix purposes via ``slug=user-{id}``).
        """
        is_personal = p.kind == PrincipalType.USER
        return cls(
            id=p.id,
            name="Personal" if is_personal else p.name,
            slug=p.slug,
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
    ``principal_description``) come off the ``principals`` row.

    ``full_name`` is the one exception — it lives on the ``users``
    table today (not on ``principals``), and ``principal.name`` for a
    USER currently holds only the username. We surface ``full_name``
    here so the Org-members UI can render a proper display name in
    one round trip. Once identity consolidation moves the display
    name onto the principal row, this field collapses into
    ``principal_name`` and the dedicated ``full_name`` can go away.
    """

    principal_id: int
    principal_kind: str
    principal_name: Optional[str] = None
    principal_description: Optional[str] = None
    full_name: Optional[str] = None
    organization_id: int
    role: Optional[str] = None
    created_at: datetime
