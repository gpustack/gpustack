"""Principal — the unified owner-identity model.

Every namespaced actor in the system (a User, an Organization, a User
Group) is a row in ``principals``. The kind-specific extension lives in
its own table:

- ``users`` — credentials, role, system flags, cluster/worker FKs
- (no extension for ORG / GROUP — every column they need is on
  ``principals`` itself)

Resources (``models``, ``model_routes``, ``clusters``, ...) record their
owner via ``owner_principal_id``. Memberships connect principals to
principals (a user-principal joining an org-principal, or a user-
principal joining a group-principal). ACLs reference principals
directly.
"""

from datetime import datetime
from enum import Enum
from typing import ClassVar, List, Optional

from sqlalchemy import Enum as SQLEnum
from sqlmodel import (
    Column,
    Field,
    ForeignKey,
    Integer,
    SQLModel,
    UniqueConstraint,
)
from sqlalchemy import Text

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import ListParams, PaginatedList


# Canonical slug of the built-in platform Org-principal. Created by the
# multi-tenancy foundation migration; system / infrastructure resources
# default to it.
#
# ``PLATFORM_PRINCIPAL_ID`` is the id we *seed* it with in a fresh DB.
# It happens to be ``1`` today, but anywhere we don't have to bake the
# integer into the SQL — primarily migrations and any future bootstrap
# code that runs against a populated DB — we look it up by slug instead.
# That's the form that survives any future renumbering (e.g. when
# ``users`` and ``principals`` get unified and USER-kind principals
# inherit ``users.id``, the platform Org will get a new id above
# ``max(users.id)``).
PLATFORM_PRINCIPAL_SLUG = 'default'
PLATFORM_PRINCIPAL_ID = 1


class PrincipalType(str, Enum):
    """Discriminator for the kind of principal a row represents.

    Kept named ``PrincipalType`` (rather than ``PrincipalKind``) for
    continuity with the cluster-access / model-route ACL APIs that
    already accepted this enum on the wire.
    """

    USER = "user"
    ORG = "org"
    GROUP = "group"


class OrgRole(str, Enum):
    # Two-tier Org membership model: OWNER can manage the Org's infra
    # (resources, members, settings); MEMBER is a plain consumer. The
    # platform-wide superuser lives on `users.is_admin` and is distinct
    # from `OrgRole.OWNER` — always disambiguate with `is_platform_admin`
    # vs `org_role == OrgRole.OWNER` in code. The names intentionally
    # diverge from the platform admin/user role so the two never get
    # conflated.
    OWNER = "owner"
    MEMBER = "member"


class PrincipalBase(SQLModel):
    kind: PrincipalType = Field(
        sa_column=Column(SQLEnum(PrincipalType), nullable=False),
    )
    # User-facing identifier used in URLs and ``effective_route_name``.
    # Globally unique among non-NULL values. Auto-set to ``user-{user.id}``
    # for USER principals; user-supplied for ORG; NULL for GROUP (groups
    # never appear in URL prefixes).
    slug: Optional[str] = Field(default=None, nullable=True)
    name: str = Field(nullable=False)
    description: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    # Structural parent. NULL for USER and ORG; for GROUP, points at the
    # owning ORG-principal so the group lives inside it.
    parent_principal_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("principals.id", ondelete="CASCADE"),
            nullable=True,
        ),
    )


class Principal(PrincipalBase, BaseModelMixin, table=True):
    __tablename__ = 'principals'
    __table_args__ = (
        UniqueConstraint('slug', name='uix_principals_slug'),
        # Group names must be unique within their parent org. NULL parent
        # (USER / ORG) doesn't participate — UNIQUE treats NULL as
        # distinct, so users and orgs can share names freely.
        UniqueConstraint(
            'parent_principal_id', 'name', name='uix_principals_parent_name'
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)


class PrincipalListParams(ListParams):
    kind: Optional[PrincipalType] = None
    parent_principal_id: Optional[int] = None
    sortable_fields: ClassVar[List[str]] = [
        "name",
        "slug",
        "kind",
        "created_at",
        "updated_at",
    ]


class PrincipalPublic(SQLModel):
    id: int
    kind: PrincipalType
    slug: Optional[str] = None
    name: str
    description: Optional[str] = None
    parent_principal_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime


PrincipalsPublic = PaginatedList[PrincipalPublic]


class PrincipalMembershipBase(SQLModel):
    parent_principal_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey("principals.id", ondelete="CASCADE"),
            nullable=False,
        ),
    )
    member_principal_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey("principals.id", ondelete="CASCADE"),
            nullable=False,
        ),
    )
    # Only meaningful when parent is an ORG; NULL for GROUP memberships
    # (groups don't have role tiers — you're either in or out).
    role: Optional[OrgRole] = Field(
        default=None,
        sa_column=Column(SQLEnum(OrgRole), nullable=True),
    )


class PrincipalMembership(PrincipalMembershipBase, BaseModelMixin, table=True):
    """Membership of one principal inside another.

    Surrogate ``id`` PK so soft-delete + re-add doesn't collide on a
    composite key. At any point in time at most one row per
    ``(parent_principal_id, member_principal_id)`` may have
    ``deleted_at IS NULL``; that invariant is enforced in the route
    handlers (the add path looks up an active row first and re-uses
    any soft-deleted row by clearing ``deleted_at`` and updating
    ``role``).
    """

    __tablename__ = 'principal_memberships'

    id: Optional[int] = Field(default=None, primary_key=True)
