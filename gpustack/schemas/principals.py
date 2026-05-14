"""Principal — the unified owner-identity model.

Every namespaced actor in the system (a User, an Organization, a User
Group) is a row in ``principals``. The kind-specific extension lives in
its own table:

- ``users`` — credentials, role, system flags, cluster/worker FKs
- (no extension for ORG / GROUP — every column they need is on
  ``principals`` itself)

All three kinds are peer-level: there is no structural parent. A
Group's relationship to an Org (if any) is expressed by a row in
``principal_memberships`` with ``parent=Org, member=Group`` — the same
mechanism used for a User joining an Org. The semantics: every active
user in that Group inherits the membership's role inside the Org.

Resources (``models``, ``model_routes``, ``clusters``, ...) record their
owner via ``owner_principal_id``. Memberships connect principals to
principals (a user-principal joining an org-principal, a user-
principal joining a group-principal, or a group-principal joining an
org-principal). ACLs reference principals directly.
"""

from datetime import datetime
from enum import Enum
from typing import ClassVar, List, Optional

from sqlalchemy import Enum as SQLEnum, text
from sqlmodel import (
    Column,
    Field,
    ForeignKey,
    Index,
    Integer,
    SQLModel,
    UniqueConstraint,
)
from sqlalchemy import Text

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import ListParams, PaginatedList
from gpustack.schemas.users import AuthProviderEnum


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
    #
    # Both roles apply to User-members and to Group-members of an Org;
    # a Group-membership confers the role on every active user inside
    # the Group.
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
    # Where this principal originated. ``Local`` for admin-created
    # rows (the default for USER / ORG, and Group rows created via the
    # ``/groups`` admin surface); ``OIDC`` / ``SAML`` for Group rows
    # auto-created by IdP sync the first time a new claim name was
    # seen. Surfaced on the Groups list so admins can tell which
    # entries are IdP-managed; not meaningful for USER (use
    # ``users.source`` for users) and ORG (always Local in v1) but
    # cheaper to keep one column for all kinds than to special-case.
    source: AuthProviderEnum = Field(
        default=AuthProviderEnum.Local,
        sa_column=Column(
            SQLEnum(AuthProviderEnum),
            nullable=False,
            server_default=AuthProviderEnum.Local.value,
        ),
    )


class Principal(PrincipalBase, BaseModelMixin, table=True):
    __tablename__ = 'principals'
    __table_args__ = (
        UniqueConstraint('slug', name='uix_principals_slug'),
        # Group names are globally unique among active groups. USER /
        # ORG names are not constrained here (Users key off
        # ``users.username``, Orgs off ``slug``).
        #
        # Postgres supports partial unique indexes — declared here for
        # autogenerate parity with the migration. MySQL has no partial
        # index, so the migration creates a plain non-unique index and
        # the route handlers / sync service enforce uniqueness at the
        # application layer.
        Index(
            'uix_principals_group_name',
            'name',
            unique=True,
            postgresql_where=text("kind = 'GROUP' AND deleted_at IS NULL"),
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)


class PrincipalListParams(ListParams):
    kind: Optional[PrincipalType] = None
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
    source: AuthProviderEnum = AuthProviderEnum.Local
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
    # Carries an ``OrgRole`` whenever the parent is an ORG (regardless of
    # whether the member is a User or a Group — for a Group-member, the
    # role propagates to every active user in the Group). NULL only for
    # User-in-Group memberships (groups don't have role tiers — you're
    # either in or out).
    role: Optional[OrgRole] = Field(
        default=None,
        sa_column=Column(SQLEnum(OrgRole), nullable=True),
    )
    # Where this membership row originated. ``Local`` for rows the
    # admin (or a route handler on behalf of the admin) created;
    # ``OIDC`` / ``SAML`` for rows written by IdP group-sync. Sync
    # logic only ever rewrites rows where ``source`` matches the
    # current provider — admin-added memberships (source=Local) are
    # untouched by sync runs.
    source: AuthProviderEnum = Field(
        default=AuthProviderEnum.Local,
        sa_column=Column(
            SQLEnum(AuthProviderEnum),
            nullable=False,
            server_default=AuthProviderEnum.Local.value,
        ),
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
