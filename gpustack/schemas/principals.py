"""Principal — the unified identity model.

Every namespaced actor in the system — a User, an Organization, a
User Group, or a SYSTEM service account (cluster / worker bootstrap)
— is a row in the ``principals`` table. The discriminator is
:attr:`Principal.kind`. There is no separate ``users`` extension
table; the row is the principal.

Kind-specific data lives off the identity row:

- **Login credentials** (hashed password + force-change flag) →
  :class:`gpustack.schemas.user_passwords.UserPassword`,
  keyed by ``owner_principal_id``.
- **System actor → infra link**: which Cluster / Worker a SYSTEM
  principal serves is recorded on the infra row
  (``clusters.system_principal_id`` / ``workers.system_principal_id``,
  back-populated to :attr:`Principal.cluster` /
  :attr:`Principal.worker`). The legacy ``role`` / ``is_system`` flags
  were dropped.

All four kinds are peer-level. A Group's relationship to an Org (if
any) is expressed as a row in ``principal_memberships`` with
``parent=Org, member=Group`` — the same mechanism used for a User
joining an Org. Every active member of the Group inherits the
membership's role inside the Org.

Resources (``models``, ``model_routes``, ``clusters``, …) record their
owner via ``owner_principal_id``. Memberships connect principals to
principals. ACLs reference principals directly.
"""

from datetime import datetime
from enum import Enum
from typing import ClassVar, List, Optional, TYPE_CHECKING

from sqlalchemy import Enum as SQLEnum, Text, text
from sqlmodel import (
    Column,
    Field,
    ForeignKey,
    Index,
    Integer,
    Relationship,
    SQLModel,
)
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.clusters import Cluster
from gpustack.schemas.common import ListParams, PaginatedList
from gpustack.schemas.workers import Worker

if TYPE_CHECKING:
    from gpustack.schemas.api_keys import ApiKey


# --------------------------------------------------------------------
# Auth / identity enums — historically defined in schemas/users.py and
# re-exported there for back-compat. Living here now lets the unified
# Principal table reference them without a circular import.
# --------------------------------------------------------------------


class AuthProviderEnum(str, Enum):
    Local = "Local"
    OIDC = "OIDC"
    SAML = "SAML"


# Canonical ``name`` of the built-in platform Org-principal. Created
# by the multi-tenancy foundation migration; system / infrastructure
# resources default to it.
PLATFORM_PRINCIPAL_NAME = 'default'

# Canonical ``name`` of the built-in "all authenticated users" group
# principal. Shares the ``system/`` reserved prefix with
# ``system/cluster-…`` / ``system/worker-…`` so every built-in
# reserved-name principal follows one convention. Every authenticated
# request is treated as a member of this group at auth-context
# resolution time, without a backing ``principal_memberships`` row.
# ClusterAccess (and any future ACL table) can grant to this
# principal_id, and the grant takes effect for every authenticated
# user. Lazy-seeded on server startup by
# :func:`init_authenticated_principal_id` so no schema migration is
# required.
AUTHENTICATED_PRINCIPAL_NAME = 'system/authenticated'
AUTHENTICATED_PRINCIPAL_DISPLAY_NAME = 'All Authenticated Users'

# Names starting with this prefix are reserved for internal /
# built-in principals (``system/cluster-…``, ``system/worker-…``,
# ``system/authenticated``). Group-CRUD routes hide and lock these
# rows so admins can't accidentally rename / delete them; cluster_access
# / ACL surfaces continue to render them for revoke flows.
RESERVED_PRINCIPAL_NAME_PREFIX = 'system/'


def is_reserved_principal_name(name: Optional[str]) -> bool:
    """Return True if ``name`` is in the reserved built-in namespace."""
    return bool(name) and name.startswith(RESERVED_PRINCIPAL_NAME_PREFIX)


# Module-private mutable resolved at server startup by
# :func:`init_platform_principal_id`. Used to be exported as
# ``PLATFORM_PRINCIPAL_ID`` (a hardcoded ``1`` baked into schema
# defaults), but the identity-consolidation migration renumbered the
# platform principal above ``MAX(users.id)`` so the id is no longer a
# stable compile-time constant — and a direct
# ``from gpustack.schemas.principals import PLATFORM_PRINCIPAL_ID``
# would capture the placeholder ``1`` at import time and silently miss
# the runtime update. Underscore prefix discourages that import path;
# every reader goes through :func:`platform_principal_id`.
_PLATFORM_PRINCIPAL_ID: int = 1
_AUTHENTICATED_PRINCIPAL_ID: int = 0


def platform_principal_id() -> int:
    """Sync getter for the platform principal id.

    Every call site that needs the *current* id (schema field defaults
    via ``default_factory``; runtime comparisons; fallback values) must
    use this function rather than reading the module-private
    ``_PLATFORM_PRINCIPAL_ID`` directly.
    """
    return _PLATFORM_PRINCIPAL_ID


def authenticated_principal_id() -> int:
    """Sync getter for the ``system/authenticated`` group principal id.

    Used by ``_accessible_clusters`` (and any future ACL resolver) to
    union grants written against this group with the caller's direct /
    group / org grants — every authenticated principal is an implicit
    member, so the row applies universally.
    """
    return _AUTHENTICATED_PRINCIPAL_ID


# Public alias for ``default_factory=`` on SQLModel fields. Identical
# to :func:`platform_principal_id`; kept distinct to make field-default
# usage greppable.
_platform_principal_id = platform_principal_id


class PrincipalType(str, Enum):
    """Discriminator for the kind of principal a row represents.

    Kept named ``PrincipalType`` (rather than ``PrincipalKind``) for
    continuity with the cluster-access / model-route ACL APIs that
    already accepted this enum on the wire.

    ``SYSTEM`` covers internal service accounts (cluster bootstrap
    tokens, per-worker registration tokens). Which specific infra row
    a SYSTEM principal serves is recorded on the *infra* side —
    ``clusters.system_principal_id`` / ``workers.system_principal_id``
    point at the principal, not the other way around — so the principal
    table doesn't need a polymorphic ``role`` column to disambiguate.
    """

    USER = "user"
    ORG = "org"
    GROUP = "group"
    SYSTEM = "system"


class OrgRole(str, Enum):
    # Two-tier Org membership model: OWNER can manage the Org's infra
    # (resources, members, settings); MEMBER is a plain consumer. The
    # platform-wide superuser lives on `principals.is_admin` and is
    # distinct from `OrgRole.OWNER` — always disambiguate with
    # `is_platform_admin` vs `org_role == OrgRole.OWNER` in code.
    #
    # Both roles apply to User-members and to Group-members of an Org;
    # a Group-membership confers the role on every active user inside
    # the Group.
    OWNER = "owner"
    MEMBER = "member"


# --------------------------------------------------------------------
# Principal — unified identity table
# --------------------------------------------------------------------


class PrincipalBase(SQLModel):
    """Columns common to every Principal kind.

    The only USER-context column left is ``is_admin`` (platform admin
    flag — meaningless on ORG / GROUP / SYSTEM and just stays False
    there). Everything else is shared identity surface: kind, name,
    display_name, description, source, parent_principal_id,
    is_active, avatar_url.
    """

    # Discriminator. Defaults to USER so legacy ``User(...)`` calls
    # (now aliased to ``Principal(...)``) continue to work without
    # being updated; ORG / GROUP creation paths already pass
    # ``kind=`` explicitly.
    kind: PrincipalType = Field(
        default=PrincipalType.USER,
        sa_column=Column(SQLEnum(PrincipalType), nullable=False),
    )

    # Stable identifier for the principal. Uniqueness is partitioned:
    # GROUP has its own namespace; USER / ORG / SYSTEM share one. The
    # GROUP partition exists because IdP-supplied group names commonly
    # coincide with admin-chosen Org names — forcing them globally
    # unique would break OIDC/SAML group sync the moment an admin
    # happens to name an Org the same as an IdP group. The shared
    # namespace for the admin-curated kinds is conservative: most
    # name lookups already scope by kind (``init_platform_principal_id``
    # → ORG, ``get_default_cluster_principal`` → SYSTEM, the
    # ``<owner-name>/<route>`` URL resolver → ORG), but
    # ``get_by_username`` (login) does not, and relies on USER / ORG /
    # SYSTEM not colliding to avoid mis-resolving a login to an ORG
    # row. Matches the k8s convention where ``metadata.name`` is the
    # URL-safe identifier — the upcoming GPU service API and k8s
    # namespace plumbing both key off this column directly.
    #
    # * USER: the login name (what the legacy schema called
    #   ``username``). Not used as a URL prefix — USER-owned model
    #   routes don't exist, so format is unconstrained beyond the
    #   uniqueness check.
    # * ORG: user-supplied URL-safe identifier; appears as the
    #   ``<owner-name>/<route-name>`` prefix in inference URLs.
    # * GROUP: admin- or IdP-supplied identifier. Not used in URLs
    #   (groups are not route-owners), so format is unconstrained
    #   beyond the global uniqueness check.
    name: Optional[str] = Field(default=None, nullable=True)

    # Human-readable display label. For USER this was the legacy
    # ``full_name`` (falling back to login at creation time); for
    # ORG / GROUP it's the user-supplied label. Cluster-access
    # grants, model-route ACL pickers, and the Org members list all
    # render this string.
    display_name: Optional[str] = Field(default=None, nullable=True)

    description: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )

    # Where this principal originated. Local for admin/UI-created
    # rows; OIDC / SAML for IdP-synced rows. Meaningful for USER (auth
    # source) and GROUP (IdP-sync provenance); typically Local for ORG.
    source: AuthProviderEnum = Field(
        default=AuthProviderEnum.Local,
        sa_column=Column(
            SQLEnum(AuthProviderEnum),
            nullable=False,
            server_default=AuthProviderEnum.Local.value,
        ),
    )

    # Optional hierarchy edge. Currently unused for ORG / GROUP rows
    # (Group→Org affiliation is expressed via ``principal_memberships``
    # to preserve the soft-delete + role semantics there) but
    # available for future kinds that need a structural parent.
    parent_principal_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("principals.id", ondelete="CASCADE"),
            nullable=True,
        ),
    )

    # ``is_admin`` is the platform-wide superuser flag — meaningful
    # only on USER rows. ORG-membership ``OrgRole.OWNER`` is the
    # per-Org admin tier; always disambiguate ``is_platform_admin``
    # from ``org_role == OrgRole.OWNER`` in code.
    is_admin: bool = Field(default=False, nullable=False)
    is_active: bool = Field(default=True, nullable=False)
    avatar_url: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )

    def model_post_init(self, __context) -> None:
        # Defense-in-depth: ``is_admin`` is the platform-wide superuser
        # flag and the 20+ ``user.is_admin`` checks across the codebase
        # treat the bit uniformly. An ORG / GROUP / SYSTEM row with
        # ``is_admin=True`` would silently slip through them. Reject at
        # construction so a misuse fails loudly instead of escalating
        # privilege. ``model_post_init`` fires on both ``__init__`` and
        # ``model_validate`` paths; ``@model_validator`` would not, since
        # SQLModel skips Pydantic validation for ``table=True`` ``__init__``.
        if self.is_admin and self.kind != PrincipalType.USER:
            raise ValueError(
                f"is_admin=True is only valid for USER principals, "
                f"got kind={self.kind!r}"
            )


class Principal(PrincipalBase, BaseModelMixin, table=True):
    """Unified identity row.

    Four kinds:

    * ``USER`` — human accounts; have a login credential in
      ``user_passwords`` (local) or come from an SSO provider.
    * ``ORG`` — tenancy boundary; owns resources.
    * ``GROUP`` — IdP-synced group; membership confers Org roles on
      its members.
    * ``SYSTEM`` — internal service accounts for cluster bootstrap
      and per-worker registration. Which infra row a SYSTEM principal
      serves is recorded on the *infra* side
      (``clusters.system_principal_id`` / ``workers.system_principal_id``).
    """

    __tablename__ = 'principals'
    __table_args__ = (
        # Two partial unique indexes — see the ``name`` field comment
        # for the partitioning rationale (non-GROUP kinds share one
        # namespace; GROUP has its own). Both are declared here for
        # autogenerate parity with the migration. On MySQL the
        # migration falls back to a single plain (non-unique) index
        # since partial indexes are unsupported; uniqueness within
        # each partition is then enforced by the create routes.
        Index(
            'uix_principals_non_group_name',
            'name',
            unique=True,
            postgresql_where=text("kind <> 'GROUP' AND deleted_at IS NULL"),
        ),
        Index(
            'uix_principals_group_name',
            'name',
            unique=True,
            postgresql_where=text("kind = 'GROUP' AND deleted_at IS NULL"),
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)

    # Login credentials (hashed password + force-change flag) live in
    # the dedicated ``user_passwords`` table; see
    # :mod:`gpustack.schemas.user_passwords`. SSO-only USER rows and
    # system actors simply have no row there.

    # Inverse 1:1 back-populations of the ``system_principal_id`` FK
    # on ``clusters`` / ``workers``. ``foreign_keys`` pins each to the
    # ``system_principal_id`` column so the mapper doesn't conflate
    # them with the ``owner_principal_id`` FK on the same tables.
    # ``uselist=False`` because the column is UNIQUE — at most one
    # cluster / worker can claim a given SYSTEM principal.
    cluster: Optional[Cluster] = Relationship(
        back_populates="system_principal",
        sa_relationship_kwargs={
            "lazy": "noload",
            "uselist": False,
            "foreign_keys": "[Cluster.system_principal_id]",
        },
    )
    worker: Optional[Worker] = Relationship(
        back_populates="system_principal",
        sa_relationship_kwargs={
            "lazy": "noload",
            "uselist": False,
            "foreign_keys": "[Worker.system_principal_id]",
        },
    )
    # ApiKey carries two FKs to principals (user_id + owner_principal_id);
    # ``foreign_keys`` pins this back-population to the user-owner
    # column.
    api_keys: List["ApiKey"] = Relationship(
        back_populates='user',
        sa_relationship_kwargs={
            "cascade": "delete",
            "lazy": "noload",
            "foreign_keys": "[ApiKey.user_id]",
        },
    )


class PrincipalListParams(ListParams):
    kind: Optional[PrincipalType] = None
    sortable_fields: ClassVar[List[str]] = [
        "name",
        "display_name",
        "kind",
        "created_at",
        "updated_at",
    ]


class PrincipalPublic(SQLModel):
    id: int
    kind: PrincipalType
    name: Optional[str] = None
    display_name: Optional[str] = None
    description: Optional[str] = None
    source: AuthProviderEnum = AuthProviderEnum.Local
    created_at: datetime
    updated_at: datetime


PrincipalsPublic = PaginatedList[PrincipalPublic]


# --------------------------------------------------------------------
# PrincipalMembership
# --------------------------------------------------------------------


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
    # ``OrgRole`` whenever the parent is an ORG (regardless of whether
    # the member is a User or a Group — for a Group-member the role
    # propagates to every active user in the Group). NULL for
    # User-in-Group memberships (groups don't have role tiers).
    role: Optional[OrgRole] = Field(
        default=None,
        sa_column=Column(SQLEnum(OrgRole), nullable=True),
    )
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
    ``deleted_at IS NULL``; the route handlers enforce that invariant.
    """

    __tablename__ = 'principal_memberships'
    id: Optional[int] = Field(default=None, primary_key=True)


# --------------------------------------------------------------------
# Platform principal resolver
# --------------------------------------------------------------------

_PLATFORM_PRINCIPAL_ID_INITIALIZED: bool = False
_AUTHENTICATED_PRINCIPAL_ID_INITIALIZED: bool = False


async def init_platform_principal_id(session: AsyncSession) -> int:
    """Resolve and bind the platform principal id at server startup.
    Must run after the platform principal row exists (the
    multi-tenancy foundation migration seeds it; the
    identity-consolidation migration may renumber it).

    Idempotent. Callable from tests.
    """
    global _PLATFORM_PRINCIPAL_ID, _PLATFORM_PRINCIPAL_ID_INITIALIZED
    # Platform principal is always an ORG — scope by kind so an
    # IdP-synced GROUP that happens to be named 'default' can't shadow
    # it under the partitioned name uniqueness model.
    p = await Principal.one_by_fields(
        session=session,
        fields={'kind': PrincipalType.ORG, 'name': PLATFORM_PRINCIPAL_NAME},
    )
    if p is None:
        raise RuntimeError(
            "platform principal not found; database may be uninitialized"
        )
    _PLATFORM_PRINCIPAL_ID = p.id
    _PLATFORM_PRINCIPAL_ID_INITIALIZED = True
    return p.id


async def init_authenticated_principal_id(session: AsyncSession) -> int:
    """Resolve and bind the ``system/authenticated`` group principal id.

    The row is seeded by the shared-GPU-services migration alongside
    the cluster_access backfill — same pattern as the platform
    principal — so this is a read-only resolver. Raises if absent so
    a missing seed surfaces loudly instead of letting auth-resolution
    silently widen.
    """
    global _AUTHENTICATED_PRINCIPAL_ID, _AUTHENTICATED_PRINCIPAL_ID_INITIALIZED
    p = await Principal.one_by_fields(
        session=session,
        fields={'kind': PrincipalType.GROUP, 'name': AUTHENTICATED_PRINCIPAL_NAME},
    )
    if p is None:
        raise RuntimeError(
            f"Authenticated principal not found by name="
            f"{AUTHENTICATED_PRINCIPAL_NAME!r}; database may be uninitialized"
        )
    _AUTHENTICATED_PRINCIPAL_ID = p.id
    _AUTHENTICATED_PRINCIPAL_ID_INITIALIZED = True
    return p.id


async def get_platform_principal_id(session: AsyncSession) -> int:
    """Read the platform principal's id, resolving by name if startup
    init has not been performed yet (tests, scripts).
    """
    if _PLATFORM_PRINCIPAL_ID_INITIALIZED:
        return _PLATFORM_PRINCIPAL_ID
    return await init_platform_principal_id(session)


async def get_authenticated_principal_id(session: AsyncSession) -> int:
    """Read the ``system/authenticated`` principal id, seeding it on
    first call. Counterpart to :func:`get_platform_principal_id`.
    """
    if _AUTHENTICATED_PRINCIPAL_ID_INITIALIZED:
        return _AUTHENTICATED_PRINCIPAL_ID
    return await init_authenticated_principal_id(session)


def _reset_platform_principal_for_tests() -> None:
    """Test hook — clears the initialised flag and resets the value."""
    global _PLATFORM_PRINCIPAL_ID, _PLATFORM_PRINCIPAL_ID_INITIALIZED
    global _AUTHENTICATED_PRINCIPAL_ID, _AUTHENTICATED_PRINCIPAL_ID_INITIALIZED
    _PLATFORM_PRINCIPAL_ID = 1
    _PLATFORM_PRINCIPAL_ID_INITIALIZED = False
    _AUTHENTICATED_PRINCIPAL_ID = 0
    _AUTHENTICATED_PRINCIPAL_ID_INITIALIZED = False
