"""Principal — the unified identity model.

Every namespaced actor in the system (a User, an Organization, a User
Group) is a row in the ``principals`` table. There is no separate
``users`` extension table any more: User-specific columns (credentials,
admin / system flags, cluster / worker FKs, …) live directly on
``principals`` and are simply NULL / default-valued on ORG and GROUP
rows.

All three kinds are peer-level. A Group's relationship to an Org (if
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
    UniqueConstraint,
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


class UserRole(Enum):
    """System-actor role on USER rows (cluster / worker service
    accounts). Orthogonal to organization membership roles.
    """

    Worker = "Worker"
    Cluster = "Cluster"


# Canonical slug of the built-in platform Org-principal. Created by the
# multi-tenancy foundation migration; system / infrastructure resources
# default to it.
PLATFORM_PRINCIPAL_SLUG = 'default'

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


def platform_principal_id() -> int:
    """Sync getter for the platform principal id.

    Every call site that needs the *current* id (schema field defaults
    via ``default_factory``; runtime comparisons; fallback values) must
    use this function rather than reading the module-private
    ``_PLATFORM_PRINCIPAL_ID`` directly.
    """
    return _PLATFORM_PRINCIPAL_ID


# Public alias for ``default_factory=`` on SQLModel fields. Identical
# to :func:`platform_principal_id`; kept distinct to make field-default
# usage greppable.
_platform_principal_id = platform_principal_id


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
    """Columns common to every Principal kind, plus USER-only columns
    that are NULL / default-valued on ORG and GROUP rows.

    Reads as "everything that used to live on ``users`` or
    ``principals`` before consolidation", on one schema.
    """

    # Discriminator. Defaults to USER so legacy ``User(...)`` calls
    # (now aliased to ``Principal(...)``) continue to work without
    # being updated; ORG / GROUP creation paths already pass
    # ``kind=`` explicitly.
    kind: PrincipalType = Field(
        default=PrincipalType.USER,
        sa_column=Column(SQLEnum(PrincipalType), nullable=False),
    )

    # Stable identifier for the principal. Globally unique among
    # non-NULL values.
    #
    # * USER: the login name (what the legacy schema called
    #   ``username``). Not used as a URL prefix — USER-owned model
    #   routes don't exist, so format is unconstrained beyond the
    #   uniqueness check.
    # * ORG: user-supplied URL-safe slug; appears as the
    #   ``<owner-slug>/<route-name>`` prefix in inference URLs.
    # * GROUP: NULL (groups have no canonical identifier; their
    #   ``name`` is uniquely indexed instead).
    slug: Optional[str] = Field(default=None, nullable=True)

    # Display name. For USER this is the human label (legacy
    # ``full_name``, falling back to login at creation time). For
    # ORG / GROUP it's the user-supplied label.
    name: Optional[str] = Field(default=None, nullable=True)

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

    # ---- USER-only columns (NULL / default-valued on ORG / GROUP) ----
    # Login (``username``) and display name (``full_name``) collapsed
    # into ``slug`` and ``name`` above — same columns are used for all
    # principal kinds.

    is_admin: bool = Field(default=False, nullable=False)
    is_active: bool = Field(default=True, nullable=False)
    avatar_url: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )

    # ``is_system`` and ``role`` describe internal system actors
    # (cluster / worker service accounts) — still USER-kind, not the
    # SYSTEM kind that the Phase 3 cleanup will introduce.
    is_system: bool = Field(default=False, nullable=False)
    role: Optional[UserRole] = Field(
        default=None,
        description="Role of the system user (Worker / Cluster)",
    )
    cluster_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("clusters.id", ondelete="CASCADE")),
    )
    worker_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("workers.id", ondelete="CASCADE")),
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

    USER rows carry credentials and the system-actor fields
    (``cluster_id`` / ``worker_id`` / ``role``). ORG and GROUP rows
    leave those NULL. The discriminator is :attr:`kind`.
    """

    __tablename__ = 'principals'
    __table_args__ = (
        UniqueConstraint('slug', name='uix_principals_slug'),
        # Group names are globally unique among active groups.
        # Postgres supports partial unique indexes — declared here for
        # autogenerate parity with the migration. MySQL / SQLite have
        # no partial index, so the migration creates a plain non-unique
        # index and the route handlers enforce uniqueness in code.
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

    # ``foreign_keys`` disambiguates these from ``clusters.owner_principal_id``
    # / ``workers.owner_principal_id``, which are the inverse direction
    # (Cluster→Principal, Worker→Principal) and would otherwise produce
    # an AmbiguousForeignKeysError on mapper configuration.
    cluster: Optional[Cluster] = Relationship(
        back_populates="cluster_users",
        sa_relationship_kwargs={
            "lazy": "noload",
            "foreign_keys": "[Principal.cluster_id]",
        },
    )
    worker: Optional[Worker] = Relationship(
        sa_relationship_kwargs={
            "lazy": "noload",
            "foreign_keys": "[Principal.worker_id]",
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
        "slug",
        "kind",
        "created_at",
        "updated_at",
    ]


class PrincipalPublic(SQLModel):
    id: int
    kind: PrincipalType
    slug: Optional[str] = None
    name: Optional[str] = None
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


async def init_platform_principal_id(session: AsyncSession) -> int:
    """Resolve and bind the platform principal id at server startup.
    Must run after the platform principal row exists (the
    multi-tenancy foundation migration seeds it; the
    identity-consolidation migration may renumber it).

    Idempotent. Callable from tests.
    """
    global _PLATFORM_PRINCIPAL_ID, _PLATFORM_PRINCIPAL_ID_INITIALIZED
    p = await Principal.one_by_field(
        session=session, field='slug', value=PLATFORM_PRINCIPAL_SLUG
    )
    if p is None:
        raise RuntimeError(
            "platform principal not found; database may be uninitialized"
        )
    _PLATFORM_PRINCIPAL_ID = p.id
    _PLATFORM_PRINCIPAL_ID_INITIALIZED = True
    return p.id


async def get_platform_principal_id(session: AsyncSession) -> int:
    """Read the platform principal's id, resolving by slug if startup
    init has not been performed yet (tests, scripts).
    """
    if _PLATFORM_PRINCIPAL_ID_INITIALIZED:
        return _PLATFORM_PRINCIPAL_ID
    return await init_platform_principal_id(session)


def _reset_platform_principal_for_tests() -> None:
    """Test hook — clears the initialised flag and resets the value."""
    global _PLATFORM_PRINCIPAL_ID, _PLATFORM_PRINCIPAL_ID_INITIALIZED
    _PLATFORM_PRINCIPAL_ID = 1
    _PLATFORM_PRINCIPAL_ID_INITIALIZED = False
