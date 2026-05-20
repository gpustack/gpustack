"""Tenant context resolution for multi-tenant request handling.

Each authenticated request resolves to a TenantContext that captures:
- the user identity
- whether they are a platform-level super-admin
- which principal the request is operating as (current_principal_id) —
  either an Org-principal the user is a member of, or the user's own
  USER-principal for personal scope
- which Org-level role they hold there (admin / user) — None for
  personal scope
- which clusters are accessible in that context

Read resolution order for current_principal_id:
1. If authenticated via API key, use api_key.owner_principal_id (header
   is ignored)
2. Else, X-Organization-Id request header if provided
3. Else, for non-admin: user.id (the user's USER-principal id — NOT
   NULL by schema, since every user IS a principal, so non-admin
   requests are structurally never context-less and can't bypass
   tenant filters with a NULL current_principal_id)
4. Else, for platform admin: None — "act across all principals", read
   paths skip tenant filters via bypass_tenant_filter
"""

from dataclasses import dataclass, field
from typing import Annotated, Any, List, Optional, Set

from fastapi import Depends, Header, Request
from sqlalchemy.orm import aliased
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.auth import get_current_user
from gpustack.api.exceptions import (
    ForbiddenException,
    InvalidException,
    NotFoundException,
)
from gpustack.schemas.api_keys import ApiKey
from gpustack.schemas.cluster_access import ClusterAccess
from gpustack.schemas.principals import (
    OrgRole,
    Principal,
    PrincipalMembership,
    PrincipalType,
)
from gpustack.schemas.users import User
from gpustack.server.db import get_session


PlatformAdminError = ForbiddenException
OrgRoleError = ForbiddenException


@dataclass
class TenantContext:
    """Per-request tenant resolution result."""

    # The authenticated principal for this request. Can be any kind
    # ``get_current_user`` resolves to — USER (cookie / API key /
    # password), SYSTEM (worker / cluster bootstrap via Basic auth or
    # service API key) — so the field is typed as ``Principal`` and
    # call sites that need to discriminate read ``user.kind`` directly
    # (see :func:`bypass_tenant_filter`).
    user: Principal
    is_platform_admin: bool
    # The principal the request is operating as. ORG-principal for an
    # Org context; the user's own USER-principal for personal scope;
    # None for platform-admin "All" mode.
    current_principal_id: Optional[int]
    org_role: Optional[OrgRole]
    accessible_cluster_ids: Set[int] = field(default_factory=set)
    # True when ``current_principal_id`` is the user's own
    # USER-principal (personal scope) rather than an ORG-principal.
    # Lets endpoints treat the "owner of a one-member namespace" case
    # differently from a real multi-user Org owner — e.g.
    # ``scope='all'`` on the Usage page should behave like ``self``
    # here (no other tenants share this scope).
    current_is_personal_scope: bool = False

    @property
    def has_org_context(self) -> bool:
        return self.current_principal_id is not None

    def assert_org_role(self, *allowed: OrgRole) -> None:
        """Raise if the caller doesn't hold one of the ``allowed`` roles
        in the current Org. Platform admins bypass.
        """
        if self.is_platform_admin:
            return
        if self.org_role is None or self.org_role not in allowed:
            raise OrgRoleError(message="Insufficient organization role")

    def target_principal_id_for_write(self) -> Optional[int]:
        """Resolve the principal a CREATE / write request should land in.

        Reads happily honor the platform-admin "no current_principal_id
        = all-orgs" mode, but writes need an actual principal to stamp
        on the new row. When the request didn't pin a context (no
        header on an admin request, or a built-in client like the OSS
        host that never sends ``X-Organization-Id``), fall back to the
        user's own USER-principal — guaranteed non-null by schema —
        instead of failing.
        """
        if self.current_principal_id is not None:
            return self.current_principal_id
        return self.user.id


async def _resolve_effective_org_role(
    session: AsyncSession,
    user_principal_id: int,
    org_principal_id: int,
) -> Optional[OrgRole]:
    """Effective OrgRole of a user inside an org.

    Two membership paths confer a role:

    - Direct: ``(parent=org, member=user)`` row.
    - Via group: ``(parent=org, member=group)`` AND ``(parent=group,
      member=user)`` — a Group joined to the Org propagates its role
      to every active user in the Group.

    Returns the max role across all paths (OWNER beats MEMBER). None
    if the user has no path into the org. Soft-deleted membership and
    group rows are ignored — removal cuts the role immediately while
    leaving an audit trail.
    """
    direct_stmt = select(PrincipalMembership.role).where(
        PrincipalMembership.parent_principal_id == org_principal_id,
        PrincipalMembership.member_principal_id == user_principal_id,
        PrincipalMembership.deleted_at.is_(None),
    )

    group_pm = aliased(PrincipalMembership)
    org_pm = aliased(PrincipalMembership)
    via_group_stmt = (
        select(org_pm.role)
        .join(group_pm, group_pm.parent_principal_id == org_pm.member_principal_id)
        .join(Principal, Principal.id == group_pm.parent_principal_id)
        .where(
            org_pm.parent_principal_id == org_principal_id,
            org_pm.deleted_at.is_(None),
            group_pm.member_principal_id == user_principal_id,
            group_pm.deleted_at.is_(None),
            Principal.kind == PrincipalType.GROUP,
            Principal.deleted_at.is_(None),
        )
    )

    # One round trip: direct + transitive memberships in a single
    # UNION ALL. Auth context is resolved on every request that
    # carries an Org context, so collapsing two SELECTs to one is
    # measurable at scale.
    #
    # ``select(Col).union_all(...)`` returns ``Row`` tuples (one cell
    # each) rather than scalars — a SQLAlchemy quirk where the union
    # output is treated as a multi-column shape regardless of input
    # arity. ``OrgRole.OWNER in [Row(OWNER)]`` is False, so we have
    # to unpack the cell explicitly. Calling ``.scalars()`` on the
    # union result is the canonical fix; ``r[0] for r in ...`` would
    # work too but is less obvious about what's going on.
    stmt = direct_stmt.union_all(via_group_stmt)
    result = await session.exec(stmt)
    roles = [r for r in result.scalars().all() if r is not None]
    if not roles:
        return None
    return OrgRole.OWNER if OrgRole.OWNER in roles else OrgRole.MEMBER


async def _user_group_principal_ids(
    session: AsyncSession,
    user_principal_id: int,
) -> List[int]:
    """All GROUP-principal ids the user is an active member of.

    Groups are no longer org-scoped — a Group is a peer-level principal
    that may be a member of zero or more Orgs. Cluster_access grants
    against a Group apply wherever the user acts, mirroring how AD/IAM
    groups behave.
    """
    stmt = (
        select(PrincipalMembership.parent_principal_id)
        .join(
            Principal,
            Principal.id == PrincipalMembership.parent_principal_id,
        )
        .where(
            PrincipalMembership.member_principal_id == user_principal_id,
            PrincipalMembership.deleted_at.is_(None),
            Principal.kind == PrincipalType.GROUP,
            Principal.deleted_at.is_(None),
        )
    )
    return list((await session.exec(stmt)).all())


async def _accessible_clusters(
    session: AsyncSession,
    user_principal_id: int,
    org_principal_id: Optional[int],
    group_principal_ids: List[int],
) -> Set[int]:
    """Cluster ids reachable from any of: org, user, or any joined group."""
    principal_ids = [user_principal_id, *group_principal_ids]
    if org_principal_id is not None:
        principal_ids.append(org_principal_id)

    if not principal_ids:
        return set()

    stmt = select(ClusterAccess.cluster_id).where(
        ClusterAccess.principal_id.in_(principal_ids),
    )
    return set((await session.exec(stmt)).all())


def _resolve_requested_principal_id(
    request: Request,
    user: Principal,
    header_value: Optional[str],
) -> Optional[int]:
    api_key: Optional[ApiKey] = getattr(request.state, "api_key", None)
    if api_key is not None and api_key.owner_principal_id is not None:
        return api_key.owner_principal_id

    if header_value:
        try:
            return int(header_value)
        except ValueError as exc:
            # 400 — the header is structurally bad, not a permission issue.
            raise InvalidException(message="Invalid X-Organization-Id") from exc

    # Platform admins default to "no context" (cross-principal platform
    # view) when nothing is supplied. They opt into act-as mode by
    # sending X-Organization-Id explicitly. Non-admins fall back to
    # their own USER-principal (NOT NULL by schema), which guarantees
    # ``current_principal_id`` is never None for non-admin callers —
    # closing the bypass that would otherwise let an empty filter run
    # against tenant-scoped lists.
    if user.is_admin:
        return None
    return user.id


async def get_tenant_context(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
    x_organization_id: Annotated[Optional[str], Header()] = None,
) -> TenantContext:
    """Resolve the per-request TenantContext.

    Result is cached on `request.state.tenant_context` so multiple downstream
    dependencies in the same request share one resolution.
    """
    if hasattr(request.state, "tenant_context"):
        return request.state.tenant_context

    is_platform_admin = bool(user.is_admin)
    current_principal_id = _resolve_requested_principal_id(
        request, user, x_organization_id
    )

    org_role: Optional[OrgRole] = None
    accessible_cluster_ids: Set[int] = set()
    current_is_personal_scope = False

    if current_principal_id is not None and user.kind != PrincipalType.SYSTEM:
        # Personal scope short-circuit: when the request points at the
        # caller's own USER-principal there's no org membership to
        # resolve. Group grants still apply (the user's groups are
        # principals in their own right), so include them in the
        # cluster_access lookup.
        if current_principal_id == user.id:
            current_is_personal_scope = True
            group_ids = await _user_group_principal_ids(session, user.id)
            accessible_cluster_ids = await _accessible_clusters(
                session,
                user.id,
                None,
                group_ids,
            )
        else:
            org_role = await _resolve_effective_org_role(
                session, user.id, current_principal_id
            )
            if org_role is None and not is_platform_admin:
                # Non-admin users cannot operate as a principal they are
                # not a member of (directly or via a Group).
                raise ForbiddenException(
                    message=(f"Not a member of organization " f"{current_principal_id}")
                )

            group_ids = await _user_group_principal_ids(session, user.id)
            accessible_cluster_ids = await _accessible_clusters(
                session,
                user.id,
                current_principal_id,
                group_ids,
            )

            # Validate the org-principal exists and isn't soft-deleted
            # before letting the request continue. Org soft-delete is
            # "removed for users" — block context resolution against it
            # so the membership row (still present, since CASCADE
            # doesn't fire on soft delete) can't be used to keep
            # operating in a logically-removed Org.
            org_row = await Principal.first_by_field(
                session, "id", current_principal_id
            )
            if (
                org_row is None
                or org_row.deleted_at is not None
                or org_row.kind != PrincipalType.ORG
            ):
                raise NotFoundException(
                    message=(f"Organization {current_principal_id} not found")
                )

    ctx = TenantContext(
        user=user,
        is_platform_admin=is_platform_admin,
        current_principal_id=current_principal_id,
        org_role=org_role,
        accessible_cluster_ids=accessible_cluster_ids,
        current_is_personal_scope=current_is_personal_scope,
    )
    request.state.tenant_context = ctx
    return ctx


async def require_platform_admin(
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
) -> TenantContext:
    """Allow only platform-level super-admin (`users.is_admin = True`)."""
    if not ctx.is_platform_admin:
        raise PlatformAdminError(message="Platform admin permission required")
    return ctx


def bypass_tenant_filter(ctx: TenantContext) -> bool:
    """Identify request contexts that should not be tenant-scoped.

    Two categories bypass:
    - Platform admin with no principal context (cross-principal platform
      view).
    - System principals (worker / cluster service accounts that the server
      itself spawns). They authenticate as ``kind=SYSTEM`` and need to
      read every tenant's resources to do their job — e.g. a worker
      fetching the Model row for an instance assigned to it.
    """
    if ctx.user is not None and ctx.user.kind == PrincipalType.SYSTEM:
        return True
    if ctx.is_platform_admin and ctx.current_principal_id is None:
        return True
    return False


def tenant_list_conditions(
    ctx: TenantContext,
    model: Any,
) -> List[Any]:
    """Build SQLAlchemy WHERE clauses to scope a list query to the caller.

    Visibility model:
    - System principals (workers / cluster service accounts) and
      platform admin without org context see everything — returns no
      conditions.
    - Everyone else with a principal context filters by
      ``model.owner_principal_id == ctx.current_principal_id``.
      Membership in the org is already enforced by
      ``get_tenant_context``.
    """
    conditions: List[Any] = []
    if bypass_tenant_filter(ctx):
        return conditions

    if ctx.current_principal_id is not None and hasattr(model, "owner_principal_id"):
        conditions.append(model.owner_principal_id == ctx.current_principal_id)

    return conditions


def cluster_visibility_conditions(
    ctx: TenantContext,
    model: Any,
) -> List[Any]:
    """Visibility filter specific to Cluster-like infrastructure rows.

    Clusters can be visible to a non-admin caller through TWO independent
    paths, so the regular ``owner_principal_id`` equality filter would
    be too narrow:

    - **Own-principal cluster**
      (``cluster.owner_principal_id == current_principal_id``):
      the caller's BYO cluster.
    - **Granted via cluster_access** (``cluster.id`` ∈
      ``ctx.accessible_cluster_ids``): global clusters the admin
      authorised, or another principal's cluster sublet to us.

    Either path makes the cluster visible. System principals and
    platform admins (no-context) bypass entirely.
    """
    from sqlalchemy import or_

    if bypass_tenant_filter(ctx):
        return []

    or_clauses = []
    if ctx.current_principal_id is not None:
        or_clauses.append(model.owner_principal_id == ctx.current_principal_id)
    if ctx.accessible_cluster_ids:
        or_clauses.append(model.id.in_(ctx.accessible_cluster_ids))

    if not or_clauses:
        # No avenue to see anything; force an empty result rather than
        # leak the full table when accessible_cluster_ids is empty.
        return [model.id == -1]

    return [or_(*or_clauses)]


def cluster_resource_visibility_conditions(
    ctx: TenantContext,
    model: Any,
) -> List[Any]:
    """Visibility filter for resources that carry BOTH ``owner_principal_id``
    (denormalized from cluster) AND ``cluster_id`` — Worker, ModelFile,
    Benchmark, ModelEvaluation, etc.

    A row is visible if:
    - it's owned by the caller's current principal
      (``owner_principal_id`` match), OR
    - its cluster is granted via ``cluster_access`` (``cluster_id`` ∈
      ``accessible_cluster_ids``).

    NULL ``owner_principal_id`` rows live on global clusters; they're
    only visible through the second branch (cluster_access) for
    non-admin.
    """
    from sqlalchemy import or_

    if bypass_tenant_filter(ctx):
        return []

    or_clauses = []
    if ctx.current_principal_id is not None and hasattr(model, "owner_principal_id"):
        or_clauses.append(model.owner_principal_id == ctx.current_principal_id)
    if ctx.accessible_cluster_ids and hasattr(model, "cluster_id"):
        or_clauses.append(model.cluster_id.in_(ctx.accessible_cluster_ids))

    if not or_clauses:
        # No access path; force empty result rather than leak.
        anchor = getattr(model, "cluster_id", None) or getattr(model, "id", None)
        return [anchor == -1]
    return [or_(*or_clauses)]


def assert_cluster_resource_visible(
    ctx: TenantContext,
    resource: Any,
    *,
    not_found_message: str = "Resource not found",
) -> None:
    """Single-row mirror of ``cluster_resource_visibility_conditions``."""
    if resource is None:
        raise NotFoundException(message=not_found_message)
    if bypass_tenant_filter(ctx):
        return

    owner = getattr(resource, "owner_principal_id", None)
    cluster_id = getattr(resource, "cluster_id", None)

    if (
        ctx.current_principal_id is not None
        and owner is not None
        and owner == ctx.current_principal_id
    ):
        return
    if cluster_id is not None and cluster_id in ctx.accessible_cluster_ids:
        return
    raise NotFoundException(message=not_found_message)


def assert_cluster_visible(
    ctx: TenantContext,
    cluster: Any,
    *,
    not_found_message: str = "Cluster not found",
) -> None:
    """404 if the caller can't see this cluster (own-principal OR cluster_access)."""
    if cluster is None:
        raise NotFoundException(message=not_found_message)
    if bypass_tenant_filter(ctx):
        return
    cluster_owner = getattr(cluster, "owner_principal_id", None)
    if (
        ctx.current_principal_id is not None
        and cluster_owner is not None
        and cluster_owner == ctx.current_principal_id
    ):
        return
    if cluster.id in ctx.accessible_cluster_ids:
        return
    raise NotFoundException(message=not_found_message)


def assert_org_owned_writable(
    ctx: TenantContext,
    resource: Any,
    *,
    resource_label: str = "resource",
) -> None:
    """403 if the caller can't mutate an org-owned infrastructure row.

    Used for clusters / cloud_credentials / worker_pools / inference
    backends — anything with a nullable ``owner_principal_id`` and these
    write rules:

    - Platform admin / system principal → allowed (bypass via
      ``bypass_tenant_filter`` for "All" mode admin and system
      principals; admin in act-as falls through to row-owner check,
      where they're treated like an Org owner).
    - **Owned by current principal**: an Org owner can write; platform
      admin in act-as bypasses the role check (admin is admin
      everywhere, even when scoped to one Org).
    - **Global** (owner IS NULL): only "All"-mode admin — Org owners
      and admin-in-act-as cannot mutate Global rows directly. Resource
      handlers redirect such writes to the caller's own row instead.
    - **Other principal's row**: never writable for non-admin.
    """
    if bypass_tenant_filter(ctx):
        return
    res_owner = getattr(resource, "owner_principal_id", None)
    if res_owner is None:
        raise PlatformAdminError(
            message=f"Only platform admin can modify global {resource_label}"
        )
    if res_owner != ctx.current_principal_id:
        raise OrgRoleError(
            message=(
                f"{resource_label.capitalize()} does not belong to "
                "current organization"
            )
        )
    # Platform admin acting-as the Org passes the role check unconditionally;
    # for non-admin we require Org owner.
    if not ctx.is_platform_admin and ctx.org_role != OrgRole.OWNER:
        raise OrgRoleError(
            message=(
                f"Insufficient organization role to modify this " f"{resource_label}"
            )
        )


def assert_cluster_writable(
    ctx: TenantContext,
    cluster: Any,
) -> None:
    assert_org_owned_writable(ctx, cluster, resource_label="cluster")


def validate_owner_principal(
    input_owner_principal_id: Optional[int],
    ctx: TenantContext,
    *,
    resource_label: str = "resource",
) -> None:
    """Decide whether the caller can create a row owned by
    ``input_owner_principal_id``.

    - Platform admin: any value (including NULL = global)
    - Org owner: must equal ``current_principal_id``; can't create global
    """
    if ctx.is_platform_admin:
        return
    if input_owner_principal_id is None:
        raise InvalidException(
            message=(f"Only platform admin can create global {resource_label}s")
        )
    if (
        ctx.current_principal_id is None
        or input_owner_principal_id != ctx.current_principal_id
    ):
        raise InvalidException(
            message="owner_principal_id must match the current organization"
        )
    if ctx.org_role != OrgRole.OWNER:
        raise InvalidException(
            message=(f"Insufficient organization role to create a " f"{resource_label}")
        )


def assert_resource_visible(
    ctx: TenantContext,
    resource: Any,
    *,
    not_found_message: str = "Resource not found",
) -> None:
    """Raise 404 if the caller is not allowed to see ``resource``.

    Mirrors the semantics of ``tenant_list_conditions`` for single-item
    GET / PUT / DELETE handlers: same visibility rules, raised as 404
    rather than 403 to avoid leaking the existence of cross-tenant rows.
    """
    if resource is None:
        raise NotFoundException(message=not_found_message)

    if bypass_tenant_filter(ctx):
        return

    owner = getattr(resource, "owner_principal_id", None)
    if (
        ctx.current_principal_id is not None
        and owner is not None
        and owner != ctx.current_principal_id
    ):
        raise NotFoundException(message=not_found_message)


def require_org_role(*allowed: OrgRole):
    """Build a dependency that requires the requesting user to hold one of the
    given roles in `current_principal_id`. Platform admins always pass.
    """

    async def _dep(
        ctx: Annotated[TenantContext, Depends(get_tenant_context)],
    ) -> TenantContext:
        if ctx.is_platform_admin:
            return ctx
        if ctx.current_principal_id is None:
            raise OrgRoleError(message="Organization context required")
        ctx.assert_org_role(*allowed)
        return ctx

    return _dep
