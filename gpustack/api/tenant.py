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
from gpustack.schemas.clusters import Cluster
from gpustack.schemas.principals import (
    OrgRole,
    Principal,
    PrincipalMembership,
    PrincipalType,
    get_authenticated_principal_id,
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
    # Owner-Org principal ids of the clusters in ``accessible_cluster_ids``.
    # Surfaces "which Orgs run infra I'm allowed to touch" — used by
    # cluster-bound resource visibility (PV types, future analogs)
    # without re-querying ``clusters`` per call.
    accessible_cluster_owner_ids: Set[int] = field(default_factory=set)
    # True when ``current_principal_id`` is the user's own
    # USER-principal (personal scope) rather than an ORG-principal.
    # Lets endpoints treat the "owner of a one-member namespace" case
    # differently from a real multi-user Org owner — e.g.
    # ``scope='all'`` on the Usage page should behave like ``self``
    # here (no other tenants share this scope).
    current_is_personal_scope: bool = False
    # For SYSTEM principals tied to exactly one cluster — worker
    # registration accounts (workers.system_principal_id) and cluster
    # bootstrap accounts (clusters.system_principal_id) — the cluster
    # they serve and that cluster's owner principal. Read paths use
    # these to scope the SYSTEM bypass down to the principal's own
    # cluster instead of every tenant's rows. Both stay None for the
    # legacy ``config.token`` principal (in-memory, platform-level
    # secret), which keeps the full bypass.
    scoped_cluster_id: Optional[int] = None
    scoped_cluster_owner_id: Optional[int] = None

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
) -> tuple[Set[int], Set[int]]:
    """Cluster ids + their owner-principal ids reachable from any of:
    org, user, joined group, or the built-in ``system/authenticated``
    group.

    ``system/authenticated`` is always part of the lookup set — every
    authenticated caller is an implicit member of it. ClusterAccess
    rows written against it become global grants that any logged-in
    user inherits. Default-Org clusters use this mechanism:
    ``create_cluster`` seeds one such row per Default-Org cluster,
    which UI surfaces as an "Everyone" grant that admin can revoke or
    re-apply.

    Returns ``(cluster_ids, owner_principal_ids)`` so downstream code
    (PV-type visibility, etc.) can scope cluster-bound resources by
    "Orgs whose infrastructure I can use" without re-querying
    ``clusters``.
    """
    principal_ids = [
        user_principal_id,
        *group_principal_ids,
        # Async getter so CLI / offline call sites that never ran
        # ``_init_data`` fail loudly (init raises) rather than silently
        # use the uninitialised sentinel ``0`` — which would make the
        # IN-clause skip every ``system/authenticated`` grant.
        await get_authenticated_principal_id(session),
    ]
    if org_principal_id is not None:
        principal_ids.append(org_principal_id)

    stmt = (
        select(Cluster.id, Cluster.owner_principal_id)
        .join(ClusterAccess, ClusterAccess.cluster_id == Cluster.id)
        .where(
            ClusterAccess.principal_id.in_(principal_ids),
            Cluster.deleted_at.is_(None),
        )
    )
    cluster_ids: Set[int] = set()
    owner_ids: Set[int] = set()
    for cid, oid in (await session.exec(stmt)).all():
        cluster_ids.add(cid)
        if oid is not None:
            owner_ids.add(oid)
    return cluster_ids, owner_ids


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
    accessible_cluster_owner_ids: Set[int] = set()
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
            (
                accessible_cluster_ids,
                accessible_cluster_owner_ids,
            ) = await _accessible_clusters(
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
            (
                accessible_cluster_ids,
                accessible_cluster_owner_ids,
            ) = await _accessible_clusters(
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

    # Resolve the cluster a SYSTEM service account belongs to. The
    # ``worker`` / ``cluster`` back-references are eager-loaded by
    # ``UserService.get_by_id`` on the API-key auth path; the legacy
    # ``config.token`` principal is in-memory and has neither, so it
    # stays unscoped (full bypass).
    scoped_cluster_id = None
    scoped_cluster_owner_id = None
    if user.kind == PrincipalType.SYSTEM:
        if user.worker is not None:
            scoped_cluster_id = user.worker.cluster_id
            scoped_cluster_owner_id = user.worker.owner_principal_id
        elif user.cluster is not None:
            scoped_cluster_id = user.cluster.id
            scoped_cluster_owner_id = user.cluster.owner_principal_id

    ctx = TenantContext(
        user=user,
        is_platform_admin=is_platform_admin,
        current_principal_id=current_principal_id,
        org_role=org_role,
        accessible_cluster_ids=accessible_cluster_ids,
        accessible_cluster_owner_ids=accessible_cluster_owner_ids,
        current_is_personal_scope=current_is_personal_scope,
        scoped_cluster_id=scoped_cluster_id,
        scoped_cluster_owner_id=scoped_cluster_owner_id,
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

    NOTE on reads: for SYSTEM principals tied to one cluster, the read
    helpers below narrow this bypass to that cluster's rows via
    :func:`cluster_scoped_system` BEFORE consulting this function. This
    function stays permissive because write/role gates
    (``require_org_role``, ``assert_org_owned_writable``) rely on the
    SYSTEM bypass regardless of cluster.
    """
    if ctx.user is not None and ctx.user.kind == PrincipalType.SYSTEM:
        return True
    if ctx.is_platform_admin and ctx.current_principal_id is None:
        return True
    return False


def cluster_scoped_system(ctx: Optional[TenantContext]) -> bool:
    """True for SYSTEM service accounts that belong to exactly one
    cluster (worker registration / cluster bootstrap accounts). Their
    reads of cluster-bound resources are narrowed to that cluster: an
    Org-owned cluster's credentials must not read other tenants' rows.
    The legacy ``config.token`` principal has no cluster linkage and is
    excluded (full bypass — it's the platform-level secret)."""
    return (
        ctx is not None
        and ctx.user is not None
        and ctx.user.kind == PrincipalType.SYSTEM
        and ctx.scoped_cluster_id is not None
    )


# Sentinel distinguishing "resource has no cluster_id attribute" from
# "cluster_id is NULL" in scoped-row checks.
_UNSET = object()


def scoped_cluster_row_visible(ctx: TenantContext, resource: Any) -> bool:
    """Row-level mirror of the scoped-SYSTEM cluster filter.

    Visible when the row belongs to the principal's cluster, or isn't
    pinned to any cluster (NULL ``cluster_id`` — e.g. models that rely
    on default-cluster resolution), or isn't cluster-bound at all (no
    ``cluster_id`` attribute — such resources keep the plain SYSTEM
    bypass).
    """
    cluster_id = getattr(resource, "cluster_id", _UNSET)
    if cluster_id is _UNSET:
        return True
    return cluster_id is None or cluster_id == ctx.scoped_cluster_id


def _scoped_cluster_conditions(ctx: TenantContext, model: Any) -> List[Any]:
    """List-query mirror of :func:`scoped_cluster_row_visible`."""
    from sqlalchemy import or_

    if not hasattr(model, "cluster_id"):
        return []
    return [
        or_(
            model.cluster_id == ctx.scoped_cluster_id,
            model.cluster_id.is_(None),
        )
    ]


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
    if cluster_scoped_system(ctx):
        return _scoped_cluster_conditions(ctx, model)
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

    if cluster_scoped_system(ctx):
        # A cluster-bound service account sees its own cluster row only.
        return [model.id == ctx.scoped_cluster_id]
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


def assert_cluster_visible(
    ctx: TenantContext,
    cluster: Any,
    *,
    not_found_message: str = "Cluster not found",
) -> None:
    """404 if the caller can't see this cluster (own-principal OR cluster_access)."""
    if cluster is None:
        raise NotFoundException(message=not_found_message)
    if cluster_scoped_system(ctx):
        if cluster.id == ctx.scoped_cluster_id:
            return
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
    allow_member: bool = False,
) -> None:
    """403 if the caller can't mutate an org-owned row.

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

    ``allow_member`` (default False) keeps infra-level writes
    (cluster, cloud_credential, worker_pool, inference_backend, PV
    type, template) gated on the OWNER role. Workload-level rows
    (GPU instance, SSH key, persistent volume) set it True so an Org
    Member can manage workloads scoped to the Org they belong to —
    consistent with how Personal-scope users manage their own.
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
                "the current organization"
            )
        )
    # Platform admin acting-as the Org passes the role check unconditionally.
    # ``allow_member=True`` accepts any Org role AND Personal scope (the
    # user managing their own USER-principal namespace). ``allow_member=False``
    # is the infra-level gate: only Admin or Org Owner — Personal scope has
    # no Org-role ladder, so it falls on the non-OWNER side here too.
    if ctx.is_platform_admin or allow_member:
        return
    if ctx.current_is_personal_scope or ctx.org_role != OrgRole.OWNER:
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
    allow_member: bool = False,
) -> None:
    """Decide whether the caller can create a row owned by
    ``input_owner_principal_id``.

    - Platform admin: any value (including NULL = global)
    - Org owner: must equal ``current_principal_id``; can't create global
    - Personal scope: must equal the caller's own USER-principal id.
      No Org-role ladder applies — the user owns their own namespace.

    ``allow_member`` (default False) keeps infra-level creates
    (cluster, cloud_credential, worker_pool, inference_backend, PV
    type, template) gated on the OWNER role. Workload-level rows
    (GPU instance, SSH key, persistent volume) set it True so an Org
    Member can spin up workloads in the Org they belong to — same
    permission ladder as Personal scope, just stamped on the Org.
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
    # ``allow_member=True`` accepts any Org role AND Personal scope (the
    # user creating in their own USER-principal namespace).
    # ``allow_member=False`` is the infra-level gate: only Admin or Org
    # Owner — Personal scope has no Org-role ladder, so it falls on the
    # non-OWNER side here too.
    if allow_member:
        return
    if ctx.current_is_personal_scope or ctx.org_role != OrgRole.OWNER:
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

    if cluster_scoped_system(ctx):
        if scoped_cluster_row_visible(ctx, resource):
            return
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
    given roles in `current_principal_id`. Platform admins and SYSTEM
    principals (worker / cluster service accounts) always pass — mirrors
    ``bypass_tenant_filter`` so write endpoints shared with worker callbacks
    (e.g. benchmark metrics) keep working.
    """

    async def _dep(
        ctx: Annotated[TenantContext, Depends(get_tenant_context)],
    ) -> TenantContext:
        if bypass_tenant_filter(ctx) or ctx.is_platform_admin:
            return ctx
        if ctx.current_principal_id is None:
            raise OrgRoleError(message="Organization context required")
        ctx.assert_org_role(*allowed)
        return ctx

    return _dep
