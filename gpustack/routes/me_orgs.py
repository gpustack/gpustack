"""Self-service tenant endpoints — what orgs am I in, what clusters can I use."""

from typing import Dict, List

from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy.orm import aliased
from sqlmodel import select

from gpustack.api.exceptions import ForbiddenException, NotFoundException
from gpustack.schemas.cluster_access import ClusterAccess
from gpustack.schemas.clusters import Cluster, ClusterPublic
from gpustack.schemas.organizations import OrganizationPublic
from gpustack.schemas.principals import (
    OrgRole,
    Principal,
    PrincipalMembership,
    PrincipalType,
    get_authenticated_principal_id,
)
from gpustack.server.deps import CurrentUserDep, SessionDep, TenantContextDep

router = APIRouter()


class MyOrganization(BaseModel):
    organization: OrganizationPublic
    role: OrgRole

    model_config = {"from_attributes": True}


@router.get("/organizations", response_model=List[MyOrganization])
async def list_my_orgs(session: SessionDep, user: CurrentUserDep):
    """Return the org switcher list — user's Personal scope first,
    then any ORG-principals they're a member of (directly or via a
    Group-membership).

    "Personal" is no longer a stored Org row. After the principals
    refactor it's the user's own USER-principal, rendered through
    ``OrganizationPublic.from_principal`` with ``is_personal=True``.
    The DTO passes the user's real ``name`` / ``display_name`` through
    unmodified; the UI is responsible for substituting its localized
    "Personal" label when ``is_personal`` is set.
    """
    items: List[MyOrganization] = []

    # ``user`` IS the user-principal row (USER kind); no extra fetch.
    if user.deleted_at is None:
        items.append(
            MyOrganization(
                organization=OrganizationPublic.from_principal(user),
                role=OrgRole.OWNER,
            )
        )

    # Direct memberships: user joined the Org directly.
    direct_stmt = (
        select(PrincipalMembership.role, Principal)
        .join(
            Principal,
            Principal.id == PrincipalMembership.parent_principal_id,
        )
        .where(
            PrincipalMembership.member_principal_id == user.id,
            PrincipalMembership.deleted_at.is_(None),
            Principal.deleted_at.is_(None),
            Principal.kind == PrincipalType.ORG,
        )
    )

    # Transitive memberships: user is in a Group that is a Member of the Org.
    group_pm = aliased(PrincipalMembership)
    org_pm = aliased(PrincipalMembership)
    via_group_stmt = (
        select(org_pm.role, Principal)
        .join(group_pm, group_pm.parent_principal_id == org_pm.member_principal_id)
        .join(Principal, Principal.id == org_pm.parent_principal_id)
        .where(
            group_pm.member_principal_id == user.id,
            group_pm.deleted_at.is_(None),
            org_pm.deleted_at.is_(None),
            Principal.deleted_at.is_(None),
            Principal.kind == PrincipalType.ORG,
        )
    )

    # Two SELECTs here (not a single ``union_all``): each SELECT
    # returns a Principal ORM entity, and SA flattens entity columns
    # across a UNION, breaking the ``(role, Principal)`` row shape
    # we destructure below — it surfaces as
    # ``ValueError: too many values to unpack (expected 2)`` on the
    # ``/v2/users/me/organizations`` endpoint, fired for any user who
    # belongs to one or more Orgs. Two round trips on a UI-page-load
    # path is acceptable; the same optimization belongs only on hot
    # paths that select scalar columns (see ``api/tenant.py``'s
    # ``_resolve_effective_org_role``, which unions cleanly because
    # its SELECTs return only ``role``).
    best_by_org: Dict[int, tuple[OrgRole, Principal]] = {}
    for stmt in (direct_stmt, via_group_stmt):
        for role, org in (await session.exec(stmt)).all():
            effective = role or OrgRole.MEMBER
            existing = best_by_org.get(org.id)
            if existing is None or (
                existing[0] != OrgRole.OWNER and effective == OrgRole.OWNER
            ):
                best_by_org[org.id] = (effective, org)

    items.extend(
        MyOrganization(
            organization=OrganizationPublic.from_principal(org),
            role=role,
        )
        for role, org in best_by_org.values()
    )
    return items


@router.get("/organizations/{org_id}/clusters", response_model=List[ClusterPublic])
async def list_my_clusters_in_org(
    session: SessionDep, ctx: TenantContextDep, org_id: int
):
    """List clusters accessible to the caller in a specific Org context."""
    org = await Principal.one_by_id(session, org_id)
    if not org or org.deleted_at is not None or org.kind != PrincipalType.ORG:
        raise NotFoundException(message="Organization not found")

    if not ctx.is_platform_admin and ctx.current_principal_id != org_id:
        raise ForbiddenException(
            message="Cannot inspect clusters of an organization you are not in"
        )

    # All Group-principals the user is in. Groups are no longer
    # org-scoped — a Group is a peer-level principal that may carry
    # cluster_access grants applicable wherever the user acts.
    user_principal_id = ctx.user.id
    group_stmt = (
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
    group_principal_ids = list((await session.exec(group_stmt)).all())

    # ``system/authenticated`` covers Default-Org "shared with all
    # users" grants (and any future admin-applied global grant) on the
    # same code path as explicit principal grants — every authenticated
    # caller is an implicit member.
    principal_ids = [
        user_principal_id,
        org_id,
        await get_authenticated_principal_id(session),
        *group_principal_ids,
    ]
    cluster_id_stmt = select(ClusterAccess.cluster_id).where(
        ClusterAccess.principal_id.in_(principal_ids)
    )
    cluster_ids = set((await session.exec(cluster_id_stmt)).all())

    if not cluster_ids:
        return []

    cluster_stmt = select(Cluster).where(
        Cluster.id.in_(cluster_ids), Cluster.deleted_at.is_(None)
    )
    return list((await session.exec(cluster_stmt)).all())
