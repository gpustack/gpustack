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
    refactor it's the user's own USER-principal (pre-refactor flag
    ``is_personal=True`` is now ``kind == USER`` rendered by
    ``OrganizationPublic.from_principal``). Synthesizing it here keeps
    the OrgSwitcher render path unchanged on the UI side.
    """
    items: List[MyOrganization] = []

    user_principal = await Principal.one_by_id(session, user.principal_id)
    if user_principal is not None and user_principal.deleted_at is None:
        items.append(
            MyOrganization(
                organization=OrganizationPublic.from_principal(user_principal),
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
            PrincipalMembership.member_principal_id == user.principal_id,
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
            group_pm.member_principal_id == user.principal_id,
            group_pm.deleted_at.is_(None),
            org_pm.deleted_at.is_(None),
            Principal.deleted_at.is_(None),
            Principal.kind == PrincipalType.ORG,
        )
    )

    # One round trip: direct + transitive Org memberships unioned.
    # OWNER beats MEMBER when both paths report on the same Org.
    best_by_org: Dict[int, tuple[OrgRole, Principal]] = {}
    for role, org in (await session.exec(direct_stmt.union_all(via_group_stmt))).all():
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
    user_principal_id = ctx.user.principal_id
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

    principal_ids = [user_principal_id, org_id, *group_principal_ids]
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
