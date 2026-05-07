"""Self-service tenant endpoints — what orgs am I in, what clusters can I use."""

from typing import List

from fastapi import APIRouter
from pydantic import BaseModel
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
    then any ORG-principals they're a member of.

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
                role=OrgRole.ADMIN,
            )
        )

    stmt = (
        select(PrincipalMembership, Principal)
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
    rows = (await session.exec(stmt)).all()
    items.extend(
        MyOrganization(
            organization=OrganizationPublic.from_principal(org),
            role=membership.role or OrgRole.USER,
        )
        for membership, org in rows
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

    # Group-principals that the user is a member of, scoped to this Org.
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
            Principal.parent_principal_id == org_id,
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
