"""Cluster access authorization.

Lets the cluster's owner Org delegate accessibility to other
principals (USER / ORG / GROUP); platform admin can manage any
cluster. The grant row stores a single ``principal_id`` FK; kind
comes from the joined principals row at read time.

Permission gates:
- GET — anyone who can see the cluster (cluster-detail audience).
- POST / DELETE — platform admin, or the owner-role member of the
  cluster's owner Org (``assert_cluster_writable``). Non-owner
  orgs that merely hold an access grant can't re-grant.
"""

from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel
from sqlmodel import select

from gpustack.api.exceptions import (
    AlreadyExistsException,
    InvalidException,
    NotFoundException,
)
from gpustack.api.tenant import assert_cluster_visible, assert_cluster_writable
from gpustack.schemas.cluster_access import ClusterAccess, ClusterAccessPublic
from gpustack.schemas.clusters import Cluster
from gpustack.schemas.principals import Principal, PrincipalType
from gpustack.server.deps import SessionDep, TenantContextDep

router = APIRouter()


class ClusterAccessGrant(BaseModel):
    # Discriminator kept on the input shape for client-side validation
    # — the server cross-checks against the joined ``principals`` row,
    # so a mismatched kind is rejected at the validator below.
    principal_type: PrincipalType
    principal_id: int


async def _load_cluster(session, ctx, cluster_id: int) -> Cluster:
    cluster = await Cluster.one_by_id(session, cluster_id)
    assert_cluster_visible(ctx, cluster, not_found_message="Cluster not found")
    return cluster


async def _validate_principal(
    session, principal_type: PrincipalType, principal_id: int
) -> Principal:
    """Ensure the principal exists, isn't soft-deleted, and matches
    the declared kind."""
    target = await Principal.one_by_id(session, principal_id)
    if not target or target.deleted_at is not None:
        raise InvalidException(message=f"Principal {principal_id} not found")
    if target.kind != principal_type:
        raise InvalidException(
            message=(
                f"Principal {principal_id} is a {target.kind.value}, "
                f"not a {principal_type.value}"
            )
        )
    # System principals (kind=SYSTEM) bypass cluster_access entirely
    # via :func:`bypass_tenant_filter`; the API surface here is
    # constrained to USER / GROUP / ORG kinds so SYSTEM never reaches
    # this validator.
    return target


async def _resolve_principal_views(
    session, rows: List[ClusterAccess]
) -> List[ClusterAccessPublic]:
    """Bulk-resolve display labels and kind for each row in a single
    principals lookup.
    """
    principal_ids = {r.principal_id for r in rows}
    principal_by_id: dict[int, Principal] = {}
    if principal_ids:
        result = await session.exec(
            select(Principal).where(Principal.id.in_(principal_ids))
        )
        principal_by_id = {p.id: p for p in result.all()}

    out: List[ClusterAccessPublic] = []
    for r in rows:
        p: Optional[Principal] = principal_by_id.get(r.principal_id)
        kind = p.kind if p else PrincipalType.USER
        # ORG principals' "parent" for display purposes is themselves;
        # GROUPs are now peer-level (may join zero or more Orgs via
        # membership rows) so they have no single parent to surface
        # here — UI quota slots for a Group should be resolved
        # separately from the Group's Org-memberships if needed.
        parent = p.id if p and p.kind == PrincipalType.ORG else None
        out.append(
            ClusterAccessPublic(
                cluster_id=r.cluster_id,
                principal_id=r.principal_id,
                principal_type=kind,
                principal_display_name=p.display_name if p else None,
                principal_parent_id=parent,
                granted_by=r.granted_by,
                created_at=r.created_at,
            )
        )
    return out


@router.get("/clusters/{cluster_id}/access", response_model=List[ClusterAccessPublic])
async def list_cluster_access(
    session: SessionDep, ctx: TenantContextDep, cluster_id: int
):
    await _load_cluster(session, ctx, cluster_id)
    stmt = select(ClusterAccess).where(ClusterAccess.cluster_id == cluster_id)
    rows = list((await session.exec(stmt)).all())
    return await _resolve_principal_views(session, rows)


@router.post("/clusters/{cluster_id}/access", response_model=ClusterAccessPublic)
async def grant_cluster_access(
    session: SessionDep,
    ctx: TenantContextDep,
    cluster_id: int,
    body: ClusterAccessGrant,
):
    cluster = await _load_cluster(session, ctx, cluster_id)
    # Cluster owner Org's owner (and platform admin) can delegate
    # access to other tenants; non-owner orgs that merely have a
    # grant can't re-grant.
    assert_cluster_writable(ctx, cluster)
    await _validate_principal(session, body.principal_type, body.principal_id)

    existing_stmt = select(ClusterAccess).where(
        ClusterAccess.cluster_id == cluster_id,
        ClusterAccess.principal_id == body.principal_id,
    )
    if (await session.exec(existing_stmt)).first() is not None:
        raise AlreadyExistsException(message="Access already granted")

    try:
        access = await ClusterAccess.create(
            session,
            ClusterAccess(
                cluster_id=cluster_id,
                principal_id=body.principal_id,
                granted_by=ctx.user.id,
            ),
        )
    except Exception as e:
        await session.rollback()
        raise InvalidException(message=f"Failed to grant cluster access: {e}")

    enriched = await _resolve_principal_views(session, [access])
    return enriched[0]


@router.delete("/clusters/{cluster_id}/access/{principal_id}")
async def revoke_cluster_access(
    session: SessionDep,
    ctx: TenantContextDep,
    cluster_id: int,
    principal_id: int,
):
    cluster = await _load_cluster(session, ctx, cluster_id)
    assert_cluster_writable(ctx, cluster)
    stmt = select(ClusterAccess).where(
        ClusterAccess.cluster_id == cluster_id,
        ClusterAccess.principal_id == principal_id,
    )
    access = (await session.exec(stmt)).first()
    if not access:
        raise NotFoundException(message="Access grant not found")

    try:
        await access.delete(session)
    except Exception as e:
        await session.rollback()
        raise InvalidException(message=f"Failed to revoke cluster access: {e}")
