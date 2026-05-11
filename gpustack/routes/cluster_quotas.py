"""Per-cluster quota rows for the cluster-detail Quotas tab.

A cluster's quota set covers every ORG that can deploy on it: the
cluster's own owner Org (implicit) plus any Org granted explicit
access via ``cluster_access`` (ORG grants expose themselves; GROUP
grants expose the group's owning Org). USER grants don't surface
here — user-level deploys are attributed to whichever Org the
caller is acting under at deploy time.

Surface design:
- ``GET /clusters/{id}/quotas`` — list rows for everyone who can
  see the cluster (platform admin OR a member of one of the
  accessible Orgs). Each row carries the org name so the client
  doesn't need a separate orgs lookup.
- ``PUT /clusters/{id}/quotas/{owner_principal_id}`` — upsert one
  org's quotas on this cluster. Platform admin OR the owner-role
  member of the cluster's owner Org (``assert_cluster_writable``);
  non-owner Orgs that merely hold an access grant cannot mutate
  quotas.
"""

from typing import List, Optional, Set

from fastapi import APIRouter
from pydantic import BaseModel
from sqlmodel import col, select

from gpustack.api.exceptions import (
    InvalidException,
)
from gpustack.api.tenant import assert_cluster_visible, assert_cluster_writable
from gpustack.schemas.cluster_access import ClusterAccess
from gpustack.schemas.clusters import Cluster
from gpustack.schemas.principals import Principal, PrincipalType
from gpustack.schemas.tenant_quotas import TenantQuota, TenantQuotaUpdate
from gpustack.server.deps import SessionDep, TenantContextDep

router = APIRouter()


class ClusterQuotaPublic(BaseModel):
    cluster_id: int
    owner_principal_id: int
    owner_name: str
    gpu: Optional[int] = None
    cpu_milli: Optional[int] = None
    memory_bytes: Optional[int] = None
    gpu_instance: Optional[int] = None


async def _load_cluster(session, ctx, cluster_id: int) -> Cluster:
    cluster = await Cluster.one_by_id(session, cluster_id)
    assert_cluster_visible(ctx, cluster, not_found_message="Cluster not found")
    return cluster


async def _accessible_org_ids(session, cluster: Cluster) -> Set[int]:
    """Org-principal ids that can deploy on this cluster.

    Owner Org always counts. cluster_access ORG grants add themselves;
    GROUP grants add the group's owning Org. USER grants are skipped
    (user-level access is per-deploy, not per-Org).
    """
    org_ids: Set[int] = set()
    if cluster.owner_principal_id is not None:
        org_ids.add(cluster.owner_principal_id)
    access_rows = (
        await session.exec(
            select(ClusterAccess).where(ClusterAccess.cluster_id == cluster.id)
        )
    ).all()
    principal_ids = {r.principal_id for r in access_rows}
    if principal_ids:
        principals = (
            await session.exec(
                select(Principal).where(col(Principal.id).in_(principal_ids))
            )
        ).all()
        for p in principals:
            if p.deleted_at is not None:
                continue
            if p.kind == PrincipalType.ORG:
                org_ids.add(p.id)
            elif p.kind == PrincipalType.GROUP and p.parent_principal_id is not None:
                org_ids.add(p.parent_principal_id)
    return org_ids


def _to_public(
    cluster_id: int,
    owner_principal_id: int,
    owner_name: str,
    quota: Optional[TenantQuota],
) -> ClusterQuotaPublic:
    return ClusterQuotaPublic(
        cluster_id=cluster_id,
        owner_principal_id=owner_principal_id,
        owner_name=owner_name,
        gpu=quota.gpu if quota else None,
        cpu_milli=quota.cpu_milli if quota else None,
        memory_bytes=quota.memory_bytes if quota else None,
        gpu_instance=quota.gpu_instance if quota else None,
    )


@router.get("/clusters/{cluster_id}/quotas", response_model=List[ClusterQuotaPublic])
async def list_cluster_quotas(
    session: SessionDep, ctx: TenantContextDep, cluster_id: int
):
    cluster = await _load_cluster(session, ctx, cluster_id)
    org_ids = await _accessible_org_ids(session, cluster)
    if not org_ids:
        return []

    quotas = (
        await session.exec(
            select(TenantQuota).where(
                TenantQuota.cluster_id == cluster_id,
                col(TenantQuota.owner_principal_id).in_(org_ids),
            )
        )
    ).all()
    quota_by_owner = {q.owner_principal_id: q for q in quotas}

    orgs = (
        await session.exec(
            select(Principal).where(
                col(Principal.id).in_(org_ids),
                Principal.kind == PrincipalType.ORG,
                Principal.deleted_at.is_(None),
            )
        )
    ).all()
    name_by_id = {p.id: p.name for p in orgs}

    out: List[ClusterQuotaPublic] = []
    for org_id in org_ids:
        if org_id not in name_by_id:
            continue
        out.append(
            _to_public(
                cluster_id, org_id, name_by_id[org_id], quota_by_owner.get(org_id)
            )
        )
    # Stable order: iterating ``org_ids`` (a set) is non-deterministic,
    # which would shuffle the Quotas table on every refresh. Sort by
    # owner Org name so the UI list stays stable.
    out.sort(key=lambda r: r.owner_name)
    return out


@router.put(
    "/clusters/{cluster_id}/quotas/{owner_principal_id}",
    response_model=ClusterQuotaPublic,
)
async def upsert_cluster_quota(
    session: SessionDep,
    ctx: TenantContextDep,
    cluster_id: int,
    owner_principal_id: int,
    body: TenantQuotaUpdate,
):
    cluster = await _load_cluster(session, ctx, cluster_id)
    # Same gate as cluster_access POST / DELETE: platform admin OR the
    # owner-role member of the cluster's owner Org. Setting quotas is
    # part of running the cluster — if you can share it, you can
    # decide how much of it each tenant gets. Non-owner Orgs that
    # merely hold an access grant cannot mutate quotas.
    assert_cluster_writable(ctx, cluster)

    org_ids = await _accessible_org_ids(session, cluster)
    if owner_principal_id not in org_ids:
        raise InvalidException(
            message="Org is not accessible to this cluster; grant access first"
        )

    owner = await Principal.one_by_id(session, owner_principal_id)
    if owner is None or owner.deleted_at is not None or owner.kind != PrincipalType.ORG:
        raise InvalidException(message=f"Org {owner_principal_id} not found")

    existing = await TenantQuota.first_by_fields(
        session,
        {"cluster_id": cluster_id, "owner_principal_id": owner_principal_id},
    )
    try:
        if existing is None:
            row = await TenantQuota.create(
                session,
                TenantQuota(
                    cluster_id=cluster_id,
                    owner_principal_id=owner_principal_id,
                    gpu=body.gpu,
                    cpu_milli=body.cpu_milli,
                    memory_bytes=body.memory_bytes,
                    gpu_instance=body.gpu_instance,
                ),
            )
        else:
            await existing.update(session, body)
            row = existing
    except Exception as e:
        await session.rollback()
        raise InvalidException(message=f"Failed to update cluster quota: {e}")

    return _to_public(cluster_id, owner_principal_id, owner.name, row)
