"""Catalog-driven startup membership for static PD routers."""

from typing import Any, Dict, List, Mapping, Optional, Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from gpustack.schemas.models import Model, ModelInstance, ModelInstanceCreate
from gpustack.schemas.pd_modes import PDRouterMembershipUpdateEnum
from gpustack.schemas.workers import Worker
from gpustack.server.pd_mode_catalog import get_pd_mode
from gpustack.utils.pd_router_membership import (
    membership_update_policy,
    router_peer_snapshot,
)

MEMBERSHIP_REFRESH_REASON = "router membership changed; rebuilding router"


def membership_policy(model: Model) -> PDRouterMembershipUpdateEnum:
    """Resolve the catalog policy with user-authored commands respected."""
    mode = (
        get_pd_mode(model.disaggregation.mode.value) if model.disaggregation else None
    )
    return membership_update_policy(model, mode)


async def expected_router_peers(
    session: AsyncSession,
    model: Model,
    instances: Sequence[ModelInstance],
    group_id: Optional[str],
    worker_ip_by_id: Optional[Mapping[int, str]] = None,
) -> List[Dict[str, Any]]:
    """Use the same worker addresses and filters as command rendering."""
    if worker_ip_by_id is None:
        worker_ip_by_id = await router_worker_ips(session, model.cluster_id)
    return router_peer_snapshot(instances, group_id, worker_ip_by_id)


async def router_worker_ips(
    session: AsyncSession, cluster_id: Optional[int]
) -> Dict[int, str]:
    """Load addresses once for all router generations in a reconciliation pass.

    Args:
        session: The current reconciliation session.
        cluster_id: The cluster whose workers provide router peer addresses.

    Returns:
        Worker IP addresses keyed by worker id.
    """
    workers = await Worker.all_by_field(session, "cluster_id", cluster_id)
    return {worker.id: worker.ip for worker in workers if worker.ip}


def router_refresh_ready(
    model: Model,
    snapshot: List[Dict[str, Any]],
    previous: Optional[List[Dict[str, Any]]],
) -> bool:
    """Wait for pure scale-out, but remove invalid bootstrap metadata promptly.

    Both sides must remain renderable. A removed/restarted member needs an
    immediate refresh even while its replacement is pending; pure additions
    wait until the requested counts are ready to coalesce router replacements.
    """
    counts = {
        name: sum(p["role"] == name for p in snapshot) for name in ("prefill", "decode")
    }
    if not all(counts.values()):
        return False
    if previous is None or any(peer not in snapshot for peer in previous):
        return True
    return all(
        counts[role.name] >= role.replicas
        for role in (model.roles or [])
        if role.name in counts
    )


async def seed_router_peer_snapshots(
    session: AsyncSession,
    model: Model,
    instances: Sequence[ModelInstance],
    group_id: str,
    pending: Sequence[ModelInstanceCreate],
) -> None:
    """Give static routers a baseline even when an older worker launches them.

    Current workers replace it with the actual rendered peers. A change during
    startup can then cause an extra replacement, but never an unbounded loop
    caused by a worker that does not yet write this field.
    """
    routers = [member for member in pending if member.role == "router"]
    if not routers or membership_policy(model) != PDRouterMembershipUpdateEnum.RESTART:
        return
    snapshot = await expected_router_peers(session, model, instances, group_id)
    for router in routers:
        router.router_peer_snapshot = snapshot
