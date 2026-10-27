"""Canonical startup peers for managed PD routers."""

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from gpustack.schemas.pd_modes import PDMode, PDRouterMembershipUpdateEnum

if TYPE_CHECKING:
    from gpustack.schemas.models import Model


def active_router_peers(
    instances: Sequence[object],
    group_id: Optional[str],
    worker_ip_by_id: Mapping[int, str],
) -> Iterator[Tuple[object, str, int, Dict[str, int]]]:
    """Yield reachable, non-draining GPU members of one generation."""
    for instance in instances:
        if getattr(instance, "role", None) not in ("prefill", "decode"):
            continue
        if getattr(instance, "group_id", None) != group_id:
            continue
        if getattr(instance, "state", None) != "running":
            continue
        if getattr(instance, "draining_since", None) is not None:
            continue
        ip = worker_ip_by_id.get(getattr(instance, "worker_id", None))
        port = getattr(instance, "port", None)
        if not ip or not port:
            continue
        ports = {
            name: band.base
            for name, band in (getattr(instance, "named_ports", None) or {}).items()
            if getattr(band, "base", None) is not None
        }
        yield instance, ip, int(port), ports


def router_peer_snapshot(
    instances: Sequence[object],
    group_id: Optional[str],
    worker_ip_by_id: Mapping[int, str],
) -> List[Dict[str, Any]]:
    """Record peer identities, addresses and bootstrap ports used at startup.

    Identity and restart timestamps distinguish a fresh engine at the same
    URL, whose bootstrap metadata must be queried again at router startup.
    """
    result = []
    for instance, ip, port, ports in active_router_peers(
        instances, group_id, worker_ip_by_id
    ):
        restarted = getattr(instance, "last_restart_time", None)
        result.append(
            {
                "id": getattr(instance, "id", None),
                "role": instance.role,
                "ip": ip,
                "port": port,
                "ports": ports,
                "restart_count": getattr(instance, "restart_count", 0) or 0,
                "last_restart_time": restarted.isoformat() if restarted else None,
            }
        )
    return sorted(result, key=lambda peer: (peer["role"], peer["ip"], peer["port"]))


def membership_update_policy(
    model: "Model", mode: Optional[PDMode]
) -> PDRouterMembershipUpdateEnum:
    """User-authored router commands own their membership configuration."""
    router = next((r for r in (model.roles or []) if r.name == "router"), None)
    if mode is None or router is None or router.run_command:
        return PDRouterMembershipUpdateEnum.MANUAL
    return mode.router.membership_update
