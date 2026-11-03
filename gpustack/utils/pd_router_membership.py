"""Canonical startup peers for managed PD routers."""

import logging
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

logger = logging.getLogger(__name__)


def instance_router_peers(
    instance: object, worker_ip_by_id: Optional[Mapping[int, str]] = None
) -> Iterator[Tuple[str, int, Dict[str, int]]]:
    """Resolve a serving PD member without falling back from managed tasks."""
    from gpustack.utils.model_instance_workers import model_instance_endpoints

    if getattr(instance, "role", None) not in ("prefill", "decode"):
        return
    if (
        getattr(instance, "state", None) != "running"
        or getattr(instance, "draining_since", None) is not None
        or getattr(instance, "deleted_at", None) is not None
    ):
        return
    servers = getattr(instance, "distributed_servers", None)
    managed = servers is not None and getattr(servers, "workloads", None) is not None
    addresses = worker_ip_by_id
    if addresses is None:
        addresses = {
            getattr(instance, "worker_id", None): getattr(instance, "worker_ip", None)
        }
        for worker in getattr(servers, "subordinate_workers", None) or []:
            addresses[worker.worker_id] = worker.worker_ip
    ports = {
        name: band.base
        for name, band in (getattr(instance, "named_ports", None) or {}).items()
        if getattr(band, "base", None) is not None
    }
    if not managed:
        ip = addresses.get(getattr(instance, "worker_id", None))
        port = getattr(instance, "port", None)
        if ip and port:
            yield ip, int(port), ports
        return
    try:
        endpoints = model_instance_endpoints(instance)
    except (AttributeError, TypeError, ValueError):
        return
    missing = sorted(
        {
            endpoint.worker_id
            for endpoint in endpoints
            if not addresses.get(endpoint.worker_id)
        }
    )
    if missing:
        logger.debug(
            "Skipping managed PD instance %s: no current address for Workers %s",
            getattr(instance, "id", None),
            missing,
        )
        return
    bootstrap = (getattr(instance, "named_ports", None) or {}).get("bootstrap")
    ranked_bootstrap = bootstrap is not None and getattr(bootstrap, "count", 1) > 1
    if ranked_bootstrap and any(
        endpoint.dp_rank is None or not 0 <= endpoint.dp_rank < bootstrap.count
        for endpoint in endpoints
    ):
        return
    for endpoint in endpoints:
        peer_ports = dict(ports)
        if ranked_bootstrap:
            peer_ports["bootstrap"] = bootstrap.base + endpoint.dp_rank
        yield addresses[endpoint.worker_id], endpoint.port, peer_ports


def active_router_peers(
    instances: Sequence[object],
    group_id: Optional[str],
    worker_ip_by_id: Optional[Mapping[int, str]] = None,
) -> Iterator[Tuple[object, str, int, Dict[str, int]]]:
    """Yield reachable, non-draining GPU members of one generation."""
    for instance in instances:
        if getattr(instance, "group_id", None) != group_id:
            continue
        for ip, port, ports in instance_router_peers(instance, worker_ip_by_id):
            yield instance, ip, port, ports


def router_peer_snapshot(
    instances: Sequence[object],
    group_id: Optional[str],
    worker_ip_by_id: Optional[Mapping[int, str]] = None,
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
