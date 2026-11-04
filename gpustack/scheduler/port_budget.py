"""How many members of a role one host has ports for.

Ports are allocated on the **worker**, when the container starts, out of
`service_port_range` — 64 ports by default. The scheduler never saw that, and
for a single-role deployment it did not need to: one instance takes one port,
so a host runs out of accelerators long before it runs out of ports.

A PD member is not one port. A connector declares bands whose width is its own
rule — Mooncake binds one port per *worker rank*, so a TP8 member holds nine
(1 + 8) — and a group deliberately concentrates its members onto as few hosts
as it can. Nine ports times seven members is the whole default pool, on a host
that may still have cards free.

**What it costs to not check this is not a refusal, it is a wedge.** The
scheduler places the member, the worker then cannot find a band for it, and
the instance sits in `starting` with a `PortRangeExhaustedError` — after the
placement decision, so nothing reconsiders it. Counting here turns that into
one fewer slot on that host, which the solver handles like any other shortage:
it moves to the next host, or refuses with an arithmetic the operator can act
on ("widen the port range").

Used by group capacity and ordinary candidate admission. The shared ordered plan
includes Engine APIs, internal communication windows, SMG listeners and connector bands.
"""

from __future__ import annotations

import asyncio
import logging

import aiohttp
from typing import Dict, Iterable, List, Optional, Set, Tuple
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.utils.network import parse_port_range

from gpustack.schemas.models import Model, member_worker_ids
from gpustack.schemas.workers import Worker
from gpustack.schemas.cache_services import CacheServiceInstance
from gpustack.policies.base import ModelInstanceScheduleCandidate
from gpustack.utils.model_ports import PDPortScopeUnsupportedError, model_port_requests
from gpustack.schemas.port_reservations import WorkerPortAvailability
from gpustack.server.worker_request import request_to_worker

logger = logging.getLogger(__name__)


def member_port_demand(
    model, role: Optional[str], cards: int, gpu_per_node: Optional[List[int]] = None
) -> int:
    """Ports one member of `role` occupies on the host it lands on.

    `cards` is the single-host allocation. Spanning placements pass
    `gpu_per_node` so the shared band covers every global rank offset on
    each host, just as the worker allocator does.

    Managed vLLM counts the complete API, RPC, store and compatibility-window
    allocation, including ports reserved on a headless host without a listener.

    **A band that cannot be resolved counts as zero, and that is not an
    underestimate.** The worker skips such a band too (leaving
    `{{ports.<name>}}` in the launch for the engine to reject by name), so no
    port is reserved for it there either. Reserving one here would make this
    the stricter of two checks that are supposed to be the same check.
    """
    return sum(
        count for count, _ in member_port_requests(model, role, gpu_per_node or [cards])
    )


def member_port_requests(
    model: Model, role: Optional[str], allocation: List[int]
) -> List[Tuple[int, Optional[int]]]:
    """Ordered bands, including fixed listeners, as the Worker allocates them."""
    return [
        (count, fixed)
        for _, count, fixed in model_port_requests(model, role, allocation)
    ]


def fits_port_layout(
    snapshots: List[WorkerPortAvailability], requests: List[Tuple[int, Optional[int]]]
) -> bool:
    """Check first-fit bands in the intersection of participating Workers' ports.

    Raises:
        ValueError: Workers have no common service-port range.
    """
    ranges = [parse_port_range(snapshot.port_range) for snapshot in snapshots]
    start, end = max(low for low, _ in ranges), min(high for _, high in ranges)
    if start > end:
        configured = ', '.join(
            f'{snapshot.worker_id}: {snapshot.port_range}' for snapshot in snapshots
        )
        raise ValueError(
            f'Workers have no common service port range ({configured}); '
            'align service_port_range across these Workers.'
        )
    busy = {port for snapshot in snapshots for port in snapshot.unavailable_ports}
    free = set(range(start, end + 1)) - busy
    for count, fixed in requests:
        if fixed is not None:
            if not set(range(fixed, fixed + count)) <= free:
                return False
            base = fixed
        else:
            run = 0
            previous = None
            base = None
            for port in sorted(free):
                run = run + 1 if previous == port - 1 else 1
                previous = port
                if run == count:
                    base = port - count + 1
                    break
            if base is None:
                return False
        free.difference_update(range(base, base + count))
    return True


async def worker_port_snapshots(
    workers: List[Worker], exclude_instance_id: Optional[int] = None
) -> Dict[int, WorkerPortAvailability]:
    """Fetch each candidate Worker's advisory port state once per scheduling pass."""
    semaphore = asyncio.Semaphore(8)
    async with (
        aiohttp.ClientSession(trust_env=True) as proxy_client,
        aiohttp.ClientSession() as no_proxy_client,
    ):

        async def fetch(worker):
            async with semaphore:
                try:
                    _, body = await request_to_worker(
                        worker=worker,
                        method="GET",
                        path="port-reservations",
                        params=(
                            {"exclude_instance_id": exclude_instance_id}
                            if exclude_instance_id is not None
                            else None
                        ),
                        proxy_client=proxy_client,
                        no_proxy_client=no_proxy_client,
                        timeout=aiohttp.ClientTimeout(total=5, sock_connect=2),
                    )
                    snapshot = WorkerPortAvailability.model_validate_json(body)
                    if snapshot.worker_id == worker.id:
                        return worker.id, snapshot
                except (aiohttp.ClientError, TimeoutError, ValueError):
                    logger.debug(
                        "Cannot establish port availability on Worker %s", worker.id
                    )
                return worker.id, None

        results = await asyncio.gather(*(fetch(worker) for worker in workers))
    return {
        worker_id: snapshot for worker_id, snapshot in results if snapshot is not None
    }


async def filter_candidates(
    session: Optional[AsyncSession],
    model: Model,
    role: Optional[str],
    candidates: List[ModelInstanceScheduleCandidate],
    workers: List[Worker],
    model_instances: Iterable[object],
    exclude_instance_id: Optional[int] = None,
) -> Tuple[List[ModelInstanceScheduleCandidate], List[str]]:
    """Remove candidates that cannot fit the same port layout on every member."""
    if not candidates:
        return candidates, []
    plans = []
    messages = []
    for candidate in candidates:
        members = [candidate.worker.id] + [
            w.worker_id for w in candidate.subordinate_workers or []
        ]
        counts = [len(candidate.gpu_indexes or [])] + [
            len(w.gpu_indexes or []) for w in candidate.subordinate_workers or []
        ]
        try:
            requests = member_port_requests(model, role, counts)
        except (ValueError, PDPortScopeUnsupportedError) as error:
            messages.append(str(error))
            continue
        if role is None and sum(count for count, _ in requests) == 1:
            plans.append((candidate, members, None))
        else:
            plans.append((candidate, members, requests))
    needed = {
        worker_id for _, members, requests in plans if requests for worker_id in members
    }
    if not needed:
        return [candidate for candidate, _, _ in plans], messages
    snapshots = await worker_port_snapshots(
        [worker for worker in workers if worker.id in needed], exclude_instance_id
    )
    # Persisted declarations come from this pass's database view; Worker RPCs
    # return only local reservations and sockets, avoiding a DB callback cycle.
    caches = (
        await CacheServiceInstance.all_by_fields(
            session,
            {"cluster_id": model.cluster_id} if model.cluster_id is not None else {},
            extra_conditions=[CacheServiceInstance.worker_id.in_(sorted(needed))],
        )
        if session is not None
        else []
    )
    for worker_id, snapshot in snapshots.items():
        snapshot.unavailable_ports = sorted(
            set(snapshot.unavailable_ports)
            | reserved_ports_on(worker_id, model_instances, caches)
        )
    accepted = []
    for candidate, members, requests in plans:
        if requests is None:
            accepted.append(candidate)
            continue
        if not all(worker_id in snapshots for worker_id in members):
            messages.append(
                f"Cannot establish port availability for Workers {members}."
            )
            continue
        try:
            fits = fits_port_layout(
                [snapshots[worker_id] for worker_id in members], requests
            )
        except ValueError as error:
            messages.append(str(error))
            continue
        if fits:
            accepted.append(candidate)
        else:
            demand = sum(count for count, _ in requests)
            messages.append(
                f"Workers {members} cannot fit {demand} service ports with the required common contiguous bands."
            )
    return accepted, messages


def ports_taken_on(
    worker_id: int,
    model_instances: Iterable[object],
    cache_instances: Iterable[object] = (),
) -> int:
    """Count distinct persisted model and cache ports on one Worker."""
    return len(reserved_ports_on(worker_id, model_instances, cache_instances))


def reserved_ports_on(
    worker_id: int,
    model_instances: Iterable[object],
    cache_instances: Iterable[object] = (),
) -> Set[int]:
    """Ports already spoken for on one host.

    Counted as a set of port numbers rather than a sum of lengths: `ports` and
    `named_ports` overlap by design — a band's base appears in both — and
    adding the two lengths would charge a TP8 member for its own base twice.

    Cache service instances are counted alongside model instances because
    they come out of the *same* `service_port_range`. A host running a cache
    server has fewer ports for members by its number of allocated ports, and
    a group is placed onto cache-bearing hosts on purpose (F6).

    **A member counts on every machine it spans, not on the one its row is
    filed under.** One `ModelInstance` holds one set of ports:
    The shared port plan sizes the bands once on the primary, and each host
    the member landed on then fences that same set through
    `_register_assigned_ports` -- a subordinate's own pass finds `mi.port`
    already set and re-registers rather than allocating. Filtering on
    `worker_id` made a running spanning member invisible on its subordinates,
    so this read handed back more free ports than the host has and the solve
    promised room the allocator would not find. `_committed_port_demand`
    charges the members of the solve in hand the same way, for the same
    reason.

    A cache instance has no spanning form -- one per worker, by its own
    constraint -- so that loop still asks about the one machine it is on.
    """
    taken: Set[int] = set()

    for instance in model_instances:
        if worker_id not in member_worker_ids(instance):
            continue
        for port in getattr(instance, "ports", None) or []:
            taken.add(port)
        for band in (getattr(instance, "named_ports", None) or {}).values():
            base = getattr(band, "base", None)
            if base is None:
                continue
            taken.update(range(base, base + max(getattr(band, "count", 1) or 1, 1)))

    for instance in cache_instances:
        if getattr(instance, "worker_id", None) != worker_id:
            continue
        taken.update((getattr(instance, "ports", None) or {}).values())
        port = getattr(instance, "port", None)
        if port is not None:
            taken.add(port)

    return taken


def port_capacity(
    port_range: Optional[str],
    demand_per_member: int,
    taken: int,
) -> Optional[int]:
    """How many more members this host has ports for, or None for no limit.

    None rather than a large number when the range cannot be read: an
    unparseable configuration must not silently become a capacity of zero,
    which is the one answer that stops an operator looking for a mistake.
    """
    if demand_per_member <= 0 or not port_range:
        return None
    try:
        start, end = parse_port_range(port_range)
    except Exception:
        logger.warning(
            "Could not read the service port range %r; placing without a port "
            "budget.",
            port_range,
        )
        return None

    free = (end - start + 1) - taken
    if free <= 0:
        return 0
    return free // demand_per_member


def describe(worker_id: int, allowed: int, demand: int, taken: int) -> str:
    """One line for the log, with the arithmetic rather than the verdict."""
    return (
        f"worker {worker_id}: ports allow {allowed} more member(s) "
        f"({demand} port(s) each, {taken} already taken)"
    )
