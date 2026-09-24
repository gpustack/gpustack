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

Only reached from the group path. A model without `roles` has no PD mode, so
`member_port_demand` returns 1 for it and the capacity it computes is the
count it already had.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Set

from gpustack.schemas.pd_modes import PDPortScopeEnum
from gpustack.utils.network import parse_port_range

# The same resolution `_assign_named_ports` runs, imported rather than
# restated. The two answers have to agree: this one decides whether a member is
# placed on a host, that one decides whether it can start there, and a private
# copy here would drift into a scheduler that promises room the allocator does
# not find.
from gpustack.schemas.models import member_worker_ids
from gpustack.worker.pd_injection import band_specs_for, band_width

logger = logging.getLogger(__name__)

# The HTTP serving port every instance gets before any band is considered.
_BASE_PORTS = 1


def member_port_demand(model, role: Optional[str], cards: int) -> int:
    """Ports one member of `role` occupies on the host it lands on.

    `cards` is how many accelerators this member would be given there, which
    is what `{{accelerator_count}}` resolves to — the same number the worker
    will read off `gpu_indexes` once the member is placed.

    **A band that cannot be resolved counts as zero, and that is not an
    underestimate.** The worker skips such a band too (leaving
    `{{ports.<name>}}` in the launch for the engine to reject by name), so no
    port is reserved for it there either. Reserving one here would make this
    the stricter of two checks that are supposed to be the same check.
    """
    demand = _BASE_PORTS
    for spec in band_specs_for(model, role, context=f"Role {role!r}"):
        if spec.scope == PDPortScopeEnum.ROLE:
            # Refused outright on the worker. Budgeting for it would be
            # reserving ports for a member that will not start.
            continue
        count = band_width(
            spec,
            cards=cards,
            backend_parameters=getattr(model, "backend_parameters", None),
        )
        if count:
            demand += count
    return demand


def ports_taken_on(
    worker_id: int,
    model_instances: Iterable[object],
    cache_instances: Iterable[object] = (),
) -> int:
    """Ports already spoken for on one host.

    Counted as a set of port numbers rather than a sum of lengths: `ports` and
    `named_ports` overlap by design — a band's base appears in both — and
    adding the two lengths would charge a TP8 member for its own base twice.

    Cache service instances are counted alongside model instances because
    they come out of the *same* `service_port_range`. A host running a cache
    server has two fewer ports for members, and a group is exactly the kind of
    workload that is placed onto cache-bearing hosts on purpose (F6).

    **A member counts on every machine it spans, not on the one its row is
    filed under.** One `ModelInstance` holds one set of ports:
    `_assign_named_ports` sizes the bands once on the primary, and each host
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
        for port in (
            getattr(instance, "port", None),
            getattr(instance, "metrics_port", None),
        ):
            if port is not None:
                taken.add(port)

    return len(taken)


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
