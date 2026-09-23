"""How good one whole-group placement is, compared with another at the same layer.

Beside the per-candidate scorers rather than with the solver, and beside
``pairing_affinity_scorer`` in particular: that one scores a single member by
this module's own objective differentiated, so the two are one rule at two
granularities. Split across packages, the next change to how prefill and
decode pull on each other lands in one of them.

The per-instance path scores a *candidate* -- one member, one worker, one set
of cards -- and picks the best. A group cannot be ranked that way: its unit is
an assignment of every member at once, and the two things worth ranking are
properties of the assignment rather than of any member in it.

**Pairing is not the sum of its members' scores.** With ``x`` prefills, ``y``
decodes, and ``p_j`` / ``d_j`` of each on worker ``j``, a router that picks the
two ends of a request independently gives

    P(a request's prefill and decode are on one host) = (1/xy) * sum_j p_j*d_j

which is the objective itself, already derived in ``PairingAffinityScorer``.
That scorer's per-candidate value is this objective's *derivative* -- "adding a
prefill on worker j raises the numerator by d_j".

Adding up the members' values would in fact rank identically: every pair is
counted once from each end, so the sum is exactly twice the numerator. What it
would not be is a **ratio**. It grows with the group's size and with whatever
scale the per-candidate scorer happens to use, so setting it beside the file
term would turn the weight between them into an exchange rate -- the very
thing the layer is held fixed to avoid. The objective is already normalised by
construction, so the weights stay ordinary constants.

**Both terms are ratios**, in ``[0, 1]``. That is what lets the weights be
ordinary constants: they trade two commensurable quantities rather than setting
an exchange rate between a count and a distance.

The layer is deliberately not a term. See ``solve_group_placement``: the
tightest layer holding the group wins outright, so every placement this module
compares already sits at the same layer and the dimension has been removed
rather than priced.
"""

from __future__ import annotations

import logging
from typing import Callable, Iterable, Optional, Set

from gpustack.schemas.models import RoleNameEnum
from gpustack.scheduler.group_solver import GroupPlacement, GroupScore

logger = logging.getLogger(__name__)


def pair_locality(placement: GroupPlacement) -> float:
    """The share of prefill/decode pairs that land on one host.

    ``sum_j p_j*d_j / (x*y)`` while every member sits on one machine, which is
    every member placed today: 1.0 when every pair is local, 0.0 when none is.

    A member wide enough to span machines generalises it rather than breaking
    it. The prefill holds the KV across its own machines and the decode reads
    what shares a machine with it, so one pair contributes
    ``|prefill machines & decode machines| / |prefill machines|`` -- which is
    1 or 0 for single-machine members, reducing to the sum above.

    Zero rather than undefined when either role is absent: a group with no
    prefill or no decode has no pair to keep local, so the term has nothing to
    say about it and must not decide the ranking. Returning a fraction of
    nothing would be a division by zero; returning 1.0 would make a
    single-role group outrank every real one.

    Args:
        placement: The assignment to measure.

    Returns:
        A ratio in ``[0, 1]``.
    """
    prefills = placement.assignments.get(RoleNameEnum.PREFILL.value) or []
    decodes = placement.assignments.get(RoleNameEnum.DECODE.value) or []
    if not prefills or not decodes:
        return 0.0

    prefill_machines = [
        set(placement.machines_of(RoleNameEnum.PREFILL.value, i))
        for i in range(len(prefills))
    ]
    decode_machines = [
        set(placement.machines_of(RoleNameEnum.DECODE.value, i))
        for i in range(len(decodes))
    ]

    total = 0.0
    for held in prefill_machines:
        for reader in decode_machines:
            # The prefill holds the KV, spread over its own machines; the
            # decode reads whatever sits on a machine it is also on. So the
            # share is measured against the prefill's width -- a decode
            # sharing one of a two-machine prefill's hosts reads half of it
            # locally, not none of it and not all.
            total += len(held & reader) / len(held)

    return total / (len(prefills) * len(decodes))


def file_locality(placement: GroupPlacement, ready_worker_ids: Set[int]) -> float:
    """The share of members landing where the model's files already are.

    Unlike pairing, this one *is* a per-member property, so averaging is the
    honest composition: members do not interact through it, and a member on a
    worker holding the files skips a download whatever its siblings do.

    Args:
        placement: The assignment to measure.
        ready_worker_ids: Workers whose copy of the model is READY.

    Returns:
        A ratio in ``[0, 1]``; 0.0 when nothing is cached anywhere, which
        makes the term inert rather than arbitrary.
    """
    if not ready_worker_ids:
        return 0.0
    shares = []
    for role, primaries in placement.assignments.items():
        for index in range(len(primaries)):
            machines = placement.machines_of(role, index)
            # A member spanning machines needs the weights on every one of
            # them, so it is warm in proportion to how many are.
            shares.append(
                len([m for m in machines if m in ready_worker_ids]) / len(machines)
            )
    if not shares:
        return 0.0
    return sum(shares) / len(shares)


def group_scorer(
    ready_worker_ids: Optional[Iterable[int]] = None,
    pair_weight: float = 1.0,
    file_weight: float = 0.3,
) -> Optional[Callable[[GroupPlacement], GroupScore]]:
    """A ``GroupScoreFn`` over the terms above, or ``None`` if every term is off.

    ``None`` rather than a function returning a constant, because the solver
    reads ``score is None`` as "take the tightest fitting domain" and stops
    enumerating a layer after the first fit. A scorer that cannot distinguish
    anything would buy a full selector sweep per extra domain to rank them all
    the same.

    Args:
        ready_worker_ids: Workers whose copy of the model is READY. Absent or
            empty leaves the file term inert.
        pair_weight: Weight on prefill/decode co-location.
        file_weight: Weight on landing where the files already are.

    Returns:
        A callable taking a ``GroupPlacement`` and returning its
        ``GroupScore`` -- the total that ranks it and the terms behind it --
        or None.
    """
    cached: Set[int] = set(ready_worker_ids or ())
    pair_on = pair_weight > 0
    file_on = file_weight > 0 and bool(cached)
    if not pair_on and not file_on:
        return None

    def score(placement: GroupPlacement) -> GroupScore:
        # The terms are kept beside the total rather than recovered later:
        # whatever is shown about a choice has to be what the choice was made
        # with, and only the terms say *why* one domain beat another. A total
        # alone cannot distinguish "paired badly but warm" from the reverse.
        terms = {}
        total = 0.0
        if pair_on:
            terms["pair"] = pair_locality(placement)
            total += pair_weight * terms["pair"]
        if file_on:
            terms["file"] = file_locality(placement, cached)
            total += file_weight * terms["file"]
        return GroupScore(total=total, terms=terms)

    return score
