"""How many members of one role a worker can still take.

The group scheduler cannot ask "does this group fit in that rack" without first
asking "how many members fit on each worker in it". There is no such query
today: the resource-fit selectors answer "where does *one more* instance go",
once. This gets the count by asking that question repeatedly and pretending the
answer was taken.

**The pretence is exact, not approximate.** Allocation is already derived from
the set of model-instance bindings rather than from anything a worker reports
(``compute_worker_allocated``), and every selector is constructed with that set
(``model_instances``). So appending a stand-in for the candidate just returned
and re-running the selector recomputes availability the same way the real
scheduler will once the row exists — there is no second accounting path to keep
in step with the first.

**Why the loop and not arithmetic.** Dividing free VRAM by a per-member claim
would ignore everything the selectors actually enforce: per-GPU fragmentation,
whole-card versus sliced modes, GPU-type matching, unified memory, the
multi-replica overcommit rule. The loop inherits all of it by construction, and
inherits future rules for free.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

from gpustack.policies.base import MemberResourceClaim

logger = logging.getLogger(__name__)

# A group whose members each fit many times over on one worker would otherwise
# spin here for as long as the fit holds. Bounded by what the caller could
# possibly place: asking for more slots than the group has members answers a
# question nobody asked.
_ABSURD_SLOTS = 1024


@dataclass
class _PlacedStandIn:
    """What ``compute_worker_allocated`` reads off a model instance.

    A real ``ModelInstance`` is a table row with required columns and an
    identity; constructing one here would mean either touching the session or
    filling in fields whose values would be lies. Only these four are read
    while summing allocation, so only these four are carried — and the class is
    named so that anything finding it in a list knows it is not a row.
    """

    worker_id: Optional[int]
    gpu_indexes: Optional[List[int]]
    gpu_type: Optional[str]
    computed_resource_claim: Any
    distributed_servers: Any = None


@dataclass
class OfferSlot:
    """One worker's capacity for one role, and the placements behind it."""

    worker_id: int
    slots: int = 0
    # The candidate the selector returned for each hypothetical member, in the
    # order they were taken. Kept because the group scheduler needs the actual
    # GPU indexes to write onto the members it decides to place here, and
    # re-deriving them later would be a second, divergent placement.
    placements: List[Any] = field(default_factory=list)
    # Why counting stopped. `slots` alone cannot tell "this worker is full"
    # from "the selector broke before it could say", and the two look identical
    # to whoever reads the refusal: a plain misconfiguration then presents as
    # "the cluster is full", which is the one message that makes an operator
    # stop looking for a mistake. Measured on a live host — a forgotten
    # `set_global_config` produced zeros on every worker and a group refusal
    # that named capacity.
    unavailable: Optional[str] = None
    # What the selector said on the round that found no room, in its own words
    # -- the claim in GiB, what the roomiest card had, the shortfall. The
    # single-instance refusal has carried these all along; a group refusal
    # could only count members, so "the group needs 4 placements and the
    # cluster has room for 2" never said how big one member is or what stood
    # in its way. Same source, so the two paths cannot describe one cluster
    # differently.
    notes: List[str] = field(default_factory=list)
    # What the selector priced ONE member of this role at, whether or not any
    # fit. A group that does not fit has no placement to read a claim off, so
    # without this the refusal can say how many members are missing but never
    # how big one is -- which is the first thing asked of a group whose roles
    # are sized differently from each other.
    claim: Optional[MemberResourceClaim] = None

    @property
    def counted_to_exhaustion(self) -> bool:
        """Whether ``slots`` is this worker's real capacity or only a floor."""
        return self.unavailable is None


async def count_offer_slots(
    make_selector,
    workers: Sequence[Any],
    model_instances: Sequence[Any],
    limit: int,
) -> OfferSlot:
    """How many more members of this role ``workers`` can take, up to ``limit``.

    ``make_selector`` is a callable taking the (growing) instance list and
    returning a fresh selector. A fresh one each round rather than a mutated
    one: the selectors compute their claims in ``__init__`` and cache them, so
    reusing an instance would answer the first round's question every time.

    ``limit`` is normally the group's member count. Capacity beyond what the
    caller wants to place is not a number anyone needs, and computing it costs
    a full selector pass per extra slot.

    **A set of machines, not one.** Handed a single worker this is what it
    always was. Handed several, the selectors' own cross-node branch becomes
    reachable -- it refuses a list shorter than two outright -- and a member
    that needs more cards than any one machine has can be counted at all. The
    combinations that come back are disjoint by construction, because each one
    is stood in before the next is asked for, so counting a domain this way
    yields the number of members it holds rather than a number per machine.
    """
    result = OfferSlot(worker_id=workers[0].id if workers else 0)
    if not workers:
        return result
    if limit <= 0:
        return result

    hypothetical = list(model_instances)
    bound = min(limit, _ABSURD_SLOTS)

    for _ in range(bound):
        try:
            selector = make_selector(hypothetical)
            candidates = await selector.select_candidates(list(workers))
        except Exception as e:
            # A selector that raises means this worker's capacity is unknown,
            # not zero. Returning what was already proven keeps the group
            # scheduler working off a floor rather than off a guess, and the
            # difference is only ever "this domain looks smaller than it is" —
            # which costs a tighter placement, never a wrong one.
            logger.warning(
                "Stopped counting capacity on %s after %d: %s",
                ", ".join(str(getattr(w, "name", w.id)) for w in workers),
                result.slots,
                e,
            )
            result.unavailable = f"{type(e).__name__}: {e}"
            return result

        # Taken on every round rather than only on the one that finds no room:
        # a role that fits still has to appear in a group's breakdown, priced
        # the same way as the role that did not.
        result.claim = _claim_of(selector) or result.claim

        candidate = _usable(candidates)
        if candidate is None:
            # The round that found no room is the one worth quoting, and it is
            # always this one: the loop stops here. Asked of the selector
            # rather than reconstructed, because the sentences it produces are
            # the same ones the single-instance refusal shows.
            result.notes = list(_notes_of(selector))
            return result

        result.slots += 1
        result.placements.append(candidate)
        hypothetical.append(_stand_in_for(candidate))

    return result


def _notes_of(selector) -> List[str]:
    """The selector's own account of why nothing fit.

    Defensive because `make_selector` is a caller-supplied callable and the
    tests hand in stubs: a counting pass must not fail over a diagnostic, and
    a missing explanation is a worse message rather than a broken schedule.
    """
    getter = getattr(selector, "get_messages", None)
    if getter is None:
        return []
    try:
        return [note for note in (getter() or []) if note]
    except Exception as e:  # pragma: no cover - diagnostics only
        logger.debug("Selector could not explain itself: %s", e)
        return []


def _claim_of(selector) -> Optional[MemberResourceClaim]:
    """What the selector priced one member at, if it prices that way.

    Defensive for the same reason `_notes_of` is: `make_selector` is
    caller-supplied and the tests hand in stubs, and a counting pass must not
    fail over a figure that only ever appears in a message.
    """
    getter = getattr(selector, "get_resource_claim", None)
    if getter is None:
        return None
    try:
        return getter()
    except Exception as e:  # pragma: no cover - diagnostics only
        logger.debug("Selector could not price a member: %s", e)
        return None


def _usable(candidates) -> Optional[Any]:
    """The first candidate that is a real fit.

    Overcommitted candidates are refused. The selectors offer them so a single
    deployment can start anyway on a busy worker, which is the right answer for
    one instance and the wrong one here: counting them would let a domain
    advertise room it does not have, and a group admitted on that count starts
    every one of its members into contention at once.
    """
    for candidate in candidates or []:
        if getattr(candidate, "overcommit", None):
            continue
        return candidate
    return None


def _stand_in_for(candidate) -> _PlacedStandIn:
    """The candidate as the allocation accounting reads it.

    Keyed off the candidate's own worker rather than the one the caller asked
    about: with several machines in play they are not the same, and the
    subordinates ride along on `distributed_servers`, which
    `compute_worker_allocated` already reads and bills to the right machine.
    """
    return _PlacedStandIn(
        worker_id=candidate.worker.id,
        gpu_indexes=candidate.gpu_indexes,
        gpu_type=candidate.gpu_type,
        computed_resource_claim=candidate.computed_resource_claim,
        distributed_servers=_subordinates_of(candidate),
    )


def _subordinates_of(candidate):
    """Carry a multi-worker placement's other halves, or nothing.

    A distributed candidate consumes VRAM on workers besides this one, and
    ``compute_worker_allocated`` finds that through ``distributed_servers``.
    Dropping it would leave those workers looking untouched, and a later round
    would hand out the same cards twice.
    """
    subordinates = getattr(candidate, "subordinate_workers", None)
    if not subordinates:
        return None
    # Duck-typed rather than constructing the schema object: the only consumer
    # walks `.subordinate_workers`, and importing the model here would tie this
    # module to the ORM for one attribute.
    return _Subordinates(subordinate_workers=list(subordinates))


@dataclass
class _Subordinates:
    subordinate_workers: List[Any]
