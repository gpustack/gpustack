"""Place every member of a group at once, in the tightest domain that fits.

The question a per-instance scheduler cannot answer is "can all of these land
together". Asking it once per member and taking the first fit each time is not
the same question: a greedy walk can place members one and two in a way that
makes three and four impossible, while a different assignment would have fitted
all four. This solves for the whole group instead, and returns the placement it
proved rather than a verdict someone else has to reproduce.

That last part is the point. A feasibility *pre-check* — compute a packing, say
"yes", then let the ordinary greedy scheduler place the members — can be right
and still fail: the greedy walk is not obliged to rediscover the packing the
check found. Both are correct and the group still never starts. Here the search
and the assignment are the same pass, so the answer cannot disagree with itself.

Adapted from koordinator's network-topology solver, whose shape has been in
production; the parts it does not have are noted where they appear.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from gpustack.topology.tree import (
    NODE_LAYER,
    ROOT_LAYER,
    GatherScope,
    TopologyNode,
    tree_scopes,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoleDemand:
    """How many members of one role, and how big each is.

    ``weight`` orders the roles when they compete for the same cards: the
    hungriest goes first, because a role that needs whole cards cannot use what
    a role taking slices has left behind, while the reverse usually works.
    Classic first-fit-decreasing, and it is the difference between "4P4D fits"
    and "4P4D fits only if you happen to place P first".
    """

    role: str
    replicas: int
    weight: float = 0.0


@dataclass(frozen=True)
class GatherRequest:
    """Where the group must fit, and whether "must" is meant literally.

    ``layer`` is a bare name: a name identifies one rung of the cluster's one
    chain, so there is nothing to carry beside it.
    """

    layer: Optional[str] = None
    must: bool = False


@dataclass(frozen=True)
class GroupScore:
    """What ranked one placement above the others, in the terms that decided it.

    ``total`` is the number compared; ``terms`` is the breakdown behind it,
    named as the scorer names them.

    **Carried, not recomputed.** Whatever a log line or a deployment form
    shows about a choice has to be the numbers the choice was made with, and
    evaluating the same terms a second time against the same placement is a
    second chance to disagree -- a stale cache, a knob read twice, a set of
    warm workers that moved in between.
    """

    total: float
    terms: Dict[str, float] = field(default_factory=dict)

    def describe(self) -> str:
        """The terms as one line, e.g. ``pair 0.50, file 0.00``."""
        if not self.terms:
            return f"score {self.total:.2f}"
        return ", ".join(f"{name} {value:.2f}" for name, value in self.terms.items())


@dataclass
class GroupPlacement:
    """One way to place the whole group: a candidate while the layer is being
    ranked, and the outcome once it has won.

    Both roles, one class, because a wrapper around this would carry nothing
    of its own -- the thing being ranked *is* the assignment. What the two
    roles do need is somewhere to keep the verdict, which is ``score``: the
    per-instance path keeps its own score on its candidates for the same
    reason, since a placement whose deciding number is thrown away cannot tell
    anyone why it beat the alternatives.

    ``path`` addresses the domain root-to-leaf instead of naming it, because a
    domain's name is unique only among its **siblings**: two zones each
    holding a ``rack-1`` are two domains under one name, and the unclassified
    bucket exists once per parent. A name is enough to print. It is not enough
    to tell two placements apart, which is exactly what ranking them needs.
    """

    layer: str
    path: List[str] = field(default_factory=list)
    # role -> [primary worker_id, ...], one entry per replica. The **contract
    # with `commit`**, which takes exactly this list and answers with one
    # candidate per entry, so its shape is not free to change.
    assignments: Dict[str, List[int]] = field(default_factory=dict)
    # (role, primary) -> every machine that member occupies, primary first.
    # Present only for a member too wide for one machine.
    #
    # Beside `assignments` rather than inside it, for the contract above. And
    # carried at all because a placement that cannot see it is ranked wrong,
    # not merely described loosely: a prefill on machines 1 and 2 with its
    # decode on 2 reads as entirely remote, and the group's machine count
    # comes out one short.
    spans: Dict[Tuple[str, int], List[int]] = field(default_factory=dict)
    # Set on the winner by `_pick`. None when nothing was ranked -- a single
    # candidate, or a caller with no policy to express -- which is a fact
    # worth being able to read rather than a zero to be mistaken for one.
    score: Optional[GroupScore] = None

    @property
    def domain(self) -> str:
        """The address as one readable string, e.g. ``zone-b / rack-1``."""
        return " / ".join(self.path)

    def worker_ids(self) -> List[int]:
        """One primary per member, in role order -- what `commit` consumes.

        Not the machines the group occupies: a member spanning machines
        appears here once. Use `machines` for that.
        """
        return [wid for ids in self.assignments.values() for wid in ids]

    def machines_of(self, role: str, index: int) -> List[int]:
        """Every machine the ``index``-th member of ``role`` occupies.

        A single machine for every role placed today. ``[primary]`` when the
        member does not span, which keeps a caller from having to know
        whether it does.
        """
        primary = self.assignments[role][index]
        return list(self.spans.get((role, primary), [primary]))

    def machines(self) -> Set[int]:
        """Every machine the whole group occupies, spans included."""
        out: Set[int] = set()
        for role, primaries in self.assignments.items():
            for index in range(len(primaries)):
                out.update(self.machines_of(role, index))
        return out

    def splits_a_member(self) -> bool:
        """Whether any single member is spread over more than one machine."""
        return any(len(machines) > 1 for machines in self.spans.values())


@dataclass
class GroupInfeasible:
    """Why it did not fit, in terms the deployment form can show.

    Carries the best domain it found rather than only the shortfall, because
    "the largest rack is 2 cards short" is actionable and "it does not fit" is
    not.
    """

    reason: str
    layer: Optional[str] = None
    best_path: List[str] = field(default_factory=list)
    # The role the walk stopped on. Carried so the caller can ask the capacity
    # function what that role's selectors said -- the claim in GiB and what
    # stood in its way -- and put it beside a count that cannot express either.
    role: Optional[str] = None
    needed: int = 0
    available: int = 0
    # Workers whose capacity could not be measured at all. `available` counts
    # only what was actually established, so without this a cluster nobody
    # could measure and a cluster that is genuinely full produce the same
    # refusal — and "full" is the one answer that stops an operator looking
    # for a mistake. Measured on a live host: one missing global config, and
    # every worker reported zero.
    unmeasured: int = 0

    @property
    def best_domain(self) -> str:
        """The roomiest domain's address as one readable string."""
        return " / ".join(self.best_path)


# async capacity(role, worker_ids, already_placed) -> {worker_id: slots}
#
# A worker whose capacity could not be determined is left OUT of the mapping.
# Absent means unknown; present-and-zero means measured and full. Collapsing
# the two would make an unmeasurable cluster indistinguishable from a full
# one.
#
# Required to be decomposable per worker: the answer for a set of workers is
# the union of the answers for its members. The real implementation asks each
# worker separately anyway, and the property is what lets the domain-sizing
# pass below run once for the whole tree instead of once per domain per
# layer, each of which would be a full selector sweep.
#
# `already_placed` is what this solve has committed so far, in the shape the
# allocation accounting reads. Passing it back is what keeps the capacity of
# the second role honest about what the first role took.
#
# Async because the only real implementation is `count_offer_slots`, which
# drives the resource-fit selectors, which are async all the way down.
CapacityFn = Callable[[str, Sequence[int], Sequence[object]], Awaitable[Dict[int, int]]]

# score(placement) -> float or GroupScore, higher total is better.
#
# A bare float is accepted so a caller with one number to compare need not
# build a breakdown it has nothing to put in; `_pick` wraps it.
#
# Ranks the placements found **at one layer**, and nothing about the layer
# itself: see `solve_group_placement` for why comparing across layers is a
# question nobody can answer. Supplied by the caller because every term in it
# is policy the solver must not know -- which roles pair with which, where a
# model's files already sit. `None` means take the tightest fitting domain,
# which is what a caller with no policy to express wants.
GroupScoreFn = Callable[[GroupPlacement], float]


async def solve_group_placement(
    root: TopologyNode,
    roles: Sequence[RoleDemand],
    capacity: CapacityFn,
    scopes: Sequence[Union[GatherScope, str]],
    gather: GatherRequest = GatherRequest(),
    attendants: Sequence[RoleDemand] = (),
    preference: Optional[Dict[int, int]] = None,
    score: Optional[GroupScoreFn] = None,
    limit: int = 16,
) -> object:
    """Place the group, or say why not.

    ``scopes`` is tightest first, and the **tightest layer holding the group
    wins outright** — a looser one is never compared against it. Inside that
    layer every fitting domain is a candidate, and ``score`` ranks them.

    **Why the layer is a strict key and not a term in the score.** Comparing a
    host-level placement against a rack-level one needs an exchange rate --
    "are two more local pairs worth crossing a rack?" -- and no one can set
    it. Holding the layer fixed removes the dimension from the comparison
    instead of pricing it. The same argument runs in the per-instance path,
    where `narrow_to_tightest_internal_spread` narrows **by removal** before
    scoring, with the note that a summing chain cannot hold two strict
    priorities.

    ``score`` is optional and ``None`` reproduces "take the tightest fitting
    domain" exactly: with nothing to rank by, the first candidate found is the
    smallest domain at the tightest layer, which is the answer that needs no
    policy. ``limit`` caps how many candidates one layer yields, since domains
    are examined smallest-first and the tail of a wide layer buys ranking
    quality at a full selector sweep each.

    ``preference`` orders workers **inside** a domain, as the last-resort
    tie-break in `_share_out`. It is a rank map the caller builds, so the
    solver ranks without knowing what it is ranking by.

    The two strategies differ only in where the walk is allowed to stop:

    - ``MustGather(X)`` is a **hard floor**. The search runs from the
      tightest scope up to and including ``X`` and then stops — it does *not*
      step to X's parent, and it does not fall back to the cluster root. If
      nothing at or below ``X`` holds the group, the deployment is refused.
      That refusal is the entire behaviour the strategy adds; the solver was
      already placing into the tightest domain that fits.
    - ``PreferGather`` (``must=False``) keeps widening up the chain, to its
      top, and then to the cluster root, which always fits. It cannot fail on
      gather grounds.
    """
    total = sum(r.replicas for r in roles)
    if total <= 0:
        return GroupPlacement(layer=NODE_LAYER, assignments={})

    if scopes and isinstance(scopes[0], str):
        scopes = tree_scopes(root, list(scopes))  # type: ignore[arg-type]
    scopes = list(scopes)  # type: ignore[arg-type]
    names = [s.name for s in scopes]

    ordered_roles = sorted(roles, key=lambda r: (-r.weight, r.role))
    # Decided once, and used for both the ceiling and the root fallback below.
    # Deriving them separately is how the first version came to log "ignoring
    # this requirement" and then refuse the deployment in its name: the ceiling
    # honoured the unknown layer by standing down, while the fallback still saw
    # `must` set and stayed switched off.
    enforced = _enforced_gather(names, gather)
    ceiling = names.index(enforced.layer) if enforced.layer else len(scopes) - 1
    best: Optional[GroupInfeasible] = None

    # Every domain is sized by the hungriest role with nothing placed, which
    # makes it one question asked once for the whole tree rather than once per
    # domain per layer. The hungriest role is the honest yardstick: a domain
    # with room for eight slices and no whole card is not eight units of room
    # to a group whose first role needs whole cards.
    sizing = await _size_domains(
        capacity, ordered_roles[0].role, root.descendant_worker_ids()
    )

    # Tightest first. Stopping at `ceiling` is the whole of MustGather: without
    # it the walk continues widening until the cluster root, which always fits
    # and is exactly the outcome the operator asked not to get.
    for scope in scopes[: ceiling + 1]:
        domains = _gatherable_domains(scope.domains)
        # Tightest fitting domain first, off the one sizing pass above.
        # The sizing pass can prove a domain holds not one member of the
        # hungriest role, and such a domain cannot hold the group -- so it is
        # skipped before `_fit_in_domain`, which costs a selector sweep per
        # role. "Proven" is the operative word: see the helper.
        #
        # 🔑 Unless it would empty the layer. A cluster that is genuinely full
        # is every domain proven hopeless, and skipping them all leaves
        # nothing to build the refusal from -- the reader gets "no domain has
        # any capacity" where they had "the roomiest one holds 0", which is
        # the sentence that says the shortfall. So the skip is an
        # optimisation over the *examinable* domains and never the reason a
        # layer goes unexamined.
        viable = [
            d
            for d in domains
            if not _proven_hopeless(d.descendant_worker_ids(), sizing)
        ]
        sized = [
            (
                sum(sizing.get(w, 0) for w in d.descendant_worker_ids()),
                d.path(),
                d,
            )
            for d in (viable or domains)
        ]

        fits: List[GroupPlacement] = []
        for _size, _path, domain in sorted(sized, key=lambda t: (t[0], t[1])):
            placement = await _fit_in_domain(
                domain,
                scope.name,
                ordered_roles,
                capacity,
                attendants,
                attendant_worker_ids=(
                    None if enforced.must else root.descendant_worker_ids()
                ),
                preference=preference,
            )
            if isinstance(placement, GroupPlacement):
                fits.append(placement)
                if score is None or len(fits) >= max(limit, 1):
                    # Nothing to rank by, or enough to rank: the domains were
                    # examined smallest-first, so `fits[0]` is already the
                    # answer a caller with no policy wants and the tail of a
                    # wide layer buys ranking quality at a sweep each.
                    break
                continue
            # `>=` so a tie is won by the later, wider scope. With `>` the
            # refusal for a rack-level requirement could name a single host,
            # since the leaf scope is examined first and its domains are just
            # as short of room — a message that contradicts itself.
            if best is None or placement.available >= best.available:
                best = placement

        if fits:
            # This layer holds the group, so no looser one is consulted -- not
            # even to compare. See the docstring: across layers the comparison
            # needs an exchange rate nobody can set.
            return _pick(fits, score)

    # The cluster root, last. It is not in `layers` — `layer_names` returns the
    # declared layers plus the leaf — and leaving it out of the search entirely
    # would mean a group too big for any declared domain could never be placed
    # at all, `must` or not. It is a real fallback rather than a domain anyone
    # gathers into: everything is under it, so reaching here means only that
    # the members are somewhere in this cluster.
    if not enforced.must:
        placement = await _fit_in_domain(
            root,
            root.layer,
            ordered_roles,
            capacity,
            attendants,
            # The root's own name is `ClusterTopologyLayer`, an internal layer
            # id. Passing its workers explicitly takes the "this cluster"
            # branch, which is both what this call means and the only wording
            # an operator can read.
            attendant_worker_ids=root.descendant_worker_ids(),
            preference=preference,
        )
        if isinstance(placement, GroupPlacement):
            # Not ranked, and there is nothing to rank: the root is one domain
            # and it is a fallback rather than a choice -- reaching it means
            # only that the members are somewhere in this cluster.
            return placement
        # `>=`, for the same reason the loop above uses it: a tie goes to the
        # wider scope, and nothing is wider than the root. It also decides
        # which sentence the operator gets, because the root is the only
        # domain that sees every worker — so it is the only one that can say
        # "capacity could not be measured on 1 of 2 workers" rather than the
        # flat "not enough room" a single host reports about itself. With `>`
        # those two tie and the host wins, and a fleet with a worker whose
        # telemetry had stopped read as simply full.
        if best is None or placement.available >= best.available:
            best = placement

    if best is None:
        return GroupInfeasible(
            reason="No topology domain has any capacity for this group.",
            needed=total,
            # Carried for the same reason every other refusal carries it: the
            # caller asks the capacity function what this role's filters said,
            # and `notes_for(None)` answers nothing at all. Reached when a
            # `must` layer has no domain to walk — a cluster with no workers
            # left in it — which is precisely when the filter lines ("Matched
            # 0/3 workers by label selector") are the only account of why. The
            # hungriest role is the one the sizing pass above already swept, so
            # it is both the meaningful answer and the only one with notes.
            role=ordered_roles[0].role,
        )
    return _describe(best, enforced, attendants)


async def _size_domains(
    capacity: CapacityFn, role: str, worker_ids: Sequence[int]
) -> Dict[int, int]:
    """One sweep of the tree, for ordering domains and skipping empty ones.

    Asked of ``capacity.sizing`` where the capacity function offers it, and
    that distinction is load-bearing rather than tidy. A capacity function may
    apply rules that are only decidable *inside* a domain -- keeping one role
    on one size of GPU is one, and it works by picking the band holding the
    most members. Picked across the whole tree, it zeroes every worker of the
    losing bands, so a domain built entirely from them sizes as empty and is
    dropped from the search, although its members would have sat on one size
    of card perfectly well.

    Falls back to the ordinary call for a capacity function with no such
    rules, which is every stub and the shape this took before.

    Args:
        capacity: The capacity function this solve was given.
        role: The hungriest role, which is the honest yardstick for a domain.
        worker_ids: Every worker under the tree.

    Returns:
        worker_id -> slots, with absence still meaning unmeasured.
    """
    sizing = getattr(capacity, "sizing", None)
    if sizing is not None:
        return await sizing(role, worker_ids)
    return await capacity(role, worker_ids, [])


def _proven_hopeless(members: Sequence[int], sizing: Dict[int, int]) -> bool:
    """Whether the sizing pass **established** this domain holds no member.

    "Established" is the whole of it. Absent from `sizing` means *unmeasured*,
    and a domain holding an unmeasured worker is not proven hopeless -- so it
    is examined, because the refusal it produces is the only one that can say
    "capacity could not be measured on 4 workers" rather than "the cluster is
    full". Those two call for opposite reactions from an operator, and "full"
    is the one that makes them stop looking for a mistake.

    An empty domain is hopeless without being unmeasured, and saying so here
    saves `_fit_in_domain` a call that only reports the same thing.

    Args:
        members: Worker ids under the domain.
        sizing: The one-pass count for the hungriest role, nothing placed.

    Returns:
        True when every member was measured and measured at zero.
    """
    if not members:
        return True
    return all(sizing.get(worker_id) == 0 for worker_id in members)


def _pick(
    fits: Sequence[GroupPlacement],
    score: Optional[GroupScoreFn],
) -> GroupPlacement:
    """The best of one layer's fitting placements.

    Two steps, and the order is the point.

    **Narrowing comes first, by removal.** A placement that splits one member
    across machines pays an all-reduce per layer per token, and the score
    cannot be trusted to outweigh that: the pairing term it is built from is
    unbounded in the per-instance path for exactly this reason, so any finite
    weight only moves the threshold. Dropped from the running
    instead, and only when a placement that keeps every member whole exists --
    a role wider than any single machine still places.

    The score sees what survives. Ties keep the earlier candidate, and the
    candidates arrived smallest-domain-first, so an unranked tie resolves to
    the tightest domain.
    """
    whole = [p for p in fits if not p.splits_a_member()]
    field_of = whole or list(fits)
    if score is None or len(field_of) == 1:
        return field_of[0]

    ranked = [(placement, _as_score(score(placement))) for placement in field_of]
    best, best_score = ranked[0]
    for placement, value in ranked[1:]:
        if value.total > best_score.total:
            best, best_score = placement, value
    best.score = best_score

    logger.debug(
        "Ranked %d placements at %s: %s",
        len(ranked),
        best.layer,
        "; ".join(
            f"{placement.domain or '<root>'} = {value.total:.3f} "
            f"({value.describe()})"
            for placement, value in ranked
        ),
    )
    return best


def _as_score(value: Union[float, GroupScore]) -> GroupScore:
    """A scorer's answer as a `GroupScore`, whichever shape it came in.

    Accepting a bare float keeps the protocol cheap for a caller that has one
    number and nothing to break it down into -- which is most callers, and
    every test that only needs an ordering.
    """
    if isinstance(value, GroupScore):
        return value
    return GroupScore(total=float(value))


def _span_of(capacity: CapacityFn, role: str, worker_id: int) -> List[int]:
    """The machines a member of ``role`` at ``worker_id`` occupies.

    Duck-typed, like every other optional thing asked of a caller-supplied
    callable here: the tests hand in plain functions, and a capacity function
    with nothing to say is read as "one machine" -- which is the truth for
    every role that fits on a single machine, i.e. every role placed today.
    """
    spans_for = getattr(capacity, "spans_for", None)
    if spans_for is None:
        return [worker_id]
    try:
        return list(spans_for(role, worker_id)) or [worker_id]
    except Exception as e:  # pragma: no cover - diagnostics only
        logger.debug("Capacity could not report a span for %r: %s", role, e)
        return [worker_id]


def _describe(
    best: GroupInfeasible,
    enforced: GatherRequest,
    attendants: Sequence[RoleDemand],
) -> GroupInfeasible:
    """Turn the numbers every refusal already carries into its sentence.

    Spent on every path, not only the two MustGather branches. The default
    path -- no gather requirement at all, which is most deployments -- would
    otherwise reach the caller as the bare string `_fit_in_domain` sets, "not
    enough room", with the shortfall sitting in the fields unread. Beside the
    single-instance refusal, which names the claim and what the roomiest worker
    had, that is an order of magnitude less information about the same event.

    Without a floor the search ends at the cluster root, and the root wins ties
    (see the `>=` in the caller), so `best` is the root's own attempt: its
    numbers are the whole cluster's, and the sentence says so rather than
    naming a domain whose only name is an internal layer id.
    """
    if best.role and best.role in {a.role for a in attendants}:
        # An attendant refusal already says which role and where, and it is the
        # one case the counting sentences below would misdescribe: the gang
        # fits, so "the cluster has room for 4" of 4 would read as a
        # contradiction of the refusal it is attached to.
        return best
    if enforced.must and not best.unmeasured:
        best.reason = (
            f"The group needs {best.needed} placements in one "
            f"{enforced.layer!r}, and the roomiest one holds {best.available}."
        )
    elif enforced.must:
        # Deliberately not phrased as a capacity verdict: the number behind it
        # is a floor, not a measurement.
        best.reason = (
            f"The group needs {best.needed} placements in one "
            f"{enforced.layer!r}, but capacity could not be measured on "
            f"{best.unmeasured} worker(s), so whether it fits is unknown."
        )
    elif not best.unmeasured:
        best.reason = (
            f"The group needs {best.needed} placements and the cluster has "
            f"room for {best.available}."
        )
    else:
        # Same distinction the MustGather branch draws, and for the same
        # reason: "full" is the one answer that stops an operator looking for
        # a mistake, and here the mistake is usually a worker that stopped
        # reporting rather than a cluster that is out of cards.
        best.reason = (
            f"The group needs {best.needed} placements and the cluster has "
            f"room for {best.available}, but capacity could not be measured "
            f"on {best.unmeasured} worker(s) — the shortfall may be smaller "
            f"than it looks, or there may be none."
        )
    return best


def _enforced_gather(names: Sequence[str], gather: GatherRequest) -> GatherRequest:
    """The requirement as it will actually be applied.

    A `must` naming a layer this cluster no longer declares is dropped
    *entirely* — not just from the ceiling. Half-dropping it is the bug this
    function exists to make impossible: the walk would stand down for the
    unknown layer while the root fallback stayed disabled, so a group that fits
    only at the cluster root would be refused in the name of a layer the code
    had just announced it was ignoring.

    A stale name means someone renamed or removed a layer somewhere else. That
    must not take a running deployment down.
    """
    # A `prefer` loses its layer here, and that costs nothing at placement
    # time — which is worth writing down, because dropping a value the operator
    # chose reads like a bug until you check what the layer is for. It is a
    # CEILING: the walk stops there instead of widening to the cluster root.
    # And the walk is already tightest-first, returning the first domain that
    # fits, so «prefer rack» and «prefer host» produce the same placement — the
    # tightest one available. The only thing a ceiling adds is the refusal, and
    # refusing is exactly what `prefer` says not to do.
    #
    # The layer is not discarded, only unused here: `_gather_unmet` reads it
    # off the model afterwards to say whether the target was met.
    if not gather.must or not gather.layer:
        return GatherRequest(layer=None, must=False)
    if gather.layer not in names:
        logger.warning(
            "Ignoring a gather requirement on unknown topology layer %r; "
            "the group will be placed as if none had been asked for.",
            gather.layer,
        )
        return GatherRequest(layer=None, must=False)
    return gather


def _gatherable_domains(domains: Sequence[TopologyNode]) -> List[TopologyNode]:
    """The domains of one scope that mean something to gather into.

    The unclassified bucket is excluded, and that is not a detail. It holds the
    workers whose position is *unknown*; gathering a group into it would be
    claiming they are together on the strength of them all being unlabelled.
    The same rule makes two unclassified workers report no common layer, and
    the two have to agree — otherwise the solver would gather onto a domain
    that the distance function says does not exist.
    """
    return [d for d in domains if not d.is_unclassified and d.layer != ROOT_LAYER]


async def _fit_in_domain(
    domain: TopologyNode,
    layer: str,
    roles: Sequence[RoleDemand],
    capacity: CapacityFn,
    attendants: Sequence[RoleDemand] = (),
    attendant_worker_ids: Optional[Sequence[int]] = None,
    preference: Optional[Dict[int, int]] = None,
) -> object:
    """Place every role inside one domain, or report how far it got.

    ``attendants`` are the group's members that occupy no accelerator -- the
    router. They are checked, never assigned: the solver answers "which worker"
    for the gang, and a router is created a pass later by the dependency gate,
    against peer addresses that do not exist while this runs.

    **Checked at all because they were invisible.** They are left out of
    ``roles`` on purpose: counting a router among the demands would make a 4P4D
    need room for nine placements in one domain, and under ``MustGather`` that
    refuses racks that would have served. But left out entirely, the group is
    admitted onto hardware with no room for its router, which then fails to
    schedule and the group never becomes servable -- a router answers every
    request, so a group without one is a group that serves nothing. Its
    ``evaluate_group`` twin said so outright: "a cluster with room for the GPU
    members but not for the router still evaluates as compatible".

    So they constrain feasibility without constraining size: they are not in
    ``needed``, they do not order domains, and they never widen the gang.

    ``attendant_worker_ids`` is where they are allowed to land. The caller
    passes this domain's workers under ``MustGather`` -- a floor the operator
    set, and a router away from its members crosses that boundary on every
    request -- and the whole cluster otherwise, so a tight domain is not
    rejected over a router that had somewhere else to go.
    """
    worker_ids = domain.descendant_worker_ids()
    total = sum(r.replicas for r in roles)
    if not worker_ids:
        return GroupInfeasible(
            reason="empty domain",
            layer=layer,
            best_path=domain.path(),
            needed=total,
            # A domain with no workers is the one refusal whose count explains
            # nothing -- "room for 0" is true of an empty cluster however the
            # fleet got that way. The role is what lets the caller attach the
            # filters' own lines, which do say. First in the order the caller
            # sorted them: the hungriest role, the same one the sizing pass
            # swept, so its filter chain has already run.
            role=roles[0].role if roles else None,
        )

    placement = GroupPlacement(layer=layer, path=domain.path())
    placed: List[object] = []
    placed_total = 0

    for role in roles:
        slots = await capacity(role.role, worker_ids, placed)
        share = _share_out(slots, role.replicas, placed, preference)
        if share is None:
            unmeasured = len([w for w in worker_ids if w not in slots])
            return GroupInfeasible(
                reason=(
                    "not enough room"
                    if not unmeasured
                    else f"capacity could not be measured on {unmeasured} of "
                    f"{len(worker_ids)} workers, and what could be measured is "
                    "not enough"
                ),
                layer=layer,
                best_path=domain.path(),
                role=role.role,
                needed=total,
                available=placed_total + sum(slots.values()),
                unmeasured=unmeasured,
            )
        assigned: List[int] = []
        for worker_id, count in share:
            assigned.extend([worker_id] * count)
            placed.extend([_Committed(worker_id, role.role)] * count)
            # Copied onto the placement as it is decided. The width was
            # discovered by the capacity function and lives in its own
            # bookkeeping; a placement that has to go back and ask cannot be
            # ranked on its own, and ranking is the whole reason it exists
            # separately from the commit that follows.
            span = _span_of(capacity, role.role, worker_id)
            if len(span) > 1:
                placement.spans[(role.role, worker_id)] = span
        placement.assignments[role.role] = assigned
        placed_total += role.replicas

    # Last, and with the gang standing in: the question is whether a router
    # fits *beside* the members, on what they leave behind.
    # `None` means "inside this domain", which is what `MustGather` asks for.
    # Stated as its own name rather than left to an identity comparison on the
    # two lists: the caller that widened the search to the whole cluster passes
    # the root's workers, and `where is not worker_ids` read that correctly
    # only by accident -- it also reported the root's own domain name, which is
    # the internal layer id `ClusterTopologyLayer`.
    scoped_to_domain = attendant_worker_ids is None
    where = worker_ids if scoped_to_domain else list(attendant_worker_ids)
    for attendant in attendants:
        slots = await capacity(attendant.role, where, placed)
        if _share_out(slots, attendant.replicas, placed, preference) is None:
            unmeasured = len([w for w in where if w not in slots])
            scope = (
                f"{' / '.join(domain.path())!r}" if scoped_to_domain else "this cluster"
            )
            return GroupInfeasible(
                reason=(
                    f"The group's accelerator-bearing members fit, but nothing "
                    f"in {scope} has room for the {attendant.role!r} role"
                    + (
                        "."
                        if not unmeasured
                        else f", and capacity could not be measured on "
                        f"{unmeasured} of {len(where)} workers."
                    )
                ),
                layer=layer,
                best_path=domain.path(),
                role=attendant.role,
                needed=total,
                # The gang's own arithmetic, unchanged: an attendant is not a
                # placement the group is short of, and counting it here would
                # make the shortfall read as one card too few.
                available=placed_total,
                unmeasured=unmeasured,
            )

    return placement


@dataclass
class _Committed:
    """A placement this solve has made but not written anywhere yet.

    The worker id is what the capacity function needs; the role is what the
    tie-break below needs. Nothing else — the capacity function re-derives the
    real resource claim itself, and inventing one here would be a second,
    quieter accounting of the same placement.
    """

    worker_id: int
    role: str = ""


def _share_out(
    slots: Dict[int, int],
    replicas: int,
    placed: Sequence[object] = (),
    preference: Optional[Dict[int, int]] = None,
) -> Optional[List[Tuple[int, int]]]:
    """Hand ``replicas`` placements to the workers with the most room first.

    The direction here is the opposite of the one used to pick the domain,
    and both are right. Between domains the *smallest* one that fits wins, so
    the group leaves the least fragmentation behind for whoever comes next.
    Inside the chosen domain the *roomiest* worker fills first, so the group
    occupies as few workers as it can and its members stay as close as the
    domain allows. Swapping either is the single most likely way to port this
    wrongly, which is why both have a test that fails on exactly that swap.

    Among workers with equal room, the one already carrying fewer of this
    group's members wins. That tie-break is the cheap half of keeping the roles
    mixed: placing role by role, roomiest-first, otherwise packs all of one
    role onto the first workers and all of the next onto the rest — and with a
    router that pairs prefill and decode independently, an all-P/all-D split
    across two domains is the one arrangement where *no* pair is local.
    Homogeneous workers make this the common case, not a corner.

    ``preference`` is the caller's own ranking, consulted only once room and
    balance have tied. It is a rank map and nothing more -- what it ranks by
    (a worker that already holds the model's files, today) is policy, and the
    layer that knows the policy is the layer that builds the map. Absent, the
    ordering is what it was.
    """
    if replicas <= 0:
        return []
    if sum(slots.values()) < replicas:
        return None

    load: Dict[int, int] = {}
    for entry in placed:
        worker_id = getattr(entry, "worker_id", None)
        if worker_id is not None:
            load[worker_id] = load.get(worker_id, 0) + 1

    # Four keys, each with a name: room, then this group's own balance, then
    # whatever the caller prefers, then the worker id. The id is last so a
    # re-solve of an unchanged cluster produces an unchanged plan; otherwise
    # every reconcile would look like a spec change to anything comparing
    # placements.
    prefer = preference or {}
    ranked = [
        worker_id
        for worker_id, _ in sorted(
            slots.items(),
            key=lambda kv: (
                -kv[1],
                load.get(kv[0], 0),
                # Absent ranks worse than any stated preference. The caller
                # maps every warm worker to 0 to say "these are warm", so a
                # default of 0 made warm and cold tie on this key and dropped
                # the preference silently -- the between-domain score still
                # counted file locality, which is why nothing looked wrong.
                prefer.get(kv[0], len(prefer) + 1),
                kv[0],
            ),
        )
    ]

    # One at a time around the ranked workers, not "fill the first, then the
    # next". Greedy filling is what produces the all-P-here/all-D-there split:
    # the first role exhausts the roomiest workers, and the second has nowhere
    # left but the rest. Dealing round-robin costs nothing in locality — by the
    # time this runs, the group has already failed to fit on any single host,
    # so its members are spanning workers either way — and it is the difference
    # between every pair being remote and half of them being local.
    taken: Dict[int, int] = {}
    left = replicas
    while left > 0:
        progressed = False
        for worker_id in ranked:
            if left <= 0:
                break
            if taken.get(worker_id, 0) >= slots[worker_id]:
                continue
            taken[worker_id] = taken.get(worker_id, 0) + 1
            left -= 1
            progressed = True
        if not progressed:
            # Cannot happen: the total was checked above. Guarded anyway
            # because the alternative is an infinite loop.
            return None
    return [(worker_id, taken[worker_id]) for worker_id in ranked if worker_id in taken]
