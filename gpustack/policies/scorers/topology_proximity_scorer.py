import logging
from typing import Dict, List, Optional, Sequence

from gpustack.policies.base import (
    ModelInstanceScheduleCandidate,
    ScheduleCandidatesScorer,
)
from gpustack.schemas.models import ModelInstance, member_worker_ids
from gpustack.topology.tree import (
    NODE_LAYER,
    ROOT_LAYER,
    common_layer,
    order_layers,
)
from gpustack.topology.view import TopologyView

logger = logging.getLogger(__name__)


class TopologyProximityScorer(ScheduleCandidatesScorer):
    """Prefer a worker close to the group's placed members, by the declared tree.

    **Host identity is not distance.** `PairingAffinityScorer`, the other
    scorer that pulls a later member toward its group, compares `worker_id` and
    nothing else: between a worker in the members' own rack and one three racks
    away it is indifferent. Read on its own, «as close as possible» would stop
    at the host, and the tiers below it would be a target to report against
    rather than one to aim at.

    This reads the cluster's own tree instead: a candidate scores by the
    tightest layer it shares with the members already placed, deeper being
    better. Sharing only the root -- or only the unclassified bucket, which
    `common_layer` refuses to treat as togetherness -- scores nothing, which is
    the honest answer for "we do not know where these are".

    **Orthogonal to `PairingAffinityScorer`, not a replacement.** That one
    answers "which host has the most of the opposite role", which is the
    probability that a request's two ends land together and is a different
    question from distance -- under 3P1D it deliberately prefers the host with
    the single decode over the one with two prefills, and no notion of
    proximity would produce that. The two are summed: pairing decides among
    hosts, proximity decides among racks.

    **A scorer, never a filter.** A candidate reaches here only after the
    selector found it can hold the member, so this reorders workers that all
    fit. Being far away is a latency cost; the refusal for a floor that must be
    honoured is `GatherFloorFilter`, and it is a separate decision on purpose.

    **The one thing here that is NOT a score: internal spread.** How far a
    candidate is from the group and how far it is from *itself* are not
    comparable quantities -- the first carries the KV transfer, once per
    request; the second carries the tensor-parallel all-reduce, once per layer
    per token -- so no amount of the first may buy a worse second. That is a
    strict priority, and `narrow_to_tightest_internal_spread` is what expresses
    it: the caller drops the looser candidates *before* anything is scored.

    Deliberately not a number instead -- a penalty one rung larger than the
    closeness term can reach. That holds inside this scorer and stops holding
    on the chain, because the chain SUMS: `PairingAffinityScorer` is sized to
    outrank the rest of the chain and is unbounded by design (`max_score` per
    opposite sibling), so two siblings outweigh any such penalty and buy a
    split member. No finite constant can
    dominate an unbounded one; a bigger number only moves the threshold.

    The penalty stays in `score` anyway, and is not duplication: it is what a
    caller that skips the narrowing still gets. Once the narrowing has run,
    every surviving candidate has the same internal spread, so the term is a
    constant added to all of them and cannot reorder anything.
    """

    def __init__(
        self,
        group_id: Optional[str],
        model_instances: Sequence[ModelInstance],
        view: Optional[TopologyView],
        anchors: Sequence[str] = (),
        max_score: float = 100.0,
    ):
        self._group_id = group_id
        self._model_instances = list(model_instances)
        self._view = view
        # Which roles say where the group *is*. The router holds no weights, so
        # it cannot anchor the group -- the same reason `role_demands` leaves it
        # out of the gang -- but it is scored against the anchors like anyone.
        self._anchors = set(anchors)
        self._max_score = max_score

    @property
    def score_ceiling(self) -> float:
        """`max_score` times the tightest rung, not `max_score`.

        The one scorer on the chain whose `_max_score` is a per-rung step
        rather than a total: `score` pays `max_score * near`, and `near` is the
        depth of the tightest layer a candidate shares with a placed member --
        3 on the built-in zone/rack/host chain, more under a declared one. So
        the default 150 buys up to 450, which is what made the 200 written down
        for `PairingAffinityScorer` too small. Read off `_depths` rather than
        assumed for the same reason `score` does: the chain is the operator's.

        The `apart` term only ever subtracts, so it cannot raise this.
        """
        if self._max_score <= 0 or self._view is None:
            return 0.0
        return self._max_score * max(self._depths().values(), default=0)

    def narrow_to_tightest_internal_spread(
        self, candidates: List[ModelInstanceScheduleCandidate]
    ) -> List[ModelInstanceScheduleCandidate]:
        """Keep only the candidates that are tightest inside themselves.

        The first key of the ordering, applied by removal rather than by score
        -- see the class note for why it cannot be a number. Returns the list
        unchanged whenever the question does not arise, which is almost always:
        no topology, no group, the scorer switched off, or -- the common case --
        every candidate sitting on one machine, since a single machine has no
        pair to be apart and they all tie at the tightest rung.

        **Never empties the set, so it cannot make a scale-out unschedulable.**
        The bucket kept is the best one that exists: when a role is wider than
        any single machine every candidate spans, they tie, and this is a no-op.
        That is the difference between "prefer not to split" and "refuse to
        split", and only the first is wanted here -- refusing would turn a
        deployment that runs today into one that does not schedule.

        A candidate whose machines are not in the tree is kept regardless. Its
        position is unknown, and unknown is not the same as loose; dropping it
        would be a judgement made from missing data.
        """
        if (
            len(candidates) < 2
            or self._max_score <= 0
            or not self._group_id
            or self._view is None
        ):
            return candidates

        depth = self._depths()
        if not depth:
            return candidates

        leaves = {
            worker_id: node
            for node in _leaves(self._view.root)
            for worker_id in node.descendant_worker_ids()
        }

        located: List[tuple] = []
        unlocatable: List[ModelInstanceScheduleCandidate] = []
        for candidate in candidates:
            span = [
                leaves[worker_id]
                for worker_id in member_worker_ids(candidate)
                if worker_id in leaves
            ]
            if not span:
                unlocatable.append(candidate)
                continue
            located.append((_internal_depth(span, depth), candidate))

        if not located:
            return candidates

        tightest = max(internal for internal, _ in located)
        kept = [candidate for internal, candidate in located if internal == tightest]
        dropped = len(located) - len(kept)
        if dropped:
            # Logged at INFO because it is the one place a candidate leaves the
            # running for a reason no score can be inspected for afterwards.
            logger.info(
                "Dropped %d candidate(s) of group %s that would split the "
                "member more widely than necessary; %d remain at the tightest "
                "internal layer (depth %d).",
                dropped,
                self._group_id,
                len(kept) + len(unlocatable),
                tightest,
            )
        return kept + unlocatable

    async def score(
        self, candidates: List[ModelInstanceScheduleCandidate]
    ) -> List[ModelInstanceScheduleCandidate]:
        if (
            not candidates
            or self._max_score <= 0
            or not self._group_id
            or self._view is None
        ):
            return candidates

        placed = {
            worker_id
            for instance in self._model_instances
            if instance.group_id == self._group_id and instance.role in self._anchors
            # A member that spans machines is near a candidate from any of
            # them; measuring only from its primary would call a candidate
            # sharing a rack with the member's other half "far".
            for worker_id in member_worker_ids(instance)
        }
        if not placed:
            # Nothing to be near yet. Every candidate scores zero and the
            # resource scorers decide alone.
            return candidates

        depth = self._depths()
        if not depth:
            return candidates

        leaves = {
            worker_id: node
            for node in _leaves(self._view.root)
            for worker_id in node.descendant_worker_ids()
        }
        anchors = [leaves[worker_id] for worker_id in placed if worker_id in leaves]
        if not anchors:
            return candidates

        # **Lexicographic, not a weighted sum**, and the weight below is how
        # that is spelled. A candidate that spans machines has two distances
        # and they pull against each other: how far its own machines are from
        # each other, and how far the whole of it is from the rest of the
        # group. They are not comparable quantities -- the first carries the
        # tensor-parallel all-reduce, once per layer per token, and the second
        # carries the KV transfer, once per request. Adding them would let a
        # large enough gain on the second buy a worse first, which is never
        # right at that ratio.
        #
        # So internal spread decides, and closeness to the group only breaks
        # ties among candidates that are equally tight inside. Written as a
        # penalty rather than a bonus so a single machine -- which has no pair
        # to be apart, and is every candidate on the non-group path -- scores
        # zero for a worker whose position is unknown, and the closeness term
        # alone otherwise. `depth`
        # is bounded by the declared rungs, so a multiplier one past the
        # tightest makes any penalty outweigh any closeness gain.
        #
        # That last sentence is true of THIS scorer and false of the chain,
        # which is what `narrow_to_tightest_internal_spread` exists to fix: the
        # chain sums, and `PairingAffinityScorer` pays an unbounded
        # `max_score` per opposite sibling, so two of them outweigh this
        # penalty and buy a split member. Where the caller
        # narrows first this term is a constant across the survivors and
        # decides nothing; it is kept for the caller that does not.
        tightest = max(depth.values(), default=0)

        for candidate in candidates:
            span = [
                leaves[worker_id]
                for worker_id in member_worker_ids(candidate)
                if worker_id in leaves
            ]
            if not span:
                continue
            # The tightest layer this candidate shares with ANY placed member.
            # Any rather than all: the member is one process talking to one
            # peer at a time, so being in a rack with three of them is not
            # three times better than being in a rack with one -- that is
            # `PairingAffinityScorer`'s question, and it is already asked.
            near = max(
                (
                    depth.get(common_layer(anchor, node) or ROOT_LAYER, 0)
                    for anchor in anchors
                    for node in span
                ),
                default=0,
            )
            apart = tightest - _internal_depth(span, depth)
            candidate.score = (
                (candidate.score or 0)
                + self._max_score * near
                - self._max_score * (tightest + 1) * apart
            )

        return candidates

    def _depths(self) -> Dict[str, int]:
        """Layer id -> how tight it is, tightest highest.

        Read off the cluster's declaration rather than assumed, because the
        chain is the operator's: whether an accelerator domain sits above or
        below a rack is something they decided, and hardcoding an order here
        would be a second opinion about their own topology.
        """
        try:
            order = [spec.layer for spec in order_layers(self._view.specs)]
        except Exception as e:
            logger.debug("Could not order the topology layers: %s", e)
            return {}
        # Root-to-leaf, so the last is the tightest. The built-in host layer is
        # not in `specs`; it is tighter than anything declared, so it takes the
        # rank above the deepest one.
        depths = {layer: index + 1 for index, layer in enumerate(order)}
        depths[NODE_LAYER] = len(order) + 1
        depths[ROOT_LAYER] = 0
        return depths


def _internal_depth(span: List, depth: Dict[str, int]) -> int:
    """How tight this candidate is inside itself, loosest pair deciding.

    A single machine is as tight as it gets -- there is no pair to be apart --
    which is what keeps every placement made today scoring exactly as it did.
    """
    if len(span) < 2:
        return max(depth.values(), default=0)
    return min(
        depth.get(common_layer(a, b) or ROOT_LAYER, 0)
        for index, a in enumerate(span)
        for b in span[index + 1 :]
    )


def _leaves(node) -> List:
    """Every host node under `node`."""
    if not node:
        return []
    if node.layer == NODE_LAYER:
        return [node]
    out: List = []
    for child in getattr(node, "children", []) or []:
        out.extend(_leaves(child))
    return out
