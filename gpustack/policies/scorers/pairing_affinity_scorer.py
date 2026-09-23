import logging
from typing import Dict, List, Optional, Sequence

from gpustack.policies.base import (
    ModelInstanceScheduleCandidate,
    ModelInstanceScore,
    ModelInstanceScorer,
    ScheduleCandidatesScorer,
)
from gpustack.schemas.models import ModelInstance, RoleNameEnum, member_worker_ids

logger = logging.getLogger(__name__)


# prefill and decode pull on each other; nothing else in a group has an
# opposite. Written as a map rather than an `if` so a third paired role would
# be one line here and nowhere else.
_OPPOSITE = {
    RoleNameEnum.PREFILL.value: RoleNameEnum.DECODE.value,
    RoleNameEnum.DECODE.value: RoleNameEnum.PREFILL.value,
}


def local_share(entry, counts: Dict[int, int]) -> float:
    """What one member placed on ``entry``'s machines is worth, per `counts`.

    The objective is ``P(a pair is local) = (1/xy) * sum over pairs of
    |machines(p) & machines(d)| / |machines(p)|``, and this is its derivative:
    adding a member whose machines are ``M`` raises the numerator by
    ``(1/|M|) * sum over m in M of counts[m]``.

    So the **siblings** each count once per machine they sit on, while the
    **candidate's own width** divides -- a member spread over two hosts gets
    half of what each host offers, because only half of its ranks are there to
    use it. For a member on one machine, which is every member placed today,
    this is ``counts[worker]`` unchanged.
    """
    machines = member_worker_ids(entry)
    if not machines:
        return 0.0
    return sum(counts.get(machine, 0) for machine in machines) / len(machines)


def opposite_counts_by_worker(
    instances: Sequence[ModelInstance],
    group_id: Optional[str],
    opposite: Optional[str],
    include_draining: bool = True,
) -> Dict[int, int]:
    """`d_j` — this group's members of `opposite`, per worker.

    One ledger for both directions. Scaling a role out reads it for the
    largest `d_j` and scaling the same role in reads it for the smallest, and
    the two only stay each other's inverse while they count the same thing —
    so they count it here rather than twice.

    Counted from `worker_id` rather than from a RUNNING state. A sibling that
    has been placed but is still starting holds that worker's cards and will
    pair from there; excluding it would make the first two scale-outs of a
    burst both choose the same "empty" host for the same wrong reason.

    `include_draining` is the one thing the two directions disagree on today,
    and it defaults to the scale-out reading. A member inside its drain window
    is still serving and still holds its cards, but it is leaving — so for
    choosing a victim it is not a partner worth keeping a prefill next to.
    """
    counts: Dict[int, int] = {}
    if not group_id or not opposite:
        return counts
    for instance in instances:
        if getattr(instance, "group_id", None) != group_id:
            continue
        if getattr(instance, "role", None) != opposite:
            continue
        if not include_draining and getattr(instance, "draining_since", None):
            continue
        # Once per machine the member occupies, at full weight on each: a
        # decode spanning two hosts really does give a prefill on either one
        # something local to read. The division belongs to the *candidate's*
        # own width, not to the sibling being counted -- see `local_share`.
        for machine in member_worker_ids(instance):
            counts[machine] = counts.get(machine, 0) + 1
    return counts


class PairingAffinityScorer(ScheduleCandidatesScorer):
    """Pull a scaled-out prefill toward decode, and a decode toward prefill.

    **The opposite role, not "the group".** The obvious rule — prefer the
    worker already holding most of this group — is wrong, and only looks right
    because it coincides with this one when P and D are balanced. What a
    request actually pays for is a prefill and a decode being on the same host,
    so with `x` prefills, `y` decodes, and `p_j` / `d_j` of each on worker `j`,
    a router that picks the two ends independently gives::

        P(a P/D pair is local) = (1/xy) * sum_j p_j * d_j

    Adding a prefill on worker `j` raises the numerator by `d_j`, and the
    denominator `(x+1)*y` is the same wherever it lands. So the best worker is
    the one with the most *decodes*. Under 3P1D the group-count rule would pile
    the new prefill onto the prefills and move that sum by nothing at all.

    **The direction is the reverse of the group solver's tie-break**, and
    both are right. Forming a group, `_share_out` prefers the worker holding
    *fewer* of the group's members, which keeps one role from monopolising the
    roomiest workers and producing the all-P-here/all-D-there split — the one
    arrangement where no pair is local. That is a defence against a worst case
    on an empty board. This runs against a board that already has a
    distribution on it, and improves the sum it actually has.

    **A scorer, not a filter**, like every locality policy beside it: a
    candidate only reaches here once the selector has found it can hold the
    member, so affinity reorders workers that all fit and can never make a
    scale-out unschedulable.

    **Never reached by a role-less model.** The scorer is added to the chain
    only for a member that has a `group_id`, which only a model with `roles`
    ever has. Existing single-role deployments score exactly as they did.
    """

    def __init__(
        self,
        group_id: Optional[str],
        role: Optional[str],
        model_instances: Sequence[ModelInstance],
        # A fallback, not the production weight. The scale-up chain sizes
        # this against what the rest of that chain can pay
        # (`_pairing_affinity_max_score`), because no constant survives the
        # next scorer being added -- 200 was chosen against 100 + 5 and was
        # already beaten by `TopologyProximityScorer`'s 450. Kept at the old
        # value so a caller that builds this scorer alone behaves as it did.
        max_score: float = 200.0,
    ):
        self._group_id = group_id
        self._opposite = _OPPOSITE.get(role or "")
        self._model_instances = model_instances
        self._max_score = max_score

    async def score(
        self, candidates: List[ModelInstanceScheduleCandidate]
    ) -> List[ModelInstanceScheduleCandidate]:
        if (
            not candidates
            or self._max_score <= 0
            or not self._group_id
            or not self._opposite
        ):
            return candidates

        counts = self._opposite_counts()
        if not counts:
            # No member of the opposite role is placed yet — a group whose
            # decodes have not been scheduled, or a scale-out that happens to
            # be first. Nothing to be near, so every candidate scores zero and
            # the resource scorers decide alone.
            return candidates

        for candidate in candidates:
            # Multiplied, not normalised into the 0..max_score band. The
            # band would make the gap between two and three decodes
            # `max_score / max_count`, which shrinks as a group grows until
            # `PlacementScorer`'s 100-point spread outranks it — and then the
            # rule silently becomes "roomiest worker", which is what this
            # exists to override. Multiplying keeps every adjacent step worth
            # a full `max_score`, so the ordering is affinity first and
            # capacity only among equals. The total is unbounded; nothing
            # downstream does anything with it but take the maximum.
            candidate.score = self._max_score * local_share(candidate, counts)

        return candidates

    def _opposite_counts(self) -> Dict[int, int]:
        return opposite_counts_by_worker(
            self._model_instances, self._group_id, self._opposite
        )


class PairingRetentionScorer(ModelInstanceScorer):
    """Pick the surplus prefill that costs the fewest local pairs to lose.

    **The inverse of `PairingAffinityScorer`, on the same ledger.** Scaling a
    role out prefers the worker holding the most of the opposite role; scaling
    the same role in must therefore delete the member on the worker holding
    the *fewest*, or the two halves of one policy spend their time undoing each
    other. Same derivation: with `p_j` / `d_j` members of each role on worker
    `j`, a router picking the two ends independently gives::

        P(a P/D pair is local) = (1/xy) * sum_j p_j * d_j

    Removing a prefill from worker `j` drops the numerator by `d_j`, and the
    denominator `(x-1)*y` is the same whichever one goes. So the cheapest
    victim is the one with the fewest decodes beside it.

    **Without this the scale-down chain does the opposite of the scale-up
    chain.** `PlacementScorer` on a group falls to its SPREAD branch — no PD
    deployment ever chooses `placement_strategy`, it is the column default —
    and SPREAD counts a worker's instances by `model_id`, which lumps prefill,
    decode and router into one distribution. The prefill on a host that also
    holds a decode therefore scores *below* a prefill sitting alone, and the
    pair that scheduling just built is the first thing scaling in takes apart.

    **Bounded, and deliberately small — the one place this differs from
    scale-up.** `PairingAffinityScorer` multiplies, paying a full `max_score`
    per sibling with no ceiling, because every candidate it ranks is a healthy
    worker that already fits and affinity is the only axis that means
    anything. Here the candidates are running members and one of them may be
    broken, in which case deleting it is free and no amount of affinity should
    buy it a reprieve. So this normalises into its own band, and that band has
    to stay under `StatusScorer`'s 50-point step (0 / 50 / 100) or a starting
    member with company would tie a healthy member sitting alone.

    Inert — all zeros — for a role with no opposite (the router, which also
    never reaches a scale-down: the API pins it to one replica), for a group
    whose opposite role is not placed anywhere yet, and for any model without
    roles. A role-less deployment's scale-down scoring is unchanged.
    """

    def __init__(
        self,
        group_id: Optional[str],
        role: Optional[str],
        peers: Sequence[ModelInstance],
        max_score: float = 20.0,
    ):
        self._group_id = group_id
        self._opposite = _OPPOSITE.get(role or "")
        # The whole generation, not just the role being scaled in: `d_j`
        # counts the *other* role, and the caller's candidate list is
        # single-role by construction (`find_scale_down_candidates` rejects
        # anything else).
        self._peers = peers
        self._max_score = max_score

    async def score_instances(
        self, instances: List[ModelInstance]
    ) -> List[ModelInstanceScore]:
        counts = opposite_counts_by_worker(
            self._peers, self._group_id, self._opposite, include_draining=False
        )
        peak = max(counts.values(), default=0)
        if self._max_score <= 0 or peak == 0:
            return [
                ModelInstanceScore(model_instance=instance, score=0)
                for instance in instances
            ]

        scored = []
        for instance in instances:
            score = self._max_score * local_share(instance, counts) / peak
            scored.append(ModelInstanceScore(model_instance=instance, score=score))
        return scored
