"""How much one opposite-role sibling is worth, measured against its own chain.

`CandidateScoreChain` sums. `PairingAffinityScorer` pays a full `max_score`
per sibling rather than normalising, so "affinity first, capacity and topology
only among equals" is a statement about *arithmetic on the assembled chain*,
not about the scorer alone: it holds while one sibling outweighs everything the
rest of that chain can add together.

No constant can encode that: whatever bound is written down stops being one as
soon as another scorer joins the chain. These tests pin the derivation instead
of a number.
"""

from types import SimpleNamespace
from typing import List

import pytest

from gpustack import envs
from gpustack.policies.base import (
    ModelInstanceScheduleCandidate,
    ScheduleCandidatesScorer,
)
from gpustack.policies.scorers.model_file_locality_scorer import (
    ModelFileLocalityScorer,
)
from gpustack.policies.scorers.pairing_affinity_scorer import PairingAffinityScorer
from gpustack.policies.scorers.placement_scorer import PlacementScorer
from gpustack.policies.scorers.score_chain import CandidateScoreChain
from gpustack.policies.scorers.topology_proximity_scorer import (
    TopologyProximityScorer,
)
from gpustack.scheduler.scheduler import _group_scorer, _pairing_affinity_max_score
from gpustack.topology.view import build_view
from gpustack.schemas.models import ComputedResourceClaim, RoleNameEnum
from tests.utils.model import new_model
from tests.utils.topology_layers import layer_obj

GROUP = "g-1"
RACK = "topology.gpustack.ai/rack"
ZONE = "topology.gpustack.ai/zone"

# The 3P1D board these tests score against. Worker 1 carries the prefills;
# worker 2 is the empty machine in their rack; worker 3 is a rack away and
# carries the decodes a scaled-out prefill wants to pair with.
PREFILL_HOST = 1
EMPTY_NEIGHBOUR = 2
DECODE_HOST = 3


def _worker(id_: int, rack=None, zone=None):
    labels = {}
    if rack:
        labels[RACK] = rack
    if zone:
        labels[ZONE] = zone
    return SimpleNamespace(id=id_, name=f"w{id_}", labels=labels, cluster_id=1)


def _member(worker_id, role, group_id=GROUP):
    return SimpleNamespace(
        id=worker_id, worker_id=worker_id, role=role, group_id=group_id
    )


def _candidate(worker):
    return ModelInstanceScheduleCandidate(
        worker=worker,
        gpu_indexes=[0],
        computed_resource_claim=ComputedResourceClaim(ram=0, vram={}),
        score=None,
    )


class _CeilingScorer(ScheduleCandidatesScorer):
    """A scorer that declares a ceiling and pays all of it to one worker.

    Stands in for the resource scorers, whose real scores need a database and
    whose exact values are beside the point: what a test of the *ordering* has
    to fix is the worst case, which is every other scorer on the chain agreeing
    on the wrong worker and paying its maximum to say so.
    """

    def __init__(self, ceiling: float, winner_id: int):
        self._max_score = ceiling
        self._winner_id = winner_id

    async def score(self, candidates: List[ModelInstanceScheduleCandidate]):
        for candidate in candidates:
            candidate.score = (
                self._max_score if candidate.worker.id == self._winner_id else 0.0
            )
        return candidates


def _board():
    """The 3P1D group and the two workers it is being scaled out onto."""
    workers = {
        PREFILL_HOST: _worker(PREFILL_HOST, "rack-a", "zone-1"),
        EMPTY_NEIGHBOUR: _worker(EMPTY_NEIGHBOUR, "rack-a", "zone-1"),
        DECODE_HOST: _worker(DECODE_HOST, "rack-b", "zone-1"),
    }
    members = [
        _member(PREFILL_HOST, RoleNameEnum.PREFILL.value),
        _member(PREFILL_HOST, RoleNameEnum.PREFILL.value),
        _member(PREFILL_HOST, RoleNameEnum.PREFILL.value),
        _member(DECODE_HOST, RoleNameEnum.DECODE.value),
        _member(DECODE_HOST, RoleNameEnum.DECODE.value),
    ]
    return workers, members


def _proximity(workers, members):
    view = build_view(
        SimpleNamespace(layers=[layer_obj("zone", [ZONE]), layer_obj("rack", [RACK])]),
        list(workers.values()),
    )
    return TopologyProximityScorer(
        GROUP,
        members,
        view,
        (RoleNameEnum.PREFILL.value, RoleNameEnum.DECODE.value),
        max_score=envs.SCHEDULER_TOPOLOGY_PROXIMITY_MAX_SCORE,
    )


def _shipped_rest(workers, members, winner_id):
    """The scale-up chain minus pairing affinity, as `find_candidate` builds it.

    The two resource scorers are stood in for at their real ceilings; the
    topology scorer is the real one, because its ceiling is the one that is not
    simply `_max_score` and the point of the test is that the sum is read off
    the chain rather than written down.
    """
    model = new_model(1, "m", huggingface_repo_id="a/b")
    return [
        _CeilingScorer(PlacementScorer(model, []).score_ceiling, winner_id),
        _CeilingScorer(
            ModelFileLocalityScorer(
                model, max_score=envs.SCHEDULER_SCALE_UP_LOCALITY_MAX_SCORE
            ).score_ceiling,
            winner_id,
        ),
        _proximity(workers, members),
    ]


async def _chain_scores(scorers, workers, candidate_ids):
    candidates = [_candidate(workers[id_]) for id_ in candidate_ids]
    scored = await CandidateScoreChain(scorers).score(candidates)
    return {c.worker.id: c.score for c in scored}


@pytest.mark.asyncio
async def test_a_scaled_out_prefill_lands_on_the_decode_host_across_the_rack():
    """A scale-out scenario end to end on the assembled chain.

    Same 3P1D, two candidates: the empty machine in the prefills' rack, and the
    machine a rack away already holding two decodes. The scale-out belongs on
    the decodes.
    """
    workers, members = _board()
    rest = _shipped_rest(workers, members, winner_id=EMPTY_NEIGHBOUR)
    pairing = PairingAffinityScorer(
        GROUP,
        RoleNameEnum.PREFILL.value,
        members,
        max_score=_pairing_affinity_max_score(
            envs.SCHEDULER_PAIRING_AFFINITY_MAX_SCORE, rest
        ),
    )

    scores = await _chain_scores(
        rest + [pairing], workers, [EMPTY_NEIGHBOUR, DECODE_HOST]
    )

    assert scores[DECODE_HOST] > scores[EMPTY_NEIGHBOUR]


@pytest.mark.asyncio
async def test_one_more_sibling_outweighs_the_whole_rest_of_the_chain():
    """The guarantee itself, stated as the invariant rather than a scenario.

    Every scorer on the shipped chain pays its full ceiling to the empty
    neighbour -- 100 + 5 + 450 = 555 with the defaults -- and the decode host
    still has to win on siblings alone. A constant written here would have to
    exceed that sum, and would fail the next time a scorer joins.

    Deliberately harsher than the real `TopologyProximityScorer` can be on
    the board above, which is why it stands in here at its ceiling rather than
    scoring: a host that already holds a sibling is an anchor, so it shares the
    *host* rung with the group and collects proximity's maximum too, and the
    two scorers cannot actually disagree by proximity's full range on a
    single-machine candidate. That coincidence is not the invariant, does not
    survive a multi-machine candidate's internal-spread penalty, and would not
    survive the next scorer. What is being pinned is the bound.
    """
    workers, members = _board()
    rest = _shipped_rest(workers, members, winner_id=EMPTY_NEIGHBOUR)
    rest = [_CeilingScorer(scorer.score_ceiling, EMPTY_NEIGHBOUR) for scorer in rest]
    ceiling = sum(scorer.score_ceiling for scorer in rest)
    pairing = PairingAffinityScorer(
        GROUP,
        RoleNameEnum.PREFILL.value,
        members,
        max_score=_pairing_affinity_max_score(
            envs.SCHEDULER_PAIRING_AFFINITY_MAX_SCORE, rest
        ),
    )

    scores = await _chain_scores(
        rest + [pairing], workers, [EMPTY_NEIGHBOUR, DECODE_HOST]
    )

    assert scores[EMPTY_NEIGHBOUR] == pytest.approx(ceiling)
    assert scores[DECODE_HOST] > scores[EMPTY_NEIGHBOUR]
    # One step, not two: the guarantee has to hold for the smallest difference
    # in siblings, which is what a scale-out onto an almost-tied board sees.
    assert pairing._max_score > ceiling


@pytest.mark.asyncio
async def test_the_topology_scorers_ceiling_counts_its_rungs_not_its_step():
    """The one ceiling on the chain that is not simply `_max_score`.

    `TopologyProximityScorer` pays `max_score` *per rung* -- zone, rack, host
    on the built-in chain -- so its default 150 buys 450. Read off the scorer
    so a cluster that declares a deeper chain raises the sum by itself.
    """
    workers, members = _board()
    proximity = _proximity(workers, members)

    assert proximity.score_ceiling == pytest.approx(
        envs.SCHEDULER_TOPOLOGY_PROXIMITY_MAX_SCORE * 3
    )

    # And it is a real ceiling: the best any candidate actually scores is the
    # host that holds a member, which is the tightest rung there is.
    scores = await _chain_scores([proximity], workers, list(workers))
    assert max(scores.values()) == pytest.approx(proximity.score_ceiling)


def test_zero_switches_the_scorer_off_and_is_never_raised_to_the_floor():
    """The configured value is a floor for every value but this one.

    `GPUSTACK_SCHEDULER_PAIRING_AFFINITY_MAX_SCORE=0` is documented as "scale a
    group out by resource fit alone". A floor that applied here would turn the
    off switch into the loudest scorer on the chain -- the exact opposite of
    what the operator asked for.
    """
    workers, members = _board()
    rest = _shipped_rest(workers, members, winner_id=EMPTY_NEIGHBOUR)

    assert _pairing_affinity_max_score(0.0, rest) == 0.0
    assert _pairing_affinity_max_score(-1.0, rest) == -1.0


@pytest.mark.asyncio
async def test_a_group_scorer_built_with_the_off_switch_scores_nothing(monkeypatch):
    """The same switch through the production path, not the helper alone."""
    workers, members = _board()
    monkeypatch.setattr(envs, "SCHEDULER_PAIRING_AFFINITY_MAX_SCORE", 0.0)
    rest = _shipped_rest(workers, members, winner_id=EMPTY_NEIGHBOUR)

    scorer = _group_scorer(GROUP, RoleNameEnum.PREFILL.value, False, members, rest)

    candidates = [_candidate(workers[DECODE_HOST])]
    scored = await scorer.score(candidates)
    assert all(c.score is None for c in scored)


def test_a_chain_without_a_topology_is_not_inflated():
    """Summed over the scorers actually appended, not the ones that could be.

    A cluster that declares no topology gets no `TopologyProximityScorer`
    (`find_candidate` only builds one when the view exists), and a fleet with
    file locality turned off gets no `ModelFileLocalityScorer`. Sizing pairing
    against absent scorers would quietly raise the weight for everyone.
    """
    model = new_model(1, "m", huggingface_repo_id="a/b")
    placement_only = [_CeilingScorer(PlacementScorer(model, []).score_ceiling, 0)]
    with_locality = placement_only + [
        _CeilingScorer(
            ModelFileLocalityScorer(
                model, max_score=envs.SCHEDULER_SCALE_UP_LOCALITY_MAX_SCORE
            ).score_ceiling,
            0,
        )
    ]

    configured = envs.SCHEDULER_PAIRING_AFFINITY_MAX_SCORE
    # 100 + 1 and 105 + 1 are both under the configured 200, so the floor wins
    # and the weight is exactly what the operator set.
    assert _pairing_affinity_max_score(configured, placement_only) == configured
    assert _pairing_affinity_max_score(configured, with_locality) == configured


def test_the_next_scorer_on_the_chain_raises_the_weight_with_no_code_change():
    """What "derived" buys: the weight survives the chain growing.

    A new scorer is added to the chain and nothing in `PairingAffinityScorer`
    or in the env default is touched; the weight has to move on its own, and
    keep the strict inequality that "affinity first" means.
    """
    workers, members = _board()
    rest = _shipped_rest(workers, members, winner_id=EMPTY_NEIGHBOUR)
    before = _pairing_affinity_max_score(
        envs.SCHEDULER_PAIRING_AFFINITY_MAX_SCORE, rest
    )

    newcomer = _CeilingScorer(1000.0, EMPTY_NEIGHBOUR)
    after = _pairing_affinity_max_score(
        envs.SCHEDULER_PAIRING_AFFINITY_MAX_SCORE, rest + [newcomer]
    )

    assert after > before
    assert after > sum(scorer.score_ceiling for scorer in rest + [newcomer])


# --- the level above affinity: do not split the member ----------------------- #


def _spanning_candidate(primary, others):
    candidate = _candidate(primary)
    candidate.subordinate_workers = [
        SimpleNamespace(worker_id=worker.id) for worker in others
    ]
    return candidate


@pytest.mark.asyncio
async def test_affinity_may_not_buy_a_member_split_across_machines():
    """Why the split decision is a narrowing step and not part of the scoring.

    An affinity weight that outranks the rest of the chain also outranks
    `TopologyProximityScorer`'s split penalty, a fixed 600 on the built-in
    rungs: two opposite siblings pay 1112 and buy a member spread over two
    machines. The trade is indefensible at any weight — a split member pays a
    tensor-parallel all-reduce once per layer per token to save a KV transfer
    that runs once per request — and no constant can forbid it, because
    affinity is `max_score` PER sibling and so has no ceiling.

    So the narrowing runs first, and the scores below never get to vote.
    """
    workers, members = _board()
    # The decode host and its rack-mate can only hold the member between them;
    # the empty neighbour holds it whole but has no decode to pair with.
    workers[4] = _worker(4, "rack-b", "zone-1")
    proximity = _proximity(workers, members)

    whole_but_lonely = _candidate(workers[EMPTY_NEIGHBOUR])
    split_but_paired = _spanning_candidate(workers[DECODE_HOST], [workers[4]])

    kept = proximity.narrow_to_tightest_internal_spread(
        [split_but_paired, whole_but_lonely]
    )
    assert kept == [whole_but_lonely], (
        "the split candidate has to leave the running before scoring; while "
        "this was a penalty, two decodes on its machines outweighed it"
    )

    # And the arithmetic that made it necessary: had it stayed, it would have
    # won — so this is not a test of a case that could not arise.
    rest = _shipped_rest(workers, members, winner_id=EMPTY_NEIGHBOUR)
    pairing = PairingAffinityScorer(
        GROUP,
        RoleNameEnum.PREFILL.value,
        members,
        max_score=_pairing_affinity_max_score(
            envs.SCHEDULER_PAIRING_AFFINITY_MAX_SCORE, rest
        ),
    )
    scored = await CandidateScoreChain(rest + [pairing]).score(
        [
            _spanning_candidate(workers[DECODE_HOST], [workers[4]]),
            _candidate(workers[EMPTY_NEIGHBOUR]),
        ]
    )
    split_score, whole_score = (c.score for c in scored)
    assert split_score > whole_score
