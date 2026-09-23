"""«As close as possible», meaning more than «the same host».

The two scorers that pull a later member toward its group — pairing affinity
for a scaled-out prefill or decode, group locality for the router — both
compare `worker_id` and nothing else. Between a worker in the members' own rack
and one three racks away they were indifferent, so the deploy form's top option
stopped at the host, and every tier below it was a target to be reported
against rather than one to aim at.
"""

from types import SimpleNamespace

import pytest

from gpustack.policies.scorers.topology_proximity_scorer import (
    TopologyProximityScorer,
)
from gpustack.topology.view import build_view
from tests.utils.topology_layers import layer_obj

RACK = "topology.gpustack.ai/rack"
ZONE = "topology.gpustack.ai/zone"


def _worker(id_: int, rack=None, zone=None):
    labels = {}
    if rack:
        labels[RACK] = rack
    if zone:
        labels[ZONE] = zone
    return SimpleNamespace(id=id_, name=f"w{id_}", labels=labels, cluster_id=1)


def _instance(id_, role, worker_id=None, group_id="g1"):
    return SimpleNamespace(id=id_, role=role, worker_id=worker_id, group_id=group_id)


def _candidate(worker):
    return SimpleNamespace(worker=worker, score=None)


def _view(workers, layers=None):
    return build_view(
        SimpleNamespace(
            layers=layers or [layer_obj("zone", [ZONE]), layer_obj("rack", [RACK])]
        ),
        workers,
    )


async def _score(workers, placed, candidates=None, anchors=("prefill", "decode")):
    scorer = TopologyProximityScorer(
        "g1", placed, _view(workers), anchors, max_score=150.0
    )
    scored = await scorer.score([_candidate(w) for w in (candidates or workers)])
    return {c.worker.id: (c.score or 0) for c in scored}


@pytest.mark.asyncio
async def test_a_worker_in_the_members_rack_beats_one_a_zone_away():
    """Neither candidate is the host a member sits on, so a `worker_id`
    comparison calls them equal and only topology can separate them."""
    workers = [
        _worker(1, "rack-a", "zone-1"),
        _worker(2, "rack-a", "zone-1"),
        _worker(3, "rack-b", "zone-2"),
    ]
    placed = [_instance(1, "prefill", worker_id=1)]

    scores = await _score(workers, placed, candidates=workers[1:])

    assert scores[2] > scores[3]


@pytest.mark.asyncio
async def test_the_same_zone_still_beats_another_zone():
    """Every declared rung counts, not just the tightest one — a group split
    across two racks of one zone is closer than one split across two zones."""
    workers = [
        _worker(1, "rack-a", "zone-1"),
        _worker(2, "rack-b", "zone-1"),
        _worker(3, "rack-c", "zone-2"),
    ]
    placed = [_instance(1, "prefill", worker_id=1)]

    scores = await _score(workers, placed, candidates=workers[1:])

    assert scores[2] > scores[3]


@pytest.mark.asyncio
async def test_the_members_own_host_scores_highest():
    workers = [_worker(1, "rack-a", "zone-1"), _worker(2, "rack-a", "zone-1")]
    placed = [_instance(1, "prefill", worker_id=1)]

    scores = await _score(workers, placed)

    assert scores[1] > scores[2]


@pytest.mark.asyncio
async def test_unclassified_workers_are_not_treated_as_close():
    """The bucket means "we do not know where these are". Reading that as
    "these are together" turns a missing label into a confident wrong answer,
    and `common_layer` refuses to do it — this must not undo that refusal."""
    workers = [_worker(1), _worker(2)]
    placed = [_instance(1, "prefill", worker_id=1)]

    scores = await _score(workers, placed)

    assert scores[2] == 0


@pytest.mark.asyncio
async def test_nothing_placed_yet_scores_nothing():
    """A group whose first member is being placed has nothing to be near, so
    this scorer stays out of it and the resource scorers decide alone."""
    workers = [_worker(1, "rack-a"), _worker(2, "rack-b")]

    scores = await _score(workers, [])

    assert set(scores.values()) == {0}


@pytest.mark.asyncio
async def test_the_router_does_not_anchor_the_group():
    """It holds no weights, so where it sits is not where the group is. A
    stray router would otherwise pull every later member after it."""
    workers = [_worker(1, "rack-a"), _worker(2, "rack-b")]
    placed = [_instance(9, "router", worker_id=2)]

    scores = await _score(workers, placed)

    assert set(scores.values()) == {0}


@pytest.mark.asyncio
async def test_the_router_is_still_pulled_toward_the_members():
    """Not an anchor, but scored against them like anyone: it forwards every
    token the group serves."""
    workers = [_worker(1, "rack-a"), _worker(2, "rack-b")]
    placed = [_instance(1, "prefill", worker_id=1)]

    scores = await _score(workers, placed, anchors=("prefill", "decode"))

    assert scores[1] > scores[2]


@pytest.mark.asyncio
async def test_another_groups_members_are_not_anchors():
    workers = [_worker(1, "rack-a"), _worker(2, "rack-b")]
    placed = [_instance(1, "prefill", worker_id=2, group_id="other")]

    scores = await _score(workers, placed)

    assert set(scores.values()) == {0}


@pytest.mark.asyncio
async def test_a_cluster_with_no_declared_topology_still_prefers_the_host():
    """The built-in host rung exists whatever the operator declared, so «as
    close as possible» keeps the one meaning it always had."""
    workers = [_worker(1), _worker(2)]
    placed = [_instance(1, "prefill", worker_id=1)]
    scorer = TopologyProximityScorer(
        "g1", placed, _view(workers, layers=[]), ("prefill",), max_score=150.0
    )

    scored = await scorer.score([_candidate(w) for w in workers])
    scores = {c.worker.id: (c.score or 0) for c in scored}

    assert scores[1] > scores[2]


@pytest.mark.asyncio
async def test_a_zero_weight_turns_it_off():
    workers = [_worker(1, "rack-a"), _worker(2, "rack-b")]
    placed = [_instance(1, "prefill", worker_id=1)]
    scorer = TopologyProximityScorer(
        "g1", placed, _view(workers), ("prefill",), max_score=0
    )

    scored = await scorer.score([_candidate(w) for w in workers])

    assert all(c.score is None for c in scored)


# --- candidates that span machines ------------------------------------------- #


def _spanning_candidate(primary, others):
    return SimpleNamespace(
        worker=primary,
        score=None,
        subordinate_workers=[SimpleNamespace(worker_id=w.id) for w in others],
    )


async def _score_candidates(workers, placed, candidates, anchors=("prefill",)):
    scorer = TopologyProximityScorer(
        "g1", placed, _view(workers), anchors, max_score=150.0
    )
    scored = await scorer.score(candidates)
    return [c.score or 0 for c in scored]


@pytest.mark.asyncio
async def test_a_tight_candidate_beats_a_closer_but_looser_one():
    """The whole reason this is lexicographic. The two distances are not
    comparable quantities: a candidate's own machines carry the tensor-parallel
    all-reduce, once per layer per token, and the gap to the rest of the group
    carries the KV transfer, once per request. Summing them would let a large
    enough gain on the second buy a worse first."""
    workers = [
        _worker(1, "rack-a", "zone-1"),  # where the group already is
        _worker(2, "rack-a", "zone-1"),
        _worker(3, "rack-b", "zone-1"),
        _worker(4, "rack-b", "zone-1"),
    ]
    placed = [_instance(1, "prefill", worker_id=1)]

    # 甲: both its machines in one rack, but a rack away from the group.
    tight_but_far = _spanning_candidate(workers[2], [workers[3]])
    # 乙: one machine in the group's own rack, the other a rack away.
    loose_but_near = _spanning_candidate(workers[1], [workers[2]])

    tight, loose = await _score_candidates(
        workers, placed, [tight_but_far, loose_but_near]
    )

    assert tight > loose


@pytest.mark.asyncio
async def test_two_equally_tight_candidates_are_split_by_closeness():
    """The second term still decides, just only among candidates the first
    term calls equal — which is what makes it a tie-break rather than a
    weight."""
    workers = [
        _worker(1, "rack-a", "zone-1"),
        _worker(2, "rack-a", "zone-1"),
        _worker(3, "rack-b", "zone-1"),
        _worker(4, "rack-b", "zone-1"),
    ]
    placed = [_instance(9, "prefill", worker_id=1)]

    near = _spanning_candidate(workers[1], [workers[0]])
    far = _spanning_candidate(workers[2], [workers[3]])

    near_score, far_score = await _score_candidates(workers, placed, [near, far])

    assert near_score > far_score


@pytest.mark.asyncio
async def test_a_single_machine_candidate_is_never_penalised():
    """It has no pair to be apart, so the internal-spread penalty never
    applies and its score comes from closeness alone."""
    workers = [_worker(1, "rack-a"), _worker(2, "rack-a")]
    placed = [_instance(9, "prefill", worker_id=1)]

    scores = await _score_candidates(
        workers, placed, [_candidate(workers[1])], anchors=("prefill",)
    )

    assert scores[0] > 0


@pytest.mark.asyncio
async def test_a_spanning_candidate_is_measured_from_all_its_machines():
    """Scoring only the primary would call a candidate far from a group that
    its other half is sitting next to."""
    workers = [
        _worker(1, "rack-a", "zone-1"),
        _worker(2, "rack-b", "zone-1"),
        _worker(3, "rack-a", "zone-1"),
        _worker(4, "rack-c", "zone-2"),
    ]
    placed = [_instance(9, "prefill", worker_id=1)]

    # Both are equally loose inside — primary in rack-b, one machine elsewhere
    # — so the penalty cancels and only the closeness term can separate them.
    reaching = _spanning_candidate(workers[1], [workers[2]])  # other half: rack-a
    away = _spanning_candidate(workers[1], [workers[3]])  # other half: zone-2

    reaching_score, away_score = await _score_candidates(
        workers, placed, [reaching, away]
    )

    assert reaching_score > away_score


# --- internal spread is removed from the running, not scored down ------------ #
#
# The penalty above is lexicographic *within this scorer*, but the chain SUMS
# and `PairingAffinityScorer` is unbounded by design (`max_score` per opposite
# sibling), so enough siblings outweigh any penalty and buy a member split
# across machines. No finite constant dominates an unbounded one, which is why
# the first key is decided by removal rather than by arithmetic.


def _narrow(workers, placed, candidates, anchors=("prefill",), max_score=150.0):
    scorer = TopologyProximityScorer(
        "g1", placed, _view(workers), anchors, max_score=max_score
    )
    return scorer.narrow_to_tightest_internal_spread(candidates)


def test_a_candidate_that_would_split_the_member_is_dropped_outright():
    """The invariant has to hold by removal: as a penalty, any large enough
    bonus elsewhere on the chain could pay for it. A dropped candidate cannot
    be bought back."""
    workers = [
        _worker(1, "rack-a", "zone-1"),
        _worker(2, "rack-b", "zone-1"),
        _worker(3, "rack-c", "zone-2"),
    ]
    placed = [_instance(9, "prefill", worker_id=1)]

    whole = _candidate(workers[0])
    split = _spanning_candidate(workers[1], [workers[2]])

    assert _narrow(workers, placed, [split, whole]) == [whole]


def test_every_candidate_splitting_equally_is_a_no_op():
    """The shape that must NOT be narrowed: a role wider than any one machine
    has no un-split candidate to prefer. Dropping here would turn «prefer not
    to split» into «refuse to split» and fail a deployment that runs today."""
    workers = [
        _worker(1, "rack-a", "zone-1"),
        _worker(2, "rack-a", "zone-1"),
        _worker(3, "rack-b", "zone-1"),
        _worker(4, "rack-b", "zone-1"),
    ]
    placed = [_instance(9, "prefill", worker_id=1)]

    near = _spanning_candidate(workers[1], [workers[0]])
    far = _spanning_candidate(workers[2], [workers[3]])

    assert _narrow(workers, placed, [near, far]) == [near, far]


def test_the_tightest_bucket_is_kept_whole_not_reduced_to_one():
    """Narrowing answers the first key only. Everything that ties on it stays
    in, for the scorers to order — otherwise this would quietly become the
    placement decision."""
    workers = [
        _worker(1, "rack-a", "zone-1"),
        _worker(2, "rack-a", "zone-1"),
        _worker(3, "rack-b", "zone-1"),
        _worker(4, "rack-c", "zone-2"),
    ]
    placed = [_instance(9, "prefill", worker_id=1)]

    whole_near = _candidate(workers[1])
    whole_far = _candidate(workers[3])
    split = _spanning_candidate(workers[2], [workers[3]])

    kept = _narrow(workers, placed, [whole_near, split, whole_far])

    assert kept == [whole_near, whole_far]


def test_a_candidate_off_the_tree_is_kept_because_unknown_is_not_loose():
    """Its position is missing, not bad. Dropping it would be a judgement made
    from absent data — the same rule `derive_net_device` follows when it
    refuses to guess rather than inventing a value."""
    workers = [_worker(1, "rack-a", "zone-1"), _worker(2, "rack-a", "zone-1")]
    placed = [_instance(9, "prefill", worker_id=1)]

    whole = _candidate(workers[1])
    off_tree = _candidate(_worker(99, "rack-z", "zone-9"))  # not in the view

    kept = _narrow(workers, placed, [whole, off_tree])

    assert kept == [whole, off_tree]


def test_narrowing_is_off_wherever_the_scorer_is():
    """Same switches, same reasons: a cluster that declares no topology and a
    weight of 0 both mean «this notion does not apply here», and neither may
    start removing candidates."""
    workers = [
        _worker(1, "rack-a", "zone-1"),
        _worker(2, "rack-b", "zone-1"),
        _worker(3, "rack-c", "zone-2"),
    ]
    placed = [_instance(9, "prefill", worker_id=1)]
    candidates = [_spanning_candidate(workers[1], [workers[2]]), _candidate(workers[0])]

    assert _narrow(workers, placed, candidates, max_score=0) == candidates

    no_topology = TopologyProximityScorer(
        "g1", placed, None, ("prefill",), max_score=150.0
    )
    assert no_topology.narrow_to_tightest_internal_spread(candidates) == candidates

    no_group = TopologyProximityScorer(
        None, placed, _view(workers), ("prefill",), max_score=150.0
    )
    assert no_group.narrow_to_tightest_internal_spread(candidates) == candidates


@pytest.mark.asyncio
async def test_the_penalty_cannot_reorder_what_the_narrowing_left():
    """Why keeping the penalty is not duplication: after narrowing it is the
    same constant on every survivor, so it decides nothing — while still being
    what a caller that skips the narrowing gets."""
    workers = [
        _worker(1, "rack-a", "zone-1"),
        _worker(2, "rack-a", "zone-1"),
        _worker(3, "rack-b", "zone-1"),
        _worker(4, "rack-b", "zone-1"),
    ]
    placed = [_instance(9, "prefill", worker_id=1)]
    near = _spanning_candidate(workers[1], [workers[0]])
    far = _spanning_candidate(workers[2], [workers[3]])

    kept = _narrow(workers, placed, [near, far])
    scores = await _score_candidates(workers, placed, kept)

    # Both paid the same penalty, so the closeness term alone separates them.
    assert scores[0] > scores[1]
