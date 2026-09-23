"""The placement policy end to end: which domain a group lands in, and why.

The unit files beside this one pin the pieces -- `test_group_placement_scorer.py`
the two ratios, `test_group_capacity_gpu_size.py` the one-card-size rule,
`test_group_solver.py` the layer walk and the sharing. This one drives the
whole policy over a real topology tree and asserts the *decisions*, because
every one of them is a place where a plausible implementation gets a defensible
but wrong answer:

* a looser layer outscoring a tighter one and winning;
* the smallest fitting domain winning because it was found first;
* file locality outweighing pair locality, or being unable to settle anything;
* the one-card-size rule, applied while sizing the tree, deleting a domain
  that was homogeneous on its own;
* an optimisation that skips domains taking the refusal's numbers with it.

The capacity stub is deliberately explicit about slots per worker: a group
scheduler's interesting behaviour is all in *which* worker gets a member, and
a stub that says "everyone has room" cannot show any of it.
"""

from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence

import pytest

from gpustack.policies.scorers.group_placement_scorer import group_scorer, pair_locality
from gpustack.scheduler.group_solver import (
    GatherRequest,
    GroupInfeasible,
    GroupPlacement,
    RoleDemand,
    solve_group_placement,
)
from gpustack.topology.tree import (
    NODE_LAYER,
    TopologyLayerSpec,
    build_topology,
    layer_names,
)

ZONE = "topology.kubernetes.io/zone"
RACK = "topology.gpustack.ai/rack"


def host(worker_id: int, zone: Optional[str] = None, rack: Optional[str] = None):
    labels = {}
    if zone:
        labels[ZONE] = zone
    if rack:
        labels[RACK] = rack
    return SimpleNamespace(id=worker_id, name=f"w{worker_id}", labels=labels)


def rack_only():
    return [TopologyLayerSpec(layer="RackLayer", label_keys=[RACK])]


def zone_and_rack():
    return [
        TopologyLayerSpec(layer="ZoneLayer", label_keys=[ZONE]),
        TopologyLayerSpec(
            layer="RackLayer", label_keys=[RACK], parent_layer="ZoneLayer"
        ),
    ]


def tree(workers, specs=None):
    specs = rack_only() if specs is None else specs
    return build_topology(specs, workers), layer_names(specs)


class Capacity:
    """Slots per worker, with the two optional behaviours the solver looks for.

    ``bands`` stands in for the one-card-size rule: workers sharing a band id
    share a card size, and a call offers only the band holding the most slots
    among the workers it was asked about. ``sizing`` deliberately does not
    apply it -- that asymmetry is the behaviour several tests below exist for.

    ``spans`` stands in for a member too wide for one machine, reported the
    way `GroupCapacity` reports it.
    """

    def __init__(
        self,
        slots: Dict[int, int],
        bands: Optional[Dict[int, int]] = None,
        spans: Optional[Dict[int, List[int]]] = None,
        unmeasured: Sequence[int] = (),
    ):
        self._slots = slots
        self._bands = bands or {}
        self._spans = spans or {}
        self._unmeasured = set(unmeasured)
        self.asked: List[Sequence[int]] = []

    def _raw(self, worker_ids, placed):
        used: Dict[int, int] = {}
        for entry in placed:
            used[entry.worker_id] = used.get(entry.worker_id, 0) + 1
        return {
            w: max(0, self._slots.get(w, 0) - used.get(w, 0))
            for w in worker_ids
            if w not in self._unmeasured
        }

    async def __call__(self, _role, worker_ids, placed):
        self.asked.append(tuple(worker_ids))
        out = self._raw(worker_ids, placed)
        if not self._bands:
            return out
        totals: Dict[int, int] = {}
        for worker_id, slots in out.items():
            band = self._bands.get(worker_id)
            totals[band] = totals.get(band, 0) + slots
        if not totals:
            return out
        winner = max(totals, key=lambda band: (totals[band], band))
        return {
            worker_id: (slots if self._bands.get(worker_id) == winner else 0)
            for worker_id, slots in out.items()
        }

    async def sizing(self, _role, worker_ids):
        return self._raw(worker_ids, ())

    def spans_for(self, _role, worker_id):
        return list(self._spans.get(worker_id, [worker_id]))


def pd(prefill=1, decode=1):
    return [
        RoleDemand(role="prefill", replicas=prefill, weight=2.0),
        RoleDemand(role="decode", replicas=decode, weight=1.0),
    ]


def scored(**by_worker):
    """A score reading one number off whichever worker the group used."""

    table = {int(k[1:]): v for k, v in by_worker.items()}

    def score(placement: GroupPlacement) -> float:
        return max(table.get(w, 0.0) for w in placement.worker_ids())

    return score


# --- 1. the layer is a strict key ------------------------------------------ #


@pytest.mark.asyncio
async def test_a_tighter_layer_wins_however_the_looser_one_scores():
    """The exchange rate nobody can set. A rack that would score ten times
    higher must not beat a host that holds the group."""
    root, layers = tree([host(1, rack="a"), host(2, rack="a")])

    got = await solve_group_placement(
        root,
        pd(),
        Capacity({1: 2, 2: 2}),
        layers,
        score=lambda p: 10.0 if p.layer == "RackLayer" else 1.0,
    )

    assert got.layer == NODE_LAYER


@pytest.mark.asyncio
async def test_the_walk_widens_one_rung_at_a_time():
    """Host, then rack, then zone -- and it stops at the first rung that
    holds the group rather than at the first that is large enough to be
    obviously safe."""
    workers = [
        host(1, zone="z1", rack="a"),
        host(2, zone="z1", rack="b"),
        host(3, zone="z2", rack="c"),
    ]
    root, layers = tree(workers, zone_and_rack())

    one_host = await solve_group_placement(root, pd(), Capacity({1: 2}), layers)
    one_zone = await solve_group_placement(
        root, pd(), Capacity({1: 1, 2: 1}), layers, score=lambda p: 0.0
    )

    assert one_host.layer == NODE_LAYER
    assert one_zone.layer == "ZoneLayer"
    assert one_zone.path == ["z1"]


@pytest.mark.asyncio
async def test_must_gather_ranks_inside_its_floor_and_refuses_above_it():
    """Both halves at once: the floor does not switch ranking off, and
    ranking does not let the walk step over the floor."""
    workers = [
        host(1, zone="z1", rack="a"),
        host(2, zone="z1", rack="b"),
    ]
    root, layers = tree(workers, zone_and_rack())

    ranked = await solve_group_placement(
        root,
        pd(),
        Capacity({1: 2, 2: 2}),
        layers,
        GatherRequest(layer="RackLayer", must=True),
        score=scored(w2=1.0, w1=0.0),
    )
    assert isinstance(ranked, GroupPlacement)
    assert set(ranked.worker_ids()) == {2}

    refused = await solve_group_placement(
        root,
        pd(2, 2),
        Capacity({1: 2, 2: 2}),
        layers,
        GatherRequest(layer="RackLayer", must=True),
        score=lambda p: 100.0,
    )
    assert isinstance(refused, GroupInfeasible)


@pytest.mark.asyncio
async def test_the_cluster_root_is_a_fallback_and_is_not_ranked():
    """Everything is under the root, so reaching it says only "somewhere in
    this cluster". There is nothing to compare it against."""
    root, layers = tree([host(1, rack="a"), host(2, rack="b")])
    calls = []

    got = await solve_group_placement(
        root,
        pd(1, 1),
        Capacity({1: 1, 2: 1}),
        layers,
        score=lambda p: calls.append(p.layer) or 0.0,
    )

    assert isinstance(got, GroupPlacement)
    assert sorted(got.worker_ids()) == [1, 2]
    # Neither rack held both members, so the root placed them and no ranking
    # was asked for.
    assert calls == []


# --- 2. enumerating one layer ---------------------------------------------- #


@pytest.mark.asyncio
async def test_every_fitting_domain_of_the_layer_reaches_the_score():
    root, layers = tree([host(i, rack=chr(ord("a") + i)) for i in range(3)])
    seen = []

    await solve_group_placement(
        root,
        pd(1, 1),
        Capacity({0: 2, 1: 2, 2: 2}),
        layers,
        score=lambda p: seen.append(p.path) or 0.0,
    )

    assert len(seen) == 3


@pytest.mark.asyncio
async def test_the_best_scoring_domain_wins_not_the_smallest():
    """Without ranking the smallest fitting domain wins by arriving first,
    which is the right answer only when nothing else is known."""
    root, layers = tree([host(1, rack="a"), host(2, rack="b")])

    got = await solve_group_placement(
        root,
        pd(),
        # Worker 2's domain is larger, so it is examined second.
        Capacity({1: 2, 2: 8}),
        layers,
        score=scored(w2=1.0, w1=0.0),
    )

    assert set(got.worker_ids()) == {2}


@pytest.mark.asyncio
async def test_the_candidate_limit_keeps_the_smallest_domains():
    """Domains are examined smallest-first, so a cap keeps the tightest ones
    and drops the tail -- the opposite would spend the budget on the domains
    the policy least wants."""
    root, layers = tree([host(i, rack=f"r{i}") for i in range(1, 6)])
    seen = []

    await solve_group_placement(
        root,
        pd(),
        Capacity({1: 2, 2: 4, 3: 6, 4: 8, 5: 10}),
        layers,
        score=lambda p: seen.append(p.worker_ids()[0]) or 0.0,
        limit=2,
    )

    assert seen == [1, 2]


@pytest.mark.asyncio
async def test_a_single_candidate_is_not_scored():
    """Nothing to compare it with. Calling the score anyway would be harmless
    and is still worth not doing: it is the only signal that ranking happened
    at all."""
    root, layers = tree([host(1, rack="a")])
    calls = []

    got = await solve_group_placement(
        root,
        pd(),
        Capacity({1: 2}),
        layers,
        score=lambda p: calls.append(p) or 0.0,
    )

    assert isinstance(got, GroupPlacement)
    assert calls == []


# --- 3. what the score actually prefers ------------------------------------ #


def two_racks():
    """Two racks of equal room, 2P2D, and no single host big enough.

    The leaf layer therefore cannot answer and the rack layer genuinely
    decides -- which is the only way a test says anything about ranking
    domains. Rack `a`'s two roomy hosts end up with one prefill and one decode
    each; rack `b`'s four one-slot hosts take one member apiece, so no pair of
    its members shares a host.
    """
    workers = [
        host(1, rack="a"),
        host(2, rack="a"),
        host(3, rack="b"),
        host(4, rack="b"),
        host(5, rack="b"),
        host(6, rack="b"),
    ]
    root, layers = tree(workers)
    return root, layers, Capacity({1: 2, 2: 2, 3: 1, 4: 1, 5: 1, 6: 1})


@pytest.mark.asyncio
async def test_pair_locality_picks_the_rack_that_keeps_the_ends_together():
    """The reason disaggregation has a placement policy at all."""
    root, layers, capacity = two_racks()

    got = await solve_group_placement(
        root, pd(2, 2), capacity, layers, score=group_scorer()
    )

    assert got.layer == "RackLayer"
    assert got.path == ["a"]
    # Each of rack `a`'s hosts carries one prefill and one decode, so half of
    # the four possible pairings are local. Rack `b` manages none.
    assert pair_locality(got) == 0.5


@pytest.mark.asyncio
async def test_file_locality_settles_a_pairing_tie():
    """Two racks that pair equally well. The tie-break is the download the
    group does not have to do."""
    workers = [host(1, rack="a"), host(2, rack="b")]
    root, layers = tree(workers)

    got = await solve_group_placement(
        root,
        pd(1, 1),
        Capacity({1: 2, 2: 2}),
        layers,
        score=group_scorer(ready_worker_ids={2}),
    )

    assert set(got.worker_ids()) == {2}


@pytest.mark.asyncio
async def test_file_locality_cannot_outrank_pairing_at_the_default_weights():
    """Co-locating a request's two ends is what the feature is for; landing on
    a warm worker saves a one-off download. The defaults have to say so, and a
    reversed pair of weights is the mutation this catches.

    Rack `a` pairs at 0.5 and is cold (score 0.5); rack `b` pairs at 0 and is
    entirely warm (score 0.3).
    """
    root, layers, capacity = two_racks()

    got = await solve_group_placement(
        root,
        pd(2, 2),
        capacity,
        layers,
        score=group_scorer(ready_worker_ids={3, 4, 5, 6}),
    )

    assert got.path == ["a"]


@pytest.mark.asyncio
async def test_the_weights_are_weights_and_not_a_fixed_priority():
    """The same fixture, with file locality weighted past pairing. A hard-coded
    ordering would ignore the knob and keep rack `a`."""
    root, layers, capacity = two_racks()

    got = await solve_group_placement(
        root,
        pd(2, 2),
        capacity,
        layers,
        score=group_scorer(
            ready_worker_ids={3, 4, 5, 6}, pair_weight=1.0, file_weight=2.0
        ),
    )

    assert got.path == ["b"]


@pytest.mark.asyncio
async def test_an_unbalanced_ratio_is_scored_against_its_own_size():
    """3P1D. The denominator is x*y, so "as local as 3P1D can be" scores the
    same as "as local as 1P1D can be" -- otherwise the ratio would quietly
    rank groups by shape."""
    root, layers = tree([host(1, rack="a")])

    got = await solve_group_placement(
        root, pd(3, 1), Capacity({1: 4}), layers, score=group_scorer()
    )

    assert pair_locality(got) == 1.0


@pytest.mark.asyncio
async def test_a_group_with_one_paired_role_scores_zero_rather_than_dividing():
    root, layers = tree([host(1, rack="a"), host(2, rack="b")])

    got = await solve_group_placement(
        root,
        [RoleDemand(role="prefill", replicas=2, weight=1.0)],
        Capacity({1: 2, 2: 2}),
        layers,
        score=group_scorer(),
    )

    assert isinstance(got, GroupPlacement)
    assert pair_locality(got) == 0.0


# --- 4. a member too wide for one machine ---------------------------------- #


@pytest.mark.asyncio
async def test_a_split_member_is_dropped_before_anything_is_scored():
    """An all-reduce per layer per token against a KV transfer per request:
    not a trade a bounded score can be trusted to make, so the candidate
    leaves the running instead of losing points."""
    root, layers = tree([host(1, rack="a"), host(2, rack="b")])

    got = await solve_group_placement(
        root,
        pd(),
        Capacity({1: 2, 2: 2}, spans={1: [1, 3]}),
        layers,
        score=scored(w1=100.0, w2=1.0),
    )

    assert set(got.worker_ids()) == {2}


@pytest.mark.asyncio
async def test_dropping_split_members_never_empties_the_field():
    """A role wider than any single machine has to place somewhere."""
    root, layers = tree([host(1, rack="a"), host(2, rack="b")])

    got = await solve_group_placement(
        root,
        pd(),
        Capacity({1: 2, 2: 2}, spans={1: [1, 3], 2: [2, 4]}),
        layers,
        score=scored(w2=1.0, w1=0.0),
    )

    assert isinstance(got, GroupPlacement)


@pytest.mark.asyncio
async def test_a_capacity_function_that_cannot_be_asked_reports_no_splits():
    """Every stub, and every role that fits on one machine -- which is every
    role placed today. The narrowing has to degrade to a no-op, not to
    dropping everything."""

    async def plain(_role, worker_ids, placed):
        used = {}
        for entry in placed:
            used[entry.worker_id] = used.get(entry.worker_id, 0) + 1
        return {w: max(0, 2 - used.get(w, 0)) for w in worker_ids}

    root, layers = tree([host(1, rack="a"), host(2, rack="b")])

    got = await solve_group_placement(
        root, pd(), plain, layers, score=scored(w2=1.0, w1=0.0)
    )

    assert set(got.worker_ids()) == {2}


# --- 5. sharing a domain out, and the caller's preference ------------------ #


@pytest.mark.asyncio
async def test_preference_decides_only_what_room_and_balance_left_tied():
    root, layers = tree([host(1, rack="a"), host(2, rack="a"), host(3, rack="a")])

    got = await solve_group_placement(
        root,
        [RoleDemand(role="prefill", replicas=2, weight=1.0)],
        # The rack is the only domain that fits two members one-per-host.
        Capacity({1: 1, 2: 1, 3: 1}),
        layers,
        preference={3: 0, 1: 1, 2: 1},
        score=lambda p: 0.0,
    )

    assert 3 in got.worker_ids()


@pytest.mark.asyncio
async def test_preference_cannot_move_a_member_onto_a_fuller_worker():
    root, layers = tree([host(1, rack="a"), host(2, rack="a")])

    got = await solve_group_placement(
        root,
        pd(2, 2),
        Capacity({1: 1, 2: 3}),
        layers,
        preference={1: 0, 2: 1},
        score=lambda p: 0.0,
    )

    assert got.worker_ids().count(2) == 3
    assert got.worker_ids().count(1) == 1


@pytest.mark.asyncio
async def test_preference_cannot_undo_the_round_robin_that_mixes_roles():
    """Two hosts, one prefill and one decode. Dealing them both onto the
    preferred host would leave the other empty and the pair local -- but it is
    the *balance* key, ahead of preference, that decides, and it deals them
    apart. The point is the ordering, not the outcome being prettier."""
    root, layers = tree([host(1, rack="a"), host(2, rack="a")])

    got = await solve_group_placement(
        root,
        pd(1, 1),
        Capacity({1: 1, 2: 1}),
        layers,
        preference={1: 0, 2: 1},
        score=lambda p: 0.0,
    )

    assert sorted(got.worker_ids()) == [1, 2]


@pytest.mark.asyncio
async def test_no_preference_and_no_score_is_the_unpolicied_answer():
    """The shape a caller with nothing to express gets: the smallest fitting
    domain at the tightest layer, and no domain examined past it."""
    root, layers = tree([host(i, rack=f"r{i}") for i in (1, 2, 3)])
    capacity = Capacity({1: 2, 2: 2, 3: 2})

    got = await solve_group_placement(root, pd(), capacity, layers)

    assert set(got.worker_ids()) == {1}
    # One host asked about, per role, and then the walk stopped.
    assert [c for c in capacity.asked if len(c) == 1] == [(1,), (1,)]


# --- 6. one card size, and the sweep that sizes the tree ------------------- #


@pytest.mark.asyncio
async def test_a_rack_of_the_losing_card_size_is_still_examined():
    """The interaction this pair of features gets wrong by default. The rule
    picks the band holding the most members; picked over the whole tree, the
    48 GiB band loses to the three 32 GiB hosts and its rack sizes as empty.
    Sizing therefore does not apply the rule, and the rack is examined -- on
    its own it is perfectly homogeneous."""
    workers = [
        host(1, rack="big"),
        host(2, rack="big"),
        host(3, rack="small"),
        host(4, rack="small"),
        host(5, rack="small"),
    ]
    root, layers = tree(workers)
    bands = {1: 48, 2: 48, 3: 32, 4: 32, 5: 32}
    seen = []

    await solve_group_placement(
        root,
        pd(1, 1),
        Capacity({w: 1 for w in range(1, 6)}, bands=bands),
        layers,
        score=lambda p: seen.append(p.path) or 0.0,
    )

    assert ["big"] in seen
    assert ["small"] in seen


@pytest.mark.asyncio
async def test_inside_a_mixed_domain_only_one_card_size_is_offered():
    """The rule does its job where it is decided. One rack, two card sizes,
    and a role of two: the band with two hosts serves it and the odd host is
    not used."""
    workers = [host(1, rack="a"), host(2, rack="a"), host(3, rack="a")]
    root, layers = tree(workers)

    got = await solve_group_placement(
        root,
        [RoleDemand(role="prefill", replicas=2, weight=1.0)],
        Capacity({1: 1, 2: 1, 3: 1}, bands={1: 48, 2: 32, 3: 32}),
        layers,
        score=lambda p: 0.0,
    )

    assert sorted(got.worker_ids()) == [2, 3]


# --- 7. the refusal's numbers survive the optimisations -------------------- #


@pytest.mark.asyncio
async def test_a_layer_of_provably_empty_domains_is_still_examined():
    """Skipping empty domains is an optimisation over the *examinable* ones.
    Skipping them all leaves nothing to build the refusal from, and the reader
    gets "no domain has any capacity" where they had a shortfall.

    Under a floor, because that is where it bites: without one the walk ends
    at the cluster root, whose own attempt populates the numbers whatever the
    layers above it skipped. `MustGather` removes that fallback, so a fleet
    that is genuinely full has nothing else to report from.
    """
    root, layers = tree([host(1, rack="a"), host(2, rack="b")])

    got = await solve_group_placement(
        root,
        pd(1, 1),
        Capacity({1: 0, 2: 0}),
        layers,
        GatherRequest(layer="RackLayer", must=True),
        score=lambda p: 0.0,
    )

    assert isinstance(got, GroupInfeasible)
    assert got.needed == 2
    assert got.available == 0
    assert "holds 0" in got.reason


@pytest.mark.asyncio
async def test_a_full_cluster_without_a_floor_still_reports_its_shortfall():
    """The same fleet, no floor: the root fallback runs and reports. Asserted
    beside the case above so the pair shows which path carries the numbers."""
    root, layers = tree([host(1, rack="a"), host(2, rack="b")])

    got = await solve_group_placement(
        root, pd(1, 1), Capacity({1: 0, 2: 0}), layers, score=lambda p: 0.0
    )

    assert isinstance(got, GroupInfeasible)
    assert got.available == 0
    assert "room for 0" in got.reason


@pytest.mark.asyncio
async def test_an_unmeasured_worker_is_not_a_proof_of_emptiness():
    """Absent from the sizing sweep means unmeasured, and a domain holding one
    is examined -- because only that examination can tell "we could not look"
    from "the cluster is full", and the second answer is the one that stops an
    operator looking for a mistake."""
    root, layers = tree([host(1, rack="a"), host(2, rack="b")])

    got = await solve_group_placement(
        root,
        pd(1, 1),
        Capacity({1: 0, 2: 0}, unmeasured=[2]),
        layers,
        score=lambda p: 0.0,
    )

    assert isinstance(got, GroupInfeasible)
    assert got.unmeasured == 1
    assert "could not be measured" in got.reason


@pytest.mark.asyncio
async def test_two_racks_of_one_name_under_two_zones_stay_apart():
    """Domain names are unique among siblings and nowhere wider, which is why
    a placement carries a path. Both zones name a rack `r1`; the answer has to
    say which."""
    workers = [
        host(1, zone="z1", rack="r1"),
        host(2, zone="z2", rack="r1"),
    ]
    root, layers = tree(workers, zone_and_rack())

    got = await solve_group_placement(
        root,
        pd(),
        Capacity({1: 2, 2: 2}),
        layers,
        score=scored(w2=1.0, w1=0.0),
    )

    # The leaf is the host itself, so the address runs all the way down --
    # which is the point: `r1` alone names two different racks.
    assert got.path == ["z2", "r1", "w2"]
    assert got.domain == "z2 / r1 / w2"


@pytest.mark.asyncio
async def test_the_winner_comes_back_carrying_the_score_that_chose_it():
    """A placement whose deciding number is thrown away cannot tell anyone why
    it beat the alternatives -- and the per-instance candidate has carried its
    own score since long before groups existed."""
    root, layers, capacity = two_racks()

    got = await solve_group_placement(
        root, pd(2, 2), capacity, layers, score=group_scorer(ready_worker_ids={1})
    )

    assert got.score is not None
    # Rack `a`'s two hosts take one prefill and one decode each, so half the
    # pairings are local and half the members sit on the one warm worker.
    assert got.score.terms == {"pair": 0.5, "file": 0.5}
    assert got.score.total == pytest.approx(0.5 + 0.3 * 0.5)
    assert got.score.describe() == "pair 0.50, file 0.50"


@pytest.mark.asyncio
async def test_an_unranked_placement_says_so_instead_of_scoring_zero():
    """One fitting domain, or no policy to express. `None` is a different fact
    from 0.0 and the log line says so rather than reporting a tie nobody
    computed."""
    root, layers = tree([host(1, rack="a")])

    got = await solve_group_placement(
        root, pd(), Capacity({1: 2}), layers, score=group_scorer()
    )

    assert got.score is None
