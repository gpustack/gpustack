from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from gpustack import envs
from gpustack.policies.base import ModelInstanceScheduleCandidate
from gpustack.policies.scorers.pairing_affinity_scorer import (
    PairingAffinityScorer,
    PairingRetentionScorer,
)
from gpustack.schemas.models import ComputedResourceClaim, RoleNameEnum

GROUP = "g-1"


def _member(worker_id, role, group_id=GROUP):
    return SimpleNamespace(worker_id=worker_id, role=role, group_id=group_id)


def _candidate(worker_id):
    return ModelInstanceScheduleCandidate(
        worker=SimpleNamespace(id=worker_id),
        gpu_indexes=[],
        computed_resource_claim=ComputedResourceClaim(ram=0, vram={}),
        score=None,
    )


async def _scores(role, members, worker_ids, **kwargs):
    candidates = [_candidate(worker_id) for worker_id in worker_ids]
    scorer = PairingAffinityScorer(GROUP, role, members, **kwargs)
    scored = await scorer.score(candidates)
    return {c.worker.id: c.score for c in scored}


@pytest.mark.asyncio
async def test_prefill_prefers_the_worker_with_most_decodes():
    """The rule itself: scaling prefill out follows decode, not prefill."""
    members = [
        _member(1, RoleNameEnum.DECODE.value),
        _member(1, RoleNameEnum.DECODE.value),
        _member(2, RoleNameEnum.DECODE.value),
        _member(3, RoleNameEnum.PREFILL.value),
    ]

    scores = await _scores(RoleNameEnum.PREFILL.value, members, [1, 2, 3])

    assert scores[1] > scores[2] > scores[3]
    assert scores[3] == 0


@pytest.mark.asyncio
async def test_decode_prefers_the_worker_with_most_prefills():
    """Symmetric, and not the same worker as the test above picks."""
    members = [
        _member(1, RoleNameEnum.DECODE.value),
        _member(1, RoleNameEnum.DECODE.value),
        _member(2, RoleNameEnum.PREFILL.value),
    ]

    scores = await _scores(RoleNameEnum.DECODE.value, members, [1, 2])

    assert scores[2] > scores[1]


@pytest.mark.asyncio
async def test_group_member_count_is_not_the_rule():
    """The failure this scorer exists to prevent.

    Worker 1 holds three prefills and no decode; worker 2 holds one decode.
    "Most members of this group" would pick worker 1 and move
    `sum p_j * d_j` by exactly nothing. 3P1D is the shape that separates the
    two rules — under a balanced ratio they agree, which is why the wrong one
    survives review.
    """
    members = [
        _member(1, RoleNameEnum.PREFILL.value),
        _member(1, RoleNameEnum.PREFILL.value),
        _member(1, RoleNameEnum.PREFILL.value),
        _member(2, RoleNameEnum.DECODE.value),
    ]

    scores = await _scores(RoleNameEnum.PREFILL.value, members, [1, 2])

    assert scores[2] > scores[1]
    assert scores[1] == 0


@pytest.mark.asyncio
async def test_one_more_sibling_outweighs_every_other_scale_up_scorer():
    """Affinity is the primary order; capacity may only break ties.

    The chain sums the scorers, so this holds only while a single step of
    affinity is worth more than the whole range the resource scorers can
    move a candidate. Normalising the score into a fixed band would break it
    silently, and only for large groups.

    The env value is a floor, not the production weight: the scale-up chain
    sizes this against every scorer actually on it, which is more than the two
    summed here. `tests/scheduler/test_pairing_affinity_weight.py` owns that
    part -- this one only pins that the scorer alone clears the two resource
    scorers summed here.
    """
    import gpustack.envs as envs

    members = [
        _member(1, RoleNameEnum.DECODE.value),
        _member(2, RoleNameEnum.DECODE.value),
        _member(2, RoleNameEnum.DECODE.value),
    ]

    scores = await _scores(
        RoleNameEnum.PREFILL.value,
        members,
        [1, 2],
        max_score=envs.SCHEDULER_PAIRING_AFFINITY_MAX_SCORE,
    )

    other_scorers_ceiling = (
        envs.SCHEDULER_SCALE_UP_PLACEMENT_MAX_SCORE
        + envs.SCHEDULER_SCALE_UP_LOCALITY_MAX_SCORE
    )
    assert scores[2] - scores[1] > other_scorers_ceiling


@pytest.mark.asyncio
async def test_placed_but_not_yet_running_siblings_count():
    """Counted off `worker_id`, so a burst of scale-outs does not stack.

    A sibling that is scheduled but still starting already holds that worker's
    cards and will pair from there.
    """
    members = [_member(7, RoleNameEnum.DECODE.value)]

    scores = await _scores(RoleNameEnum.PREFILL.value, members, [7, 8])

    assert scores[7] > scores[8]


@pytest.mark.asyncio
async def test_other_groups_and_unplaced_members_are_ignored():
    members = [
        _member(1, RoleNameEnum.DECODE.value, group_id="another-group"),
        _member(2, RoleNameEnum.DECODE.value),
        _member(None, RoleNameEnum.DECODE.value),
    ]

    scores = await _scores(RoleNameEnum.PREFILL.value, members, [1, 2])

    assert scores[1] == 0
    assert scores[2] > 0


@pytest.mark.asyncio
async def test_no_opposite_member_placed_leaves_every_score_untouched():
    """Not zero — untouched, so the chain adds nothing and the resource
    scorers decide alone."""
    members = [_member(1, RoleNameEnum.PREFILL.value)]

    candidates = [_candidate(1), _candidate(2)]
    scorer = PairingAffinityScorer(GROUP, RoleNameEnum.PREFILL.value, members)
    scored = await scorer.score(candidates)

    assert all(c.score is None for c in scored)


@pytest.mark.asyncio
async def test_router_and_role_less_members_are_not_scored():
    """The two ways a member reaches this scorer with nothing to pair to."""
    members = [_member(1, RoleNameEnum.DECODE.value)]
    candidates = [_candidate(1), _candidate(2)]

    router = PairingAffinityScorer(GROUP, RoleNameEnum.ROUTER.value, members)
    assert all(c.score is None for c in await router.score(candidates))

    role_less = PairingAffinityScorer(None, None, members)
    assert all(c.score is None for c in await role_less.score(candidates))


class TestRetention:
    """The same ledger read from the other end: which surplus member to drop.

    `PairingAffinityScorer` answers "where should the next prefill go"; this
    one answers "which prefill costs the fewest local pairs to lose". They have
    to stay each other's inverse, so most of these assert on the relationship
    rather than on a number.
    """

    @staticmethod
    def _instance(worker_id, role, group_id=GROUP, draining_since=None):
        return SimpleNamespace(
            worker_id=worker_id,
            role=role,
            group_id=group_id,
            draining_since=draining_since,
        )

    async def _scores(self, role, peers, candidates, **kwargs):
        scorer = PairingRetentionScorer(GROUP, role, peers, **kwargs)
        scored = await scorer.score_instances(candidates)
        return [s.score for s in scored]

    @pytest.mark.asyncio
    async def test_the_lonely_prefill_scores_lowest(self):
        """3P1D down to 2P1D: the prefill with no decode beside it goes."""
        paired = self._instance(1, RoleNameEnum.PREFILL.value)
        also_paired = self._instance(1, RoleNameEnum.PREFILL.value)
        lonely = self._instance(3, RoleNameEnum.PREFILL.value)
        peers = [
            paired,
            also_paired,
            lonely,
            self._instance(1, RoleNameEnum.DECODE.value),
        ]

        scores = await self._scores(
            RoleNameEnum.PREFILL.value, peers, [paired, also_paired, lonely]
        )

        assert scores[2] == 0
        assert scores[0] == scores[1] > 0

    @pytest.mark.asyncio
    async def test_it_is_the_inverse_of_the_scale_out_rule(self):
        """Scaling out picks the worker this would delete from last.

        The regression it guards: the SPREAD placement branch the group
        inherits ranks a prefill sharing a host with a decode *below* one
        sitting alone, so scaling out and scaling in were undoing each other.
        """
        members = [
            _member(1, RoleNameEnum.DECODE.value),
            _member(1, RoleNameEnum.DECODE.value),
            _member(2, RoleNameEnum.DECODE.value),
            _member(3, RoleNameEnum.PREFILL.value),
        ]
        scale_out = await _scores(RoleNameEnum.PREFILL.value, members, [1, 2, 3])

        candidates = [self._instance(w, RoleNameEnum.PREFILL.value) for w in (1, 2, 3)]
        peers = [
            self._instance(1, RoleNameEnum.DECODE.value),
            self._instance(1, RoleNameEnum.DECODE.value),
            self._instance(2, RoleNameEnum.DECODE.value),
            *candidates,
        ]
        keep = await self._scores(RoleNameEnum.PREFILL.value, peers, candidates)

        best_to_add = max(scale_out, key=scale_out.get)
        worst_to_keep = candidates[keep.index(min(keep))].worker_id
        assert best_to_add == 1
        assert worst_to_keep == 3

    @pytest.mark.asyncio
    async def test_it_stays_below_the_status_scorer_step(self):
        """The band, and the reason it is a band at all.

        `StatusScorer` steps 50 between "starting" and "running". A pairing
        score that reaches 50 lets a starting member with company tie a
        running member sitting alone — and deleting the broken one is the only
        free move a scale-down has.
        """
        crowded = self._instance(1, RoleNameEnum.PREFILL.value)
        peers = [crowded] + [
            self._instance(1, RoleNameEnum.DECODE.value) for _ in range(8)
        ]

        scores = await self._scores(RoleNameEnum.PREFILL.value, peers, [crowded])

        assert scores[0] == envs.SCHEDULER_SCALE_DOWN_PAIRING_MAX_SCORE
        assert envs.SCHEDULER_SCALE_DOWN_PAIRING_MAX_SCORE < 50

    @pytest.mark.asyncio
    async def test_a_draining_partner_does_not_count(self):
        """It is leaving inside the window, so it is not a pair worth keeping
        a prefill next to."""
        beside_draining = self._instance(1, RoleNameEnum.PREFILL.value)
        beside_staying = self._instance(2, RoleNameEnum.PREFILL.value)
        peers = [
            beside_draining,
            beside_staying,
            self._instance(
                1, RoleNameEnum.DECODE.value, draining_since=datetime.now(timezone.utc)
            ),
            self._instance(2, RoleNameEnum.DECODE.value),
        ]

        scores = await self._scores(
            RoleNameEnum.PREFILL.value, peers, [beside_draining, beside_staying]
        )

        assert scores[0] == 0
        assert scores[1] > 0

    @pytest.mark.asyncio
    async def test_inert_when_there_is_nothing_to_pair_with(self):
        """Three ways in, one answer: all zeros, so the scorer contributes
        nothing and the rest of the chain decides alone."""
        prefill = self._instance(1, RoleNameEnum.PREFILL.value)
        router = self._instance(1, RoleNameEnum.ROUTER.value)
        decode = self._instance(1, RoleNameEnum.DECODE.value)

        no_opposite_placed = await self._scores(
            RoleNameEnum.PREFILL.value, [prefill], [prefill]
        )
        router_has_no_opposite = await self._scores(
            RoleNameEnum.ROUTER.value, [router, decode], [router]
        )
        turned_off = await self._scores(
            RoleNameEnum.PREFILL.value, [prefill, decode], [prefill], max_score=0
        )

        assert no_opposite_placed == [0]
        assert router_has_no_opposite == [0]
        assert turned_off == [0]

    @pytest.mark.asyncio
    async def test_another_generation_is_not_counted(self):
        """A `d_j` summed across two generations ranks a member by peers it
        can never pair with."""
        candidate = self._instance(1, RoleNameEnum.PREFILL.value)
        peers = [
            candidate,
            self._instance(1, RoleNameEnum.DECODE.value, group_id="older-generation"),
        ]

        scores = await self._scores(RoleNameEnum.PREFILL.value, peers, [candidate])

        assert scores == [0]


# --- members wide enough to span machines ---------------------------------- #


def _spanning_member(primary, subordinates, role, group_id=GROUP):
    """A placed member holding cards on several hosts, as the row records it."""
    return SimpleNamespace(
        worker_id=primary,
        role=role,
        group_id=group_id,
        distributed_servers=SimpleNamespace(
            subordinate_workers=[SimpleNamespace(worker_id=w) for w in subordinates]
        ),
    )


def _spanning_candidate(primary, subordinates):
    candidate = _candidate(primary)
    candidate.subordinate_workers = [SimpleNamespace(worker_id=w) for w in subordinates]
    return candidate


@pytest.mark.asyncio
async def test_a_spanning_decode_pairs_from_every_host_it_holds():
    """It holds cards on both, so a prefill on either has something local to
    fetch. Counted at the primary alone, the second host reads as empty and
    the scale-out refuses to put a prefill beside half a decode."""
    scorer = PairingAffinityScorer(
        GROUP,
        RoleNameEnum.PREFILL.value,
        [_spanning_member(1, [2], RoleNameEnum.DECODE.value)],
        max_score=100.0,
    )

    on_primary, on_subordinate, elsewhere = await scorer.score(
        [_candidate(1), _candidate(2), _candidate(3)]
    )

    assert on_primary.score == 100.0
    assert on_subordinate.score == 100.0
    assert elsewhere.score == 0.0


@pytest.mark.asyncio
async def test_a_spanning_prefill_gets_a_share_of_what_its_hosts_offer():
    """The candidate's own width divides. Only half its ranks sit beside the
    decode, so the pairing it buys is half -- which is exactly the objective's
    derivative and matches what the group-level score gives the same layout."""
    scorer = PairingAffinityScorer(
        GROUP,
        RoleNameEnum.PREFILL.value,
        [_member(2, RoleNameEnum.DECODE.value)],
        max_score=100.0,
    )

    whole, spread = await scorer.score([_candidate(2), _spanning_candidate(2, [3])])

    assert whole.score == 100.0
    assert spread.score == 50.0


@pytest.mark.asyncio
async def test_a_single_machine_member_scores_exactly_as_before():
    """The generalisation has to leave every deployment placed today alone:
    one machine means the share is the count, unchanged."""
    scorer = PairingAffinityScorer(
        GROUP,
        RoleNameEnum.PREFILL.value,
        [_member(1, RoleNameEnum.DECODE.value), _member(1, RoleNameEnum.DECODE.value)],
        max_score=100.0,
    )

    two, none = await scorer.score([_candidate(1), _candidate(9)])

    assert two.score == 200.0
    assert none.score == 0.0


@pytest.mark.asyncio
async def test_scale_down_reads_the_same_ledger_as_scale_up():
    """The two are inverses on one ledger, so a spanning member has to be
    valued the same going in and coming out -- otherwise scaling out places a
    prefill beside half a decode and scaling in deletes it again."""
    peers = [_spanning_member(1, [2], RoleNameEnum.DECODE.value)]
    scorer = PairingRetentionScorer(
        GROUP, RoleNameEnum.PREFILL.value, peers, max_score=20.0
    )

    beside_primary = _member(1, RoleNameEnum.PREFILL.value)
    beside_subordinate = _member(2, RoleNameEnum.PREFILL.value)
    alone = _member(7, RoleNameEnum.PREFILL.value)

    scored = await scorer.score_instances([beside_primary, beside_subordinate, alone])
    by_worker = {s.model_instance.worker_id: s.score for s in scored}

    # The host holding the decode's other half is worth keeping too.
    assert by_worker[1] == by_worker[2] == 20.0
    assert by_worker[7] == 0.0


@pytest.mark.asyncio
async def test_a_spanning_victim_is_valued_by_what_all_its_hosts_hold():
    """Deleting a member spread over two hosts gives up the pairing on both,
    weighted by how much of it sat on each. Valued at its primary alone, a
    prefill whose other half sits beside the decode reads as costing nothing
    and is taken apart first."""
    peers = [_member(2, RoleNameEnum.DECODE.value)]
    scorer = PairingRetentionScorer(
        GROUP, RoleNameEnum.PREFILL.value, peers, max_score=20.0
    )

    spanning = _spanning_member(1, [2], RoleNameEnum.PREFILL.value)
    beside_it = _member(2, RoleNameEnum.PREFILL.value)
    alone = _member(7, RoleNameEnum.PREFILL.value)

    scored = await scorer.score_instances([spanning, beside_it, alone])
    by_worker = {s.model_instance.worker_id: s.score for s in scored}

    # Half of the spanning member sits with the decode, so it is half as
    # expensive to lose as the one wholly beside it -- and not free.
    assert by_worker[2] == 20.0
    assert by_worker[1] == 10.0
    assert by_worker[7] == 0.0
