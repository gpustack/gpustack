"""An instance must not be weighed against its own resource claim.

`get_worker_allocatable_resource` computes a GPU's free VRAM as
`total - sum(claims of every instance on it) - system_reserved`, and the
scheduler hands it every row from `ModelInstance.all()` -- including the one it
is currently placing.

A freshly created instance has `computed_resource_claim = None`, so the sum
skips it. The hazard appears once an instance has been placed, written a claim,
and gone back to PENDING (a retry, a re-deploy, a group re-solve): its own claim
now counts against the GPUs it is asking for. Because the claim is
`gpu_memory_utilization x total`, allocatable collapses to the remaining 10%,
the GPU fails the `allocatable/total >= gpu_memory_utilization` test, every
candidate is classed overcommit, and a multi-replica model refuses overcommit
outright. The retry re-reads the same stale claim, so it never recovers.

It looks from the outside like a scheduler that refuses *empty* cards: the
replica stays PENDING while each candidate GPU reports allocatable/total = 0.10
with that instance's own claim as the only occupant.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gpustack.policies.utils import get_worker_allocatable_resource
from gpustack.scheduler import scheduler

TOTAL = 68719476736  # 64 GiB, one 910B2 die
UTIL = 0.9
CLAIM = int(TOTAL * UTIL)  # what a --gpu-memory-utilization=0.9 instance claims


def _worker(worker_id=3, gpu_indexes=(6, 7)):
    return SimpleNamespace(
        id=worker_id,
        name="asc-w22",
        system_reserved=SimpleNamespace(ram=0, vram=0),
        status=SimpleNamespace(
            memory=SimpleNamespace(total=2 * 1024**4, is_unified_memory=False),
            gpu_devices=[
                SimpleNamespace(
                    index=i,
                    type="ascend",
                    memory=SimpleNamespace(total=TOTAL),
                )
                for i in gpu_indexes
            ],
        ),
    )


def _instance(instance_id, worker_id=3, gpu_indexes=(6, 7), claim=CLAIM):
    """A PENDING instance that has already written a claim -- the state the
    retry loop gets stuck in."""
    return SimpleNamespace(
        id=instance_id,
        worker_id=worker_id,
        gpu_type="ascend",
        gpu_indexes=list(gpu_indexes),
        distributed_servers=None,
        computed_resource_claim=SimpleNamespace(
            ram=0, vram={i: claim for i in gpu_indexes}
        ),
    )


class TestTheArithmeticThatBlocksIt:
    def test_its_own_claim_eats_the_gpu_it_is_asking_for(self):
        """The defect, stated as numbers: one PENDING instance on two empty
        cards drives allocatable/total to 0.10."""
        worker = _worker()
        stuck = _instance(404)

        allocatable = get_worker_allocatable_resource([stuck], worker)

        for idx in (6, 7):
            ratio = allocatable.vram[idx] / TOTAL
            assert ratio == pytest.approx(1 - UTIL, abs=1e-6)
            assert ratio < UTIL, "this is what makes every candidate overcommit"

    def test_excluding_it_gives_the_empty_cards_back(self):
        """The fix, stated as numbers: drop the row being placed and the same
        two cards report fully free."""
        worker = _worker()
        stuck = _instance(404)

        allocatable = get_worker_allocatable_resource(
            [mi for mi in [stuck] if mi.id != 404], worker
        )

        for idx in (6, 7):
            assert allocatable.vram[idx] == TOTAL
            assert allocatable.vram[idx] / TOTAL >= UTIL

    def test_a_sibling_on_the_same_cards_still_counts(self):
        """Only the instance being placed is exempt. A *different* instance
        holding those cards is a real occupant and must keep them."""
        worker = _worker()
        sibling = _instance(500)
        stuck = _instance(404)

        allocatable = get_worker_allocatable_resource(
            [mi for mi in [sibling, stuck] if mi.id != 404], worker
        )

        for idx in (6, 7):
            assert allocatable.vram[idx] == TOTAL - CLAIM

    def test_a_claimless_pending_row_was_always_harmless(self):
        """Why this went unnoticed: a never-placed instance carries no claim,
        so the sum skipped it and the bug needed a re-deploy to show up."""
        worker = _worker()
        fresh = _instance(404)
        fresh.computed_resource_claim = None

        allocatable = get_worker_allocatable_resource([fresh], worker)

        for idx in (6, 7):
            assert allocatable.vram[idx] == TOTAL


class _Recorder:
    """Stands in for whichever resource-fit selector gets built."""

    seen = []

    def __init__(self, *args, **kwargs):
        self.model = next(a for a in args if hasattr(a, "backend_parameters"))
        self.instances = next(a for a in args if isinstance(a, list))
        type(self).seen.append(self)

    async def select_candidates(self, workers):
        return []

    def get_messages(self):
        return []


@pytest.fixture
def harness():
    _Recorder.seen = []
    scorers_seen = []

    class _FilterChain:
        def __init__(self, filters):
            pass

        async def filter(self, workers):
            return workers, []

    class _ScoreChain:
        def __init__(self, scorers):
            pass

        async def score(self, candidates):
            return candidates

    def _placement_scorer(model, model_instances):
        scorers_seen.append(model_instances)
        return SimpleNamespace()

    with patch.multiple(
        scheduler,
        WorkerFilterChain=_FilterChain,
        CandidateScoreChain=_ScoreChain,
        PlacementScorer=_placement_scorer,
        pick_highest_score_candidate=lambda candidates: None,
        VLLMResourceFitSelector=_Recorder,
    ):
        yield SimpleNamespace(selector=_Recorder, scorers_seen=scorers_seen)


def _model():
    from gpustack.schemas.models import BackendEnum, Model

    return Model(
        id=1,
        name="pd",
        source="local_path",
        local_path="/models/m",
        backend=BackendEnum.VLLM.value,
        replicas=7,
    )


async def _run(instances, exclude_instance_id=None):
    return await scheduler.find_candidate(
        None,
        SimpleNamespace(cache_dir=None),
        _model(),
        [_worker()],
        instances,
        exclude_instance_id=exclude_instance_id,
    )


class TestFindCandidateDropsTheRowItIsPlacing:
    @pytest.mark.asyncio
    async def test_the_selector_never_sees_the_instance_being_placed(self, harness):
        stuck, sibling = _instance(404), _instance(500)

        await _run([stuck, sibling], exclude_instance_id=404)

        got = harness.selector.seen[0].instances
        assert [mi.id for mi in got] == [500]

    @pytest.mark.asyncio
    async def test_the_placement_scorer_gets_the_same_filtered_list(self, harness):
        """The scorer spreads replicas across workers by counting placed
        instances; an unplaced one must not be counted either."""
        stuck, sibling = _instance(404), _instance(500)

        await _run([stuck, sibling], exclude_instance_id=404)

        assert [mi.id for mi in harness.scorers_seen[0]] == [500]

    @pytest.mark.asyncio
    async def test_without_the_argument_nothing_is_filtered(self, harness):
        """The evaluator's dry run places nothing, so it passes no id and must
        keep seeing the cluster exactly as it is."""
        stuck, sibling = _instance(404), _instance(500)

        await _run([stuck, sibling])

        assert [mi.id for mi in harness.selector.seen[0].instances] == [404, 500]

    @pytest.mark.asyncio
    async def test_an_id_that_matches_nothing_is_a_no_op(self, harness):
        stuck, sibling = _instance(404), _instance(500)

        await _run([stuck, sibling], exclude_instance_id=999)

        assert [mi.id for mi in harness.selector.seen[0].instances] == [404, 500]
