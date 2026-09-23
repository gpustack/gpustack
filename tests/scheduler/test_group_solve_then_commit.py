"""Does the solve's "it fits" survive the commit that turns it into cards?

The two passes are deliberately separate -- the solve answers *which worker*
over a tree of domains, the commit answers *which cards* for the one domain
that won -- and `commit` re-derives rather than replays, because the count runs
once per domain per layer while searching and a cache would hold whichever
domain was examined last rather than the one that won.

That separation is exactly where the two can disagree, and a disagreement is
not a small matter: the group is refused after the search already said it
fits. Ranking several domains per layer widened the gap by giving the shared
bookkeeping (`_offers`, `_spans`) more chances to be overwritten between the
two passes.

So these drive the **real** `GroupCapacity` through solve-then-commit and
assert the commit produces exactly as many members as the solve promised, on
disjoint cards.
"""

from types import SimpleNamespace

import pytest

from gpustack.schemas.models import ComputedResourceClaim, Model, RoleSpec
from gpustack.scheduler.group_capacity import GroupCapacity, _RoleProjection
from gpustack.policies.scorers.group_placement_scorer import group_scorer
from gpustack.scheduler.group_solver import (
    GroupPlacement,
    RoleDemand,
    solve_group_placement,
)
from gpustack.topology.tree import TopologyLayerSpec, build_topology, layer_names

GIB = 1024**3
RACK = "topology.gpustack.ai/rack"


def _worker(worker_id: int, rack: str, cards: int = 2):
    return SimpleNamespace(
        id=worker_id,
        name=f"w{worker_id}",
        ip=f"10.0.0.{worker_id}",
        labels={RACK: rack},
        status=SimpleNamespace(
            memory=SimpleNamespace(total=256 * GIB),
            gpu_devices=[
                SimpleNamespace(index=i, memory=SimpleNamespace(total=48 * GIB))
                for i in range(cards)
            ],
        ),
    )


def _tree(workers):
    specs = [TopologyLayerSpec(layer="RackLayer", label_keys=[RACK])]
    return build_topology(specs, workers), layer_names(specs)


class _OneCardPerMember:
    """Hands out one free card at a time, per worker, as the real ones do.

    Re-derives what is taken from the instance list it is handed, which is the
    behaviour both passes depend on: the count stands each pick in before
    asking again, and the commit replays that against the winning assignment.
    A stub that ignored the list would make the two passes agree for the wrong
    reason.
    """

    def __init__(self, instances):
        self._taken = {}
        for entry in instances:
            worker_id = getattr(entry, "worker_id", None)
            for index in getattr(entry, "gpu_indexes", None) or []:
                self._taken.setdefault(worker_id, set()).add(index)

    async def select_candidates(self, workers):
        worker = workers[0]
        used = self._taken.get(worker.id, set())
        free = [g.index for g in worker.status.gpu_devices if g.index not in used]
        if not free:
            return []
        return [
            SimpleNamespace(
                worker=worker,
                gpu_indexes=[free[0]],
                gpu_type="cuda",
                gpu_addresses=None,
                computed_resource_claim=ComputedResourceClaim(
                    vram={free[0]: 40 * GIB}, ram=GIB
                ),
                subordinate_workers=None,
                overcommit=False,
            )
        ]


def _capacity(workers, model):
    cap = GroupCapacity(SimpleNamespace(), model, workers, [])
    cap._selector = lambda m, instances, cpu_only, ram_claim=None: _OneCardPerMember(
        instances
    )
    for role in (spec.name for spec in model.roles):
        cap._eligible[role] = {w.id: w for w in workers}
        cap._projected[role] = _RoleProjection(model, False)
    return cap


def _model(prefill: int, decode: int) -> Model:
    model = Model(name="pd", source="huggingface", huggingface_repo_id="x/y")
    model.id = 1
    model.roles = [
        RoleSpec(name="prefill", replicas=prefill),
        RoleSpec(name="decode", replicas=decode),
    ]
    return model


def _demands(model):
    return [
        RoleDemand(role=spec.name, replicas=spec.replicas, weight=float(2 - i))
        for i, spec in enumerate(model.roles)
    ]


async def _solve_then_commit(workers, model, **kwargs):
    """The two passes, in the order and with the accumulation `schedule_group`
    uses -- because the order is what keeps them consistent."""
    root, layers = _tree(workers)
    capacity = _capacity(workers, model)
    placement = await solve_group_placement(
        root, _demands(model), capacity, layers, **kwargs
    )
    assert isinstance(placement, GroupPlacement), placement

    already = []
    committed = {}
    for role, worker_ids in placement.assignments.items():
        candidates = await capacity.commit(role, worker_ids, already)
        committed[role] = candidates
        for candidate in candidates:
            already.append(
                SimpleNamespace(
                    worker_id=candidate.worker.id,
                    gpu_indexes=candidate.gpu_indexes,
                    gpu_type=candidate.gpu_type,
                    computed_resource_claim=candidate.computed_resource_claim,
                    distributed_servers=None,
                )
            )
    return placement, committed


def _cards(committed):
    return sorted(
        (c.worker.id, index)
        for candidates in committed.values()
        for c in candidates
        for index in c.gpu_indexes
    )


@pytest.mark.asyncio
async def test_a_single_host_group_commits_exactly_what_was_promised():
    workers = [_worker(1, "a", cards=2)]
    model = _model(prefill=1, decode=1)

    placement, committed = await _solve_then_commit(workers, model)

    assert {r: len(c) for r, c in committed.items()} == {"prefill": 1, "decode": 1}
    # Two members, two cards, no card handed out twice.
    assert _cards(committed) == [(1, 0), (1, 1)]
    assert placement.machines() == {1}


@pytest.mark.asyncio
async def test_a_group_spanning_a_rack_commits_exactly_what_was_promised():
    """Four members over two two-card hosts: the leaf layer cannot hold them,
    so the rack wins and both hosts are used. The commit has to reproduce the
    same split, or the group is refused after being told it fits."""
    workers = [_worker(1, "a", cards=2), _worker(2, "a", cards=2)]
    model = _model(prefill=2, decode=2)

    placement, committed = await _solve_then_commit(workers, model)

    assert placement.layer == "RackLayer"
    assert {r: len(c) for r, c in committed.items()} == {"prefill": 2, "decode": 2}
    assert _cards(committed) == [(1, 0), (1, 1), (2, 0), (2, 1)]


@pytest.mark.asyncio
async def test_ranking_several_domains_does_not_spoil_the_winner_s_commit():
    """The risk ranking introduced. Counting domain after domain writes the
    same per-(role, worker) bookkeeping the commit reads back, so a domain
    examined *after* the winner could leave the winner's entry describing
    someone else's cards.

    Three racks, all of which fit, all of which are therefore counted.
    """
    workers = [_worker(1, "a"), _worker(2, "b"), _worker(3, "c")]
    model = _model(prefill=1, decode=1)

    placement, committed = await _solve_then_commit(
        workers, model, score=group_scorer(ready_worker_ids={3})
    )

    # Rack `c` is the warm one, and it is examined last.
    assert placement.path == ["c", "w3"]
    assert {r: len(c) for r, c in committed.items()} == {"prefill": 1, "decode": 1}
    assert _cards(committed) == [(3, 0), (3, 1)]


@pytest.mark.asyncio
async def test_a_role_of_two_on_one_host_gets_two_distinct_cards():
    """The count stands each pick in before asking for the next, so the two
    members of one role on one host must commit onto different cards. Handing
    the same card twice is the failure the commit guard reports as a cluster
    change."""
    workers = [_worker(1, "a", cards=3)]
    model = _model(prefill=2, decode=1)

    _placement, committed = await _solve_then_commit(workers, model)

    assert {r: len(c) for r, c in committed.items()} == {"prefill": 2, "decode": 1}
    assert _cards(committed) == [(1, 0), (1, 1), (1, 2)]


@pytest.mark.asyncio
async def test_the_solve_refuses_rather_than_promising_more_than_the_cards():
    """The other side of the same property: when the cards genuinely do not
    add up the refusal happens in the solve, where it can name the shortfall,
    and never in the commit."""
    workers = [_worker(1, "a", cards=1)]
    model = _model(prefill=2, decode=2)
    root, layers = _tree(workers)

    got = await solve_group_placement(
        root, _demands(model), _capacity(workers, model), layers
    )

    assert not isinstance(got, GroupPlacement)
    assert got.needed == 4
