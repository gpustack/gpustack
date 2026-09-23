"""What a PD deployment's evaluation says it costs.

A role-bearing model must not be evaluated by asking `find_candidate` where
ONE instance of the model-level spec would go. That answer is a single
replica's claim, computed without the role's overrides, against a placement
nobody ever makes — a 2P4D group would read like a 1x deployment and a cluster
that could not hold the group would still evaluate as compatible.

So the assertions below are about three things: every replica is counted, the
router is counted, and an infeasible group is refused in the solver's own
words.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gpustack.policies.base import MemberResourceClaim
from gpustack.schemas.model_sets import ModelSpec
from gpustack.schemas.models import ComputedResourceClaim, RoleSpec
from gpustack.scheduler import evaluator
from gpustack.scheduler.evaluator import evaluate_group
from gpustack.scheduler.group_solver import GroupInfeasible, GroupPlacement

GIB = 1024**3


def _spec(roles=None, **kwargs) -> ModelSpec:
    return ModelSpec(
        name="pd",
        source="huggingface",
        huggingface_repo_id="x/y",
        backend="vLLM",
        roles=roles,
        **kwargs,
    )


def _roles(prefill=2, decode=2, router=1):
    return [
        RoleSpec(name="prefill", replicas=prefill),
        RoleSpec(name="decode", replicas=decode),
        RoleSpec(name="router", replicas=router),
    ]


def _candidate(worker_id: int, ram: int, vram: int):
    return SimpleNamespace(
        worker=SimpleNamespace(id=worker_id, name=f"w{worker_id}"),
        gpu_indexes=[0],
        gpu_type="cuda",
        computed_resource_claim=ComputedResourceClaim(ram=ram, vram={0: vram}),
        subordinate_workers=None,
    )


async def _run(placement, commit_map=None, spec=None, workers=None, demand_map=None):
    """Drive `evaluate_group` with the solve and the commit pass stubbed.

    Everything between them — the role projection, the router's own claim, the
    ordering and the summing — is the real code, because that is what the old
    path got wrong.
    """

    class FakeCapacity:
        def __init__(self, *a, **kw):
            pass

        async def commit(self, role, worker_ids, already):
            return (commit_map or {}).get(role, [])

        def notes_for(self, role):
            # What the selectors said about this role, in GiB. The evaluation
            # and the scheduler read it from the same place on purpose: a form
            # that explains a refusal differently from the deployment that
            # follows is two answers about one cluster.
            return [f"{role} needs 20.31 GiB of VRAM."] if role else []

        async def demand_for(self, role, worker_ids):
            return (demand_map or {}).get(role, (None, 0))

    view = SimpleNamespace(
        root=SimpleNamespace(descendant_worker_ids=lambda: [1, 2]),
        scopes=lambda: [],
    )

    with (
        patch.object(
            evaluator.Cluster,
            "one_by_id",
            AsyncMock(return_value=SimpleNamespace(id=1, topology=None)),
        ),
        patch.object(evaluator, "cache_instances_in", AsyncMock(return_value=[])),
        patch.object(evaluator, "build_view", lambda *a, **kw: view),
        patch.object(evaluator, "GroupCapacity", FakeCapacity),
        patch.object(
            evaluator, "solve_group_placement", AsyncMock(return_value=placement)
        ),
    ):
        return await evaluate_group(
            config=SimpleNamespace(),
            session=None,
            model=spec if spec is not None else _spec(roles=_roles()),
            workers=workers if workers is not None else [],
            model_instances=[],
            cluster_id=1,
        )


@pytest.mark.asyncio
async def test_the_total_counts_every_replica_and_the_router():
    """The headline number. Four GPU members at 40 GiB VRAM each is 160
    GiB, not the 40 the single-instance path reported — and the router's
    container memory is part of what the group asks a cluster for."""
    placement = GroupPlacement(
        layer="Rack",
        path=["rack-a"],
        assignments={"prefill": [1, 1], "decode": [2, 2]},
    )
    commit = {
        "prefill": [_candidate(1, 2 * GIB, 40 * GIB) for _ in range(2)],
        "decode": [_candidate(2, 2 * GIB, 40 * GIB) for _ in range(2)],
    }
    group = await _run(placement, commit)

    assert group.messages == []
    assert group.total.vram == 160 * GIB
    # 4 members x 2 GiB + the router's 2 GiB floor.
    assert group.total.ram == 10 * GIB
    assert {c.role: c.replicas for c in group.claims} == {
        "prefill": 2,
        "decode": 2,
        "router": 1,
    }


@pytest.mark.asyncio
async def test_the_breakdown_is_in_the_declared_role_order():
    """The solver sorts roles by weight and a dict keeps whatever order it was
    given; the breakdown is read by a person, so it follows the deployment."""
    placement = GroupPlacement(
        layer="Rack",
        path=["rack-a"],
        assignments={"decode": [2, 2], "prefill": [1, 1]},
    )
    commit = {
        "prefill": [_candidate(1, GIB, 40 * GIB) for _ in range(2)],
        "decode": [_candidate(2, GIB, 20 * GIB) for _ in range(2)],
    }
    group = await _run(placement, commit)

    assert [c.role for c in group.claims] == ["prefill", "decode", "router"]


@pytest.mark.asyncio
async def test_a_uniform_role_reports_what_one_replica_costs():
    placement = GroupPlacement(
        layer="Rack", path=["rack-a"], assignments={"prefill": [1, 1]}
    )
    commit = {"prefill": [_candidate(1, GIB, 40 * GIB) for _ in range(2)]}
    group = await _run(
        placement, commit, spec=_spec(roles=[RoleSpec(name="prefill", replicas=2)])
    )

    prefill = next(c for c in group.claims if c.role == "prefill")
    assert prefill.per_replica.vram == 40 * GIB
    assert prefill.vram == 80 * GIB


@pytest.mark.asyncio
async def test_a_role_whose_members_differ_reports_no_per_replica_figure():
    """A role spread over two accelerator types sizes differently on each.
    Showing the first member's number as if it were every member's is how a
    mixed group would read as half its real size."""
    placement = GroupPlacement(
        layer="Rack", path=["rack-a"], assignments={"prefill": [1, 2]}
    )
    commit = {
        "prefill": [_candidate(1, GIB, 40 * GIB), _candidate(2, GIB, 80 * GIB)],
    }
    group = await _run(
        placement, commit, spec=_spec(roles=[RoleSpec(name="prefill", replicas=2)])
    )

    prefill = next(c for c in group.claims if c.role == "prefill")
    assert prefill.per_replica is None
    assert prefill.vram == 120 * GIB


@pytest.mark.asyncio
async def test_an_infeasible_group_is_refused_in_the_solvers_own_words():
    """ "The group needs 8 placements in one 'Rack', and the roomiest one holds
    6" is actionable; "unable to find a schedulable worker" is not."""
    placement = GroupInfeasible(
        reason="The group needs 8 placements in one 'Rack', and the roomiest one holds 6.",
        layer="Rack",
        best_path=["rack-a"],
        needed=8,
        available=6,
    )
    group = await _run(placement)

    assert group.total is None
    assert group.claims == []
    assert "roomiest" in group.messages[0]


@pytest.mark.asyncio
async def test_a_refusal_breaks_the_group_down_the_way_an_approval_does():
    """A group that fits is described role by role, with what one member of
    each costs. A group that does not fit was described by a count of members
    alone — the same question answered in units nobody can compare — so the
    breakdown is built from what each role's own selector priced."""
    placement = GroupInfeasible(
        reason="The group needs 4 placements and the cluster has room for 0.",
        role="prefill",
        needed=4,
        available=0,
    )
    group = await _run(
        placement,
        demand_map={
            "prefill": (MemberResourceClaim(vram=40 * GIB, ram=GIB), 0),
            "decode": (MemberResourceClaim(vram=20 * GIB, ram=GIB), 2),
            "router": (MemberResourceClaim(vram=0, ram=2 * GIB), 1),
        },
    )

    assert [d.role for d in group.demands] == ["prefill", "decode", "router"]
    prefill, decode, router = group.demands
    # Two replicas of a 40 GiB member, and none of them placeable.
    assert prefill.vram == 80 * GIB
    assert prefill.per_replica.vram == 40 * GIB
    assert (prefill.replicas, prefill.placeable) == (2, 0)
    assert (decode.replicas, decode.placeable) == (2, 2)
    assert router.ram == 2 * GIB


@pytest.mark.asyncio
async def test_a_refusal_names_the_role_its_explanation_is_about():
    """The notes under a refusal are one role's, and they say "the model"
    rather than "prefill". In a group of three that reads as the whole
    deployment's footprint, which is the one thing it is not."""
    placement = GroupInfeasible(reason="no room", role="decode")
    group = await _run(placement)

    assert any("Blocked on the 'decode' role" in m for m in group.messages)


@pytest.mark.asyncio
async def test_a_role_no_selector_could_price_still_reports_its_count():
    """A breakdown is worth more with a hole in it than not at all: how many
    of a role fit is the part an operator acts on, and a size nobody measured
    must not be shown as zero bytes."""
    placement = GroupInfeasible(reason="no room", role="prefill")
    group = await _run(placement, demand_map={"decode": (None, 1)})

    decode = next(d for d in group.demands if d.role == "decode")
    assert decode.per_replica is None
    assert decode.vram == 0
    assert decode.placeable == 1


@pytest.mark.asyncio
async def test_a_breakdown_that_cannot_be_built_does_not_take_the_refusal_with_it():
    """The refusal is what the deployment form acts on; the breakdown only
    explains it."""

    class Exploding:
        def __init__(self, *a, **kw):
            pass

        def notes_for(self, role):
            return ["the cluster is full"]

        async def demand_for(self, role, worker_ids):
            raise RuntimeError("selector blew up")

    view = SimpleNamespace(
        root=SimpleNamespace(descendant_worker_ids=lambda: [1]),
        scopes=lambda: [],
    )
    with (
        patch.object(
            evaluator.Cluster,
            "one_by_id",
            AsyncMock(return_value=SimpleNamespace(id=1, topology=None)),
        ),
        patch.object(evaluator, "cache_instances_in", AsyncMock(return_value=[])),
        patch.object(evaluator, "build_view", lambda *a, **kw: view),
        patch.object(evaluator, "GroupCapacity", Exploding),
        patch.object(
            evaluator,
            "solve_group_placement",
            AsyncMock(return_value=GroupInfeasible(reason="no room", role="prefill")),
        ),
    ):
        group = await evaluate_group(
            config=SimpleNamespace(),
            session=None,
            model=_spec(roles=_roles()),
            workers=[],
            model_instances=[],
            cluster_id=1,
        )

    assert group.demands == []
    assert "the cluster is full" in group.messages


@pytest.mark.asyncio
async def test_a_commit_that_comes_up_short_refuses_rather_than_underprices():
    """The solve said it fits; if turning that into cards disagrees, the
    cluster moved under us. Reporting the members that did resolve would be a
    total that understates the group."""
    placement = GroupPlacement(
        layer="Rack", path=["rack-a"], assignments={"prefill": [1, 1]}
    )
    group = await _run(placement, {"prefill": [_candidate(1, GIB, 40 * GIB)]})

    assert group.total is None
    assert "GPU assignments" in group.messages[0]


@pytest.mark.asyncio
async def test_a_group_of_only_routers_is_priced_without_a_solve():
    """No member occupies an accelerator, so there is nothing to solve — but
    the group still has a footprint, and reporting it beats reporting
    nothing."""
    spec = _spec(roles=[RoleSpec(name="router", replicas=1)])
    group = await _run(GroupInfeasible(reason="never consulted"), spec=spec)

    assert group.messages == []
    assert group.total.vram == 0
    assert group.total.ram == 2 * GIB
    assert [c.role for c in group.claims] == ["router"]


@pytest.mark.asyncio
async def test_a_declared_router_memory_replaces_the_floor():
    spec = _spec(
        roles=[
            RoleSpec(name="router", replicas=1, resources={"memory": 8 * GIB}),
        ]
    )
    group = await _run(GroupInfeasible(reason="never consulted"), spec=spec)

    assert group.total.ram == 8 * GIB


@pytest.mark.asyncio
async def test_a_role_bearing_model_never_reaches_find_candidate():
    """The gate, stated as a test. `find_candidate` answers "where would one
    more instance go", which is not a question a group has."""
    placement = GroupPlacement(
        layer="Rack", path=["rack-a"], assignments={"prefill": [1]}
    )
    commit = {"prefill": [_candidate(1, GIB, 40 * GIB)]}

    class FakeCapacity:
        def __init__(self, *a, **kw):
            pass

        async def commit(self, role, worker_ids, already):
            return commit.get(role, [])

        def notes_for(self, role):
            return []

    view = SimpleNamespace(root=SimpleNamespace(), scopes=lambda: [])
    find_candidate = AsyncMock(return_value=(None, []))
    worker = SimpleNamespace(id=1, cluster_id=1)

    with (
        patch.object(
            evaluator.Cluster,
            "one_by_id",
            AsyncMock(return_value=SimpleNamespace(id=1, topology=None)),
        ),
        patch.object(evaluator, "cache_instances_in", AsyncMock(return_value=[])),
        patch.object(evaluator, "build_view", lambda *a, **kw: view),
        patch.object(evaluator, "GroupCapacity", FakeCapacity),
        patch.object(
            evaluator, "solve_group_placement", AsyncMock(return_value=placement)
        ),
        patch.object(evaluator.scheduler, "find_candidate", find_candidate),
        patch.object(evaluator, "set_default_spec", AsyncMock(return_value=False)),
        patch.object(evaluator, "set_gguf_model_file_path", AsyncMock()),
        patch.object(
            evaluator, "evaluate_model_input", AsyncMock(return_value=(True, []))
        ),
        patch.object(
            evaluator, "evaluate_model_metadata", AsyncMock(return_value=(True, []))
        ),
        patch.object(
            evaluator, "evaluate_environment", AsyncMock(return_value=(True, []))
        ),
    ):
        result = await evaluator.evaluate_model(
            config=SimpleNamespace(),
            session=None,
            model=_spec(roles=[RoleSpec(name="prefill", replicas=1)]),
            workers=[worker],
            model_instances=[],
            cluster_id=1,
        )

    find_candidate.assert_not_called()
    assert result.compatible is True
    assert result.resource_claim.vram == 40 * GIB
    assert result.role_resource_claims[0].role == "prefill"
    assert result.role_resource_claims_by_cluster_id[1][0].replicas == 1
