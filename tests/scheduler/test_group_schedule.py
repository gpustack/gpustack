"""The group-scheduling entry point, and mostly the gate in front of it.

What is worth pinning here is not that a group gets placed — the solver has
its own suite for that — but that **nothing else can reach this path**. Group
scheduling decides "the group is never spread", and that decision is only safe
because a role-less deployment cannot enter. Every test below that returns
`False` from the gate is protecting an existing deployment.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gpustack.schemas.clusters import ClusterTopology, GatherStrategyEnum
from gpustack.schemas.models import GatherSpec, Model, RoleSpec
from gpustack.scheduler import group_schedule
from gpustack.scheduler.group_schedule import (
    gather_request,
    is_group_forming,
    schedule_group,
)
from gpustack.scheduler.group_solver import GroupInfeasible, GroupPlacement
from tests.utils.topology_layers import layer_dict, layer_obj

RACK = "topology.gpustack.ai/rack"


def _instance(id: int, role: str, worker_id=None, group_id="g1"):
    return SimpleNamespace(
        id=id,
        name=f"m-{role}-{id}",
        role=role,
        worker_id=worker_id,
        group_id=group_id,
        model_id=1,
    )


def _model(roles=True, gather=None, cluster_id=1):
    model = Model(
        name="pd",
        source="huggingface",
        huggingface_repo_id="x/y",
        gather=gather,
    )
    model.id = 1
    model.cluster_id = cluster_id
    if roles:
        model.roles = [
            RoleSpec(name="prefill", replicas=2),
            RoleSpec(name="decode", replicas=2),
            RoleSpec(name="router", replicas=1),
        ]
    return model


# --- the gate: everything that must NOT be group-scheduled ------------------ #


def test_a_role_less_model_is_never_a_group():
    """The test the whole safety argument rests on.

    A deployment without roles has replicas, and replicas are
    interchangeable — spreading them is correct and is what the existing path
    does. If this ever returns True, every pre-PD deployment silently changes
    placement policy.
    """
    model = _model(roles=False)
    instances = [_instance(1, "", group_id=None)]
    assert is_group_forming(model, instances) is False


def test_a_group_with_a_placed_member_is_a_scale_out_not_a_forming():
    """Solving again would either move running members — it cannot — or place
    the new one against a stale picture. Scale-out stays per-instance."""
    model = _model()
    instances = [
        _instance(1, "prefill", worker_id=7),
        _instance(2, "prefill"),
        _instance(3, "decode"),
        _instance(4, "decode"),
    ]
    assert is_group_forming(model, instances) is False


def test_a_fully_placed_group_is_not_forming():
    model = _model()
    instances = [
        _instance(1, "prefill", worker_id=7),
        _instance(2, "decode", worker_id=8),
    ]
    assert is_group_forming(model, instances) is False


def test_a_group_of_only_routers_is_not_forming():
    """The router takes no accelerator, so it neither competes for cards nor
    constrains the domain — there is no group to solve."""
    model = _model()
    instances = [_instance(9, "router")]
    assert is_group_forming(model, instances) is False


def test_an_unplaced_group_is_forming():
    model = _model()
    instances = [
        _instance(1, "prefill"),
        _instance(2, "prefill"),
        _instance(3, "decode"),
        _instance(4, "decode"),
        _instance(9, "router"),
    ]
    assert is_group_forming(model, instances) is True


def test_a_router_already_placed_does_not_block_forming():
    """The router is excluded from the gang (D11), so its placement says
    nothing about whether the GPU members have been solved."""
    model = _model()
    instances = [
        _instance(1, "prefill"),
        _instance(3, "decode"),
        _instance(9, "router", worker_id=7),
    ]
    assert is_group_forming(model, instances) is True


# --- where the gather requirement comes from -------------------------------- #


def test_the_models_gather_is_the_requirement():
    model = _model(
        gather=GatherSpec(strategy=GatherStrategyEnum.MUST_GATHER, layer="Rack")
    )
    request = gather_request(model)
    assert request.must is True
    assert request.layer == "Rack"


def test_a_stale_cluster_level_default_has_no_effect():
    """A cluster row may still hold `defaultGatherStrategy` /
    `defaultGatherLayer`; `ClusterTopology` ignores unknown keys and nothing
    reads them. A silent model is unconstrained no matter what its cluster
    stored.

    There is no cluster-level inheritance because the two ways of being wrong
    are not the same size: without a default, a group that wanted `rack` and
    did not say so is merely placed looser than ideal. With one, it inherits
    `MustGather` and the deployment is *refused*, for a floor the deploy form
    never showed."""
    topology = ClusterTopology.model_validate(
        {
            "layers": [layer_dict("Cabinet", [RACK])],
            "defaultGatherStrategy": "MustGather",
            "defaultGatherLayer": "Rack",
        }
    )
    assert not hasattr(topology, "default_gather_strategy")
    assert "defaultGatherStrategy" not in topology.model_dump(by_alias=True)

    request = gather_request(_model())
    assert request.must is False
    assert request.layer is None


def test_no_gather_is_a_preference_not_a_requirement():
    """Absent must never mean "refuse": the solver's own default is to widen
    to the cluster root rather than fail."""
    request = gather_request(_model())
    assert request.must is False
    assert request.layer is None


def test_an_accelerator_domain_layer_is_named_like_any_other():
    """The layer name travels alone. There is one chain, so there is nothing
    to derive from the name, and a domain rung is requested exactly the way a
    rack is."""
    model = _model(
        gather=GatherSpec(
            strategy=GatherStrategyEnum.MUST_GATHER, layer="accelerator_domain"
        )
    )
    assert gather_request(model).layer == "accelerator_domain"


# --- the solve, and what it hands back -------------------------------------- #


def _candidate(worker_id: int, gpu_indexes):
    return SimpleNamespace(
        worker=SimpleNamespace(
            id=worker_id,
            name=f"w{worker_id}",
            ip=f"10.0.0.{worker_id}",
            advertise_address=None,
            ifname="eth0",
        ),
        gpu_indexes=gpu_indexes,
        gpu_type="cuda",
        gpu_addresses=None,
        computed_resource_claim=None,
        subordinate_workers=None,
    )


async def _run(placement, commit_map=None, topology=None, group_instances=None):
    cluster = SimpleNamespace(id=1, topology=topology)
    model = _model()
    # `cluster_id` is spelled out because `schedule_group` narrows the fleet to
    # the model's own cluster before it builds the tree — `Worker.all` hands it
    # every cluster's workers — and a stand-in without the field is not a
    # worker any of those paths could ever receive.
    workers = [
        SimpleNamespace(id=1, name="w1", labels={}, status=None, cluster_id=1),
        SimpleNamespace(id=2, name="w2", labels={}, status=None, cluster_id=1),
    ]
    rows = (
        group_instances
        if group_instances is not None
        else [
            _instance(1, "prefill"),
            _instance(2, "prefill"),
            _instance(3, "decode"),
            _instance(4, "decode"),
        ]
    )

    class FakeCapacity:
        def __init__(self, *a, **kw):
            pass

        async def commit(self, role, worker_ids, already):
            return (commit_map or {}).get(role, [])

        def notes_for(self, role):
            # The selectors' own account of why nothing fit, keyed by the role
            # the solver stopped on. Spelled out here rather than stubbed away
            # because a refusal that loses it is exactly the regression the
            # assertion below watches for.
            return [f"{role} needs 20.31 GiB of VRAM."] if role else []

    with (
        patch.object(
            group_schedule.Cluster, "one_by_id", AsyncMock(return_value=cluster)
        ),
        patch.object(group_schedule, "GroupCapacity", FakeCapacity),
        patch.object(
            group_schedule,
            "solve_group_placement",
            AsyncMock(return_value=placement),
        ),
    ):
        return await schedule_group(
            session=None,
            config=SimpleNamespace(),
            model=model,
            workers=workers,
            model_instances=[],
            group_instances=rows,
        )


@pytest.mark.asyncio
async def test_a_solved_group_hands_back_one_candidate_per_row():
    placement = GroupPlacement(
        layer="Rack",
        path=["rack-a"],
        assignments={"prefill": [1, 1], "decode": [2, 2]},
    )
    commit = {
        "prefill": [_candidate(1, [0]), _candidate(1, [1])],
        "decode": [_candidate(2, [0]), _candidate(2, [1])],
    }
    by_instance, messages = await _run(placement, commit)

    assert messages == []
    assert set(by_instance) == {1, 2, 3, 4}
    # The GPU indexes are the half the solver does not answer. Two members
    # of one role on one worker must get *different* cards.
    prefill_gpus = [by_instance[1].gpu_indexes, by_instance[2].gpu_indexes]
    assert prefill_gpus == [[0], [1]]


@pytest.mark.asyncio
async def test_an_infeasible_group_places_nothing_and_says_why():
    """All-or-nothing (D14). A partial group is the state the strictness
    exists to prevent, so the mapping must be None rather than short."""
    placement = GroupInfeasible(
        reason="The group needs 4 placements in one 'Rack', and the roomiest one holds 2.",
        layer="Rack",
        best_path=["rack-a"],
        role="prefill",
        needed=4,
        available=2,
    )
    by_instance, messages = await _run(placement)
    assert by_instance is None
    assert "roomiest" in messages[0]
    # And the size beside the count. The solver reasons in placements and
    # the selectors in GiB; a refusal carrying only the first cannot tell a
    # group that is slightly too big from one that was never going to fit.
    assert any("20.31 GiB" in message for message in messages[1:])


@pytest.mark.asyncio
async def test_a_commit_pass_that_comes_up_short_refuses_the_whole_group():
    """The solve said it fits; if turning that into cards disagrees, the
    cluster moved. Placing the members that did resolve would leave a
    half-formed group."""
    placement = GroupPlacement(
        layer="Rack", path=["rack-a"], assignments={"prefill": [1, 1]}
    )
    by_instance, messages = await _run(placement, {"prefill": [_candidate(1, [0])]})
    assert by_instance is None
    assert "GPU assignments" in messages[0]


@pytest.mark.asyncio
async def test_fewer_rows_than_the_solve_waits_rather_than_placing_part():
    """The convergence loop creates rows over time. Placing part of a role now
    would need a second solve later against a picture this one already
    changed."""
    placement = GroupPlacement(
        layer="Rack", path=["rack-a"], assignments={"prefill": [1, 1]}
    )
    by_instance, messages = await _run(
        placement,
        {"prefill": [_candidate(1, [0]), _candidate(1, [1])]},
        group_instances=[_instance(1, "prefill")],
    )
    assert by_instance is None
    assert "waiting for the rest" in messages[0]


@pytest.mark.asyncio
async def test_an_invalid_cluster_topology_refuses_with_the_reason():
    """An operator error, not a capacity one — and refusing beats placing the
    group against a tree built from a guess."""
    bad = SimpleNamespace(
        layers=[layer_obj("A", parent="nope")],
    )
    by_instance, messages = await _run(
        GroupPlacement(layer="Rack", path=["rack-a"], assignments={}), topology=bad
    )
    assert by_instance is None
    assert "topology is invalid" in messages[0]
