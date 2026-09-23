"""Keeping a group's later members inside the floor its formation honoured.

`MustGather` was enforced on exactly the first four rows of a deployment.
The formation goes through `solve_group_placement`, which walks up to the named
layer and stops; everything created afterwards — a scaled-out prefill, a
scaled-out decode, and the router, which the dependency gate creates a pass
later on *every* PD deployment — went through the ordinary per-instance path,
which had never heard of a floor. So a group the operator told the scheduler to
refuse rather than spread was spread anyway, one member at a time, and
`_gather_unmet` did not report it either: it computed for `PreferGather` only,
on the reasoning that `MustGather` "refused at admission instead".
"""

from types import SimpleNamespace

import pytest

from gpustack.policies.worker_filters.gather_floor_filter import GatherFloorFilter
from gpustack.topology.view import build_view
from tests.utils.topology_layers import layer_obj, lid

RACK = "topology.gpustack.ai/rack"


def _worker(id_: int, rack=None):
    return SimpleNamespace(
        id=id_,
        name=f"w{id_}",
        labels={RACK: rack} if rack else {},
        cluster_id=1,
    )


def _instance(id_, role, worker_id=None, group_id="g1"):
    return SimpleNamespace(id=id_, role=role, worker_id=worker_id, group_id=group_id)


def _view(workers):
    return build_view(SimpleNamespace(layers=[layer_obj("rack", [RACK])]), workers)


# The floor names a layer by its id, not by what it is called — that is what
# `gather.layer` carries and what `view.nodes()` resolves.
RACK_LAYER = lid("rack")


def _filter(workers, instances, layer=RACK_LAYER, group_id="g1"):
    return GatherFloorFilter(
        layer=layer,
        group_id=group_id,
        model_instances=instances,
        view=_view(workers),
        weight_bearing=["prefill", "decode"],
    )


WORKERS = [_worker(1, "rack-a"), _worker(2, "rack-a"), _worker(3, "rack-b")]


@pytest.mark.asyncio
async def test_a_scaled_out_member_stays_in_the_members_rack():
    """The deployment asked to be refused rather than placed outside the rack.
    Placing the fifth member in another one honours the ask on four rows and
    breaks it on the fifth."""
    placed = [_instance(1, "prefill", worker_id=1), _instance(2, "decode", worker_id=1)]

    kept, messages = await _filter(WORKERS, placed).filter(list(WORKERS))

    assert [w.id for w in kept] == [1, 2]
    assert RACK_LAYER in messages[0]


@pytest.mark.asyncio
async def test_the_router_is_held_to_the_same_floor():
    """It is outside the gang, which is right, and that left it outside the
    floor as well — the solver checks a worker in the domain could host it and
    the ordinary path then put it wherever it liked. A promise the placement
    did not keep."""
    placed = [_instance(1, "prefill", worker_id=1), _instance(2, "decode", worker_id=2)]

    kept, _messages = await _filter(WORKERS, placed).filter(list(WORKERS))

    assert 3 not in [w.id for w in kept]


@pytest.mark.asyncio
async def test_nothing_placed_yet_constrains_nothing():
    """Either this is the formation — which does not come through here — or the
    group is being rebuilt, and the solver is applying the floor in both."""
    kept, messages = await _filter(WORKERS, []).filter(list(WORKERS))

    assert len(kept) == 3
    assert messages == []


@pytest.mark.asyncio
async def test_the_router_does_not_anchor_the_group():
    """It holds no weights, so where it sits is not where the group is — the
    same reason `role_demands` leaves it out of the gang. Anchoring on it would
    let a stray router drag every later member after it."""
    placed = [_instance(9, "router", worker_id=3)]

    kept, _messages = await _filter(WORKERS, placed).filter(list(WORKERS))

    assert len(kept) == 3


@pytest.mark.asyncio
async def test_members_already_spread_are_not_forced_back():
    """Deliberately not a refusal. The floor is already broken — by a
    rescheduled member, or a topology edited underneath a running group — and
    refusing a new member would not put it back. Picking one of the two racks
    would be arbitrary. `_gather_unmet` reports the breach instead."""
    placed = [_instance(1, "prefill", worker_id=1), _instance(2, "decode", worker_id=3)]

    kept, messages = await _filter(WORKERS, placed).filter(list(WORKERS))

    assert len(kept) == 3
    assert messages == []


@pytest.mark.asyncio
async def test_unclassified_members_constrain_nothing():
    """ "Together" is not a fact anyone established about two unlabelled hosts,
    and the rest of the scheduler refuses to read it as one."""
    workers = [_worker(1), _worker(2)]
    placed = [_instance(1, "prefill", worker_id=1)]

    kept, _messages = await _filter(workers, placed).filter(list(workers))

    assert len(kept) == 2


@pytest.mark.asyncio
async def test_another_groups_members_are_not_anchors():
    placed = [_instance(1, "prefill", worker_id=3, group_id="other")]

    kept, _messages = await _filter(WORKERS, placed).filter(list(WORKERS))

    assert len(kept) == 3


@pytest.mark.asyncio
async def test_no_layer_no_floor():
    placed = [_instance(1, "prefill", worker_id=1)]

    kept, _messages = await _filter(WORKERS, placed, layer=None).filter(list(WORKERS))

    assert len(kept) == 3


@pytest.mark.asyncio
async def test_a_message_names_the_domain_and_the_ask():
    """A member that stays PENDING because of this has to be diagnosable from
    the row: which rack, and that refusing was the configured answer."""
    placed = [_instance(1, "prefill", worker_id=1)]

    _kept, messages = await _filter(WORKERS, placed).filter([_worker(3, "rack-b")])

    assert messages
    assert "rack-a" in messages[0]
    assert "refused" in messages[0]
