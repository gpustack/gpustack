"""`PreferGather` with a layer: ship it anyway, but say if you missed.

A lenient target — "aim for X, ship anyway" — is the combination most
deployments want, and the marker is what makes it mean something: without one
afterwards, the ask lives in the spec and the outcome lives nowhere.

The second half of the file is the other gather marker, `MustGather`'s:
`gather_blocked_scale_out`, for the refusal that SUCCEEDED and was therefore
invisible. The two never overlap — one says the floor was broken, the other
says it held and a member is waiting behind it.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gpustack.schemas.clusters import ClusterTopology
from gpustack.schemas.models import (
    DegradationReasonEnum,
    GatherSpec,
    GatherStrategyEnum,
    ModelInstanceStateEnum,
)
from gpustack.server.controllers import _gather_blocked_scale_out, _gather_unmet
from tests.utils.topology_layers import layer_dict, lid

RACK = "topology.gpustack.ai/rack"
ZONE = "topology.gpustack.ai/zone"


def _worker(id_, labels):
    return SimpleNamespace(
        id=id_, name=f"w{id_}", labels=labels, status=SimpleNamespace(topology_facts={})
    )


def _instance(worker_id, state=ModelInstanceStateEnum.RUNNING, role="prefill"):
    return SimpleNamespace(worker_id=worker_id, state=state, role=role)


async def _check(gather, workers, instances, topology=None):
    # `roles` is read by `role_takes_no_accelerator`, which is how the router
    # gets excluded — a stub without it would pass by accident.
    model = SimpleNamespace(
        gather=gather,
        cluster_id=1,
        roles=[SimpleNamespace(name=name) for name in ("prefill", "decode", "router")],
    )
    cluster = SimpleNamespace(
        topology=ClusterTopology.model_validate(topology) if topology else None
    )
    with (
        patch(
            "gpustack.schemas.clusters.Cluster.one_by_id",
            return_value=cluster,
        ),
        patch(
            "gpustack.schemas.workers.Worker.all_by_field",
            return_value=workers,
        ),
    ):
        return await _gather_unmet(None, model, instances)


PREFER_RACK = GatherSpec(strategy=GatherStrategyEnum.PREFER_GATHER, layer=lid("rack"))


@pytest.mark.asyncio
async def test_members_inside_the_wanted_rack_are_not_degraded():
    workers = [_worker(1, {RACK: "R1"}), _worker(2, {RACK: "R1"})]
    assert await _check(PREFER_RACK, workers, [_instance(1), _instance(2)]) is False


@pytest.mark.asyncio
async def test_members_split_across_racks_are_degraded():
    """The whole point: the deployment went out, and this is the only record
    that it went out looser than asked."""
    workers = [
        _worker(1, {ZONE: "H", RACK: "R1"}),
        _worker(2, {ZONE: "H", RACK: "R2"}),
    ]
    assert await _check(PREFER_RACK, workers, [_instance(1), _instance(2)]) is True


@pytest.mark.asyncio
async def test_a_tighter_placement_than_asked_for_is_not_degraded():
    """Asked for same-zone, got same-rack. Tighter is never a miss — and this
    is the direction the comparison is easiest to write backwards, since
    "looser" means *earlier* in a root-to-leaf order."""
    prefer_zone = GatherSpec(
        strategy=GatherStrategyEnum.PREFER_GATHER, layer=lid("zone")
    )
    workers = [
        _worker(1, {ZONE: "H", RACK: "R1"}),
        _worker(2, {ZONE: "H", RACK: "R1"}),
    ]
    assert await _check(prefer_zone, workers, [_instance(1), _instance(2)]) is False


@pytest.mark.asyncio
async def test_must_gather_reports_a_miss_it_should_never_have():
    """This asserted the opposite, on the reasoning that `MustGather`
    "refused at admission instead". True of the formation, which goes through
    the solver; false of everything after it. A scaled-out member and the
    router are placed by the per-instance path, so the one strategy whose whole
    point is strictness was also the one that could be violated in total
    silence.

    `GatherFloorFilter` is the enforcement. This is the net under it, for the
    cases that filter deliberately declines to force — members already spread,
    or unclassified at the layer, where refusing a new member would not put
    back a floor that is already gone. So under `MustGather` this should always
    be false, and a true here is an invariant that broke, not a report."""
    must = GatherSpec(strategy=GatherStrategyEnum.MUST_GATHER, layer=lid("rack"))
    workers = [
        _worker(1, {ZONE: "H", RACK: "R1"}),
        _worker(2, {ZONE: "H", RACK: "R2"}),
    ]
    assert await _check(must, workers, [_instance(1), _instance(2)]) is True


@pytest.mark.asyncio
async def test_must_gather_inside_the_floor_reports_nothing():
    must = GatherSpec(strategy=GatherStrategyEnum.MUST_GATHER, layer=lid("rack"))
    workers = [_worker(1, {RACK: "R1"}), _worker(2, {RACK: "R1"})]
    assert await _check(must, workers, [_instance(1), _instance(2)]) is False


@pytest.mark.asyncio
async def test_no_layer_means_nothing_to_miss():
    """`PreferGather` on its own is "anywhere is fine", which cannot be
    disappointed."""
    bare = GatherSpec(strategy=GatherStrategyEnum.PREFER_GATHER)
    workers = [_worker(1, {RACK: "R1"}), _worker(2, {RACK: "R2"})]
    assert await _check(bare, workers, [_instance(1), _instance(2)]) is False


@pytest.mark.asyncio
async def test_one_placed_member_is_silence_not_a_pass():
    """There is no distance between members to be wrong about yet. Reporting
    `False` here is the same answer as "it fits", which is why the early
    return is about the question not applying rather than about it passing."""
    workers = [_worker(1, {RACK: "R1"}), _worker(2, {RACK: "R2"})]
    one = [_instance(1), _instance(2, state=ModelInstanceStateEnum.PENDING)]
    assert await _check(PREFER_RACK, workers, one) is False


@pytest.mark.asyncio
async def test_unlabelled_members_count_as_a_miss():
    """Two workers in the unclassified bucket are NOT close: the bucket means
    "we do not know where these are", and reading it as "together" would turn
    a missing label into a confident wrong answer (`common_layer`'s own
    rule). A group that cannot be shown to meet the target has not met it."""
    workers = [_worker(1, {}), _worker(2, {})]
    assert await _check(PREFER_RACK, workers, [_instance(1), _instance(2)]) is True


@pytest.mark.asyncio
async def test_a_custom_rung_is_compared_like_any_other():
    topology = {"layers": [layer_dict("Pod", ["dc/pod"], parent="zone")]}
    prefer_pod = GatherSpec(strategy=GatherStrategyEnum.PREFER_GATHER, layer=lid("Pod"))
    same = [
        _worker(1, {ZONE: "H", "dc/pod": "P1"}),
        _worker(2, {ZONE: "H", "dc/pod": "P1"}),
    ]
    assert (
        await _check(prefer_pod, same, [_instance(1), _instance(2)], topology) is False
    )

    split = [
        _worker(1, {ZONE: "H", "dc/pod": "P1"}),
        _worker(2, {ZONE: "H", "dc/pod": "P2"}),
    ]
    assert (
        await _check(prefer_pod, split, [_instance(1), _instance(2)], topology) is True
    )


@pytest.mark.asyncio
async def test_the_router_does_not_count_towards_the_target():
    """The solver excludes it for the same reason (`role_demands`): a
    router holds no weights, so it "neither competes for cards nor constrains
    which domain the group lands in". Counting it here would report a miss
    because the *proxy* landed elsewhere — a placement nothing constrained
    and which would be made again on the next reschedule."""
    workers = [
        _worker(1, {ZONE: "H", RACK: "R1"}),
        _worker(2, {ZONE: "H", RACK: "R1"}),
        _worker(3, {ZONE: "H", RACK: "R9"}),
    ]
    group = [
        _instance(1, role="prefill"),
        _instance(2, role="decode"),
        _instance(3, role="router"),
    ]
    assert await _check(PREFER_RACK, workers, group) is False


@pytest.mark.asyncio
async def test_a_router_alone_beside_one_engine_is_not_a_group():
    """Dropping the router can leave fewer than two members to compare, and
    that is silence rather than a pass."""
    workers = [_worker(1, {RACK: "R1"}), _worker(3, {RACK: "R9"})]
    assert (
        await _check(
            PREFER_RACK,
            workers,
            [_instance(1, role="prefill"), _instance(3, role="router")],
        )
        is False
    )


def test_the_reason_is_its_own_enum_value():
    assert DegradationReasonEnum.GATHER_UNMET.value == "gather_unmet"


# --- the refusal that worked ------------------------------------------------ #
#
# `_gather_unmet` above fires when the floor has already been BROKEN, which
# under `MustGather` is an invariant check. The case that actually happens is
# the opposite one: `GatherFloorFilter` refusing a member *successfully*, so
# the member is never placed — the strategy working exactly as asked, and
# reported on the model nowhere at all. Measured in an e2e round: a 1P1D told
# to grow to 2P sat with a pending prefill behind a `running` model row with an
# empty `degradations` list, and the only trace was one member's
# `state_message`.


MUST_RACK = GatherSpec(strategy=GatherStrategyEnum.MUST_GATHER, layer=lid("rack"))


def _member(worker_id, group_id="g1", role="prefill", age_seconds=600):
    # Naive UTC, which is what the column actually stores
    # (`TimestampsMixin._datetime_func` strips the tzinfo before writing).
    created = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(
        seconds=age_seconds
    )
    return SimpleNamespace(
        worker_id=worker_id, group_id=group_id, role=role, created_at=created
    )


def _blocked(gather, instances):
    model = SimpleNamespace(
        gather=gather,
        roles=[SimpleNamespace(name=name) for name in ("prefill", "decode", "router")],
    )
    return _gather_blocked_scale_out(model, instances)


def test_a_member_that_cannot_be_placed_inside_the_domain_is_reported():
    """The whole point: the group is serving and the scale-up will never
    complete, so the model row has to say both."""
    group = [_member(1), _member(2, role="decode"), _member(None)]
    assert _blocked(MUST_RACK, group) is True


def test_formation_is_not_a_blocked_scale_out():
    """Nothing placed means no domain has been established, so there is no
    floor for a filter to enforce — the solver is the one refusing, and a group
    that cannot form fails scheduling with the shortfall named. Marking it here
    would put a placement-strategy badge on every group during its first
    seconds, which is how a marker stops being read."""
    group = [_member(None), _member(None, role="decode"), _member(None, role="router")]
    assert _blocked(MUST_RACK, group) is False


def test_the_dwell_suppresses_a_member_created_seconds_ago():
    """Every ordinary scale-up spends a few seconds with a row created and no
    worker chosen yet. Without the dwell the marker fires on all of them and
    clears again, which teaches people to ignore it by the time it means
    something."""
    group = [_member(1), _member(None, age_seconds=2)]
    assert _blocked(MUST_RACK, group) is False


def test_the_dwell_reads_an_aware_stamp_too():
    """The column stores naive UTC, but a caller assembling instances in
    memory hands over aware ones — and subtracting the two raises rather than
    answering wrongly, so this is the difference between a marker and a
    reconcile that dies."""
    aware = _member(None)
    aware.created_at = datetime.now(timezone.utc) - timedelta(seconds=600)
    assert _blocked(MUST_RACK, [_member(1), aware]) is True


def test_a_member_with_no_stamp_cannot_be_shown_to_have_waited():
    stampless = _member(None)
    stampless.created_at = None
    assert _blocked(MUST_RACK, [_member(1), stampless]) is False


def test_prefer_gather_never_refuses_so_it_never_blocks():
    """An unplaced member under `PreferGather` is a capacity fact with nothing
    to do with gather: that strategy widens to the cluster root rather than
    holding a member back."""
    prefer = GatherSpec(strategy=GatherStrategyEnum.PREFER_GATHER, layer=lid("rack"))
    assert _blocked(prefer, [_member(1), _member(None)]) is False


def test_no_gather_requirement_at_all_blocks_nothing():
    assert _blocked(None, [_member(1), _member(None)]) is False


def test_no_layer_means_no_floor_to_be_held_behind():
    """`GatherSpec` refuses this pair at construction — a refusal needs
    something to refuse below — so the only way to hold it is a row written
    before that validator existed. Checked anyway, because the alternative on
    such a row is a marker naming a domain the deployment never named."""
    layerless = SimpleNamespace(strategy=GatherStrategyEnum.MUST_GATHER, layer=None)
    assert _blocked(layerless, [_member(1), _member(None)]) is False


def test_a_group_with_every_member_placed_is_not_blocked():
    assert _blocked(MUST_RACK, [_member(1), _member(2, role="decode")]) is False


def test_a_placed_router_does_not_anchor_the_floor():
    """Same exclusion the filter makes and for the same reason: a router
    holds no weights, so it is subject to the domain without being what
    defines it. A group whose router landed first has established nothing, and
    anchoring on it would arm the marker during formation — the one case this
    has to stay quiet through."""
    group = [_member(3, role="router"), _member(None), _member(None, role="decode")]
    assert _blocked(MUST_RACK, group) is False


def test_an_unplaced_router_is_still_held_behind_the_floor():
    """Subject to it, though — the filter constrains the router too, which is
    how a router stopped being placed outside the domain its own engines are
    in."""
    group = [_member(1), _member(2, role="decode"), _member(None, role="router")]
    assert _blocked(MUST_RACK, group) is True


def test_another_generations_unplaced_member_is_not_this_ones_blockage():
    """A generation being rebuilt has unplaced rows of its own. Reading them
    against the live generation's anchors would put the marker on a restart,
    which is a group forming again."""
    group = [_member(1), _member(None, group_id="g2")]
    assert _blocked(MUST_RACK, group) is False


def test_an_orphan_row_anchors_nothing():
    """No `group_id` means no membership, so it cannot be the thing that
    establishes where the group is."""
    group = [_member(1, group_id=None), _member(None)]
    assert _blocked(MUST_RACK, group) is False


def test_the_blocked_reason_is_its_own_enum_value():
    assert (
        DegradationReasonEnum.GATHER_BLOCKED_SCALE_OUT.value
        == "gather_blocked_scale_out"
    )
