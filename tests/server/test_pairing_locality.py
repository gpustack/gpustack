"""Whether a group's KV transfers can stay off the network.

The gap this closes is on the manual-selection path: a user picks host A's
cards for prefill and host B's for decode, both roles fit, the
group starts, every request works — and every single KV transfer crosses the
network. Nothing says a word. On a link without RDMA that arrangement makes PD
strictly worse than not disaggregating at all, and the only other feedback is a
TTFT regression nobody attributes to placement.
"""

from types import SimpleNamespace

import pytest

from gpustack.schemas.models import ModelInstanceStateEnum
from gpustack.server.controllers import _pairing_remote, pairing_locality


def _model(roles=("prefill", "decode")):
    return SimpleNamespace(
        roles=[SimpleNamespace(name=name, replicas=1) for name in roles]
    )


def _instance(role, worker_id, state=ModelInstanceStateEnum.RUNNING):
    return SimpleNamespace(role=role, worker_id=worker_id, state=state)


def test_one_host_makes_every_transfer_local():
    instances = [_instance("prefill", 1), _instance("decode", 1)]
    assert pairing_locality(_model(), instances).value == 1.0
    assert _pairing_remote(_model(), instances) is False


def test_roles_split_across_hosts_can_never_be_local():
    """The arrangement manual selection produces by accident."""
    instances = [
        _instance("prefill", 1),
        _instance("prefill", 1),
        _instance("decode", 2),
        _instance("decode", 2),
    ]
    assert pairing_locality(_model(), instances).value == 0.0
    assert _pairing_remote(_model(), instances) is True


def test_an_even_spread_reproduces_the_one_over_x_ceiling():
    """The reason the threshold is zero rather than a fraction.

    A 2P2D placed one pair per host — the best placement there is — still only
    keeps half its transfers local, because the router picks a prefill and a
    decode independently. A threshold expressed as a fraction would fire here,
    on the arrangement it is supposed to reward.
    """
    for x in (2, 3, 4):
        instances = [_instance("prefill", w) for w in range(1, x + 1)]
        instances += [_instance("decode", w) for w in range(1, x + 1)]
        assert pairing_locality(_model(), instances).value == pytest.approx(1 / x)
        assert _pairing_remote(_model(), instances) is False


def test_packing_the_same_group_tighter_beats_one_over_x():
    """`1/x` is a FLOOR, not a ceiling — the machine count is the driver.

    The same 4P4D, three placements. Nothing about the deployment changes
    between them: same replica counts, same roles, same cards. Only the number
    of hosts the members sit on moves, and the locality moves with it as
    `1/m`.

    This is what the deploy form cannot say. It knows `x` and not `m`, so the
    most it can honestly offer is the `m == x` row below; the group summary
    reads the row that actually happened.
    """
    # m == x: one prefill and one decode per host.
    spread = [_instance("prefill", w) for w in range(1, 5)]
    spread += [_instance("decode", w) for w in range(1, 5)]
    assert pairing_locality(_model(), spread).value == pytest.approx(1 / 4)

    # m == 2: two of each per host.
    packed = [_instance("prefill", w) for w in (1, 1, 2, 2)]
    packed += [_instance("decode", w) for w in (1, 1, 2, 2)]
    assert pairing_locality(_model(), packed).value == pytest.approx(1 / 2)

    # m == 1.
    single = [_instance("prefill", 1) for _ in range(4)]
    single += [_instance("decode", 1) for _ in range(4)]
    assert pairing_locality(_model(), single).value == pytest.approx(1.0)


def test_spreading_past_one_pair_per_host_falls_below_the_floor():
    """The floor only holds while the roles stay mixed.

    Four hosts, but each carries one role and not the other. `m` went up and
    the mixing went away, so this lands below `1/x` — at zero, the one reading
    `_pairing_remote` marks. The solver deals round-robin precisely so that a
    group it places never looks like this; manual card selection still can.
    """
    unmixed = [_instance("prefill", w) for w in (1, 2)]
    unmixed += [_instance("decode", w) for w in (3, 4)]
    assert pairing_locality(_model(), unmixed).value == 0.0
    assert _pairing_remote(_model(), unmixed) is True


def test_a_partial_overlap_is_between_the_two():
    # prefill on 1,2 · decode both on 1 -> half the prefill picks are local.
    instances = [
        _instance("prefill", 1),
        _instance("prefill", 2),
        _instance("decode", 1),
    ]
    assert pairing_locality(_model(), instances).value == pytest.approx(0.5)
    assert _pairing_remote(_model(), instances) is False


def test_only_running_members_count():
    """A member that is not up occupies no host yet, and counting it would let
    a starting group look local before anything is placed."""
    instances = [
        _instance("prefill", 1),
        _instance("decode", 2),
        _instance("decode", 1, state=ModelInstanceStateEnum.INITIALIZING),
    ]
    assert pairing_locality(_model(), instances).value == 0.0


def test_the_router_is_not_part_of_the_pairing():
    """It holds no KV, so where it sits cannot make a transfer local."""
    instances = [
        _instance("prefill", 1),
        _instance("decode", 1),
        _instance("router", 2),
    ]
    assert pairing_locality(_model(), instances).value == 1.0


@pytest.mark.parametrize(
    "instances",
    [
        [],
        [_instance("prefill", 1)],
        [_instance("decode", 1)],
        [_instance("prefill", None), _instance("decode", 1)],
    ],
)
def test_silence_rather_than_zero_when_the_question_does_not_apply(instances):
    """`None` and `0.0` are different answers.

    A role with no running member has no placement to judge, and reporting 0
    there would mark every group PAIRING_REMOTE for the whole window between
    the first member starting and the last."""
    assert pairing_locality(_model(), instances).value is None
    assert _pairing_remote(_model(), instances) is False


def test_a_model_without_roles_is_not_a_group():
    model = SimpleNamespace(roles=None)
    assert pairing_locality(model, [_instance("prefill", 1)]).value is None
    assert _pairing_remote(model, [_instance("prefill", 1)]) is False


# --- a member that spans machines -------------------------------------------- #


def _spanning(role, worker_id, others, state=ModelInstanceStateEnum.RUNNING):
    """A member holding `worker_id` plus the workers in `others`.

    The extra machines live on `distributed_servers`, which is where a
    multi-worker instance records them and which every other "where is this
    member" reader in the server still forgets.
    """
    return SimpleNamespace(
        role=role,
        worker_id=worker_id,
        state=state,
        distributed_servers=SimpleNamespace(
            subordinate_workers=[SimpleNamespace(worker_id=w) for w in others]
        ),
    )


def test_a_spanning_member_turns_the_zero_into_silence():
    """A member on several machines leaves the two roles on disjoint sets of
    them, so the arithmetic bottoms out for a reason that has nothing to do
    with how well the group was placed. Printed as a verdict it would put a
    permanent degradation on exactly the deployments that have to span, naming
    something no operator can act on."""
    instances = [
        _spanning("prefill", 1, [2]),
        _spanning("decode", 3, [4]),
    ]

    locality = pairing_locality(_model(), instances)

    assert locality.value is None
    assert locality.source == "spanning_members"


def test_the_marker_goes_quiet_with_it():
    """Same function, so the suppression is not a second decision that could
    disagree with the figure printed beside it."""
    instances = [
        _spanning("prefill", 1, [2]),
        _spanning("decode", 3, [4]),
    ]

    assert _pairing_remote(_model(), instances) is False


def test_a_spanning_member_is_silence_even_when_the_roles_share_a_host():
    """This asserted the opposite one commit ago, on the reading that a
    shared host makes the sum an ordinary measurement. It does not.

    The formula assumes each pick lands somewhere — true while an instance is
    one machine. KV is sharded by TP rank, so a decode rank needs particular
    prefill ranks' shards; once one member's ranks are spread over several
    machines, whether a pair is local depends on the rank mapping, which a sum
    over worker counts cannot see. Prefill on {1,2} and decode on {2,3} at
    equal TP with ranks laid out in order: every rank pair is remote, and the
    sum says 0.25.

    Absent beats wrong, and this figure is not decoration — `pairing_remote`
    is derived from it."""
    instances = [
        _spanning("prefill", 1, [2]),
        _instance("decode", 1),
    ]

    locality = pairing_locality(_model(), instances)

    assert locality.value is None
    assert locality.source == "spanning_members"


def test_single_machine_members_keep_reporting_zero():
    """The case the marker was written for: pick host A's cards for prefill and
    host B's for decode and nothing spans anything. That zero is a placement
    that could have gone better, and it must keep saying so."""
    instances = [_instance("prefill", 1), _instance("decode", 2)]

    locality = pairing_locality(_model(), instances)

    assert locality.value == 0.0
    assert locality.source == "measured"
    assert _pairing_remote(_model(), instances) is True


def test_the_router_does_not_make_a_group_look_spanning():
    """It occupies no accelerator and is excluded everywhere else for that
    reason; a router with subordinates would otherwise silence the figure for
    the two roles that do the pairing."""
    instances = [
        _instance("prefill", 1),
        _instance("decode", 2),
        _spanning("router", 3, [4]),
    ]

    assert pairing_locality(
        _model(("prefill", "decode", "router")), instances
    ).source == ("measured")
