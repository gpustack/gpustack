"""PD group placement, end to end, with nothing on the decision path stubbed.

Three scheduling defects in a row shipped past a green suite, and all three had
the same shape: **the test replaced the very seam that was broken**.

- `tests/scheduler/test_group_schedule.py` swaps the whole of `GroupCapacity`
  for a `FakeCapacity`. `GroupCapacity._translate` was handing the solver back
  a resource claim with no GPU index attached, so the second role saw every
  card on the worker as free and a 2P1D was dealt onto a two-card host. The
  fake had no `_translate` to get wrong (fixed in `bf9962c2`).
- `tests/server/test_soft_scale_down.py` mocks both `find_scale_down_candidates`
  *and* `ModelInstance.update` on every case. With the write stubbed out,
  nothing noticed that the scale-down mark never reached the database
  (fixed in `b496a768`).

So the rule for this file is: **no test double stands between the fixture and
the decision**. The workers are the real JSON fixtures, the topology is a real
`build_view`, the capacity is a real `GroupCapacity` driving real selectors
through the real `count_offer_slots`, and the placement is a real
`solve_group_placement`. The only things patched are the ones that would reach
the network or the database and answer nothing about placement:

- `get_pretrained_config_with_workers` -- a HuggingFace download. Pinned to
  Llama-3.1-8B's published hyperparameters so the VRAM estimate is a fixed
  number rather than whatever the hub serves today.
- the `async_session` behind `BackendFrameworkFilter`, and the session handed
  to `schedule_group` -- both pure reads (runner catalog, cache instances).
- `Cluster.one_by_id` -- the row the cluster topology is read from.

Every assertion names **which worker and which cards**, because "a placement
was returned" is the one thing all three defects above also did.
"""

from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gpustack.schemas.clusters import GatherStrategyEnum
from gpustack.schemas.models import (
    BackendEnum,
    GatherSpec,
    Model,
    ModelInstance,
    ModelInstanceStateEnum,
    RoleSpec,
    SourceEnum,
)
from gpustack.scheduler import group_schedule
from gpustack.scheduler.group_capacity import GroupCapacity
from gpustack.topology.tree import NODE_LAYER
from gpustack.scheduler.group_schedule import schedule_group
from tests.fixtures.workers.fixtures import (
    linux_nvidia_22_H100_80gx8,
    linux_nvidia_23_H100_80gx8,
    linux_nvidia_24_H100_80gx8,
    linux_nvidia_25_H100_80gx8,
    linux_nvidia_5_a100_80gx2,
    linux_nvidia_6_a100_80gx2,
    linux_nvidia_7_a100_80gx2,
)
from tests.utils.topology_layers import layer_obj, lid

GIB = 1024**3

RACK_KEY = "topology.gpustack.ai/rack"
# A layer is addressed by its resolved id everywhere the scheduler touches it —
# `gather.layer`, `view.nodes()`, the solver's scope names. Spelling "rack"
# into a `GatherSpec` would match no domain at all and the requirement would be
# dropped rather than enforced, which is a green test that proves nothing.
RACK_LAYER = lid("rack")

# meta-llama/Llama-3.1-8B-Instruct, as published. Fixed here so the VRAM claim
# the selectors compute (19.95 GiB at the default utilization) is a property of
# this file rather than of network weather.
LLAMA_31_8B = SimpleNamespace(
    architectures=["LlamaForCausalLM"],
    num_hidden_layers=32,
    hidden_size=4096,
    intermediate_size=14336,
    vocab_size=128256,
    num_attention_heads=32,
    num_key_value_heads=8,
    torch_dtype="bfloat16",
    max_position_embeddings=131072,
)


async def _pretrained_config(model, workers=None, trust_remote_code=False):
    return LLAMA_31_8B


def _read_only_session():
    """A session that answers every query with nothing.

    Two readers need one: `BackendFrameworkFilter` looks up the inference
    backend rows and the runner overrides, and `cache_instances_in` looks up
    the cluster's cache servers. Empty is the honest answer for both in a
    fleet built from fixtures — and empty overrides is what makes the filter
    fall through to the packaged runner catalog, so the filter itself still
    runs for real.
    """
    session = MagicMock()
    result = MagicMock()
    result.all = MagicMock(return_value=[])
    result.first = MagicMock(return_value=None)
    session.exec = AsyncMock(return_value=result)
    return session


def _session_cm():
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=_read_only_session())
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _offline():
    """The two patches that keep this file off the network and off a database.

    Returned as a pair rather than applied here so the same two cover both
    drivers below: `_place`, which goes through `schedule_group`, and the one
    test that asks `GroupCapacity` directly what it measured.
    """
    return (
        patch(
            "gpustack.policies.candidate_selectors.base_candidate_selector."
            "get_pretrained_config_with_workers",
            new=_pretrained_config,
        ),
        patch(
            "gpustack.policies.worker_filters.backend_framework_filter.async_session",
            side_effect=_session_cm,
        ),
    )


# --- the fleet, the model, the members -------------------------------------- #

# Real worker fixtures, grouped by what one machine is. The ids come from the
# JSON and are kept, so an assertion names the host the fixture describes.
FLEETS = {
    # 8 x H100 80G, ids 22..25.
    "H100x8": (
        linux_nvidia_22_H100_80gx8,
        linux_nvidia_23_H100_80gx8,
        linux_nvidia_24_H100_80gx8,
        linux_nvidia_25_H100_80gx8,
    ),
    # 2 x A100 80G, ids 8..10. The narrow machine: one member per card, so the
    # arithmetic a test is pinning down is visible without 16 cards of slack.
    "A100x2": (
        linux_nvidia_5_a100_80gx2,
        linux_nvidia_6_a100_80gx2,
        linux_nvidia_7_a100_80gx2,
    ),
}


def _fleet(kind: str, count: int, racks=(), cards=None, ram=None):
    """`count` machines of one card type, optionally narrowed or placed.

    `cards` trims the reported devices (a scalar, or one entry per machine) —
    the fixtures are 2- and 8-card hosts and a scenario sometimes needs a
    1-card one. `ram` overwrites `memory.total`, which is how a host that
    cannot measure itself (`0`) is told apart from one that is merely small.
    """
    workers = []
    for n, load in enumerate(FLEETS[kind][:count]):
        worker = load()
        # The fixtures predate clusters and carry no id; `ClusterFilter` reads
        # it, so without this every worker is filtered out for every role.
        worker.cluster_id = 1
        if cards is not None:
            want = cards[n] if isinstance(cards, (list, tuple)) else cards
            worker.status.gpu_devices = worker.status.gpu_devices[:want]
        if ram is not None:
            worker.status.memory.total = (
                ram[n] if isinstance(ram, (list, tuple)) else ram
            )
        if racks and racks[n]:
            worker.labels = dict(worker.labels or {})
            worker.labels[RACK_KEY] = racks[n]
        workers.append(worker)
    return workers


def _pd_model(prefill=1, decode=1, router=1, params=None, gather=None) -> Model:
    """An xPyD deployment of one real model on one real engine."""
    model = Model(
        name="pd",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="meta-llama/Llama-3.1-8B-Instruct",
        backend=BackendEnum.VLLM.value,
        backend_parameters=params,
        gather=gather,
    )
    model.id = 1
    model.cluster_id = 1
    model.roles = [
        RoleSpec(name=name, replicas=replicas)
        for name, replicas in (
            ("prefill", prefill),
            ("decode", decode),
            ("router", router),
        )
        if replicas
    ]
    return model


def _members(model: Model) -> List[ModelInstance]:
    """One unplaced row per replica, the way the convergence loop creates them."""
    rows: List[ModelInstance] = []
    for spec in model.roles:
        for _ in range(spec.replicas):
            index = len(rows) + 1
            rows.append(
                ModelInstance(
                    id=index,
                    name=f"pd-{spec.name}-{index}",
                    model_id=model.id,
                    model_name=model.name,
                    role=spec.name,
                    group_id="g1",
                    state=ModelInstanceStateEnum.PENDING,
                )
            )
    return rows


Placement = Dict[str, List[Tuple[int, List[int]]]]


async def _place(
    config,
    model: Model,
    workers: Sequence,
    topology=None,
    raw: bool = False,
) -> Tuple[Optional[Placement], List[str]]:
    """Drive `schedule_group` and read the answer off the member rows.

    Returns `({role: [(worker_id, gpu_indexes), ...]}, messages)`, or
    `(None, messages)` when the group was refused. A role with no entry was
    given no worker — which is the expected state for the router and the
    refused state for everything else.

    `raw=True` hands back the candidates themselves instead, for the few
    assertions that are about what a member holds besides its primary machine
    — the summary above is keyed by one worker and cannot express a member
    that spans.
    """
    rows = _members(model)
    cluster = SimpleNamespace(id=1, topology=topology)
    pretrained, backend_session = _offline()
    with (
        patch.object(
            group_schedule.Cluster, "one_by_id", AsyncMock(return_value=cluster)
        ),
        pretrained,
        backend_session,
    ):
        by_instance, messages = await schedule_group(
            session=_read_only_session(),
            config=config,
            model=model,
            workers=list(workers),
            # Nothing else is running on this fleet: every scenario below is
            # about what the group's own members do to each other.
            model_instances=[],
            group_instances=rows,
        )

    if by_instance is None:
        return None, messages
    if raw:
        return by_instance, messages

    placed: Placement = {}
    for row in rows:
        candidate = by_instance.get(row.id)
        if candidate is None:
            continue
        placed.setdefault(row.role, []).append(
            (candidate.worker.id, list(candidate.gpu_indexes))
        )
    return placed, messages


def _rack_topology():
    return SimpleNamespace(layers=[layer_obj("rack", [RACK_KEY])])


def _cards(placed: Placement) -> List[Tuple[int, int]]:
    """Every (worker, card) the group took, so overlap is one assertion."""
    return [
        (worker_id, index)
        for entries in placed.values()
        for worker_id, indexes in entries
        for index in indexes
    ]


# --- A. basic placement ----------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_1p1d_lands_on_one_host_without_sharing_a_card(config):
    """The smallest group there is, on a machine with room to spare.

    The cards are the assertion. A 1P1D that "succeeded" while handing both
    members card 0 is the defect this file exists for, and it reports itself
    as a success everywhere except here.
    """
    placed, messages = await _place(config, _pd_model(1, 1), _fleet("H100x8", 1))

    assert messages == []
    assert placed == {"prefill": [(22, [1])], "decode": [(22, [0])]}
    assert len(set(_cards(placed))) == 2


@pytest.mark.asyncio
async def test_a_2p2d_at_tp4_fills_two_eight_card_hosts_exactly(config):
    """Four members of four cards each onto two 8-card hosts: no slack at all.

    Pins the two directions that are deliberately opposite — the roomiest
    worker fills first *within* the domain, and the roles are dealt
    round-robin rather than packed — because packing greedily would put both
    prefills on host 22 and both decodes on host 23, the one arrangement in
    which no prefill/decode pair is local.
    """
    model = _pd_model(2, 2, params=["--tensor-parallel-size=4"])
    placed, messages = await _place(config, model, _fleet("H100x8", 2))

    assert messages == []
    assert placed == {
        "prefill": [(22, [4, 5, 6, 7]), (23, [4, 5, 6, 7])],
        "decode": [(22, [0, 1, 2, 3]), (23, [0, 1, 2, 3])],
    }
    assert len(set(_cards(placed))) == 16


@pytest.mark.asyncio
async def test_a_low_utilization_lets_two_members_share_one_card(config):
    """`--gpu-memory-utilization` is the whole of "how many fit on a card".

    At 0.45 two decodes sit on card 0 of the same host and the prefill takes
    card 1 — three members on a two-card machine. The point is not that
    sharing is desirable but that the count comes from the selectors rather
    than from counting cards: anything that re-derives capacity from the
    device list would say two.
    """
    model = _pd_model(1, 2, params=["--gpu-memory-utilization=0.45"])
    placed, messages = await _place(config, model, _fleet("A100x2", 1))

    assert messages == []
    assert placed == {"prefill": [(8, [1])], "decode": [(8, [0]), (8, [0])]}


@pytest.mark.asyncio
async def test_a_high_utilization_gives_each_member_its_own_card(config):
    """The other half of the previous test, on the same two-card host.

    At the default 0.9 the same 1P2D no longer fits, which is what makes the
    0.45 result above a measurement rather than a coincidence.
    """
    placed, messages = await _place(config, _pd_model(1, 2), _fleet("A100x2", 1))

    assert placed is None
    assert messages[0] == (
        "The group needs 3 placements and the cluster has room for 2."
    )


@pytest.mark.asyncio
async def test_a_group_that_does_not_fit_places_no_member_at_all(config):
    """All-or-nothing (D14), and a refusal that carries both units.

    A partial group is the state the strictness exists to prevent, so the
    mapping must be `None` rather than short. And the refusal has to say both
    how many placements are missing *and* what one member costs in GiB: the
    count alone cannot tell a group that is two cards short from one that was
    never going to fit on this hardware, and those call for different actions.
    """
    placed, messages = await _place(config, _pd_model(4, 4), _fleet("A100x2", 1))

    assert placed is None
    assert messages[0] == (
        "The group needs 8 placements and the cluster has room for 2."
    )
    assert any("19.95 GiB of VRAM" in message for message in messages[1:])


# --- B. one role paying for what the other took ----------------------------- #


@pytest.mark.asyncio
async def test_b_a_2p1d_is_refused_on_a_two_card_host(config):
    """The direct regression for `bf9962c2`.

    Prefill counted two slots on the host; decode, asked afterwards, counted
    one *more* — because the commits handed back for prefill carried a claim
    with no card attached, so the allocation accounting could not subtract it
    from any GPU. The solver dealt three members onto two cards and the
    mistake only surfaced in the commit pass, which reported it as "the
    cluster changed during scheduling" — a sentence that was never true, and
    that sent operators looking at their fleet instead of at this code.

    So two assertions, and the second is the one that fails on a regression:
    nothing is placed, and the reason is arithmetic rather than a phantom
    race.
    """
    placed, messages = await _place(config, _pd_model(2, 1), _fleet("A100x2", 1))

    assert placed is None
    assert messages[0] == (
        "The group needs 3 placements and the cluster has room for 2."
    )
    assert not any("changed during scheduling" in message for message in messages)


@pytest.mark.asyncio
async def test_b_an_overflowing_role_is_refused_by_arithmetic_not_a_phantom_race(
    config,
):
    """The sharpest form of the same regression, on an odd number of cards.

    Three cards, two decodes and two prefills. Decode is placed first and takes
    two of the three, so prefill has one card for two members and the group is
    refused — with the count, from the solver.

    Three cards rather than two because that is what separates the two ways
    `_translate` has been wrong. Losing the card entirely lets both roles
    believe all three cards are free and the group is *placed*, double-booked.
    Remembering only the *first* candidate per (role, worker) under-reports
    decode by one, the solve believes it fits, and the failure lands in the
    commit pass as "the cluster changed during scheduling" — a race that never
    happened, reported about a cluster that never moved. The final assertion
    is the one that tells those apart.
    """
    placed, messages = await _place(
        config, _pd_model(2, 2), _fleet("H100x8", 1, cards=3)
    )

    assert placed is None
    assert messages[0] == (
        "The group needs 4 placements and the cluster has room for 3."
    )
    assert not any("changed during scheduling" in message for message in messages)


@pytest.mark.asyncio
async def test_b_two_members_of_one_role_get_different_cards(config):
    """`_translate` takes the n-th candidate the count produced, not the first.

    Remembering only the first made every later member of a role invisible to
    the next one, which is the same root cause as the test above seen from
    inside a single role.
    """
    placed, messages = await _place(config, _pd_model(2, 0), _fleet("A100x2", 1))

    assert messages == []
    assert placed == {"prefill": [(8, [0]), (8, [1])]}


# --- C. the router: counted, never placed ----------------------------------- #


@pytest.mark.asyncio
async def test_c_a_group_whose_router_has_no_room_is_refused(config):
    """The regression for `9484dc8b`.

    A router occupies no accelerator, so it is kept out of the gang — counting
    it would make a 4P4D need nine placements in one domain and refuse racks
    that would have served. Left out *entirely*, though, the group is admitted
    onto hardware with no room for the one member that answers requests, and
    the deployment never becomes servable.

    Built by shrinking the host's RAM to 1 GiB: the GPU members claim no RAM
    worth speaking of and still fit, while the router's 2 GiB floor does not.
    """
    fleet = _fleet("A100x2", 1, ram=[1 * GIB])
    placed, messages = await _place(config, _pd_model(1, 1), fleet)

    assert placed is None
    assert "'router'" in messages[0]
    assert "accelerator-bearing members fit" in messages[0]


@pytest.mark.asyncio
async def test_c_a_router_that_fits_is_checked_but_not_assigned(config):
    """It is created a pass later by the dependency gate, from peer addresses
    that do not exist while the solve runs. Assigning it here would be a
    placement made against hosts the peers had not been written to yet."""
    placed, messages = await _place(config, _pd_model(1, 1), _fleet("H100x8", 1))

    assert messages == []
    assert "router" not in placed


@pytest.mark.asyncio
async def test_c_the_router_does_not_inflate_the_groups_size(config):
    """4P4D needs eight placements in one domain, not nine.

    The number is load-bearing under `MustGather`: a ninth placement is a rack
    that has room for the group being refused on behalf of a member that fits
    anywhere.
    """
    placed, messages = await _place(config, _pd_model(4, 4), _fleet("A100x2", 1))

    assert placed is None
    assert "needs 8 placements" in messages[0]


# --- D. topology and gather ------------------------------------------------- #


@pytest.mark.asyncio
async def test_d_must_gather_places_the_group_inside_one_rack(config):
    """Two racks, and the group fits in the first — so the floor costs nothing.

    Hosts 8 and 9 are in rack-a and host 10 is in rack-b; the 2P2D lands
    entirely on the rack-a pair.
    """
    model = _pd_model(
        2,
        2,
        gather=GatherSpec(strategy=GatherStrategyEnum.MUST_GATHER, layer=RACK_LAYER),
    )
    fleet = _fleet("A100x2", 3, racks=["rack-a", "rack-a", "rack-b"])
    placed, messages = await _place(config, model, fleet, topology=_rack_topology())

    assert messages == []
    assert placed == {
        "prefill": [(8, [1]), (9, [1])],
        "decode": [(8, [0]), (9, [0])],
    }


@pytest.mark.asyncio
async def test_d_must_gather_refuses_rather_than_crossing_a_rack(config):
    """The refusal *is* the strategy.

    The solver already placed into the tightest domain that fits; all
    `MustGather` adds is a ceiling on the widening walk. One card in rack-a
    and two in rack-b hold the four-member group between them and in neither
    rack alone, so the walk stops and the message names the layer it stopped
    at — by its resolved id, which is what `gather.layer` carries.
    """
    model = _pd_model(
        2,
        2,
        gather=GatherSpec(strategy=GatherStrategyEnum.MUST_GATHER, layer=RACK_LAYER),
    )
    fleet = _fleet("A100x2", 3, racks=["rack-a", "rack-b", "rack-b"], cards=1)
    placed, messages = await _place(config, model, fleet, topology=_rack_topology())

    assert placed is None
    assert messages[0] == (
        f"The group needs 4 placements in one {RACK_LAYER!r}, "
        "and the roomiest one holds 2."
    )


@pytest.mark.asyncio
async def test_d_prefer_gather_crosses_the_rack_rather_than_refusing(config):
    """A target, not a floor — the whole difference between the two strategies.

    Same shape as the refusal above, with enough cards that the group fits
    across the two racks and in neither alone. `PreferGather` deploys it and
    records the breach elsewhere; it cannot fail on gather grounds.
    """
    model = _pd_model(
        2,
        2,
        gather=GatherSpec(strategy=GatherStrategyEnum.PREFER_GATHER, layer=RACK_LAYER),
    )
    fleet = _fleet("A100x2", 3, racks=["rack-a", "rack-b", "rack-b"], cards=[2, 1, 1])
    placed, messages = await _place(config, model, fleet, topology=_rack_topology())

    assert messages == []
    assert placed == {
        "prefill": [(10, [0]), (8, [1])],
        "decode": [(8, [0]), (9, [0])],
    }


@pytest.mark.asyncio
async def test_d_without_a_gather_the_tightest_fitting_domain_wins(config):
    """Between domains the *smallest* that fits wins, so the group leaves the
    least fragmentation behind. (Inside the chosen domain the roomiest worker
    fills first — the opposite direction, and both are deliberate.)

    One card in rack-a and one in each of rack-b's two hosts. The pair could
    be taken from rack-a plus rack-b, which is what the cluster root would
    offer; the rack scope is reached first and rack-b holds the whole group,
    so nothing lands in rack-a.
    """
    fleet = _fleet("A100x2", 3, racks=["rack-a", "rack-b", "rack-b"], cards=1)
    placed, messages = await _place(
        config, _pd_model(1, 1), fleet, topology=_rack_topology()
    )

    assert messages == []
    assert placed == {"prefill": [(10, [0])], "decode": [(9, [0])]}


@pytest.mark.asyncio
async def test_d_unclassified_workers_are_not_a_domain(config):
    """ "Together" is not a fact anyone established about an unlabelled host.

    Host 8 is in rack-a and host 9 has no rack at all. Gathering the group
    into the unclassified bucket would be claiming the two are adjacent on the
    strength of one of them being unlabelled — so the bucket is skipped, the
    only real rack holds one member, and the `MustGather` is refused.
    """
    model = _pd_model(
        1,
        1,
        gather=GatherSpec(strategy=GatherStrategyEnum.MUST_GATHER, layer=RACK_LAYER),
    )
    fleet = _fleet("A100x2", 2, racks=["rack-a", None], cards=1)
    placed, messages = await _place(config, model, fleet, topology=_rack_topology())

    assert placed is None
    assert messages[0] == (
        f"The group needs 2 placements in one {RACK_LAYER!r}, "
        "and the roomiest one holds 1."
    )


# --- E. unmeasurable is not full -------------------------------------------- #


@pytest.mark.asyncio
async def test_e_a_worker_reporting_no_memory_is_unknown_not_zero(config):
    """A host that says `memory.total = 0` has told us nothing, not that it
    is full — no host runs on none.

    Two consequences, and both are asserted because only the first is visible
    from the message. The worker is *absent* from the capacity mapping rather
    than present with a zero, which is what lets the refusal distinguish a
    cluster nobody could measure from one that is genuinely out of cards; and
    the refusal says so, because "full" is the one answer that stops an
    operator looking for a mistake.
    """
    fleet = _fleet("A100x2", 2, ram=[0, 185695764480])
    model = _pd_model(2, 2)

    pretrained, backend_session = _offline()
    with pretrained, backend_session:
        capacity = GroupCapacity(config, model, fleet, [], [])
        slots = await capacity("prefill", [w.id for w in fleet], [])

    # Host 8 reports no memory; host 9 has two cards' worth of room.
    assert slots == {9: 2}

    placed, messages = await _place(config, model, fleet)
    assert placed is None
    assert "capacity could not be measured on 1 worker(s)" in messages[0]


@pytest.mark.asyncio
async def test_e_a_fleet_that_is_merely_full_says_so_in_placements(config):
    """The other side of the same rule. Every worker measured, the answer is
    short, and the refusal is a plain count — no hedging about telemetry,
    which would send someone looking for a broken agent that is not there."""
    placed, messages = await _place(config, _pd_model(2, 2), _fleet("A100x2", 1))

    assert placed is None
    assert messages[0] == (
        "The group needs 4 placements and the cluster has room for 2."
    )
    assert "could not be measured" not in messages[0]


@pytest.mark.asyncio
async def test_e_a_label_selector_matching_nothing_is_a_measured_zero(config):
    """A filter that excluded every worker measured the cluster; it did not
    fail to.

    From a live e2e run: a group whose `worker_selector` named a label no
    worker carries was refused with "the cluster has room for 0, but capacity
    could not be measured on 17 worker(s) — the shortfall may be smaller than
    it looks, or there may be none". Nothing was unmeasurable. The filters had
    answered, definitively, about every worker, and the hedge sent the reader
    looking for a broken agent instead of at the typo in their own selector —
    while the single-instance path, given the same selector, printed "Matched
    0/3 workers by label selector".

    So both halves are asserted. The wording must be the plain count, with no
    telemetry clause; and the refusal must carry the filters' own lines, which
    are the only place the *reason* for the zero is written down.
    """
    model = _pd_model(2, 2)
    model.worker_selector = {"worker-name": "no-such-worker-e2e"}

    placed, messages = await _place(config, model, _fleet("A100x2", 3))

    assert placed is None
    assert messages[0] == (
        "The group needs 4 placements and the cluster has room for 0."
    )
    assert "could not be measured" not in messages[0]
    # The note the single-instance path prints, now on this one too. Matched
    # as a prefix because the backend appends its own aside about Linux.
    assert any(
        "Matched 0/3 workers by label selector: "
        "{'worker-name': 'no-such-worker-e2e'}." in message
        for message in messages[1:]
    ), messages
    assert any(
        "Matched 3 workers by cluster selector." in message for message in messages[1:]
    ), messages


@pytest.mark.asyncio
async def test_e_an_empty_cluster_is_refused_with_the_filters_own_account(config):
    """The same rule as the test above, on the one path that had no role to ask
    about.

    A cluster with nothing in it reaches the solver as a domain with no
    workers, and that refusal was built without a role — so the caller asked
    `notes_for(None)`, which answers nothing by contract, and the operator got
    "room for 0" with no statement of what was counted. The count is identical
    for a fleet that was filtered away and one that was never joined, so the
    filter line is the only thing that tells the two apart.
    """
    placed, messages = await _place(config, _pd_model(2, 2), [])

    assert placed is None
    assert messages[0] == (
        "The group needs 4 placements and the cluster has room for 0."
    )
    assert any(
        "Matched 0 workers by cluster selector." in message for message in messages[1:]
    ), messages


@pytest.mark.asyncio
async def test_e_a_must_gather_with_no_domain_to_walk_names_the_filter_too(config):
    """The other unattributed refusal: `must` at a layer whose every domain is
    empty, so the walk never enters one and there is no best attempt to
    describe.

    The host rung is the case that survives an empty fleet — it is the one
    scope that exists whatever the operator declared, which is why a `must` on
    it is enforced rather than dropped as unknown, and with no worker there is
    nothing at it to walk into. The sentence is the solver's own; what was
    missing is the line underneath saying which fleet produced no domain.
    """
    model = _pd_model(
        2,
        2,
        gather=GatherSpec(strategy=GatherStrategyEnum.MUST_GATHER, layer=NODE_LAYER),
    )
    placed, messages = await _place(config, model, [])

    assert placed is None
    assert messages[0] == "No topology domain has any capacity for this group."
    assert any(
        "Matched 0 workers by cluster selector." in message for message in messages[1:]
    ), messages


# --- F. members wider than one machine -------------------------------------- #


@pytest.mark.asyncio
async def test_f_a_role_wider_than_one_machine_spans_several(config):
    """Thirty-two H100s and a 1P1D at `--tensor-parallel-size=16`: each member
    needs sixteen cards and no host has more than eight. Fitting this is a
    structural question, not a sizing one — the fleet holds VRAM to spare.
    `count_offer_slots` must offer the selector several workers at once,
    because the selectors' cross-node branch refuses a list shorter than two:
    offered one worker at a time, a member wider than a single machine could
    never be counted however much the cluster held.

    Each member lands on two hosts, and the two members take four hosts
    between them without overlapping.
    """
    model = _pd_model(1, 1, params=["--tensor-parallel-size=16"])
    placed, messages = await _place(config, model, _fleet("H100x8", 4))

    assert placed is not None, messages
    prefill_host, prefill_cards = placed["prefill"][0]
    decode_host, decode_cards = placed["decode"][0]
    # Whole machines: the selector takes every card of every host it combines.
    assert prefill_cards == list(range(8))
    assert decode_cards == list(range(8))
    assert prefill_host != decode_host


@pytest.mark.asyncio
async def test_f_a_member_that_spans_reports_every_machine_it_holds(config):
    """The half that the rest of the server reads. A spanning member records
    its other machines on `distributed_servers`, and four places in the server
    ask "where is this member" — the gather floor, the breach report, the
    locality sum, the proximity scorer. A candidate that came back without
    them would leave all four looking at half a placement."""
    model = _pd_model(1, 1, params=["--tensor-parallel-size=16"])
    by_instance, _messages = await _place(config, model, _fleet("H100x8", 4), raw=True)

    assert by_instance is not None
    spans = {
        len(
            [candidate.worker.id]
            + [s.worker_id for s in (candidate.subordinate_workers or [])]
        )
        for candidate in by_instance.values()
    }
    assert spans == {2}


@pytest.mark.asyncio
async def test_f_a_deployment_that_forbids_splitting_a_member_is_refused(config):
    """The switch is an answer, not a gap. A deployment that turned off
    cross-node inference has said its members may not be split, and the group
    is refused rather than quietly spanning anyway — the refusal is what sends
    the reader to that switch instead of to a bigger machine."""
    model = _pd_model(1, 1, params=["--tensor-parallel-size=16"])
    model.distributed_inference_across_workers = False
    placed, messages = await _place(config, model, _fleet("H100x8", 4))

    assert placed is None
    assert "room for 0" in messages[0]


@pytest.mark.asyncio
async def test_f_a_role_that_fits_one_machine_never_spans(config):
    """The compatibility guarantee, and the reason the cost of all this is
    zero for every deployment placed today: the wider search runs only when no
    single machine holds even one member."""
    placed, _messages = await _place(
        config,
        _pd_model(1, 1, params=["--tensor-parallel-size=4"]),
        _fleet("H100x8", 4),
    )

    assert placed is not None
    assert len(placed["prefill"][0][1]) == 4


# --- G. the fleet is one cluster's, not the whole server's ------------------ #


def _foreign(workers, cluster_id: int = 2):
    """Move these machines to another cluster, the way `Worker.all` finds them.

    `_fleet` stamps `cluster_id = 1` on everything because `ClusterFilter` is
    real here; this undoes it for the hosts a test wants present-but-forbidden.
    """
    for worker in workers:
        worker.cluster_id = cluster_id
    return workers


@pytest.mark.asyncio
async def test_g_the_tree_is_built_over_one_cluster_not_the_whole_fleet(config):
    """`schedule_group` is handed `Worker.all(session)` — every worker this
    server knows, in every cluster — and must narrow it before it builds
    anything.

    The two sibling paths already do: the per-instance topology read spells
    `[w for w in workers if w.cluster_id == model.cluster_id]`, and
    `evaluate_group` is only ever given one cluster's workers because its
    caller groups them first. This one did not, and an e2e run on a 3-worker
    cluster inside a 17-worker fleet was refused "capacity could not be
    measured on 17 worker(s)" — fourteen machines the deployment could never
    have used, and nothing an operator did to them could change the answer.

    The tree is the assertion because it is where the damage is: every foreign
    host becomes a domain of its own that the solver walks and pays a full
    selector sweep on, and the sweep can only ever answer zero — `ClusterFilter`
    sits at the head of the capacity chain. That last fact is also why the
    *refusal* below reads the same either way, so it is asserted too but it is
    not the thing under test: with filtered-out workers now counted as measured
    zeros (see section E), foreign hosts are invisible in the numbers and only
    the tree and the wasted sweeps give them away.

    `build_view` and `GroupCapacity` are wrapped, not replaced — both spies
    call straight through, so the placement below is still decided by the real
    tree and the real selectors, which is the rule this file is built on.
    """
    own = _fleet("A100x2", 1)
    foreign = _foreign(_fleet("H100x8", 2))
    seen: Dict[str, List[int]] = {}

    real_build_view = group_schedule.build_view
    real_capacity = group_schedule.GroupCapacity

    def spy_build_view(topology, workers):
        seen["tree"] = sorted(w.id for w in workers)
        return real_build_view(topology, workers)

    def spy_capacity(config_, model, workers, *args, **kwargs):
        seen["measured"] = sorted(w.id for w in workers)
        return real_capacity(config_, model, workers, *args, **kwargs)

    with (
        patch.object(group_schedule, "build_view", spy_build_view),
        patch.object(group_schedule, "GroupCapacity", spy_capacity),
    ):
        placed, messages = await _place(config, _pd_model(2, 2), own + foreign)

    # Host 8 is this model's cluster; 22 and 23 belong to another one.
    assert seen["tree"] == [8]
    assert seen["measured"] == [8]

    # And the counts are the single-cluster counts: two cards, four members.
    assert placed is None
    assert messages[0] == (
        "The group needs 4 placements and the cluster has room for 2."
    )
    assert "could not be measured" not in messages[0]
