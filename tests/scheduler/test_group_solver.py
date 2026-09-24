import pytest

from gpustack.scheduler.group_solver import (
    GatherRequest,
    GroupInfeasible,
    GroupPlacement,
    RoleDemand,
    solve_group_placement,
)
from gpustack.topology.tree import (
    NODE_LAYER,
    ROOT_LAYER,
    TopologyLayerSpec,
    build_topology,
    layer_names,
)
from types import SimpleNamespace

RACK = "topology.gpustack.ai/rack"


def worker(id_, name, rack=None):
    labels = {RACK: rack} if rack else {}
    return SimpleNamespace(id=id_, name=name, labels=labels)


def rack_layer():
    return [TopologyLayerSpec(layer="RackLayer", label_keys=[RACK])]


def tree(workers, specs=None):
    specs = rack_layer() if specs is None else specs
    return build_topology(specs, workers), layer_names(specs)


def flat_capacity(per_worker):
    """Every worker has the same room for every role, minus what is committed."""

    async def capacity(_role, worker_ids, placed):
        used = {}
        for entry in placed:
            used[entry.worker_id] = used.get(entry.worker_id, 0) + 1
        return {w: max(0, per_worker - used.get(w, 0)) for w in worker_ids}

    return capacity


def pd(prefill=1, decode=1):
    return [
        RoleDemand(role="prefill", replicas=prefill, weight=2.0),
        RoleDemand(role="decode", replicas=decode, weight=1.0),
    ]


# --- the tightest layer wins ----------------------------------------------- #


@pytest.mark.asyncio
async def test_a_group_that_fits_one_host_is_placed_on_one_host():
    """Leaf-to-root: the tightest domain holding the whole group wins, and no
    wider layer is even considered."""
    root, layers = tree([worker(1, "w1", "rack-a"), worker(2, "w2", "rack-a")])

    got = await solve_group_placement(root, pd(), flat_capacity(4), layers)

    assert isinstance(got, GroupPlacement)
    assert got.layer == NODE_LAYER
    assert len(set(got.worker_ids())) == 1


@pytest.mark.asyncio
async def test_a_group_too_big_for_one_host_widens_to_the_rack():
    root, layers = tree([worker(1, "w1", "rack-a"), worker(2, "w2", "rack-a")])

    got = await solve_group_placement(root, pd(2, 2), flat_capacity(2), layers)

    assert isinstance(got, GroupPlacement)
    assert got.layer == "RackLayer"
    assert got.domain == "rack-a"
    assert sorted(got.worker_ids()) == [1, 1, 2, 2]


@pytest.mark.asyncio
async def test_it_widens_only_as_far_as_it_has_to():
    """Two racks, and a group that fits in one. Widening to the cluster root
    would also "work" and would be the wrong answer."""
    root, layers = tree(
        [
            worker(1, "w1", "rack-a"),
            worker(2, "w2", "rack-a"),
            worker(3, "w3", "rack-b"),
        ]
    )

    got = await solve_group_placement(root, pd(2, 2), flat_capacity(2), layers)

    assert got.layer == "RackLayer"
    assert got.domain == "rack-a"


# --- the two opposite sort directions -------------------------------------- #


@pytest.mark.asyncio
async def test_between_domains_the_tightest_that_fits_wins():
    """Binpack across domains: the group takes the smaller rack and leaves the
    bigger one whole for whoever needs it next.

    Every worker holds one member, so the group cannot fit a single host and
    the rack layer is genuinely the one deciding — otherwise this asserts
    nothing about domain ordering."""
    workers = [
        worker(1, "s1", "small"),
        worker(2, "s2", "small"),
        worker(3, "b1", "big"),
        worker(4, "b2", "big"),
        worker(5, "b3", "big"),
        worker(6, "b4", "big"),
    ]
    root, layers = tree(workers)

    got = await solve_group_placement(root, pd(1, 1), flat_capacity(1), layers)

    assert got.layer == "RackLayer"
    assert got.domain == "small"


@pytest.mark.asyncio
async def test_a_group_that_fits_the_roomiest_host_goes_there_whole():
    """The leaf layer is searched first, so a group small enough for one host
    never reaches the rack-level distribution at all — compactness comes from
    the layer walk, not from how a domain's workers are filled."""

    async def capacity(_role, worker_ids, placed):
        used = {}
        for entry in placed:
            used[entry.worker_id] = used.get(entry.worker_id, 0) + 1
        room = {1: 1, 2: 4}
        return {w: max(0, room[w] - used.get(w, 0)) for w in worker_ids}

    root, layers = tree([worker(1, "small", "rack-a"), worker(2, "roomy", "rack-a")])

    got = await solve_group_placement(root, pd(2, 1), capacity, layers)

    assert got.layer == NODE_LAYER
    assert set(got.worker_ids()) == {2}


@pytest.mark.asyncio
async def test_inside_a_domain_the_roomiest_worker_gets_the_larger_share():
    """Round-robin, but starting from the roomiest — so when the members do not
    divide evenly, the spare goes where there is most room.

    Five members over rooms of 3 and 4 — bigger than either host, so the rack
    layer really is deciding, and with slack left over so the order of the
    round decides who carries the spare. Roomiest-first gives 3/2; starting
    from the tightest gives 2/3, which is the mutation this has to catch."""

    async def capacity(_role, worker_ids, placed):
        used = {}
        for entry in placed:
            used[entry.worker_id] = used.get(entry.worker_id, 0) + 1
        room = {1: 3, 2: 4}
        return {w: max(0, room[w] - used.get(w, 0)) for w in worker_ids}

    root, layers = tree([worker(1, "small", "rack-a"), worker(2, "roomy", "rack-a")])

    got = await solve_group_placement(root, pd(3, 2), capacity, layers)

    assert got.layer == "RackLayer"
    assert sum(1 for w in got.worker_ids() if w == 2) == 3
    assert sum(1 for w in got.worker_ids() if w == 1) == 2


# --- gather ---------------------------------------------------------------- #


ZONE = "topology.kubernetes.io/zone"


def zone_rack_layers():
    return [
        TopologyLayerSpec(layer="ZoneLayer", label_keys=[ZONE]),
        TopologyLayerSpec(
            layer="RackLayer", label_keys=[RACK], parent_layer="ZoneLayer"
        ),
    ]


def zoned(id_, name, zone, rack):
    return SimpleNamespace(id=id_, name=name, labels={ZONE: zone, RACK: rack})


@pytest.mark.asyncio
async def test_must_gather_stops_the_walk_at_the_named_layer():
    """The whole of MustGather. Three layers on purpose: the group fits the
    zone but no rack, so a solver that ignored the ceiling would widen one step
    and succeed — which is exactly the outcome the operator asked not to get.
    A two-layer fixture cannot tell the two apart."""
    workers = [
        zoned(1, "w1", "z1", "rack-a"),
        zoned(2, "w2", "z1", "rack-a"),
        zoned(3, "w3", "z1", "rack-b"),
        zoned(4, "w4", "z1", "rack-b"),
    ]
    root, layers = tree(workers, zone_rack_layers())
    assert layers == ["ZoneLayer", "RackLayer", NODE_LAYER]

    loose = await solve_group_placement(root, pd(3, 3), flat_capacity(2), layers)
    assert isinstance(loose, GroupPlacement)
    assert loose.layer == "ZoneLayer"

    strict = await solve_group_placement(
        root,
        pd(3, 3),
        flat_capacity(2),
        layers,
        GatherRequest(layer="RackLayer", must=True),
    )

    assert isinstance(strict, GroupInfeasible)
    assert "RackLayer" in strict.reason and "6" in strict.reason


@pytest.mark.asyncio
async def test_must_gather_does_not_fall_back_to_the_cluster_root_either():
    """The other half of "MustGather is a hard floor", and the one a reader
    is likelier to get wrong.

    Stopping the *layer* walk at the named rung is not enough: the cluster root
    is not one of the layers, it is a fallback below the loop, and it always
    fits. Reaching it would turn every refusal into a silent placement
    somewhere in the cluster — the exact outcome `MustGather` is bought to
    prevent. Here the two workers share no rack, so the root is the only domain
    that could hold the group, and the requirement has to refuse instead.
    """
    workers = [worker(1, "w1", "rack-a"), worker(2, "w2", "rack-b")]
    root, layers = tree(workers)

    loose = await solve_group_placement(root, pd(2, 2), flat_capacity(2), layers)
    assert isinstance(loose, GroupPlacement)
    assert loose.layer == ROOT_LAYER, "the fixture must force the root fallback"

    strict = await solve_group_placement(
        root,
        pd(2, 2),
        flat_capacity(2),
        layers,
        GatherRequest(layer="RackLayer", must=True),
    )
    assert isinstance(strict, GroupInfeasible)


@pytest.mark.asyncio
async def test_without_must_gather_the_same_group_is_placed_across_racks():
    root, layers = tree(
        [
            worker(1, "w1", "rack-a"),
            worker(2, "w2", "rack-a"),
            worker(3, "w3", "rack-b"),
            worker(4, "w4", "rack-b"),
        ]
    )

    got = await solve_group_placement(root, pd(3, 3), flat_capacity(2), layers)

    assert isinstance(got, GroupPlacement)
    assert len(got.worker_ids()) == 6


@pytest.mark.asyncio
async def test_the_refusal_says_how_much_room_the_roomiest_domain_had():
    """ "It does not fit" is not actionable; "the roomiest rack holds 4" is."""
    root, layers = tree([worker(1, "w1", "rack-a"), worker(2, "w2", "rack-a")])

    got = await solve_group_placement(
        root,
        pd(4, 4),
        flat_capacity(2),
        layers,
        GatherRequest(layer="RackLayer", must=True),
    )

    assert isinstance(got, GroupInfeasible)
    assert got.needed == 8
    assert got.available == 4
    assert got.best_domain == "rack-a"


@pytest.mark.asyncio
async def test_a_gather_layer_that_no_longer_exists_does_not_block_scheduling():
    """A layer renamed or removed after the model was saved. Refusing here
    would take a running deployment down for an edit made somewhere else."""
    root, layers = tree([worker(1, "w1", "rack-a")])

    got = await solve_group_placement(
        root,
        pd(1, 1),
        flat_capacity(4),
        layers,
        GatherRequest(layer="LayerThatWasDeleted", must=True),
    )

    assert isinstance(got, GroupPlacement)


@pytest.mark.asyncio
async def test_an_unknown_gather_layer_does_not_disable_the_root_fallback():
    """Found on a live cluster, not here — the version above passes either way,
    because its group fits at the rack layer and never reaches the fallback.

    The workers sit in different zones *and* different racks, so the cluster
    root is the only domain holding both. Dropping the unknown requirement from
    the ceiling but not from the fallback leaves the group refused in the name
    of a layer the code has just logged that it is ignoring."""
    workers = [zoned(1, "w1", "z1", "rack-a"), zoned(2, "w2", "z2", "rack-b")]
    root, layers = tree(workers, zone_rack_layers())

    loose = await solve_group_placement(root, pd(2, 2), flat_capacity(2), layers)
    assert isinstance(loose, GroupPlacement)
    assert loose.layer == ROOT_LAYER, "the fixture must force the root fallback"

    got = await solve_group_placement(
        root,
        pd(2, 2),
        flat_capacity(2),
        layers,
        GatherRequest(layer="SuperPodLayer", must=True),
    )

    assert isinstance(got, GroupPlacement)
    assert got.layer == ROOT_LAYER


# --- the unclassified bucket ----------------------------------------------- #


@pytest.mark.asyncio
async def test_unlabelled_workers_are_not_a_domain_to_gather_into():
    """They are the workers whose position is *unknown*. Gathering into that
    bucket would claim they are together on the strength of them all being
    unlabelled — and the distance function already says they are not."""
    root, layers = tree([worker(1, "w1"), worker(2, "w2")])

    got = await solve_group_placement(
        root,
        pd(1, 1),
        flat_capacity(1),
        layers,
        GatherRequest(layer="RackLayer", must=True),
    )

    assert isinstance(got, GroupInfeasible)


@pytest.mark.asyncio
async def test_unlabelled_workers_are_still_schedulable_without_a_requirement():
    """Not gatherable is not unusable: the leaf layer is per-worker and always
    real, so a group that fits on one host still lands."""
    root, layers = tree([worker(1, "w1"), worker(2, "w2")])

    got = await solve_group_placement(root, pd(1, 1), flat_capacity(4), layers)

    assert isinstance(got, GroupPlacement)
    assert got.layer == NODE_LAYER


# --- role order and role mixing -------------------------------------------- #


@pytest.mark.asyncio
async def test_the_hungriest_role_is_placed_first():
    """First-fit-decreasing. A role needing whole cards cannot use what a role
    taking slices left behind; the reverse usually works."""
    seen = []

    async def capacity(role, worker_ids, placed):
        seen.append(role)
        used = {}
        for entry in placed:
            used[entry.worker_id] = used.get(entry.worker_id, 0) + 1
        return {w: max(0, 4 - used.get(w, 0)) for w in worker_ids}

    root, layers = tree([worker(1, "w1", "rack-a")])
    await solve_group_placement(
        root,
        [
            RoleDemand(role="light", replicas=1, weight=1.0),
            RoleDemand(role="heavy", replicas=1, weight=9.0),
        ],
        capacity,
        layers,
    )

    assert seen[0] == "heavy"


@pytest.mark.asyncio
async def test_equal_room_prefers_the_worker_carrying_fewer_of_the_group():
    """Placing role by role, roomiest-first, otherwise packs all of one role
    onto the first workers and all of the next onto the rest. With a router
    that pairs prefill and decode independently, an all-P/all-D split is the
    one arrangement where no pair is local."""
    root, layers = tree([worker(1, "w1", "rack-a"), worker(2, "w2", "rack-a")])

    got = await solve_group_placement(root, pd(2, 2), flat_capacity(2), layers)

    # Each worker ends up with one prefill and one decode, not two of one kind.
    assert sorted(got.assignments["prefill"]) == [1, 2]
    assert sorted(got.assignments["decode"]) == [1, 2]


# --- degenerate inputs ----------------------------------------------------- #


@pytest.mark.asyncio
async def test_an_empty_group_is_trivially_placed():
    root, layers = tree([worker(1, "w1", "rack-a")])

    got = await solve_group_placement(root, [], flat_capacity(4), layers)

    assert isinstance(got, GroupPlacement)
    assert got.worker_ids() == []


@pytest.mark.asyncio
async def test_a_cluster_with_no_capacity_at_all_says_so():
    root, layers = tree([worker(1, "w1", "rack-a")])

    got = await solve_group_placement(root, pd(1, 1), flat_capacity(0), layers)

    assert isinstance(got, GroupInfeasible)


@pytest.mark.asyncio
async def test_the_plan_is_stable_across_re_solves():
    """An unchanged cluster must produce an unchanged plan, or every reconcile
    looks like a spec change to anything comparing placements."""
    root, layers = tree([worker(i, f"w{i}", "rack-a") for i in range(1, 5)])

    first = await solve_group_placement(root, pd(2, 2), flat_capacity(2), layers)
    second = await solve_group_placement(root, pd(2, 2), flat_capacity(2), layers)

    assert first.assignments == second.assignments


# --- what the refusal is allowed to claim ---------------------------------- #


@pytest.mark.asyncio
async def test_a_worker_that_could_not_be_measured_is_not_reported_as_full():
    """A plain misconfiguration once produced zero on every worker and a
    refusal that named capacity — the one answer that stops an operator looking
    for a mistake. Absent from the mapping means unknown; present-and-zero
    means measured and full."""
    root, layers = tree([worker(1, "w1", "rack-a"), worker(2, "w2", "rack-a")])

    async def nothing_measurable(_role, _worker_ids, _placed):
        return {}

    got = await solve_group_placement(
        root,
        pd(1, 1),
        nothing_measurable,
        layers,
        GatherRequest(layer="RackLayer", must=True),
    )

    assert isinstance(got, GroupInfeasible)
    assert got.unmeasured == 2
    assert "could not be measured" in got.reason
    assert "holds 0" not in got.reason


@pytest.mark.asyncio
async def test_a_genuinely_full_cluster_still_says_so():
    """The other side of the same rule: measured and zero is a capacity
    verdict, and must keep reading like one."""
    root, layers = tree([worker(1, "w1", "rack-a")])

    got = await solve_group_placement(
        root,
        pd(1, 1),
        flat_capacity(0),
        layers,
        GatherRequest(layer="RackLayer", must=True),
    )

    assert isinstance(got, GroupInfeasible)
    assert got.unmeasured == 0
    assert "holds 0" in got.reason


@pytest.mark.asyncio
async def test_domains_are_sized_with_one_capacity_pass_not_one_each():
    """Sizing asks the same question of the same workers at every layer, and
    each ask is a full selector sweep in production. One pass for the tree."""
    # Counted in total, not by shape: sizing per domain would *add* calls
    # rather than change the one the tree-wide pass makes, so a predicate that
    # only recognises the tree-wide call cannot see the difference.
    calls = []

    async def counting(role, worker_ids, placed):
        calls.append((role, tuple(sorted(worker_ids)), len(placed)))
        return {w: 4 for w in worker_ids}

    workers = [
        zoned(1, "w1", "z1", "rack-a"),
        zoned(2, "w2", "z1", "rack-b"),
        zoned(3, "w3", "z2", "rack-c"),
    ]
    root, layers = tree(workers, zone_rack_layers())

    await solve_group_placement(root, pd(1, 1), counting, layers)

    # One tree-wide sizing pass, then one call per role placing into the
    # winning leaf. Sizing each of the three leaf domains separately would add
    # three more.
    assert calls[0] == ("prefill", (1, 2, 3), 0), "the sizing pass comes first"
    assert len(calls) == 3, f"expected 1 sizing + 2 placements, got {calls}"


@pytest.mark.asyncio
async def test_a_refusal_names_the_workers_it_could_not_look_at():
    """The sentence an operator gets when one worker stopped reporting.

    Only the root sees every worker, so it is the only domain that can say
    "we could not measure one of them"; a single host, asked about itself,
    can only ever say "not enough room". The two tie on measured room -- the
    root's total IS the reachable host's -- and with a strict `>` the host won
    and the unmeasured worker vanished from the message. A fleet with an agent
    that had stopped reporting then read as simply full, which sends the
    reader to free up memory instead of to that agent.
    """

    async def capacity(_role, worker_ids, placed):
        # Worker 2 is absent rather than zero: this module's way of saying
        # "unknown", and what a host with no system telemetry produces.
        used = {}
        for entry in placed:
            used[entry.worker_id] = used.get(entry.worker_id, 0) + 1
        return {w: max(0, 1 - used.get(w, 0)) for w in worker_ids if w != 2}

    root, layers = tree([worker(1, "w1", "rack-a"), worker(2, "w2", "rack-b")])

    got = await solve_group_placement(root, pd(2, 2), capacity, layers, GatherRequest())

    assert isinstance(got, GroupInfeasible)
    assert got.unmeasured == 1
    assert "could not be measured" in got.reason


@pytest.mark.asyncio
async def test_a_refusal_without_a_floor_still_carries_the_numbers():
    """The default path: a refusal with no gather requirement behind it.

    `needed` / `available` are filled in on every refusal, and the reason has
    to spend them on a sentence here too. A deployment with no gather
    requirement is most of them, and a bare "not enough room" leaves the
    shortfall sitting unread in the fields beside it, where the
    single-instance refusal names the claim and what the roomiest worker had.
    """
    root, layers = tree([worker(1, "w1", "rack-a"), worker(2, "w2", "rack-a")])

    got = await solve_group_placement(
        root, pd(4, 4), flat_capacity(1), layers, GatherRequest()
    )

    assert isinstance(got, GroupInfeasible)
    assert got.needed == 8
    assert got.available == 2
    assert "8" in got.reason and "2" in got.reason
    # No floor was asked for, so the search ran to the cluster root and the
    # sentence must not name a domain whose only name is an internal layer id.
    assert "ClusterTopologyLayer" not in got.reason


@pytest.mark.asyncio
async def test_a_floorless_refusal_separates_short_from_unmeasured():
    """ "Full" is the one answer that stops an operator looking for a mistake,
    and without a floor the mistake is usually a worker that stopped reporting
    rather than a cluster out of cards."""

    async def capacity(_role, worker_ids, placed):
        used = {}
        for entry in placed:
            used[entry.worker_id] = used.get(entry.worker_id, 0) + 1
        return {w: max(0, 1 - used.get(w, 0)) for w in worker_ids if w != 2}

    root, layers = tree([worker(1, "w1", "rack-a"), worker(2, "w2", "rack-b")])

    got = await solve_group_placement(root, pd(2, 2), capacity, layers, GatherRequest())

    assert isinstance(got, GroupInfeasible)
    assert got.needed == 4
    assert got.unmeasured == 1
    assert "could not be measured" in got.reason
    # Both halves, because either alone points somewhere wrong.
    assert "4" in got.reason


# --- the router: checked, never assigned ------------------------------------ #


def _router(replicas=1):
    return [RoleDemand(role="router", replicas=replicas, weight=0.0)]


def _capacity_with_router(per_worker, router_on):
    """GPU roles get `per_worker` slots each; the router only fits on the
    workers named in `router_on`.

    `per_worker` may be an int (the same everywhere) or a dict keyed by worker
    id, which is how a test forces the gang onto one particular host.
    """

    def gpu_slots(worker_id):
        if isinstance(per_worker, dict):
            return per_worker.get(worker_id, 0)
        return per_worker

    async def capacity(role, worker_ids, placed):
        if role == "router":
            return {w: (1 if w in router_on else 0) for w in worker_ids}
        used = {}
        for entry in placed:
            if getattr(entry, "role", "") != "router":
                used[entry.worker_id] = used.get(entry.worker_id, 0) + 1
        return {w: max(0, gpu_slots(w) - used.get(w, 0)) for w in worker_ids}

    return capacity


@pytest.mark.asyncio
async def test_a_group_whose_router_has_nowhere_to_go_is_refused():
    """The gap this closes. The router takes no accelerator, so it is not a
    demand and the solver never saw it — and a group admitted onto hardware
    with no room for its router never becomes servable, because the router is
    what answers requests. `evaluate_group` said so in its own docstring: "a
    cluster with room for the GPU members but not for the router still
    evaluates as compatible"."""
    root, layers = tree([worker(1, "w1", "rack-a")])

    got = await solve_group_placement(
        root,
        pd(1, 1),
        _capacity_with_router(2, router_on=set()),
        layers,
        GatherRequest(),
        attendants=_router(),
    )

    assert isinstance(got, GroupInfeasible)
    assert got.role == "router"
    assert "'router'" in got.reason
    # The gang fit, so the counting sentence must not be pasted over this one:
    # "the cluster has room for 2" of 2 would contradict the refusal it is
    # attached to.
    assert "placements" not in got.reason


@pytest.mark.asyncio
async def test_a_router_that_fits_does_not_change_the_placement():
    """It is checked, not assigned: the row does not exist yet — the
    dependency gate creates it a pass later, from peer addresses that are not
    known while this runs."""
    root, layers = tree([worker(1, "w1", "rack-a")])

    got = await solve_group_placement(
        root,
        pd(1, 1),
        _capacity_with_router(2, router_on={1}),
        layers,
        GatherRequest(),
        attendants=_router(),
    )

    assert isinstance(got, GroupPlacement)
    assert "router" not in got.assignments


@pytest.mark.asyncio
async def test_the_router_does_not_inflate_what_the_group_needs():
    """Counting it among the demands would make a 4P4D need nine placements in
    one domain and refuse racks that would have served. It constrains
    feasibility, never size."""
    root, layers = tree([worker(1, "w1", "rack-a")])

    got = await solve_group_placement(
        root,
        pd(4, 4),
        _capacity_with_router(1, router_on={1}),
        layers,
        GatherRequest(),
        attendants=_router(),
    )

    assert isinstance(got, GroupInfeasible)
    assert got.needed == 8  # not 9
    assert got.role != "router"


@pytest.mark.asyncio
async def test_without_a_floor_the_router_may_live_anywhere_in_the_cluster():
    """A tight domain must not be rejected over a router that had somewhere
    else to go: no gather requirement means no claim about where members sit,
    so refusing here would spread the gang for nothing."""
    root, layers = tree([worker(1, "w1", "rack-a"), worker(2, "w2", "rack-b")])

    got = await solve_group_placement(
        root,
        pd(1, 1),
        _capacity_with_router({1: 2, 2: 0}, router_on={2}),
        layers,
        GatherRequest(),
        attendants=_router(),
    )

    assert isinstance(got, GroupPlacement)
    assert set(got.assignments["prefill"]) == {1}


@pytest.mark.asyncio
async def test_must_gather_keeps_the_router_inside_the_floor():
    """The operator set a floor, and a router away from its members crosses it
    on every request — it answers all of them.

    The gang can only fit on w1, so the floor puts it in rack-a; the router
    only fits on w2, in rack-b. Without the floor this is the previous test
    and it deploys.
    """
    root, layers = tree([worker(1, "w1", "rack-a"), worker(2, "w2", "rack-b")])

    got = await solve_group_placement(
        root,
        pd(1, 1),
        _capacity_with_router({1: 2, 2: 0}, router_on={2}),
        layers,
        GatherRequest(layer="RackLayer", must=True),
        attendants=_router(),
    )

    assert isinstance(got, GroupInfeasible)
    assert got.role == "router"


@pytest.mark.asyncio
async def test_a_group_with_no_router_is_unaffected():
    """Every pre-PD deployment and every group whose router role is absent
    takes exactly the path it took before."""
    root, layers = tree([worker(1, "w1", "rack-a")])

    got = await solve_group_placement(
        root, pd(1, 1), _capacity_with_router(2, router_on=set()), layers
    )

    assert isinstance(got, GroupPlacement)


@pytest.mark.asyncio
async def test_a_cluster_wide_router_refusal_does_not_name_an_internal_layer():
    """The root's own name is `ClusterTopologyLayer`, a layer id nobody
    outside the scheduler has seen. The wording branch keyed off an identity
    comparison between two lists, and the root fallback did not pass its
    workers -- so a refusal that had searched the whole cluster reported it as
    a domain, by that name."""
    root, layers = tree([worker(1, "w1", "rack-a")])

    got = await solve_group_placement(
        root,
        pd(1, 1),
        _capacity_with_router(2, router_on=set()),
        layers,
        GatherRequest(),
        attendants=_router(),
    )

    assert isinstance(got, GroupInfeasible)
    assert "this cluster" in got.reason
    assert "ClusterTopologyLayer" not in got.reason


@pytest.mark.asyncio
async def test_a_floored_router_refusal_still_names_the_domain():
    """The other half: under `MustGather` the search really was confined to one
    domain, and naming it is the whole value of the message."""
    root, layers = tree([worker(1, "w1", "rack-a"), worker(2, "w2", "rack-b")])

    got = await solve_group_placement(
        root,
        pd(1, 1),
        _capacity_with_router({1: 2, 2: 0}, router_on={2}),
        layers,
        GatherRequest(layer="RackLayer", must=True),
        attendants=_router(),
    )

    assert isinstance(got, GroupInfeasible)
    assert "rack-a" in got.reason


@pytest.mark.asyncio
async def test_a_preferred_layer_changes_no_placement_because_the_walk_is_already_tightest_first():
    """Why `_enforced_gather` may drop a `prefer`'s layer.

    Dropping a value the operator chose reads like a bug until you check what
    the layer is for. It is a CEILING — the walk stops there instead of
    widening to the cluster root — and refusing above it is the whole of what
    it buys. `prefer` says do not refuse, so the ceiling has nothing to do.

    What makes that safe is the walk, not the gather: it goes tightest rung
    first and returns the first domain that fits, so it is already doing
    «as close as possible». This asserts the equality rather than the
    reasoning — three different preferred rungs and no requirement at all must
    all put the group in the same place, or the layer was load-bearing after
    all and dropping it was a bug.
    """
    workers = [
        zoned(1, "w1", "z1", "rack-a"),
        zoned(2, "w2", "z1", "rack-a"),
        zoned(3, "w3", "z1", "rack-b"),
    ]
    root, layers = tree(workers, zone_rack_layers())

    asks = [
        GatherRequest(),
        GatherRequest(layer=NODE_LAYER, must=False),
        GatherRequest(layer="RackLayer", must=False),
        GatherRequest(layer="ZoneLayer", must=False),
    ]
    placements = []
    for ask in asks:
        got = await solve_group_placement(root, pd(2, 2), flat_capacity(2), layers, ask)
        assert isinstance(got, GroupPlacement)
        placements.append((got.layer, got.domain, sorted(got.assignments.items())))

    assert len(set(map(str, placements))) == 1, placements
    # And it is the tightest rung that fits, not the widest — otherwise the
    # equality above would hold for an uninteresting reason.
    assert placements[0][0] == "RackLayer"


@pytest.mark.asyncio
async def test_a_preferred_floor_never_refuses_what_must_gather_would():
    """The same fixture that makes `MustGather(rack)` refuse, asked as a
    preference: it has to place. This is the half a future «let prefer use its
    layer too» change would break first."""
    workers = [worker(1, "w1", "rack-a"), worker(2, "w2", "rack-b")]
    root, layers = tree(workers)

    refused = await solve_group_placement(
        root, pd(2, 2), flat_capacity(2), layers, GatherRequest("RackLayer", must=True)
    )
    assert isinstance(refused, GroupInfeasible)

    placed = await solve_group_placement(
        root, pd(2, 2), flat_capacity(2), layers, GatherRequest("RackLayer", must=False)
    )
    assert isinstance(placed, GroupPlacement)
    assert placed.layer == ROOT_LAYER


# --- ranking the fitting domains of one layer ------------------------------ #


def _capacity_by_worker(slots):
    """Room per worker, fixed, minus what this solve has committed."""

    async def capacity(_role, worker_ids, placed):
        used = {}
        for entry in placed:
            used[entry.worker_id] = used.get(entry.worker_id, 0) + 1
        return {w: max(0, slots.get(w, 0) - used.get(w, 0)) for w in worker_ids}

    return capacity


@pytest.mark.asyncio
async def test_every_fitting_domain_of_the_winning_layer_is_offered_to_the_score():
    """Ranking needs more than one candidate, and the walk stops as soon as it
    has them: a domain that fits is not a reason to stop looking within its
    own layer."""
    root, layers = tree(
        [worker(i, f"w{i}", "rack-a") for i in (1, 2, 3)],
    )
    seen = []

    got = await solve_group_placement(
        root,
        pd(),
        flat_capacity(4),
        layers,
        score=lambda p: seen.append(sorted(p.worker_ids())) or 0.0,
    )

    assert isinstance(got, GroupPlacement)
    # Three single-host domains, each holding the group on its own.
    assert sorted(seen) == [[1, 1], [2, 2], [3, 3]]


@pytest.mark.asyncio
async def test_the_highest_scoring_domain_wins_not_the_smallest():
    """The whole point of ranking. Without it the smallest fitting domain wins
    by arriving first, which is the answer only when nothing else is known."""
    root, layers = tree([worker(i, f"w{i}", "rack-a") for i in (1, 2, 3)])

    got = await solve_group_placement(
        root,
        pd(),
        flat_capacity(4),
        layers,
        score=lambda p: 1.0 if 3 in p.worker_ids() else 0.0,
    )

    assert isinstance(got, GroupPlacement)
    assert set(got.worker_ids()) == {3}


@pytest.mark.asyncio
async def test_a_looser_layer_is_never_ranked_against_a_tighter_one():
    """Holding the layer fixed is what removes the exchange rate nobody can
    set. A rack that would score higher must not beat a host that fits."""
    root, layers = tree([worker(1, "w1", "rack-a"), worker(2, "w2", "rack-b")])

    got = await solve_group_placement(
        root,
        pd(),
        flat_capacity(4),
        layers,
        # Rack-level placements score far higher, and must still lose.
        score=lambda p: 100.0 if p.layer == "RackLayer" else 1.0,
    )

    assert isinstance(got, GroupPlacement)
    assert got.layer == NODE_LAYER


@pytest.mark.asyncio
async def test_without_a_score_the_first_fitting_domain_still_wins():
    """The no-policy path, unchanged: nothing to rank by means the smallest
    fitting domain at the tightest layer, found first and returned."""
    root, layers = tree([worker(i, f"w{i}", "rack-a") for i in (1, 2, 3)])
    calls = []

    async def counting(role, worker_ids, placed):
        calls.append(tuple(worker_ids))
        return {w: 4 for w in worker_ids}

    got = await solve_group_placement(root, pd(), counting, layers)

    assert isinstance(got, GroupPlacement)
    # One host examined, not three: no ranking means no reason to look on.
    assert len([c for c in calls if len(c) == 1]) == 2  # two roles, one host


@pytest.mark.asyncio
async def test_the_candidate_limit_caps_how_wide_a_layer_is_explored():
    root, layers = tree([worker(i, f"w{i}", "rack-a") for i in range(1, 9)])
    seen = []

    await solve_group_placement(
        root,
        pd(),
        flat_capacity(4),
        layers,
        score=lambda p: seen.append(p.domain) or 0.0,
        limit=3,
    )

    assert len(seen) == 3


@pytest.mark.asyncio
async def test_a_domain_the_sizing_pass_proved_empty_is_not_examined():
    """Skipping it saves a selector sweep per role. Proven is the operative
    word -- an unmeasured worker is not proof of anything."""
    root, layers = tree([worker(1, "w1", "rack-a"), worker(2, "w2", "rack-a")])
    examined = []

    async def capacity(role, worker_ids, placed):
        if len(worker_ids) == 1:
            examined.append(worker_ids[0])
        used = {}
        for entry in placed:
            used[entry.worker_id] = used.get(entry.worker_id, 0) + 1
        room = {1: 0, 2: 4}
        return {w: max(0, room[w] - used.get(w, 0)) for w in worker_ids}

    got = await solve_group_placement(root, pd(), capacity, layers, score=lambda p: 0.0)

    assert isinstance(got, GroupPlacement)
    assert set(got.worker_ids()) == {2}
    assert 1 not in examined


# --- the caller's worker preference ---------------------------------------- #


@pytest.mark.asyncio
async def test_preference_breaks_a_tie_that_room_and_balance_left_open():
    root, layers = tree([worker(1, "w1", "rack-a"), worker(2, "w2", "rack-a")])

    got = await solve_group_placement(
        root,
        [RoleDemand(role="prefill", replicas=1, weight=1.0)],
        flat_capacity(4),
        layers,
        # Both hosts are identical, so only the preference can separate them.
        # Domains are single hosts here, so this decides within the chosen one.
        preference={2: 0, 1: 1},
        score=lambda p: 0.0,
    )

    assert isinstance(got, GroupPlacement)


def test_a_worker_outside_the_preference_map_ranks_below_every_worker_in_it():
    """The shape the scheduler actually passes.

    It maps every warm worker to 0 — the map says "these are warm", not how
    they order among themselves — so a default of 0 for an absent worker made
    warm and cold tie and dropped the preference entirely. Nothing looked
    wrong because the between-domain score still counted file locality, and
    the existing tie-break test maps both workers and so never exercised the
    absent case.

    Asserted on `_share_out` directly: the preference only reaches it once a
    group has failed to fit on one host, which a placement-level test would
    have to build a whole spanning cluster to reach.
    """
    from gpustack.scheduler.group_solver import _share_out

    # Identical room and nothing placed, so only the preference can separate
    # them — and the warm one is the HIGHER id on purpose: with the buggy
    # default both workers tied here and the final `worker_id` tie-break
    # handed it to worker 1, which is the right answer for the wrong reason.
    got = _share_out({1: 1, 2: 1}, replicas=1, preference={2: 0})

    assert got == [(2, 1)], got


@pytest.mark.asyncio
async def test_preference_never_outranks_room():
    """A rank map orders what capacity has already tied. Letting it move a
    member onto a fuller worker would make it a capacity decision."""
    root, layers = tree([worker(1, "w1", "rack-a"), worker(2, "w2", "rack-a")])

    got = await solve_group_placement(
        root,
        pd(2, 2),
        # Neither host holds all four, so the rack wins and both are used --
        # which is what makes the split between them observable at all.
        _capacity_by_worker({1: 1, 2: 3}),
        layers,
        preference={1: 0, 2: 1},
        score=lambda p: 0.0,
    )

    assert isinstance(got, GroupPlacement)
    assert got.layer == "RackLayer"
    # Room decides the split: three members where there is room for three, one
    # where there is room for one. The preference for worker 1 moves nothing.
    assert got.worker_ids().count(2) == 3
    assert got.worker_ids().count(1) == 1


@pytest.mark.asyncio
async def test_a_placement_that_splits_a_member_loses_to_one_that_does_not():
    """Narrowed by removal before anything is scored. A member spread over two
    machines pays an all-reduce per layer per token, which no bounded score
    can be trusted to outweigh -- the pairing term it competes with is
    unbounded in the per-instance path for exactly this reason."""
    root, layers = tree([worker(i, f"w{i}", "rack-a") for i in (1, 2)])

    class Capacity:
        """Both hosts fit the group; only host 1 has to split a member."""

        async def __call__(self, _role, worker_ids, placed):
            used = {}
            for entry in placed:
                used[entry.worker_id] = used.get(entry.worker_id, 0) + 1
            return {w: max(0, 4 - used.get(w, 0)) for w in worker_ids}

        def spans_for(self, _role, worker_id):
            return [1, 2] if worker_id == 1 else [worker_id]

    got = await solve_group_placement(
        root,
        pd(),
        Capacity(),
        layers,
        # Host 1 scores far higher and must still lose: the split is settled
        # before the score is consulted at all.
        score=lambda p: 100.0 if 1 in p.worker_ids() else 1.0,
    )

    assert isinstance(got, GroupPlacement)
    assert set(got.worker_ids()) == {2}


@pytest.mark.asyncio
async def test_a_role_wider_than_any_machine_still_places():
    """The narrowing drops split placements only while a whole one exists.
    Removing the last candidate would refuse a group whose members genuinely
    do not fit on one machine."""
    root, layers = tree([worker(i, f"w{i}", "rack-a") for i in (1, 2)])

    class Capacity:
        async def __call__(self, _role, worker_ids, placed):
            used = {}
            for entry in placed:
                used[entry.worker_id] = used.get(entry.worker_id, 0) + 1
            return {w: max(0, 4 - used.get(w, 0)) for w in worker_ids}

        def spans_for(self, _role, worker_id):
            return [1, 2]

    got = await solve_group_placement(
        root, pd(), Capacity(), layers, score=lambda p: 0.0
    )

    assert isinstance(got, GroupPlacement)
