from types import SimpleNamespace

import pytest

from gpustack.topology.tree import (
    NODE_LAYER,
    ROOT_LAYER,
    UNCLASSIFIED,
    TopologyError,
    TopologyLayerSpec,
    build_topology,
    common_layer,
    layer_names,
    nodes_at_layer,
    order_layers,
    unclassified_at,
)

RACK = "topology.gpustack.ai/rack"
ZONE = "topology.kubernetes.io/zone"


def worker(id_: int, name: str, **labels):
    return SimpleNamespace(id=id_, name=name, labels=labels)


def two_layers():
    return [
        TopologyLayerSpec(layer="RackLayer", label_keys=[RACK, ZONE]),
        TopologyLayerSpec(layer=NODE_LAYER, parent_layer="RackLayer"),
    ]


def leaf_for(root, worker_id):
    for node in nodes_at_layer(root, NODE_LAYER):
        if worker_id in node.worker_ids:
            return node
    raise AssertionError(f"worker {worker_id} is not in the tree")


# --- layer ordering -------------------------------------------------------- #


def test_the_chain_is_ordered_root_to_leaf_not_by_declaration_order():
    """The stored form is a parent chain precisely so a declaration can be
    written in any order; deriving the order is what lets a new layer be
    inserted without renumbering the layers below it."""
    specs = [
        TopologyLayerSpec(layer="RackLayer", parent_layer="ZoneLayer"),
        TopologyLayerSpec(layer="ZoneLayer"),
    ]

    assert [s.layer for s in order_layers(specs)] == ["ZoneLayer", "RackLayer"]


def test_a_cluster_with_no_declaration_still_offers_the_tightest_choice():
    """The leaf is appended unconditionally, so "at least on the same host" is
    reachable without any configuration at all."""
    assert layer_names([]) == [NODE_LAYER]


def test_layer_names_are_root_to_leaf():
    assert layer_names(two_layers()[:1]) == ["RackLayer", NODE_LAYER]


def test_a_fork_off_the_root_is_refused():
    """Two layers hanging off the root would leave "how many layers up" with no
    single answer."""
    specs = [
        TopologyLayerSpec(layer="RackLayer"),
        TopologyLayerSpec(layer="ZoneLayer"),
    ]

    with pytest.raises(TopologyError, match="hangs off the cluster root"):
        order_layers(specs)


def test_a_fork_below_the_root_is_refused():
    specs = [
        TopologyLayerSpec(layer="ZoneLayer"),
        TopologyLayerSpec(layer="RackA", parent_layer="ZoneLayer"),
        TopologyLayerSpec(layer="RackB", parent_layer="ZoneLayer"),
    ]

    with pytest.raises(TopologyError, match="more than one child"):
        order_layers(specs)


def test_a_dangling_parent_is_refused():
    specs = [TopologyLayerSpec(layer="RackLayer", parent_layer="NoSuchLayer")]

    with pytest.raises(TopologyError, match="unknown parent"):
        order_layers(specs)


def test_a_cycle_is_refused_rather_than_looping():
    """A cycle leaves nothing hanging off the root, so it is caught there
    rather than by a guard of its own — every layer names at most one parent,
    so a cycle can never be *reached* from a root and a dedicated check inside
    the walk could not fire."""
    specs = [
        TopologyLayerSpec(layer="A", parent_layer="B"),
        TopologyLayerSpec(layer="B", parent_layer="A"),
    ]

    with pytest.raises(TopologyError):
        order_layers(specs)


def test_a_cycle_beside_a_real_chain_is_caught_as_unreachable():
    """The other half of the same argument: with a valid root present, the
    cycle is simply never walked into, and the reachability check names it."""
    specs = [
        TopologyLayerSpec(layer="Root"),
        TopologyLayerSpec(layer="A", parent_layer="B"),
        TopologyLayerSpec(layer="B", parent_layer="A"),
    ]

    with pytest.raises(TopologyError, match="not reachable"):
        order_layers(specs)


def test_the_root_name_cannot_be_taken_by_a_declared_layer():
    with pytest.raises(TopologyError, match="implicit root"):
        order_layers([TopologyLayerSpec(layer=ROOT_LAYER)])


def test_a_duplicate_layer_is_refused():
    specs = [
        TopologyLayerSpec(layer="RackLayer"),
        TopologyLayerSpec(layer="RackLayer", parent_layer="RackLayer"),
    ]

    with pytest.raises(TopologyError, match="Duplicate"):
        order_layers(specs)


# --- building -------------------------------------------------------------- #


def test_workers_sharing_a_label_value_land_in_one_domain():
    root = build_topology(
        two_layers(),
        [worker(1, "w1", **{RACK: "rack-a"}), worker(2, "w2", **{RACK: "rack-a"})],
    )

    racks = nodes_at_layer(root, "RackLayer")
    assert [r.name for r in racks] == ["rack-a"]
    assert sorted(racks[0].descendant_worker_ids()) == [1, 2]


def test_any_of_falls_through_to_the_next_key_and_reports_which_matched():
    """One layer, several vendor spellings. Which key matched has to survive to
    the UI, otherwise an operator cannot tell why a worker landed where it did."""
    root = build_topology(
        two_layers(),
        [worker(1, "w1", **{ZONE: "zone-a"})],
    )

    rack = nodes_at_layer(root, "RackLayer")[0]
    assert rack.name == "zone-a"
    assert rack.matched_label_key == ZONE


def test_the_first_declared_key_wins_when_several_are_present():
    root = build_topology(
        two_layers(),
        [worker(1, "w1", **{RACK: "rack-a", ZONE: "zone-a"})],
    )

    rack = nodes_at_layer(root, "RackLayer")[0]
    assert (rack.name, rack.matched_label_key) == ("rack-a", RACK)


def test_an_empty_label_value_counts_as_absent():
    """An empty label is a labelling accident. Reading it as a domain name would
    gather every half-labelled worker into one bogus domain that looks real."""
    root = build_topology(two_layers(), [worker(1, "w1", **{RACK: ""})])

    assert [r.name for r in nodes_at_layer(root, "RackLayer")] == [UNCLASSIFIED]


def test_a_missing_label_leaves_the_worker_schedulable_not_dropped():
    """The whole fail-open argument in one assertion: a worker with no topology
    label is still in the tree, still has a leaf, and is still schedulable."""
    root = build_topology(
        two_layers(),
        [worker(1, "labelled", **{RACK: "rack-a"}), worker(2, "bare")],
    )

    assert sorted(root.descendant_worker_ids()) == [1, 2]
    assert unclassified_at(root, "RackLayer") == [2]
    assert leaf_for(root, 2).name == "bare"


def test_no_declaration_at_all_still_produces_one_leaf_per_worker():
    root = build_topology([], [worker(1, "w1"), worker(2, "w2")])

    assert [n.name for n in nodes_at_layer(root, NODE_LAYER)] == ["w1", "w2"]
    assert sorted(root.descendant_worker_ids()) == [1, 2]


def test_the_leaf_is_keyed_by_worker_name_not_by_any_label():
    """The leaf takes the worker's name, which is why it cannot collapse: no
    label is consulted, so no label can be missing."""
    root = build_topology(
        two_layers(),
        [worker(1, "w1", **{RACK: "r"}), worker(2, "w2", **{RACK: "r"})],
    )

    rack = nodes_at_layer(root, "RackLayer")[0]
    assert sorted(c.name for c in rack.children) == ["w1", "w2"]


def test_three_layers_nest_in_the_declared_chain():
    specs = [
        TopologyLayerSpec(layer="ZoneLayer", label_keys=[ZONE]),
        TopologyLayerSpec(
            layer="RackLayer", label_keys=[RACK], parent_layer="ZoneLayer"
        ),
    ]
    root = build_topology(specs, [worker(1, "w1", **{ZONE: "z1", RACK: "r1"})])

    zone = nodes_at_layer(root, "ZoneLayer")[0]
    rack = zone.children[0]
    assert (zone.name, rack.name, rack.children[0].name) == ("z1", "r1", "w1")


# --- distance -------------------------------------------------------------- #


def test_two_workers_on_one_host_are_closest():
    root = build_topology([], [worker(1, "same"), worker(2, "same")])

    leaf = nodes_at_layer(root, NODE_LAYER)[0]
    assert common_layer(leaf, leaf) == NODE_LAYER


def test_same_rack_different_host_reports_the_rack():
    root = build_topology(
        two_layers(),
        [worker(1, "w1", **{RACK: "rack-a"}), worker(2, "w2", **{RACK: "rack-a"})],
    )

    assert common_layer(leaf_for(root, 1), leaf_for(root, 2)) == "RackLayer"


def test_different_racks_share_nothing_below_the_root():
    root = build_topology(
        two_layers(),
        [worker(1, "w1", **{RACK: "rack-a"}), worker(2, "w2", **{RACK: "rack-b"})],
    )

    assert common_layer(leaf_for(root, 1), leaf_for(root, 2)) is None


def test_two_unclassified_workers_are_not_treated_as_close():
    """The bucket means "we do not know where these are". Reading it as "these
    are together" would turn a missing label into a confident wrong answer, and
    the group scheduler would gather onto a domain that does not exist."""
    root = build_topology(two_layers(), [worker(1, "w1"), worker(2, "w2")])

    assert common_layer(leaf_for(root, 1), leaf_for(root, 2)) is None


def test_an_unclassified_worker_is_not_close_to_a_labelled_one():
    root = build_topology(
        two_layers(),
        [worker(1, "w1", **{RACK: "rack-a"}), worker(2, "w2")],
    )

    assert common_layer(leaf_for(root, 1), leaf_for(root, 2)) is None


def test_the_tightest_common_layer_wins_not_the_first_shared_one():
    specs = [
        TopologyLayerSpec(layer="ZoneLayer", label_keys=[ZONE]),
        TopologyLayerSpec(
            layer="RackLayer", label_keys=[RACK], parent_layer="ZoneLayer"
        ),
    ]
    root = build_topology(
        specs,
        [
            worker(1, "w1", **{ZONE: "z1", RACK: "r1"}),
            worker(2, "w2", **{ZONE: "z1", RACK: "r1"}),
            worker(3, "w3", **{ZONE: "z1", RACK: "r2"}),
        ],
    )

    assert common_layer(leaf_for(root, 1), leaf_for(root, 2)) == "RackLayer"
    assert common_layer(leaf_for(root, 1), leaf_for(root, 3)) == "ZoneLayer"
