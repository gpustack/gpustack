"""The vocabulary: fill a value in and a layer exists; declare nothing.

What is pinned here is the contract the table, the tree and the solver all
rely on — the owned key is always tried first, a field is active only once
someone filled it in, custom layers slot in by their parent, and the same
declarations are refused everywhere for the same reasons.

And the property this file exists for: **there is one chain, and the
accelerator domain is an ordinary rung of it.** Nothing here returns a second
chain, a chain marker, or a scope the tree did not produce. The keys a domain
is published under survive as candidate keys, so an operator who adds a layer
pointing at one gets values immediately.
"""

from types import SimpleNamespace

import pytest

from gpustack.topology.tree import (
    NODE_LAYER,
    NODE_LAYER_SLUG,
    ROOT_LAYER,
    TopologyError,
    UNCLASSIFIED,
)
from gpustack.topology.view import build_view
from tests.utils.topology_layers import layer_obj, lid
from gpustack.topology.vocabulary import (
    KNOWN_KEYS,
    RESERVED_IDS,
    VOCABULARY,
    VOCABULARY_IDS,
    gather_layer_names,
    primary_key_for,
    resolve,
    validate_declaration,
)

RACK = "topology.gpustack.ai/rack"
K8S_RACK = "topology.kubernetes.io/rack"
ZONE = "topology.gpustack.ai/zone"
K8S_ZONE = "topology.kubernetes.io/zone"
# Not a rung any more — kept as a plain label key a custom layer can read.
ROW = "topology.gpustack.ai/row"
CLIQUE = "nvidia.com/gpu.clique"
DOMAIN = "topology.gpustack.ai/accelerator-domain"
SWITCH = "topology.gpustack.ai/switch"

# The rung an operator declares for the accelerator domain. It is a custom
# layer like any other now — the name is not reserved, and nothing in the code
# knows it is special.
DOMAIN_LAYER = "accelerator_domain"


def worker(id_, name, labels=None, facts=None):
    return SimpleNamespace(
        id=id_,
        name=name,
        labels=labels or {},
        status=SimpleNamespace(topology_facts=facts),
    )


# Tests go on naming layers in domain terms ("rack", "Hall"); `layer_obj`
# turns a name into the id/name pair the resolver actually reads. See
# tests/utils/topology_layers.py.
layer = layer_obj


def layer_host(**kw):
    """The leaf's own entry, which exists only so it can be renamed."""
    return layer_obj(NODE_LAYER_SLUG, **kw)


def layer_root():
    """An entry claiming the implicit root's id — refused, since the root is
    not a layer at all. Built by hand: `lid` only knows real layers."""
    entry = layer_obj("root")
    entry.id = ROOT_LAYER
    return entry


def topology(layers=()):
    return SimpleNamespace(layers=list(layers))


def domain_layer(parent="rack"):
    """The declaration an operator writes to place the domain on the chain."""
    return layer(DOMAIN_LAYER, [DOMAIN, CLIQUE], parent=parent)


# --- resolve ---------------------------------------------------------------- #


def test_no_declaration_is_the_vocabulary_as_is():
    resolved = resolve(None)
    assert [x.id for x in resolved.layers] == list(VOCABULARY_IDS)
    assert [x.name for x in resolved.layers] == ["zone", "rack"]
    assert resolved.layer(lid("rack")).label_keys == (RACK, K8S_RACK)


def test_naming_a_vocabulary_field_overrides_its_keys_but_keeps_the_owned_key_first():
    """The owned key is what the table writes; if it were not tried first a
    hand-filled value could lose to a discovered one."""
    resolved = resolve(topology([layer("rack", ["dc/rack"])]))
    assert resolved.layer(lid("rack")).label_keys == (RACK, "dc/rack")

    resolved = resolve(topology([layer("rack", ["dc/rack", RACK])]))
    assert resolved.layer(lid("rack")).label_keys == (RACK, "dc/rack")


def test_a_custom_layer_slots_in_below_its_parent():
    resolved = resolve(topology([layer("Pod", ["dc/pod"], parent="zone")]))
    ids = [x.name for x in resolved.layers]
    assert ids.index("Pod") == ids.index("zone") + 1
    assert resolved.layer(lid("Pod")).builtin is False


def test_the_accelerator_domain_is_declared_as_an_ordinary_custom_layer():
    """The domain is a layer the operator inserts where their hardware puts
    it, reading the keys the runtime already writes."""
    resolved = resolve(topology([domain_layer(parent="zone")]))
    ids = [x.name for x in resolved.layers]
    assert ids == ["zone", DOMAIN_LAYER, "rack"]
    assert resolved.layer(lid(DOMAIN_LAYER)).builtin is False
    assert resolved.layer(lid(DOMAIN_LAYER)).label_keys == (DOMAIN, CLIQUE)


def test_tiers_inside_a_domain_chain_under_it_with_no_schema_change():
    """The Atlas 950 case — three bandwidth tiers inside one domain (blade 1008,
    cabinet 896, across cabinets 448 GB/s). Each is a layer; none is a field."""
    resolved = resolve(
        topology(
            [
                domain_layer(parent="zone"),
                layer("cabinet", ["hw/cabinet"], parent=DOMAIN_LAYER),
                layer("blade", ["hw/blade"], parent="cabinet"),
            ]
        )
    )
    ids = [x.name for x in resolved.layers]
    assert ids == ["zone", DOMAIN_LAYER, "cabinet", "blade", "rack"]


def test_a_parentless_custom_layer_sits_above_the_vocabulary():
    resolved = resolve(topology([layer("Campus", ["dc/campus"])]))
    assert [x.name for x in resolved.layers][:2] == ["Campus", "zone"]


def test_custom_layers_chain_under_each_other():
    resolved = resolve(
        topology([layer("B", parent="A"), layer("A", parent="rack")]),
    )
    ids = [x.name for x in resolved.layers]
    assert ids[ids.index("rack") + 1 :][:2] == ["A", "B"]


def test_a_name_is_looked_up_once_because_there_is_one_chain():
    resolved = validate_declaration(
        topology([layer("cage", ["dc/cage"], parent="rack")])
    )
    assert resolved.layer(lid("cage")) is not None
    assert resolved.layer(lid("nonsense")) is None


# --- active ----------------------------------------------------------------- #


def test_a_field_is_active_only_once_a_worker_has_a_value():
    resolved = resolve(None)
    assert resolved.active([worker(1, "w1")]) == []
    assert [x.name for x in resolved.active([worker(1, "w1", {RACK: "R1"})])] == [
        "rack"
    ]
    assert [x.name for x in resolved.active([worker(1, "w1", {K8S_RACK: "R1"})])] == [
        "rack"
    ]


def test_a_discovered_fact_activates_a_declared_domain_layer_too():
    """The worker keeps writing the domain as a label (the whole of
    `topology_facts` is kept), so declaring the rung is all an operator does."""
    resolved = resolve(topology([domain_layer()]))
    active = resolved.active([worker(1, "w1", facts={CLIQUE: "u.1"})])
    assert [x.name for x in active] == [DOMAIN_LAYER]


def test_a_custom_layer_stays_visible_when_nobody_matches_it():
    """An operator who wrote it down wants to see that nobody matches."""
    resolved = resolve(topology([layer("Pod", ["dc/pod"], parent="zone")]))
    assert [x.name for x in resolved.active([worker(1, "w1")])] == ["Pod"]


def test_active_layers_chain_in_vocabulary_order_whatever_is_skipped():
    resolved = resolve(None)
    active = resolved.active([worker(1, "w1", {ZONE: "hall-1", RACK: "r"})])
    specs = resolved.specs(active)
    assert [(s.layer, s.parent_layer) for s in specs] == [
        (lid("zone"), None),
        (lid("rack"), lid("zone")),
    ]


# --- primary keys ----------------------------------------------------------- #


def test_primary_keys_are_found_by_name_alone():
    resolved = resolve(
        topology(
            [
                layer("Pod", ["dc/pod", "other/pod"], parent="zone"),
                domain_layer(),
            ]
        )
    )
    assert primary_key_for(resolved, lid("rack")) == RACK
    assert primary_key_for(resolved, lid("zone")) == ZONE
    assert primary_key_for(resolved, lid(DOMAIN_LAYER)) == DOMAIN
    assert primary_key_for(resolved, lid("Pod")) == "dc/pod"
    assert primary_key_for(resolved, NODE_LAYER) is None
    assert primary_key_for(resolved, lid("nonsense")) is None


# --- validation: the same refusals everywhere ------------------------------- #


@pytest.mark.parametrize(
    "layers, message",
    [
        ([layer("A", parent="nope")], "unknown parent"),
        ([layer("A"), layer("B")], "share a parent"),
        ([layer("A", parent="B"), layer("B", parent="A")], "not reachable"),
        # The root is not a layer at all. The leaf IS one — declarable, but
        # only to rename: it takes the worker's own name, which is what lets a
        # tree survive a fleet with no labels, so it may not read a key, move,
        # or be switched off.
        ([layer_root()], "reserved"),
        ([layer_host(keys=[RACK])], "cannot read label keys"),
        ([layer_host(parent="rack")], "cannot name a parent"),
        ([layer_host(disabled=True)], "cannot be disabled"),
        ([layer("A"), layer("A")], "Duplicate"),
    ],
)
def test_unbuildable_declarations_are_refused(layers, message):
    with pytest.raises(TopologyError, match=message):
        validate_declaration(topology(layers))


def test_only_the_root_and_the_leaf_are_reserved_names():
    """The set shrank with the second chain. `rack` was never reserved
    against its own chain — naming it is how an operator overrides its keys —
    and `accelerator_domain` was only reserved because it was the *other*
    chain's built-in rung. There is no other chain."""
    assert RESERVED_IDS == frozenset({"ClusterTopologyLayer", NODE_LAYER})
    assert DOMAIN_LAYER not in RESERVED_IDS
    assert "zone" not in RESERVED_IDS
    assert "rack" not in RESERVED_IDS


def test_a_custom_layer_may_be_called_accelerator_domain():
    """Which is the recommended spelling, now that it is just a name."""
    resolved = validate_declaration(topology([domain_layer()]))
    assert resolved.layer(lid(DOMAIN_LAYER)) is not None


def test_a_parent_on_a_vocabulary_entry_is_ignored():
    """The UI serialises the whole chain uniformly; a builtin's place is fixed
    regardless of what it says its parent is."""
    resolved = validate_declaration(
        topology([layer("rack", ["dc/rack"], parent="zone")])
    )
    assert [x.id for x in resolved.layers] == list(VOCABULARY_IDS)
    assert resolved.layer(lid("rack")).label_keys == (RACK, "dc/rack")


def test_a_valid_declaration_is_returned_resolved():
    resolved = validate_declaration(topology([layer("Pod", ["dc/pod"], parent="zone")]))
    assert resolved.layer(lid("Pod")) is not None


def test_gather_layer_names_are_the_host_then_the_chain():
    names = gather_layer_names(resolve(None))
    assert names == [NODE_LAYER, lid("zone"), lid("rack")]
    assert len(names) == len(set(names))


def test_an_undeclared_domain_is_not_a_gather_target():
    """What the deployment form has to see: `gather.layer` has no
    `accelerator_domain` special value. A cluster that did not declare the rung
    does not offer it."""
    assert lid(DOMAIN_LAYER) not in gather_layer_names(resolve(None))
    assert lid(DOMAIN_LAYER) in gather_layer_names(resolve(topology([domain_layer()])))


# --- the candidate keys ----------------------------------------------------- #


def test_the_domain_and_switch_keys_are_offered_as_candidates():
    """There are two built-in layers, but the keys a fleet already publishes
    stay in the suggestion list so adding the layer yields values at once
    rather than after a relabelling campaign."""
    keys = {k.key for k in KNOWN_KEYS}
    assert {
        DOMAIN,
        CLIQUE,
        "accelerator.topograph.run/domain",
        "network.topology.nvidia.com/accelerator",
        "fabric.topograph.run/tier-0",
    } <= keys


def test_our_own_switch_key_is_not_offered():
    """A candidate key earns its place by having a source. Nothing writes
    `topology.gpustack.ai/switch`, and its value would be a switch chassis MAC
    — not something an operator can look up and type — so offering it would
    advertise a rung that can only ever be empty. Topograph's tier-0 is the
    same rung from a source that does exist, which is why it stays in the list
    above."""
    assert SWITCH not in {k.key for k in KNOWN_KEYS}


def test_every_candidate_key_fits_a_rung_that_exists():
    """`fits` is where the Advanced panel offers to insert the layer, so it can
    only name the three built-ins that survive."""
    for known in KNOWN_KEYS:
        assert known.fits, known.key
        assert set(known.fits) <= {f.slug for f in VOCABULARY}, known.key


# --- the domain is an ordinary tree rung ------------------------------------ #


def test_a_declared_domain_is_a_tree_rung_with_an_unclassified_bucket():
    workers = [
        worker(1, "w1", facts={CLIQUE: "u.1"}),
        worker(2, "w2", facts={CLIQUE: "u.1"}),
        worker(3, "w3", facts={CLIQUE: "u.2"}),
        worker(4, "w4"),
    ]
    view = build_view(topology([domain_layer()]), workers)
    groups = view.nodes(lid(DOMAIN_LAYER))

    by_name = {g.name: sorted(g.descendant_worker_ids()) for g in groups}
    assert by_name == {"u.1": [1, 2], "u.2": [3], UNCLASSIFIED: [4]}


def test_a_hand_filled_domain_wins_over_the_discovered_one():
    view = build_view(
        topology([domain_layer()]),
        [worker(1, "w1", {DOMAIN: "hccs-b"}, facts={CLIQUE: "u.1"})],
    )
    assert [g.name for g in view.nodes(lid(DOMAIN_LAYER))] == ["hccs-b"]


def test_a_tier_inside_the_domain_nests_under_it_not_beside_it():
    """The pair, not the cabinet alone: cabinet R1 in super pod 3 and cabinet
    R1 in super pod 4 are not the same place — and with a chain that falls out
    of the tree rather than out of a string concatenation."""
    workers = [
        worker(1, "w1", {"hw/cabinet": "R1"}, facts={DOMAIN: "spod-3"}),
        worker(2, "w2", {"hw/cabinet": "R1"}, facts={DOMAIN: "spod-3"}),
        worker(3, "w3", {"hw/cabinet": "R1"}, facts={DOMAIN: "spod-4"}),
        worker(4, "w4", facts={DOMAIN: "spod-3"}),  # no cabinet: unclassified there
    ]
    view = build_view(
        topology(
            [
                domain_layer(),
                layer("cabinet", ["hw/cabinet"], parent=DOMAIN_LAYER),
            ]
        ),
        workers,
    )

    cabinets = view.nodes(lid("cabinet"))
    named = {
        (c.parent.name, c.name): sorted(c.descendant_worker_ids())
        for c in cabinets
        if not c.is_unclassified
    }
    assert named == {("spod-3", "R1"): [1, 2], ("spod-4", "R1"): [3]}
    assert view.unclassified_at(lid("cabinet")) == [4]


# --- the view's scopes: one list, and it comes out of the tree --------------- #


def test_the_search_runs_host_first_then_the_declared_rungs_outward():
    workers = [
        worker(1, "w1", {RACK: "R1", ZONE: "hall-1"}, facts={CLIQUE: "u.1"}),
        worker(2, "w2", {RACK: "R1", ZONE: "hall-1"}, facts={CLIQUE: "u.1"}),
    ]
    view = build_view(topology([domain_layer()]), workers)

    assert [s.name for s in view.scopes()] == [
        NODE_LAYER,
        lid(DOMAIN_LAYER),
        lid("rack"),
        lid("zone"),
    ]


def test_the_domain_is_a_scope_like_any_other():
    """The domain is not a scope list of its own, unranked against the tree's:
    it is a rung between the host and the rack, in the one order the solver
    widens along."""
    workers = [
        worker(1, "w1", {RACK: "R1"}, facts={CLIQUE: "u.1"}),
        worker(2, "w2", {RACK: "R1"}, facts={CLIQUE: "u.1"}),
    ]
    view = build_view(topology([domain_layer()]), workers)

    names = [s.name for s in view.scopes()]
    assert names.index(lid(DOMAIN_LAYER)) < names.index(lid("rack"))
    assert view.tiers() == names


def test_a_field_nobody_filled_is_not_a_scope():
    view = build_view(None, [worker(1, "w1"), worker(2, "w2")])
    assert [s.name for s in view.scopes()] == [NODE_LAYER]
    assert view.domain_count("rack") == 0


def test_a_custom_layer_nobody_matches_is_shown_but_not_offered():
    view = build_view(
        topology([layer("Pod", ["dc/pod"], parent="zone")]), [worker(1, "w1")]
    )
    assert [x.name for x in view.active] == ["Pod"]
    assert "Pod" not in view.tiers()


# --- identity, renaming, disabling ----------------------------------------- #


def test_a_layers_id_is_its_identity_and_its_name_is_not():
    """The split the whole structure exists for. Two layers may read the same
    keys and be called the same thing by two different clusters; what a
    `parentLayer` and a `Model.gather.layer` point at is neither."""
    resolved = resolve(None)
    assert [x.id for x in resolved.layers] == [
        "builtin-000001",
        "builtin-000003",
    ]
    assert [x.name for x in resolved.layers] == ["zone", "rack"]
    assert NODE_LAYER == "builtin-000004"
    # `builtin-000002` is absent, not renumbered. It was `row`, and it is
    # retired rather than reused: a stored `parentLayer` or `Model.gather.
    # layer` still naming it must resolve to nothing rather than silently to
    # whatever took the number. Ids are allocation order, never position.


def test_renaming_a_builtin_changes_what_it_is_called_and_nothing_else():
    resolved = resolve(topology([layer("rack", display_name="A区机柜")]))
    rung = resolved.layer(lid("rack"))

    assert rung.display_name == "A区机柜"
    assert rung.label == "A区机柜"
    # The three things a rename must not touch: what points at it, what it
    # reads, and the key the table writes.
    assert rung.id == lid("rack")
    assert rung.name == "rack"
    assert rung.label_keys == (RACK, K8S_RACK)


def test_an_unrenamed_layer_has_no_display_name_at_all():
    """Absence is the only way to say "never renamed", which is why there is
    no boolean beside it: a flag and a string can contradict each other, and
    one of the two states would then be unreachable."""
    rung = resolve(None).layer(lid("rack"))
    assert rung.display_name is None
    assert rung.label == "rack"


def test_a_builtin_cannot_be_renamed_by_rewriting_its_canonical_name():
    """The failure this refuses is silent: a client that puts the *translated*
    label in `name` freezes the row into one person's UI language, and every
    other reader gets it. Refused rather than corrected, so the bug surfaces
    where it is made."""
    bad = layer("rack")
    bad.name = "机柜"
    with pytest.raises(TopologyError, match="must keep the name 'rack'"):
        validate_declaration(topology([bad]))


def test_the_host_can_be_renamed_too():
    resolved = resolve(topology([layer_host(display_name="裸机")]))
    assert resolved.host_display_name == "裸机"
    # And it stays the leaf rather than becoming a rung of the chain.
    assert [x.name for x in resolved.layers] == ["zone", "rack"]


def test_two_layers_cannot_end_up_with_the_same_label():
    """The deployment form's "at least in the same ___" is a list of these,
    so a duplicate is two options a deployer cannot tell apart."""
    with pytest.raises(TopologyError, match="both called 'rack'"):
        validate_declaration(topology([layer("Rack")]))

    with pytest.raises(TopologyError, match="both called"):
        validate_declaration(
            topology([layer("zone", display_name="X"), layer("rack", display_name="X")])
        )


def test_a_disabled_builtin_leaves_the_tree_but_stays_in_the_list():
    """Distinct from a rung nobody filled in: that one is a fact about the
    data and comes back the moment a worker grows the label, this one is a
    decision and does not.

    It stays in `layers`. That list is what `GET /topology` renders the
    Advanced panel from, so dropping it here took away the very switch that
    turns it back on — disabling became a one-way door. Only `active()`
    excludes it, which is what keeps it out of the tree and the tiers."""
    declared = topology([layer("zone", disabled=True)])
    resolved = resolve(declared)
    assert [x.name for x in resolved.layers] == ["zone", "rack"]
    assert resolved.layer(lid("zone")).disabled is True

    # Labelled, so "not a tier" can only be the switch and not missing data.
    labelled = [worker(1, "w1", {ZONE: "H", RACK: "R1"})]
    view = build_view(declared, labelled)
    assert [x.name for x in view.active] == ["rack"]
    assert lid("zone") not in view.tiers()
    assert lid("rack") in view.tiers()


def test_only_a_builtin_can_be_disabled():
    """A custom layer is deleted instead — there is no vocabulary entry for it
    to fall back to, so a disabled one would be a row that means nothing."""
    with pytest.raises(TopologyError, match="cannot be disabled"):
        validate_declaration(topology([layer("Pod", ["dc/pod"], disabled=True)]))


def test_a_custom_layer_under_a_disabled_rung_survives_and_the_chain_closes_up():
    """Not refused, and not stranded either.

    It *was* refused, back when disabling removed the rung from `layers`:
    the child then had no parent to be placed under and came out unreachable.
    With the rung kept, the child is placed as usual and simply drops out of
    `active` along with its parent — the chain closes over the gap instead of
    breaking at it, so there is nothing to refuse."""
    declared = topology(
        [
            layer("zone", disabled=True),
            layer("Pod", ["dc/pod"], parent="zone"),
        ]
    )
    resolved = validate_declaration(declared)
    assert [x.name for x in resolved.layers] == ["zone", "Pod", "rack"]

    labelled = [worker(1, "w1", {ROW: "H", "dc/pod": "P1", RACK: "R1"})]
    view = build_view(declared, labelled)
    assert [x.name for x in view.active] == ["Pod", "rack"]
