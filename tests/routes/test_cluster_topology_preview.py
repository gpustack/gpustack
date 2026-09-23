"""The topology view: what the vocabulary does to a real fleet.

The endpoints are thin shells over `topology.view`, so what is
tested here is the part the scheduler has no opinion about — the counts, the
per-worker locations with their sources, the unfilled bucket's payload, the
fact that a field exists in the tree only once someone filled it in, and that
the preview body wins over the saved mapping.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gpustack.api.exceptions import BadRequestException
from gpustack.routes import cluster_topology as route
from gpustack.schemas.clusters import ClusterTopology
from gpustack.topology.tree import NODE_LAYER, UNCLASSIFIED
from tests.utils.topology_layers import layer_dict, layer_obj, lid

RACK = "topology.gpustack.ai/rack"
ZONE = "topology.gpustack.ai/zone"
CLIQUE = "nvidia.com/gpu.clique"
DOMAIN = "topology.gpustack.ai/accelerator-domain"
# The leaf-switch rung, as a fleet running Topograph publishes it — the one
# source that fills this rung, which is why the tests read it here.
SWITCH = "fabric.topograph.run/tier-0"

# The two layers an operator declares for facts the fleet publishes. Neither is
# built in any more, which is the whole of the change: something else writes
# the label, the operator decides it names a place.
DOMAIN_LAYER = layer_dict("accelerator_domain", [DOMAIN, CLIQUE], parent="rack")
SWITCH_LAYER = layer_dict("switch", [SWITCH], parent="rack")


def _worker(id: int, name: str, labels=None, gpus=2, facts=None):
    return SimpleNamespace(
        id=id,
        name=name,
        labels=labels or {},
        state="ready",
        status=SimpleNamespace(
            gpu_devices=[SimpleNamespace(index=i) for i in range(gpus)],
            topology_facts=facts,
        ),
    )


def _topology(**kwargs):
    return ClusterTopology.model_validate(kwargs)


def _stubs(workers, saved=None, allocated=None, models=None):
    allocated = allocated or {}

    async def fake_allocated(worker_id):
        entry = allocated.get(worker_id, {})
        if entry == "raise":
            raise RuntimeError("no global config")
        return SimpleNamespace(ram=0, vram=entry)

    cluster = SimpleNamespace(id=1, topology=saved)
    return (
        patch.object(route.Cluster, "one_by_id", AsyncMock(return_value=cluster)),
        patch.object(route.Worker, "all_by_field", AsyncMock(return_value=workers)),
        patch.object(route, "assert_cluster_visible", lambda *a, **k: None),
        patch.object(route, "assert_org_owned_writable", lambda *a, **k: None),
        patch(
            "gpustack.server.worker_allocated_cache.get_worker_allocated",
            new=AsyncMock(side_effect=fake_allocated),
        ),
        patch(
            "gpustack.schemas.models.Model.all_by_field",
            new=AsyncMock(return_value=models or []),
        ),
    )


async def _get(workers, saved=None, allocated=None, models=None):
    stubs = _stubs(workers, saved, allocated, models)
    with stubs[0], stubs[1], stubs[2], stubs[3], stubs[4], stubs[5]:
        return await route.get_cluster_topology(session=None, ctx=None, id=1)


async def _preview(workers, saved=None, body=None, allocated=None):
    stubs = _stubs(workers, saved, allocated)
    with stubs[0], stubs[1], stubs[2], stubs[3], stubs[4], stubs[5]:
        return await route.preview_cluster_topology(
            session=None,
            ctx=None,
            id=1,
            body=route.TopologyPreviewRequest(topology=body) if body else None,
        )


def _find(node, name):
    if node.name == name:
        return node
    for child in node.children:
        found = _find(child, name)
        if found:
            return found
    return None


def _layer(result, layer_id):
    return next(layer for layer in result.layers if layer.id == layer_id)


# --- a cluster that filled nothing in --------------------------------------- #


@pytest.mark.asyncio
async def test_an_empty_cluster_is_hosts_under_the_root():
    """The leaf is built in and takes the worker's name, so a cluster with no
    values is not a degraded state — it simply cannot tell two workers apart
    above the host."""
    result = await _get([_worker(1, "w1"), _worker(2, "w2")])

    assert [c.name for c in result.tree.children] == ["w1", "w2"]
    assert result.total_workers == 2
    assert result.unclassified_workers == 0
    assert result.layers[-1].id == NODE_LAYER
    assert result.layers[-1].active is True


@pytest.mark.asyncio
async def test_every_vocabulary_field_is_listed_inactive_until_someone_fills_it():
    """The fields are what the "add a field" menu offers; none is declared,
    each simply waits for a value."""
    result = await _get([_worker(1, "w1")])

    assert [layer.id for layer in result.layers] == [
        lid("zone"),
        lid("rack"),
        NODE_LAYER,
    ]
    assert not any(layer.active for layer in result.layers[:-1])
    assert _layer(result, lid("rack")).primary_key == RACK
    assert _layer(result, lid("rack")).unclassified == 1


@pytest.mark.asyncio
async def test_a_value_on_one_worker_brings_the_layer_into_the_tree():
    """Fill it in and it exists; there is no declaration step."""
    result = await _get([_worker(1, "w1", {RACK: "R1"}), _worker(2, "w2")])

    assert _layer(result, lid("rack")).active is True
    assert _layer(result, lid("rack")).domains == 1
    assert _layer(result, lid("rack")).classified == 1
    assert _find(result.tree, "R1").workers == 1
    bucket = _find(result.tree, UNCLASSIFIED)
    assert bucket.worker_ids == [2]
    assert result.unclassified_workers == 1


# --- locations: value, source, what a hand-filled value hides -------------- #


@pytest.mark.asyncio
async def test_locations_report_where_each_value_came_from():
    """One flat map keyed by layer name, which is what lets the table keep one
    column per name."""
    # rack → switch → domain, one chain: a fork would be refused, and the
    # refusal is the point of the chain being single.
    saved = _topology(
        layers=[SWITCH_LAYER, dict(DOMAIN_LAYER, parentLayer=lid("switch"))]
    )
    workers = [
        _worker(
            1,
            "w1",
            {RACK: "R1"},
            facts={CLIQUE: "u.3", SWITCH: "aa:bb"},
        ),
    ]
    result = await _get(workers, saved=saved)
    location = result.workers[0].location

    assert location[lid("rack")].value == "R1"
    assert location[lid("rack")].source == "user"
    assert location[lid("switch")].value == "aa:bb"
    assert location[lid("accelerator_domain")].value == "u.3"
    assert location[lid("accelerator_domain")].source == "discovered"


@pytest.mark.asyncio
async def test_a_hand_filled_value_hides_the_discovered_one_and_says_so():
    """Clearing the label must be able to promise "nvl-a comes back", which
    needs the hidden value carried alongside."""
    saved = _topology(layers=[DOMAIN_LAYER])
    workers = [_worker(1, "w1", {DOMAIN: "hccs-b"}, facts={CLIQUE: "nvl-a"})]

    result = await _get(workers, saved=saved)
    location = result.workers[0].location[lid("accelerator_domain")]

    assert location.value == "hccs-b"
    assert location.source == "user"
    assert location.discovered_value == "nvl-a"


@pytest.mark.asyncio
async def test_a_discovered_domain_makes_the_declared_domain_rung_active():
    """The fact is published by the worker either way; declaring the rung is
    what turns it into a place, and then it counts like every other layer."""
    saved = _topology(layers=[DOMAIN_LAYER])
    workers = [
        _worker(1, "w1", facts={CLIQUE: "u.1"}),
        _worker(2, "w2", facts={CLIQUE: "u.1"}),
        _worker(3, "w3"),
    ]
    result = await _get(workers, saved=saved)
    domain = _layer(result, lid("accelerator_domain"))

    assert domain.active is True
    assert domain.domains == 1
    assert domain.classified == 2
    assert domain.unclassified == 1
    assert result.unclassified_workers == 1


@pytest.mark.asyncio
async def test_an_undeclared_domain_is_not_a_layer_at_all():
    """The built-in list is zone/rack and nothing else. A fleet whose runtime
    publishes a clique gets no layer for it until someone says so — the cost
    of letting the domain be placed anywhere on the chain."""
    result = await _get([_worker(1, "w1", facts={CLIQUE: "u.1"})])

    assert "accelerator_domain" not in [layer.id for layer in result.layers]
    assert "accelerator_domain" not in result.workers[0].location
    # And the key is still offered, so declaring the layer is one click.
    assert any(k.key == CLIQUE for k in result.vocabulary.known_keys)


@pytest.mark.asyncio
async def test_there_is_one_tree_and_the_domain_is_a_rung_of_it():
    """One tree holds every layer, the domain included. A domain can span
    racks, so the operator says which contains which — here, the domain sits
    under the rack — and the single tree follows that chain."""
    saved = _topology(layers=[DOMAIN_LAYER])
    workers = [
        _worker(1, "w1", {RACK: "R1"}, facts={CLIQUE: "u.1"}),
        _worker(2, "w2", {RACK: "R1"}, facts={CLIQUE: "u.2"}),
    ]
    result = await _get(workers, saved=saved)

    assert not hasattr(result, "accelerator_tree")
    assert not hasattr(result, "accelerator_layers")
    assert [c.name for c in result.tree.children] == ["R1"]
    assert [c.name for c in _find(result.tree, "R1").children] == ["u.1", "u.2"]


@pytest.mark.asyncio
async def test_the_host_is_the_leaf_and_carries_no_chain_marker():
    result = await _get([_worker(1, "w1")])

    assert result.layers[-1].id == NODE_LAYER
    assert not hasattr(result.layers[-1], "chain")


# --- counts ----------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_counts_roll_up_through_the_tree():
    workers = [
        _worker(1, "w1", {RACK: "rack-a"}, gpus=4),
        _worker(2, "w2", {RACK: "rack-a"}, gpus=4),
        _worker(3, "w3", {RACK: "rack-b"}, gpus=2),
    ]
    result = await _get(workers)

    assert result.tree.workers == 3
    assert result.tree.gpus == 10
    rack_a = _find(result.tree, "rack-a")
    assert (rack_a.workers, rack_a.gpus) == (2, 8)
    rack_b = _find(result.tree, "rack-b")
    assert (rack_b.workers, rack_b.gpus) == (1, 2)


@pytest.mark.asyncio
async def test_free_gpus_excludes_what_is_already_allocated():
    """ "Can my 2P2D fit in that rack" is the question, so an allocated card is
    not free. Allocation comes from the model-instance bindings, the same
    source the scheduler reads."""
    result = await _get(
        [_worker(1, "w1", {RACK: "rack-a"}, gpus=4)],
        allocated={1: {0: 1024, 1: 2048}},
    )
    rack = _find(result.tree, "rack-a")
    assert rack.gpus == 4
    assert rack.free_gpus == 2
    assert result.workers[0].free_gpus == 2


@pytest.mark.asyncio
async def test_a_zero_allocation_still_counts_as_free():
    result = await _get([_worker(1, "w1", gpus=2)], allocated={1: {0: 0, 1: 0}})
    assert result.tree.free_gpus == 2


@pytest.mark.asyncio
async def test_an_unreadable_allocation_counts_as_used_not_free():
    """The pessimistic direction is the only safe one: an optimistic guess
    is the one answer that sends an operator to a rack that cannot take the
    group."""
    result = await _get([_worker(1, "w1", gpus=4)], allocated={1: "raise"})
    assert result.tree.gpus == 4
    assert result.tree.free_gpus == 0


# --- the unfilled bucket ---------------------------------------------------- #


@pytest.mark.asyncio
async def test_unfilled_workers_are_deduplicated_across_layers():
    """One worker missing two values is one worker to go and fill in."""
    workers = [
        _worker(1, "w1"),
        _worker(2, "w2", {ZONE: "hall-1"}),
        _worker(3, "w3", {RACK: "R1"}),
    ]
    result = await _get(workers)
    # w1 is unfilled at zone *and* at rack; w2 only at rack; w3 only at zone.
    assert result.unclassified_workers == 3


@pytest.mark.asyncio
async def test_a_blank_label_value_is_absent_not_a_domain():
    result = await _get([_worker(1, "w1", {RACK: ""}), _worker(2, "w2", {RACK: "R1"})])
    assert _find(result.tree, UNCLASSIFIED) is not None
    assert "rack" not in result.workers[0].location


# --- any-of ----------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_the_matching_key_is_reported_so_a_mixed_fleet_is_readable():
    workers = [
        _worker(1, "w1", {RACK: "rack-a"}),
        _worker(2, "w2", {"topology.kubernetes.io/rack": "rack-b"}),
    ]
    result = await _get(workers)
    assert _find(result.tree, "rack-a").matched_label_key == RACK
    assert (
        _find(result.tree, "rack-b").matched_label_key == "topology.kubernetes.io/rack"
    )
    assert result.workers[1].location[lid("rack")].key == "topology.kubernetes.io/rack"


@pytest.mark.asyncio
async def test_the_gpustack_key_wins_when_both_are_present():
    """The owned key is first whatever the mapping says, so a hand-filled value
    cannot lose to a cloud's label."""
    result = await _get(
        [_worker(1, "w1", {RACK: "mine", "topology.kubernetes.io/rack": "theirs"})]
    )
    assert _find(result.tree, "mine") is not None
    assert _find(result.tree, "theirs") is None


# --- the preview body wins -------------------------------------------------- #


@pytest.mark.asyncio
async def test_an_unsaved_mapping_in_the_body_overrides_the_saved_one():
    """The reason the preview exists: the Advanced panel redraws the tree
    *before* saving."""
    workers = [_worker(1, "w1", {"dc/rack": "rack-a"})]
    candidate = _topology(layers=[layer_dict("rack", ["dc/rack"])])

    saved_result = await _get(workers)
    previewed = await _preview(workers, body=candidate)

    assert _find(saved_result.tree, "rack-a") is None
    assert _find(previewed.tree, "rack-a") is not None
    assert _layer(previewed, lid("rack")).label_keys == [RACK, "dc/rack"]


@pytest.mark.asyncio
async def test_no_body_falls_back_to_the_saved_mapping():
    saved = _topology(layers=[layer_dict("rack", ["dc/rack"])])
    result = await _preview([_worker(1, "w1", {"dc/rack": "rack-a"})], saved=saved)
    assert _find(result.tree, "rack-a") is not None


@pytest.mark.asyncio
async def test_a_custom_layer_appears_where_its_parent_puts_it():
    saved = _topology(layers=[layer_dict("Pod", ["dc/pod"], parent="zone")])
    result = await _get([_worker(1, "w1", {"dc/pod": "p1", RACK: "R1"})], saved=saved)

    ids = [layer.id for layer in result.layers]
    assert ids.index(lid("Pod")) == ids.index(lid("zone")) + 1
    assert _layer(result, lid("Pod")).builtin is False
    assert _find(result.tree, "p1") is not None
    assert result.workers[0].location[lid("Pod")].value == "p1"


# --- refusals: the declaration only, never the data ------------------------- #


@pytest.mark.asyncio
async def test_a_mapping_that_cannot_become_a_tree_is_refused():
    bad = SimpleNamespace(
        layers=[layer_obj("A", parent="nope")],
    )
    with pytest.raises(BadRequestException):
        await _preview([_worker(1, "w1")], body=None, saved=bad)


@pytest.mark.asyncio
async def test_missing_values_are_never_a_refusal():
    result = await _get([_worker(1, "w1"), _worker(2, "w2", {RACK: "R1"})])
    assert result.unclassified_workers == 1


# --- payload discipline ----------------------------------------------------- #


@pytest.mark.asyncio
async def test_worker_ids_are_carried_only_where_they_are_acted_on():
    result = await _get([_worker(1, "w1", {RACK: "rack-a"}), _worker(2, "w2")])

    assert result.tree.worker_ids == []
    assert _find(result.tree, "rack-a").worker_ids == []
    assert _find(result.tree, "w1").worker_ids == [1]
    assert _find(result.tree, UNCLASSIFIED).worker_ids == [2]


@pytest.mark.asyncio
async def test_layers_name_the_models_that_gather_into_them():
    """A custom layer with references cannot be deleted without stranding
    them, and the Advanced panel says which."""
    models = [
        SimpleNamespace(
            name="pd-a", gather=SimpleNamespace(layer=lid("rack"), chain=None)
        ),
        SimpleNamespace(
            name="pd-b", gather=SimpleNamespace(layer=lid("rack"), chain="network")
        ),
        SimpleNamespace(name="plain", gather=None),
    ]
    result = await _get([_worker(1, "w1", {RACK: "R1"})], models=models)

    assert _layer(result, lid("rack")).referenced_by_models == ["pd-a", "pd-b"]
    assert _layer(result, lid("zone")).referenced_by_models == []


@pytest.mark.asyncio
async def test_a_reference_to_a_custom_rung_lands_on_it():
    saved = _topology(
        layers=[
            DOMAIN_LAYER,
            layer_dict("cabinet", ["hw/cabinet"], parent="accelerator_domain"),
        ],
    )
    models = [
        SimpleNamespace(name="pd-domain", gather=SimpleNamespace(layer=lid("cabinet"))),
        SimpleNamespace(name="pd-net", gather=SimpleNamespace(layer=lid("rack"))),
    ]
    result = await _get([_worker(1, "w1")], saved=saved, models=models)

    assert _layer(result, lid("rack")).referenced_by_models == ["pd-net"]
    assert _layer(result, lid("cabinet")).referenced_by_models == ["pd-domain"]


@pytest.mark.asyncio
async def test_workers_carry_their_labels_and_a_domain_tier_counts_like_any_rung():
    """A tier *inside* the domain is an ordinary layer, so it reports the same
    numbers every other layer does — which is the property the second chain
    existed to provide and the one chain keeps."""
    saved = _topology(
        layers=[
            DOMAIN_LAYER,
            layer_dict("cabinet", ["hw/cabinet"], parent="accelerator_domain"),
        ]
    )
    workers = [
        _worker(1, "w1", {"hw/cabinet": "C1"}, facts={DOMAIN: "spod-3"}),
        _worker(2, "w2", facts={DOMAIN: "spod-3"}),
    ]
    result = await _get(workers, saved=saved)

    assert result.workers[0].labels == {"hw/cabinet": "C1"}
    cabinet = _layer(result, lid("cabinet"))
    assert cabinet.classified == 1
    assert cabinet.unclassified == 1
    assert cabinet.domains == 1
    assert _find(result.tree, "C1").workers == 1


@pytest.mark.asyncio
async def test_the_vocabulary_ships_with_the_view_root_to_leaf():
    """Two built-in fields, no chain marker on any of them."""
    result = await _get([_worker(1, "w1")])
    assert [f.id for f in result.vocabulary.fields] == [
        lid("zone"),
        lid("rack"),
    ]
    assert not hasattr(result.vocabulary.fields[0], "chain")


@pytest.mark.asyncio
async def test_the_domain_and_switch_keys_are_offered_as_candidates():
    """The built-in layers are few, the candidate keys are not. Adding the
    layer has to produce values at once, or the automatic collection the worker
    still does would be wasted."""
    result = await _get([_worker(1, "w1")])
    offered = {k.key for k in result.vocabulary.known_keys}

    assert {
        DOMAIN,
        CLIQUE,
        "accelerator.topograph.run/domain",
        "network.topology.nvidia.com/accelerator",
        "fabric.topograph.run/tier-0",
    } <= offered
    # And one key is deliberately NOT offered: nothing writes
    # `topology.gpustack.ai/switch`, and its value would be a chassis MAC that
    # no operator types, so offering it advertises a rung that can only be
    # empty. Topograph's tier-0 above is the same rung with a real source.
    assert "topology.gpustack.ai/switch" not in offered
    # And every `fits` points at a rung that still exists.
    for known in result.vocabulary.known_keys:
        assert set(known.fits) <= {"zone", "rack"}, known.key


# --- locations: filling a value in ----------------------------------------- #


class _FakeService:
    def __init__(self, session):
        pass

    async def update(self, worker, patch):
        worker.labels = patch["labels"]


async def _set(workers, assignments, saved=None):
    stubs = _stubs(workers, saved)
    with (
        stubs[0],
        stubs[1],
        stubs[2],
        stubs[3],
        stubs[4],
        stubs[5],
        patch("gpustack.server.services.WorkerService", _FakeService),
    ):
        body = route.LocationsRequest(
            assignments=[route.LocationAssignment(**a) for a in assignments]
        )
        return await route.set_cluster_topology_locations(
            session=None, ctx=None, id=1, body=body
        )


@pytest.mark.asyncio
async def test_setting_a_location_writes_the_fields_own_key():
    workers = [_worker(1, "w1"), _worker(2, "w2")]
    result = await _set(
        workers, [{"worker_ids": [1, 2], "layer": lid("rack"), "value": "R3"}]
    )

    assert workers[0].labels == {RACK: "R3"}
    assert workers[1].labels == {RACK: "R3"}
    assert _find(result.topology.tree, "R3").workers == 2


@pytest.mark.asyncio
async def test_the_response_carries_the_inverse_for_undo():
    """Undo is the same call with the previous values: the two that had R1 get
    R1 back and the one that had nothing gets cleared, in two assignments."""
    workers = [
        _worker(1, "w1", {RACK: "R1"}),
        _worker(2, "w2", {RACK: "R1"}),
        _worker(3, "w3"),
    ]
    result = await _set(
        workers, [{"worker_ids": [1, 2, 3], "layer": lid("rack"), "value": "R3"}]
    )

    inverse = {(a.value, tuple(a.worker_ids)) for a in result.previous}
    assert inverse == {(None, (3,)), ("R1", (1, 2))}

    undone = await _set(workers, [a.model_dump() for a in result.previous])
    assert workers[0].labels == {RACK: "R1"}
    assert workers[2].labels == {}
    assert _find(undone.topology.tree, "R3") is None


@pytest.mark.asyncio
async def test_clearing_only_removes_the_owned_key():
    """Other sources' keys are never touched: clearing a hand-filled value is
    what uncovers a discovered or cloud-provided one."""
    workers = [_worker(1, "w1", {RACK: "R1", "topology.kubernetes.io/rack": "cloud-a"})]
    result = await _set(
        workers, [{"worker_ids": [1], "layer": lid("rack"), "value": None}]
    )

    assert workers[0].labels == {"topology.kubernetes.io/rack": "cloud-a"}
    assert result.topology.workers[0].location[lid("rack")].value == "cloud-a"


@pytest.mark.asyncio
async def test_a_blank_value_clears():
    workers = [_worker(1, "w1", {RACK: "R1"})]
    await _set(workers, [{"worker_ids": [1], "layer": lid("rack"), "value": "  "}])
    assert workers[0].labels == {}


@pytest.mark.asyncio
async def test_a_declared_domain_can_be_filled_by_hand():
    """Ascend has no standard label for the super pod — `superPodID` is a node
    *annotation* — so hand-filling is the only source there, and it goes
    through the one "set position" call like a rack does."""
    saved = _topology(layers=[DOMAIN_LAYER])
    workers = [_worker(1, "w1")]
    result = await _set(
        workers,
        [{"worker_ids": [1], "layer": lid("accelerator_domain"), "value": "hccs-b"}],
        saved=saved,
    )
    assert workers[0].labels == {DOMAIN: "hccs-b"}
    assert _layer(result.topology, lid("accelerator_domain")).domains == 1


@pytest.mark.asyncio
async def test_a_custom_rung_is_filled_in_by_the_same_call():
    """No chain argument, no second endpoint: the layer name is the key."""
    saved = _topology(
        layers=[
            DOMAIN_LAYER,
            layer_dict("cabinet", ["hw/cabinet"], parent="accelerator_domain"),
        ],
    )
    workers = [_worker(1, "w1"), _worker(2, "w2")]
    result = await _set(
        workers,
        [
            {"worker_ids": [1], "layer": lid("rack"), "value": "R1"},
            {"worker_ids": [2], "layer": lid("cabinet"), "value": "C1"},
        ],
        saved=saved,
    )

    assert workers[0].labels == {RACK: "R1"}
    assert workers[1].labels == {"hw/cabinet": "C1"}
    assert _find(result.topology.tree, "C1") is not None


@pytest.mark.asyncio
async def test_an_undeclared_domain_cannot_be_filled_in():
    """The refusal that replaces the built-in rung: there is no field to write
    until the operator has said the fleet has one."""
    workers = [_worker(1, "w1")]
    with pytest.raises(BadRequestException):
        await _set(
            workers,
            [
                {
                    "worker_ids": [1],
                    "layer": lid("accelerator_domain"),
                    "value": "hccs-b",
                }
            ],
        )


@pytest.mark.asyncio
async def test_an_unknown_field_or_foreign_worker_refuses_the_whole_batch():
    workers = [_worker(1, "w1")]
    with pytest.raises(BadRequestException):
        await _set(workers, [{"worker_ids": [1], "layer": NODE_LAYER, "value": "x"}])
    with pytest.raises(BadRequestException):
        await _set(
            workers,
            [
                {"worker_ids": [1], "layer": lid("rack"), "value": "R1"},
                {"worker_ids": [99], "layer": lid("rack"), "value": "R1"},
            ],
        )
    assert workers[0].labels == {}
