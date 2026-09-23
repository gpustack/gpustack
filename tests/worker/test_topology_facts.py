"""From per-device hints to one host's position."""

from types import SimpleNamespace

from gpustack.detectors.runtime.runtime import _topology_hints
from gpustack.worker.topology_facts import (
    NVIDIA_CLIQUE_KEY,
    facts_from_devices,
)


def dev(**hints):
    return SimpleNamespace(topology_hints=hints or None)


# --- runtime appendix -> device hints --------------------------------------- #


def test_a_multi_node_fabric_becomes_a_clique():
    hints = _topology_hints({"fabric_cluster_uuid": "aaaa-bbbb", "fabric_clique_id": 7})
    assert hints == {NVIDIA_CLIQUE_KEY: "aaaa-bbbb.7"}


def test_a_single_host_hgx_reports_no_domain():
    """H100/H200 with NVSwitch report the fabric as COMPLETED with an all-zero
    UUID and clique 0; passing it through would put every HGX in the fleet into
    one shared domain."""
    hints = _topology_hints(
        {
            "fabric_cluster_uuid": "00000000-0000-0000-0000-000000000000",
            "fabric_clique_id": 0,
        }
    )
    assert NVIDIA_CLIQUE_KEY not in hints


def test_anything_but_the_fabric_keys_is_ignored():
    """Only the NVIDIA fabric keys are translated. An appendix carries
    whatever the runtime chose to attach to a device, and a key that reaches
    `topology_facts` becomes a *position* — one the layer vocabulary, the tree
    and the solver would all have to know about. Translating an unrecognised
    key would publish a position under a name nothing else in the system
    knows."""
    hints = _topology_hints(
        {
            "vendor_domain_id": 3,
            "port_neighbour_id": "c0:f9:b0:c7:13:71",
            "port_neighbour_name": "sw-1",
        }
    )
    assert hints == {}


# --- device hints -> host facts --------------------------------------------- #


def test_eight_cards_agreeing_is_one_domain():
    facts = facts_from_devices([dev(**{NVIDIA_CLIQUE_KEY: "u.1"})] * 8)
    assert facts == {NVIDIA_CLIQUE_KEY: "u.1"}


def test_cards_disagreeing_claim_no_domain():
    """The model has one domain per host; a host straddling two is not
    represented, and claiming either would be wrong for half its cards."""
    facts = facts_from_devices(
        [dev(**{NVIDIA_CLIQUE_KEY: "u.1"}), dev(**{NVIDIA_CLIQUE_KEY: "u.2"})]
    )
    assert NVIDIA_CLIQUE_KEY not in facts


def test_devices_without_hints_yield_nothing():
    assert facts_from_devices([dev(), SimpleNamespace()]) == {}
    assert facts_from_devices(None) == {}
