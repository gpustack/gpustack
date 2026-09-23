"""A cluster's topology, resolved against its fleet, for everyone who reads it.

The scheduler, the topology page and the deployment form's feasibility check
all need the same answer to "what does this cluster's declaration do to these
workers": which fields are in use, the tree they produce, and the ordered
scopes the solver walks. This is that one computation, so the three can never
disagree about the same cluster.

**One of everything**, because there is one tree (see
``topology.tree``): one resolved chain, one tree, one scope list, and
widening is a walk up it. Nothing here takes a chain argument.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from gpustack.topology.tree import (
    GatherScope,
    tree_scopes,
    TopologyLayerSpec,
    TopologyNode,
    build_topology,
    effective_topology_labels,
    layer_names,
    nodes_at_layer,
)
from gpustack.topology.vocabulary import (
    ResolvedLayer,
    ResolvedTopology,
    source_of,
    validate_declaration,
)


@dataclass
class Location:
    """One worker's value for one field, and where it came from."""

    value: str
    source: str
    key: str
    discovered_value: Optional[str] = None
    display: Optional[str] = None


@dataclass
class TopologyView:
    resolved: ResolvedTopology
    workers: List[object]
    active: List[ResolvedLayer] = field(default_factory=list)
    specs: List[TopologyLayerSpec] = field(default_factory=list)
    root: Optional[TopologyNode] = None
    layers: List[str] = field(default_factory=list)
    """Root-to-leaf, leaf included — the tree's layers."""

    locations: Dict[int, Dict[str, Location]] = field(default_factory=dict)
    """worker id -> layer name -> value. Keyed by name because a name
    identifies a rung, which is what the table (one column per name) and the
    batch "set position" call already assume."""

    def scopes(self) -> List[GatherScope]:
        """The search order, tightest first.

        The host first — it is the tightest scope there is and it exists
        whatever the operator declared — then the chain's layers upward. Scopes
        with nothing in them are left out so the deployment form does not offer
        a tier no group could satisfy.
        """
        tree = tree_scopes(self.root, self.layers)
        host, above = tree[0], tree[1:]
        return [host] + [
            scope
            for scope in above
            if any(not d.is_unclassified for d in scope.domains)
        ]

    def tiers(self) -> List[str]:
        """Every tier the deployment form may offer, tightest first.

        The host, then the rungs above it. The order means something here — it
        is the order the solver widens along — which is exactly what the old
        two-chain list could not promise.
        """
        return [s.name for s in self.scopes()]

    def nodes(self, layer_id: str) -> List[TopologyNode]:
        return nodes_at_layer(self.root, layer_id)

    def unclassified_at(self, layer_id: str) -> List[int]:
        """Workers with no value at this layer, across every bucket.

        The tree has one unclassified bucket *per parent* — the workers with a
        zone but no rack sit under their zone, the ones with neither sit under
        the zone-level bucket — so the answer is the union, not the first hit.
        """
        out: List[int] = []
        for node in self.nodes(layer_id):
            if node.is_unclassified:
                out.extend(node.descendant_worker_ids())
        return out

    def domain_count(self, layer_id: str) -> int:
        return len([n for n in self.nodes(layer_id) if not n.is_unclassified])

    def unfilled_workers(self) -> set:
        """Workers missing a value at any active layer, deduplicated.

        One worker missing two fields is one worker to go and fill in, not two
        problems.
        """
        out: set = set()
        for layer in self.active:
            out |= set(self.unclassified_at(layer.id))
        return out

    def location(self, worker_id: int, field_id: str) -> Optional[Location]:
        return self.locations.get(worker_id, {}).get(field_id)


def build_view(topology, workers: Iterable) -> TopologyView:
    """Resolve ``topology`` (a ``ClusterTopology`` or None) over ``workers``.

    Raises ``TopologyError`` only for a declaration that cannot become a tree;
    a worker missing a value is a normal state with a place in the result.
    """
    workers = list(workers)
    resolved = validate_declaration(topology)
    active = resolved.active(workers)
    specs = resolved.specs(active)

    view = TopologyView(
        resolved=resolved,
        workers=workers,
        active=active,
        specs=specs,
        root=build_topology(specs, workers),
        layers=layer_names(specs),
    )
    view.locations = {
        w.id: _locations_of(w, resolved)
        for w in workers
        if getattr(w, "id", None) is not None
    }
    return view


def _locations_of(worker, resolved: ResolvedTopology) -> Dict[str, Location]:
    """Every field this worker has a value for, hand-filled or discovered."""
    merged = effective_topology_labels(worker)
    facts = getattr(getattr(worker, "status", None), "topology_facts", None) or {}
    out: Dict[str, Location] = {}

    for layer in resolved.layers:
        for key in layer.label_keys:
            value = merged.get(key)
            if not value:
                continue
            discovered = next(
                (facts[k] for k in layer.label_keys if facts.get(k)), None
            )
            # No key pairs a readable name with its value today. The field
            # stays for the ones that will: keyed off the label key, so a value
            # that reads as an identifier can carry a name beside it.
            display = None
            out[layer.id] = Location(
                value=value,
                source=source_of(worker, key),
                key=key,
                discovered_value=(
                    discovered
                    if discovered != value or source_of(worker, key) == "user"
                    else None
                ),
                display=display,
            )
            break
    return out
