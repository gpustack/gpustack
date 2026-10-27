"""A cluster's topology, as *one* tree of workers.

A tree answers one question for the group scheduler: *how far apart are two
workers*. Everything else here exists to build that tree out of the only source
of truth a worker has for where it physically sits — its labels.

**One chain, root to leaf, and the operator says what is on it.** The built-in
rungs are zone → rack, and anything else an operator's fabric has — an
NVLink/HCCS domain, a blade, a cage — is a custom layer they insert wherever it
belongs. ``order_layers``, ``build_topology``, ``common_layer``,
``nodes_at_layer`` and ``unclassified_at`` each run once, over that one chain.

**The accelerator domain is a rung on that chain, not a chain beside it.**
As long as a domain's boundary is a run of *contiguous* cabinets it is
expressible as one rung, and on every shipping generation it is (NVL72 = 1
cabinet, NVL36×2 = 2, CloudMatrix384 = 16, Atlas 950 = 160). Where the domain
sits *inside* a machine — 910B2 — "same domain" and "same host" are the same
constraint, and the built-in leaf already covers it. Declaring the rung in the
wrong place is an operator error, not something the model cannot say.

**A layer name identifies a rung on its own**, which is what lets the whole
product treat it as a key: one column per name in the table, one entry per name
in a worker's location map, one `gather.layer`.

**The leaf never collapses.** ``NodeTopologyLayer`` takes the worker's name
rather than a label, so a cluster with no topology declared at all still gets a
usable tree: one leaf per worker under the root. That is what makes every
failure here a loss of *resolution* rather than a loss of *service* — a missing
or mistyped label can only make two workers look equally distant, never make a
worker unschedulable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

# The two layer ids the operator cannot use for a layer of their own. The
# root is implicit (every declared layer without a parent hangs off it) and the
# leaf is the worker itself.
#
# The leaf carries a registry number like every other built-in rung, because it
# IS one: it shows up in the chain, it is renameable, and `Model.gather.layer`
# stores it whenever a deployment asks to stay on one host. The root does not —
# it is never a gather tier (`group_solver` filters it out), cannot be
# declared, and never reaches the wire.
ROOT_LAYER = "ClusterTopologyLayer"
NODE_LAYER = "builtin-000004"
NODE_LAYER_SLUG = "host"

# The domain a worker lands in when the layer's label keys all miss. Named
# rather than dropped: an unclassified worker is still schedulable, and the UI
# needs something to hang "20 workers are missing this label" off.
UNCLASSIFIED = "<unclassified>"


@dataclass(frozen=True)
class TopologyLayerSpec:
    """One declared layer.

    ``label_keys`` is any-of rather than a single key on purpose: the same
    physical layer is spelled differently by every vendor and cloud
    (``topology.kubernetes.io/zone`` on one fleet, a private key on the next),
    and a cluster that mixes them should not force the operator to relabel
    everything first. The first key present on the worker wins, and which one
    matched is reported so the UI can show it.
    """

    layer: str
    label_keys: Sequence[str] = ()
    parent_layer: Optional[str] = None


@dataclass
class TopologyNode:
    """A domain, or a worker when ``layer == NODE_LAYER``."""

    layer: str
    name: str
    parent: Optional["TopologyNode"] = None
    children: List["TopologyNode"] = field(default_factory=list)
    worker_ids: List[int] = field(default_factory=list)
    # Which of the layer's any-of keys actually matched, for the worker(s)
    # under this domain. None at the root, the leaf, and the unclassified
    # bucket, none of which are reached through a label.
    matched_label_key: Optional[str] = None

    @property
    def is_unclassified(self) -> bool:
        return self.name == UNCLASSIFIED

    def path(self) -> List[str]:
        """This domain's address, root-to-here, with the root left out.

        A domain's ``name`` identifies it only among its **siblings** --
        ``_child`` dedupes within one parent's children and nowhere wider --
        so ``(layer, name)`` is a label, not an address. Two zones each
        holding a ``rack-1`` produce two nodes indistinguishable by that pair,
        and per-zone rack numbering is the normal way to name racks. The
        unclassified bucket is worse: every parent grows its own, so one layer
        can hold many nodes all called the same thing.

        The root is left out because it is an internal layer id
        (``ClusterTopologyLayer``) that means "somewhere in this cluster" --
        naming it would put a word in front of the reader that identifies
        nothing.

        Returns:
            The domain names from the outermost declared layer down to this
            node, e.g. ``["zone-b", "rack-1"]``. Empty for the root itself.
        """
        out: List[str] = []
        node: Optional["TopologyNode"] = self
        while node is not None and node.layer != ROOT_LAYER:
            out.append(node.name)
            node = node.parent
        out.reverse()
        return out

    def descendant_worker_ids(self) -> List[int]:
        if self.layer == NODE_LAYER:
            return list(self.worker_ids)
        out: List[int] = []
        for child in self.children:
            out.extend(child.descendant_worker_ids())
        return out


class TopologyError(ValueError):
    """A declaration that cannot be turned into a tree.

    Raised only for the declaration — never for the data. A cycle or a dangling
    parent means the operator's intent is unknowable; a worker missing a label
    means only that it is unclassified, which is a normal state.
    """


def order_layers(specs: Sequence[TopologyLayerSpec]) -> List[TopologyLayerSpec]:
    """Root-to-leaf order, derived from the parent chain.

    The chain is stored rather than an ordered list because inserting a layer
    into an ordered list renumbers every layer below it, and these names are
    referenced from saved model configurations. With a chain, a new layer names
    its parent and nothing else moves.

    A layer with no parent hangs off the implicit root. Declaring several such
    layers is a fork, which is rejected: the tree the scheduler walks has one
    path from root to leaf, and a fork would make "how many layers up" have no
    single answer.
    """
    if not specs:
        return []

    by_name: Dict[str, TopologyLayerSpec] = {}
    for spec in specs:
        if not spec.layer:
            raise TopologyError("A topology layer must have a name.")
        if spec.layer == ROOT_LAYER:
            raise TopologyError(
                f"{ROOT_LAYER!r} is the implicit root and cannot be declared. "
                "Leave `parent_layer` unset on the topmost layer instead."
            )
        if spec.layer in by_name:
            raise TopologyError(f"Duplicate topology layer {spec.layer!r}.")
        by_name[spec.layer] = spec

    # Before the root check, not after: a layer whose parent is a typo has no
    # root either, and "you declared no topmost layer" would send the operator
    # looking in the wrong place.
    children: Dict[str, List[TopologyLayerSpec]] = {}
    for spec in specs:
        if spec.parent_layer is None:
            continue
        if spec.parent_layer not in by_name:
            raise TopologyError(
                f"Topology layer {spec.layer!r} names an unknown parent "
                f"{spec.parent_layer!r}."
            )
        children.setdefault(spec.parent_layer, []).append(spec)

    roots = [s for s in specs if not s.parent_layer]
    if not roots:
        raise TopologyError(
            "Every topology layer names a parent, so none of them hangs off the "
            "cluster root. Leave `parent_layer` unset on the topmost layer."
        )
    if len(roots) > 1:
        names = ", ".join(sorted(r.layer for r in roots))
        raise TopologyError(
            f"More than one topology layer hangs off the cluster root ({names}). "
            "The layers must form a single chain from the cluster down to the node."
        )

    # No cycle guard below, deliberately. Every layer names at most one parent,
    # so the declaration is a forest of in-trees and a cycle can never be
    # reached from a root: a cycle whose members all have parents either leaves
    # no root at all (caught above) or sits unreachable beside one (caught by
    # the reachability check below). A guard here could not fire.
    ordered: List[TopologyLayerSpec] = []
    current: Optional[TopologyLayerSpec] = roots[0]
    while current is not None:
        ordered.append(current)
        kids = children.get(current.layer, [])
        if len(kids) > 1:
            names = ", ".join(sorted(k.layer for k in kids))
            raise TopologyError(
                f"Topology layer {current.layer!r} has more than one child "
                f"({names}). The layers must form a single chain."
            )
        current = kids[0] if kids else None

    if len(ordered) != len(specs):
        missing = ", ".join(sorted(set(by_name) - {o.layer for o in ordered}))
        raise TopologyError(
            f"Topology layers {missing} are not reachable from the cluster root; "
            "the layers must form a single chain."
        )

    return ordered


def effective_topology_labels(worker) -> Dict[str, str]:
    """What a worker's position is read from.

    The worker's own labels laid over what its runtime discovered
    (``status.topology_facts``). The order is the whole policy: a hand-filled
    value overrides a discovered one, and clearing the hand-filled key uncovers
    the discovered one again.
    """
    status = getattr(worker, "status", None)
    facts = getattr(status, "topology_facts", None) or {}
    labels = getattr(worker, "labels", None) or {}
    return {**facts, **labels}


def _domain_of(labels: Mapping[str, str], spec: TopologyLayerSpec):
    """The domain a worker belongs to at one layer, and the key that said so.

    Any-of: the declared keys are tried in order and the first one carrying a
    non-empty value wins. A blank value counts as absent — an empty label is a
    labelling accident, and treating it as a domain name would silently gather
    every half-labelled worker into one bogus domain.
    """
    for key in spec.label_keys:
        value = (labels or {}).get(key)
        if value:
            return value, key
    return None, None


def build_topology(
    specs: Sequence[TopologyLayerSpec],
    workers: Iterable,
) -> TopologyNode:
    """Build the tree. Declaration errors raise; data gaps do not.

    ``workers`` needs only ``id``, ``name`` and ``labels``; it is typed loosely
    so the scheduler can pass ORM rows and the tests can pass stubs.
    """
    ordered = order_layers([s for s in specs if s.layer != NODE_LAYER])
    root = TopologyNode(layer=ROOT_LAYER, name=ROOT_LAYER)

    for worker in workers:
        worker_id = getattr(worker, "id", None)
        if worker_id is None:
            continue
        labels = effective_topology_labels(worker)

        parent = root
        for spec in ordered:
            name, matched = _domain_of(labels, spec)
            if name is None:
                name = UNCLASSIFIED
            parent = _child(parent, spec.layer, name, matched)

        # The leaf is the worker itself, keyed by name rather than by any
        # label: this is the layer that must never collapse.
        leaf_name = getattr(worker, "name", None) or str(worker_id)
        leaf = _child(parent, NODE_LAYER, leaf_name, None)
        leaf.worker_ids.append(worker_id)

    return root


def _child(
    parent: TopologyNode, layer: str, name: str, matched_label_key: Optional[str]
) -> TopologyNode:
    for existing in parent.children:
        if existing.layer == layer and existing.name == name:
            return existing
    node = TopologyNode(
        layer=layer, name=name, parent=parent, matched_label_key=matched_label_key
    )
    parent.children.append(node)
    return node


def layer_names(specs: Sequence[TopologyLayerSpec]) -> List[str]:
    """Root-to-leaf layer names, leaf included.

    What the deployment form's "at least in the same ___" choices are built
    from. The leaf is appended unconditionally, which is why the tightest
    choice is offered even by a cluster that declared no topology at all.
    """
    return [s.layer for s in order_layers(list(specs))] + [NODE_LAYER]


def common_layer(a: TopologyNode, b: TopologyNode) -> Optional[str]:
    """The tightest layer whose domain contains both, or None.

    Two workers in the same unclassified bucket are *not* treated as close:
    the bucket means "we do not know where these are", and reading that as
    "these are together" would turn missing labels into confident wrong
    answers. This is the one place where the unclassified bucket behaves
    differently from a real domain.
    """
    # Walk b upward and stop at the first node a also sits under. By identity
    # rather than by position, so the answer does not depend on the two
    # branches having equal depth.
    ancestors_of_a = {id(node) for node in _ancestry(a)}
    for node in reversed(_ancestry(b)):
        if id(node) not in ancestors_of_a:
            continue
        if node.layer == ROOT_LAYER:
            # Sharing only the root is sharing nothing: every worker in the
            # cluster is under it.
            return None
        if node.is_unclassified:
            return None
        return node.layer
    return None


def _ancestry(node: TopologyNode) -> List[TopologyNode]:
    """Root-to-node, inclusive."""
    chain: List[TopologyNode] = []
    current: Optional[TopologyNode] = node
    while current is not None:
        chain.append(current)
        current = current.parent
    chain.reverse()
    return chain


def nodes_at_layer(root: TopologyNode, layer: str) -> List[TopologyNode]:
    """Every domain at one layer, in declaration order."""
    if root.layer == layer:
        return [root]
    out: List[TopologyNode] = []
    for child in root.children:
        out.extend(nodes_at_layer(child, layer))
    return out


def unclassified_at(root: TopologyNode, layer: str) -> List[int]:
    """Workers that fell into the unclassified bucket of one layer.

    The number the topology page leads with, because a worker landing here is
    the failure this whole module is most likely to hit and the one that
    reports nothing on its own.
    """
    out: List[int] = []
    for node in nodes_at_layer(root, layer):
        if node.is_unclassified:
            out.extend(node.descendant_worker_ids())
    return out


@dataclass
class GatherScope:
    """One rung of the search: a name and the candidate domains at that rung.

    **Every candidate set comes out of the tree**, ordered tightest first with
    the host at the front — including the accelerator domain, which is a rung
    an operator declares and is therefore built, walked and widened past
    exactly like a rack. A scope the tree did not produce would be a scope with
    no parent, and widening past it would have no defined next step.
    """

    name: str
    domains: List[TopologyNode]


def tree_scopes(root: TopologyNode, layers: Sequence[str]) -> List[GatherScope]:
    """The tree's layers as scopes, leaf first, without the cluster root.

    ``layers`` is root-to-leaf, as ``layer_names`` returns it.
    """
    return [
        GatherScope(layer, nodes_at_layer(root, layer)) for layer in reversed(layers)
    ]
