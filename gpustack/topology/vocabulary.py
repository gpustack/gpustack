"""The fixed vocabulary of places a worker can be, and how a worker's place is read.

An operator describes a machine room, not a schema: "node-9 is in rack R3".
The layers worth declaring for everyone are a small, stable set — the zone,
the rack, the host — so they are declared once here, in order, and a cluster
only ever fills in *values*. A field with a value on at least one worker is a
layer of that cluster's tree; a field nobody filled in is not. There is no
"declare a layer" step for the built-in rungs.

**One vocabulary, because there is one chain.** The NVLink/HCCS/UB domain is
not a vocabulary of its own: a domain whose boundary is a run of contiguous
cabinets is expressible as one rung of this chain, and on every shipping
generation it is contiguous; where the domain sits *inside* one machine
(910B2), "same domain" and "same host" are the same constraint and the
built-in leaf already covers it. The operator decides where the domain rung
goes, and the keys the domain is published under are **candidate keys**
(``KNOWN_KEYS``).

**Custom layers** are for every rung this list does not name — the accelerator
domain among them. They live in ``Cluster.topology.layers`` with the parent
chain the tree has always used, and slot between the vocabulary's fields.

Every field owns one key under ``topology.gpustack.ai/``, listed first among its
candidates. That is what the table writes when an operator fills in a value,
and because it is tried first, a hand-filled value always wins over whatever a
cloud, a device or a discovery tool wrote under another key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from gpustack.topology.tree import (
    NODE_LAYER,
    NODE_LAYER_SLUG,
    ROOT_LAYER,
    TopologyLayerSpec,
    effective_topology_labels,
)

GPUSTACK_PREFIX = "topology.gpustack.ai/"


@dataclass(frozen=True)
class VocabularyField:
    id: str
    """Registry number, never reused and never renumbered.

    Opaque on purpose. An id and a display name drawn from the same machine-room
    vocabulary cannot be told apart in stored data — ``{id: "rack", name:
    "zone"}`` reads as a contradiction rather than as "the rack rung, which
    this operator calls a zone". A number belongs to no vocabulary, so the two
    value spaces cannot overlap.

    **The number is allocation order, not chain order.** Today the two happen
    to agree; a layer inserted above ``zone`` later would take ``000005``, not
    ``000000``. Sorting by it is always wrong — the chain is ``parent_layer``.

    Hard-coded rather than derived from the tuple's position, and that is the
    whole point: the vocabulary is edited as hardware and practice change, and
    positional numbering would silently repoint every stored
    ``Model.gather.layer`` at a different layer.
    """

    slug: str
    """The canonical name, and the i18n lookup key. Reaches the wire as a
    layer's ``name``; what an operator sees is ``display_name or t(slug)``."""

    label_keys: Tuple[str, ...]
    """Any-of, first present wins. The ``topology.gpustack.ai/`` key is first."""

    @property
    def primary_key(self) -> str:
        return self.label_keys[0]


# Root-to-leaf. The order is the one thing this module refuses to let a
# cluster change: a tree whose zones sit inside racks is not a tree anyone
# recognises, and a fixed order is what lets two clusters mean the same thing
# by "rack".
#
# Two rungs, `zone` over `rack`, and only two: more physical rungs than that
# is more structure than an operator will fill in. `zone` earns its place by
# having a standard behind it — `topology.kubernetes.io/zone`, a Kubernetes
# well-known label — so a fleet on k8s lands in the right rung with no
# labelling at all.
#
# `switch` stays out, and now has no key of ours at all: nothing in the fleet
# publishes one, and a rung with no source is a rung that can only be empty. An
# operator who wants to place by the switch declares a layer over whatever key
# their own tooling writes — Topograph's `fabric.topograph.run/tier-0`, say —
# which is the same operation as adding one for the accelerator domain.
#
# `builtin-000002` is retired and deliberately NOT reused. Ids are allocation
# order and a stored `Model.gather.layer` points at one; handing a retired
# number to something else would silently repoint every declaration that still
# names it.
VOCABULARY: Tuple[VocabularyField, ...] = (
    VocabularyField(
        "builtin-000001",
        "zone",
        (GPUSTACK_PREFIX + "zone", "topology.kubernetes.io/zone"),
    ),
    VocabularyField(
        "builtin-000003",
        "rack",
        (GPUSTACK_PREFIX + "rack", "topology.kubernetes.io/rack"),
    ),
)

VOCABULARY_IDS = tuple(f.id for f in VOCABULARY)

VOCABULARY_SLUGS: Dict[str, str] = {f.id: f.slug for f in VOCABULARY}
"""Built-in id -> canonical name. What a stored layer's ``name`` must equal
while it is one of ours (``validate_declaration``), so a client cannot write
the *translated* label there and freeze the row into one person's UI language."""

RESERVED_IDS = frozenset({ROOT_LAYER, NODE_LAYER})
"""Names no declared layer may take: the implicit root, and the leaf.

Only these two, because they are the names that are not layers at all. A
vocabulary id is not reserved: naming ``rack`` in the declaration is how an
operator overrides that field's label keys, and ``accelerator_domain`` is free
for the recommended spelling of the domain rung as a custom layer."""


def declared_layers(topology) -> List:
    """The entries a cluster declared.

    One accessor rather than every call site reaching for the attribute, so a
    ``ClusterTopology`` and a test stub are read the same way.
    """
    return list(getattr(topology, "layers", None) or [])


@dataclass(frozen=True)
class KnownKey:
    """A label key some vendor or tool is known to write, offered in the
    Advanced panel so nobody has to remember the spelling."""

    key: str
    vendor: str
    fits: Tuple[str, ...]
    note: str = ""


# ``fits`` names the built-in rung a key is *nearest* to, and that is all it
# is: a hint for where to insert the layer that reads it. The accelerator-domain
# key is in here rather than in ``VOCABULARY`` for the reason at the top of this
# module — it is a fact the fleet publishes, and which rung it amounts to is the
# operator's call, not ours.
#
# A key earns its place here by having a source. Offering one that nothing
# writes and nobody can type advertises a rung that can only ever be empty,
# which is worse than leaving the operator to paste the key themselves.
KNOWN_KEYS: Tuple[KnownKey, ...] = (
    KnownKey(
        GPUSTACK_PREFIX + "accelerator-domain",
        "GPUStack",
        ("rack", "zone"),
        "NVLink/HCCS/UB domain. Hand-filled for now: on Ascend the super pod "
        "id is read with npu-smi, and nothing publishes it automatically.",
    ),
    KnownKey(
        "nvidia.com/gpu.clique",
        "NVIDIA",
        ("rack", "zone"),
        "NVLink domain, written by the driver.",
    ),
    KnownKey(
        "accelerator.topograph.run/domain",
        "Topograph",
        ("rack", "zone"),
        "NVLink domain as Topograph discovers it.",
    ),
    KnownKey(
        "network.topology.nvidia.com/accelerator",
        "NVIDIA",
        ("rack", "zone"),
        "NVLink domain.",
    ),
    KnownKey(
        "fabric.topograph.run/tier-0",
        "Topograph",
        ("rack",),
        "The switch closest to the node.",
    ),
    KnownKey(
        "fabric.topograph.run/tier-1",
        "Topograph",
        ("zone",),
        "One tier above the leaf switch.",
    ),
    KnownKey(
        "fabric.topograph.run/tier-2",
        "Topograph",
        ("zone",),
        "Two tiers above the leaf switch.",
    ),
    KnownKey(
        "network.topology.nvidia.com/block",
        "NVIDIA",
        ("rack", "zone"),
        "IB fabric block.",
    ),
    KnownKey(
        "network.topology.nvidia.com/spine",
        "NVIDIA",
        ("zone",),
        "IB fabric spine.",
    ),
    KnownKey(
        "network.topology.nvidia.com/datacenter",
        "NVIDIA",
        ("zone",),
        "IB fabric datacenter.",
    ),
    KnownKey(
        "cloud.google.com/gce-topology-subblock",
        "GKE",
        ("rack",),
        "On A4X this is the NVL72 domain.",
    ),
    KnownKey(
        "cloud.google.com/gce-topology-block",
        "GKE",
        ("zone",),
        "One fast network.",
    ),
    KnownKey(
        "topology.k8s.aws/network-node-layer-3",
        "EKS",
        ("rack",),
        "Finest EKS network layer.",
    ),
    KnownKey(
        "topology.k8s.aws/ultraserver-id",
        "EKS",
        ("rack",),
        "GB200 UltraServer NVL72 domain.",
    ),
    KnownKey(
        "ds.coreweave.com/nvlink.domain",
        "CoreWeave",
        ("rack",),
        "NVL72 domain.",
    ),
    KnownKey(
        "topology.kubernetes.io/region",
        "Kubernetes",
        ("zone",),
        "Well-known region label.",
    ),
)


def source_of(worker, key: str) -> str:
    """Which side of the merge a key came from, for the UI's source badge."""
    labels = getattr(worker, "labels", None) or {}
    return "user" if key in labels else "discovered"


@dataclass(frozen=True)
class ResolvedLayer:
    """One rung of a cluster's chain, vocabulary or custom, keys resolved."""

    id: str
    name: str
    """Canonical name. The vocabulary slug for a built-in rung, the operator's
    original wording for a custom one. Never the translated label."""

    label_keys: Tuple[str, ...]
    builtin: bool
    display_name: Optional[str] = None
    """What the operator renamed it to; unset means never renamed."""

    disabled: bool = False

    @property
    def primary_key(self) -> Optional[str]:
        return self.label_keys[0] if self.label_keys else None

    @property
    def label(self) -> str:
        """What to call this rung where no translation is available.

        The UI has the catalogue and does the real thing — ``displayName or
        t(name)``. This is the same rule with ``t`` as identity, for log lines
        and error messages, so a server-side sentence naming a layer says what
        the operator calls it rather than ``builtin-000003``.
        """
        return self.display_name or self.name

    def spec(self, parent: Optional[str]) -> TopologyLayerSpec:
        return TopologyLayerSpec(
            layer=self.id, label_keys=self.label_keys, parent_layer=parent
        )


@dataclass
class ResolvedTopology:
    """A cluster's declaration with the vocabulary filled in, root-to-leaf.

    ``layers`` is every rung whether or not any worker has a value for it;
    ``active`` is the subset a tree is built from.
    """

    layers: List[ResolvedLayer] = field(default_factory=list)

    host_display_name: Optional[str] = None
    """What the operator renamed the leaf to, if anything.

    Held here rather than as a rung of ``layers``: the leaf is not declared,
    reads no label and cannot be disabled or reordered, so putting it in the
    list would mean every loop over the chain had to special-case the last
    element. The only thing about it an operator can change is what it is
    called."""

    def layer(self, layer_id: str) -> Optional[ResolvedLayer]:
        for layer in self.layers:
            if layer.id == layer_id:
                return layer
        return None

    def active(self, workers: Iterable) -> List[ResolvedLayer]:
        """The layers at least one worker has a value for.

        This is the rule that replaces declaring: fill a field in and the tree
        grows a layer; leave it empty and it does not exist. Custom layers are
        kept even when empty — an operator who wrote one down wants to see
        that nobody matches it, and the preview says so — but they still do not
        become a tier the deployment form can ask for.
        """
        labels = [effective_topology_labels(w) for w in workers]
        out: List[ResolvedLayer] = []
        for layer in self.layers:
            # Switched off by the operator: not a tier, not a rung of the
            # tree, however many workers carry its label. This is the one
            # difference from "nobody filled it in" — that one comes back by
            # itself the moment a label appears, and this one does not.
            if layer.disabled:
                continue
            if not layer.builtin or any(
                _has_value(lb, layer.label_keys) for lb in labels
            ):
                out.append(layer)
        return out

    def specs(self, layers: Sequence[ResolvedLayer]) -> List[TopologyLayerSpec]:
        """Chain the given layers into what ``build_topology`` takes."""
        specs: List[TopologyLayerSpec] = []
        parent: Optional[str] = None
        for layer in layers:
            specs.append(layer.spec(parent))
            parent = layer.id
        return specs


def _has_value(labels: Mapping[str, str], keys: Sequence[str]) -> bool:
    return any(labels.get(k) for k in keys)


def resolve(topology) -> ResolvedTopology:
    """Fill the vocabulary into a cluster's ``ClusterTopology`` (or None).

    Empty declarations are the common case and mean the vocabulary as-is:
    zone/rack. A non-empty list is the Advanced panel's work — an entry
    carrying a vocabulary **id** replaces that field's keys or renames it, and
    any other entry is a custom layer whose place is fixed by its
    ``parent_layer``.

    Custom layers are spliced in by their parent: right below the parent they
    name, which may be a vocabulary field or another custom layer. A custom
    layer with no parent sits at the top, above the vocabulary. The vocabulary
    itself never moves.

    A disabled built-in rung is kept here and dropped by ``active()``. Both
    kinds of absence end at the same place — not a rung of the tree, not a
    gather tier — but only one of them comes back by itself: a rung nobody
    filled in returns the moment a worker grows the label, a disabled one
    stays gone until the operator says otherwise. Keeping it in this list is
    also what leaves the Advanced panel a row to draw the switch on.
    """
    builtin_ids = {v.id for v in VOCABULARY}

    declared: Dict[str, object] = {}
    customs = []
    host_display_name: Optional[str] = None
    for entry in declared_layers(topology):
        entry_id = getattr(entry, "id", None)
        if not entry_id:
            continue
        if entry_id == NODE_LAYER:
            # The leaf is declarable for exactly one reason — to rename it —
            # so it is read for that and otherwise ignored. Falling through to
            # `customs` would splice a second host into the middle of the
            # chain, which is the failure this branch exists to prevent.
            host_display_name = getattr(entry, "display_name", None)
        elif entry_id in builtin_ids:
            declared[entry_id] = entry
        else:
            customs.append(entry)

    layers: List[ResolvedLayer] = []
    for vocab in VOCABULARY:
        entry = declared.get(vocab.id)
        # A disabled rung stays in this list. It is dropped from `active()`
        # instead, which is what keeps it out of the tree and out of the
        # gather tiers. Skipping it here removed it from `GET /topology` as
        # well, and the Advanced panel draws its rows from that — so the rung
        # vanished on the next open and there was no switch left to turn it
        # back on. Disabling was a one-way door.
        keys = tuple(getattr(entry, "label_keys", None) or ()) or vocab.label_keys
        # The owned key stays first whatever the override said: it is the key
        # the table writes, and if it were not tried first a hand-filled value
        # could lose to a discovered one — the one ordering this design forbids.
        keys = (vocab.primary_key,) + tuple(k for k in keys if k != vocab.primary_key)
        layers.append(
            ResolvedLayer(
                vocab.id,
                # Never the stored `name`. A built-in rung's canonical name is
                # the vocabulary's to define; reading it back from the row
                # would let one bad write turn into a permanent wrong answer.
                vocab.slug,
                keys,
                builtin=True,
                display_name=getattr(entry, "display_name", None),
                disabled=bool(getattr(entry, "disabled", False)),
            )
        )

    # Splice customs below their parent. Repeated until stable so a custom
    # layer under another custom layer lands after both are placed.
    pending = list(customs)
    while pending:
        progressed = False
        for entry in list(pending):
            parent = getattr(entry, "parent_layer", None)
            keys = tuple(getattr(entry, "label_keys", None) or ())
            layer = ResolvedLayer(
                entry.id,
                getattr(entry, "name", None) or entry.id,
                keys,
                builtin=False,
                display_name=getattr(entry, "display_name", None),
            )
            if parent is None:
                layers.insert(0, layer)
            else:
                index = next((i for i, x in enumerate(layers) if x.id == parent), None)
                if index is None:
                    continue
                layers.insert(index + 1, layer)
            pending.remove(entry)
            progressed = True
        if not progressed:
            # A parent that does not exist, or one that exists but is disabled.
            # The schema validator refuses the first on save; either way the
            # entry is dropped here so a stale row cannot take the scheduler
            # down.
            break

    return ResolvedTopology(layers, host_display_name=host_display_name)


def _validate_layer(layer, builtin_ids, custom_ids, seen) -> None:
    """Everything one declared layer has to satisfy on its own.

    Split out of `validate_declaration` for its branch count alone: the checks
    below judge a single row, and the ones that stayed behind judge how the
    rows fit together. Keeping them in one function meant a reader tracing a
    fork error walked past ten single-row rules to reach it.
    """
    from gpustack.topology.tree import TopologyError

    if not layer.id:
        raise TopologyError("A topology layer must have an id.")
    if not layer.name:
        raise TopologyError(f"Topology layer {layer.id!r} must have a name.")
    if layer.id in seen:
        raise TopologyError(f"Duplicate topology layer {layer.id!r}.")
    seen.add(layer.id)
    if layer.id == ROOT_LAYER:
        raise TopologyError(f"{layer.id!r} is reserved and cannot be a layer.")
    if layer.id == NODE_LAYER:
        # Declarable, but only to rename. The leaf takes the worker's own
        # name and is what makes a tree survive a fleet with no labels at
        # all; letting it read a label or move would put that guarantee in
        # the operator's hands, which is the one place it must not be.
        if layer.label_keys:
            raise TopologyError(
                "The host layer takes the worker's name and cannot read " "label keys."
            )
        if layer.parent_layer:
            raise TopologyError(
                "The host layer is always the leaf and cannot name a parent."
            )
        if layer.disabled:
            raise TopologyError("The host layer cannot be disabled.")
        return
    # A built-in rung's canonical name belongs to the vocabulary, not to
    # the client. Refused rather than quietly corrected: a client writing
    # the *translated* label here is a real bug with a silent symptom —
    # every reader in every other language gets one operator's UI language
    # — and correcting it server-side would hide the bug while the wrong
    # value kept arriving.
    expected = VOCABULARY_SLUGS.get(layer.id)
    if expected is not None and layer.name != expected:
        raise TopologyError(
            f"Built-in topology layer {layer.id!r} must keep the name "
            f"{expected!r}; rename it with displayName instead of {layer.name!r}."
        )
    if layer.disabled and layer.id not in builtin_ids:
        raise TopologyError(
            f"Topology layer {layer.id!r} is custom and cannot be disabled; "
            "delete it instead."
        )
    # A vocabulary entry's `parent_layer` is ignored rather than refused:
    # its place in the chain is fixed, and a client that serialises the
    # whole chain uniformly (each entry pointing at its predecessor) is
    # not wrong about anything that matters.
    if (
        layer.id not in builtin_ids
        and layer.parent_layer
        and layer.parent_layer not in builtin_ids
        and layer.parent_layer not in custom_ids
    ):
        raise TopologyError(
            f"Topology layer {layer.id!r} names an unknown parent "
            f"{layer.parent_layer!r}."
        )


def validate_declaration(topology) -> ResolvedTopology:
    """Refuse a declaration that cannot become a tree; return it resolved.

    Only the declaration is judged, never the data: a custom layer naming a
    parent that does not exist, taking a reserved id, colliding with another
    rung's label or forking the chain means the operator's intent is
    unknowable, while a worker missing a value is a normal state the tree has a
    place for. Raised as ``TopologyError`` so the schema validator and the
    scheduler refuse the same declarations for the same reasons.
    """
    from gpustack.topology.tree import TopologyError, layer_names

    resolved = resolve(topology)

    builtin_ids = {v.id for v in VOCABULARY}
    layers = declared_layers(topology)
    seen = set()
    # The leaf is neither built in nor custom. Leaving it out of both sets is
    # load-bearing: in `custom_ids` it would be reported unreachable (it is
    # never a rung of `resolved.layers`), and among the parents below it would
    # look like a second top-level layer and read as a fork.
    declarable = builtin_ids | {NODE_LAYER}
    custom_ids = {
        layer_.id for layer_ in layers if layer_.id and layer_.id not in declarable
    }
    for layer in layers:
        _validate_layer(layer, builtin_ids, custom_ids, seen)

    # A fork: two custom layers under the same parent (or two at the top).
    # The tree the scheduler walks has one path from root to leaf, and a fork
    # would make "how many layers up" have no single answer.
    parents = [layer_.parent_layer for layer_ in layers if layer_.id in custom_ids]
    if len(parents) != len(set(parents)):
        raise TopologyError(
            "Two custom topology layers share a parent; the layers must form a "
            "single chain."
        )

    placed = {layer.id for layer in resolved.layers}
    unreachable = sorted(cid for cid in custom_ids if cid not in placed)
    if unreachable:
        raise TopologyError(
            f"Topology layers {', '.join(unreachable)} are not reachable from the "
            "cluster root; the layers must form a single chain."
        )

    # Two rungs the operator cannot tell apart. The deployment form's "at least
    # in the same ___" is a list of these labels, so a duplicate is two options
    # with one word on them.
    labels = [layer.label.strip().casefold() for layer in resolved.layers] + [
        (resolved.host_display_name or english_label(NODE_LAYER_SLUG))
        .strip()
        .casefold()
    ]
    duplicate = next((lb for lb in labels if labels.count(lb) > 1), None)
    if duplicate:
        raise TopologyError(
            f"Two topology layers are both called {duplicate!r}; layer names must "
            "be distinct."
        )

    layer_names(resolved.specs(resolved.layers))
    return resolved


def gather_layer_names(resolved: ResolvedTopology) -> List[str]:
    """Every layer a saved ``gather.layer`` may name: the host, then the chain.

    Disabled rungs are not among them. They stay in ``layers`` so the panel
    can switch them back on, but they are not tiers — offering one would let
    a `MustGather` name a rung the solver never groups by, which is the
    unenforceable promise `GatherSpec` exists to prevent.
    """
    return [NODE_LAYER] + [layer.id for layer in resolved.layers if not layer.disabled]


def primary_key_for(resolved: ResolvedTopology, field_id: str) -> Optional[str]:
    """The key the table writes for a field, or None for a field with none.

    The leaf has no key: a host is itself. A custom layer's first key is what
    it writes, which is why the Advanced panel tells an operator to put the
    key they mean to write first.
    """
    if field_id == NODE_LAYER:
        return None
    layer = resolved.layer(field_id)
    return layer.primary_key if layer else None


_ENGLISH = {
    "zone": "Zone",
    "rack": "Rack",
    NODE_LAYER_SLUG: "Host",
}


def english_label(name: str) -> str:
    """An English fallback for a canonical name, for clients that do not
    localise. Keyed by the slug, never by the id: an id is a registry number
    and has no English.

    The UI ignores this and renders ``displayName or t(name)`` itself, which is
    why an operator's own wording never passes through here."""
    return _ENGLISH.get(name, name)
