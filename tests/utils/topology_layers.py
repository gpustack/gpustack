"""Test-side spelling of a topology layer's identity.

A layer's id is opaque by design (`builtin-000003`, `custom-a7f3c1`), which is
right for stored data and unreadable in a test — an assertion about
`builtin-000003` says nothing about racks. These helpers let a test go on
naming layers in domain terms while exercising the real id path: `lid("rack")`
is the vocabulary's registry number, and `lid("Hall")` is a stable made-up
custom id, so the same call spells a layer, its children's `parentLayer` and a
model's `gather.layer` consistently.

Stable rather than random on purpose: a test that writes
`{"parentLayer": lid("Hall")}` and later asserts on `lid("Hall")` needs the
two to agree, and a fixture that changed between calls would be a source of
flakes rather than of coverage.
"""

import hashlib
from types import SimpleNamespace

from gpustack.topology.tree import NODE_LAYER, NODE_LAYER_SLUG
from gpustack.topology.vocabulary import VOCABULARY

BUILTIN_IDS = {f.slug: f.id for f in VOCABULARY}
BUILTIN_IDS[NODE_LAYER_SLUG] = NODE_LAYER


def lid(name: str) -> str:
    """The layer id a test means by ``name``.

    A vocabulary slug resolves to its registry number; anything else is a
    custom layer and gets a deterministic `custom-` id derived from the name.
    """
    builtin = BUILTIN_IDS.get(name)
    if builtin:
        return builtin
    digest = hashlib.blake2s(name.encode("utf-8"), digest_size=3).hexdigest()
    return f"custom-{digest}"


def layer_dict(name, keys=(), parent=None, display_name=None, disabled=False):
    """One entry of ``ClusterTopology.layers``, in wire (camelCase) form.

    ``name`` is both the canonical name the entry carries and the handle the
    test refers to it by — which is exactly the production rule: a built-in
    rung's `name` is its slug, a custom one's is what the operator typed.
    """
    out = {"id": lid(name), "name": name, "labelKeys": list(keys)}
    if parent is not None:
        out["parentLayer"] = lid(parent)
    if display_name is not None:
        out["displayName"] = display_name
    if disabled:
        out["disabled"] = True
    return out


def layer_obj(name, keys=(), parent=None, display_name=None, disabled=False):
    """The same entry as a duck-typed stub, for the resolver's own tests.

    ``resolve`` reads attributes rather than a `ClusterTopology`, so the unit
    tests stub it; this keeps the two spellings in one place so they cannot
    drift.
    """
    return SimpleNamespace(
        id=lid(name),
        name=name,
        label_keys=list(keys),
        parent_layer=lid(parent) if parent is not None else None,
        display_name=display_name,
        disabled=disabled,
    )
