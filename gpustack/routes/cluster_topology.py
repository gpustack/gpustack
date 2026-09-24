"""Where this cluster's workers are, and the two ways of changing that.

**The page opens on a filled-in table, not an empty form.** `GET /topology`
returns every field of the vocabulary with whether it is in use, every worker
with the values it has (hand-filled or discovered by its runtime) and where
each came from, and the tree those values produce — one request, so the table,
the tree and the overview bar cannot disagree.

**One tree.** An accelerator domain is an ordinary layer on the chain the
operator declares, and reports the same numbers every other layer does — the
payload carries no second tree for it.

**A worker's `location` is a flat map** keyed by layer name, which is what lets
the table keep one column per layer name.

**Filling in a value is writing a label.** `POST /topology/locations` sets one
field on a batch of workers by writing the field's own key
(`topology.gpustack.ai/rack`) into `Worker.labels`. Nothing else is stored:
the position is the label, `worker_selector` can read it, and clearing it
uncovers whatever the worker discovered on its own. The response carries the
inverse assignments so the UI's undo is the same call with the previous values.

**Changing the mapping is a different act.** Which keys a field reads from is
the Advanced panel's business and goes through `PUT /clusters/{id}` after a
`POST /topology/preview` of the unsaved mapping. Values take effect at once;
mappings are previewed and saved. The two are kept apart on purpose.

**Counts, not just names.** Every domain carries its worker, GPU and free-GPU
totals, because "which rack can hold my 2P2D" is the question, and the
unclassified bucket carries the worker ids behind it so "3 workers have no
rack yet" is one click away from being fixed.
"""

import logging
from typing import Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from gpustack.api.exceptions import BadRequestException
from gpustack.api.tenant import assert_cluster_visible, assert_org_owned_writable
from gpustack.schemas.clusters import Cluster, ClusterTopology
from gpustack.schemas.workers import Worker
from gpustack.topology.tree import (
    NODE_LAYER,
    NODE_LAYER_SLUG,
    TopologyError,
    TopologyNode,
)
from gpustack.topology.view import TopologyView, build_view
from gpustack.topology.vocabulary import (
    KNOWN_KEYS,
    VOCABULARY_IDS,
    VOCABULARY_SLUGS,
    english_label,
    primary_key_for,
)
from gpustack.server.deps import SessionDep, TenantContextDep

router = APIRouter()

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# The view: what the page reads.                                              #
# --------------------------------------------------------------------------- #


class VocabularyFieldPublic(BaseModel):
    id: str
    name: str


class KnownKeyPublic(BaseModel):
    key: str
    vendor: str
    fits: List[str]
    note: str = ""


class VocabularyPublic(BaseModel):
    fields: List[VocabularyFieldPublic]
    """The built-in fields, root-to-leaf: zone, rack."""
    known_keys: List[KnownKeyPublic]
    """Label keys some vendor or tool is known to write, with the built-in rung
    each is nearest to. The accelerator-domain and switch keys live here rather
    than as fields of their own — they are facts the fleet publishes, and which
    rung they amount to is the operator's call."""


class TopologyLayerPublic(BaseModel):
    """One field of the vocabulary or one custom layer, with whether the fleet
    uses it. Non-active fields are what the "add a field" menu offers."""

    id: str
    name: str
    """Canonical name — the vocabulary slug, or a custom layer's original
    wording. The UI's i18n key; clients that do not localise can fall back to
    `english_label`."""
    display_name: Optional[str] = None
    """What the operator renamed it to; unset means never renamed. The
    effective label is `display_name or t(name)`, and it is never translated —
    these are the operator's words."""
    builtin: bool
    active: bool
    disabled: bool = False
    label_keys: List[str] = []
    primary_key: Optional[str] = None
    domains: int = 0
    classified: int = 0
    unclassified: int = 0
    referenced_by_models: List[str] = []
    """Models whose `gather.layer` names this layer. A custom layer with
    references cannot be deleted without stranding them, and the Advanced
    panel says which."""


class LocationPublic(BaseModel):
    value: str
    source: str
    """`user` (hand-filled label) or `discovered` (the worker's runtime)."""
    key: str
    discovered_value: Optional[str] = None
    """What the worker discovered, when a hand-filled value hides it; the UI
    says "clearing this restores nvl-a"."""
    display: Optional[str] = None
    """A readable name for an opaque value — the switch's system name beside
    its chassis id."""


class TopologyWorkerPublic(BaseModel):
    id: int
    name: str
    state: Optional[str] = None
    gpus: int = 0
    free_gpus: int = 0
    location: Dict[str, LocationPublic] = {}
    """This worker's values, keyed by layer id."""
    labels: Dict[str, str] = {}
    """The worker's own labels, so the Advanced panel can count who carries a
    key being typed without a second request."""


class TopologyDomainPublic(BaseModel):
    """One node of the tree, with the numbers the page leads with."""

    layer: str
    name: str
    unclassified: bool = False
    matched_label_key: Optional[str] = None
    workers: int = 0
    gpus: int = 0
    free_gpus: int = 0
    worker_ids: List[int] = []
    """Only populated on the unclassified bucket and the leaf: on the bucket it
    is what turns "3 workers have no rack" into one bulk action."""
    children: List["TopologyDomainPublic"] = []


class SuggestionPublic(BaseModel):
    key: str
    workers: int
    distinct_values: int
    looks_like: Optional[str] = None


class TopologyViewPublic(BaseModel):
    vocabulary: VocabularyPublic
    layers: List[TopologyLayerPublic]
    """Root-to-leaf, the host leaf last and always active."""
    workers: List[TopologyWorkerPublic]
    tree: TopologyDomainPublic
    total_workers: int = 0
    unclassified_workers: int = 0
    """Workers with no value at any *active* layer, deduplicated: the number
    the overview bar leads with."""
    suggestions: List[SuggestionPublic] = []


@router.get("/{id}/topology", response_model=TopologyViewPublic)
async def get_cluster_topology(session: SessionDep, ctx: TenantContextDep, id: int):
    """The saved mapping over the live fleet."""
    cluster = await Cluster.one_by_id(session, id)
    assert_cluster_visible(ctx, cluster, not_found_message=f"cluster {id} not found")
    workers = await Worker.all_by_field(session, "cluster_id", id)
    return await _view_public(
        cluster.topology, workers, await _gather_references(session, ctx, id)
    )


async def _gather_references(session, ctx, cluster_id: int) -> Dict[str, List[str]]:
    """layer name -> names of models whose `gather.layer` names it.

    Scoped to the caller, because a cluster is not a tenant boundary. A global
    cluster, or one sublet through `cluster_access`, passes
    `assert_cluster_visible` for a principal who owns none of the models on it
    -- and these names go out in the payload of every read, preview and
    locations call. So the list answers "which of YOUR models would this layer
    strand", which is also the only version of the question the caller can act
    on.

    It is not the deletion guard. That lives in `clusters.py`, sees every
    model, and still refuses a layer another principal's deployment needs --
    so the narrower list here can understate what blocks a delete, never
    permit one it should not.
    """
    from gpustack.api.tenant import tenant_list_conditions
    from gpustack.schemas.models import Model

    out: Dict[str, List[str]] = {}
    models = await Model.all_by_field(
        session,
        "cluster_id",
        cluster_id,
        extra_conditions=list(tenant_list_conditions(ctx, Model)),
    )
    for model in models:
        gather = getattr(model, "gather", None)
        layer = getattr(gather, "layer", None)
        if layer:
            out.setdefault(layer, []).append(model.name)
    return out


class TopologyPreviewRequest(BaseModel):
    """The mapping to preview. Absent means the saved one."""

    topology: Optional[ClusterTopology] = None


@router.post("/{id}/topology/preview", response_model=TopologyViewPublic)
async def preview_cluster_topology(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    body: Optional[TopologyPreviewRequest] = None,
):
    """The view `body.topology` would produce, without saving it.

    The Advanced panel's loop is "add a key, see who comes out of the unfilled
    bucket", and making that a save against a live cluster would turn a
    keystroke into a commitment. An invalid mapping is a 400, never a stored
    mistake; a worker missing a value is a normal state with a place in the
    result.
    """
    cluster = await Cluster.one_by_id(session, id)
    assert_cluster_visible(ctx, cluster, not_found_message=f"cluster {id} not found")
    topology = body.topology if body and body.topology is not None else cluster.topology
    workers = await Worker.all_by_field(session, "cluster_id", id)
    return await _view_public(
        topology, workers, await _gather_references(session, ctx, id)
    )


def _layers_public(
    view: TopologyView,
    worker_count: int,
    gather_refs: Dict[str, List[str]],
    host_display_name: Optional[str] = None,
) -> List[TopologyLayerPublic]:
    """The chain's rungs, root-to-leaf, with the host leaf appended.

    A *disabled* rung is in this list and absent from the tree. The panel has
    to draw the switch that turns it back on, which it cannot do for a rung
    `resolve()` left out entirely.
    """
    active_ids = {layer.id for layer in view.active}
    out: List[TopologyLayerPublic] = []
    for layer in view.resolved.layers:
        active = layer.id in active_ids
        classified = sum(1 for locs in view.locations.values() if layer.id in locs)
        out.append(
            TopologyLayerPublic(
                id=layer.id,
                name=layer.name,
                display_name=layer.display_name,
                builtin=layer.builtin,
                active=active,
                disabled=layer.disabled,
                label_keys=list(layer.label_keys),
                primary_key=layer.primary_key,
                domains=view.domain_count(layer.id) if active else 0,
                classified=classified,
                unclassified=worker_count - classified,
                referenced_by_models=sorted(gather_refs.get(layer.id, [])),
            )
        )
    # The host, last and always active.
    out.append(
        TopologyLayerPublic(
            id=NODE_LAYER,
            name=NODE_LAYER_SLUG,
            display_name=host_display_name,
            builtin=True,
            active=True,
            domains=worker_count,
            classified=worker_count,
            referenced_by_models=sorted(gather_refs.get(NODE_LAYER, [])),
        )
    )
    return out


def _locations_public(locs: Dict[str, object]) -> Dict[str, LocationPublic]:
    return {
        field_id: LocationPublic(
            value=loc.value,
            source=loc.source,
            key=loc.key,
            discovered_value=loc.discovered_value,
            display=loc.display,
        )
        for field_id, loc in locs.items()
    }


async def _view_public(
    topology, workers, gather_refs: Optional[Dict[str, List[str]]] = None
) -> TopologyViewPublic:
    gather_refs = gather_refs or {}
    try:
        view = build_view(topology, workers)
    except TopologyError as e:
        raise BadRequestException(message=str(e))

    capacity = await _worker_capacity(workers)
    by_id = {w.id: w for w in workers}

    workers_public = [
        TopologyWorkerPublic(
            id=w.id,
            name=getattr(w, "name", None) or str(w.id),
            state=_state_of(w),
            gpus=capacity.get(w.id, _Capacity()).gpus,
            free_gpus=capacity.get(w.id, _Capacity()).free_gpus,
            location=_locations_public(view.locations.get(w.id, {})),
            labels=dict(getattr(w, "labels", None) or {}),
        )
        for w in workers
        if getattr(w, "id", None) is not None
    ]

    return TopologyViewPublic(
        vocabulary=VocabularyPublic(
            fields=[
                VocabularyFieldPublic(id=i, name=english_label(VOCABULARY_SLUGS[i]))
                for i in VOCABULARY_IDS
            ],
            known_keys=[
                KnownKeyPublic(
                    key=k.key, vendor=k.vendor, fits=list(k.fits), note=k.note
                )
                for k in KNOWN_KEYS
            ],
        ),
        layers=_layers_public(
            view, len(workers), gather_refs, view.resolved.host_display_name
        ),
        workers=workers_public,
        tree=_to_public(view.root, capacity),
        total_workers=len(by_id),
        # Unfilled at any active layer, deduplicated: one worker missing two
        # fields is one worker to go and fill in, not two problems.
        unclassified_workers=len(view.unfilled_workers()),
    )


def _state_of(worker) -> Optional[str]:
    state = getattr(worker, "state", None)
    return getattr(state, "value", state) if state is not None else None


class _Capacity(BaseModel):
    gpus: int = 0
    free_gpus: int = 0


async def _worker_capacity(workers) -> Dict[int, _Capacity]:
    """Per worker: how many GPUs it has, and how many carry nothing.

    Allocation comes from ``get_worker_allocated``, which derives it from the
    current model-instance bindings — the same single source of truth the
    scheduler and the workers API read. Deriving it here from anything a worker
    self-reports would let the preview and the scheduler disagree about the
    same rack.
    """
    from gpustack.server.worker_allocated_cache import get_worker_allocated

    out: Dict[int, _Capacity] = {}
    for worker in workers:
        devices = (
            (getattr(worker.status, "gpu_devices", None) or []) if worker.status else []
        )
        indexes = [d.index for d in devices if d.index is not None]
        try:
            allocated = await get_worker_allocated(worker.id)
            used = {
                index
                for index, vram in (getattr(allocated, "vram", None) or {}).items()
                if vram
            }
        except Exception as e:
            # A worker whose allocation cannot be read is reported as fully
            # used rather than fully free: the preview's job is to answer "can
            # my group fit here", and an optimistic guess is the one answer
            # that sends an operator to a rack that cannot take the group.
            logger.warning(
                "Could not read allocation for worker %s; counting its GPUs as "
                "used in the topology preview: %s",
                getattr(worker, "name", worker.id),
                e,
            )
            used = set(indexes)
        out[worker.id] = _Capacity(
            gpus=len(indexes),
            free_gpus=len([i for i in indexes if i not in used]),
        )
    return out


def _to_public(
    node: TopologyNode, capacity: Dict[int, _Capacity]
) -> TopologyDomainPublic:
    children = [_to_public(child, capacity) for child in node.children]
    worker_ids = node.descendant_worker_ids()
    is_leaf = node.layer == NODE_LAYER

    if children:
        workers = sum(child.workers for child in children)
        gpus = sum(child.gpus for child in children)
        free_gpus = sum(child.free_gpus for child in children)
    else:
        workers = len(worker_ids)
        gpus = sum(capacity.get(wid, _Capacity()).gpus for wid in worker_ids)
        free_gpus = sum(capacity.get(wid, _Capacity()).free_gpus for wid in worker_ids)

    return TopologyDomainPublic(
        layer=node.layer,
        name=node.name,
        unclassified=node.is_unclassified,
        matched_label_key=node.matched_label_key,
        workers=workers,
        gpus=gpus,
        free_gpus=free_gpus,
        worker_ids=worker_ids if (node.is_unclassified or is_leaf) else [],
        children=children,
    )


# --------------------------------------------------------------------------- #
# Locations: filling a field in.                                              #
# --------------------------------------------------------------------------- #


class LocationAssignment(BaseModel):
    worker_ids: List[int]
    layer: str
    """A vocabulary field id, or a custom layer's name."""
    value: Optional[str] = None
    """None clears the field's own key. Other sources' keys are never touched,
    which is what lets a cleared hand-filled value uncover a discovered one."""


class LocationsRequest(BaseModel):
    assignments: List[LocationAssignment]


class LocationsPublic(BaseModel):
    previous: List[LocationAssignment]
    """The inverse of what was applied: POST it back to undo."""
    topology: TopologyViewPublic


@router.post("/{id}/topology/locations", response_model=LocationsPublic)
async def set_cluster_topology_locations(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    body: LocationsRequest,
):
    """Set one field on a batch of workers, by writing the field's own key.

    Effective at once — position is a label, and a label change only affects
    scheduling from here on; nothing running is moved. Refused only for a
    field the cluster does not have or a worker outside the cluster: there is
    no such thing as an invalid rack name.
    """
    from gpustack.server.services import WorkerService

    cluster = await Cluster.one_by_id(session, id)
    assert_cluster_visible(ctx, cluster, not_found_message=f"cluster {id} not found")
    assert_org_owned_writable(ctx, cluster, resource_label="cluster")

    workers = await Worker.all_by_field(session, "cluster_id", id)
    by_id = {w.id: w for w in workers}
    try:
        resolved_view = build_view(cluster.topology, workers)
    except TopologyError as e:
        # The same translation the read paths do. A stored declaration that
        # cannot become a tree is the caller's to fix, and letting it out as a
        # 500 here would make a mutating endpoint the one place that reports it
        # as a server fault.
        raise BadRequestException(message=str(e))

    # Resolve every assignment before writing any, so a bad one refuses the
    # whole batch instead of leaving half a rack renamed.
    planned: List[tuple] = []
    for assignment in body.assignments:
        key = primary_key_for(resolved_view.resolved, assignment.layer)
        if key is None:
            raise BadRequestException(
                message=f"{assignment.layer!r} is not a field that can be filled in."
            )
        for worker_id in assignment.worker_ids:
            worker = by_id.get(worker_id)
            if worker is None:
                raise BadRequestException(
                    message=f"worker {worker_id} is not in cluster {id}."
                )
            planned.append(
                (
                    worker,
                    assignment.layer,
                    key,
                    (assignment.value or "").strip() or None,
                )
            )

    # One transaction for the batch, because the paragraph above promises one:
    # validating everything before writing anything is worth nothing if the
    # writes then land one commit at a time, since a failure partway through
    # leaves exactly the half-renamed rack that promise rules out -- and the
    # caller gets a 500 instead of the `previous` it needs to undo what did
    # land. Rows are mutated here and committed together below.
    previous: Dict[tuple, Dict[int, Optional[str]]] = {}
    changed: Dict[int, Worker] = {}
    for worker, layer, key, value in planned:
        labels = dict(worker.labels or {})
        before = labels.get(key)
        if value is None:
            labels.pop(key, None)
        else:
            labels[key] = value
        if labels == (worker.labels or {}):
            continue
        # Assigned to the row rather than passed as a patch: one batch may name
        # the same worker for two different fields, and the second has to build
        # on the first rather than on what the request read.
        worker.labels = labels
        changed[worker.id] = worker
        previous.setdefault((layer, before), {})[worker.id] = before

    if changed:
        await WorkerService(session).batch_update(list(changed.values()))

    # One inverse assignment per (field, previous value): the undo of "these
    # three got R3" is "these two get back R1 and that one gets cleared".
    inverse = [
        LocationAssignment(worker_ids=sorted(ids), layer=layer, value=value)
        for (layer, value), ids in previous.items()
        if ids
    ]

    refreshed = await Worker.all_by_field(session, "cluster_id", id)
    return LocationsPublic(
        previous=inverse,
        topology=await _view_public(
            cluster.topology, refreshed, await _gather_references(session, ctx, id)
        ),
    )
