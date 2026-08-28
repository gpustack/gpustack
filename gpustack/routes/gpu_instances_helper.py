"""Shared helpers for the GPU instance-type / flavor routes.

Most helpers sit on top of :class:`ClusterOps` (the raw ``worker.gpustack.ai/v1``
CRD client) and translate cluster access + Kubernetes failures into the
project's HTTP semantics, so the route modules stay thin. :func:`watch_event_stream`
is a transport-level helper shared by both the per-cluster and the aggregated
watch routes, framing a normalized watch source as GPUStack SSE.
:func:`display_name_label` / :func:`order_by_display_label` are shared by the
five GPU Service lists whose Name column carries a sorter, so the one rule about
what that column shows and orders by has one definition. Templates are a card
view with no column sorter, so they search the display name without ordering by
it and do not use these.
"""

import asyncio
import http
import json
import logging
from contextlib import aclosing, asynccontextmanager, suppress
from typing import Any, AsyncIterator, Callable, List, Optional, Tuple

from fastapi import Request
from fastapi.encoders import jsonable_encoder
from kubernetes_asyncio import client
from sqlalchemy import func
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.exceptions import (
    AlreadyExistsException,
    ConflictException,
    InternalServerErrorException,
    InvalidException,
    NotFoundException,
    ServiceUnavailableException,
)
from gpustack.api.tenant import (
    TenantContext,
    assert_cluster_visible,
    assert_cluster_writable,
)
from gpustack.gpu_instances.cluster_apis import ClusterOps
from gpustack.gpu_instances.cluster_apis_util import principal_namespace_identifier
from gpustack.schemas import Cluster
from gpustack.schemas.clusters import is_gpu_service_cluster
from gpustack.schemas.principals import PLATFORM_PRINCIPAL_NAME, Principal
from gpustack.server.bus import Event, EventType

logger = logging.getLogger(__name__)

# Kubernetes watch verbs → GPUStack EventType. Any other verb (notably
# BOOKMARK / ERROR) is absent here and dropped from the stream, so a GPUStack
# client only ever sees the project's own event types.
_K8S_TO_GPUSTACK = {
    "ADDED": EventType.CREATED,
    "MODIFIED": EventType.UPDATED,
    "DELETED": EventType.DELETED,
}

# Idle keepalive cadence, matching ActiveRecord.subscribe's 15s heartbeat.
_HEARTBEAT_INTERVAL = 15.0

# Bound the producer→consumer buffer so a slow client applies backpressure to
# the watch instead of growing memory without limit.
_WATCH_QUEUE_MAXSIZE = 100

# Producer-done sentinel: dequeuing it means the source ended (vs. a wait_for
# timeout, which means "no event yet — send a heartbeat").
_DONE = object()


def display_name_label(model: Any) -> Any:
    """SQL for what a GPU Service list renders in its Name column.

    ``display_name || name`` as the UI writes it, where ``nullif(..., '')`` is
    what makes an empty display name fall back the way a falsy ``||`` does.

    ``model.display_name`` is a real column on every GPU Service table but
    ``gpu_instance_types``, where it lives inside the ``spec`` JSON and the model
    exposes it as a hybrid instead — so one call covers all of them.
    """
    return func.coalesce(func.nullif(model.display_name, ""), model.name)


def order_by_display_label(
    order_by: Optional[List[Tuple[str, str]]],
    label: Any,
) -> Optional[List[Tuple[Any, str]]]:
    """Point a ``sort_by=name`` request at the label the Name column renders.

    These Name columns are ``sorter: true`` and send ``sort_by=name``, so
    ordering on the ``name`` column sorted each column by a value it does not
    display. Shared rather than repeated per route because it is one product
    rule over five lists, and a rule written five times is one that drifts.

    ``sortable_fields`` stays as it is: the wire name is still ``name``, because
    it is the Name column being sorted. The consequence, accepted, is that
    ``sort_by=name`` orders by the displayed label. Every other sortable field
    passes through untouched, for ``paginated_by_query`` to resolve by attribute
    lookup.

    The label is expanded rather than substituted, and the result always ends on
    a unique key, because ``LIMIT``/``OFFSET`` paging over a non-unique sort key
    is not deterministic: rows sharing a key have engine-defined relative order,
    which can differ between the page-1 and the page-2 query, so one row comes
    back twice and another never. Labels collide by construction — the operator
    stamps ``CPU-only`` on every cluster's collapsed generic pool — and ``name``
    does not settle it either: ``gpu_instance_types`` is unique on ``snapshot``,
    the rest only per owner, while these lists span owners and clusters. So the
    label orders, ``name`` breaks ties meaningfully, and ``id`` guarantees a
    total order.
    """
    if not order_by:
        return order_by

    translated: List[Tuple[Any, str]] = []
    for field, direction in order_by:
        if field == "name":
            translated.append((label, direction))
        translated.append((field, direction))

    if not any(field == "id" for field, _ in translated):
        translated.append(("id", translated[-1][1]))
    return translated


def ensure_visible(obj: Cluster, ctx: TenantContext) -> Cluster:
    """Return the cluster if the caller can see it, else raise 404.
    ``assert_cluster_visible`` handles ``obj is None`` (missing → 404)."""
    assert_cluster_visible(ctx, obj)
    return obj


def ensure_writable(obj: Cluster, ctx: TenantContext) -> Cluster:
    """Return the cluster if the caller can write it, else raise 404 when it is
    missing/invisible or 403 when it is visible-but-not-owned. Visibility is
    checked first so a write to a cluster the caller cannot see 404s (does not
    leak its existence) instead of 403-ing on ownership."""
    assert_cluster_visible(ctx, obj)
    assert_cluster_writable(ctx, obj)
    return obj


def assert_cluster_gpu_service(cluster: Cluster) -> None:
    """Refuse a cluster that is not registered for GPU Service.

    A cluster's purpose is decided at registration and is the presence of
    ``k8s_options.gpu_instance_options`` (see ``schemas/clusters.py``). A Model
    Service cluster's capacity is committed to model deployment, so it has no
    GPU-instance infrastructure to write to.

    409 rather than 404 or 403: the caller may well see the cluster and own it
    — what is wrong is the cluster's purpose, which is a thing the caller can
    change. Run it *after* the visibility/ownership checks so a cluster the
    caller cannot see still 404s and never has its purpose disclosed.

    Writes only. The per-cluster instance-type **read** deliberately keeps
    serving Model Service clusters: it is what the model deploy form's slicing
    picker reads.
    """
    if is_gpu_service_cluster(cluster):
        return
    raise ConflictException(
        message=(
            f"Cluster '{cluster.name}' is registered for model service, "
            f"not GPU service. Switch the cluster to GPU service first."
        )
    )


async def build_cluster_ops(
    request: Request, session: AsyncSession, cluster: Cluster
) -> ClusterOps:
    """Build a :class:`ClusterOps` for the (already access-checked) cluster.

    The owner identifier only derives the org namespace of namespaced CRDs;
    the instance-type / flavor CRDs are cluster-scoped, so it is irrelevant to
    their calls but the constructor still requires it — fall back to the
    platform identifier for a NULL-owner (global) cluster.
    """
    principal = (
        await Principal.one_by_id(session, cluster.owner_principal_id)
        if cluster.owner_principal_id is not None
        else None
    )
    owner_identifier = (
        principal_namespace_identifier(principal)
        if principal
        else PLATFORM_PRINCIPAL_NAME
    )

    return ClusterOps(
        server_api_port=request.app.state.server_config.get_api_port(),
        cluster_id=cluster.id,
        cluster_registration_token=cluster.registration_token,
        cluster_owner_principal_identifier=owner_identifier,
    )


@asynccontextmanager
async def handle_error(already_exists_message: Optional[str] = None):
    """Translate a Kubernetes ``ApiException`` into the project's HTTP
    exceptions so a failure surfaces as the right status instead of a
    blanket 500.

    ``already_exists_message`` overrides the message of the 409 branch: the
    upstream exception only carries the bare HTTP reason phrase
    (``"Conflict"``), which does not tell the caller what already exists.
    """
    try:
        yield
    except client.exceptions.ApiException as e:
        message = getattr(e, "reason", None) or str(e)
        if e.status == http.HTTPStatus.NOT_FOUND:
            raise NotFoundException(message=message)
        if e.status == http.HTTPStatus.CONFLICT:
            raise AlreadyExistsException(message=already_exists_message or message)
        if e.status == http.HTTPStatus.BAD_REQUEST:
            raise InvalidException(message=message)
        # One branch for all three: to the caller they are the same thing —
        # the upstream never answered and a retry may help. Most often it is a
        # cluster with no ready worker, whose proxy 503s. Folding them into the
        # 500 fallback made the response contradict itself: a 500 whose own
        # message read "Service Unavailable" (#6071).
        if e.status in (
            http.HTTPStatus.BAD_GATEWAY,
            http.HTTPStatus.SERVICE_UNAVAILABLE,
            http.HTTPStatus.GATEWAY_TIMEOUT,
        ):
            raise ServiceUnavailableException(message=message)
        raise InternalServerErrorException(message=message)


async def watch_event_stream(
    events: AsyncIterator[Tuple[Optional[str], dict]],
    to_public: Callable[[dict], Any],
) -> AsyncIterator[str]:
    """Frame a ``(verb, object)`` watch source as GPUStack SSE.

    A background task drains ``events`` into a bounded queue; the consumer maps
    each Kubernetes watch verb (ADDED/MODIFIED/DELETED) to a GPUStack EventType,
    drops BOOKMARK / unknown verbs, applies ``to_public`` to the object, and
    frames each mapped event as ``<json>\\n\\n``. When no event arrives within
    ``_HEARTBEAT_INTERVAL`` a bare ``"\\n\\n"`` keepalive is emitted.

    ``events`` owns its own resources: the caller wraps any client / context
    manager (e.g. a :class:`ClusterOps`) inside the source generator, so
    cancelling this stream cancels the drain task, which propagates into
    ``events`` and unwinds those ``async with`` blocks. A source that ends —
    cleanly or by raising — just ends the stream; it never becomes a data frame.
    """
    queue: asyncio.Queue = asyncio.Queue(maxsize=_WATCH_QUEUE_MAXSIZE)

    async def _drain() -> None:
        try:
            # aclosing guarantees the source generator is closed on every exit
            # path — including a cancel that parks us mid-iteration — so its
            # ``async with`` blocks (e.g. a ClusterOps client) always unwind
            # instead of being abandoned suspended at a yield.
            async with aclosing(events) as source:
                async for pair in source:
                    await queue.put(pair)
        except asyncio.CancelledError:
            # Client disconnect cancels this producer. Do NOT fall through to the
            # _DONE enqueue below: a slow consumer may have parked us on a full
            # queue, and a blocking put during teardown would deadlock the
            # consumer awaiting this task. Re-raise so the task ends.
            raise
        except Exception:
            logger.exception("watch stream source failed")
        # Source ended or errored (but was not cancelled): signal the consumer to
        # stop. Reached only while the consumer is still draining, so this put
        # cannot deadlock on a full queue.
        await queue.put(_DONE)

    producer = asyncio.create_task(_drain())
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), _HEARTBEAT_INTERVAL)
            except asyncio.TimeoutError:
                yield "\n\n"
                continue
            if item is _DONE:
                return
            verb, obj = item
            mapped = _K8S_TO_GPUSTACK.get(verb)
            if mapped is None:
                continue
            event = Event(type=mapped, data=to_public(obj))
            yield json.dumps(jsonable_encoder(event), separators=(",", ":")) + "\n\n"
    finally:
        producer.cancel()
        with suppress(asyncio.CancelledError):
            await producer
