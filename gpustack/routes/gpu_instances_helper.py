"""Shared helpers for the GPU instance-type / flavor routes.

Most helpers sit on top of :class:`ClusterOps` (the raw ``worker.gpustack.ai/v1``
CRD client) and translate cluster access + Kubernetes failures into the
project's HTTP semantics, so the route modules stay thin. :func:`watch_event_stream`
is a transport-level helper shared by both the per-cluster and the aggregated
watch routes, framing a normalized watch source as GPUStack SSE.
"""

import asyncio
import http
import json
import logging
from contextlib import aclosing, asynccontextmanager, suppress
from typing import Any, AsyncIterator, Callable, Optional, Tuple

from fastapi import Request
from fastapi.encoders import jsonable_encoder
from kubernetes_asyncio import client
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.exceptions import (
    AlreadyExistsException,
    InternalServerErrorException,
    InvalidException,
    NotFoundException,
)
from gpustack.api.tenant import (
    TenantContext,
    assert_cluster_visible,
    assert_cluster_writable,
)
from gpustack.gpu_instances.cluster_apis import ClusterOps
from gpustack.gpu_instances.cluster_apis_util import principal_namespace_identifier
from gpustack.schemas import Cluster
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
async def handle_error():
    """Translate a Kubernetes ``ApiException`` into the project's HTTP
    exceptions so a client-caused failure surfaces as the right status
    instead of a blanket 500."""
    try:
        yield
    except client.exceptions.ApiException as e:
        message = getattr(e, "reason", None) or str(e)
        if e.status == http.HTTPStatus.NOT_FOUND:
            raise NotFoundException(message=message)
        if e.status == http.HTTPStatus.CONFLICT:
            raise AlreadyExistsException(message=message)
        if e.status == http.HTTPStatus.BAD_REQUEST:
            raise InvalidException(message=message)
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
