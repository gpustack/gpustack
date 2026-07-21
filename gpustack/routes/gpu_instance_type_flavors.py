import json
import logging
from typing import AsyncIterator, Optional, Tuple

from fastapi import APIRouter, Request
from starlette.responses import StreamingResponse

from gpustack.api.tenant import cluster_visibility_conditions
from gpustack.gpu_instances import gateway_client
from gpustack.routes.gpu_instances_helper import (
    build_cluster_ops,
    ensure_visible,
    handle_error,
    watch_event_stream,
)

from gpustack.schemas import (
    Cluster,
    GPUAggregatedInstanceTypeFlavorPublic,
    GPUAggregatedInstanceTypeFlavorsPublic,
    GPUInstanceTypeFlavorPublic,
    GPUInstanceTypeFlavorsPublic,
)
from gpustack.schemas.clusters import ClusterProvider
from gpustack.server.db import async_session
from gpustack.server.deps import SessionDep, TenantContextDep

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/aggregated",
    response_model=GPUAggregatedInstanceTypeFlavorsPublic,
    response_model_exclude_none=True,
)
async def get_gpu_aggregated_instance_type_flavors(
    ctx: TenantContextDep,
    watch: bool = False,
):
    # Mirror cluster-list visibility: surface every Kubernetes cluster
    # the caller can see — both the ones they own AND the ones granted
    # via ``cluster_access``. Without the grants path an Org member
    # would see an empty flavor list even after a platform admin
    # authorised them on a K8s cluster.
    async with async_session() as session:
        clusters = await Cluster.all_by_fields(
            session=session,
            fields={"provider": ClusterProvider.Kubernetes},
            extra_conditions=cluster_visibility_conditions(ctx, Cluster),
        )

    # gateway_client list/watch take cluster ids as strings.
    cluster_ids = [str(c.id) for c in clusters]

    if not cluster_ids:
        # No visible clusters → return an empty aggregate. The gateway treats an
        # empty cluster filter as "all clusters", so forwarding the empty set
        # would leak the whole fleet to a caller who can see nothing.
        return GPUAggregatedInstanceTypeFlavorsPublic(items=[])

    if watch:
        # The gateway streams raw Kubernetes watch verbs (ADDED/MODIFIED/
        # DELETED); wrap them into GPUStack events so a caller sees the same
        # event-type contract as every other GPUStack stream.
        return StreamingResponse(
            watch_event_stream(
                _aggregated_instance_type_flavor_events(cluster_ids),
                GPUAggregatedInstanceTypeFlavorPublic.model_validate,
            ),
            media_type="text/event-stream",
        )

    return await gateway_client.list_instance_type_flavors(
        clusters=cluster_ids,
        aggregated=True,
    )


@router.get(
    "",
    response_model=GPUInstanceTypeFlavorsPublic,
    response_model_exclude_none=True,
)
async def get_gpu_instance_type_flavors(
    request: Request,
    session: SessionDep,
    ctx: TenantContextDep,
    cluster_id: int,
):
    cluster = ensure_visible(
        await Cluster.one_by_id(
            session=session,
            id=cluster_id,
        ),
        ctx,
    )
    ops = await build_cluster_ops(request, session, cluster)

    async with ops, handle_error():
        result = await ops.list_instance_type_flavors()
    return _to_instance_type_flavors_public(result)


async def _aggregated_instance_type_flavor_events(
    cluster_ids: list[str],
) -> AsyncIterator[Tuple[Optional[str], dict]]:
    """Normalize the gateway's aggregated InstanceTypeFlavor watch into ``(verb, object)``.

    The gateway re-frames each ``manager.WorkerEvent`` as ``<json>\\n\\n`` (see
    gateway_client._stream); the JSON carries a Kubernetes watch ``type`` and an
    already-aggregated ``object`` (name / spec). Malformed lines are dropped,
    mirroring controllers._on_downstream_event.
    """
    async for line in gateway_client.watch_instance_type_flavors(
        clusters=cluster_ids,
        aggregated=True,
    ):
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            logger.warning("discarding malformed aggregated event: %r", line)
            continue
        yield event.get("type"), event.get("object") or {}


def _to_instance_type_flavor_public(item: dict) -> GPUInstanceTypeFlavorPublic:
    """Map a raw ``worker.gpustack.ai/v1`` InstanceTypeFlavor dict into the
    public schema, hoisting ``metadata.name`` to ``name``. Flavors are a
    read-only projection and carry no status."""
    return GPUInstanceTypeFlavorPublic(
        name=item.get("metadata", {}).get("name"),
        spec=item.get("spec") or {},
    )


def _to_instance_type_flavors_public(result: dict) -> GPUInstanceTypeFlavorsPublic:
    return GPUInstanceTypeFlavorsPublic(
        items=[_to_instance_type_flavor_public(i) for i in result.get("items", [])]
    )
