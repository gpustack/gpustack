import json
import logging
from typing import AsyncIterator, Optional, Tuple

from fastapi import APIRouter, Request, status
from kubernetes_asyncio import client
from starlette.responses import StreamingResponse

from gpustack.api.exceptions import NotFoundException
from gpustack.api.tenant import cluster_visibility_conditions
from gpustack.gpu_instances import gateway_client
from gpustack.gpu_instances.cluster_apis import ClusterOps
from gpustack.routes.gpu_instances_helper import (
    build_cluster_ops,
    ensure_visible,
    ensure_writable,
    handle_error,
    watch_event_stream,
)

from gpustack.schemas import (
    Cluster,
    GPUAggregatedInstanceTypePublic,
    GPUAggregatedInstanceTypesPublic,
    GPUInstanceTypeCreate,
    GPUInstanceTypeUpdate,
    GPUInstanceTypePublic,
    GPUInstanceTypesPublic,
)
from gpustack.schemas.clusters import ClusterProvider
from gpustack.server.db import async_session
from gpustack.server.deps import SessionDep, TenantContextDep

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/aggregated",
    response_model=GPUAggregatedInstanceTypesPublic,
    response_model_exclude_none=True,
)
async def get_gpu_aggregated_instance_types(
    ctx: TenantContextDep,
    watch: bool = False,
):
    # Mirror cluster-list visibility: surface every Kubernetes cluster
    # the caller can see — both the ones they own AND the ones granted
    # via ``cluster_access``. Without the grants path an Org member
    # would see an empty instance-type list even after a platform
    # admin authorised them on a K8s cluster.
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
        return GPUAggregatedInstanceTypesPublic(items=[])

    if watch:
        # The gateway streams raw Kubernetes watch verbs (ADDED/MODIFIED/
        # DELETED); wrap them into GPUStack events so a caller sees the same
        # event-type contract as every other GPUStack stream.
        return StreamingResponse(
            watch_event_stream(
                _aggregated_instance_type_events(cluster_ids),
                GPUAggregatedInstanceTypePublic.model_validate,
            ),
            media_type="text/event-stream",
        )

    return await gateway_client.list_instance_types(
        clusters=cluster_ids,
        aggregated=True,
    )


@router.get(
    "",
    response_model=GPUInstanceTypesPublic,
    response_model_exclude_none=True,
)
async def get_gpu_instance_types(
    request: Request,
    session: SessionDep,
    ctx: TenantContextDep,
    cluster_id: int,
    watch: bool = False,
):
    cluster = ensure_visible(
        await Cluster.one_by_id(
            session=session,
            id=cluster_id,
        ),
        ctx,
    )
    ops = await build_cluster_ops(request, session, cluster)

    if watch:
        # The visibility check above already ran, so an invisible cluster has
        # 404'd before any stream is opened. The source owns and closes ops.
        return StreamingResponse(
            watch_event_stream(
                _cluster_instance_type_events(ops),
                _to_instance_type_public,
            ),
            media_type="text/event-stream",
        )

    async with ops, handle_error():
        result = await ops.list_instance_types()
    return _to_instance_types_public(result)


@router.post(
    "",
    response_model=GPUInstanceTypePublic,
    response_model_exclude_none=True,
)
async def create_gpu_instance_type(
    request: Request,
    session: SessionDep,
    ctx: TenantContextDep,
    create: GPUInstanceTypeCreate,
    cluster_id: int,
):
    cluster = ensure_writable(
        await Cluster.one_by_id(
            session=session,
            id=cluster_id,
        ),
        ctx,
    )
    ops = await build_cluster_ops(request, session, cluster)

    spec = create.spec.model_dump(by_alias=True, exclude_none=True)
    async with ops, handle_error():
        result = await ops.create_instance_type(create.name, spec)
    return _to_instance_type_public(result)


@router.put(
    "",
    response_model=GPUInstanceTypePublic,
    response_model_exclude_none=True,
)
async def update_gpu_instance_type(
    request: Request,
    session: SessionDep,
    ctx: TenantContextDep,
    update: GPUInstanceTypeUpdate,
    cluster_id: int,
):
    cluster = ensure_writable(
        await Cluster.one_by_id(
            session=session,
            id=cluster_id,
        ),
        ctx,
    )
    ops = await build_cluster_ops(request, session, cluster)

    spec = update.spec.model_dump(by_alias=True, exclude_none=True)
    async with ops, handle_error():
        result = await ops.update_instance_type(update.name, spec)
    if result is None:
        raise NotFoundException(message=f"Instance type {update.name} not found")
    return _to_instance_type_public(result)


@router.delete("/{name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_gpu_instance_type(
    request: Request,
    session: SessionDep,
    ctx: TenantContextDep,
    name: str,
    cluster_id: int,
):
    cluster = ensure_writable(
        await Cluster.one_by_id(
            session=session,
            id=cluster_id,
        ),
        ctx,
    )
    ops = await build_cluster_ops(request, session, cluster)

    async with ops, handle_error():
        existed = await ops.delete_instance_type(name)
    if not existed:
        raise NotFoundException(message=f"Instance type {name} not found")


@router.put(
    "/{name}/deactivate",
    response_model=GPUInstanceTypePublic,
    response_model_exclude_none=True,
)
async def deactivate_gpu_instance_type(
    request: Request,
    session: SessionDep,
    ctx: TenantContextDep,
    name: str,
    cluster_id: int,
):
    cluster = ensure_writable(
        await Cluster.one_by_id(
            session=session,
            id=cluster_id,
        ),
        ctx,
    )
    ops = await build_cluster_ops(request, session, cluster)

    async with ops, handle_error():
        result = await ops.deactivate_instance_type(name)
    if result is None:
        raise NotFoundException(message=f"Instance type {name} not found")
    return _to_instance_type_public(result)


@router.put(
    "/{name}/activate",
    response_model=GPUInstanceTypePublic,
    response_model_exclude_none=True,
)
async def activate_gpu_instance_type(
    request: Request,
    session: SessionDep,
    ctx: TenantContextDep,
    name: str,
    cluster_id: int,
):
    cluster = ensure_writable(
        await Cluster.one_by_id(
            session=session,
            id=cluster_id,
        ),
        ctx,
    )
    ops = await build_cluster_ops(request, session, cluster)

    async with ops, handle_error():
        result = await ops.activate_instance_type(name)
    if result is None:
        raise NotFoundException(message=f"Instance type {name} not found")
    return _to_instance_type_public(result)


async def _cluster_instance_type_events(
    ops: ClusterOps,
) -> AsyncIterator[Tuple[Optional[str], dict]]:
    """Normalize a cluster's native Kubernetes watch into ``(verb, raw object)``.

    ``ops`` is entered here — not in the consumer — so its client is closed on
    every exit path, including a disconnect that cancels the stream. A watch
    ERROR / unrecoverable ``resourceVersion`` expiry surfaces as an
    ``ApiException`` and ends the source cleanly (logged at WARNING because the
    watch otherwise never terminates) rather than as a data frame.
    """
    async with ops:
        try:
            async for native in ops.watch_instance_types():
                yield native["type"], native["raw_object"]
        except client.exceptions.ApiException as e:
            logger.warning(
                "instance-type watch for cluster %s ended: %s", ops.cluster_id, e
            )


async def _aggregated_instance_type_events(
    cluster_ids: list[str],
) -> AsyncIterator[Tuple[Optional[str], dict]]:
    """Normalize the gateway's aggregated InstanceType watch into ``(verb, object)``.

    The gateway re-frames each ``manager.WorkerEvent`` as ``<json>\\n\\n`` (see
    gateway_client._stream); the JSON carries a Kubernetes watch ``type`` and an
    already-aggregated ``object`` (name / spec / aggregated status). Malformed
    lines are dropped, mirroring controllers._on_downstream_event.
    """
    async for line in gateway_client.watch_instance_types(
        clusters=cluster_ids,
        aggregated=True,
    ):
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            logger.warning("discarding malformed aggregated event: %r", line)
            continue
        yield event.get("type"), event.get("object") or {}


def _to_instance_type_public(item: dict) -> GPUInstanceTypePublic:
    """Map a raw ``worker.gpustack.ai/v1`` InstanceType dict into the public
    schema, hoisting ``metadata.name`` to ``name``. A freshly-created CR may
    lack a reconciled status, so an empty status maps to ``{}`` (every
    ``GPUInstanceTypeStatus`` field is Optional)."""
    return GPUInstanceTypePublic(
        name=item.get("metadata", {}).get("name"),
        spec=item.get("spec") or {},
        status=item.get("status") or {},
    )


def _to_instance_types_public(result: dict) -> GPUInstanceTypesPublic:
    return GPUInstanceTypesPublic(
        items=[_to_instance_type_public(i) for i in result.get("items", [])]
    )
