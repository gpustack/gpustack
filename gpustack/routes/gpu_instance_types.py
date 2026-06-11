from fastapi import APIRouter
from starlette.responses import StreamingResponse

from gpustack.api.tenant import cluster_visibility_conditions
from gpustack.gpu_instances import gateway_client

from gpustack.schemas import (
    GPUAggregatedInstanceTypesPublic,
    Cluster,
)
from gpustack.schemas.clusters import ClusterProvider
from gpustack.server.db import async_session
from gpustack.server.deps import TenantContextDep

router = APIRouter()


@router.get(
    "",
    response_model=GPUAggregatedInstanceTypesPublic,
    response_model_exclude_none=True,
)
async def get_gpu_instance_types(
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

    cluster_ids = [c.id for c in clusters]

    if watch:
        return StreamingResponse(
            gateway_client.watch_instance_types(
                clusters=cluster_ids,
                aggregated=True,
            ),
            media_type="text/event-stream",
        )

    return await gateway_client.list_instance_types(
        clusters=cluster_ids,
        aggregated=True,
    )
