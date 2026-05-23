from fastapi import APIRouter
from starlette.responses import StreamingResponse

from gpustack.api.tenant import (
    bypass_tenant_filter,
)
from gpustack.gpu_instances import gateway_client

from gpustack.schemas import (
    GPUAggregatedInstanceTypesPublic,
    Cluster,
)
from gpustack.schemas.clusters import ClusterProvider
from gpustack.schemas.principals import platform_principal_id
from gpustack.server.db import async_session
from gpustack.server.deps import TenantContextDep

router = APIRouter()


@router.get("", response_model=GPUAggregatedInstanceTypesPublic)
async def get_gpu_instance_types(
    ctx: TenantContextDep,
    watch: bool = False,
):
    owner_principal_id = ctx.current_principal_id or platform_principal_id()
    if bypass_tenant_filter(ctx):
        owner_principal_id = None

    fields: dict = {
        "provider": ClusterProvider.Kubernetes,
    }
    if owner_principal_id is not None:
        fields["owner_principal_id"] = owner_principal_id

    async with async_session() as session:
        clusters = await Cluster.all_by_fields(
            session=session,
            fields=fields,
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
