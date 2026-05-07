from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from gpustack.api.tenant import (
    bypass_tenant_filter,
    assert_cluster_resource_visible,
    cluster_resource_visibility_conditions,
)
from gpustack.server.db import async_session
from gpustack.server.deps import SessionDep, TenantContextDep
from gpustack.schemas.gpu_devices import (
    GPUDevice,
    GPUDeviceListParams,
    GPUDevicesPublic,
    GPUDevicePublic,
)


router = APIRouter()


@router.get("", response_model=GPUDevicesPublic)
async def get_gpus(
    ctx: TenantContextDep,
    params: GPUDeviceListParams = Depends(),
    search: str = None,
    cluster_id: int = None,
):
    fuzzy_fields = {}
    if search:
        fuzzy_fields = {"name": search}
    fields = {}
    if cluster_id:
        fields["cluster_id"] = cluster_id

    extra_conditions = cluster_resource_visibility_conditions(ctx, GPUDevice)

    def _gpu_visible(g) -> bool:
        if bypass_tenant_filter(ctx):
            return True
        org_id = getattr(g, "owner_principal_id", None)
        if (
            ctx.current_principal_id is not None
            and org_id is not None
            and org_id == ctx.current_principal_id
        ):
            return True
        if getattr(g, "cluster_id", None) in ctx.accessible_cluster_ids:
            return True
        return False

    if params.watch:
        return StreamingResponse(
            GPUDevice.streaming(
                fuzzy_fields=fuzzy_fields, fields=fields, filter_func=_gpu_visible
            ),
            media_type="text/event-stream",
        )

    async with async_session() as session:
        return await GPUDevice.paginated_by_query(
            session=session,
            fuzzy_fields=fuzzy_fields,
            page=params.page,
            per_page=params.perPage,
            fields=fields,
            extra_conditions=extra_conditions,
            order_by=params.order_by,
        )


@router.get("/{id}", response_model=GPUDevicePublic)
async def get_gpu(session: SessionDep, ctx: TenantContextDep, id: str):
    model = await GPUDevice.one_by_id(session, id)
    assert_cluster_resource_visible(
        ctx, model, not_found_message="GPU device not found"
    )
    return model
