from typing import Dict

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from gpustack.api.tenant import (
    bypass_tenant_filter,
    assert_cluster_resource_visible,
    cluster_resource_visibility_conditions,
)
from gpustack.server.bus import Event, EventType
from gpustack.server.db import async_session
from gpustack.server.deps import SessionDep, TenantContextDep
from gpustack.server.worker_allocated_cache import (
    get_worker_allocated,
    vram_allocated_for_index,
)
from gpustack.schemas.gpu_devices import (
    GPUDevice,
    GPUDeviceListParams,
    GPUDevicesPublic,
    GPUDevicePublic,
)


router = APIRouter()


async def _lookup_vram_allocated(worker_id: int) -> Dict[int, int]:
    """Cache-backed {gpu_index: vram} lookup for a worker (see
    server.worker_allocated_cache)."""
    allocated = await get_worker_allocated(worker_id)
    return allocated.vram


def to_gpu_device_public(device: GPUDevice, vram: Dict[int, int]) -> GPUDevicePublic:
    """Override memory.allocated with the value aggregated server-side from
    current ModelInstance assignments, mirroring /v2/workers — the
    persisted view row carries only the worker-reported value, which is
    deprecated and no longer populated."""
    data = device.model_dump()
    memory = data.get('memory')
    if memory is not None:
        memory['allocated'] = vram_allocated_for_index(vram, data.get('index'))
    return GPUDevicePublic.model_validate(data)


async def _inject_allocated_into_event(event: Event):
    """Mutate a GPUDevice watch event so it carries the same
    server-computed allocated as the REST response."""
    # DELETED events arrive with data={"id": N} (ID-only) — nothing to mutate.
    if event.type == EventType.DELETED:
        return
    device = event.data
    if not isinstance(device, GPUDevicePublic) or device.memory is None:
        return
    vram = await _lookup_vram_allocated(device.worker_id)
    device.memory.allocated = vram_allocated_for_index(vram, device.index)


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
                fuzzy_fields=fuzzy_fields,
                fields=fields,
                filter_func=_gpu_visible,
                event_transform=_inject_allocated_into_event,
            ),
            media_type="text/event-stream",
        )

    async with async_session() as session:
        paginated = await GPUDevice.paginated_by_query(
            session=session,
            fuzzy_fields=fuzzy_fields,
            page=params.page,
            per_page=params.perPage,
            fields=fields,
            extra_conditions=extra_conditions,
            order_by=params.order_by,
        )
        vram_by_worker = {
            worker_id: await _lookup_vram_allocated(worker_id)
            for worker_id in {device.worker_id for device in paginated.items}
        }
        items = [
            to_gpu_device_public(device, vram_by_worker[device.worker_id])
            for device in paginated.items
        ]
        return GPUDevicesPublic(items=items, pagination=paginated.pagination)


@router.get("/{id}", response_model=GPUDevicePublic)
async def get_gpu(session: SessionDep, ctx: TenantContextDep, id: str):
    model = await GPUDevice.one_by_id(session, id)
    assert_cluster_resource_visible(
        ctx, model, not_found_message="GPU device not found"
    )
    return to_gpu_device_public(model, await _lookup_vram_allocated(model.worker_id))
