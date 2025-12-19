from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from gpustack.server.deps import SessionDep, EngineDep
from gpustack.schemas.gpu_devices import (
    GPUDevice,
    GPUDeviceListParams,
    GPUDevicesPublic,
    GPUDevicePublic,
)
from gpustack.api.exceptions import (
    NotFoundException,
)


router = APIRouter()


@router.get("", response_model=GPUDevicesPublic)
async def get_gpus(
    engine: EngineDep,
    session: SessionDep,
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
    if params.watch:
        return StreamingResponse(
            GPUDevice.streaming(engine, fuzzy_fields=fuzzy_fields, fields=fields),
            media_type="text/event-stream",
        )

    return await GPUDevice.paginated_by_query(
        session=session,
        fuzzy_fields=fuzzy_fields,
        page=params.page,
        per_page=params.perPage,
        fields=fields,
        order_by=params.order_by,
    )


@router.get("/{id}", response_model=GPUDevicePublic)
async def get_gpu(session: SessionDep, id: str):
    model = await GPUDevice.one_by_id(session, id)
    if not model:
        raise NotFoundException(message="GPU device not found")

    return model
