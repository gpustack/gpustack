from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from gpustack.server.deps import ListParamsDep, SessionDep
from gpustack.schemas.gpu_devices import (
    GPUDevice,
    GPUDevicesPublic,
    GPUDevicePublic,
)
from gpustack.api.exceptions import (
    NotFoundException,
)


router = APIRouter()


@router.get("", response_model=GPUDevicesPublic)
async def get_gpus(session: SessionDep, params: ListParamsDep, search: str = None):
    fuzzy_fields = {}
    if search:
        fuzzy_fields = {"name": search}

    if params.watch:
        return StreamingResponse(
            GPUDevice.streaming(session, fuzzy_fields=fuzzy_fields),
            media_type="text/event-stream",
        )

    return await GPUDevice.paginated_by_query(
        session=session,
        fuzzy_fields=fuzzy_fields,
        page=params.page,
        per_page=params.perPage,
    )


@router.get("/{id}", response_model=GPUDevicePublic)
async def get_gpu(session: SessionDep, id: str):
    model = await GPUDevice.one_by_id(session, id)
    if not model:
        raise NotFoundException(message="GPU device not found")

    return model
