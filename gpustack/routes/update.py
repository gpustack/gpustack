from fastapi import APIRouter

from gpustack.server.update_check import get_update


router = APIRouter()


@router.get("/")
async def update():
    update_response = await get_update()
    return update_response
