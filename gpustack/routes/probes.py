from fastapi import APIRouter


router = APIRouter()


@router.get("/healthz")
async def healthz():
    return "ok"


@router.get("/readyz")
async def readyz():
    return "ok"
