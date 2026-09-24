from fastapi import APIRouter
from gpustack.extension import resolve_version_info

router = APIRouter()


@router.get("/healthz")
async def healthz():
    return "ok"


@router.get("/readyz")
async def readyz():
    return "ok"


@router.get("/version")
async def version():
    version, git_commit = resolve_version_info()
    return {"version": version, "git_commit": git_commit}
