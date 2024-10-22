from fastapi import APIRouter
from packaging.version import Version

from gpustack.server.update_check import UpdateResponse, get_update
from gpustack import __version__

router = APIRouter()


@router.get("/")
async def update():
    update_response = await get_update()

    if is_newer_version(update_response.latest_version, __version__):
        return update_response

    return UpdateResponse(latest_version=__version__)


def is_newer_version(given: str, current: str) -> bool:
    """
    Check if the given version is newer than the current version.
    """
    try:
        givenV = Version(given)
        currentV = Version(current)
    except Exception:
        return False

    return givenV > currentV
