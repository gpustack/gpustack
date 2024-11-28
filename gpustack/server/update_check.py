import asyncio
import time
import logging
from typing import Optional
import httpx
from pydantic import BaseModel

from gpustack import __version__
from gpustack.utils import platform

logger = logging.getLogger(__name__)

cached_update = None
cache_timestamp = None
UPDATE_CACHE_TIMEOUT = 12 * 60 * 60
UPDATE_CHECK_INTERVAL = 12 * 60 * 60
UPDATE_CHECK_URL = "https://update-service.gpustack.ai"


def is_dev_version() -> bool:
    return "0.0.0" in __version__


class UpdateResponse(BaseModel):
    latest_version: str


async def get_update(update_check_url: Optional[str] = None) -> UpdateResponse:
    if is_dev_version():
        return UpdateResponse(latest_version=__version__)

    global cached_update, cache_timestamp

    if (
        cached_update is None
        or cache_timestamp is None
        or time.time() - cache_timestamp > UPDATE_CACHE_TIMEOUT
    ):
        cached_update = await do_get_update(update_check_url)
        cache_timestamp = time.time()

    return cached_update


async def do_get_update(
    update_check_url: Optional[str] = None,
    timeout: int = 3,
) -> UpdateResponse:
    if update_check_url is None:
        update_check_url = UPDATE_CHECK_URL

    try:
        os_name = platform.system()
        arch = platform.arch()
        headers = {"User-Agent": f"gpustack/{__version__} ({os_name}; {arch})"}
        params = {"os": os_name, "arch": arch, "version": __version__}

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(
                update_check_url, headers=headers, params=params
            )
            response.raise_for_status()

            response_data = response.json()
            return UpdateResponse(**response_data)
    except Exception as e:
        logger.debug(f"Failed to check for updates: {e}")
        return UpdateResponse(latest_version=__version__)


class UpdateChecker:
    """
    UpdateChecker checks for updates periodically.
    """

    def __init__(
        self, update_check_url: Optional[str] = None, interval=UPDATE_CHECK_INTERVAL
    ):
        self._interval = interval
        self._update_check_url = update_check_url

    async def start(self):
        while True:
            try:
                await get_update(self._update_check_url)
            except Exception as e:
                logger.error(f"Failed to get update: {e}")
            await asyncio.sleep(self._interval)
