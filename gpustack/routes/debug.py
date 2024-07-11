import logging
from fastapi import APIRouter, Request

from gpustack.api.exceptions import InvalidException

router = APIRouter()

logger = logging.getLogger(__name__)


@router.get("/log_level")
async def get_log_level():
    current_level = logging.getLogger().level
    return logging.getLevelName(current_level)


@router.put("/log_level")
async def set_log_level(request: Request):
    level = await request.body()
    level_str = level.decode("utf-8").upper().strip()
    numeric_level = getattr(logging, level_str, None)
    if not isinstance(numeric_level, int):
        raise InvalidException(message="Invalid log level")

    logging.getLogger().setLevel(numeric_level)
    logger.info(f"Set log level to {level_str}")
    return "ok"
