import logging
import tracemalloc
from fastapi import APIRouter, Request

from gpustack.api.exceptions import (
    BadRequestException,
    InvalidException,
)

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
    numeric_level = logging._nameToLevel.get(level_str)
    if not isinstance(numeric_level, int):
        raise InvalidException(message="Invalid log level")

    logging.getLogger().setLevel(numeric_level)
    logger.info(f"Set log level to {level_str}")
    return "ok"


@router.get("/memory")
def get_memory_profile():
    if not tracemalloc.is_tracing():
        raise BadRequestException(
            message="tracemalloc is not enabled. Please run GPUStack server in debug mode."
        )

    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    result = [str(stat) for stat in top_stats[:20]]
    return {"top_memory_lines": result}
