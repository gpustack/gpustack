import logging
import tracemalloc
from fastapi import APIRouter, Request
from typing import Any, Dict

from gpustack.api.exceptions import (
    BadRequestException,
    InvalidException,
    ForbiddenException,
)
from gpustack.config.config import Config, set_global_config
from gpustack.utils.config import (
    WHITELIST_CONFIG_FIELDS,
    READ_ONLY_CONFIG_FIELDS,
    coerce_value_by_field,
    is_local_request,
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


@router.get("/config")
async def get_config(request: Request):
    app_state = request.app.state
    cfg: Config = getattr(app_state, "server_config", None) or getattr(
        app_state, "config", None
    )
    if cfg is None:
        raise InvalidException(message="Config is not available")
    result: Dict[str, Any] = {}
    for field in READ_ONLY_CONFIG_FIELDS:
        if hasattr(cfg, field):
            result[field] = getattr(cfg, field)
    return result


@router.put("/config")
async def set_config(request: Request):
    if not is_local_request(request):
        raise ForbiddenException(message="Only localhost is allowed")
    app_state = request.app.state
    cfg: Config = getattr(app_state, "server_config", None) or getattr(
        app_state, "config", None
    )
    if cfg is None:
        raise InvalidException(message="Config is not available")
    data = await request.json()
    updates: Dict[str, Any] = {}
    for k, v in data.items():
        if k in WHITELIST_CONFIG_FIELDS:
            updates[k] = coerce_value_by_field(k, v)
    for k, v in updates.items():
        setattr(cfg, k, v)
    if "debug" in updates:
        logging.getLogger().setLevel(
            logging.DEBUG if bool(updates["debug"]) else logging.INFO
        )
    set_global_config(cfg)
    logger.info("Applied runtime config updates")
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
