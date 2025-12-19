import logging
from fastapi import APIRouter, Request
from typing import Any, Dict

from gpustack.api.exceptions import (
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
