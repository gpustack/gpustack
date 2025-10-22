import logging
from typing import Optional, Set
from fastapi import APIRouter, Request, Response
from gpustack.api.exceptions import (
    ForbiddenException,
    NotFoundException,
)
from gpustack.schemas.api_keys import ApiKey
from gpustack.server.deps import CurrentUserDep

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("")
async def token_auth(
    request: Request,
    user: CurrentUserDep,
):
    model_name = request.headers.get("X-Higress-Llm-Model")
    if model_name is None:
        logger.warning(
            "Missing X-Higress-Llm-Model header for token authentication, user %s",
            user.id,
        )
        raise NotFoundException(message=f"Model {model_name} Not Found")
    api_key: Optional[ApiKey] = getattr(request.state, "api_key", None)
    access_key = None if api_key is None else api_key.access_key
    allowed_model_names: Set[str] = getattr(
        request.state, "user_allow_model_names", set()
    )
    if model_name not in allowed_model_names:
        raise ForbiddenException(
            message=f"Api key not allowed to access model {model_name}"
        )
    comsumer = '.'.join(
        [part for part in [access_key, f"gpustack-{user.id}"] if part is not None]
    )
    headers = {
        "X-Mse-Consumer": comsumer,
    }
    return Response(
        status_code=200,
        headers=headers,
    )
