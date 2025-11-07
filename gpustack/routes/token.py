import logging
from typing import Optional, Set, Annotated
from fastapi.security import HTTPAuthorizationCredentials
from fastapi import APIRouter, Request, Response, Depends
from gpustack.api.exceptions import (
    ForbiddenException,
)
from gpustack.server.services import ModelService
from gpustack.schemas.api_keys import ApiKey
from gpustack.schemas.models import AccessPolicyEnum
from gpustack.server.deps import SessionDep
from gpustack.api.auth import (
    basic_auth,
    cookie_auth,
    bearer_auth,
    get_current_user,
    credentials_exception,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("")
async def server_auth(
    request: Request,
    session: SessionDep,
):
    model_name = request.headers.get("x-higress-llm-model")
    if model_name is None:
        logger.warning(
            "Missing x-higress-llm-model header for token authentication",
        )
        raise credentials_exception
    pair = await ModelService(session=session).get_model_auth_info_by_name(
        name=model_name
    )
    if pair is None:
        raise credentials_exception
    policy = pair[0]
    registration_token = pair[1]
    consumer = 'none'
    if policy != AccessPolicyEnum.PUBLIC:
        user = await get_current_user(
            request=request,
            session=session,
            basic_credentials=await basic_auth(request),
            bearer_token=await bearer_auth(request),
            cookie_token=await cookie_auth(request),
        )
        api_key: Optional[ApiKey] = getattr(request.state, "api_key", None)
        access_key = None if api_key is None else api_key.access_key
        allowed_model_names: Set[str] = getattr(
            request.state, "user_allow_model_names", set()
        )
        if model_name not in allowed_model_names:
            raise ForbiddenException(
                message=f"Api key not allowed to access model {model_name}"
            )
        consumer = '.'.join(
            [part for part in [access_key, f"gpustack-{user.id}"] if part is not None]
        )
    headers = {
        "X-Mse-Consumer": consumer,
        "Authorization": f"Bearer {registration_token}",
    }
    return Response(
        status_code=200,
        headers=headers,
    )


async def worker_auth(
    request: Request,
    bearer_token: Annotated[
        Optional[HTTPAuthorizationCredentials], Depends(bearer_auth)
    ] = None,
):
    token: str = request.app.state.token
    registration_token = request.app.state.config.token
    model_name = request.headers.get("X-Higress-Llm-Model")
    if model_name is None:
        logger.warning("Missing X-Higress-Llm-Model header for token authentication")
        raise credentials_exception
    if bearer_token is None:
        raise credentials_exception
    if (
        bearer_token.credentials != token
        and bearer_token.credentials != registration_token
    ):
        raise credentials_exception
    return Response(
        status_code=200,
        headers={
            "X-Mse-Consumer": "gpustack-server",
        },
    )
