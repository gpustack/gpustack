import logging
from typing import Optional, Annotated
from fastapi.security import HTTPAuthorizationCredentials
from fastapi import APIRouter, Request, Response, Depends
from gpustack.api.exceptions import (
    NotFoundException,
    ForbiddenException,
    UnauthorizedException,
    BadRequestException,
)
from gpustack.server.services import ModelService, UserService
from gpustack.schemas.api_keys import ApiKey
from gpustack.schemas.users import User
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

model_name_missing_exception = BadRequestException(
    message="Missing 'model' field",
    is_openai_exception=True,
)

model_not_found_exception = NotFoundException(
    message="Model not found",
    is_openai_exception=True,
)


@router.get("")
async def server_auth(
    request: Request,
    session: SessionDep,
):
    user: Optional[User] = None
    api_key: Optional[ApiKey] = None
    access_key: Optional[str] = None
    consumer = 'none'
    try:
        user = await get_current_user(
            request=request,
            session=session,
            basic_credentials=await basic_auth(request),
            bearer_token=await bearer_auth(request),
            cookie_token=await cookie_auth(request),
        )
        api_key = getattr(request.state, "api_key", None)
        access_key = None if api_key is None else api_key.access_key
        consumer = '.'.join(
            [part for part in [access_key, f"gpustack-{user.id}"] if part is not None]
        )
    except UnauthorizedException:
        logger.debug("Unauthenticated request to server token-auth endpoint")
    except Exception as e:
        logger.error(f"Error during authentication: {e}")
        raise e

    model_name = request.headers.get("x-higress-llm-model")
    if model_name is None or model_name == "":
        logger.debug(
            "Missing x-higress-llm-model header for token authentication",
        )
        raise credentials_exception if user is None else model_name_missing_exception
    pair = await ModelService(session=session).get_model_auth_info_by_name(
        name=model_name
    )
    if pair is None:
        raise credentials_exception if user is None else model_not_found_exception
    policy = pair[0]
    registration_token = pair[1]

    if user is None and policy != AccessPolicyEnum.PUBLIC:
        logger.debug(
            f"Unauthenticated request to access model {model_name} with policy {policy}",
        )
        raise credentials_exception

    if policy != AccessPolicyEnum.PUBLIC:
        if not await UserService(session).model_allowed_for_user(
            model_name=model_name,
            user_id=user.id,
            api_key=api_key,
        ):
            raise ForbiddenException(
                message=f"Api key not allowed to access model {model_name}"
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
