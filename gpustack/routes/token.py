import logging
from datetime import timedelta
from typing import Optional, Annotated
from fastapi.security import HTTPAuthorizationCredentials
from fastapi import APIRouter, Request, Response, Depends
from gpustack.api.exceptions import (
    NotFoundException,
    ForbiddenException,
    UnauthorizedException,
    BadRequestException,
)
from gpustack.server.services import ModelRouteService, UserService
from gpustack.schemas.api_keys import ApiKey
from gpustack.schemas.users import User
from gpustack.schemas.models import AccessPolicyEnum
from gpustack.server.deps import SessionDep
from gpustack.api.auth import (
    api_key_header_auth,
    basic_auth,
    cookie_auth,
    bearer_auth,
    get_current_user,
    credentials_exception,
    gateway_token_auth,
    inference_scope,
)
from gpustack.security import JWTManager, AUTH_CACHE_HEADER

logger = logging.getLogger(__name__)

router = APIRouter()


# DO NOT make these module-level singletons — see issue #5121. Raising the
# same instance repeatedly keeps every prior call-stack alive on the
# instance's ``__traceback__`` (the instance lives forever as a module
# attribute), retaining every ``frame.f_locals`` (Request, AsyncSession,
# user, api_key, ...) and burning ~30 KB per request. Always raise a freshly
# constructed instance instead.
def model_name_missing_exception() -> BadRequestException:
    return BadRequestException(
        message="Missing 'model' field",
        is_openai_exception=True,
    )


def model_not_found_exception() -> NotFoundException:
    return NotFoundException(
        message="Model not found",
        is_openai_exception=True,
    )


@router.get("")
async def server_auth(
    request: Request,
    session: SessionDep,
):
    jwt_manager: JWTManager = request.app.state.jwt_manager
    cached = request.headers.get(AUTH_CACHE_HEADER)
    request_model = request.headers.get("x-higress-llm-model")
    if cached and request_model:
        try:
            data = jwt_manager.decode_jwt_data(cached)
            if data.get("model") == request_model:
                return Response(
                    status_code=200,
                    headers={
                        "X-Mse-Consumer": data["consumer"],
                        "Authorization": "Bearer " + data["token"],
                        "cookie": "dummy=dummy",
                    },
                )
        except Exception:
            pass

    user: Optional[User] = None
    api_key: Optional[ApiKey] = None
    access_key: Optional[str] = None
    consumer = 'none'
    cookie_token = await cookie_auth(request)
    x_api_key = await api_key_header_auth(request)
    try:
        user = await get_current_user(
            request=request,
            session=session,
            basic_credentials=await basic_auth(request),
            bearer_token=await bearer_auth(request),
            x_api_key=x_api_key,
            cookie_token=cookie_token,
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

    if user is None:
        gateway_token_auth(request)

    model_name = request.headers.get("x-higress-llm-model")
    if model_name is None or model_name == "":
        logger.debug(
            "Missing x-higress-llm-model header for token authentication",
        )
        raise (
            credentials_exception() if user is None else model_name_missing_exception()
        )
    pair = await ModelRouteService(session=session).get_model_auth_info_by_name(
        model_name
    )
    if pair is None:
        raise credentials_exception() if user is None else model_not_found_exception()
    policy = pair[0]
    registration_token = pair[1]

    if user is None and policy != AccessPolicyEnum.PUBLIC:
        logger.debug(
            f"Unauthenticated request to access model {model_name} with policy {policy}",
        )
        raise credentials_exception()

    if policy != AccessPolicyEnum.PUBLIC:
        # llm_scope will raise exception if the api key is not allowed to access llm.
        inference_scope(request, user)
        if not await UserService(session).model_allowed_for_user(
            model_name=model_name,
            user_id=user.id,
            api_key=api_key,
        ):
            raise ForbiddenException(
                message=f"Api key not allowed to access model {model_name}"
            )
    cache_token = jwt_manager.create_token(
        {"consumer": consumer, "token": registration_token, "model": model_name},
        expires_delta=timedelta(minutes=5),
    )
    return Response(
        status_code=200,
        headers={
            "X-Mse-Consumer": consumer,
            "Authorization": f"Bearer {registration_token}",
            AUTH_CACHE_HEADER: cache_token,
            "cookie": "dummy=dummy",
        },
    )


async def worker_auth(
    request: Request,
    bearer_token: Annotated[
        Optional[HTTPAuthorizationCredentials], Depends(bearer_auth)
    ] = None,
    x_api_key: Annotated[Optional[str], Depends(api_key_header_auth)] = None,
):
    token: str = request.app.state.token
    registration_token = request.app.state.config.token
    model_name = request.headers.get("X-Higress-Llm-Model")
    if model_name is None:
        logger.warning("Missing X-Higress-Llm-Model header for token authentication")
        raise credentials_exception()
    token_value = (bearer_token.credentials if bearer_token else None) or x_api_key
    if token_value is None:
        raise credentials_exception()
    if token_value != token and token_value != registration_token:
        raise credentials_exception()
    return Response(
        status_code=200,
        headers={
            "X-Mse-Consumer": "gpustack-server",
        },
    )
