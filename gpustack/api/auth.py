import re
import uuid
from datetime import datetime, timezone
import logging
import aiohttp
from aiocache import cached
from fastapi import Depends, Request
from gpustack.config.config import Config
from gpustack.schemas.config import GatewayModeEnum
from gpustack.server.db import get_session
from typing import Annotated, Optional, Tuple, List
from fastapi.security import (
    APIKeyCookie,
    APIKeyHeader,
    HTTPAuthorizationCredentials,
    HTTPBasic,
    HTTPBasicCredentials,
    HTTPBearer,
)
from sqlmodel.ext.asyncio.session import AsyncSession
from gpustack.api.exceptions import (
    ForbiddenException,
    InternalServerErrorException,
    UnauthorizedException,
)
from gpustack.schemas.api_keys import ApiKey, PermissionScope
from gpustack.schemas.users import User, UserRole
from gpustack.security import (
    JWTManager,
    verify_hashed_secret,
    get_key_pair,
)
from gpustack.server.services import APIKeyService, UserService

logger = logging.getLogger(__name__)

SESSION_COOKIE_NAME = "gpustack_session"
OIDC_ID_TOKEN_COOKIE_NAME = "gpustack_oidc_id_token"
SSO_LOGIN_COOKIE_NAME = "gpustack_sso_login"
SYSTEM_USER_PREFIX = "system/"
SYSTEM_WORKER_USER_PREFIX = "system/worker/"
basic_auth = HTTPBasic(auto_error=False)
bearer_auth = HTTPBearer(auto_error=False)
api_key_header_auth = APIKeyHeader(name="X-API-Key", auto_error=False)
cookie_auth = APIKeyCookie(name=SESSION_COOKIE_NAME, auto_error=False)

credentials_exception = UnauthorizedException(
    message="Invalid authentication credentials"
)


def client_ip_getter(request: Request) -> str:
    if request.app.state.server_config.gateway_mode == GatewayModeEnum.embedded:
        return request.headers.get("X-GPUStack-Real-IP", "")
    else:
        return request.client.host


async def get_current_user(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
    basic_credentials: Annotated[
        Optional[HTTPBasicCredentials], Depends(basic_auth)
    ] = None,
    bearer_token: Annotated[
        Optional[HTTPAuthorizationCredentials], Depends(bearer_auth)
    ] = None,
    x_api_key: Annotated[Optional[str], Depends(api_key_header_auth)] = None,
    cookie_token: Annotated[Optional[str], Depends(cookie_auth)] = None,
) -> User:
    if hasattr(request.state, "user"):
        user: User = getattr(request.state, "user")
        return user
    api_key: Optional[ApiKey] = None
    user = None
    try:
        server_config: Config = request.app.state.server_config
        if basic_credentials and is_system_user(basic_credentials.username):
            user = await authenticate_system_user(server_config, basic_credentials)
        elif basic_credentials:
            user = await authenticate_basic_user(session, basic_credentials)
        elif cookie_token:
            jwt_manager: JWTManager = request.app.state.jwt_manager
            user = await get_user_from_jwt_token(session, jwt_manager, cookie_token)
        elif bearer_token or x_api_key:
            token = (bearer_token.credentials if bearer_token else None) or x_api_key
            if token is not None:
                user, api_key = await get_user_from_api_token(session, token)

        if user is None and client_ip_getter(request=request) == "127.0.0.1":
            if not server_config.force_auth_localhost:
                user = await User.first_by_field(session, "is_admin", True)
        if user:
            if not user.is_active:
                raise UnauthorizedException(message="User account is deactivated")
            request.state.user = user
            if api_key is not None:
                request.state.api_key = api_key
            return user

    except UnauthorizedException:
        raise
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to authenticate user: {e}")

    raise credentials_exception


async def get_admin_user(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    if not current_user.is_admin:
        raise ForbiddenException(message="No permission to access")
    return current_user


async def get_cluster_user(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    if (
        current_user.is_system
        and current_user.role == UserRole.Cluster
        and current_user.cluster_id is not None
    ):
        return current_user
    return await get_admin_user(current_user)


async def get_worker_user(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    if (
        current_user.is_system
        and current_user.role == UserRole.Worker
        and current_user.worker is not None
    ):
        return current_user
    return await get_admin_user(current_user)


def is_system_user(username: str) -> bool:
    return username.startswith(SYSTEM_USER_PREFIX)


async def authenticate_system_user(
    config: Config,
    credentials: HTTPBasicCredentials,
) -> Optional[User]:
    if credentials.username.startswith(SYSTEM_WORKER_USER_PREFIX):
        if credentials.password == config.token:
            return User(username=credentials.username, is_admin=True)
    return None


async def authenticate_basic_user(
    session: AsyncSession,
    basic_credentials: HTTPBasicCredentials,
) -> Optional[User]:
    try:
        user = await authenticate_user(
            session, basic_credentials.username, basic_credentials.password
        )
        return user
    except Exception:
        return None


def get_access_token(
    bearer_token: Optional[HTTPAuthorizationCredentials],
    oauth2_bearer_token: Optional[str],
    cookie_token: Optional[str],
) -> str:
    if bearer_token:
        return bearer_token.credentials
    elif oauth2_bearer_token:
        return oauth2_bearer_token
    elif cookie_token:
        return cookie_token
    else:
        raise credentials_exception


async def get_user_from_jwt_token(
    session: AsyncSession, jwt_manager: JWTManager, access_token: str
) -> Optional[User]:
    try:
        payload = jwt_manager.decode_jwt_token(access_token)
        username = payload.get("sub")
    except Exception:
        logger.debug("Failed to decode JWT token")
        return None

    if username is None:
        return None

    try:
        user = await UserService(session).get_by_username(username)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to get user: {e}")

    return user


def parse_hyphen_uuid(value: str) -> Optional[str]:
    if not re.match(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', value, re.I
    ):
        return None
    try:
        uuid.UUID(value)
        return value
    except ValueError:
        return None


async def get_user_from_api_token(
    session: AsyncSession, token: str
) -> Tuple[Optional[User], Optional[ApiKey]]:
    try:
        access_key, secret_key = get_key_pair(token)
        worker_uuid = parse_hyphen_uuid(access_key)
        if worker_uuid is not None:
            # if access_key is a valid uuid, it's legacy worker re-registering with legacy token
            access_key = ""
        access_keys = [access_key]
        # the custom key should have 32 chars access key as it is generated by security.custom_key_hash which will return 32 chars hex string.
        if len(access_key) == 32 and "" not in access_keys:
            # this means it is custom key or legacy worker token, we should also try to find api key with empty access key for backward compatibility
            access_keys.append("")
        for access_key in access_keys:
            api_key: ApiKey = await APIKeyService(session).get_by_access_key(access_key)
            if api_key:
                logger.trace(f"Found API key for access key: {access_key}")
                break
        if (
            api_key is not None
            and verify_hashed_secret(api_key.hashed_secret_key, secret_key)
            and (
                api_key.expires_at is None
                or api_key.expires_at > datetime.now(timezone.utc)
            )
        ):
            user: Optional[User] = await UserService(session).get_by_id(
                user_id=api_key.user_id,
            )
            if user is not None:
                return user, api_key
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to get user: {e}")

    return None, None


async def authenticate_user(
    session: AsyncSession, username: str, password: str
) -> User:
    user = await UserService(session).get_by_username(username)
    if not user:
        raise UnauthorizedException(message="Incorrect username or password")

    if not verify_hashed_secret(user.hashed_password, password):
        raise UnauthorizedException(message="Incorrect username or password")

    if not user.is_active:
        raise UnauthorizedException(message="User account is deactivated")

    return user


async def worker_auth(
    request: Request,
    bearer_token: Annotated[
        Optional[HTTPAuthorizationCredentials], Depends(bearer_auth)
    ] = None,
    x_api_key: Annotated[Optional[str], Depends(api_key_header_auth)] = None,
):
    token_value = (bearer_token.credentials if bearer_token else None) or x_api_key
    if not token_value:
        raise UnauthorizedException(message="Invalid authentication credentials")
    token = request.app.state.token
    config: Config = request.app.state.config
    registration_token = config.token
    server_url = config.get_server_url()
    if token_value in [token, registration_token]:
        return
    model_name = request.headers.get("X-Higress-Llm-Model")
    if model_name is not None:
        cred = token_value
        show_len = max(1, min(6, len(cred)))
        masked_token = f"{'*' * (len(cred) - show_len)}{cred[-show_len:]}"
        logger.debug(f"Verifying worker token {masked_token} via server authentication")
        cached_auth = make_auth_token_via_server(request.app.state.http_client_no_proxy)
        is_valid = await cached_auth(server_url, token_value, model_name)
        if is_valid:
            return
    raise UnauthorizedException(message="Invalid authentication credentials")


def make_auth_token_via_server(client: aiohttp.ClientSession):
    @cached(ttl=60)
    async def inner(server_url: str, token: str, model_name: str) -> bool:
        auth_url = f"{server_url.rstrip('/')}/token-auth"
        headers = {
            "Authorization": f"Bearer {token}",
            "X-Higress-Llm-Model": model_name,
        }
        try:
            async with client.get(auth_url, headers=headers) as resp:
                return resp.status == 200
        except aiohttp.ClientError as e:
            logger.error(f"Error verifying token via server: {e}")
            return False

    return inner


def get_scopes(
    request: Request, _current_user: Annotated[User, Depends(get_current_user)]
) -> List[str]:
    api_key: ApiKey = getattr(request.state, "api_key", None)
    if api_key is not None:
        return api_key.scope
    return [PermissionScope.ALL]


def inference_scope(
    request: Request, _current_user: Annotated[User, Depends(get_current_user)]
):
    scopes = get_scopes(request, _current_user)
    if PermissionScope.ALL not in scopes and PermissionScope.INFERENCE not in scopes:
        raise ForbiddenException(
            message="API key does not have permission to access inference features"
        )


def management_scope(
    request: Request, _current_user: Annotated[User, Depends(get_current_user)]
):
    scopes = get_scopes(request, _current_user)
    if PermissionScope.ALL not in scopes and PermissionScope.MANAGEMENT not in scopes:
        raise ForbiddenException(
            message="API key does not have permission to access management features"
        )
