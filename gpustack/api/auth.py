import uuid
from datetime import datetime, timezone
import logging
from fastapi import Depends, Request
from gpustack.config.config import Config
from gpustack.schemas.config import GatewayModeEnum
from gpustack.server.db import get_session
from typing import Annotated, Optional, Tuple
from fastapi.security import (
    APIKeyCookie,
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
from gpustack.schemas.api_keys import ApiKey
from gpustack.schemas.users import User, UserRole
from gpustack.security import (
    API_KEY_PREFIX,
    JWTManager,
    verify_hashed_secret,
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
cookie_auth = APIKeyCookie(name=SESSION_COOKIE_NAME, auto_error=False)

credentials_exception = UnauthorizedException(
    message="Invalid authentication credentials"
)


async def get_current_user(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
    basic_credentials: Annotated[
        Optional[HTTPBasicCredentials], Depends(basic_auth)
    ] = None,
    bearer_token: Annotated[
        Optional[HTTPAuthorizationCredentials], Depends(bearer_auth)
    ] = None,
    cookie_token: Annotated[Optional[str], Depends(cookie_auth)] = None,
) -> User:
    if hasattr(request.state, "user"):
        user: User = getattr(request.state, "user")
        return user
    api_key: Optional[ApiKey] = None
    user = None
    if basic_credentials and is_system_user(basic_credentials.username):
        server_config: Config = request.app.state.server_config
        user = await authenticate_system_user(server_config, basic_credentials)
    elif basic_credentials:
        user = await authenticate_basic_user(session, basic_credentials)
    elif cookie_token:
        jwt_manager: JWTManager = request.app.state.jwt_manager
        user = await get_user_from_jwt_token(session, jwt_manager, cookie_token)
    elif bearer_token:
        user, api_key = await get_user_from_bearer_token(session, bearer_token)

    server_config: Config = request.app.state.server_config

    def client_ip_getter() -> str:
        if server_config.gateway_mode == GatewayModeEnum.embedded:
            return request.headers.get("X-GPUStack-Real-IP", "")
        else:
            return request.client.host

    if user is None and client_ip_getter() == "127.0.0.1":
        if not server_config.force_auth_localhost:
            try:
                user = await User.first_by_field(session, "is_admin", True)
            except Exception as e:
                raise InternalServerErrorException(message=f"Failed to get user: {e}")
    if user:
        if not user.is_active:
            raise UnauthorizedException(message="User account is deactivated")
        request.state.user = user
        if api_key is not None:
            request.state.api_key = api_key
        return user

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


def parse_uuid(value: str) -> Optional[str]:
    try:
        uuid.UUID(value)
        return value
    except ValueError:
        return None


async def get_user_from_bearer_token(
    session: AsyncSession, bearer_token: HTTPAuthorizationCredentials
) -> Tuple[Optional[User], Optional[ApiKey]]:
    try:
        parts = bearer_token.credentials.split("_", maxsplit=2)
        if len(parts) == 3 and parts[0] == API_KEY_PREFIX:
            access_key = parts[1]
            secret_key = parts[2]
            # if access_key is a valid uuid, it's legacy worker re-registering with legacy token
            worker_uuid = parse_uuid(access_key)
            if worker_uuid is not None:
                access_key = ""

            api_key = await APIKeyService(session).get_by_access_key(access_key)
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
):
    if not bearer_token:
        raise UnauthorizedException(message="Invalid authentication credentials")
    token = request.app.state.token
    registration_token = request.app.state.config.token
    if bearer_token.credentials not in [token, registration_token]:
        raise UnauthorizedException(message="Invalid authentication credentials")
