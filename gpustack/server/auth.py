from datetime import datetime
from fastapi import Depends, Request
from gpustack.config.config import Config
from gpustack.schemas.api_keys import ApiKey
from gpustack.server.db import get_session
from typing import Annotated, Optional
from fastapi.security import (
    APIKeyCookie,
    HTTPAuthorizationCredentials,
    HTTPBasic,
    HTTPBasicCredentials,
    HTTPBearer,
)
from sqlmodel.ext.asyncio.session import AsyncSession
from gpustack.api.exceptions import ForbiddenException, UnauthorizedException
from gpustack.schemas.users import User
from gpustack.security import (
    API_KEY_PREFIX,
    decode_access_token,
    verify_hashed_secret,
)

SESSION_COOKIE_NAME = "gpustack_session"
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
    user = None
    if basic_credentials and is_system_user(basic_credentials.username):
        server_config: Config = request.app.state.server_config
        user = await authenticate_system_user(server_config, basic_credentials)
    elif basic_credentials:
        user = await authenticate_basic_user(session, basic_credentials)
    elif cookie_token:
        user = await get_user_from_jwt_token(session, cookie_token)
    elif bearer_token:
        user = await get_user_from_bearer_token(session, bearer_token)

    if user:
        request.state.user = user
        return user

    raise credentials_exception


async def get_admin_user(
    current_user: Annotated[User, Depends(get_current_user)]
) -> User:
    if not current_user.is_admin:
        raise ForbiddenException(message="No permission to access")
    return current_user


def is_system_user(username: str) -> bool:
    return username.startswith(SYSTEM_USER_PREFIX)


async def authenticate_system_user(
    config: Config,
    credentials: HTTPBasicCredentials,
) -> User:
    if credentials.username.startswith(SYSTEM_WORKER_USER_PREFIX):
        if credentials.password == config.token:
            return User(username=credentials.username, is_admin=True)
    raise credentials_exception


async def authenticate_basic_user(
    session: AsyncSession,
    basic_credentials: HTTPBasicCredentials,
) -> User:
    try:
        user = await authenticate_user(
            session, basic_credentials.username, basic_credentials.password
        )
        return user
    except Exception:
        raise credentials_exception


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


async def get_user_from_jwt_token(session: AsyncSession, access_token: str) -> User:
    try:
        payload = decode_access_token(access_token)
        username = payload.get("sub")
    except Exception:
        raise credentials_exception

    if username is None:
        raise credentials_exception

    user = await User.one_by_field(session, "username", username)
    if not user:
        raise credentials_exception
    return user


async def get_user_from_bearer_token(
    session: AsyncSession, bearer_token: HTTPAuthorizationCredentials
) -> User:
    try:
        parts = bearer_token.credentials.split("_")
        if len(parts) == 3 and parts[0] == API_KEY_PREFIX:
            access_key = parts[1]
            secret_key = parts[2]
            api_key = await ApiKey.one_by_field(session, "access_key", access_key)
            if (
                api_key is not None
                and verify_hashed_secret(api_key.hashed_secret_key, secret_key)
                and (api_key.expires_at is None or api_key.expires_at > datetime.now())
            ):
                user = await User.one_by_id(session, api_key.user_id)
                if user is not None:
                    return user
    except Exception:
        raise credentials_exception

    raise credentials_exception


async def authenticate_user(
    session: AsyncSession, username: str, password: str
) -> User:
    user = await User.one_by_field(session, "username", username)
    if not user:
        raise UnauthorizedException(message="Incorrect username or password")

    if not verify_hashed_secret(user.hashed_password, password):
        raise UnauthorizedException(message="Incorrect username or password")

    return user
