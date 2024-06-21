from fastapi import Depends, Request
from gpustack.config.config import Config
from gpustack.server.db import get_session
from typing import Annotated, Optional
from fastapi.security import (
    APIKeyCookie,
    HTTPAuthorizationCredentials,
    HTTPBasic,
    HTTPBasicCredentials,
    HTTPBearer,
    OAuth2PasswordBearer,
)
from sqlmodel.ext.asyncio.session import AsyncSession
from gpustack.api.exceptions import UnauthorizedException
from gpustack.schemas.users import User
from gpustack.security import (
    decode_access_token,
    verify_password,
)

SESSION_COOKIE_NAME = "gpustack_session"
SYSTEM_USER_PREFIX = "system/"
SYSTEM_WORKER_USER_PREFIX = "system/worker/"
basic_auth = HTTPBasic(auto_error=False)
bearer_auth = HTTPBearer(auto_error=False)
oauth2_bearer_auth = OAuth2PasswordBearer(tokenUrl="/auth/token", auto_error=False)
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
    oauth2_bearer_token: Annotated[Optional[str], Depends(oauth2_bearer_auth)] = None,
    cookie_token: Annotated[Optional[str], Depends(cookie_auth)] = None,
) -> User:
    if basic_credentials and is_system_user(basic_credentials.username):
        server_config: Config = request.app.state.server_config
        return await authenticate_system_user(server_config, basic_credentials)
    elif basic_credentials:
        return await authenticate_basic_user(session, basic_credentials)

    access_token = get_access_token(bearer_token, oauth2_bearer_token, cookie_token)

    return await get_user_from_token(session, access_token)


def is_system_user(username: str) -> bool:
    return username.startswith(SYSTEM_USER_PREFIX)


async def authenticate_system_user(
    config: Config,
    credentials: HTTPBasicCredentials,
) -> User:
    if credentials.username.startswith(SYSTEM_WORKER_USER_PREFIX):
        if credentials.password == config.token:
            return User(username=credentials.username)
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


async def get_user_from_token(session: AsyncSession, access_token: str) -> User:
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


async def authenticate_user(
    session: AsyncSession, username: str, password: str
) -> User:
    user = await User.one_by_field(session, "username", username)
    if not user:
        raise UnauthorizedException(message="Incorrect username or password")

    if not verify_password(user.hashed_password, password):
        raise UnauthorizedException(message="Incorrect username or password")

    return user
