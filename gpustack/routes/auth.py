from datetime import timedelta
from typing import Annotated
from fastapi import APIRouter, Depends, Form, Response
from fastapi.security import (
    OAuth2PasswordRequestForm,
)
from pydantic import BaseModel
from gpustack.api.exceptions import UnauthorizedException
from gpustack.schemas.users import User
from gpustack.security import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    create_access_token,
    verify_password,
)
from gpustack.server.auth import SESSION_COOKIE_NAME, authenticate_user
from gpustack.server.deps import SessionDep


router = APIRouter()


class Token(BaseModel):
    access_token: str
    token_type: str


@router.post("/token")
async def login_for_access_token(
    session: SessionDep, form_data: Annotated[OAuth2PasswordRequestForm, Depends()]
):
    user = await User.one_by_field(session, "username", form_data.username)
    if not user:
        raise UnauthorizedException(message="Incorrect username or password")

    if not verify_password(user.hashed_password, form_data.password):
        raise UnauthorizedException(message="Incorrect username or password")

    access_token = create_access_token(
        username=user.username,
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return Token(access_token=access_token, token_type="bearer")


@router.post("/login")
async def login(
    response: Response,
    session: SessionDep,
    username: Annotated[str, Form()],
    password: Annotated[str, Form()],
):
    user = await authenticate_user(session, username, password)

    access_token = create_access_token(
        username=user.username,
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )

    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=access_token,
        httponly=True,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        expires=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post("/logout")
async def logout(response: Response):
    response.delete_cookie(key=SESSION_COOKIE_NAME)
