from datetime import timedelta
from typing import Annotated
from fastapi import APIRouter, Form, Response
from pydantic import BaseModel
from gpustack.api.exceptions import UnauthorizedException
from gpustack.schemas.users import UpdatePassword
from gpustack.security import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    create_access_token,
    get_secret_hash,
    verify_hashed_secret,
)
from gpustack.server.auth import SESSION_COOKIE_NAME, authenticate_user
from gpustack.server.deps import CurrentUserDep, SessionDep


router = APIRouter()


class Token(BaseModel):
    access_token: str
    token_type: str


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


@router.post("/update-password")
async def update_password(
    session: SessionDep,
    user: CurrentUserDep,
    update_in: UpdatePassword,
):
    if not verify_hashed_secret(user.hashed_password, update_in.current_password):
        raise UnauthorizedException(message="Incorrect current password")

    hashed_password = get_secret_hash(update_in.new_password)
    await user.update(session, {"hashed_password": hashed_password})
