from typing import Annotated
from fastapi import APIRouter, Form, Request, Response
from pydantic import BaseModel
from gpustack.api.exceptions import InvalidException
from gpustack.schemas.users import UpdatePassword
from gpustack.security import (
    JWT_TOKEN_EXPIRE_MINUTES,
    JWTManager,
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
    request: Request,
    response: Response,
    session: SessionDep,
    username: Annotated[str, Form()],
    password: Annotated[str, Form()],
):
    user = await authenticate_user(session, username, password)

    jwt_manager: JWTManager = request.app.state.jwt_manager
    access_token = jwt_manager.create_jwt_token(
        username=user.username,
    )

    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=access_token,
        httponly=True,
        max_age=JWT_TOKEN_EXPIRE_MINUTES * 60,
        expires=JWT_TOKEN_EXPIRE_MINUTES * 60,
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
        raise InvalidException(message="Incorrect current password")

    hashed_password = get_secret_hash(update_in.new_password)
    patch = {"hashed_password": hashed_password, "require_password_change": False}
    await user.update(session, patch)
