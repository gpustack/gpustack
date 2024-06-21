from datetime import timedelta
from typing import Annotated
from fastapi import APIRouter, Form, Response
from pydantic import BaseModel
from gpustack.security import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    create_access_token,
)
from gpustack.server.auth import SESSION_COOKIE_NAME, authenticate_user
from gpustack.server.deps import SessionDep


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
