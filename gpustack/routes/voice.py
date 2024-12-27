import logging
from typing import List

from fastapi import APIRouter, Request
from pydantic import BaseModel

from gpustack.routes.openai import proxy_request_by_model
from gpustack.server.deps import SessionDep

router = APIRouter()

logger = logging.getLogger(__name__)


class VoicesResponse(BaseModel):
    model: str
    voices: List[str]


@router.get(
    "/voices",
    response_model=VoicesResponse,
    deprecated=True,
    description=(
        "This endpoint is deprecated and will be removed in future versions. "
        "Please use the `/v1-openai/models` endpoint with the `with_meta` query parameter instead."
    ),
)
async def voices(
    session: SessionDep,
    request: Request,
):
    return await proxy_request_by_model(request, session, "voices")
