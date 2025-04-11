import logging
from typing import List, Optional

from fastapi import APIRouter, Body, Request
from pydantic import BaseModel

from gpustack.routes.openai import proxy_request_by_model

router = APIRouter()

logger = logging.getLogger(__name__)


class RerankRequest(BaseModel):
    model: str
    query: str
    documents: List[str]
    top_n: Optional[int] = None
    return_documents: Optional[bool] = True


class RerankUsage(BaseModel):
    total_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None


class RerankResultDocument(BaseModel):
    text: str


class RerankResult(BaseModel):
    index: int
    document: RerankResultDocument
    relevance_score: float


class RerankResponse(BaseModel):
    model: str
    # object: str
    usage: RerankUsage
    results: List[RerankResult]


@router.post("/rerank", response_model=RerankResponse)
async def rerank(request: Request, input_model: RerankRequest = Body(...)):
    return await proxy_request_by_model(request, "rerank")
