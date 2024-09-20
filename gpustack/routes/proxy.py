from fastapi.responses import JSONResponse
import httpx
import logging

from fastapi import APIRouter, Request, Response

from gpustack.api.exceptions import (
    BadRequestException,
    ForbiddenException,
)
from gpustack.server.deps import SessionDep

router = APIRouter()

logger = logging.getLogger(__name__)


ALLOWED_SITES = ["https://modelscope.cn", "https://www.modelscope.cn"]

HEADER_FORWARDED_PREFIX = "x-forwarded-"
HEADER_SKIPPED = ["host", "content-length", "transfer-encoding", "cookie"]


@router.api_route("", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy(session: SessionDep, request: Request, url: str):
    if not url:
        raise BadRequestException(message="Missing 'url' query parameter")

    if not any(url.startswith(domain) for domain in ALLOWED_SITES):
        raise ForbiddenException(message="This domain is not allowed")

    if request.method not in ["GET", "POST", "PUT", "DELETE"]:
        raise BadRequestException(message="Method not allowed")

    forwarded_headers = process_headers(request.headers)

    async with httpx.AsyncClient() as client:
        try:
            if request.method == "GET":
                response = await client.request(
                    request.method, url, headers=forwarded_headers
                )
            else:
                data = (
                    await request.body()
                    if request.method in ["POST", "PUT", "DELETE"]
                    else None
                )
                response = await client.request(
                    request.method, url, headers=forwarded_headers, data=data
                )

            return Response(
                status_code=response.status_code,
                content=response.content,
                media_type=response.headers.get("Content-Type"),
            )
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"detail": str(e)},
                media_type="application/json",
            )


def process_headers(headers):
    processed_headers = {}
    for key, value in headers.items():
        if key.lower() in HEADER_SKIPPED:
            continue
        elif key.lower().startswith(HEADER_FORWARDED_PREFIX):
            new_key = key[len(HEADER_FORWARDED_PREFIX) :]
            processed_headers[new_key] = value
        else:
            processed_headers[key] = value
    return processed_headers
