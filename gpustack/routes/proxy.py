import os
from urllib.parse import urlparse
from fastapi.responses import JSONResponse
import httpx
import logging

from fastapi import APIRouter, Request, Response

from gpustack.api.exceptions import (
    BadRequestException,
    ForbiddenException,
)

router = APIRouter()

logger = logging.getLogger(__name__)


ALLOWED_SITES = [
    "https://modelscope.cn",
    "https://www.modelscope.cn",
    "https://huggingface.co",
]

HEADER_FORWARDED_PREFIX = "x-forwarded-"
HEADER_SKIPPED = [
    "host",
    "port",
    "proto",
    "referer",
    "server",
    "content-length",
    "transfer-encoding",
    "cookie",
    "x-forwarded-host",
    "x-forwarded-port",
    "x-forwarded-proto",
    "x-forwarded-server",
]
HF_ENDPOINT = os.getenv("HF_ENDPOINT")

timeout = httpx.Timeout(connect=15.0, read=60.0, write=60.0, pool=10.0)


@router.api_route("", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy(request: Request, url: str):

    validate_http_method(request.method)
    validate_url(url)

    url = replace_hf_endpoint(url)

    forwarded_headers = process_headers(request.headers)

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
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
                headers=response.headers,
                media_type=response.headers.get("Content-Type"),
            )
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"detail": str(e)},
                media_type="application/json",
            )


def validate_http_method(method: str):
    allowed_methods = ["GET", "POST", "PUT", "DELETE"]
    if method not in allowed_methods:
        raise BadRequestException(message=f"HTTP method '{method}' is not allowed")


def validate_url(url: str):
    if not url:
        raise BadRequestException(message="Missing 'url' query parameter")

    try:
        parsed_url = urlparse(url)
    except Exception:
        raise BadRequestException(message="Invalid 'url' query parameter")

    if not parsed_url.netloc or not parsed_url.scheme:
        raise BadRequestException(message="Invalid 'url' query parameter")

    for allowed_site in ALLOWED_SITES:
        parsed_allowed_site_url = urlparse(allowed_site)
        if (
            parsed_url.netloc == parsed_allowed_site_url.netloc
            and parsed_url.scheme == parsed_allowed_site_url.scheme
        ):
            return

    raise ForbiddenException(message="This site is not allowed")


def replace_hf_endpoint(url: str) -> str:
    """
    Replace the huggingface.co domain with the specified endpoint if set.
    """
    if HF_ENDPOINT and url.startswith("https://huggingface.co"):
        return url.replace("https://huggingface.co", HF_ENDPOINT, 1)
    return url


def process_headers(headers):
    processed_headers = {}
    for key, value in headers.items():
        if key.lower() in HEADER_SKIPPED:
            continue
        elif key.lower().startswith(HEADER_FORWARDED_PREFIX):
            new_key = key[len(HEADER_FORWARDED_PREFIX) :]
            processed_headers[new_key] = value
        # set accept-encoding to identity to avoid decompression
        # httpx automatically decodes the content and we want to keep it raw
        # See https://www.python-httpx.org/quickstart/#binary-response-content
        elif key.lower() == "accept-encoding":
            processed_headers[key] = "identity"
        else:
            processed_headers[key] = value

    return processed_headers
