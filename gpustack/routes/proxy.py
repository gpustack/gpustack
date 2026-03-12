import os
from urllib.parse import urlparse
import aiohttp
from fastapi.responses import JSONResponse
import logging
from typing import Callable, Optional
from functools import partial

from fastapi import APIRouter, Request, Response

from gpustack.api.exceptions import (
    BadRequestException,
    ForbiddenException,
)
from gpustack.config.config import get_global_config
from gpustack.utils.network import use_proxy_env_for_url

router = APIRouter()

logger = logging.getLogger(__name__)


ALLOWED_SITES = [
    "https://modelscope.cn",
    "https://www.modelscope.cn",
    "https://huggingface.co",
]

HEADER_FORWARDED_PREFIX = "x-forwarded-"
HEADER_SKIPPED = [
    "date",
    "set-cookie",
    "host",
    "port",
    "proto",
    "referer",
    "server",
    "content-length",
    "transfer-encoding",
    "content-encoding",
    "cookie",
    "x-forwarded-host",
    "x-forwarded-port",
    "x-forwarded-proto",
    "x-forwarded-server",
]
HF_ENDPOINT = os.getenv("HF_ENDPOINT")

timeout = aiohttp.ClientTimeout(
    connect=15.0,
    sock_read=60.0,
    sock_connect=10.0,
)


def hf_token_process(url: str, headers: dict) -> dict:
    global_config = get_global_config()
    if global_config.huggingface_token and (
        url.startswith("https://huggingface.co") or HF_ENDPOINT
    ):
        headers["Authorization"] = f"Bearer {global_config.huggingface_token}"
    return headers


@router.api_route("", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy(request: Request, url: str):

    validate_http_method(request.method)
    validate_url(url)

    url = replace_hf_endpoint(url)
    return await proxy_to(request, url, header_func=partial(hf_token_process, url))


async def proxy_to(
    request: Request, url: str, header_func: Optional[Callable[[dict], dict]] = None
):
    forwarded_headers = process_headers(request.headers)
    if header_func is not None:
        forwarded_headers = header_func(forwarded_headers)
    try:
        data = (
            await request.body()
            if request.method in ["POST", "PUT", "DELETE"]
            else None
        )

        use_proxy_env = use_proxy_env_for_url(url)
        async with aiohttp.ClientSession(
            timeout=timeout, trust_env=use_proxy_env
        ) as session:
            async with session.request(
                method=request.method,
                url=url,
                headers=forwarded_headers,
                data=data,
            ) as resp:
                content = await resp.read()
                headers = {
                    k: v
                    for k, v in resp.headers.items()
                    if k.lower() not in HEADER_SKIPPED
                }
                return Response(
                    status_code=resp.status,
                    content=content,
                    headers=headers,
                    media_type=headers.get("Content-Type"),
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


def process_headers(headers: dict) -> dict:
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
