import asyncio
import logging
import aiohttp
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from gpustack.api.auth import worker_auth
from gpustack.api.exceptions import GatewayTimeoutException, ServiceUnavailableException
from gpustack import envs

router = APIRouter(dependencies=[Depends(worker_auth)])

logger = logging.getLogger(__name__)


@router.api_route(
    "/proxy/{path:path}",
    methods=["GET", "POST", "OPTIONS", "HEAD"],
)
async def proxy(path: str, request: Request):
    target_service_port = request.headers.get("X-Target-Port")
    if not target_service_port:
        raise HTTPException(status_code=400, detail="Missing X-Target-Port header")

    try:
        url = f"http://127.0.0.1:{target_service_port}/{path}"
        content = await request.body()
        headers = dict(request.headers)
        headers.pop("host", None)

        async def stream_response(resp):
            async for chunk in resp.content.iter_chunked(1024):
                yield chunk

        http_client: aiohttp.ClientSession = request.app.state.http_client
        timeout = aiohttp.ClientTimeout(total=envs.PROXY_TIMEOUT)
        resp = await http_client.request(
            method=request.method,
            url=url,
            headers=headers,
            data=content,
            timeout=timeout,
        )

        return StreamingResponse(
            stream_response(resp),
            status_code=resp.status,
            headers=dict(resp.headers),
            background=BackgroundTask(resp.close),
        )

    except asyncio.TimeoutError as e:
        error_message = f"Request to {url} timed out"
        if str(e):
            error_message += f": {e}"
        raise GatewayTimeoutException(
            message=error_message,
            is_openai_exception=True,
        )
    except Exception as e:
        error_message = "An unexpected error occurred"
        if str(e):
            error_message += f": {e}"
        raise ServiceUnavailableException(
            message=error_message,
            is_openai_exception=True,
        )
