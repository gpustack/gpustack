import logging
import os
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
import httpx
from starlette.background import BackgroundTask

from gpustack.api.auth import worker_auth
from gpustack.api.exceptions import GatewayTimeoutException, ServiceUnavailableException

router = APIRouter(dependencies=[Depends(worker_auth)])

logger = logging.getLogger(__name__)

PROXY_TIMEOUT = int(os.getenv("GPUSTACK_PROXY_TIMEOUT_SECONDS", 1800))

# Configure connection pool for high concurrency workloads
limits = httpx.Limits(
    max_connections=1000,  # Total connection pool size
    max_keepalive_connections=800,  # Keep-alive connections
    keepalive_expiry=30  # Keep connections alive for 30 seconds
)
client = httpx.AsyncClient(timeout=PROXY_TIMEOUT, limits=limits)


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

        req_proxy = client.build_request(
            request.method,
            httpx.URL(url),
            headers=headers,
            content=content,
        )
        resp_proxy = await client.send(req_proxy, stream=True)

        return StreamingResponse(
            resp_proxy.aiter_raw(),
            status_code=resp_proxy.status_code,
            headers=resp_proxy.headers,
            background=BackgroundTask(resp_proxy.aclose),
        )

    except httpx.TimeoutException as e:
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
