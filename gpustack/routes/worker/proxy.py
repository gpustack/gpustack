import logging
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
import httpx
from starlette.background import BackgroundTask

from gpustack.api.auth import worker_auth

router = APIRouter(dependencies=[Depends(worker_auth)])

logger = logging.getLogger(__name__)

client = httpx.AsyncClient(timeout=600)


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

    except httpx.RequestError as e:
        logger.error(f"Error during request to {url}: {e}")
        raise HTTPException(
            status_code=502, detail=f"Error connecting to target service: {e}"
        )
