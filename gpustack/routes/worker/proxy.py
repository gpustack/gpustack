import asyncio
import logging
import aiohttp
import random
from typing import Callable, Dict
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from gpustack.api.auth import worker_auth
from gpustack.api.exceptions import GatewayTimeoutException, ServiceUnavailableException
from gpustack import envs
from gpustack.utils.network import use_proxy_env_for_url

router = APIRouter(dependencies=[Depends(worker_auth)])

logger = logging.getLogger(__name__)

router_header_key = "x-gpustack-model"


@router.api_route(
    "/proxy/{path:path}",
    methods=["GET", "POST", "OPTIONS", "HEAD"],
)
async def proxy(path: str, request: Request):
    worker_ip_getter: Callable[[], str] = request.app.state.worker_ip_getter
    if worker_ip_getter is None:
        worker_ip_getter = localhost_fallback
    target_service_port = getattr(
        request.state, "x_target_port", request.headers.get("X-Target-Port")
    )
    if not target_service_port:
        raise HTTPException(status_code=400, detail="Missing X-Target-Port header")

    try:
        logger.debug(
            f"Proxying request to worker at port {target_service_port} for path: {path}"
        )
        url = f"http://{worker_ip_getter()}:{target_service_port}/{path}"
        headers = dict(request.headers)
        headers.pop("host", None)
        headers.pop("transfer-encoding", None)
        if headers.get("transfer-encoding", "").lower() == "chunked":

            async def body_generator():
                async for chunk in request.stream():
                    yield chunk

            content = body_generator()
        else:
            content = await request.body()

        async def stream_response(resp):
            async for chunk in resp.content.iter_chunked(1024):
                yield chunk

        use_proxy_env = use_proxy_env_for_url(url)
        http_client: aiohttp.ClientSession = (
            request.app.state.http_client
            if use_proxy_env
            else request.app.state.http_client_no_proxy
        )
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


def localhost_fallback() -> str:
    return "127.0.0.1"


def get_model_instance_info_from_model_name(request: Request) -> int:
    """
    Get model instance port and generic proxy support from model name in header "x-gpustack-model"

    Return the model instance port and support of generic proxy or not.
    """
    model_name = request.headers.get(router_header_key, None)
    if model_name is None:
        raise HTTPException(
            status_code=400, detail=f"Missing {router_header_key} header"
        )
    instance_ports: Dict[int, int] = request.app.state.instance_port_by_model_name(
        model_name
    )
    if len(instance_ports) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No running model instance found for model name: {model_name}",
        )
    instances = list(instance_ports.keys())
    model_instance_id = random.choice(instances)
    port = instance_ports[model_instance_id]
    logger.debug(
        f"Found ports for model instances {instances} of model {model_name}, selected port: {port}"
    )
    return port


async def set_port_from_model_name(request: Request, call_next):
    model_name = request.headers.get(router_header_key, None)
    if model_name is None:
        return await call_next(request)
    try:
        port = get_model_instance_info_from_model_name(request)
        request.scope["path"] = f"/proxy{request.url.path}"
        request.state.x_target_port = str(port)
        return await call_next(request)
    except HTTPException as e:
        logger.debug("failed to find model instance for proxying: %s", e.detail)
        return await call_next(request)
