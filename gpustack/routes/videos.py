"""OpenAI video jobs pinned to their creating inference instance."""

import asyncio
import json
import re
from datetime import timedelta
from typing import Dict
from urllib.parse import quote

import aiohttp
import jwt
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from gpustack import envs
from gpustack.api.exceptions import (
    GatewayTimeoutException,
    NotFoundException,
    ServiceUnavailableException,
)
from gpustack.api.responses import StreamingResponseWithStatusCode
from gpustack.gateway.utils import router_header_key
from gpustack.routes import openai
from gpustack.schemas.models import ModelInstance
from gpustack.server.db import async_session
from gpustack.server.deps import CurrentUserDep, TenantContextDep
from gpustack.server.services import WorkerService
from gpustack.server.worker_request import request_to_worker, stream_to_worker
from gpustack.utils.model_instance_workers import model_instance_endpoints

router = APIRouter()
_ID_PREFIX = "video_"
_JOB_KIND = "video-job-v1"


def _not_found() -> NotFoundException:
    return NotFoundException(message="Video not found", is_openai_exception=True)


def _job_payload(request: Request, video_id: str, user, ctx) -> Dict:
    try:
        if not video_id.startswith(_ID_PREFIX) or len(video_id) > 4096:
            raise _not_found()
        payload = request.app.state.jwt_manager.decode_jwt_data(
            video_id[len(_ID_PREFIX) :]
        )
        if (
            payload.get("kind") != _JOB_KIND
            or payload.get("user_id") != user.id
            or payload.get("principal_id") != ctx.current_principal_id
        ):
            raise _not_found()
        return payload
    except (jwt.InvalidTokenError, KeyError, TypeError, AttributeError):
        raise _not_found() from None


@router.post("/videos")
async def create_video(
    request: Request, user: CurrentUserDep, ctx: TenantContextDep
) -> Response:
    """Create a job and return an authenticated, instance-bound public ID."""
    request.state.tenant_context = ctx
    response = await openai.proxy_request_by_model(request, user)
    if not 200 <= response.status_code < 300:
        return response
    try:
        job = json.loads(response.body)
        native_id = job["id"]
        if not isinstance(native_id, str) or not re.fullmatch(
            r"[A-Za-z0-9_-]{1,256}", native_id
        ):
            raise ValueError("Invalid task ID")
    except (ValueError, TypeError, KeyError):
        raise ServiceUnavailableException(
            message="Inference service did not return a video task ID",
            is_openai_exception=True,
        ) from None

    # A signed handle survives API restarts without a process-local job map.
    # It is scoped to its creator and tenant, and cannot reroute to another replica.
    payload = {
        **request.state.video_target,
        "kind": _JOB_KIND,
        "native_id": native_id,
        "user_id": user.id,
        "principal_id": ctx.current_principal_id,
    }
    token = request.app.state.jwt_manager.create_token(
        payload, expires_delta=timedelta(days=1)
    )
    job["id"] = _ID_PREFIX + token
    headers = openai._worker_response_headers(response.headers)
    return JSONResponse(job, status_code=response.status_code, headers=headers)


async def _resolve_job(request: Request, payload: Dict, user, ctx):
    async with async_session() as session:
        await openai._assert_video_route_access(
            session,
            payload["model"],
            payload["route_id"],
            user,
            ctx,
            getattr(request.state, "api_key", None),
        )
        instance = await ModelInstance.one_by_id(
            session, payload["instance_id"], refresh=True
        )
        if instance is None or instance.deleted_at is not None:
            raise _not_found()
        endpoints = model_instance_endpoints(instance)
        available = any(
            endpoint.worker_id == payload["worker_id"]
            and f"{endpoint.registry_name}.static" == payload["registry"]
            and endpoint.port == payload["port"]
            for endpoint in endpoints
        )
        if (
            not available
            or instance.model_id != payload["model_id"]
            or (instance.restart_count or 0) != payload["restart_count"]
        ):
            raise ServiceUnavailableException(
                message="The inference instance that owns this video is unavailable",
                is_openai_exception=True,
            )
        worker = await WorkerService(session).get_by_id(payload["worker_id"])
        if worker is None or worker.deleted_at is not None:
            raise _not_found()
    request.state.model_route_id = payload["route_id"]
    return worker


@router.get("/videos/{video_id}")
async def retrieve_video(
    video_id: str, request: Request, user: CurrentUserDep, ctx: TenantContextDep
) -> Response:
    """Retrieve the creating instance's job status and progress."""
    payload = _job_payload(request, video_id, user, ctx)
    worker = await _resolve_job(request, payload, user, ctx)
    try:
        response, body = await request_to_worker(
            worker=worker,
            method="GET",
            path=f"v1/videos/{quote(payload['native_id'], safe='')}",
            headers={router_header_key: payload["registry"]},
            proxy_client=request.app.state.http_client,
            no_proxy_client=request.app.state.http_client_no_proxy,
            raise_on_error=False,
        )
    except asyncio.TimeoutError:
        raise GatewayTimeoutException(
            message="Video status request timed out", is_openai_exception=True
        ) from None
    except aiohttp.ClientError:
        raise ServiceUnavailableException(
            message="Video inference service is unavailable", is_openai_exception=True
        ) from None
    headers = openai._worker_response_headers(response.headers)
    if not 200 <= response.status < 300:
        return Response(body, status_code=response.status, headers=headers)
    try:
        job = json.loads(body)
        if job["id"] != payload["native_id"]:
            raise ValueError("Unexpected task ID")
        job["id"] = video_id
    except (ValueError, TypeError, KeyError):
        raise ServiceUnavailableException(
            message="Invalid video task response", is_openai_exception=True
        ) from None
    return JSONResponse(job, status_code=response.status, headers=headers)


def _video_content_error(
    error: Exception, _timeout: aiohttp.ClientTimeout
) -> tuple[str, int]:
    """Return public transfer errors; worker_request logs the private exception."""
    timed_out = isinstance(error, asyncio.TimeoutError)
    status = 504 if timed_out else 503
    message = (
        "Video content request timed out"
        if timed_out
        else "Video content is unavailable"
    )
    body = {
        "error": {
            "message": message,
            "type": "GatewayTimeout" if timed_out else "ServiceUnavailable",
            "code": status,
        }
    }
    return json.dumps(body), status


@router.get("/videos/{video_id}/content")
async def download_video(
    video_id: str, request: Request, user: CurrentUserDep, ctx: TenantContextDep
) -> Response:
    """Stream completed video bytes without buffering the complete MP4 in the API."""
    payload = _job_payload(request, video_id, user, ctx)
    worker = await _resolve_job(request, payload, user, ctx)

    async def content():
        async for chunk, headers, status in stream_to_worker(
            worker=worker,
            method="GET",
            path=f"v1/videos/{quote(payload['native_id'], safe='')}/content",
            headers={router_header_key: payload["registry"]},
            params=dict(request.query_params),
            proxy_client=request.app.state.http_client,
            no_proxy_client=request.app.state.http_client_no_proxy,
            timeout=aiohttp.ClientTimeout(total=envs.PROXY_TIMEOUT),
            on_exception=_video_content_error,
            raw=True,
        ):
            headers = openai._worker_response_headers(headers)
            if status >= 400:
                headers.setdefault("Content-Type", "application/json")
            yield chunk, headers, status

    return StreamingResponseWithStatusCode(content(), media_type="video/mp4")
