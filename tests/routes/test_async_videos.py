import json
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.responses import JSONResponse, Response
from starlette.requests import Request

from gpustack.api.exceptions import NotFoundException, ServiceUnavailableException
from gpustack.api.tenant import TenantContext
from gpustack.routes import openai, videos
from gpustack.security import JWTManager
from gpustack.utils.model_instance_workers import ModelInstanceEndpoint
from tests.utils.mock import mock_async_session

original_proxy = openai.proxy_request_by_model


@pytest.fixture
def video_api(monkeypatch):
    app = SimpleNamespace(
        state=SimpleNamespace(
            jwt_manager=JWTManager("video-test-secret" * 4),
            http_client=object(),
            http_client_no_proxy=object(),
        )
    )
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/videos",
            "query_string": b"",
            "headers": [],
            "app": app,
        }
    )
    request.state.video_target = {
        "model": "published-wan",
        "model_id": 2,
        "route_id": 9,
        "instance_id": 4,
        "worker_id": 3,
        "registry": "model-2-4.static",
        "port": 40000,
        "restart_count": 0,
    }
    job = {"id": "native-task", "status": "queued", "progress": 0}
    create = AsyncMock(return_value=JSONResponse(job))
    monkeypatch.setattr(openai, "proxy_request_by_model", create)
    session_context = mock_async_session()
    session = session_context.__aenter__.return_value
    access_result = MagicMock()
    access_result.first.return_value = 9
    session.exec.return_value = access_result
    monkeypatch.setattr(videos, "async_session", lambda: session_context)
    monkeypatch.setattr(openai, "async_session", lambda: session_context)
    instance = SimpleNamespace(id=4, model_id=2, deleted_at=None, restart_count=0)
    lookup = AsyncMock(return_value=instance)
    monkeypatch.setattr(videos.ModelInstance, "one_by_id", lookup)
    monkeypatch.setattr(
        videos,
        "model_instance_endpoints",
        lambda _: [ModelInstanceEndpoint(2, 4, 3, 40000)],
    )
    worker = SimpleNamespace(id=3, deleted_at=None)
    monkeypatch.setattr(
        videos.WorkerService, "get_by_id", AsyncMock(return_value=worker)
    )
    forward = AsyncMock(
        return_value=(
            SimpleNamespace(status=200, headers={"Content-Type": "application/json"}),
            json.dumps({**job, "status": "in_progress", "progress": 42}).encode(),
        )
    )
    monkeypatch.setattr(videos, "request_to_worker", forward)
    user = SimpleNamespace(id=11, is_admin=False)
    ctx = TenantContext(
        user=user, is_platform_admin=False, current_principal_id=7, org_role=None
    )
    return SimpleNamespace(
        request=request,
        user=user,
        ctx=ctx,
        job=job,
        create=create,
        access=session.exec,
        access_result=access_result,
        instance=instance,
        lookup=lookup,
        worker=worker,
        forward=forward,
    )


async def _create(api):
    response = await videos.create_video(api.request, api.user, api.ctx)
    return json.loads(response.body)


def test_async_routes_are_on_api_not_model_load_balancer():
    paths = {
        (route.path, method)
        for route in openai.get_api_router().routes
        for method in route.methods
    }
    assert {
        ("/videos", "POST"),
        ("/videos/{video_id}", "GET"),
        ("/videos/{video_id}/content", "GET"),
    } <= paths
    assert all(
        "/videos" not in prefix.prefixes for prefix in openai.openai_model_prefixes
    )


@pytest.mark.asyncio
async def test_create_and_poll_preserve_public_id_and_pin_original_instance(video_api):
    api = video_api
    job = await _create(api)
    assert job["id"] != "native-task"
    assert job["status"] == "queued"
    response = await videos.retrieve_video(job["id"], api.request, api.user, api.ctx)
    assert json.loads(response.body) == {**job, "status": "in_progress", "progress": 42}
    assert api.forward.call_args.kwargs["path"] == "v1/videos/native-task"
    assert api.forward.call_args.kwargs["worker"] is api.worker
    assert api.forward.call_args.kwargs["headers"] == {
        videos.router_header_key: "model-2-4.static"
    }
    api.lookup.assert_awaited_once_with(api.lookup.call_args.args[0], 4, refresh=True)
    api.access.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "access", ["another-user", "another-tenant", "tampered", "invalid"]
)
async def test_video_id_cannot_cross_user_tenant_or_be_forged(video_api, access):
    api = video_api
    job = await _create(api)
    video_id = job["id"]
    if access == "another-user":
        api.user.id = 12
    elif access == "another-tenant":
        api.ctx.current_principal_id = 8
    elif access == "tampered":
        video_id = video_id[:-8] + "tampered"
    else:
        video_id = "native-task"
    with pytest.raises(NotFoundException):
        await videos.retrieve_video(video_id, api.request, api.user, api.ctx)
    api.access.assert_not_awaited()
    api.forward.assert_not_awaited()


@pytest.mark.asyncio
async def test_revoked_model_access_blocks_status_and_download(video_api):
    api = video_api
    job = await _create(api)
    api.access_result.first.return_value = None
    for handler in (videos.retrieve_video, videos.download_video):
        with pytest.raises(NotFoundException):
            await handler(job["id"], api.request, api.user, api.ctx)
    api.forward.assert_not_awaited()


@pytest.mark.asyncio
async def test_restarted_instance_does_not_send_job_to_another_replica(video_api):
    api = video_api
    job = await _create(api)
    api.instance.restart_count = 1
    with pytest.raises(ServiceUnavailableException):
        await videos.retrieve_video(job["id"], api.request, api.user, api.ctx)
    api.forward.assert_not_awaited()


@pytest.mark.asyncio
async def test_engine_error_status_and_body_are_preserved(video_api):
    api = video_api
    job = await _create(api)
    body = b'{"detail":"Unknown video"}'
    api.forward.return_value = (SimpleNamespace(status=404, headers={}), body)
    response = await videos.retrieve_video(job["id"], api.request, api.user, api.ctx)
    assert response.status_code == 404
    assert response.body == body
    api.create.return_value = Response(body, status_code=422)
    response = await videos.create_video(api.request, api.user, api.ctx)
    assert response.status_code == 422
    assert response.body == body


@pytest.mark.asyncio
@pytest.mark.parametrize("native_id", ["../healthz", "", 42, None])
async def test_create_rejects_invalid_engine_job_ids(video_api, native_id):
    api = video_api
    api.create.return_value = JSONResponse({**api.job, "id": native_id})
    with pytest.raises(ServiceUnavailableException):
        await _create(api)


@pytest.mark.asyncio
async def test_download_streams_binary_and_removes_worker_transfer_headers(
    video_api, monkeypatch
):
    api = video_api
    job = await _create(api)
    calls = []

    async def stream(**kwargs):
        calls.append(kwargs)
        headers = {
            "Content-Type": "video/mp4",
            "Content-Length": "10",
            "Content-Encoding": "gzip",
        }
        yield b"\x00\x00ftyp", headers, 200
        yield b"mp42", headers, 200

    monkeypatch.setattr(videos, "stream_to_worker", stream)
    response = await videos.download_video(job["id"], api.request, api.user, api.ctx)
    chunks = [item async for item in response.body_iterator]
    assert b"".join(item[0] for item in chunks) == b"\x00\x00ftypmp42"
    assert chunks[0][1] == {"Content-Type": "video/mp4"}
    assert calls[0]["path"] == "v1/videos/native-task/content"
    assert calls[0]["worker"] is api.worker
    assert calls[0]["raw"] is True


@pytest.mark.asyncio
async def test_expired_handle_is_not_usable(video_api):
    api = video_api
    job = await _create(api)
    manager = api.request.app.state.jwt_manager
    payload = manager.decode_jwt_data(job["id"][len(videos._ID_PREFIX) :])
    expired = videos._ID_PREFIX + manager.create_token(
        payload, expires_delta=timedelta(seconds=-1)
    )
    with pytest.raises(NotFoundException):
        await videos.retrieve_video(expired, api.request, api.user, api.ctx)
    api.forward.assert_not_awaited()


@pytest.mark.asyncio
async def test_async_create_rejects_streaming_requests(video_api, monkeypatch):
    from gpustack.api.exceptions import BadRequestException

    monkeypatch.setattr(
        openai,
        "parse_json_body",
        AsyncMock(return_value=({"model": "wan", "stream": True}, "wan", True)),
    )
    with pytest.raises(BadRequestException) as error:
        await original_proxy(video_api.request, video_api.user)
    assert error.value.status_code == 400


@pytest.mark.asyncio
async def test_video_creation_checks_context_before_dispatch(video_api, monkeypatch):
    api = video_api
    api.access_result.first.return_value = None
    monkeypatch.setattr(openai, "proxy_request_by_model", original_proxy)
    monkeypatch.setattr(
        openai,
        "parse_request_body",
        AsyncMock(
            return_value=("published-wan", False, {"model": "published-wan"}, None)
        ),
    )
    monkeypatch.setattr(
        openai.UserService, "model_allowed_for_user", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(
        openai.ModelRouteService,
        "resolve_route_targets",
        AsyncMock(
            return_value=[
                SimpleNamespace(model_id=2, weight=1, overridden_model_name=None)
            ]
        ),
    )
    monkeypatch.setattr(
        openai.ModelRouteService,
        "get_by_name",
        AsyncMock(return_value=SimpleNamespace(id=9)),
    )
    monkeypatch.setattr(
        openai.ModelService, "get_by_id", AsyncMock(return_value=SimpleNamespace(id=2))
    )
    choose_instance = AsyncMock()
    monkeypatch.setattr(openai, "get_running_instance", choose_instance)
    with pytest.raises(NotFoundException):
        await videos.create_video(api.request, api.user, api.ctx)
    choose_instance.assert_not_awaited()
    api.forward.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [False, True])
async def test_video_transfer_errors_do_not_expose_exception_details(
    video_api, monkeypatch, timeout, caplog
):
    from gpustack.server import worker_request

    api = video_api
    job = await _create(api)
    secret = "private-worker-host: traceback with sensitive details"
    error = TimeoutError(secret) if timeout else RuntimeError(secret)

    connection = MagicMock()
    connection.__aenter__ = AsyncMock(side_effect=error)
    connection.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(
        worker_request, "_request_to_worker", lambda **kwargs: connection
    )
    response = await videos.download_video(job["id"], api.request, api.user, api.ctx)
    messages = []

    async def send(message):
        messages.append(message)

    await response.stream_response(send)
    assert messages[0]["status"] == (504 if timeout else 503)
    body = b"".join(message.get("body", b"") for message in messages)
    assert secret.encode() not in body
    assert secret in caplog.text
    assert json.loads(body)["error"]["message"] == (
        "Video content request timed out" if timeout else "Video content is unavailable"
    )
