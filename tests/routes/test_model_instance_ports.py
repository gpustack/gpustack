import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import aiohttp
import httpx
import pytest
from fastapi import Depends, FastAPI

from gpustack.api.exceptions import (
    ForbiddenException,
    NotFoundException,
    ServiceUnavailableException,
    register_handlers,
)
from gpustack.api.auth import get_current_user, get_worker_principal
from gpustack.routes import model_instance_ports as routes
from gpustack.schemas.models import (
    DistributedServerCoordinateModeEnum,
    DistributedServers,
    ModelInstanceSubordinateWorker,
)
from gpustack.schemas.port_reservations import ModelPortClaim
from gpustack.schemas.principals import PrincipalType
from tests.utils.model import new_model_instance


@pytest.fixture
def api_request():
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                http_client=MagicMock(), http_client_no_proxy=MagicMock()
            )
        )
    )


def _workers():
    return [SimpleNamespace(id=1, cluster_id=1), SimpleNamespace(id=2, cluster_id=1)]


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["reject", "timeout", "cancel"])
async def test_partial_claim_rolls_back_every_participant(
    api_request, monkeypatch, outcome
):
    monkeypatch.setattr(routes, "_members", AsyncMock(return_value=_workers()))
    calls = []

    async def rpc(**kwargs):
        calls.append((kwargs["worker"].id, kwargs["method"]))
        if kwargs["method"] == "DELETE":
            return SimpleNamespace(status=204), None
        if kwargs["worker"].id == 2:
            if outcome == "timeout":
                raise aiohttp.ClientError("connection lost after accepting")
            if outcome == "cancel":
                raise asyncio.CancelledError()
            return (
                SimpleNamespace(status=200),
                json.dumps({"accepted": False, "unavailable_ports": [40021]}).encode(),
            )
        return SimpleNamespace(status=200), b'{"accepted": true}'

    monkeypatch.setattr(routes, "request_to_worker", rpc)
    claim = ModelPortClaim(token=uuid4(), ports=[40020, 40021])
    if outcome == "reject":
        result = await routes.reserve(7, claim, api_request, None, None)
        assert not result.accepted
        assert result.unavailable_ports == [40021]
    else:
        error = (
            ServiceUnavailableException
            if outcome == "timeout"
            else asyncio.CancelledError
        )
        with pytest.raises(error):
            await routes.reserve(7, claim, api_request, None, None)
    assert calls[:2] == [(1, "POST"), (2, "POST")]
    assert set(calls[2:]) == {(1, "DELETE"), (2, "DELETE")}


@pytest.mark.asyncio
async def test_successful_claim_is_retained_until_explicit_cleanup(
    api_request, monkeypatch
):
    monkeypatch.setattr(routes, "_members", AsyncMock(return_value=_workers()))
    rpc = AsyncMock(return_value=(SimpleNamespace(status=200), b'{"accepted": true}'))
    monkeypatch.setattr(routes, "request_to_worker", rpc)
    claim = ModelPortClaim(token=uuid4(), ports=[40020])
    assert (await routes.reserve(7, claim, api_request, None, None)).accepted
    assert [call.kwargs["method"] for call in rpc.call_args_list] == ["POST", "POST"]
    await routes.release(7, claim.token, api_request, None, None)
    assert [call.kwargs["method"] for call in rpc.call_args_list][2:] == [
        "DELETE",
        "DELETE",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["follower", "admin", "cross_cluster", "invisible"])
async def test_member_resolution_enforces_primary_and_visibility(monkeypatch, case):
    instance = new_model_instance(7, "model", 1, worker_id=1)
    instance.cluster_id = 1
    instance.distributed_servers = DistributedServers(
        mode=DistributedServerCoordinateModeEnum.INITIALIZE_LATER
    )
    caller = (
        None if case == "admin" else SimpleNamespace(id=2 if case == "follower" else 1)
    )
    ctx = SimpleNamespace(user=SimpleNamespace(worker=caller))
    visibility = MagicMock()
    if case == "invisible":
        visibility.side_effect = NotFoundException("not visible")
    monkeypatch.setattr(routes, "assert_resource_visible", visibility)
    monkeypatch.setattr(
        routes.ModelInstance, "one_by_id", AsyncMock(return_value=instance)
    )
    worker = SimpleNamespace(id=1, cluster_id=2 if case == "cross_cluster" else 1)
    lookup = AsyncMock(return_value=worker)
    monkeypatch.setattr(routes.Worker, "one_by_id", lookup)
    error = ForbiddenException if case in ("follower", "admin") else NotFoundException
    with pytest.raises(error):
        await routes._members(None, ctx, 7)
    visibility.assert_called_once()
    if case != "cross_cluster":
        lookup.assert_not_called()


@pytest.mark.asyncio
async def test_members_release_database_connection_before_worker_rpc(monkeypatch):
    instance = new_model_instance(7, "model", 1, worker_id=2)
    instance.cluster_id = 1
    instance.distributed_servers = DistributedServers(
        mode=DistributedServerCoordinateModeEnum.INITIALIZE_LATER,
        subordinate_workers=[ModelInstanceSubordinateWorker(worker_id=1)],
    )
    ctx = SimpleNamespace(user=SimpleNamespace(worker=SimpleNamespace(id=2)))
    session = AsyncMock()
    monkeypatch.setattr(routes, "assert_resource_visible", MagicMock())
    monkeypatch.setattr(
        routes.ModelInstance, "one_by_id", AsyncMock(return_value=instance)
    )
    monkeypatch.setattr(
        routes.Worker,
        "one_by_id",
        AsyncMock(side_effect=lambda session, worker_id: _workers()[worker_id - 1]),
    )
    workers = await routes._members(session, ctx, 7)
    assert [worker.id for worker in workers] == [2, 1]
    session.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_snapshot_requires_matching_worker_identity(api_request, monkeypatch):
    monkeypatch.setattr(routes, "_members", AsyncMock(return_value=_workers()))
    rpc = AsyncMock(
        return_value=(
            SimpleNamespace(status=200),
            json.dumps(
                {
                    "worker_id": 1,
                    "generation": str(uuid4()),
                    "port_range": "40000-40063",
                    "unavailable_ports": [],
                }
            ).encode(),
        )
    )
    monkeypatch.setattr(routes, "request_to_worker", rpc)
    with pytest.raises(ServiceUnavailableException) as error:
        await routes.availability(7, api_request, None, None)
    assert "identity mismatch" in error.value.message


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["GET", "POST", "DELETE"])
async def test_internal_coordination_router_denies_ordinary_users(monkeypatch, method):
    from gpustack.routes.routes import worker_client_router
    from gpustack.server.deps import get_session, get_tenant_context

    app = FastAPI()
    app.include_router(
        worker_client_router,
        prefix="/v2",
        dependencies=[Depends(get_worker_principal)],
    )
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(
        kind=PrincipalType.USER, is_admin=False
    )
    app.dependency_overrides[get_session] = lambda: None
    app.dependency_overrides[get_tenant_context] = lambda: None
    register_handlers(app)
    members = AsyncMock()
    monkeypatch.setattr(routes, "_members", members)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://server"
    ) as client:
        response = await client.request(
            method, "/v2/model-instances/7/port-reservations"
        )
    assert response.status_code == 403
    members.assert_not_awaited()


@pytest.mark.asyncio
async def test_release_waits_for_and_reports_all_failed_workers(
    api_request, monkeypatch, caplog
):
    monkeypatch.setattr(routes, "_members", AsyncMock(return_value=_workers()))
    second_started, finish_second = asyncio.Event(), asyncio.Event()

    async def rpc(**kwargs):
        if kwargs["worker"].id == 2:
            second_started.set()
            await finish_second.wait()
        raise aiohttp.ClientError("worker unavailable")

    monkeypatch.setattr(routes, "request_to_worker", rpc)
    task = asyncio.create_task(routes.release(7, uuid4(), api_request, None, None))
    try:
        await asyncio.wait_for(second_started.wait(), timeout=1)
        await asyncio.sleep(0)
        assert not task.done(), "Cleanup must collect every participant's result"
    finally:
        finish_second.set()
        with pytest.raises(ServiceUnavailableException) as error:
            await task
    assert "[1, 2]" in error.value.message
    assert "worker 1" in caplog.text
    assert "worker 2" in caplog.text


@pytest.mark.asyncio
async def test_cancelled_reservation_observes_background_rollback_failure(
    api_request, monkeypatch, caplog
):
    monkeypatch.setattr(routes, "_members", AsyncMock(return_value=_workers()))
    posting, rolling_back, finish_cleanup, finished = (
        asyncio.Event() for _ in range(4)
    )
    original_rollback = routes._rollback

    async def rollback(*args):
        try:
            await original_rollback(*args)
        finally:
            finished.set()

    async def rpc(**kwargs):
        if kwargs["method"] == "POST":
            posting.set()
            await asyncio.Event().wait()
        rolling_back.set()
        await finish_cleanup.wait()
        raise aiohttp.ClientError("worker unavailable during cleanup")

    monkeypatch.setattr(routes, "request_to_worker", rpc)
    monkeypatch.setattr(routes, "_rollback", rollback)
    task = asyncio.create_task(
        routes.reserve(
            7, ModelPortClaim(token=uuid4(), ports=[40020]), api_request, None, None
        )
    )
    try:
        await asyncio.wait_for(posting.wait(), timeout=1)
        task.cancel()
        await asyncio.wait_for(rolling_back.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        finish_cleanup.set()
        await asyncio.wait_for(finished.wait(), timeout=1)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
    assert "Tentative port rollback failed" in caplog.text


@pytest.mark.asyncio
async def test_cleanup_token_is_forwarded_in_a_header(api_request, monkeypatch):
    rpc = AsyncMock(return_value=(SimpleNamespace(status=204), None))
    monkeypatch.setattr(routes, "request_to_worker", rpc)
    token = uuid4()
    await routes._call(api_request, _workers()[0], 7, "DELETE", token=token)
    assert not rpc.call_args.kwargs.get("params")
    assert rpc.call_args.kwargs["headers"]["X-Claim-Token"] == str(token)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["rpc_error", "cancel"])
async def test_rollback_failure_preserves_the_original_reservation_error(
    api_request, monkeypatch, outcome
):
    monkeypatch.setattr(routes, "_members", AsyncMock(return_value=_workers()))
    primary = (
        aiohttp.ClientConnectionError("claim response lost")
        if outcome == "rpc_error"
        else asyncio.CancelledError()
    )
    deleted = []

    async def rpc(**kwargs):
        worker_id = kwargs["worker"].id
        if kwargs["method"] == "DELETE":
            deleted.append(worker_id)
            raise aiohttp.ClientConnectionError("cleanup unavailable")
        if worker_id == 2:
            raise primary
        return SimpleNamespace(status=200), b'{"accepted": true}'

    monkeypatch.setattr(routes, "request_to_worker", rpc)
    error_type = (
        ServiceUnavailableException
        if outcome == "rpc_error"
        else asyncio.CancelledError
    )
    with pytest.raises(error_type) as error:
        await routes.reserve(
            7, ModelPortClaim(token=uuid4(), ports=[40020]), api_request, None, None
        )
    assert sorted(deleted) == [1, 2]
    if outcome == "rpc_error":
        assert "Cannot coordinate model ports with worker 2" in error.value.message
        assert error.value.__cause__ is primary
    else:
        assert error.value is primary


@pytest.mark.asyncio
async def test_rejected_claim_with_failed_cleanup_reports_both_outcomes(
    api_request, monkeypatch
):
    monkeypatch.setattr(routes, "_members", AsyncMock(return_value=_workers()))

    async def rpc(**kwargs):
        if kwargs["method"] == "DELETE":
            raise aiohttp.ClientConnectionError("cleanup unavailable")
        return (
            SimpleNamespace(status=200),
            b'{"accepted": false, "unavailable_ports": [40021]}',
        )

    monkeypatch.setattr(routes, "request_to_worker", rpc)
    with pytest.raises(ServiceUnavailableException) as error:
        await routes.reserve(
            7,
            ModelPortClaim(token=uuid4(), ports=[40020, 40021]),
            api_request,
            None,
            None,
        )
    assert "[1, 2]" in error.value.message
    assert error.value.details["rejection"] == {
        "accepted": False,
        "unavailable_ports": [40021],
    }


@pytest.mark.asyncio
async def test_member_resolution_rejects_an_instance_without_shared_worker_ports(
    monkeypatch,
):
    from gpustack.api.exceptions import BadRequestException

    instance = new_model_instance(7, "model", 1, worker_id=1)
    instance.distributed_servers = None
    monkeypatch.setattr(routes, "assert_resource_visible", MagicMock())
    monkeypatch.setattr(
        routes.ModelInstance, "one_by_id", AsyncMock(return_value=instance)
    )
    worker_lookup = AsyncMock()
    monkeypatch.setattr(routes.Worker, "one_by_id", worker_lookup)
    ctx = SimpleNamespace(user=SimpleNamespace(worker=SimpleNamespace(id=1)))
    with pytest.raises(BadRequestException) as error:
        await routes._members(None, ctx, 7)
    assert error.value.message == "Instance has no shared worker ports"
    worker_lookup.assert_not_awaited()
