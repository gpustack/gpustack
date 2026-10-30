from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gpustack.api.exceptions import BadRequestException, ForbiddenException
from gpustack.api.tenant import TenantContext
from gpustack.routes import benchmarks as route
from gpustack.schemas.benchmark import (
    Benchmark,
    BenchmarkSnapshot,
    BenchmarkStateEnum,
    BenchmarkStateUpdate,
    BenchmarkTargetModeEnum,
)
from gpustack.schemas.principals import PrincipalType
from gpustack.schemas.principals import Principal
from gpustack.api.auth import get_worker_principal, is_server_token_principal


@pytest.fixture
def gateway_report(monkeypatch):
    benchmark = Benchmark(
        id=1,
        name="bm",
        model_instance_name="mi",
        worker_id=12,
        cluster_id=7,
        owner_principal_id=10,
        target_mode=BenchmarkTargetModeEnum.ROUTE,
        state=BenchmarkStateEnum.QUEUED,
        snapshot=BenchmarkSnapshot(),
    )
    save = AsyncMock()
    monkeypatch.setattr(Benchmark, "one_by_id", AsyncMock(return_value=benchmark))
    monkeypatch.setattr(Benchmark, "update", save)
    principal = SimpleNamespace(
        kind=PrincipalType.SYSTEM, worker=SimpleNamespace(id=12)
    )
    ctx = TenantContext(
        user=principal,
        is_platform_admin=False,
        current_principal_id=None,
        org_role=None,
    )
    return SimpleNamespace(benchmark=benchmark, ctx=ctx, save=save)


async def report(fixture, url):
    return await route.update_benchmark_state(
        None, fixture.ctx, 1, BenchmarkStateUpdate(gateway_url=url)
    )


@pytest.mark.asyncio
async def test_assigned_worker_can_freeze_the_resolved_target(gateway_report):
    await report(gateway_report, "https://gateway.example:8443/prefix/")
    update = gateway_report.save.await_args.args[1]
    assert update["snapshot"].gateway_url == "https://gateway.example:8443/prefix"
    assert "gateway_url" not in update


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "principal",
    [
        None,
        SimpleNamespace(kind=PrincipalType.USER, worker=None),
        SimpleNamespace(kind=PrincipalType.SYSTEM, worker=SimpleNamespace(id=13)),
        SimpleNamespace(kind=PrincipalType.SYSTEM, worker=None),
    ],
)
async def test_only_assigned_worker_can_report_a_target(gateway_report, principal):
    gateway_report.ctx.user = principal
    gateway_report.ctx.is_platform_admin = True
    with pytest.raises(ForbiddenException):
        await report(gateway_report, "https://attacker.example")
    gateway_report.save.assert_not_awaited()


@pytest.mark.asyncio
async def test_legacy_basic_principal_keeps_worker_api_access_but_cannot_report_target(
    gateway_report,
):
    principal = Principal(name="system/worker/legacy", kind=PrincipalType.SYSTEM)
    assert is_server_token_principal(principal)
    assert await get_worker_principal(principal) is principal
    gateway_report.ctx.user = principal
    with pytest.raises(ForbiddenException):
        await report(gateway_report, "https://gateway.example")
    gateway_report.save.assert_not_awaited()


@pytest.mark.asyncio
async def test_report_cannot_replace_a_frozen_target(gateway_report):
    gateway_report.benchmark.snapshot.gateway_url = "https://trusted.example"
    with pytest.raises(BadRequestException):
        await report(gateway_report, "https://different.example")
    gateway_report.save.assert_not_awaited()


@pytest.mark.asyncio
async def test_repeated_report_of_same_target_is_idempotent(gateway_report):
    gateway_report.benchmark.snapshot.gateway_url = "https://trusted.example"
    await report(gateway_report, "https://trusted.example/")
    assert (
        gateway_report.save.await_args.args[1]["snapshot"].gateway_url
        == "https://trusted.example"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/target",
        "https://user:secret@gateway.example",
        "https://gateway.example?query=1",
        "https://gateway.example#fragment",
        "http://gateway.example:bad",
        "http://gateway.example:0",
        "http://gateway.example:-1",
        "http://gateway.example:65536",
    ],
)
async def test_report_requires_a_valid_inference_base_url(gateway_report, url):
    with pytest.raises(BadRequestException):
        await report(gateway_report, url)
    gateway_report.save.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("port", [1, 65535])
async def test_report_accepts_valid_tcp_port_boundaries(gateway_report, port):
    url = f"https://gateway.example:{port}"
    await report(gateway_report, url)
    assert gateway_report.save.await_args.args[1]["snapshot"].gateway_url == url


@pytest.mark.asyncio
async def test_completed_run_cannot_acquire_a_new_target(gateway_report):
    gateway_report.benchmark.state = BenchmarkStateEnum.COMPLETED
    with pytest.raises(BadRequestException):
        await report(gateway_report, "https://gateway.example")
    gateway_report.save.assert_not_awaited()


@pytest.mark.asyncio
async def test_instance_run_cannot_report_a_route_target(gateway_report):
    gateway_report.benchmark.target_mode = BenchmarkTargetModeEnum.INSTANCE
    with pytest.raises(BadRequestException):
        await report(gateway_report, "https://gateway.example")
    gateway_report.save.assert_not_awaited()
