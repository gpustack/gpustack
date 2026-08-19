"""Top-level cache-service-instance route checks.

Instances carry no owner_principal_id: list/watch visibility derives from
the parent service (Org callers) or the caller's cluster (cluster-bound
system accounts), and writes are reserved for system principals (worker
state write-back).
"""

from contextlib import asynccontextmanager
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gpustack.api.exceptions import ForbiddenException, NotFoundException
from gpustack.api.tenant import TenantContext
from gpustack.routes import cache_service_instances as instances_route
from gpustack.schemas.cache_services import (
    CacheServiceInstancePublic,
    CacheServiceInstanceUpdate,
    CacheServiceStateEnum,
)
from gpustack.schemas.principals import PrincipalType

ORG_PRINCIPAL = 42


def _user_ctx(principal_id: int = ORG_PRINCIPAL) -> TenantContext:
    user = MagicMock()
    user.kind = PrincipalType.USER
    return TenantContext(
        user=user,
        is_platform_admin=False,
        current_principal_id=principal_id,
        org_role=None,
    )


def _system_ctx(scoped_cluster_id=None) -> TenantContext:
    user = MagicMock()
    user.kind = PrincipalType.SYSTEM
    return TenantContext(
        user=user,
        is_platform_admin=False,
        current_principal_id=None,
        org_role=None,
        scoped_cluster_id=scoped_cluster_id,
    )


def _params(watch=False) -> SimpleNamespace:
    return SimpleNamespace(page=1, perPage=100, watch=watch)


def _patch_session(monkeypatch):
    @asynccontextmanager
    async def fake_session():
        yield MagicMock()

    monkeypatch.setattr(instances_route, "async_session", fake_session)


def _instance_row(**overrides):
    fields = dict(
        id=21,
        name="svc-a1b2c",
        cache_service_id=9,
        worker_id=5,
        cluster_id=3,
        state=CacheServiceStateEnum.RUNNING,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


# ---- list scoping ----


@pytest.mark.asyncio
async def test_list_scopes_org_users_to_their_services(monkeypatch):
    _patch_session(monkeypatch)
    paginated = AsyncMock(return_value="page")
    monkeypatch.setattr(
        instances_route.CacheServiceInstance, "paginated_by_query", paginated
    )

    result = await instances_route.get_cache_service_instances(
        ctx=_user_ctx(), params=_params(), cache_service_id=9
    )

    assert result == "page"
    call_kwargs = paginated.await_args.kwargs
    assert call_kwargs["fields"] == {"cache_service_id": 9}
    # One derived condition: cache_service_id IN (caller's services).
    conditions = call_kwargs["extra_conditions"]
    assert len(conditions) == 1
    assert "cache_service_id IN" in str(conditions[0])
    assert "owner_principal_id" in str(conditions[0])


@pytest.mark.asyncio
async def test_list_scopes_cluster_system_to_its_cluster(monkeypatch):
    _patch_session(monkeypatch)
    paginated = AsyncMock(return_value="page")
    monkeypatch.setattr(
        instances_route.CacheServiceInstance, "paginated_by_query", paginated
    )

    await instances_route.get_cache_service_instances(
        ctx=_system_ctx(scoped_cluster_id=3), params=_params()
    )

    conditions = paginated.await_args.kwargs["extra_conditions"]
    assert len(conditions) == 1
    assert "cluster_id" in str(conditions[0])


@pytest.mark.asyncio
async def test_list_unscoped_for_platform_system(monkeypatch):
    """The legacy platform-level system principal keeps the full bypass."""
    _patch_session(monkeypatch)
    paginated = AsyncMock(return_value="page")
    monkeypatch.setattr(
        instances_route.CacheServiceInstance, "paginated_by_query", paginated
    )

    await instances_route.get_cache_service_instances(
        ctx=_system_ctx(), params=_params(), worker_id=5
    )

    call_kwargs = paginated.await_args.kwargs
    assert call_kwargs["fields"] == {"worker_id": 5}
    assert call_kwargs["extra_conditions"] == []


# ---- watch scoping ----


def _patch_streaming(monkeypatch):
    captured = {}

    def fake_streaming(fields=None, filter_func=None, **kwargs):
        captured["fields"] = fields
        captured["filter_func"] = filter_func

        async def _empty():
            if False:
                yield ""

        return _empty()

    monkeypatch.setattr(
        instances_route.CacheServiceInstance, "streaming", fake_streaming
    )
    return captured


@pytest.mark.asyncio
async def test_watch_filters_cluster_system_to_its_cluster(monkeypatch):
    captured = _patch_streaming(monkeypatch)

    await instances_route.get_cache_service_instances(
        ctx=_system_ctx(scoped_cluster_id=3), params=_params(watch=True)
    )

    filter_func = captured["filter_func"]
    assert filter_func(_instance_row(cluster_id=3)) is True
    assert filter_func(_instance_row(cluster_id=4)) is False


@pytest.mark.asyncio
async def test_watch_filters_org_users_to_their_services(monkeypatch):
    captured = _patch_streaming(monkeypatch)
    _patch_session(monkeypatch)
    monkeypatch.setattr(
        instances_route.CacheService,
        "all_by_fields",
        AsyncMock(return_value=[SimpleNamespace(id=9)]),
    )

    await instances_route.get_cache_service_instances(
        ctx=_user_ctx(), params=_params(watch=True)
    )

    filter_func = captured["filter_func"]
    assert filter_func(_instance_row(cache_service_id=9))
    assert not filter_func(_instance_row(cache_service_id=8))


@pytest.mark.asyncio
async def test_watch_unfiltered_for_platform_system(monkeypatch):
    captured = _patch_streaming(monkeypatch)

    await instances_route.get_cache_service_instances(
        ctx=_system_ctx(), params=_params(watch=True)
    )

    assert captured["filter_func"] is None


# ---- get by id ----


@pytest.mark.asyncio
async def test_get_by_id_serves_system_callers(monkeypatch):
    """The worker's state write-back reads the instance back through this
    endpoint whenever its watch cache is cold."""
    instance = _instance_row()
    monkeypatch.setattr(
        instances_route.CacheServiceInstance,
        "one_by_id",
        AsyncMock(return_value=instance),
    )

    result = await instances_route.get_cache_service_instance(
        session=MagicMock(), ctx=_system_ctx(scoped_cluster_id=3), id=21
    )

    assert result is instance


@pytest.mark.asyncio
async def test_get_by_id_hides_other_clusters_from_cluster_system(monkeypatch):
    monkeypatch.setattr(
        instances_route.CacheServiceInstance,
        "one_by_id",
        AsyncMock(return_value=_instance_row(cluster_id=4)),
    )

    with pytest.raises(NotFoundException):
        await instances_route.get_cache_service_instance(
            session=MagicMock(), ctx=_system_ctx(scoped_cluster_id=3), id=21
        )


@pytest.mark.asyncio
async def test_get_by_id_scopes_org_users_to_their_services(monkeypatch):
    monkeypatch.setattr(
        instances_route.CacheServiceInstance,
        "one_by_id",
        AsyncMock(return_value=_instance_row()),
    )
    monkeypatch.setattr(
        instances_route.CacheService,
        "one_by_id",
        AsyncMock(
            return_value=SimpleNamespace(
                id=9, owner_principal_id=ORG_PRINCIPAL + 1, deleted_at=None
            )
        ),
    )

    with pytest.raises(NotFoundException):
        await instances_route.get_cache_service_instance(
            session=MagicMock(), ctx=_user_ctx(), id=21
        )


@pytest.mark.asyncio
async def test_get_by_id_missing_instance_is_not_found(monkeypatch):
    monkeypatch.setattr(
        instances_route.CacheServiceInstance,
        "one_by_id",
        AsyncMock(return_value=None),
    )

    with pytest.raises(NotFoundException):
        await instances_route.get_cache_service_instance(
            session=MagicMock(), ctx=_system_ctx(), id=21
        )


# ---- public serialization ----


def test_public_serialization_includes_name():
    """The API-facing shape carries the instance's display name."""
    public = CacheServiceInstancePublic(
        id=21,
        name="svc-a1b2c",
        cache_service_id=9,
        worker_id=5,
        cluster_id=3,
        state=CacheServiceStateEnum.RUNNING,
        created_at=datetime(2026, 7, 21),
        updated_at=datetime(2026, 7, 21),
    )

    assert public.model_dump()["name"] == "svc-a1b2c"


# ---- state write-back ----


def _update_in(**overrides) -> CacheServiceInstanceUpdate:
    fields = dict(
        name="svc-a1b2c",
        cache_service_id=9,
        worker_id=5,
        cluster_id=3,
        state=CacheServiceStateEnum.RUNNING,
    )
    fields.update(overrides)
    return CacheServiceInstanceUpdate(**fields)


@pytest.mark.asyncio
async def test_update_allows_system_writeback(monkeypatch):
    instance = _instance_row(update=AsyncMock())
    monkeypatch.setattr(
        instances_route.CacheServiceInstance,
        "one_by_id",
        AsyncMock(return_value=instance),
    )

    result = await instances_route.update_cache_service_instance(
        session=MagicMock(),
        ctx=_system_ctx(),
        id=21,
        instance_in=_update_in(port=40001, healthy=True),
    )

    instance.update.assert_awaited_once()
    assert result is instance


@pytest.mark.asyncio
async def test_update_rejects_non_system_callers(monkeypatch):
    instance = _instance_row(update=AsyncMock())
    monkeypatch.setattr(
        instances_route.CacheServiceInstance,
        "one_by_id",
        AsyncMock(return_value=instance),
    )

    with pytest.raises(ForbiddenException):
        await instances_route.update_cache_service_instance(
            session=MagicMock(),
            ctx=_user_ctx(),
            id=21,
            instance_in=_update_in(),
        )
    instance.update.assert_not_called()


@pytest.mark.asyncio
async def test_update_missing_instance_is_not_found(monkeypatch):
    monkeypatch.setattr(
        instances_route.CacheServiceInstance,
        "one_by_id",
        AsyncMock(return_value=None),
    )

    with pytest.raises(NotFoundException):
        await instances_route.update_cache_service_instance(
            session=MagicMock(),
            ctx=_system_ctx(),
            id=21,
            instance_in=_update_in(),
        )
