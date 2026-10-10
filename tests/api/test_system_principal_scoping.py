"""Unit tests for cluster-scoped SYSTEM principal visibility."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gpustack.api.exceptions import NotFoundException
from gpustack.api.tenant import (
    TenantContext,
    assert_cluster_visible,
    assert_resource_visible,
    cluster_scoped_system,
    cluster_visibility_conditions,
    scoped_cluster_row_visible,
    tenant_list_conditions,
)
from gpustack.routes.inference_backend import _hybrid_backend_conditions
from gpustack.schemas.clusters import Cluster
from gpustack.schemas.models import Model
from gpustack.schemas.principals import PrincipalType


def _system_ctx(cluster_id=None, cluster_owner_id=None) -> TenantContext:
    user = MagicMock()
    user.kind = PrincipalType.SYSTEM
    return TenantContext(
        user=user,
        is_platform_admin=False,
        current_principal_id=None,
        org_role=None,
        scoped_cluster_id=cluster_id,
        scoped_cluster_owner_id=cluster_owner_id,
    )


def _user_ctx(principal_id=7) -> TenantContext:
    user = MagicMock()
    user.kind = PrincipalType.USER
    return TenantContext(
        user=user,
        is_platform_admin=False,
        current_principal_id=principal_id,
        org_role=None,
    )


def test_cluster_scoped_system_detection():
    assert cluster_scoped_system(_system_ctx(cluster_id=3, cluster_owner_id=42))
    # Legacy config.token principal: SYSTEM but no cluster linkage.
    assert not cluster_scoped_system(_system_ctx())
    assert not cluster_scoped_system(_user_ctx())
    assert not cluster_scoped_system(None)


def test_scoped_cluster_row_visible():
    ctx = _system_ctx(cluster_id=3)
    assert scoped_cluster_row_visible(ctx, SimpleNamespace(cluster_id=3))
    assert not scoped_cluster_row_visible(ctx, SimpleNamespace(cluster_id=4))
    # NULL cluster rows stay visible (default-cluster resolution).
    assert scoped_cluster_row_visible(ctx, SimpleNamespace(cluster_id=None))
    # Resources without a cluster_id attribute keep the plain bypass.
    assert scoped_cluster_row_visible(ctx, SimpleNamespace())


def test_tenant_list_conditions_scopes_cluster_bound_system():
    scoped = tenant_list_conditions(_system_ctx(cluster_id=3), Model)
    assert len(scoped) == 1
    rendered = str(scoped[0])
    assert "cluster_id" in rendered

    # Legacy SYSTEM principal keeps the full bypass.
    assert tenant_list_conditions(_system_ctx(), Model) == []


def test_cluster_visibility_conditions_scoped_to_own_cluster():
    scoped = cluster_visibility_conditions(_system_ctx(cluster_id=3), Cluster)
    assert len(scoped) == 1
    rendered = str(scoped[0])
    assert "id" in rendered
    assert cluster_visibility_conditions(_system_ctx(), Cluster) == []


def test_assert_resource_visible_scoped():
    ctx = _system_ctx(cluster_id=3)
    assert_resource_visible(ctx, SimpleNamespace(cluster_id=3))
    assert_resource_visible(ctx, SimpleNamespace(cluster_id=None))
    with pytest.raises(NotFoundException):
        assert_resource_visible(ctx, SimpleNamespace(cluster_id=4))
    # Legacy SYSTEM principal still sees everything.
    assert_resource_visible(_system_ctx(), SimpleNamespace(cluster_id=4))


def test_assert_cluster_visible_scoped():
    ctx = _system_ctx(cluster_id=3)
    assert_cluster_visible(ctx, SimpleNamespace(id=3))
    with pytest.raises(NotFoundException):
        assert_cluster_visible(ctx, SimpleNamespace(id=4))
    assert_cluster_visible(_system_ctx(), SimpleNamespace(id=4))


@pytest.mark.parametrize("mine", [False, True])
def test_cluster_list_and_watch_share_system_scope(mine):
    from gpustack.routes.clusters import (
        _cluster_manageable_conditions,
        _is_cluster_manageable,
        _is_cluster_visible,
    )

    visible = _is_cluster_manageable if mine else _is_cluster_visible
    ctx = _system_ctx(cluster_id=3)
    assert visible(SimpleNamespace(id=3), ctx)
    assert not visible(SimpleNamespace(id=4), ctx)
    assert visible(SimpleNamespace(id=4), _system_ctx())
    conditions = (
        _cluster_manageable_conditions(ctx)
        if mine
        else cluster_visibility_conditions(ctx, Cluster)
    )
    (condition,) = conditions
    assert condition.compare(Cluster.id == 3)


@pytest.mark.asyncio
async def test_gpu_instance_list_scopes_cluster_bound_system(monkeypatch):
    from gpustack.routes import gpu_instances
    from gpustack.schemas.common import Pagination
    from gpustack.schemas.gpu_instances import GPUInstance, GPUInstanceListParams

    query = AsyncMock(
        return_value=SimpleNamespace(
            items=[], pagination=Pagination(page=1, perPage=20, total=0, totalPage=0)
        )
    )
    monkeypatch.setattr(GPUInstance, "paginated_by_query", query)
    monkeypatch.setattr(gpu_instances, "async_session", MagicMock())
    monkeypatch.setattr(gpu_instances, "_types_by_snapshot", AsyncMock(return_value={}))
    ctx = _system_ctx(cluster_id=3)
    await gpu_instances.get_gpu_instances(ctx=ctx, params=GPUInstanceListParams())
    conditions = query.call_args.kwargs["extra_conditions"]
    expected = tenant_list_conditions(ctx, GPUInstance)
    assert len(conditions) == len(expected) == 1
    assert conditions[0].compare(expected[0])


def test_hybrid_backend_conditions_scoped_to_cluster_owner():
    """Cluster-bound service accounts see Platform backend rows plus the
    cluster owner Org's rows only; the legacy token keeps full bypass."""
    scoped = _hybrid_backend_conditions(_system_ctx(cluster_id=3, cluster_owner_id=42))
    assert len(scoped) == 1
    rendered = str(scoped[0])
    assert "owner_principal_id" in rendered

    assert _hybrid_backend_conditions(_system_ctx()) == []
