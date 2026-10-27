"""Spanning previews report role widths only for visible clusters."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest


def _worker(gpus: int):
    return SimpleNamespace(
        id=1,
        status=SimpleNamespace(
            gpu_devices=[SimpleNamespace(index=i) for i in range(gpus)]
        ),
    )


def _ctx(is_admin=True, accessible_cluster_ids=None):
    from unittest.mock import MagicMock

    from gpustack.api.tenant import TenantContext
    from gpustack.schemas.principals import PrincipalType

    user = MagicMock()
    user.id = 99
    user.is_admin = is_admin
    user.kind = PrincipalType.USER
    return TenantContext(
        user=user,
        is_platform_admin=is_admin,
        current_principal_id=None if is_admin else 7,
        org_role=None,
        accessible_cluster_ids=set(accessible_cluster_ids or []),
    )


def _create(**kwargs):
    from gpustack.schemas.models import ModelCreate, SourceEnum

    return ModelCreate(
        name="m1",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/repo",
        **kwargs,
    )


@pytest.mark.asyncio
async def test_the_route_answers_with_both_numbers():
    """Both are needed by the one sentence the form writes, so both cross the
    wire — the form must not have to hold half the comparison."""
    from gpustack.routes.models import preview_spanning_roles
    from gpustack.schemas.models import RoleSpec

    model_in = _create(
        cluster_id=1,
        backend="vLLM",
        backend_parameters=["--tensor-parallel-size=16"],
        roles=[RoleSpec(name="prefill", replicas=1)],
    )
    with (
        patch(
            "gpustack.schemas.clusters.Cluster.one_by_id",
            new=AsyncMock(return_value=SimpleNamespace(id=1, owner_principal_id=1)),
        ),
        patch(
            "gpustack.schemas.workers.Worker.all_by_field",
            new=AsyncMock(return_value=[_worker(8)]),
        ),
    ):
        answer = await preview_spanning_roles(None, _ctx(), model_in)

    assert [(r.name, r.gpus) for r in answer.roles] == [("prefill", 16)]
    assert answer.widest_worker_gpus == 8


@pytest.mark.asyncio
async def test_a_cluster_the_caller_cannot_see_is_not_answered_about():
    """The answer describes someone else's machines. It is a small fact — the
    widest card count in a fleet — but it is still a fact about a cluster the
    caller was not given."""
    from gpustack.api.exceptions import NotFoundException
    from gpustack.routes.models import preview_spanning_roles
    from gpustack.schemas.models import RoleSpec

    model_in = _create(
        cluster_id=1,
        backend="vLLM",
        backend_parameters=["--tensor-parallel-size=16"],
        roles=[RoleSpec(name="prefill", replicas=1)],
    )
    with patch(
        "gpustack.schemas.clusters.Cluster.one_by_id",
        new=AsyncMock(return_value=SimpleNamespace(id=1, owner_principal_id=1)),
    ):
        with pytest.raises(NotFoundException):
            await preview_spanning_roles(None, _ctx(is_admin=False), model_in)
