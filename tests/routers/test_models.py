from unittest.mock import AsyncMock, MagicMock

import pytest

from gpustack.api.exceptions import (
    AlreadyExistsException,
    ForbiddenException,
    NotFoundException,
)
from gpustack.api.tenant import TenantContext
from gpustack.routes.models import create_model, update_model
from gpustack.schemas.models import ModelCreate, ModelUpdate, SourceEnum
from gpustack.schemas.principals import PrincipalType, platform_principal_id

DEFAULT_ORG_ID = platform_principal_id()
CUSTOM_ORG_ID = 5
OTHER_ORG_ID = 7
CLUSTER_ID = 101


def _ctx(current_principal_id, is_admin=False, accessible_cluster_ids=None):
    user = MagicMock()
    user.id = 99
    user.is_admin = is_admin
    # Tenant helpers compare user.kind against PrincipalType.SYSTEM; pin it
    # to a non-SYSTEM kind so a bare mock can't drift into the SYSTEM bypass.
    user.kind = PrincipalType.USER
    return TenantContext(
        user=user,
        is_platform_admin=is_admin,
        current_principal_id=current_principal_id,
        org_role=None,
        accessible_cluster_ids=set(accessible_cluster_ids or []),
    )


def _model_create(cluster_id=None):
    return ModelCreate(
        name="m1",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/repo",
        cluster_id=cluster_id,
    )


def _cluster(owner_principal_id, cluster_id=CLUSTER_ID, deleted=False):
    cluster = MagicMock()
    cluster.id = cluster_id
    cluster.owner_principal_id = owner_principal_id
    cluster.deleted_at = object() if deleted else None
    return cluster


@pytest.mark.asyncio
async def test_create_model_rejects_default_org_cluster_for_custom_org(monkeypatch):
    """A custom org cannot deploy onto a visible cluster owned by another
    org (e.g. the Default org's shared cluster) — 403, not 404."""
    monkeypatch.setattr(
        "gpustack.routes.models.Cluster.one_by_id",
        AsyncMock(return_value=_cluster(DEFAULT_ORG_ID)),
    )

    with pytest.raises(ForbiddenException):
        await create_model(
            MagicMock(),
            _ctx(CUSTOM_ORG_ID, accessible_cluster_ids=[CLUSTER_ID]),
            _model_create(cluster_id=CLUSTER_ID),
        )


@pytest.mark.asyncio
async def test_create_model_hides_non_visible_cluster_as_missing(monkeypatch):
    """A cluster the caller can't see is reported as missing (404), not
    forbidden (403), so cross-tenant cluster ids can't be probed."""
    monkeypatch.setattr(
        "gpustack.routes.models.Cluster.one_by_id",
        AsyncMock(return_value=_cluster(OTHER_ORG_ID)),
    )

    with pytest.raises(NotFoundException):
        await create_model(
            MagicMock(),
            _ctx(CUSTOM_ORG_ID),
            _model_create(cluster_id=CLUSTER_ID),
        )


@pytest.mark.asyncio
async def test_create_model_rejects_missing_cluster(monkeypatch):
    monkeypatch.setattr(
        "gpustack.routes.models.Cluster.one_by_id",
        AsyncMock(return_value=None),
    )

    with pytest.raises(NotFoundException):
        await create_model(
            MagicMock(),
            _ctx(CUSTOM_ORG_ID),
            _model_create(cluster_id=CLUSTER_ID),
        )


@pytest.mark.asyncio
async def test_create_model_rejects_deleted_cluster(monkeypatch):
    """A soft-deleted cluster is treated as missing (404)."""
    monkeypatch.setattr(
        "gpustack.routes.models.Cluster.one_by_id",
        AsyncMock(return_value=_cluster(CUSTOM_ORG_ID, deleted=True)),
    )

    with pytest.raises(NotFoundException):
        await create_model(
            MagicMock(),
            _ctx(CUSTOM_ORG_ID),
            _model_create(cluster_id=CLUSTER_ID),
        )


@pytest.mark.asyncio
async def test_create_model_allows_own_org_cluster(monkeypatch):
    """An own-org cluster passes the org-alignment check and proceeds to
    the name-uniqueness check (signalled here by AlreadyExists)."""
    monkeypatch.setattr(
        "gpustack.routes.models.Cluster.one_by_id",
        AsyncMock(return_value=_cluster(CUSTOM_ORG_ID)),
    )
    monkeypatch.setattr(
        "gpustack.routes.models.Model.one_by_fields",
        AsyncMock(return_value=MagicMock()),
    )

    with pytest.raises(AlreadyExistsException):
        await create_model(
            MagicMock(),
            _ctx(CUSTOM_ORG_ID),
            _model_create(cluster_id=CLUSTER_ID),
        )


@pytest.mark.asyncio
async def test_create_model_admin_all_mode_derives_owner_from_cluster(monkeypatch):
    """Admin in "All" mode (no principal context) derives the owning org
    from the chosen cluster; the ownership check then passes even for a
    non-default org's cluster, and the model is stamped with that owner."""
    monkeypatch.setattr(
        "gpustack.routes.models.Cluster.one_by_id",
        AsyncMock(return_value=_cluster(OTHER_ORG_ID)),
    )
    one_by_fields = AsyncMock(return_value=MagicMock())
    monkeypatch.setattr(
        "gpustack.routes.models.Model.one_by_fields",
        one_by_fields,
    )

    with pytest.raises(AlreadyExistsException):
        await create_model(
            MagicMock(),
            _ctx(current_principal_id=None, is_admin=True),
            _model_create(cluster_id=CLUSTER_ID),
        )

    # The uniqueness pre-check runs against the org derived from the
    # cluster, not the platform default.
    assert one_by_fields.await_args.args[1]["owner_principal_id"] == OTHER_ORG_ID


@pytest.mark.asyncio
async def test_create_model_admin_all_mode_rejects_missing_cluster(monkeypatch):
    """Admin "All" mode still rejects a non-existent cluster rather than
    stamping the model with the platform default."""
    monkeypatch.setattr(
        "gpustack.routes.models.Cluster.one_by_id",
        AsyncMock(return_value=None),
    )

    with pytest.raises(NotFoundException):
        await create_model(
            MagicMock(),
            _ctx(current_principal_id=None, is_admin=True),
            _model_create(cluster_id=999),
        )


async def _run_update(monkeypatch, ctx, cluster_return):
    """Drive update_model for an owned model pointed at ``cluster_return``."""
    model = MagicMock()
    model.owner_principal_id = CUSTOM_ORG_ID
    monkeypatch.setattr(
        "gpustack.routes.models.Model.one_by_id",
        AsyncMock(return_value=model),
    )
    monkeypatch.setattr(
        "gpustack.routes.models.assert_resource_visible",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "gpustack.routes.models.Cluster.one_by_id",
        AsyncMock(return_value=cluster_return),
    )
    await update_model(
        MagicMock(),
        ctx,
        1,
        ModelUpdate(
            name="m1",
            source=SourceEnum.HUGGING_FACE,
            huggingface_repo_id="org/repo",
            cluster_id=CLUSTER_ID,
        ),
    )


@pytest.mark.asyncio
async def test_update_model_rejects_cross_org_cluster(monkeypatch):
    """A visible cluster owned by another org is a 403 on update."""
    with pytest.raises(ForbiddenException):
        await _run_update(
            monkeypatch,
            _ctx(CUSTOM_ORG_ID, accessible_cluster_ids=[CLUSTER_ID]),
            _cluster(DEFAULT_ORG_ID),
        )


@pytest.mark.asyncio
async def test_update_model_hides_non_visible_cluster_as_missing(monkeypatch):
    """A non-visible cluster is a 404 on update, not a 403 — no probing."""
    with pytest.raises(NotFoundException):
        await _run_update(monkeypatch, _ctx(CUSTOM_ORG_ID), _cluster(OTHER_ORG_ID))


@pytest.mark.asyncio
async def test_update_model_rejects_missing_cluster(monkeypatch):
    with pytest.raises(NotFoundException):
        await _run_update(monkeypatch, _ctx(CUSTOM_ORG_ID), None)
