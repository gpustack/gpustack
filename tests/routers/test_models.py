from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gpustack.api.exceptions import (
    AlreadyExistsException,
    BadRequestException,
    ForbiddenException,
    NotFoundException,
)
from gpustack.api.tenant import TenantContext
from gpustack.routes import models as models_route
from gpustack.routes import model_instances as model_instances_route
from gpustack.routes.models import create_model, update_model, validate_model_in
from gpustack.routes.model_instances import delete_model_instance
from gpustack.routes.model_common import ModelStateFilterEnum
from gpustack.schemas.models import (
    BackendEnum,
    Model,
    ModelCreate,
    ModelInstance,
    ModelUpdate,
    SourceEnum,
)
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


@pytest.mark.parametrize(
    "ready, replicas, state, expected",
    [
        (2, 3, ModelStateFilterEnum.READY, True),
        (0, 3, ModelStateFilterEnum.READY, False),
        (0, 3, ModelStateFilterEnum.NOT_READY, True),
        (2, 3, ModelStateFilterEnum.NOT_READY, False),
        (0, 0, ModelStateFilterEnum.STOPPED, True),
        (0, 3, ModelStateFilterEnum.STOPPED, False),
        (0, 3, None, True),
    ],
)
def test_model_watch_filter_applies_state(
    monkeypatch, ready, replicas, state, expected
):
    """The /models watch stream honors ``state`` via replica counts."""
    monkeypatch.setattr(models_route, "cluster_scoped_system", lambda ctx: False)

    visible = models_route._make_model_watch_filter(
        ctx=None, categories=None, state=state
    )
    data = SimpleNamespace(ready_replicas=ready, replicas=replicas)
    assert visible(data) is expected


def test_model_watch_filter_passes_id_only_delete_events(monkeypatch):
    """ID-only DELETED payloads lack replica counts and must not be dropped
    by the state filter, else watch clients hold stale rows."""
    monkeypatch.setattr(models_route, "cluster_scoped_system", lambda ctx: False)

    visible = models_route._make_model_watch_filter(
        ctx=None, categories=None, state=ModelStateFilterEnum.READY
    )
    assert visible({"id": 7}) is True


def _dp_model(cls=ModelCreate, **overrides):
    base = dict(
        name="m",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="a/b",
        backend=BackendEnum.VLLM.value,
        distributed_inference_across_workers=True,
    )
    base.update(overrides)
    return cls(**base)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "backend_parameters, replicas_in, expected",
    [
        # external-LB: replicas derived from --data-parallel-size (dpl == 1).
        (["--data-parallel-external-lb", "--data-parallel-size=4"], None, 4),
        # hybrid-LB: replicas == dp // dpl.
        (
            [
                "--data-parallel-hybrid-lb",
                "--data-parallel-size=8",
                "--data-parallel-size-local=2",
            ],
            None,
            4,
        ),
        # An explicit replicas that matches the derived value is accepted.
        (["--data-parallel-external-lb", "--data-parallel-size=4"], 4, 4),
    ],
)
async def test_dp_replicas_derived(backend_parameters, replicas_in, expected):
    kwargs = {"backend_parameters": backend_parameters}
    if replicas_in is not None:
        kwargs["replicas"] = replicas_in
    model_in = _dp_model(**kwargs)
    # Derivation runs before any session use (gpu_selector is None), so a real
    # DB session is unnecessary here.
    await validate_model_in(None, model_in)
    assert model_in.replicas == expected


@pytest.mark.asyncio
async def test_dp_replicas_conflict_rejected():
    model_in = _dp_model(
        replicas=3,
        backend_parameters=["--data-parallel-external-lb", "--data-parallel-size=4"],
    )
    with pytest.raises(BadRequestException):
        await validate_model_in(None, model_in)


@pytest.mark.asyncio
async def test_dp_replicas_zero_stops_without_override():
    # replicas == 0 stops the model; it must not be derived-over or rejected.
    model_in = _dp_model(
        replicas=0,
        backend_parameters=["--data-parallel-external-lb", "--data-parallel-size=4"],
    )
    await validate_model_in(None, model_in)
    assert model_in.replicas == 0


@pytest.mark.asyncio
async def test_dp_replicas_update_persists():
    # On update the derived replicas must join model_fields_set so
    # ActiveRecord.update writes it back to the DB.
    model_in = _dp_model(
        cls=ModelUpdate,
        backend_parameters=["--data-parallel-external-lb", "--data-parallel-size=6"],
    )
    await validate_model_in(None, model_in)
    assert model_in.replicas == 6
    assert "replicas" in model_in.model_fields_set


def _stub_instance_service(monkeypatch):
    service = MagicMock()
    service.delete = AsyncMock()
    service.batch_delete = AsyncMock(return_value=[])
    monkeypatch.setattr(
        model_instances_route, "ModelInstanceService", lambda session: service
    )
    monkeypatch.setattr(
        model_instances_route, "assert_resource_visible", lambda *a, **k: None
    )
    return service


@pytest.mark.asyncio
async def test_delete_dp_member_deletes_whole_group(monkeypatch):
    # Deleting one DP member (dp_rank > 0) tears down the whole group; a lone
    # replacement can never schedule, so the group is rebuilt intact instead.
    target = ModelInstance(id=19, model_id=1, model_name="m", dp_rank=2)
    siblings = [
        ModelInstance(id=i, model_id=1, model_name="m", dp_rank=i) for i in range(4)
    ]
    dp_model = _dp_model(
        cls=Model,
        id=1,
        backend_parameters=["--data-parallel-external-lb", "--data-parallel-size=4"],
    )
    monkeypatch.setattr(ModelInstance, "one_by_id", AsyncMock(return_value=target))
    monkeypatch.setattr(Model, "one_by_id", AsyncMock(return_value=dp_model))
    monkeypatch.setattr(ModelInstance, "all_by_field", AsyncMock(return_value=siblings))
    service = _stub_instance_service(monkeypatch)

    await delete_model_instance(session=MagicMock(), ctx=MagicMock(), id=19)

    service.batch_delete.assert_awaited_once_with(siblings)
    service.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_plain_instance_deletes_only_target(monkeypatch):
    target = ModelInstance(id=19, model_id=1, model_name="m")
    plain_model = _dp_model(cls=Model, id=1, distributed_inference_across_workers=False)
    monkeypatch.setattr(ModelInstance, "one_by_id", AsyncMock(return_value=target))
    monkeypatch.setattr(Model, "one_by_id", AsyncMock(return_value=plain_model))
    service = _stub_instance_service(monkeypatch)

    await delete_model_instance(session=MagicMock(), ctx=MagicMock(), id=19)

    service.delete.assert_awaited_once_with(target)
    service.batch_delete.assert_not_awaited()
