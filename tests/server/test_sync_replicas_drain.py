"""Tests for graceful scale-down via DRAINING."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gpustack.schemas.models import ModelInstanceStateEnum
from gpustack.server.controllers import sync_replicas
from tests.utils.model import new_model, new_model_instance


def _patch_service(begin_drain=None, batch_delete=None, create=None):
    service = MagicMock()
    service.begin_drain = begin_drain or AsyncMock(return_value=[])
    service.batch_delete = batch_delete or AsyncMock(return_value=[])
    service.create = create or AsyncMock()
    return service


@pytest.mark.asyncio
async def test_sync_replicas_scale_down_drains_running_not_delete():
    model = new_model(1, "m", replicas=1, huggingface_repo_id="org/m")
    running_a = new_model_instance(
        1, "m-a", 1, worker_id=1, state=ModelInstanceStateEnum.RUNNING
    )
    running_b = new_model_instance(
        2, "m-b", 1, worker_id=1, state=ModelInstanceStateEnum.RUNNING
    )
    instances = [running_a, running_b]

    session = MagicMock()
    begin_drain = AsyncMock(return_value=["m-b"])
    batch_delete = AsyncMock(return_value=[])
    service = _patch_service(begin_drain=begin_drain, batch_delete=batch_delete)

    with (
        patch(
            "gpustack.server.controllers.Model.one_by_id",
            new=AsyncMock(return_value=model),
        ),
        patch(
            "gpustack.server.controllers.ModelInstance.all_by_field",
            new=AsyncMock(return_value=instances),
        ),
        patch(
            "gpustack.server.controllers.find_scale_down_candidates",
            new=AsyncMock(
                return_value=[
                    MagicMock(model_instance=running_b, score=10),
                    MagicMock(model_instance=running_a, score=100),
                ]
            ),
        ),
        patch(
            "gpustack.server.controllers.ModelInstanceService",
            return_value=service,
        ),
    ):
        await sync_replicas(session, model)

    begin_drain.assert_awaited_once()
    drained = begin_drain.await_args.args[0]
    assert drained == [running_b]
    batch_delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_sync_replicas_does_not_drain_already_draining_twice():
    model = new_model(1, "m", replicas=1, huggingface_repo_id="org/m")
    running = new_model_instance(
        1, "m-a", 1, worker_id=1, state=ModelInstanceStateEnum.RUNNING
    )
    draining = new_model_instance(
        2, "m-b", 1, worker_id=1, state=ModelInstanceStateEnum.DRAINING
    )
    instances = [running, draining]

    session = MagicMock()
    begin_drain = AsyncMock(return_value=[])
    batch_delete = AsyncMock(return_value=[])
    create = AsyncMock()
    service = _patch_service(
        begin_drain=begin_drain, batch_delete=batch_delete, create=create
    )

    with (
        patch(
            "gpustack.server.controllers.Model.one_by_id",
            new=AsyncMock(return_value=model),
        ),
        patch(
            "gpustack.server.controllers.ModelInstance.all_by_field",
            new=AsyncMock(return_value=instances),
        ),
        patch(
            "gpustack.server.controllers.ModelInstanceService",
            return_value=service,
        ),
    ):
        await sync_replicas(session, model)

    begin_drain.assert_not_awaited()
    batch_delete.assert_not_awaited()
    create.assert_not_awaited()


@pytest.mark.asyncio
async def test_sync_replicas_no_scale_up_while_draining_keeps_count():
    """Draining rows count toward len(instances); no premature scale-up."""
    model = new_model(1, "m", replicas=1, huggingface_repo_id="org/m")
    draining = new_model_instance(
        1, "m-a", 1, worker_id=1, state=ModelInstanceStateEnum.DRAINING
    )
    instances = [draining]

    session = MagicMock()
    create = AsyncMock()
    service = _patch_service(create=create)

    with (
        patch(
            "gpustack.server.controllers.Model.one_by_id",
            new=AsyncMock(return_value=model),
        ),
        patch(
            "gpustack.server.controllers.ModelInstance.all_by_field",
            new=AsyncMock(return_value=instances),
        ),
        patch(
            "gpustack.server.controllers.ModelInstanceService",
            return_value=service,
        ),
    ):
        await sync_replicas(session, model)

    create.assert_not_awaited()


@pytest.mark.asyncio
async def test_sync_replicas_hard_deletes_non_running_scale_down():
    model = new_model(1, "m", replicas=1, huggingface_repo_id="org/m")
    running = new_model_instance(
        1, "m-a", 1, worker_id=1, state=ModelInstanceStateEnum.RUNNING
    )
    pending = new_model_instance(
        2, "m-b", 1, worker_id=1, state=ModelInstanceStateEnum.PENDING
    )
    instances = [running, pending]

    session = MagicMock()
    begin_drain = AsyncMock(return_value=[])
    batch_delete = AsyncMock(return_value=["m-b"])
    service = _patch_service(begin_drain=begin_drain, batch_delete=batch_delete)

    with (
        patch(
            "gpustack.server.controllers.Model.one_by_id",
            new=AsyncMock(return_value=model),
        ),
        patch(
            "gpustack.server.controllers.ModelInstance.all_by_field",
            new=AsyncMock(return_value=instances),
        ),
        patch(
            "gpustack.server.controllers.find_scale_down_candidates",
            new=AsyncMock(
                return_value=[
                    MagicMock(model_instance=pending, score=0),
                    MagicMock(model_instance=running, score=100),
                ]
            ),
        ),
        patch(
            "gpustack.server.controllers.ModelInstanceService",
            return_value=service,
        ),
    ):
        await sync_replicas(session, model)

    begin_drain.assert_not_awaited()
    batch_delete.assert_awaited_once()
    assert batch_delete.await_args.args[0] == [pending]
