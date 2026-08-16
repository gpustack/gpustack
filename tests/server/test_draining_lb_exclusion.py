"""LB / ready_replicas exclusion of DRAINING instances."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gpustack.schemas.models import ModelInstanceStateEnum
from gpustack.server.controllers import sync_ready_replicas
from gpustack.server.services import ModelInstanceService
from tests.utils.model import new_model_instance


@pytest.mark.asyncio
async def test_sync_ready_replicas_excludes_draining():
    model = MagicMock()
    model.deleted_at = None
    model.id = 1
    model.ready_replicas = 2
    running = new_model_instance(
        1, "m-a", 1, worker_id=1, state=ModelInstanceStateEnum.RUNNING
    )
    draining = new_model_instance(
        2, "m-b", 1, worker_id=1, state=ModelInstanceStateEnum.DRAINING
    )

    session = MagicMock()
    update = AsyncMock()

    with (
        patch(
            "gpustack.server.controllers.ModelInstance.all_by_field",
            new=AsyncMock(return_value=[running, draining]),
        ),
        patch(
            "gpustack.server.controllers.ModelService",
        ) as service_cls,
    ):
        service = MagicMock()
        service.update = update
        service_cls.return_value = service

        updated = await sync_ready_replicas(session, model)

    assert updated is True
    assert model.ready_replicas == 1
    update.assert_awaited_once_with(model)


@pytest.mark.asyncio
async def test_get_running_instances_filter_is_running_only():
    """Document the LB contract: get_running_instances queries state=RUNNING."""
    session = MagicMock()
    service = ModelInstanceService(session)

    with patch(
        "gpustack.server.services.ModelInstance.all_by_fields",
        new=AsyncMock(return_value=[]),
    ) as all_by_fields:
        await service.get_running_instances.__wrapped__(service, 42)

    all_by_fields.assert_awaited_once()
    kwargs = all_by_fields.await_args.kwargs
    assert kwargs["fields"] == {
        "model_id": 42,
        "state": ModelInstanceStateEnum.RUNNING,
    }
