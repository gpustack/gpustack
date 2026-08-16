"""Tests for DELETE /model-instances graceful drain."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status

from gpustack.routes import model_instances as routes
from gpustack.schemas.models import ModelInstanceStateEnum


def _ctx():
    return MagicMock()


def _mi(*, state, id=1, name="m-a"):
    mi = MagicMock()
    mi.id = id
    mi.name = name
    mi.state = state
    mi.refresh = AsyncMock()
    return mi


@pytest.mark.asyncio
async def test_delete_running_enters_draining_202():
    mi = _mi(state=ModelInstanceStateEnum.RUNNING)

    async def _refresh(_session):
        mi.state = ModelInstanceStateEnum.DRAINING

    mi.refresh = _refresh
    session = MagicMock()
    begin_drain = AsyncMock()

    with (
        patch(
            "gpustack.routes.model_instances.ModelInstance.one_by_id",
            new=AsyncMock(return_value=mi),
        ),
        patch("gpustack.routes.model_instances.assert_resource_visible"),
        patch("gpustack.routes.model_instances.ModelInstanceService") as service_cls,
        patch(
            "gpustack.routes.model_instances.ModelInstancePublic.model_validate",
            side_effect=lambda obj: MagicMock(
                model_dump=lambda mode="json": {
                    "id": obj.id,
                    "name": obj.name,
                    "state": str(obj.state),
                }
            ),
        ),
    ):
        service = MagicMock()
        service.begin_drain = begin_drain
        service.delete = AsyncMock()
        service_cls.return_value = service

        resp = await routes.delete_model_instance(session, _ctx(), 1)

    assert resp.status_code == status.HTTP_202_ACCEPTED
    begin_drain.assert_awaited_once()
    service.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_already_draining_idempotent_202():
    mi = _mi(state=ModelInstanceStateEnum.DRAINING)
    session = MagicMock()
    begin_drain = AsyncMock()
    delete = AsyncMock()

    with (
        patch(
            "gpustack.routes.model_instances.ModelInstance.one_by_id",
            new=AsyncMock(return_value=mi),
        ),
        patch("gpustack.routes.model_instances.assert_resource_visible"),
        patch("gpustack.routes.model_instances.ModelInstanceService") as service_cls,
        patch(
            "gpustack.routes.model_instances.ModelInstancePublic.model_validate",
            side_effect=lambda obj: MagicMock(
                model_dump=lambda mode="json": {
                    "id": obj.id,
                    "state": "draining",
                }
            ),
        ),
    ):
        service = MagicMock()
        service.begin_drain = begin_drain
        service.delete = delete
        service_cls.return_value = service

        resp = await routes.delete_model_instance(session, _ctx(), 1)

    assert resp.status_code == status.HTTP_202_ACCEPTED
    begin_drain.assert_not_awaited()
    delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_pending_hard_deletes_204():
    mi = _mi(state=ModelInstanceStateEnum.PENDING)
    session = MagicMock()
    begin_drain = AsyncMock()
    delete = AsyncMock()

    with (
        patch(
            "gpustack.routes.model_instances.ModelInstance.one_by_id",
            new=AsyncMock(return_value=mi),
        ),
        patch("gpustack.routes.model_instances.assert_resource_visible"),
        patch("gpustack.routes.model_instances.ModelInstanceService") as service_cls,
    ):
        service = MagicMock()
        service.begin_drain = begin_drain
        service.delete = delete
        service_cls.return_value = service

        resp = await routes.delete_model_instance(session, _ctx(), 1)

    assert resp.status_code == status.HTTP_204_NO_CONTENT
    begin_drain.assert_not_awaited()
    delete.assert_awaited_once_with(mi)
