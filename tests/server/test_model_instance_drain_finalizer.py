"""Tests for ModelInstanceDrainFinalizer."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gpustack.schemas.models import ModelInstanceStateEnum
from gpustack.server.model_instance_drain_finalizer import (
    ModelInstanceDrainFinalizer,
    _drain_timed_out,
)
from tests.utils.mock import mock_async_session
from tests.utils.model import new_model_instance


def test_drain_timed_out_false_within_window():
    mi = new_model_instance(1, "m-a", 1, state=ModelInstanceStateEnum.DRAINING)
    mi.drain_started_at = datetime.now(timezone.utc) - timedelta(seconds=10)
    now = datetime.now(timezone.utc)
    assert _drain_timed_out(mi, now, timeout_seconds=120) is False


def test_drain_timed_out_true_after_window():
    mi = new_model_instance(1, "m-a", 1, state=ModelInstanceStateEnum.DRAINING)
    mi.drain_started_at = datetime.now(timezone.utc) - timedelta(seconds=200)
    now = datetime.now(timezone.utc)
    assert _drain_timed_out(mi, now, timeout_seconds=120) is True


def test_drain_timed_out_missing_started_at():
    mi = new_model_instance(1, "m-a", 1, state=ModelInstanceStateEnum.DRAINING)
    mi.drain_started_at = None
    now = datetime.now(timezone.utc)
    assert _drain_timed_out(mi, now, timeout_seconds=120) is True


@pytest.mark.asyncio
async def test_reconcile_deletes_when_idle():
    mi = new_model_instance(1, "m-a", 1, state=ModelInstanceStateEnum.DRAINING)
    mi.drain_idle = True
    mi.drain_started_at = datetime.now(timezone.utc)

    session_cm = mock_async_session()
    batch_delete = AsyncMock(return_value=["m-a"])
    delete = AsyncMock()

    with (
        patch(
            "gpustack.server.model_instance_drain_finalizer.async_session",
            return_value=session_cm,
        ),
        patch(
            "gpustack.server.model_instance_drain_finalizer.ModelInstance.all_by_field",
            new=AsyncMock(return_value=[mi]),
        ),
        patch(
            "gpustack.server.model_instance_drain_finalizer.ModelInstanceService"
        ) as service_cls,
    ):
        service = MagicMock()
        service.batch_delete = batch_delete
        service.delete = delete
        service_cls.return_value = service

        finalizer = ModelInstanceDrainFinalizer(timeout_seconds=120)
        deleted = await finalizer.reconcile()

    assert deleted == ["m-a"]
    batch_delete.assert_awaited_once_with([mi])
    delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_reconcile_keeps_busy_until_timeout():
    mi = new_model_instance(1, "m-a", 1, state=ModelInstanceStateEnum.DRAINING)
    mi.drain_idle = False
    mi.drain_started_at = datetime.now(timezone.utc) - timedelta(seconds=5)

    session_cm = mock_async_session()
    batch_delete = AsyncMock()
    delete = AsyncMock()

    with (
        patch(
            "gpustack.server.model_instance_drain_finalizer.async_session",
            return_value=session_cm,
        ),
        patch(
            "gpustack.server.model_instance_drain_finalizer.ModelInstance.all_by_field",
            new=AsyncMock(return_value=[mi]),
        ),
        patch(
            "gpustack.server.model_instance_drain_finalizer.ModelInstanceService"
        ) as service_cls,
    ):
        service = MagicMock()
        service.batch_delete = batch_delete
        service.delete = delete
        service_cls.return_value = service

        finalizer = ModelInstanceDrainFinalizer(timeout_seconds=120)
        deleted = await finalizer.reconcile()

    assert deleted == []
    batch_delete.assert_not_awaited()
    delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_reconcile_force_deletes_on_timeout():
    mi = new_model_instance(1, "m-a", 1, state=ModelInstanceStateEnum.DRAINING)
    mi.drain_idle = False
    mi.drain_started_at = datetime.now(timezone.utc) - timedelta(seconds=200)

    session_cm = mock_async_session()
    batch_delete = AsyncMock(return_value=["m-a"])
    delete = AsyncMock()

    with (
        patch(
            "gpustack.server.model_instance_drain_finalizer.async_session",
            return_value=session_cm,
        ),
        patch(
            "gpustack.server.model_instance_drain_finalizer.ModelInstance.all_by_field",
            new=AsyncMock(return_value=[mi]),
        ),
        patch(
            "gpustack.server.model_instance_drain_finalizer.ModelInstanceService"
        ) as service_cls,
    ):
        service = MagicMock()
        service.batch_delete = batch_delete
        service.delete = delete
        service_cls.return_value = service

        finalizer = ModelInstanceDrainFinalizer(timeout_seconds=120)
        deleted = await finalizer.reconcile()

    assert deleted == ["m-a"]
    batch_delete.assert_awaited_once_with([mi])
    delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_reconcile_batch_deletes_multiple_instances():
    mi_a = new_model_instance(1, "m-a", 1, state=ModelInstanceStateEnum.DRAINING)
    mi_a.drain_idle = True
    mi_a.drain_started_at = datetime.now(timezone.utc)

    mi_b = new_model_instance(2, "m-b", 2, state=ModelInstanceStateEnum.DRAINING)
    mi_b.drain_idle = False
    mi_b.drain_started_at = datetime.now(timezone.utc) - timedelta(seconds=200)

    session_cm = mock_async_session()
    batch_delete = AsyncMock(return_value=["m-a", "m-b"])
    delete = AsyncMock()

    with (
        patch(
            "gpustack.server.model_instance_drain_finalizer.async_session",
            return_value=session_cm,
        ),
        patch(
            "gpustack.server.model_instance_drain_finalizer.ModelInstance.all_by_field",
            new=AsyncMock(return_value=[mi_a, mi_b]),
        ),
        patch(
            "gpustack.server.model_instance_drain_finalizer.ModelInstanceService"
        ) as service_cls,
    ):
        service = MagicMock()
        service.batch_delete = batch_delete
        service.delete = delete
        service_cls.return_value = service

        finalizer = ModelInstanceDrainFinalizer(timeout_seconds=120)
        deleted = await finalizer.reconcile()

    assert deleted == ["m-a", "m-b"]
    batch_delete.assert_awaited_once_with([mi_a, mi_b])
    delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_reconcile_no_delete_when_none_ready():
    mi = new_model_instance(1, "m-a", 1, state=ModelInstanceStateEnum.DRAINING)
    mi.drain_idle = False
    mi.drain_started_at = datetime.now(timezone.utc) - timedelta(seconds=5)

    session_cm = mock_async_session()
    batch_delete = AsyncMock()
    delete = AsyncMock()

    with (
        patch(
            "gpustack.server.model_instance_drain_finalizer.async_session",
            return_value=session_cm,
        ),
        patch(
            "gpustack.server.model_instance_drain_finalizer.ModelInstance.all_by_field",
            new=AsyncMock(return_value=[mi]),
        ),
        patch(
            "gpustack.server.model_instance_drain_finalizer.ModelInstanceService"
        ) as service_cls,
    ):
        service = MagicMock()
        service.batch_delete = batch_delete
        service.delete = delete
        service_cls.return_value = service

        finalizer = ModelInstanceDrainFinalizer(timeout_seconds=120)
        deleted = await finalizer.reconcile()

    assert deleted == []
    batch_delete.assert_not_awaited()
    delete.assert_not_awaited()
