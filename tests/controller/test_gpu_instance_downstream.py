"""C1: the downstream queue maps worker watch events onto upstream triggers.

``_on_downstream_event`` resolves the CR to its upstream id (mechanism X:
``gpustack.ai/instance-id`` label first, else namespace reverse-resolution) and
enqueues an id-only stub (the object is never carried). A CR going away — a
``DELETED`` or a ``MODIFIED`` with a ``deletionTimestamp`` — becomes an upstream
``DELETED`` so the delete intent gets coalesce priority; everything else is a
``MODIFIED``. Consumption stays phase-keyed, so the type only orders the queue.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gpustack.gpu_instances.controllers import GPUInstanceController
from gpustack.schemas.gpu_instances import KUBERES_INSTANCE_ID_LABEL
from gpustack.schemas.principals import platform_principal_id
from gpustack.server.workqueue import WorkEventType


@pytest.fixture
def controller():
    return GPUInstanceController(SimpleNamespace(get_api_port=lambda: 80))


def _line(
    etype, *, name="gi-1", namespace="gpustack-user-1", labels=None, deleting=False
):
    metadata = {"name": name, "namespace": namespace}
    if labels:
        metadata["labels"] = labels
    if deleting:
        metadata["deletionTimestamp"] = "2026-07-02T00:00:00Z"
    return json.dumps({"type": etype, "object": {"metadata": metadata}})


def _pending(controller, iid):
    return controller._queue._pending[(iid,)]


# --- type mapping ---------------------------------------------------------- #


@pytest.mark.asyncio
async def test_modified_maps_to_modified(controller, monkeypatch):
    monkeypatch.setattr(controller, "_resolve_instance_id", AsyncMock(return_value=1))

    await controller._on_downstream_event(_line("MODIFIED"))

    assert _pending(controller, 1).type == WorkEventType.MODIFIED


@pytest.mark.asyncio
async def test_added_maps_to_modified(controller, monkeypatch):
    monkeypatch.setattr(controller, "_resolve_instance_id", AsyncMock(return_value=1))

    await controller._on_downstream_event(_line("ADDED"))

    assert _pending(controller, 1).type == WorkEventType.MODIFIED


@pytest.mark.asyncio
async def test_deleted_maps_to_deleted(controller, monkeypatch):
    monkeypatch.setattr(controller, "_resolve_instance_id", AsyncMock(return_value=1))

    await controller._on_downstream_event(_line("DELETED"))

    assert _pending(controller, 1).type == WorkEventType.DELETED


@pytest.mark.asyncio
async def test_modified_with_deletion_timestamp_maps_to_deleted(
    controller, monkeypatch
):
    monkeypatch.setattr(controller, "_resolve_instance_id", AsyncMock(return_value=1))

    await controller._on_downstream_event(_line("MODIFIED", deleting=True))

    assert _pending(controller, 1).type == WorkEventType.DELETED


# --- coalesce priority + object handling ----------------------------------- #


@pytest.mark.asyncio
async def test_deletion_timestamp_coalesces_over_pending_modified(
    controller, monkeypatch
):
    monkeypatch.setattr(controller, "_resolve_instance_id", AsyncMock(return_value=1))

    await controller._on_downstream_event(_line("MODIFIED"))
    await controller._on_downstream_event(_line("MODIFIED", deleting=True))

    # DELETED intent wins the slot over the earlier plain MODIFIED.
    assert _pending(controller, 1).type == WorkEventType.DELETED


@pytest.mark.asyncio
async def test_object_is_never_carried(controller, monkeypatch):
    monkeypatch.setattr(controller, "_resolve_instance_id", AsyncMock(return_value=7))

    await controller._on_downstream_event(_line("MODIFIED"))

    stub = _pending(controller, 7).object.data
    assert stub.id == 7
    assert stub.status is None  # id-only stub, not the downstream object


# --- ignored / unresolvable ------------------------------------------------ #


@pytest.mark.asyncio
async def test_malformed_line_is_ignored(controller):
    await controller._on_downstream_event("{not json")
    assert len(controller._queue._pending) == 0


@pytest.mark.asyncio
async def test_bookmark_event_is_ignored(controller, monkeypatch):
    resolve = AsyncMock(return_value=1)
    monkeypatch.setattr(controller, "_resolve_instance_id", resolve)

    await controller._on_downstream_event(_line("BOOKMARK"))

    resolve.assert_not_called()
    assert len(controller._queue._pending) == 0


@pytest.mark.asyncio
async def test_unresolvable_cr_is_not_enqueued(controller, monkeypatch):
    monkeypatch.setattr(
        controller, "_resolve_instance_id", AsyncMock(return_value=None)
    )

    await controller._on_downstream_event(_line("MODIFIED"))

    assert len(controller._queue._pending) == 0


# --- mechanism X resolver (label-first / namespace fallback) --------------- #


@pytest.mark.asyncio
async def test_resolve_by_label_needs_no_db(controller):
    iid = await controller._resolve_instance_id(
        {"metadata": {"labels": {KUBERES_INSTANCE_ID_LABEL: "42"}}}
    )
    assert iid == 42


@pytest.mark.asyncio
async def test_resolve_owner_from_user_namespace(controller):
    # ``user-<id>`` carries the owner principal id directly — no DB lookup.
    owner = await controller._resolve_owner_principal_id(None, "gpustack-user-9")
    assert owner == 9


@pytest.mark.asyncio
async def test_resolve_owner_from_default_namespace_is_platform(controller):
    owner = await controller._resolve_owner_principal_id(None, "gpustack-default")
    assert owner == platform_principal_id()
