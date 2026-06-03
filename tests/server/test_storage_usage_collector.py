"""Tests for ``StorageUsageCollector`` — volume window extraction + lifecycle
dispatch (PV meters from created to deleted, no attach split)."""

from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from gpustack.schemas.resource_events import (
    EVENT_TYPE_ATTACHED,
    EVENT_TYPE_CREATED,
    EVENT_TYPE_DELETED,
    RESOURCE_TYPE_PERSISTENT_VOLUME,
    ResourceEvent,
)
from gpustack.server.storage_usage_collector import (
    StorageUsageCollector,
    _open_volume_from_event,
)


def make_pv_event(
    *,
    occurred_at: datetime,
    event_type: str,
    resource_id: int = 88,
    resource_name: str = "pv-models",
    capacity: str = "200Gi",
    storage_type: str = "ssd",
) -> ResourceEvent:
    return ResourceEvent(
        occurred_at=occurred_at,
        owner_principal_id=1,
        creator_id=7,
        resource_type=RESOURCE_TYPE_PERSISTENT_VOLUME,
        resource_id=resource_id,
        resource_name=resource_name,
        event_type=event_type,
        spec_snapshot={
            "name": resource_name,
            "spec": {"type_": storage_type, "capacity": capacity},
        },
    )


def test_open_volume_extracts_capacity_and_type():
    vol = _open_volume_from_event(
        make_pv_event(
            occurred_at=datetime(2026, 5, 26, 17, 0, 0), event_type=EVENT_TYPE_CREATED
        )
    )
    assert vol is not None
    assert vol.volume_id == 88
    assert vol.storage_type == "ssd"
    assert vol.capacity_mib == 200 * 1024
    assert vol.window_start == datetime(2026, 5, 26, 17, 0, 0)


@pytest.mark.asyncio
async def test_created_opens_deleted_settles():
    c = StorageUsageCollector()
    with patch.object(c, "_settle_locked", new=AsyncMock()) as settle:
        await c._handle_event(
            make_pv_event(
                occurred_at=datetime(2026, 5, 26, 17, 0, 0),
                event_type=EVENT_TYPE_CREATED,
            )
        )
        assert 88 in c._open
        await c._handle_event(
            make_pv_event(
                occurred_at=datetime(2026, 5, 27, 17, 0, 0),
                event_type=EVENT_TYPE_DELETED,
            )
        )
        assert 88 not in c._open
        settle.assert_awaited_once()


@pytest.mark.asyncio
async def test_attached_is_audit_only_no_rollup():
    c = StorageUsageCollector()
    with patch.object(c, "_settle_locked", new=AsyncMock()) as settle:
        await c._handle_event(
            make_pv_event(
                occurred_at=datetime(2026, 5, 26, 17, 0, 0),
                event_type=EVENT_TYPE_CREATED,
            )
        )
        await c._handle_event(
            make_pv_event(
                occurred_at=datetime(2026, 5, 26, 17, 25, 0),
                event_type=EVENT_TYPE_ATTACHED,
            )
        )
        # attach doesn't pause or settle — PV meters regardless of attachment
        assert 88 in c._open
        settle.assert_not_awaited()


def test_zero_capacity_not_tracked():
    # defensive: unparseable capacity → not opened (capacity_mib == 0)
    vol = _open_volume_from_event(
        make_pv_event(
            occurred_at=datetime(2026, 5, 26, 17, 0, 0),
            event_type=EVENT_TYPE_CREATED,
            capacity="",
        )
    )
    assert vol is not None and vol.capacity_mib == 0
