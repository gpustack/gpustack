"""V4: what does and does not stop a persistent volume's meter.

A PV has an independent lifecycle, so its meter is driven exclusively by the
``created`` / ``deleted`` resource events the logger writes — never by whether
an instance happens to hold it. Two consequences are easy to get backwards and
are pinned here:

* **Swapping an instance's volume keeps billing the old one.** The swap only
  re-points ``GPUInstance.persistent_volume_id``; the PV row is untouched, so no
  ``deleted`` event is written and the window stays open. Correct — the volume
  still exists and still holds the user's data — but it means the old volume
  must be deleted explicitly.
* **Asking to delete a PV does NOT stop billing.** ``DELETE`` only stamps
  ``status.phase = Deleting`` (an ``UPDATED`` bus event, which the logger
  deliberately does not record); billing continues until the finalizer clears
  the downstream CRs and hard-deletes the row, which is the only path that
  publishes ``EventType.DELETED``. A blocking holder therefore extends the
  billed period — see the note in §6.5 of the metering design doc.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gpustack.schemas.resource_events import (
    EVENT_TYPE_CREATED,
    EVENT_TYPE_DELETED,
)
from gpustack.server.bus import EventType
from gpustack.server.resource_event_logger import ResourceEventLogger


def _volume_bus_event(event_type, *, rid=88, phase=None):
    """A bus event carrying a PV row, as the logger's subscription sees it."""
    volume = SimpleNamespace(
        id=rid,
        name="pv-models",
        owner_principal_id=1,
        creator_id=7,
        status={"phase": phase} if phase else None,
        spec={"type_": "ssd", "capacity": "200Gi"},
    )
    return SimpleNamespace(type=event_type, data=volume, id=rid, changed_fields={})


async def _drive(logger, events):
    """Feed bus events to the volume handler; return the resource-event types
    it decided to write."""
    with patch.object(logger, "_write_event", new=AsyncMock()) as w:
        for e in events:
            await logger._handle_volume(e)
        return [call.kwargs["event_type"] for call in w.await_args_list]


@pytest.mark.asyncio
async def test_volume_swap_and_detach_do_not_stop_the_meter():
    """A swap/attach/detach reaches the logger as ``UPDATED`` on the PV row.

    Nothing is recorded for it, so the collector never sees a ``deleted`` event
    and the window stays open — the old volume keeps being billed after an
    instance is re-pointed at a different one.
    """
    logger = ResourceEventLogger()
    written = await _drive(
        logger,
        [
            _volume_bus_event(EventType.CREATED),
            # instance re-pointed at another volume: the PV row itself only ever
            # sees an update (and in practice not even that)
            _volume_bus_event(EventType.UPDATED),
            _volume_bus_event(EventType.UPDATED),
        ],
    )
    assert written == [EVENT_TYPE_CREATED]


@pytest.mark.asyncio
async def test_delete_request_keeps_billing_until_the_row_is_gone():
    """``phase = Deleting`` does not close the window; the hard delete does.

    This is the opposite of "asking to delete stops the meter": while the
    finalizer waits on a holder (a Stopped instance blocks reclaim just as a
    running one does), the volume stays billable.
    """
    logger = ResourceEventLogger()
    written = await _drive(
        logger,
        [
            _volume_bus_event(EventType.CREATED),
            # DELETE /gpu-instance-persistent-volumes/{id} — a phase stamp
            _volume_bus_event(EventType.UPDATED, phase="Deleting"),
            # ... finalizer still blocked, more reconcile writes
            _volume_bus_event(EventType.UPDATED, phase="Deleting"),
        ],
    )
    assert written == [EVENT_TYPE_CREATED], "delete request must not settle the window"

    # Only the finalizer's hard delete publishes DELETED, and that is what the
    # collector turns into "stop metering".
    written = await _drive(logger, [_volume_bus_event(EventType.DELETED)])
    assert written == [EVENT_TYPE_DELETED]


@pytest.mark.asyncio
async def test_created_is_recorded_once_across_repeated_updates():
    """Guard the dedup that makes the two tests above meaningful: a storm of
    updates on a volume the logger has not seen created yet must still yield a
    single ``created``, not one per event."""
    logger = ResourceEventLogger()
    written = await _drive(
        logger,
        [
            _volume_bus_event(EventType.UPDATED),
            _volume_bus_event(EventType.UPDATED),
            _volume_bus_event(EventType.CREATED),
        ],
    )
    assert written == [EVENT_TYPE_CREATED]
