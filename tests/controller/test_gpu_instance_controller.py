"""Unit tests for GPUInstanceController's in-memory event coalescing.

Coalescing lives in the generic ``WorkQueue`` via a controller-supplied policy
(``GPUInstanceController._coalesce_events``): ``DELETED`` is sticky (terminal
delete intent), and otherwise the latest event wins — the consumer always
re-fetches the row, so a superseded event's stale snapshot never matters.
"""

from gpustack.schemas.gpu_instances import (
    GPUInstance,
    GPUInstancePhase,
    GPUInstanceStatus,
)
from gpustack.server.bus import Event, EventType
from gpustack.gpu_instances.controllers import GPUInstanceController


def _gi(id_: int, phase: str) -> GPUInstance:
    return GPUInstance(
        id=id_,
        name=f"gi-{id_}",
        owner_principal_id=1,
        cluster_id=2,
        spec={"type_": "gpu", "image": "busybox"},
        status=GPUInstanceStatus(phase=phase),
    )


def _controller() -> GPUInstanceController:
    return GPUInstanceController(cfg=None)


def _pending(controller: GPUInstanceController, iid: int) -> Event:
    """The bus Event currently occupying the queue slot for ``iid``."""
    return controller._queue._pending[(iid,)].object


def test_first_event_fills_empty_slot():
    controller = _controller()

    controller._enqueue(
        Event(type=EventType.UPDATED, data=_gi(1, GPUInstancePhase.READY))
    )

    assert _pending(controller, 1).data.status.phase == GPUInstancePhase.READY


def test_latest_event_wins():
    controller = _controller()

    controller._enqueue(
        Event(type=EventType.UPDATED, data=_gi(2, GPUInstancePhase.STOPPING))
    )
    # A newer event for the same iid supersedes the pending one.
    controller._enqueue(
        Event(type=EventType.UPDATED, data=_gi(2, GPUInstancePhase.READY))
    )

    assert _pending(controller, 2).data.status.phase == GPUInstancePhase.READY


def test_deleted_is_sticky():
    controller = _controller()

    controller._enqueue(
        Event(type=EventType.DELETED, data=_gi(3, GPUInstancePhase.READY))
    )
    # A pending DELETED is terminal — a later event does not displace it.
    controller._enqueue(
        Event(type=EventType.UPDATED, data=_gi(3, GPUInstancePhase.READY))
    )

    assert _pending(controller, 3).type == EventType.DELETED


def test_deleted_upgrades_pending_update():
    controller = _controller()

    controller._enqueue(
        Event(type=EventType.UPDATED, data=_gi(4, GPUInstancePhase.DELETING))
    )
    # DELETED supersedes a pending non-terminal event (delete intent wins).
    controller._enqueue(
        Event(type=EventType.DELETED, data=_gi(4, GPUInstancePhase.DELETING))
    )

    assert _pending(controller, 4).type == EventType.DELETED
