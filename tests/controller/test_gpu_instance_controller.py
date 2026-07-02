"""Unit tests for GPUInstanceController's in-memory event coalescing.

Coalescing now lives in the generic ``WorkQueue`` via a controller-supplied
policy (``GPUInstanceController._coalesce_events``). These exercise that policy
end-to-end through ``_enqueue`` and by inspecting the queue's pending slot.

The reconfirm sweep (#5587) routes best-effort re-confirmation events through
the same slot, and must never displace a real /stop, /start, or /delete that is
already pending. A pending DELETED is terminal, and a pending DELETING must not
be superseded by a later UPDATED (only upgraded by a DELETED).
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
    # ``_enqueue`` now only routes into the queue (no per-iid worker task is
    # spawned until the dispatch loop runs), so no test stubbing is needed.
    return GPUInstanceController(cfg=None)


def _pending(controller: GPUInstanceController, iid: int) -> Event:
    """The bus Event currently occupying the queue slot for ``iid``."""
    return controller._queue._pending[(iid,)].object


def test_reconfirm_does_not_displace_pending_stop():
    controller = _controller()

    # A user /stop is already queued for iid 1.
    controller._enqueue(
        Event(type=EventType.UPDATED, data=_gi(1, GPUInstancePhase.STOPPING))
    )
    # The sweep then enqueues a reconfirm for the same iid.
    controller._enqueue(
        Event(
            type=EventType.UPDATED, data=_gi(1, GPUInstancePhase.READY), reconfirm=True
        )
    )

    pending = _pending(controller, 1)
    assert pending.reconfirm is False
    assert pending.data.status.phase == GPUInstancePhase.STOPPING


def test_reconfirm_does_not_displace_pending_delete():
    controller = _controller()

    controller._enqueue(
        Event(type=EventType.DELETED, data=_gi(2, GPUInstancePhase.READY))
    )
    controller._enqueue(
        Event(
            type=EventType.UPDATED, data=_gi(2, GPUInstancePhase.READY), reconfirm=True
        )
    )

    assert _pending(controller, 2).type == EventType.DELETED


def test_reconfirm_fills_empty_slot():
    controller = _controller()

    controller._enqueue(
        Event(
            type=EventType.UPDATED, data=_gi(3, GPUInstancePhase.READY), reconfirm=True
        )
    )

    assert _pending(controller, 3).reconfirm is True


def test_real_event_displaces_pending_reconfirm():
    controller = _controller()

    controller._enqueue(
        Event(
            type=EventType.UPDATED, data=_gi(4, GPUInstancePhase.READY), reconfirm=True
        )
    )
    # A real /stop arriving after a queued reconfirm must win.
    controller._enqueue(
        Event(type=EventType.UPDATED, data=_gi(4, GPUInstancePhase.STOPPING))
    )

    pending = _pending(controller, 4)
    assert pending.reconfirm is False
    assert pending.data.status.phase == GPUInstancePhase.STOPPING


def test_deleting_not_displaced_by_later_update():
    controller = _controller()

    # An in-flight UPDATED already reading DELETING is terminal-bound.
    controller._enqueue(
        Event(type=EventType.UPDATED, data=_gi(5, GPUInstancePhase.DELETING))
    )
    controller._enqueue(
        Event(type=EventType.UPDATED, data=_gi(5, GPUInstancePhase.READY))
    )

    assert _pending(controller, 5).data.status.phase == GPUInstancePhase.DELETING


def test_deleted_upgrades_pending_deleting():
    controller = _controller()

    controller._enqueue(
        Event(type=EventType.UPDATED, data=_gi(6, GPUInstancePhase.DELETING))
    )
    # DELETED is the one event allowed to supersede a pending DELETING.
    controller._enqueue(
        Event(type=EventType.DELETED, data=_gi(6, GPUInstancePhase.DELETING))
    )

    assert _pending(controller, 6).type == EventType.DELETED
