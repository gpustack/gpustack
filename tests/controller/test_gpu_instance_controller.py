"""Unit tests for GPUInstanceController's in-memory event coalescing.

These exercise ``_enqueue`` in isolation — the per-iid pending-slot policy is
concurrency-sensitive and easy to regress. The reconfirm sweep (#5587) routes
best-effort re-confirmation events through the same slot, and must never
displace a real /stop, /start, or /delete that is already pending.
"""

from gpustack.schemas.gpu_instances import (
    GPUInstance,
    GPUInstancePhase,
    GPUInstanceStatus,
)
from gpustack.server.bus import Event, EventType
from gpustack.server.controllers import GPUInstanceController


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
    controller = GPUInstanceController(cfg=None)
    # Pre-seed the per-iid worker slots so ``_enqueue`` treats a worker as
    # already running and does not spawn real asyncio tasks (which would open
    # DB sessions). We only assert on the synchronous ``_pending`` slot here.
    controller._workers = _AlwaysContains()
    return controller


class _AlwaysContains:
    """Stand-in for ``_workers`` that reports every iid as already having a
    worker, so ``_enqueue`` never calls ``asyncio.create_task``."""

    def __contains__(self, _key) -> bool:
        return True

    def __setitem__(self, _key, _value) -> None:  # pragma: no cover - unused
        pass


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

    pending = controller._pending[1]
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

    assert controller._pending[2].type == EventType.DELETED


def test_reconfirm_fills_empty_slot():
    controller = _controller()

    controller._enqueue(
        Event(
            type=EventType.UPDATED, data=_gi(3, GPUInstancePhase.READY), reconfirm=True
        )
    )

    assert controller._pending[3].reconfirm is True


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

    pending = controller._pending[4]
    assert pending.reconfirm is False
    assert pending.data.status.phase == GPUInstancePhase.STOPPING
