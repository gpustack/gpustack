"""Per-model reconcile debouncing.

For a role-bearing model this is a correctness requirement, not a throughput
one. The trigger chain is "instance DELETED -> Model UPDATED -> reconcile", so
retiring a 4P4D generation is nine deletions and nine reconciles — and every
middle one observes a group short of members. Reconciling on each recreates
what the deletion is halfway through removing.

The coalescing policy carries the load here, and the part worth pinning is that
it is NOT plain latest-wins: `notify_model_route_target` decides whether to
publish by asking which fields moved, so keeping only the newest event's
`changed_fields` silently drops the notification an earlier event carried.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from gpustack.api.exceptions import ConflictException

from gpustack.config.config import Config
from gpustack.schemas.config import GatewayModeEnum
from gpustack.schemas.models import (
    ComputedResourceClaim,
    DistributedServers,
    Model,
    ModelInstance,
    ModelInstanceStateEnum,
    ModelInstanceSubordinateWorker,
    ModelInstanceWorkload,
    ModelInstanceWorkloadStateEnum,
    SourceEnum,
)
from gpustack.server.bus import Event, EventType
from gpustack.server.controllers import ModelController, ModelInstanceController
from gpustack.server.services import ModelInstanceService
from gpustack.server.workqueue import WorkEvent, WorkEventType
from tests.utils.model import new_model_instance, new_model_instance_workload


def _controller() -> ModelController:
    config = Config(
        token="t",
        jwt_secret_key="s",
        gateway_mode=GatewayModeEnum.disabled,
        data_dir="/tmp/gpustack-test",
    )
    return ModelController(config)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'unavailable', ['none', 'missing', 'deleted', 'unreachable', 'offline', 'not_ready']
)
async def test_managed_worker_readiness_uses_one_fresh_complete_batch(unavailable):
    from gpustack.schemas.workers import Worker

    rows = [
        SimpleNamespace(
            id=id_,
            deleted_at=None,
            unreachable=False,
            state='maintenance' if id_ == 2 else 'ready',
            heartbeat_time=datetime.now(timezone.utc),
        )
        for id_ in (1, 2)
    ]
    if unavailable == 'missing':
        rows.pop()
    elif unavailable == 'deleted':
        rows[-1].deleted_at = datetime.now(timezone.utc)
    elif unavailable == 'unreachable':
        rows[-1].unreachable = True
    elif unavailable == 'offline':
        rows[-1].heartbeat_time = datetime.now(timezone.utc) - timedelta(days=1)
    elif unavailable == 'not_ready':
        rows[-1].state = 'not_ready'
    session = MagicMock()
    service = ModelInstanceService(session)
    with (
        patch.object(Worker, 'all_by_ids', AsyncMock(return_value=rows)) as batch,
        patch.object(Worker, 'one_by_id', AsyncMock()) as single,
    ):
        assert await service._managed_workers_ready({1, 2}) == (unavailable == 'none')
        batch.assert_awaited_once_with(session, {1, 2}, refresh=True)
        single.assert_not_awaited()


def _work(model_id, type_, changed_fields):
    return WorkEvent(
        keys=(model_id,),
        type=type_,
        object=Event(
            type=EventType.UPDATED,
            data=SimpleNamespace(id=model_id, name="m"),
            changed_fields=changed_fields,
        ),
    )


# --- the coalescing policy ------------------------------------------------- #


def test_the_newest_row_wins():
    merge = ModelController._merge_events
    older = _work(1, WorkEventType.MODIFIED, {"replicas": (1, 2)})
    newer = _work(1, WorkEventType.MODIFIED, {"replicas": (2, 3)})

    merged = merge(older, newer)

    assert merged.object is newer.object


def test_changed_fields_are_unioned_not_replaced():
    """The reason this is not latest-wins. A `state` transition followed by an
    unrelated edit would otherwise leave the gateway holding a target it was
    never told to update."""
    merge = ModelController._merge_events
    older = _work(1, WorkEventType.MODIFIED, {"state": ("pending", "running")})
    newer = _work(1, WorkEventType.MODIFIED, {"description": ("a", "b")})

    merged = merge(older, newer)

    assert set(merged.object.changed_fields) == {"state", "description"}


def test_a_field_that_moved_twice_keeps_the_whole_span():
    """One reconcile of everything since the last one means the pair has to
    describe the span, not its final step."""
    merge = ModelController._merge_events
    older = _work(1, WorkEventType.MODIFIED, {"ready_replicas": (0, 1)})
    newer = _work(1, WorkEventType.MODIFIED, {"ready_replicas": (1, 3)})

    merged = merge(older, newer)

    assert merged.object.changed_fields["ready_replicas"] == (0, 3)


def test_a_pending_deleted_is_sticky():
    """A row that is gone must not be reconciled as if merely updated."""
    merge = ModelController._merge_events
    deleted = _work(1, WorkEventType.DELETED, {})
    updated = _work(1, WorkEventType.MODIFIED, {"replicas": (1, 2)})

    assert merge(deleted, updated) is deleted


def test_a_deleted_upgrades_whatever_is_pending():
    merge = ModelController._merge_events
    updated = _work(1, WorkEventType.MODIFIED, {"replicas": (1, 2)})
    deleted = _work(1, WorkEventType.DELETED, {})

    assert merge(updated, deleted) is deleted


# --- the burst it exists to absorb ----------------------------------------- #


@pytest.mark.asyncio
async def test_a_burst_of_deletions_reconciles_once():
    """Nine members going away is nine bus events. Nine reconciles would each
    see an incomplete group; one reconcile sees the settled state."""
    controller = _controller()
    for _ in range(9):
        controller._queue.add(
            _work(1, WorkEventType.MODIFIED, {"ready_replicas": (1, 0)})
        )

    assert len(controller._queue) == 1


@pytest.mark.asyncio
async def test_different_models_are_not_collapsed_together():
    """Keyed per model, so one busy model cannot delay another."""
    controller = _controller()
    controller._queue.add(_work(1, WorkEventType.MODIFIED, {"replicas": (1, 2)}))
    controller._queue.add(_work(2, WorkEventType.MODIFIED, {"replicas": (1, 2)}))

    assert len(controller._queue) == 2


@pytest.mark.asyncio
async def test_one_models_events_reconcile_serially():
    """`sync_replicas` reads the instance set and then writes to it, so two
    passes overlapping on one model would both see the pre-write count and
    both create."""
    controller = _controller()
    running = []
    peak = 0

    async def _slow_reconcile(event):
        nonlocal peak
        running.append(1)
        peak = max(peak, len(running))
        await asyncio.sleep(0.01)
        running.pop()

    with patch.object(controller, "_reconcile", AsyncMock(side_effect=_slow_reconcile)):
        dispatch = asyncio.create_task(controller._dispatch())
        for i in range(4):
            controller._queue.add(
                _work(1, WorkEventType.MODIFIED, {"replicas": (i, i + 1)})
            )
            await asyncio.sleep(0.005)
        await asyncio.sleep(0.1)
        dispatch.cancel()
        await asyncio.gather(dispatch, return_exceptions=True)

    assert peak == 1, "two reconciles of one model overlapped"


@pytest.mark.asyncio
async def test_a_failed_reconcile_is_retried_with_backoff():
    controller = _controller()

    with patch.object(
        controller, "_reconcile", AsyncMock(side_effect=RuntimeError("boom"))
    ):
        await controller._process(
            _work(1, WorkEventType.MODIFIED, {"replicas": (1, 2)})
        )

    assert controller._queue.failures((1,)) == 1


@pytest.mark.asyncio
async def test_a_successful_reconcile_clears_the_backoff():
    controller = _controller()
    event = _work(1, WorkEventType.MODIFIED, {"replicas": (1, 2)})

    with patch.object(
        controller, "_reconcile", AsyncMock(side_effect=RuntimeError("boom"))
    ):
        await controller._process(event)
    with patch.object(controller, "_reconcile", AsyncMock()):
        await controller._process(event)

    assert controller._queue.failures((1,)) == 0


# --- the one pass this controller books for itself ------------------------- #


async def _reconcile_once(controller, drain_due, recovery_due=None):
    """Drive one `_reconcile` with everything after `sync_replicas` stubbed.

    `_reconcile` swallows its own exceptions, so a half-built harness would
    show up as a booking that silently never happened rather than as an error.
    Everything it touches is patched for that reason.
    """
    model = Model(
        id=1,
        name="m",
        replicas=1,
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/repo",
        owner_principal_id=1,
        cluster_id=1,
    )

    @asynccontextmanager
    async def _session():
        yield MagicMock()

    with (
        patch("gpustack.server.controllers.async_session", _session),
        patch(
            "gpustack.server.controllers.sync_replicas",
            AsyncMock(return_value=drain_due),
        ),
        patch(
            "gpustack.server.controllers.Model.one_by_id",
            AsyncMock(return_value=model),
        ),
        patch("gpustack.server.controllers.sync_model_status", AsyncMock()),
        patch("gpustack.server.controllers.notify_model_route_target", AsyncMock()),
        patch("gpustack.server.controllers.sync_categories_and_meta", AsyncMock()),
        patch.object(controller, "_ensure_model_mcp_bridge", AsyncMock()),
        patch.object(controller, "_sync_model_ai_proxy", AsyncMock()),
        patch.object(
            controller,
            "_sync_managed_model_instances",
            AsyncMock(return_value=recovery_due),
        ),
        patch.object(controller._queue, "add_after") as add_after,
    ):
        await controller._reconcile(
            Event(type=EventType.UPDATED, data=model, changed_fields={})
        )
    return add_after


@pytest.mark.asyncio
async def test_an_open_drain_window_books_its_own_reap():
    """Nothing else would arrive.

    A member marked for drain changes no field `sync_model_status` publishes,
    so the mark produces no Model event, and a deployment that has settled
    produces none either. `_reap_drained` runs off this loop, so without the
    booking the window expires against a reconcile that never comes and the
    member holds its accelerators until someone happens to edit the model.
    """
    add_after = await _reconcile_once(_controller(), drain_due=42.0)

    add_after.assert_called_once()
    booked_event, delay = add_after.call_args[0]
    assert delay == pytest.approx(42.0, abs=0.1)
    assert booked_event.keys == (1,)


@pytest.mark.asyncio
async def test_nothing_draining_books_nothing():
    """The booking is the exception, not the routine: an ordinary reconcile
    must leave no timer behind, or every settled model would hold one."""
    add_after = await _reconcile_once(_controller(), drain_due=None)

    add_after.assert_not_called()


@pytest.fixture
def managed_recovery():
    model = Model(
        id=1,
        name="m",
        replicas=1,
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/repo",
        cluster_id=1,
        restart_on_error=True,
    )
    instance = new_model_instance(
        1,
        "mi",
        model.id,
        worker_id=1,
        state=ModelInstanceStateEnum.RUNNING,
        gpu_indexes=[0],
        computed_resource_claim=ComputedResourceClaim(vram={0: 1024}),
    )
    instance.port = 40000
    instance.ports = [40000]
    instance.distributed_servers = DistributedServers(
        subordinate_workers=[
            ModelInstanceSubordinateWorker(worker_id=2, gpu_indexes=[0])
        ],
        workloads=[
            ModelInstanceWorkload.model_validate(
                new_model_instance_workload(
                    worker_id=worker_id, node_rank=worker_id - 1, state="running"
                )
            )
            for worker_id in (1, 2)
        ],
    )
    session = AsyncMock()
    session.get.side_effect = lambda cls, key, **kwargs: (
        instance if cls is ModelInstance else model
    )

    async def update(row, source=None):
        for key, value in (source or {}).items():
            setattr(row, key, value)
        await session.commit()

    with patch.object(
        ModelInstanceService, "update", AsyncMock(side_effect=update)
    ) as write:
        yield SimpleNamespace(
            model=model,
            instance=instance,
            session=session,
            service=ModelInstanceService(session),
            write=write,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("worker_index", [0, 1])
@pytest.mark.parametrize("failure", ["failed", "stopped"])
async def test_managed_task_failure_stops_the_complete_instance(
    managed_recovery, worker_index, failure
):
    ctx = managed_recovery
    task = ctx.instance.distributed_servers.workloads[worker_index]
    task.state = ModelInstanceWorkloadStateEnum(failure)
    resources = ctx.instance.computed_resource_claim.model_copy(deep=True)

    await ctx.service.reconcile_managed_instance(ctx.instance.id, 0)

    assert ctx.instance.state == ModelInstanceStateEnum.ERROR
    assert task.name in ctx.instance.state_message
    assert str(task.worker_id) in ctx.instance.state_message
    assert "rank" in ctx.instance.state_message
    assert "stopping" not in ctx.instance.state_message
    assert ctx.instance.restart_count == 0
    assert ctx.instance.computed_resource_claim == resources
    assert ctx.instance.ports == [40000]
    assert len(ctx.instance.distributed_servers.workloads) == 2
    ctx.write.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("task_state", ["pending", "running", "unknown"])
async def test_managed_nonterminal_observation_does_not_request_restart(
    managed_recovery, task_state
):
    ctx = managed_recovery
    ctx.instance.distributed_servers.workloads[1].state = (
        ModelInstanceWorkloadStateEnum(task_state)
    )

    assert await ctx.service.reconcile_managed_instance(ctx.instance.id, 0) is None

    ctx.write.assert_not_awaited()
    assert ctx.instance.state == ModelInstanceStateEnum.RUNNING


@pytest.mark.asyncio
@pytest.mark.parametrize("remaining", ["pending", "running", "failed", "unknown"])
async def test_managed_recovery_requires_every_registered_task_stopped(
    managed_recovery, remaining
):
    ctx = managed_recovery
    ctx.instance.state = ModelInstanceStateEnum.ERROR
    ctx.instance.state_message = "first rank failure"
    ctx.instance.spec_digest = 'old'
    ctx.instance.resolved_path = '/models/running-generation'
    ctx.instance.distributed_servers.workloads[0].state = (
        ModelInstanceWorkloadStateEnum.STOPPED
    )
    ctx.instance.distributed_servers.workloads[1].state = (
        ModelInstanceWorkloadStateEnum(remaining)
    )

    await ctx.service.reconcile_managed_instance(ctx.instance.id, 0)

    ctx.write.assert_not_awaited()
    assert ctx.instance.state_message == "first rank failure"
    assert ctx.instance.resolved_path == '/models/running-generation'
    ctx.session.execute.assert_not_awaited()
    assert ctx.instance.restart_count == 0
    assert ctx.instance.distributed_servers.workloads


@pytest.mark.asyncio
async def test_managed_stopped_generation_advances_once_with_fresh_locked_state(
    managed_recovery,
):
    ctx = managed_recovery
    ctx.instance.state = ModelInstanceStateEnum.ERROR
    for item in ctx.instance.distributed_servers.workloads:
        item.state = ModelInstanceWorkloadStateEnum.STOPPED
    resources = ctx.instance.computed_resource_claim.model_copy(deep=True)

    await ctx.service.reconcile_managed_instance(ctx.instance.id, 0)
    await ctx.service.reconcile_managed_instance(ctx.instance.id, 0)
    await ctx.service.reconcile_managed_instance(ctx.instance.id, 1)

    assert ctx.instance.restart_count == 1
    assert ctx.instance.last_restart_time is not None
    assert ctx.instance.state == ModelInstanceStateEnum.SCHEDULED
    assert ctx.instance.distributed_servers.workloads == []
    assert ctx.instance.computed_resource_claim == resources
    assert ctx.instance.worker_id == 1
    assert ctx.instance.distributed_servers.subordinate_workers[0].worker_id == 2
    ctx.write.assert_awaited_once()
    assert ctx.session.get.call_args_list[0].kwargs == {
        "with_for_update": True,
        "populate_existing": True,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason",
    [
        "restart_off",
        "draining",
        "model_restart",
        "model_stopped",
        "model_deleted",
        "stale_pd",
    ],
)
async def test_managed_recovery_respects_current_policy_and_user_operations(
    managed_recovery, reason
):
    ctx = managed_recovery
    ctx.instance.state = ModelInstanceStateEnum.ERROR
    ctx.instance.state_message = "first rank failure"
    for item in ctx.instance.distributed_servers.workloads:
        item.state = ModelInstanceWorkloadStateEnum.STOPPED
    now = datetime.now(timezone.utc)
    if reason == "restart_off":
        ctx.model.restart_on_error = False
    elif reason == "draining":
        ctx.instance.draining_since = now
    elif reason == "model_restart":
        ctx.model.restarting_since = now
    elif reason == "model_stopped":
        ctx.model.replicas = 0
    elif reason == "model_deleted":
        ctx.model.deleted_at = now
    else:
        ctx.instance.group_id = "pd-group"
        ctx.model.stale = True

    await ctx.service.reconcile_managed_instance(ctx.instance.id, 0)

    ctx.write.assert_not_awaited()
    assert ctx.instance.state == ModelInstanceStateEnum.ERROR
    assert ctx.instance.state_message == "first rank failure"
    assert ctx.instance.restart_count == 0


@pytest.mark.asyncio
async def test_managed_failure_requests_cleanup_even_with_automatic_restart_off(
    managed_recovery,
):
    ctx = managed_recovery
    ctx.model.restart_on_error = False
    ctx.instance.distributed_servers.workloads[1].state = (
        ModelInstanceWorkloadStateEnum.FAILED
    )

    await ctx.service.reconcile_managed_instance(ctx.instance.id, 0)

    assert ctx.instance.state == ModelInstanceStateEnum.ERROR
    ctx.write.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "restart_count,delay",
    [(0, 0), (1, 10), (2, 20), (3, 40), (4, 80), (5, 160), (6, 300), (1000, 300)],
)
async def test_managed_restart_backoff_survives_a_new_service(
    managed_recovery, restart_count, delay
):
    ctx = managed_recovery
    ctx.instance.state = ModelInstanceStateEnum.ERROR
    ctx.instance.restart_count = restart_count
    ctx.instance.distributed_servers.workloads = []
    ctx.instance.distributed_servers.orchestration_mode = "managed"
    ctx.instance.last_restart_time = datetime.now(timezone.utc) - timedelta(seconds=1)

    due = await ModelInstanceService(ctx.session).reconcile_managed_instance(
        ctx.instance.id, restart_count
    )

    if delay:
        assert due == pytest.approx(delay - 1, abs=0.2)
        ctx.write.assert_not_awaited()
        ctx.instance.updated_at = datetime.now(timezone.utc)
        repeated_due = await ctx.service.reconcile_managed_instance(
            ctx.instance.id, restart_count
        )
        assert repeated_due == pytest.approx(due, abs=0.2)
        ctx.instance.last_restart_time -= timedelta(seconds=delay)
        await ModelInstanceService(ctx.session).reconcile_managed_instance(
            ctx.instance.id, restart_count
        )
    assert ctx.instance.restart_count == restart_count + 1
    ctx.write.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("manifest", [None, []])
async def test_managed_recovery_keeps_legacy_and_empty_manifest_distinct(
    managed_recovery, manifest
):
    ctx = managed_recovery
    ctx.instance.state = ModelInstanceStateEnum.ERROR
    ctx.instance.distributed_servers.workloads = manifest
    ctx.instance.distributed_servers.orchestration_mode = (
        "managed"
        if ctx.instance.distributed_servers.workloads is not None
        else "legacy"
    )

    await ctx.service.reconcile_managed_instance(ctx.instance.id, 0)

    assert ctx.instance.restart_count == (0 if manifest is None else 1)


@pytest.mark.asyncio
async def test_model_pass_books_managed_backoff_even_without_a_drain():
    controller = _controller()
    add_after = await _reconcile_once(controller, drain_due=None, recovery_due=17)
    assert add_after.call_args.args[1] == pytest.approx(17, abs=0.1)


@pytest.mark.asyncio
async def test_managed_retry_and_drain_share_the_earliest_model_timer():
    controller = _controller()
    add_after = await _reconcile_once(controller, drain_due=8, recovery_due=17)
    assert add_after.call_args.args[1] == pytest.approx(8, abs=0.1)


@pytest.mark.asyncio
@pytest.mark.parametrize("worker_index", [0, 1])
async def test_managed_failure_withdraws_before_gateway_work_even_if_it_fails(
    managed_recovery, worker_index
):
    ctx = managed_recovery
    controller = _controller()
    ctx.instance.distributed_servers.workloads[worker_index].state = (
        ModelInstanceWorkloadStateEnum.FAILED
    )

    @asynccontextmanager
    async def session():
        yield ctx.session

    async def gateway(*args):
        assert ctx.instance.state == ModelInstanceStateEnum.ERROR
        assert len(ctx.instance.distributed_servers.workloads) == 2
        raise RuntimeError("gateway unavailable")

    with (
        patch("gpustack.server.controllers.async_session", session),
        patch.object(Model, "one_by_id", AsyncMock(return_value=ctx.model)),
        patch.object(
            ModelInstance, "all_by_field", AsyncMock(return_value=[ctx.instance])
        ),
        patch(
            "gpustack.server.controllers.sync_replicas", AsyncMock(return_value=None)
        ),
        patch("gpustack.server.controllers.sync_model_status", AsyncMock()),
        patch("gpustack.server.controllers.notify_model_route_target", AsyncMock()),
        patch("gpustack.server.controllers.sync_categories_and_meta", AsyncMock()),
        patch.object(controller, "_ensure_model_mcp_bridge", AsyncMock()),
        patch.object(
            controller, "_sync_model_ai_proxy", AsyncMock(side_effect=gateway)
        ),
    ):
        await controller._process(
            WorkEvent(
                keys=(1,),
                type=WorkEventType.MODIFIED,
                object=Event(type=EventType.UPDATED, data=ctx.model),
            )
        )

    assert controller._queue.failures((1,)) == 1
    assert ctx.instance.restart_count == 0
    ctx.write.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("initializing", [False, True])
@pytest.mark.parametrize("guarded_legacy", [False, True])
@pytest.mark.parametrize(
    'changed_field',
    [
        None,
        'download_progress',
        'model_config_snapshot',
        'distributed_servers',
        'state',
        'restart_count',
        'draining_since',
        'worker_id',
        'worker_ip',
        'worker_advertise_address',
        'port',
        'ports',
    ],
)
async def test_rank_manifest_event_wakes_model_recovery_without_aggregate_change(
    managed_recovery, initializing, guarded_legacy, changed_field
):
    ctx = managed_recovery
    controller = ModelInstanceController(_controller()._config)
    if guarded_legacy:
        from uuid import uuid4
        from gpustack.schemas.models import ModelInstanceLegacyStop

        ctx.instance.distributed_servers.workloads = None
        ctx.instance.distributed_servers.orchestration_mode = "legacy"
        ctx.instance.distributed_servers.legacy_stop = ModelInstanceLegacyStop(
            token=uuid4(),
            restart_count=0,
            worker_ids=[1, 2],
            worker_principal_ids={1: 10, 2: 20},
            confirmed_worker_ids=[1, 2],
        )
    if initializing:
        ctx.instance.state = ModelInstanceStateEnum.INITIALIZING

    @asynccontextmanager
    async def session():
        yield ctx.session

    with (
        patch("gpustack.server.controllers.async_session", session),
        patch.object(Model, "one_by_id", AsyncMock(return_value=ctx.model)),
        patch.object(Model, "refresh", AsyncMock()),
        patch(
            "gpustack.server.controllers.sync_model_status",
            AsyncMock(return_value=False),
        ),
        patch(
            "gpustack.server.controllers.ensure_instance_model_file", AsyncMock()
        ) as ensure_files,
        patch(
            'gpustack.server.controllers.cleanup_orphan_lora_routes',
            AsyncMock(return_value=False),
        ),
        patch("gpustack.server.controllers.event_bus.publish", AsyncMock()) as publish,
        patch(
            "gpustack.server.controllers.notify_model_route_target", AsyncMock()
        ) as notify,
    ):
        await controller._reconcile(
            Event(
                type=EventType.UPDATED,
                data=ctx.instance,
                changed_fields=(
                    {changed_field: (None, getattr(ctx.instance, changed_field))}
                    if changed_field
                    else {}
                ),
            )
        )
        await asyncio.sleep(0)

    runtime_changed = changed_field not in (
        'download_progress',
        'model_config_snapshot',
    )
    assert publish.await_count == int(runtime_changed)
    if runtime_changed:
        topic, event = publish.call_args.args
        assert topic == "model"
        assert event.data.id == ctx.model.id
        assert event.type == EventType.UPDATED
    assert ensure_files.await_count == int(initializing and not guarded_legacy)
    assert notify.await_count == int(
        runtime_changed and not initializing and not guarded_legacy
    )


@pytest.mark.asyncio
async def test_managed_error_keeps_first_failure_despite_late_readiness(
    managed_recovery,
):
    ctx = managed_recovery
    ctx.instance.state = ModelInstanceStateEnum.ERROR
    ctx.instance.state_message = "original failure"
    ctx.instance.distributed_servers.workloads[0].dp_rank_placements[0].ready = True

    await ctx.service.reconcile_managed_instance(ctx.instance.id, 0)

    ctx.write.assert_not_awaited()
    assert ctx.instance.state == ModelInstanceStateEnum.ERROR
    assert ctx.instance.state_message == "original failure"


@pytest.mark.asyncio
async def test_partial_registration_can_recover_after_every_registered_task_stops(
    managed_recovery,
):
    ctx = managed_recovery
    ctx.instance.state = ModelInstanceStateEnum.ERROR
    ctx.instance.distributed_servers.workloads = (
        ctx.instance.distributed_servers.workloads[:1]
    )
    ctx.instance.distributed_servers.orchestration_mode = (
        "managed"
        if ctx.instance.distributed_servers.workloads is not None
        else "legacy"
    )
    ctx.instance.distributed_servers.workloads[0].state = (
        ModelInstanceWorkloadStateEnum.STOPPED
    )

    await ctx.service.reconcile_managed_instance(ctx.instance.id, 0)

    assert ctx.instance.restart_count == 1
    assert ctx.instance.distributed_servers.workloads == []


@pytest.mark.asyncio
async def test_manual_dp_edit_keeps_managed_recovery_until_explicit_rebuild(
    managed_recovery,
):
    ctx = managed_recovery
    ctx.model.env = {'GPUSTACK_MANUAL_DP': '1'}
    ctx.instance.state = ModelInstanceStateEnum.ERROR
    for task in ctx.instance.distributed_servers.workloads:
        task.state = ModelInstanceWorkloadStateEnum.STOPPED

    await ctx.service.reconcile_managed_instance(ctx.instance.id, 0)

    assert ctx.instance.restart_count == 1
    assert ctx.instance.distributed_servers.orchestration_mode == 'managed'
    assert ctx.instance.distributed_servers.workloads == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'edited', [None, 'parameters', 'manual_dp', 'topology', 'version']
)
async def test_ordinary_managed_recovery_registers_the_current_configuration(
    managed_recovery, edited
):
    from gpustack.routes.model_instances import _validate_workload_layout
    from gpustack.schemas.models import PortBand
    from gpustack.server.controllers import model_spec_digest

    ctx = managed_recovery
    ctx.session.info = {}
    ctx.model.backend = 'vLLM'
    ctx.model.backend_version = '0.31.0'
    ctx.instance.backend = 'vLLM'
    ctx.instance.backend_version = '0.31.0'
    ctx.instance.api_detected_backend_version = '0.31.0'
    ctx.model.backend_parameters = ['--data-parallel-external-lb', '--tp=1', '--dp=2']
    ctx.instance.spec_digest = await model_spec_digest(ctx.session, ctx.model)
    ctx.instance.state = ModelInstanceStateEnum.ERROR
    for task in ctx.instance.distributed_servers.workloads:
        task.state = ModelInstanceWorkloadStateEnum.STOPPED
    if edited:
        if edited == 'manual_dp':
            ctx.model.env = {'GPUSTACK_MANUAL_DP': '1'}
        elif edited == 'version':
            ctx.model.backend_version = '0.32.0'
        elif edited == 'topology':
            ctx.model.backend_parameters = [
                '--data-parallel-external-lb',
                '--tp=2',
                '--dp=1',
            ]
        else:
            ctx.model.backend_parameters.append('--max-model-len=4096')
        ctx.model.stale = True
        ctx.session.info.clear()
    target = await model_spec_digest(ctx.session, ctx.model)

    await ctx.service.reconcile_managed_instance(ctx.instance.id, 0)

    assert ctx.instance.state == ModelInstanceStateEnum.SCHEDULED
    assert ctx.instance.spec_digest == target
    assert ctx.instance.backend == ctx.model.backend
    assert ctx.instance.backend_version == ctx.model.backend_version
    assert ctx.instance.api_detected_backend_version == (None if edited else '0.31.0')
    assert ctx.instance.worker_id == 1
    assert ctx.instance.gpu_indexes == [0]
    assert ctx.instance.distributed_servers.orchestration_mode == 'managed'
    assert ctx.instance.port == (None if edited else 40000)
    assert ctx.instance.ports == (None if edited else [40000])
    # Preparation republishes the new generation's ports before task registration.
    api_count = 1 if edited == 'topology' else 2
    ctx.instance.named_ports = {
        '_vllm_external_api': PortBand(base=40000, count=api_count)
    }
    task = ModelInstanceWorkload.model_validate(
        new_model_instance_workload(worker_id=1, dp_rank=0)
    )
    await _validate_workload_layout(
        ctx.session,
        ctx.instance,
        [task],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('backend', ['vLLM', 'SGLang', 'custom'])
async def test_ordinary_recovery_uses_latest_single_entry_backend(
    managed_recovery, backend
):
    from gpustack.server.controllers import model_spec_digest

    ctx = managed_recovery
    ctx.session.info = {}
    ctx.model.backend = 'vLLM'
    ctx.model.backend_parameters = ['--data-parallel-external-lb', '--tp=1', '--dp=2']
    ctx.instance.spec_digest = await model_spec_digest(ctx.session, ctx.model)
    ctx.instance.state = ModelInstanceStateEnum.ERROR
    for task in ctx.instance.distributed_servers.workloads:
        task.state = ModelInstanceWorkloadStateEnum.STOPPED
    ctx.model.backend = backend
    ctx.model.backend_parameters = []
    ctx.model.stale = True
    ctx.session.info.clear()

    await ctx.service.reconcile_managed_instance(ctx.instance.id, 0)

    assert ctx.instance.backend == backend
    assert ctx.instance.distributed_servers.orchestration_mode == 'legacy'
    assert ctx.instance.distributed_servers.workloads is None
    assert ctx.instance.port is None and ctx.instance.ports is None
    assert ctx.instance.computed_resource_claim.vram == {0: 1024}


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", ["deleted", "new_generation", "missing"])
async def test_managed_recovery_ignores_a_stale_reconcile_snapshot(
    managed_recovery, changed
):
    ctx = managed_recovery
    ctx.instance.state = ModelInstanceStateEnum.ERROR
    ctx.instance.distributed_servers.workloads = []
    ctx.instance.distributed_servers.orchestration_mode = "managed"
    if changed == "deleted":
        ctx.instance.deleted_at = datetime.now(timezone.utc)
    elif changed == "new_generation":
        ctx.instance.restart_count = 2
    else:
        ctx.session.get.return_value = None
        ctx.session.get.side_effect = None

    await ctx.service.reconcile_managed_instance(ctx.instance.id, 0)

    ctx.write.assert_not_awaited()
    ctx.session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_new_model_controller_resumes_only_the_failed_managed_instance(
    managed_recovery,
):
    ctx = managed_recovery
    ctx.instance.state = ModelInstanceStateEnum.ERROR
    ctx.instance.distributed_servers.workloads = []
    ctx.instance.distributed_servers.orchestration_mode = "managed"
    healthy = ctx.instance.model_copy(deep=True)
    healthy.id = 2
    healthy.state = ModelInstanceStateEnum.RUNNING
    legacy = new_model_instance(3, "legacy", 1, state=ModelInstanceStateEnum.ERROR)

    with patch.object(
        ModelInstance,
        "all_by_field",
        AsyncMock(return_value=[ctx.instance, healthy, legacy]),
    ):
        await _controller()._sync_managed_model_instances(ctx.session, ctx.model)
        await _controller()._sync_managed_model_instances(ctx.session, ctx.model)

    ctx.write.assert_awaited_once()
    assert ctx.write.call_args.args[0].id == 1
    assert ctx.instance.restart_count == 1
    assert healthy.restart_count == 0
    assert healthy.state == ModelInstanceStateEnum.RUNNING
    assert legacy.restart_count == 0


@pytest.mark.asyncio
async def test_pd_config_change_blocks_recovery_before_stale_status_is_published(
    managed_recovery,
):
    ctx = managed_recovery
    ctx.instance.group_id = "pd-group"
    ctx.instance.state = ModelInstanceStateEnum.ERROR
    ctx.instance.distributed_servers.workloads = []
    ctx.instance.distributed_servers.orchestration_mode = "managed"
    ctx.model.stale = False

    with (
        patch.object(
            ModelInstance, "all_by_field", AsyncMock(return_value=[ctx.instance])
        ),
        patch(
            "gpustack.server.controllers.model_members_stale",
            AsyncMock(return_value=True),
        ),
    ):
        await _controller()._sync_managed_model_instances(ctx.session, ctx.model)

    ctx.write.assert_not_awaited()
    assert ctx.instance.restart_count == 0


@pytest.mark.asyncio
async def test_pd_failure_requests_cleanup_before_reading_recovery_configuration(
    managed_recovery,
):
    ctx = managed_recovery
    ctx.instance.group_id = "pd-group"
    ctx.instance.distributed_servers.workloads[1].state = (
        ModelInstanceWorkloadStateEnum.FAILED
    )
    with (
        patch.object(
            ModelInstance, "all_by_field", AsyncMock(return_value=[ctx.instance])
        ),
        patch(
            "gpustack.server.controllers.model_members_stale",
            AsyncMock(side_effect=RuntimeError("configuration unavailable")),
        ) as stale,
    ):
        await _controller()._sync_managed_model_instances(ctx.session, ctx.model)

    stale.assert_not_awaited()
    assert ctx.instance.state == ModelInstanceStateEnum.ERROR
    ctx.write.assert_awaited_once()


@pytest.mark.asyncio
async def test_managed_pd_failure_recovers_only_its_complete_instance(managed_recovery):
    ctx = managed_recovery
    ctx.instance.role, ctx.instance.group_id = 'prefill', 'g1'
    healthy = []
    for identity, role, group in [
        (2, 'decode', 'g1'),
        (3, 'router', 'g1'),
        (4, 'prefill', 'g2'),
    ]:
        peer = ctx.instance.model_copy(deep=True)
        peer.id, peer.role, peer.group_id = identity, role, group
        if role == 'router':
            peer.distributed_servers = None
        healthy.append(peer)
    before = [peer.model_dump() for peer in healthy]
    controller = ModelController.__new__(ModelController)
    with (
        patch.object(
            ModelInstance,
            'all_by_field',
            AsyncMock(return_value=[ctx.instance, *healthy]),
        ),
        patch(
            'gpustack.server.controllers.model_members_stale',
            AsyncMock(return_value=False),
        ),
    ):
        ctx.instance.distributed_servers.workloads[0].state = (
            ModelInstanceWorkloadStateEnum.FAILED
        )
        await controller._sync_managed_model_instances(ctx.session, ctx.model)
        assert ctx.instance.state == ModelInstanceStateEnum.ERROR
        for task in ctx.instance.distributed_servers.workloads:
            task.state = ModelInstanceWorkloadStateEnum.STOPPED
        await controller._sync_managed_model_instances(ctx.session, ctx.model)
    assert ctx.instance.restart_count == 1
    assert ctx.instance.distributed_servers.workloads == []
    assert [peer.model_dump() for peer in healthy] == before
    assert all(call.args[0] is ctx.instance for call in ctx.write.call_args_list)


@pytest.mark.asyncio
async def test_mixed_stop_waits_for_legacy_router_and_reuses_its_token(
    managed_recovery, monkeypatch
):
    from gpustack.server import services
    from gpustack.schemas.workers import Worker

    ctx = managed_recovery
    engine = ctx.instance
    engine.group_id = 'g1'
    for item in engine.distributed_servers.workloads:
        item.state = ModelInstanceWorkloadStateEnum.STOPPED
    router = new_model_instance(2, 'router', 1, worker_id=2)
    router.role, router.group_id = 'router', 'g1'
    router.distributed_servers = None
    rows = {1: engine, 2: router}
    monkeypatch.setattr(
        ModelInstance,
        'one_by_id',
        AsyncMock(side_effect=lambda session, identity, **kw: rows[identity]),
    )
    monkeypatch.setattr(
        Worker,
        'one_by_id',
        AsyncMock(return_value=SimpleNamespace(system_principal_id=20)),
    )

    async def persist(row, session, source=None, **kwargs):
        for key, value in (source or {}).items():
            setattr(row, key, value)
        return row

    monkeypatch.setattr(ModelInstance, 'update', persist)
    delete = AsyncMock()
    monkeypatch.setattr(ModelInstance, 'delete', delete)
    monkeypatch.setattr(ModelInstance, '_invalidate_cached_all', AsyncMock())
    monkeypatch.setattr(services, 'delete_cache_by_key', AsyncMock())
    invalidate = AsyncMock()
    monkeypatch.setattr(services, 'invalidate_workers_allocated', invalidate)
    assert (
        await ctx.service.batch_delete([engine, router], require_all_stopped=True) == []
    )
    guard = router.distributed_servers.legacy_stop
    assert guard.worker_ids == [2]
    assert guard.worker_principal_ids == {2: 20}
    assert router.distributed_servers.workloads is None
    token = guard.token
    assert (
        await ctx.service.batch_delete([engine, router], require_all_stopped=True) == []
    )
    assert router.distributed_servers.legacy_stop.token == token
    delete.assert_not_awaited()
    router.distributed_servers.legacy_stop.confirmed_worker_ids = [2]
    assert await ctx.service.batch_delete(
        [engine, router], require_all_stopped=True
    ) == [engine.name, router.name]
    assert delete.await_count == 2
    assert invalidate.await_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize('task_evidence', [False, True])
async def test_empty_stop_ownership_only_covers_unassigned_taskless_rows(
    managed_recovery, monkeypatch, task_evidence
):
    ctx = managed_recovery
    ctx.instance.group_id = 'g1'
    waiting = new_model_instance(
        2, 'waiting', 1, worker_id=None, state=ModelInstanceStateEnum.PENDING
    )
    waiting.pid = 123 if task_evidence else None
    waiting.port, waiting.ports, waiting.named_ports = None, None, None

    async def persist(row, session, source=None, **kwargs):
        for key, value in (source or {}).items():
            setattr(row, key, value)
        return row

    monkeypatch.setattr(ModelInstance, 'update', persist)
    if task_evidence:
        with pytest.raises(ConflictException):
            await ctx.service.request_stop([ctx.instance, waiting], auto_commit=False)
        assert waiting.distributed_servers is None
    else:
        await ctx.service.request_stop([ctx.instance, waiting], auto_commit=False)
        assert waiting.distributed_servers.legacy_stop.worker_ids == []
        assert waiting.distributed_servers.workloads is None


@pytest.fixture
def smg_recovery(managed_recovery):
    ctx = managed_recovery
    ctx.instance.distributed_servers.workloads = [
        ModelInstanceWorkload(
            name=f'engine-{worker}',
            worker_id=worker,
            state='running',
            component=dict(
                kind='sglang_engine',
                **(dict(api_port=40000, dp_size=2, ready=True) if worker == 1 else {}),
            ),
        )
        for worker in (1, 2)
    ] + [
        ModelInstanceWorkload(
            name='mi-r0-smg-a0',
            worker_id=1,
            state='failed',
            component=dict(kind='sglang_smg', api_port=40001, metrics_port=40002),
        )
    ]
    ctx.instance.distributed_servers.orchestration_mode = (
        "managed"
        if ctx.instance.distributed_servers.workloads is not None
        else "legacy"
    )
    workers = [
        SimpleNamespace(
            id=worker_id,
            deleted_at=None,
            unreachable=False,
            state='ready',
            heartbeat_time=datetime.now(timezone.utc),
        )
        for worker_id in (1, 2)
    ]
    with patch(
        'gpustack.server.services.Worker.all_by_ids',
        AsyncMock(return_value=workers),
    ):
        yield ctx


@pytest.mark.asyncio
async def test_smg_failure_and_recovery_preserve_healthy_engine_generation(
    smg_recovery,
):
    ctx = smg_recovery
    engines = [w.model_dump() for w in ctx.instance.distributed_servers.workloads[:2]]
    await ctx.service.reconcile_managed_instance(ctx.instance.id, 0)
    smg = ctx.instance.distributed_servers.workloads[-1]
    assert ctx.instance.state == ModelInstanceStateEnum.STARTING
    assert smg.component.stop_requested
    assert 'SMG' in ctx.instance.state_message
    smg.state = ModelInstanceWorkloadStateEnum.STOPPED
    await ctx.service.reconcile_managed_instance(ctx.instance.id, 0)
    await ctx.service.reconcile_managed_instance(ctx.instance.id, 0)
    smg = ctx.instance.distributed_servers.workloads[-1]
    assert smg.name == 'mi-r0-smg-a1'
    assert smg.component.attempt == 1
    assert not smg.component.stop_requested
    assert smg.component.api_port == 40001
    assert smg.component.metrics_port == 40002
    assert smg.state == ModelInstanceWorkloadStateEnum.PENDING
    assert ctx.instance.state_message == ''
    assert ctx.instance.restart_count == 0
    assert ctx.instance.last_restart_time is None
    assert [
        w.model_dump() for w in ctx.instance.distributed_servers.workloads[:2]
    ] == engines


@pytest.mark.asyncio
@pytest.mark.parametrize('state', ['failed', 'unknown', 'running'])
async def test_smg_unconfirmed_stop_cannot_advance_attempt(smg_recovery, state):
    ctx = smg_recovery
    smg = ctx.instance.distributed_servers.workloads[-1]
    smg.state = ModelInstanceWorkloadStateEnum(state)
    smg.component.stop_requested = True
    await ctx.service.reconcile_managed_instance(ctx.instance.id, 0)
    assert ctx.instance.distributed_servers.workloads[-1].component.attempt == 0
    assert ctx.instance.restart_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'fence',
    [
        'restart_off',
        'draining',
        'model_restart',
        'inactive',
        'stale',
        'deleted',
        'replicas_zero',
        'not_allowed',
        'engine_unknown',
        'worker_unknown',
    ],
)
async def test_smg_recovery_obeys_current_instance_and_model_fences(
    smg_recovery, fence
):
    ctx = smg_recovery
    smg = ctx.instance.distributed_servers.workloads[-1]
    smg.state = ModelInstanceWorkloadStateEnum.STOPPED
    smg.component.stop_requested = True
    if fence == 'restart_off':
        ctx.model.restart_on_error = False
    elif fence == 'draining':
        ctx.instance.draining_since = datetime.now(timezone.utc)
    elif fence == 'model_restart':
        ctx.model.restarting_since = datetime.now(timezone.utc)
    elif fence == 'inactive':
        ctx.instance.state = ModelInstanceStateEnum.SCHEDULED
    elif fence == 'stale':
        ctx.model.stale = True
    elif fence == 'deleted':
        ctx.model.deleted_at = datetime.now(timezone.utc)
    elif fence == 'replicas_zero':
        ctx.model.replicas = 0
    elif fence == 'engine_unknown':
        ctx.instance.distributed_servers.workloads[0].state = (
            ModelInstanceWorkloadStateEnum.UNKNOWN
        )
    elif fence == 'worker_unknown':
        with patch(
            'gpustack.server.services.Worker.all_by_ids',
            AsyncMock(return_value=[]),
        ):
            await ctx.service.reconcile_managed_instance(ctx.instance.id, 0)
        assert ctx.instance.distributed_servers.workloads[-1].component.attempt == 0
        return
    await ctx.service.reconcile_managed_instance(
        ctx.instance.id, 0, allow_restart=fence != 'not_allowed'
    )
    assert ctx.instance.distributed_servers.workloads[-1].component.attempt == 0
    assert ctx.instance.restart_count == 0
    assert (
        all(
            w.state == ModelInstanceWorkloadStateEnum.RUNNING
            for w in ctx.instance.distributed_servers.workloads[:2]
        )
        or fence == 'engine_unknown'
    )
    if fence == 'restart_off':
        assert 'disabled' in ctx.instance.state_message


@pytest.mark.asyncio
async def test_engine_failure_takes_priority_over_smg_recovery(smg_recovery):
    ctx = smg_recovery
    ctx.instance.distributed_servers.workloads[0].state = (
        ModelInstanceWorkloadStateEnum.FAILED
    )
    await ctx.service.reconcile_managed_instance(ctx.instance.id, 0)
    assert ctx.instance.state == ModelInstanceStateEnum.ERROR
    assert 'engine-1' in ctx.instance.state_message
    assert ctx.instance.distributed_servers.workloads[-1].component.attempt == 0


@pytest.mark.asyncio
async def test_smg_backoff_survives_service_restart(smg_recovery):
    ctx = smg_recovery
    smg = ctx.instance.distributed_servers.workloads[-1]
    smg.state = ModelInstanceWorkloadStateEnum.STOPPED
    smg.component.stop_requested = True
    smg.component.attempt = 2
    smg.component.last_restart_time = datetime.now(timezone.utc)
    due = await ModelInstanceService(ctx.session).reconcile_managed_instance(
        ctx.instance.id, 0
    )
    assert 18 < due <= 20
    assert ctx.instance.distributed_servers.workloads[-1].component.attempt == 2
    ctx.instance.distributed_servers.workloads[
        -1
    ].component.last_restart_time -= timedelta(seconds=21)
    await ModelInstanceService(ctx.session).reconcile_managed_instance(
        ctx.instance.id, 0
    )
    assert ctx.instance.distributed_servers.workloads[-1].component.attempt == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'missing',
    ['none', 'engine', 'engine_api', 'smg_ready', 'smg_stop', 'smg_owner', 'worker'],
)
async def test_smg_readiness_requires_complete_engine_and_current_smg(
    smg_recovery, missing
):
    ctx = smg_recovery
    ctx.instance.state = ModelInstanceStateEnum.STARTING
    workloads = ctx.instance.distributed_servers.workloads
    workloads[-1].state = ModelInstanceWorkloadStateEnum.RUNNING
    workloads[-1].component.ready = True
    if missing == 'engine':
        workloads.pop(1)
    elif missing == 'engine_api':
        workloads[0].component.ready = False
    elif missing == 'smg_ready':
        workloads[-1].component.ready = False
    elif missing == 'smg_stop':
        workloads[-1].component.stop_requested = True
    elif missing == 'smg_owner':
        workloads[-1].worker_id = 2
    elif missing == 'worker':
        with patch(
            'gpustack.server.services.Worker.all_by_ids',
            AsyncMock(return_value=[]),
        ):
            await ctx.service.reconcile_managed_instance(ctx.instance.id, 0)
        assert ctx.instance.state == ModelInstanceStateEnum.STARTING
        return
    await ctx.service.reconcile_managed_instance(ctx.instance.id, 0)
    assert ctx.instance.state == (
        ModelInstanceStateEnum.RUNNING
        if missing == 'none'
        else ModelInstanceStateEnum.STARTING
    )


@pytest.mark.asyncio
async def test_controller_resumes_smg_backoff_and_unknown_observation(smg_recovery):
    ctx = smg_recovery
    smg = ctx.instance.distributed_servers.workloads[-1]
    smg.state = ModelInstanceWorkloadStateEnum.UNKNOWN
    with patch.object(
        ModelInstance, 'all_by_field', AsyncMock(return_value=[ctx.instance])
    ):
        assert (
            await _controller()._sync_managed_model_instances(ctx.session, ctx.model)
            is None
        )
        assert ctx.instance.state == ModelInstanceStateEnum.RUNNING
        assert not smg.component.stop_requested
        smg = ctx.instance.distributed_servers.workloads[-1]
        smg.state = ModelInstanceWorkloadStateEnum.STOPPED
        smg.component.attempt = 1
        smg.component.last_restart_time = datetime.now(timezone.utc)
        due = await _controller()._sync_managed_model_instances(ctx.session, ctx.model)
        assert 8 < due <= 10
    assert ctx.instance.restart_count == 0


@pytest.mark.asyncio
async def test_smg_recovery_checks_current_spec_under_instance_lock(smg_recovery):
    ctx = smg_recovery
    ctx.instance.spec_digest = 'old'
    smg = ctx.instance.distributed_servers.workloads[-1]
    smg.state = ModelInstanceWorkloadStateEnum.STOPPED
    with patch(
        'gpustack.server.controllers.model_members_stale', AsyncMock(return_value=True)
    ) as stale:
        await ctx.service.reconcile_managed_instance(ctx.instance.id, 0)
    stale.assert_awaited_once_with(ctx.session, ctx.model, [ctx.instance])
    assert ctx.instance.distributed_servers.workloads[-1].component.attempt == 0


@pytest.mark.asyncio
async def test_smg_unconfirmed_stop_does_not_republish_unchanged_state(smg_recovery):
    ctx = smg_recovery
    await ctx.service.reconcile_managed_instance(ctx.instance.id, 0)
    ctx.write.reset_mock()
    await ctx.service.reconcile_managed_instance(ctx.instance.id, 0)
    ctx.write.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('pending', ['running', 'unknown', 'stopped'])
async def test_engine_generation_restart_waits_for_smg_stop(smg_recovery, pending):
    ctx = smg_recovery
    ctx.instance.state = ModelInstanceStateEnum.ERROR
    for workload in ctx.instance.distributed_servers.workloads[:2]:
        workload.state = ModelInstanceWorkloadStateEnum.STOPPED
        workload.component.ready = False if workload.component.api_port else None
    ctx.instance.distributed_servers.workloads[-1].state = (
        ModelInstanceWorkloadStateEnum(pending)
    )
    await ctx.service.reconcile_managed_instance(ctx.instance.id, 0)
    assert ctx.instance.restart_count == int(pending == 'stopped')
    if pending == 'stopped':
        assert ctx.instance.distributed_servers.workloads == []
    else:
        assert len(ctx.instance.distributed_servers.workloads) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize('fence', ['deleted', 'new_generation'])
async def test_smg_reconcile_does_not_modify_stale_instance_generation(
    smg_recovery, fence
):
    ctx = smg_recovery
    ctx.instance.distributed_servers.workloads[-1].state = (
        ModelInstanceWorkloadStateEnum.STOPPED
    )
    if fence == 'deleted':
        ctx.instance.deleted_at = datetime.now(timezone.utc)
    else:
        ctx.instance.restart_count = 1
    await ctx.service.reconcile_managed_instance(ctx.instance.id, 0)
    ctx.write.assert_not_awaited()
    assert ctx.instance.distributed_servers.workloads[-1].component.attempt == 0


@pytest.mark.asyncio
@pytest.mark.parametrize('ready', [False, True])
async def test_unreachable_managed_instance_recovers_only_after_fresh_readiness(
    managed_recovery, ready
):
    ctx = managed_recovery
    ctx.instance.state = ModelInstanceStateEnum.UNREACHABLE
    resources = ctx.instance.computed_resource_claim.model_copy(deep=True)
    with (
        patch.object(
            ModelInstance, 'all_by_field', AsyncMock(return_value=[ctx.instance])
        ),
        patch.object(
            ctx.service.__class__,
            '_managed_instance_ready',
            AsyncMock(return_value=ready),
        ) as probe,
    ):
        await _controller()._sync_managed_model_instances(ctx.session, ctx.model)
    probe.assert_awaited_once()
    assert ctx.instance.state == (
        ModelInstanceStateEnum.RUNNING if ready else ModelInstanceStateEnum.UNREACHABLE
    )
    assert ctx.instance.restart_count == 0
    assert ctx.instance.computed_resource_claim == resources
    assert len(ctx.instance.distributed_servers.workloads) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'blocked', [None, 'unready', 'unknown', 'worker', 'draining', 'epoch']
)
async def test_managed_worker_recovery_preserves_generation_and_readiness_fences(
    managed_recovery, blocked
):
    from gpustack.schemas.workers import Worker, WorkerStateEnum

    ctx = managed_recovery
    ctx.instance.state = ModelInstanceStateEnum.UNREACHABLE
    for workload in ctx.instance.distributed_servers.workloads:
        for placement in workload.dp_rank_placements:
            if placement.node_rank == 0:
                placement.ready = blocked != 'unready'
    if blocked == 'unknown':
        ctx.instance.distributed_servers.workloads[1].state = (
            ModelInstanceWorkloadStateEnum.UNKNOWN
        )
    if blocked == 'draining':
        ctx.instance.draining_since = datetime.now(timezone.utc)
    if blocked == 'epoch':
        ctx.instance.restart_count = 1
    workers = [
        SimpleNamespace(
            id=worker_id,
            deleted_at=None,
            unreachable=blocked == 'worker',
            state=WorkerStateEnum.READY,
            heartbeat_time=datetime.now(timezone.utc),
        )
        for worker_id in (1, 2)
    ]
    with patch.object(Worker, 'all_by_ids', AsyncMock(return_value=workers)):
        await ctx.service.reconcile_managed_instance(ctx.instance.id, 0)
    assert ctx.instance.state == (
        ModelInstanceStateEnum.RUNNING
        if blocked is None
        else ModelInstanceStateEnum.UNREACHABLE
    )
    assert ctx.instance.restart_count == int(blocked == 'epoch')


@pytest.mark.asyncio
@pytest.mark.parametrize('lost_worker', [1, 2])
@pytest.mark.parametrize('restart_enabled', [False, True])
async def test_smg_worker_disconnect_preserves_processes_and_recovers(
    smg_recovery, lost_worker, restart_enabled
):
    ctx = smg_recovery
    ctx.model.restart_on_error = restart_enabled
    ctx.instance.state = ModelInstanceStateEnum.UNREACHABLE
    workloads = ctx.instance.distributed_servers.workloads
    smg = workloads[-1]
    smg.state = ModelInstanceWorkloadStateEnum.RUNNING
    smg.component.ready = True
    for item in workloads:
        if item.worker_id == lost_worker:
            item.state = ModelInstanceWorkloadStateEnum.UNKNOWN
            if item.component.api_port is not None:
                item.component.ready = False
    before = ctx.instance.model_dump()
    for _ in range(2):
        assert await ctx.service.reconcile_managed_instance(ctx.instance.id, 0) is None
        assert ctx.instance.model_dump() == before
    for item in workloads:
        item.state = ModelInstanceWorkloadStateEnum.RUNNING
        if item.component.api_port is not None:
            item.component.ready = True
    await ctx.service.reconcile_managed_instance(ctx.instance.id, 0)
    assert ctx.instance.state == ModelInstanceStateEnum.RUNNING
    assert ctx.instance.restart_count == 0
    assert ctx.instance.distributed_servers.workloads[-1].component.attempt == 0
    assert not ctx.instance.distributed_servers.workloads[-1].component.stop_requested


@pytest.mark.asyncio
async def test_unreleased_drain_retry_has_a_nonzero_delay():
    add_after = await _reconcile_once(_controller(), drain_due=0)
    delay = add_after.call_args.args[1]
    assert 0.8 < delay <= 1


@pytest.mark.asyncio
@pytest.mark.parametrize('headless', [False, True])
@pytest.mark.parametrize('state', ['failed', 'stopped'])
async def test_hybrid_task_failure_uses_complete_group_recovery(
    managed_recovery, headless, state
):
    from tests.utils.model import new_hybrid_workload

    ctx = managed_recovery
    ctx.instance.distributed_servers.workloads = [
        ModelInstanceWorkload.model_validate(new_hybrid_workload(1, state='running')),
        ModelInstanceWorkload.model_validate(
            new_hybrid_workload(2, headless=headless, state=state)
        ),
    ]
    ctx.instance.distributed_servers.orchestration_mode = "managed"
    await ctx.service.reconcile_managed_instance(ctx.instance.id, 0)
    assert ctx.instance.state == ModelInstanceStateEnum.ERROR
    assert 'Worker 2' in ctx.instance.state_message
    assert ctx.instance.restart_count == 0


@pytest.fixture
def hybrid_recovery(managed_recovery):
    from tests.utils.model import new_hybrid_workload

    ctx = managed_recovery
    ctx.model.backend = 'vLLM'
    ctx.model.backend_parameters = ['--data-parallel-hybrid-lb', '--tp=4', '--dp=2']
    ctx.instance.worker_ip = '10.0.0.1'
    ctx.instance.gpu_indexes = [0, 1]
    ctx.instance.distributed_servers.subordinate_workers = [
        ModelInstanceSubordinateWorker(
            worker_id=w, worker_ip=f'10.0.0.{w}', gpu_indexes=[0, 1]
        )
        for w in (2, 3, 4)
    ]
    ctx.instance.distributed_servers.workloads = [
        ModelInstanceWorkload.model_validate(
            new_hybrid_workload(w, headless=w in (2, 4), state='running')
        )
        for w in (1, 2, 3, 4)
    ]
    ctx.instance.distributed_servers.orchestration_mode = "managed"
    for task in ctx.instance.distributed_servers.workloads:
        if task.component.api_port:
            task.component.dp_size, task.component.ready = 1, True
    return ctx


@pytest.mark.asyncio
@pytest.mark.parametrize('worker_id', [1, 2, 3, 4])
@pytest.mark.parametrize('failure', ['failed', 'stopped'])
async def test_hybrid_any_required_task_failure_withdraws_all_apis(
    hybrid_recovery, worker_id, failure
):
    from gpustack.utils.model_instance_workers import model_instance_endpoints

    ctx = hybrid_recovery
    assert len(model_instance_endpoints(ctx.instance)) == 2
    task = ctx.instance.distributed_servers.workloads[worker_id - 1]
    task.state = ModelInstanceWorkloadStateEnum(failure)
    resources = ctx.instance.computed_resource_claim.model_copy(deep=True)
    await ctx.service.reconcile_managed_instance(ctx.instance.id, 0)
    assert ctx.instance.state == ModelInstanceStateEnum.ERROR
    assert f'Worker {worker_id}' in ctx.instance.state_message
    assert model_instance_endpoints(ctx.instance) == []
    assert ctx.instance.restart_count == 0
    assert len(ctx.instance.distributed_servers.workloads) == 4
    assert ctx.instance.computed_resource_claim == resources
    assert ctx.instance.ports == [40000]


@pytest.mark.asyncio
@pytest.mark.parametrize('worker_id', [1, 2, 3, 4])
@pytest.mark.parametrize('pending', ['pending', 'running', 'failed', 'unknown'])
async def test_hybrid_recovery_waits_for_every_task_then_advances_once(
    hybrid_recovery, worker_id, pending
):
    ctx = hybrid_recovery
    ctx.instance.state = ModelInstanceStateEnum.ERROR
    for task in ctx.instance.distributed_servers.workloads:
        task.state = ModelInstanceWorkloadStateEnum.STOPPED
        task.component.ready = False if task.component.api_port else None
    last = ctx.instance.distributed_servers.workloads[worker_id - 1]
    last.state = ModelInstanceWorkloadStateEnum(pending)
    resources = ctx.instance.computed_resource_claim.model_copy(deep=True)
    await ctx.service.reconcile_managed_instance(ctx.instance.id, 0)
    ctx.write.assert_not_awaited()
    assert ctx.instance.restart_count == 0
    last.state = ModelInstanceWorkloadStateEnum.STOPPED
    await ctx.service.reconcile_managed_instance(ctx.instance.id, 0)
    await ctx.service.reconcile_managed_instance(ctx.instance.id, 0)
    await ctx.service.reconcile_managed_instance(ctx.instance.id, 1)
    assert ctx.instance.restart_count == 1
    assert ctx.instance.state == ModelInstanceStateEnum.SCHEDULED
    assert ctx.instance.distributed_servers.workloads == []
    assert ctx.instance.computed_resource_claim == resources
    assert [
        w.worker_id for w in ctx.instance.distributed_servers.subordinate_workers
    ] == [2, 3, 4]
    ctx.write.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize('pd', [False, True])
async def test_hybrid_user_delete_keeps_ordinary_fast_path_and_pd_stop_barrier(
    hybrid_recovery, monkeypatch, pd
):
    from gpustack.server import services

    ctx = hybrid_recovery
    if pd:
        ctx.instance.role, ctx.instance.group_id = 'prefill', 'group'
    deleted = AsyncMock()
    monkeypatch.setattr(ModelInstance, 'delete', deleted)
    monkeypatch.setattr(ModelInstance, '_invalidate_cached_all', AsyncMock())
    monkeypatch.setattr(services, 'delete_cache_by_key', AsyncMock())
    monkeypatch.setattr(services, 'invalidate_workers_allocated', AsyncMock())

    async def apply(row, session, source=None, **kwargs):
        for key, value in (source or {}).items():
            setattr(row, key, value)

    monkeypatch.setattr(ModelInstance, 'update', apply)
    assert await ctx.service.delete(ctx.instance) is (not pd)
    assert deleted.await_count == int(not pd)
    if pd:
        assert ctx.instance.draining_since is not None
        assert ctx.instance.distributed_servers.stop_requested
        assert len(ctx.instance.distributed_servers.workloads) == 4
        for task in ctx.instance.distributed_servers.workloads:
            task.state = ModelInstanceWorkloadStateEnum.STOPPED
            task.component.ready = False if task.component.api_port else None
        assert await ctx.service.delete(ctx.instance)
        deleted.assert_awaited_once()
    else:
        assert ctx.instance.draining_since is None
        assert not ctx.instance.distributed_servers.stop_requested


@pytest.mark.asyncio
@pytest.mark.parametrize('stopped', [False, True])
async def test_ep_only_switches_to_single_entry_only_after_all_tasks_stop(
    managed_recovery, stopped
):
    from gpustack.schemas.models import BackendEnum, ModelInstanceComponent

    ctx = managed_recovery
    ctx.model.backend = BackendEnum.SGLANG
    ctx.model.backend_parameters = ['--ep-size', '2']
    ctx.instance.computed_resource_claim.ram = 3 * 1024**3
    ctx.instance.state = ModelInstanceStateEnum.ERROR
    for item in ctx.instance.distributed_servers.workloads:
        item.component = ModelInstanceComponent(kind='sglang_engine')
        item.dp_rank_placements = []
        item.state = ModelInstanceWorkloadStateEnum.STOPPED
    if not stopped:
        ctx.instance.distributed_servers.workloads[0].state = (
            ModelInstanceWorkloadStateEnum.RUNNING
        )
    await ctx.service.reconcile_managed_instance(ctx.instance.id, 0)
    if stopped:
        assert ctx.instance.distributed_servers.workloads is None
        assert ctx.instance.port is None and ctx.instance.ports is None
        assert ctx.instance.named_ports is None
        assert ctx.instance.computed_resource_claim.ram == 1024**3
        assert ctx.instance.restart_count == 1
    else:
        ctx.write.assert_not_awaited()
        assert ctx.instance.computed_resource_claim.ram == 3 * 1024**3
        assert ctx.instance.distributed_servers.workloads


@pytest.mark.asyncio
@pytest.mark.parametrize('edited', [None, 'primary', 'draft', 'remove_draft', 'lora'])
async def test_managed_recovery_prepares_files_for_the_adopted_spec(
    managed_recovery, monkeypatch, edited
):
    from gpustack.schemas.model_files import ModelFile
    from gpustack.schemas.models import (
        LoraListEntry,
        ModelSource,
        SpeculativeConfig,
        ModelConfigSnapshot,
    )
    from gpustack.schemas.cache_services import CacheConfigSnapshot
    from gpustack.schemas.workers import Worker
    from gpustack.server import controllers
    from sqlalchemy.orm.attributes import set_committed_value

    ctx = managed_recovery
    ctx.session.info = {}
    ctx.model.backend = 'vLLM'
    ctx.model.backend_parameters = ['--data-parallel-external-lb', '--tp=1', '--dp=2']
    monkeypatch.setattr(
        'gpustack.utils.model_source.get_catalog_draft_models',
        AsyncMock(return_value=[]),
    )
    worker = SimpleNamespace(id=ctx.instance.worker_id)
    monkeypatch.setattr(Worker, 'one_by_id', AsyncMock(return_value=worker))
    cache_config = AsyncMock(return_value=None)
    monkeypatch.setattr(
        'gpustack.server.services.resolve_instance_cache_config_safe', cache_config
    )
    ctx.instance.cache_config = CacheConfigSnapshot(cache_service_id=1, injected=True)
    ctx.instance.model_config_snapshot = ModelConfigSnapshot(
        backend_parameters=['--old']
    )
    ctx.model.speculative_config = SpeculativeConfig(draft_model='/models/draft-A')
    ctx.instance.source = ctx.model.source
    ctx.instance.huggingface_repo_id = ctx.model.huggingface_repo_id
    ctx.instance.resolved_path = '/models/primary-A'
    ctx.instance.draft_model_source = ModelSource(
        source='local_path', local_path='/models/draft-A'
    )
    ctx.instance.draft_model_resolved_path = '/models/draft-A'
    ctx.instance.download_progress = 100
    ctx.instance.draft_model_download_progress = 100
    ctx.instance.spec_digest = await controllers.model_spec_digest(
        ctx.session, ctx.model
    )
    old_files = [
        ModelFile(
            id=wid,
            worker_id=wid,
            source=ctx.model.source,
            huggingface_repo_id=ctx.model.huggingface_repo_id,
            state='ready',
            resolved_paths=['/models/primary-A'],
        )
        for wid in (1, 2)
    ]
    set_committed_value(ctx.instance, 'model_files', old_files)
    old_draft_files = [
        ModelFile(
            id=wid + 2,
            worker_id=wid,
            source='local_path',
            local_path='/models/draft-A',
            state='ready',
            resolved_paths=['/models/draft-A'],
        )
        for wid in (1, 2)
    ]
    set_committed_value(ctx.instance, 'draft_model_files', old_draft_files)
    if edited == 'primary':
        ctx.model.source = SourceEnum.LOCAL_PATH
        ctx.model.local_path = '/models/primary-B'
        ctx.model.huggingface_repo_id = None
    elif edited == 'draft':
        ctx.model.speculative_config = SpeculativeConfig(draft_model='/models/draft-B')
    elif edited == 'remove_draft':
        ctx.model.speculative_config = None
    elif edited == 'lora':
        ctx.model.backend_parameters.append('--enable-lora')
        ctx.model.lora_list = [
            LoraListEntry(
                lora_name='adapter',
                source='huggingface',
                lora_repo_name='org/adapter',
            )
        ]
    ctx.session.info.clear()
    ctx.instance.state = ModelInstanceStateEnum.ERROR
    for task in ctx.instance.distributed_servers.workloads:
        task.state = ModelInstanceWorkloadStateEnum.STOPPED

    await ctx.service.reconcile_managed_instance(ctx.instance.id, 0)

    assert ctx.instance.spec_digest == await controllers.model_spec_digest(
        ctx.session, ctx.model
    )
    if not edited:
        assert ctx.instance.model_files == old_files
        assert ctx.instance.resolved_path == '/models/primary-A'
        ctx.session.execute.assert_not_awaited()
        cache_config.assert_not_awaited()
        assert ctx.instance.cache_config.injected is True
        return
    assert ctx.instance.source == ctx.model.source
    assert ctx.instance.huggingface_repo_id == ctx.model.huggingface_repo_id
    assert ctx.instance.local_path == ctx.model.local_path
    assert ctx.instance.resolved_path is None
    assert ctx.instance.draft_model_resolved_path is None
    assert ctx.instance.download_progress is None
    assert ctx.instance.draft_model_download_progress is None
    assert ctx.instance.cache_config is None
    assert ctx.instance.model_config_snapshot is None
    cache_config.assert_awaited_once_with(
        ctx.session, ctx.model, worker=worker, spans_workers=True
    )
    assert ctx.instance.model_files == []
    assert ctx.instance.draft_model_files == []
    assert ctx.session.execute.await_count == 2
    from gpustack.schemas.links import (
        ModelInstanceModelFileLink,
        ModelInstanceDraftModelFileLink,
    )

    for call, link in zip(
        ctx.session.execute.await_args_list,
        (ModelInstanceModelFileLink, ModelInstanceDraftModelFileLink),
    ):
        statement = call.args[0]
        assert statement.table.name == link.__tablename__
        assert statement.compile().params == {'model_instance_id_1': ctx.instance.id}
    assert not controllers._instance_model_files_complete(ctx.instance, ctx.model)
    if edited == 'remove_draft':
        assert ctx.instance.draft_model_source is None
    else:
        assert ctx.instance.draft_model_source.local_path == (
            '/models/draft-B' if edited == 'draft' else '/models/draft-A'
        )
    primary_files = [
        ModelFile(
            id=wid + 4,
            worker_id=wid,
            source=ctx.model.source,
            huggingface_repo_id=ctx.model.huggingface_repo_id,
            local_path=ctx.model.local_path,
            state='ready',
            resolved_paths=[ctx.model.local_path or '/models/primary-A'],
        )
        for wid in (1, 2)
    ]
    draft_source = ctx.instance.draft_model_source
    draft_files = (
        [
            ModelFile(
                id=wid + 6,
                worker_id=wid,
                source='local_path',
                local_path=draft_source.local_path,
                state='ready',
                resolved_paths=[draft_source.local_path],
            )
            for wid in (1, 2)
        ]
        if draft_source
        else []
    )
    by_source = {ctx.instance.model_source_index: primary_files}
    if draft_source:
        by_source[draft_source.model_source_index] = draft_files
    monkeypatch.setattr(
        controllers.ModelFileService,
        'get_by_source_index',
        AsyncMock(side_effect=lambda index: by_source.get(index, [])),
    )
    monkeypatch.setattr(
        controllers.ModelFileService, 'get_by_resolved_path', AsyncMock(return_value=[])
    )
    create_file = AsyncMock()
    monkeypatch.setattr(ModelFile, 'create', create_file)
    primary_files = await controllers.get_or_create_model_files_for_instance(
        ctx.session, ctx.instance
    )
    if draft_source:
        draft_files = await controllers.get_or_create_model_files_for_instance(
            ctx.session, ctx.instance, is_draft_model=True
        )
    create_file.assert_not_awaited()
    set_committed_value(ctx.instance, 'model_files', primary_files)
    set_committed_value(ctx.instance, 'draft_model_files', draft_files)
    ctx.instance.state = ModelInstanceStateEnum.INITIALIZING
    monkeypatch.setattr(
        controllers, '_build_mounted_loras_payload', AsyncMock(return_value=([], []))
    )
    assert await controllers._promote_to_starting_if_complete(ctx.session, ctx.instance)
    assert ctx.instance.resolved_path == primary_files[0].resolved_paths[0]
    assert ctx.instance.draft_model_resolved_path == (
        draft_source.local_path if draft_source else None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('stale_sibling', [False, True])
@pytest.mark.parametrize('fence', [None, 'pd', 'own_stale'])
async def test_current_smg_recovers_despite_a_stale_sibling(
    smg_recovery, stale_sibling, fence
):
    from gpustack.server.controllers import model_members_stale, model_spec_digest

    ctx = smg_recovery
    ctx.session.info = {}
    ctx.model.backend = 'SGLang'
    ctx.model.backend_parameters = ['--dp-size=2']
    ctx.model.replicas = 2
    ctx.instance.spec_digest = await model_spec_digest(ctx.session, ctx.model)
    sibling = ctx.instance.model_copy(deep=True, update={'id': 2, 'spec_digest': 'old'})
    ctx.model.stale = await model_members_stale(
        ctx.session,
        ctx.model,
        [ctx.instance, sibling] if stale_sibling else [ctx.instance],
    )
    if fence == 'pd':
        ctx.instance.group_id = 'group'
    elif fence == 'own_stale':
        ctx.instance.spec_digest = 'old'
    smg = ctx.instance.distributed_servers.workloads[-1]
    smg.state = ModelInstanceWorkloadStateEnum.STOPPED
    smg.component.stop_requested = True
    await ctx.service.reconcile_managed_instance(ctx.instance.id, 0)
    expected = int(fence != 'own_stale' and not (fence == 'pd' and stale_sibling))
    assert ctx.instance.distributed_servers.workloads[-1].component.attempt == expected


@pytest.mark.asyncio
async def test_recovery_serializes_with_inflight_file_preparation(
    managed_recovery, monkeypatch
):
    from gpustack.schemas.model_files import ModelFile
    from gpustack.server import controllers
    from sqlalchemy.orm.attributes import set_committed_value

    ctx = managed_recovery
    ctx.session.info = {}
    ctx.model.backend = 'vLLM'
    ctx.model.backend_parameters = ['--data-parallel-external-lb', '--tp=1', '--dp=2']
    current = ctx.instance
    current.source = ctx.model.source
    current.huggingface_repo_id = ctx.model.huggingface_repo_id
    current.spec_digest = await controllers.model_spec_digest(ctx.session, ctx.model)
    current.state = ModelInstanceStateEnum.INITIALIZING
    current.distributed_servers.workloads = []
    preparing = current.model_copy(deep=True)
    files = [
        ModelFile(
            id=wid,
            worker_id=wid,
            source=current.source,
            huggingface_repo_id=current.huggingface_repo_id,
            state='ready',
            resolved_paths=['/models/model-A'],
        )
        for wid in (1, 2)
    ]
    lookup_started, resume_lookup, recovery_started = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
    lock = asyncio.Lock()
    held = set()
    writer = MagicMock()

    async def acquire(session):
        if session not in held:
            await lock.acquire()
            held.add(session)

    def release(session):
        if session in held:
            held.remove(session)
            lock.release()

    async def load(session, id, *, for_update=False, options=None, **kwargs):
        if session is ctx.session:
            recovery_started.set()
        if for_update:
            await acquire(session)
        return preparing if options is not None else current

    async def load_files(session, id, *, for_update=False):
        if for_update:
            await acquire(session)
        return current

    async def lookup(index):
        assert index == preparing.model_source_index
        lookup_started.set()
        await resume_lookup.wait()
        return files

    monkeypatch.setattr(ModelInstance, 'one_by_id', AsyncMock(side_effect=load))
    monkeypatch.setattr(
        ModelInstance, 'one_by_id_with_model_files', AsyncMock(side_effect=load_files)
    )
    monkeypatch.setattr(Model, 'one_by_id', AsyncMock(return_value=ctx.model))
    monkeypatch.setattr(
        'gpustack.server.services.Worker.one_by_id', AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        controllers.ModelFileService,
        'get_by_source_index',
        AsyncMock(side_effect=lookup),
    )
    writer.exec = AsyncMock(return_value=SimpleNamespace(first=lambda: None))
    pending = []
    writer.add.side_effect = pending.append

    async def commit_writer():
        if pending:
            set_committed_value(current, 'model_files', files)
            pending.clear()
        release(writer)

    async def commit_recovery():
        release(ctx.session)
        # A Worker can begin the replacement while the old preparation unwinds.
        current.state = ModelInstanceStateEnum.INITIALIZING

    writer.commit = AsyncMock(side_effect=commit_writer)
    ctx.session.commit.side_effect = commit_recovery

    task = asyncio.create_task(
        controllers.ensure_instance_model_file(writer, preparing)
    )
    await asyncio.wait_for(lookup_started.wait(), timeout=2)
    ctx.model.source = SourceEnum.LOCAL_PATH
    ctx.model.huggingface_repo_id = None
    ctx.model.local_path = '/models/model-B'
    ctx.session.info.clear()
    current.state = ModelInstanceStateEnum.ERROR
    recovery = asyncio.create_task(
        ctx.service.reconcile_managed_instance(current.id, 0)
    )
    try:
        await asyncio.wait_for(recovery_started.wait(), timeout=2)
        resume_lookup.set()
        await asyncio.wait_for(asyncio.gather(task, recovery), timeout=2)
    finally:
        resume_lookup.set()
        task.cancel()
        recovery.cancel()
        await asyncio.gather(task, recovery, return_exceptions=True)
        release(writer)
        release(ctx.session)
    assert current.restart_count == 1
    assert current.local_path == '/models/model-B'
    assert current.spec_digest == await controllers.model_spec_digest(
        ctx.session, ctx.model
    )
    assert current.resolved_path is None
    assert current.model_files == []
