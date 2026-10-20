from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import asyncio
import pytest

from gpustack.schemas.config import GatewayModeEnum
from gpustack.schemas.models import (
    DisaggregationSpec,
    Model,
    ModelInstanceStateEnum,
    RoleSpec,
    SourceEnum,
)
from gpustack.server import controllers, pd_membership, workqueue
from gpustack.server.bus import Event, EventType
from gpustack.server.controllers import sync_model_status


class RegistryResponse:
    def __init__(self, payload=None, status=200):
        self.payload = payload
        self.status = status

    async def json(self):
        return self.payload

    async def text(self):
        return ""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


@pytest.fixture
def recovery(monkeypatch, config):
    model = Model(
        id=4246,
        name="pd",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="Qwen/Qwen3-0.6B",
        roles=[RoleSpec(name=name) for name in ("prefill", "decode", "router")],
        disaggregation=DisaggregationSpec(mode="vllm-nixl"),
        owner_principal_id=1,
        cluster_id=1,
    )
    instances = [
        SimpleNamespace(
            id=i,
            role=role,
            state=ModelInstanceStateEnum.RUNNING,
            worker_id=1,
            worker_ip="10.0.0.1",
            port=40000 + i,
            restart_count=0,
            last_restart_time=None,
        )
        for i, role in enumerate(("prefill", "decode", "router"), start=1)
    ]
    state = SimpleNamespace(
        model=model,
        instances=instances,
        registry={},
        calls=[],
        now=100.0,
        admit=True,
        drain_due=None,
    )

    def request(method, url, **kwargs):
        state.calls.append((method, url))
        if method == "GET":
            return RegistryResponse({"workers": list(state.registry.values())})
        assert method == "POST"
        body = kwargs["json"]
        if state.admit:
            state.registry[body["url"]] = body
        return RegistryResponse(status=200 if state.admit else 503)

    reconcile = pd_membership.reconcile

    async def reconcile_with_client(*args, **kwargs):
        return await reconcile(*args, **kwargs, client=SimpleNamespace(request=request))

    async def sync_status(session, current):
        if current.disaggregation and current.disaggregation.mode.value == "vllm-nixl":
            await controllers._reconcile_router_membership(
                session, current, state.instances
            )

    @asynccontextmanager
    async def session():
        yield MagicMock()

    clock = SimpleNamespace(monotonic=lambda: state.now)
    monkeypatch.setattr(pd_membership, "time", clock)
    monkeypatch.setattr(workqueue, "time", clock)
    monkeypatch.setattr(pd_membership, "_ADMISSION_WINDOW_SECONDS", 0)
    monkeypatch.setattr(pd_membership, "reconcile", reconcile_with_client)
    monkeypatch.setattr(controllers, "async_session", session)
    monkeypatch.setattr(
        controllers, "_router_dial", AsyncMock(return_value=("10.0.0.1", None, None))
    )
    monkeypatch.setattr(
        Model, "one_by_id", AsyncMock(side_effect=lambda *args: state.model)
    )
    state.sync_replicas = AsyncMock(side_effect=lambda *args: state.drain_due)
    monkeypatch.setattr(controllers, "sync_replicas", state.sync_replicas)
    monkeypatch.setattr(controllers, "sync_model_status", sync_status)
    monkeypatch.setattr(controllers, "notify_model_route_target", AsyncMock())
    monkeypatch.setattr(controllers, "sync_categories_and_meta", AsyncMock())
    state.controller = controllers.ModelController(
        config.model_copy(update={"gateway_mode": GatewayModeEnum.disabled})
    )
    monkeypatch.setattr(state.controller, "_ensure_model_mcp_bridge", AsyncMock())
    monkeypatch.setattr(state.controller, "_sync_model_ai_proxy", AsyncMock())
    pd_membership.forget(model.id)
    yield state
    pd_membership.forget(model.id)


@pytest.mark.asyncio
@pytest.mark.parametrize("changed_field", ["restart_count", "last_restart_time"])
async def test_same_router_identity_replays_members_after_restart(
    recovery, changed_field
):
    state = recovery
    await controllers._reconcile_router_membership(None, state.model, state.instances)
    assert len(state.registry) == 2
    state.registry.clear()
    state.calls.clear()
    value = 1 if changed_field == "restart_count" else datetime.now(timezone.utc)
    setattr(state.instances[-1], changed_field, value)

    await controllers._reconcile_router_membership(None, state.model, state.instances)

    assert len(state.registry) == 2
    assert sum(method == "POST" for method, _ in state.calls) == 2
    assert controllers.upstream_registration_ready(state.model)


@pytest.mark.asyncio
async def test_empty_registry_recovers_without_another_bus_event(recovery):
    state = recovery
    await state.controller._reconcile(Event(type=EventType.UPDATED, data=state.model))
    assert len(state.registry) == 2
    state.registry.clear()
    state.now += pd_membership.RESETTLE_AFTER_SECONDS

    event = await asyncio.wait_for(state.controller._queue.get(), timeout=0.1)
    await state.controller._reconcile(event.object)

    assert len(state.registry) == 2
    assert controllers.upstream_registration_ready(state.model)


@pytest.mark.asyncio
async def test_failed_registration_retries_without_restarting_engines(recovery):
    state = recovery
    state.admit = False
    await state.controller._reconcile(Event(type=EventType.UPDATED, data=state.model))
    assert not controllers.upstream_registration_ready(state.model)
    state.admit = True
    state.now += 5

    event = await asyncio.wait_for(state.controller._queue.get(), timeout=0.1)
    await state.controller._reconcile(event.object)

    assert len(state.registry) == 2
    assert controllers.upstream_registration_ready(state.model)
    assert all(instance.restart_count == 0 for instance in state.instances)


@pytest.mark.asyncio
async def test_router_absence_invalidates_previous_registration(recovery):
    state = recovery
    router = state.instances[-1]
    await controllers._reconcile_router_membership(None, state.model, state.instances)
    state.registry.clear()
    router.state = ModelInstanceStateEnum.ERROR
    await controllers._reconcile_router_membership(None, state.model, state.instances)
    assert not controllers.upstream_registration_ready(state.model)

    router.state = ModelInstanceStateEnum.RUNNING
    await controllers._reconcile_router_membership(None, state.model, state.instances)

    assert len(state.registry) == 2
    assert controllers.upstream_registration_ready(state.model)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [None, "vllm-mooncake", "custom", "stopped"])
async def test_only_active_dynamic_deployments_schedule_membership_checks(
    recovery, mode
):
    state = recovery
    if mode == "stopped":
        state.model.replicas = 0
    else:
        state.model.disaggregation = DisaggregationSpec(mode=mode) if mode else None

    await state.controller._reconcile(Event(type=EventType.UPDATED, data=state.model))

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(state.controller._queue.get(), timeout=0.01)


@pytest.mark.asyncio
@pytest.mark.parametrize("drain_due", [2.0, 120.0])
async def test_membership_check_and_drain_share_the_earliest_timer(recovery, drain_due):
    state = recovery
    state.drain_due = drain_due
    await state.controller._reconcile(Event(type=EventType.UPDATED, data=state.model))
    state.now += min(drain_due, pd_membership.RESETTLE_AFTER_SECONDS)

    event = await asyncio.wait_for(state.controller._queue.get(), timeout=0.1)

    assert event.keys == (state.model.id,)


@pytest.mark.asyncio
@pytest.mark.parametrize("deleted", [False, True])
async def test_delayed_event_reads_current_deployment_before_reconciling(
    recovery, deleted
):
    state = recovery
    old = state.model.model_copy(deep=True)
    if deleted:
        state.model = None
    else:
        state.model.replicas = 0

    await state.controller._reconcile(Event(type=EventType.UPDATED, data=old))

    if deleted:
        state.sync_replicas.assert_not_called()
    else:
        assert state.sync_replicas.call_args.args[1].replicas == 0


@pytest.mark.asyncio
async def test_temporary_model_read_failure_keeps_a_retry(recovery, monkeypatch):
    state = recovery
    read = AsyncMock(side_effect=[RuntimeError("database unavailable"), state.model])
    monkeypatch.setattr(Model, "one_by_id", read)
    event = workqueue.WorkEvent(
        keys=(state.model.id,),
        type=workqueue.WorkEventType.MODIFIED,
        object=Event(type=EventType.UPDATED, data=state.model),
    )

    await state.controller._process(event)
    assert state.controller._queue.failures(event.keys) == 1
    state.now += 1
    retry = await asyncio.wait_for(state.controller._queue.get(), timeout=0.1)
    await state.controller._process(retry)

    assert len(state.registry) == 2
    assert state.controller._queue.failures(event.keys) == 0


@pytest.mark.parametrize(
    "failures, delay", [(1, 5.0), (2, 10.0), (12, 60.0), (30, 60.0)]
)
def test_membership_failure_retries_back_off_with_a_cap(recovery, failures, delay):
    for _ in range(failures):
        pd_membership.record(
            recovery.model.id,
            pd_membership.MembershipOutcome(ok=False, reason="offline"),
        )

    assert pd_membership.reconcile_after(recovery.model) == delay


def test_successful_membership_resets_failure_backoff(recovery):
    for _ in range(12):
        pd_membership.record(
            recovery.model.id, pd_membership.MembershipOutcome(ok=False)
        )
    pd_membership.record(recovery.model.id, pd_membership.MembershipOutcome(ok=True))
    pd_membership.record(recovery.model.id, pd_membership.MembershipOutcome(ok=False))

    assert pd_membership.reconcile_after(recovery.model) == 5.0


@pytest.mark.asyncio
async def test_persistent_membership_failure_does_not_self_trigger_a_loop(
    recovery, monkeypatch
):
    state = recovery
    state.admit = False
    for instance in state.instances:
        instance.draining_since = None
        instance.cache_config = None
    for _ in range(pd_membership.PERSISTENT_FAILURE_PASSES - 1):
        pd_membership.record(state.model.id, pd_membership.MembershipOutcome(ok=False))

    async def degradation_reasons(*args, **kwargs):
        return [], kwargs["state_message"]

    async def publish_status(current):
        state.controller._queue.add(
            workqueue.WorkEvent(
                keys=(current.id,),
                type=workqueue.WorkEventType.MODIFIED,
                object=Event(
                    type=EventType.UPDATED, data=current.model_copy(deep=True)
                ),
            )
        )

    update = AsyncMock(side_effect=publish_status)
    monkeypatch.setattr(controllers, "sync_model_status", sync_model_status)
    monkeypatch.setattr(
        controllers.ModelInstance,
        "all_by_field",
        AsyncMock(return_value=state.instances),
    )
    monkeypatch.setattr(
        controllers, "ModelService", lambda session: SimpleNamespace(update=update)
    )
    monkeypatch.setattr(controllers, "_degradation_reasons", degradation_reasons)
    monkeypatch.setattr(
        controllers, "model_members_stale", AsyncMock(return_value=False)
    )
    monkeypatch.setattr(controllers, "reconcile_route_target_states", AsyncMock())

    await state.controller._reconcile(Event(type=EventType.UPDATED, data=state.model))
    event = await asyncio.wait_for(state.controller._queue.get(), timeout=0.1)
    await state.controller._process(event)

    assert not controllers.upstream_registration_ready(state.model)
    assert update.call_count == 1
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(state.controller._queue.get(), timeout=0.01)

    state.now += pd_membership.reconcile_after(state.model)
    retry = await asyncio.wait_for(state.controller._queue.get(), timeout=0.1)
    await state.controller._process(retry)

    assert pd_membership.consecutive_failures(state.model.id) == 7
    assert update.call_count == 1
