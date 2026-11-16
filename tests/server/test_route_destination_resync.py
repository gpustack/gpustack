import asyncio
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gpustack.gateway import utils as mcp_handler
from gpustack.schemas.model_routes import ModelRoute, ModelRouteTarget, TargetStateEnum
from gpustack.schemas.models import ModelInstanceStateEnum
from gpustack.server.bus import Event, EventType
from gpustack.server.controllers import (
    ModelRouteController,
    _notify_model_instance_endpoints,
    notify_model_route_target,
)
from tests.utils.model import new_model, new_model_instance

MODEL_ID = 5
CLUSTER_ID = 3


def _target(id: int):
    return ModelRouteTarget(
        id=id,
        route_id=id,
        name=f"t-{id}",
        route_name=f"r-{id}",
        model_id=MODEL_ID,
        weight=1,
        state=TargetStateEnum.ACTIVE,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changed_fields, should_notify",
    [
        # The coordinator drops ``changed_fields`` on the wire and
        # ``Event`` defaults it to an empty dict, and the status
        # change-gate has already absorbed the diff by the time the Model
        # controller sees the event -- a bare event (None or empty) must
        # still wake the route or the destination stays pinned to the old
        # member set.
        (None, True),
        ({}, True),
        ({"description": (None, "x")}, False),
        ({"ready_replicas": (1, 2)}, True),
    ],
)
async def test_bare_model_event_still_notifies_route_targets(
    changed_fields, should_notify
):
    model = new_model(MODEL_ID, "base", huggingface_repo_id="repo/base")

    publish = AsyncMock()
    with (
        patch(
            "gpustack.server.controllers.Model.one_by_id",
            AsyncMock(return_value=model),
        ),
        patch(
            "gpustack.server.controllers.ModelRouteTarget.all_by_fields",
            AsyncMock(return_value=[_target(1)]),
        ),
        patch("gpustack.server.controllers.event_bus.publish", publish),
    ):
        await notify_model_route_target(
            session=None,
            model=model,
            event=Event(
                type=EventType.UPDATED,
                data=model,
                changed_fields=changed_fields,
            ),
        )

    assert publish.await_count == (1 if should_notify else 0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changed_fields, should_notify",
    [
        # A rolling replacement swaps the member address while
        # ``ready_replicas`` stays constant; plain TP replicas carry no
        # workloads and must not be skipped on that account.
        ({"worker_ip": ("10.0.0.1", "10.0.0.2")}, True),
        (
            {
                "state": (
                    ModelInstanceStateEnum.STARTING,
                    ModelInstanceStateEnum.RUNNING,
                )
            },
            True,
        ),
        (None, True),
        ({"name": ("old", "new")}, False),
    ],
)
async def test_plain_replica_endpoint_notify(changed_fields, should_notify):
    model = new_model(MODEL_ID, "base", huggingface_repo_id="repo/base")
    instance = new_model_instance(
        1,
        "instance-1",
        MODEL_ID,
        state=ModelInstanceStateEnum.RUNNING,
    )

    notify = AsyncMock()
    with patch("gpustack.server.controllers.notify_model_route_target", notify):
        await _notify_model_instance_endpoints(
            session=None,
            model=model,
            event=Event(
                type=EventType.UPDATED,
                data=instance,
                changed_fields=changed_fields,
            ),
        )

    assert notify.await_count == (1 if should_notify else 0)
    if should_notify:
        assert "instance_endpoints" in notify.await_args.kwargs["event"].changed_fields


class _NullSession:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *args):
        return False


@pytest.mark.asyncio
async def test_resync_reconciles_every_live_route():
    live = ModelRoute(id=1, name="live-route")
    deleted = ModelRoute(id=2, name="gone-route", deleted_at="2026-01-01")
    # A route created after the sweep snapshot but before the orphan
    # cleanup: its ingress was already written by the event path, so the
    # cleanup must see its name in the expected list.
    created_mid_sweep = ModelRoute(id=3, name="mid-sweep-route")

    reconcile = AsyncMock()
    cleanups = {
        name: AsyncMock()
        for name in (
            "cleanup_ingresses",
            "cleanup_fallback_filters",
            "cleanup_model_mapper",
            "cleanup_generic_proxy_router",
        )
    }
    controller = ModelRouteController.__new__(ModelRouteController)
    controller._reconcile = reconcile
    controller._reconcile_lock = asyncio.Lock()
    controller._config = SimpleNamespace(
        get_namespace=lambda: "gpustack",
        gateway_namespace="higress-system",
    )
    controller._k8s_config = object()
    controller._networking_api = object()
    controller._networking_istio_api = object()
    controller._higress_extension_api = object()

    sweep_calls = 0

    async def all_by_fields(session, fields=None, options=None):
        nonlocal sweep_calls
        sweep_calls += 1
        if sweep_calls == 1:
            # Id snapshot that seeds the sweep.
            return [live, deleted]
        # The orphan cleanup's own fresh query.
        return [live, deleted, created_mid_sweep]

    async def one_by_id(session, route_id, options=None):
        # The soft-deleted row reads back gone; the live one is re-read
        # under the lock so a mid-sweep edit would be written with its
        # new values, not the snapshot's.
        return live if route_id == live.id else None

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "gpustack.server.controllers.async_session",
                return_value=_NullSession(),
            )
        )
        stack.enter_context(
            patch(
                "gpustack.server.controllers.ModelRoute.all_by_fields",
                all_by_fields,
            )
        )
        stack.enter_context(
            patch(
                "gpustack.server.controllers.ModelRoute.one_by_id",
                one_by_id,
            )
        )
        for name, mock in cleanups.items():
            stack.enter_context(
                patch(f"gpustack.server.controllers.mcp_handler.{name}", mock)
            )
        await controller._resync_all_routes()

    # Only the route whose fresh read still resolves is reconciled, and
    # through the ordinary path so the diff-gated writes apply.
    assert reconcile.await_count == 1
    event = reconcile.await_args.args[0]
    assert event.type == EventType.UPDATED and event.data.id == live.id

    # The soft-deleted route's artifacts are retired, and the route
    # created mid-sweep is spared: the expected names come from a fresh
    # query, not the pre-sweep snapshot. Every live route contributes its
    # fallback companion name too, so a configured fallback ingress is
    # not swept away as an orphan.
    def _expected(*route_ids):
        names = [mcp_handler.model_route_ingress_name(i) for i in route_ids]
        return names + [mcp_handler.fallback_ingress_name(n) for n in names]

    assert cleanups["cleanup_ingresses"].await_count == 1
    assert cleanups["cleanup_ingresses"].await_args.kwargs[
        "expected_names"
    ] == _expected(live.id, created_mid_sweep.id)
    assert cleanups["cleanup_fallback_filters"].await_count == 1
    assert cleanups["cleanup_model_mapper"].await_count == 1
    assert cleanups["cleanup_generic_proxy_router"].await_count == 1
