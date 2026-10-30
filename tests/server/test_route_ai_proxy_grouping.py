from unittest.mock import AsyncMock, patch

import pytest

from gpustack.schemas.clusters import Cluster
from gpustack.schemas.config import ModelInstanceProxyModeEnum
from gpustack.schemas.model_routes import ModelRoute, ModelRouteTarget, TargetStateEnum
from gpustack.schemas.models import ModelInstanceStateEnum, RoleNameEnum, RoleSpec
from gpustack.schemas.workers import Worker, WorkerStateEnum, WorkerStatus
from gpustack.server.bus import Event, EventType
from gpustack.server.controllers import (
    WorkerController,
    calculate_destinations,
    notify_model_ai_proxy_change,
    notify_model_route_target,
    sync_model_ai_proxy,
)
from tests.utils.model import new_model, new_model_instance

CLUSTER_ID = 3
MODEL_ID = 5


class _Registry:
    def __init__(self, name):
        self.name = name

    def get_service_name(self):
        return self.name


def _cluster(id: int = CLUSTER_ID, token: str = "cluster-token") -> Cluster:
    # ``system_principal_id`` stays None so ``get_cluster_registry`` returns
    # None and destinations are built from the deployment's own instances.
    return Cluster(id=id, name=f"cluster-{id}", registration_token=token)


def _running_instance(id: int, model_id: int = MODEL_ID):
    instance = new_model_instance(
        id,
        f"instance-{id}",
        model_id,
        worker_id=id,
        state=ModelInstanceStateEnum.RUNNING,
    )
    instance.worker_ip = f"10.0.0.{id}"
    instance.port = 8000
    return instance


def _target(id: int, overridden_model_name=None, weight: int = 1, model_id=MODEL_ID):
    return ModelRouteTarget(
        id=id,
        route_id=id,
        name=f"t-{id}",
        route_name=f"r-{id}",
        model_id=model_id,
        overridden_model_name=overridden_model_name,
        weight=weight,
        state=TargetStateEnum.ACTIVE,
    )


def _model(model_id: int = MODEL_ID):
    model = new_model(model_id, "base", huggingface_repo_id="repo/base")
    model.cluster_id = CLUSTER_ID
    return model


def _destinations(*services):
    return [(1, "base", _Registry(s)) for s in services]


class _Cfg:
    gateway_namespace = "higress-system"


async def _run_sync(monkeypatch, model, targets, destinations_by_override=None):
    """Run sync_model_ai_proxy with the model and its referencing targets
    mocked; returns the (providers, rules, owned) handed to the CR diff."""
    captured = {}

    async def fake_ensure(api, name, namespace, spec_diff, extra_labels=None):
        captured["name"] = name
        captured["kwargs"] = dict(spec_diff.keywords)

    async def fake_destinations(session, model, overridden_model_name=None):
        if destinations_by_override is not None:
            return destinations_by_override.get(overridden_model_name, [])
        return _destinations("model-5-1.static")

    with (
        patch(
            "gpustack.server.controllers.Model.one_by_id",
            AsyncMock(return_value=model),
        ),
        patch(
            "gpustack.server.controllers.ModelRouteTarget.all_by_field",
            AsyncMock(return_value=targets),
        ),
        patch(
            "gpustack.server.controllers.calculate_model_destinations",
            fake_destinations,
        ),
        patch(
            "gpustack.server.controllers.Cluster.one_by_id",
            AsyncMock(return_value=_cluster()),
        ),
        patch(
            "gpustack.server.controllers.mcp_handler.ensure_wasm_plugin",
            side_effect=fake_ensure,
        ),
    ):
        await sync_model_ai_proxy(
            cfg=_Cfg(), session=None, extensions_api=object(), model_id=model.id
        )
    return captured


@pytest.mark.asyncio
async def test_lora_names_never_enter_the_rule(monkeypatch):
    """Model names a deployment serves — LoRA aliases, overrides — are
    expressed on the mapper/lb CR (candidate.modelName, modelMappers),
    never as service names: the rule carries the deployment's own
    registries, identical whatever the targets alias."""
    model = _model()
    targets = [
        _target(1),
        _target(2, overridden_model_name="base:adapter-a"),
        _target(3, overridden_model_name="base:adapter-b"),
    ]

    captured = await _run_sync(monkeypatch, model, targets)

    (rule,) = captured["kwargs"]["expected_match_rules"]
    assert rule.config["activeProviderId"] == "gpustack-model-5"
    assert set(rule.service) == {"model-5-1.static"}
    # legacy per-route ids never ride this write — the startup cleanup
    # pass owns their retirement
    assert captured["kwargs"]["owned_provider_ids"] == {"gpustack-model-5"}
    (provider,) = captured["kwargs"]["expected_providers"]
    assert provider["apiTokens"] == ["cluster-token"]


@pytest.mark.asyncio
async def test_alias_target_alone_gates_the_rule(monkeypatch):
    """The route reference read is an existence gate: a single aliased
    target referencing the deployment is enough for the rule to exist,
    and the rule it produces is the deployment's base services."""
    model = _model()
    targets = [_target(2, overridden_model_name="base:adapter-a")]

    captured = await _run_sync(monkeypatch, model, targets)

    (rule,) = captured["kwargs"]["expected_match_rules"]
    assert set(rule.service) == {"model-5-1.static"}


@pytest.mark.asyncio
async def test_no_live_reference_strips_the_rule(monkeypatch):
    """No live target (or none ACTIVE) leaves the expected rule set empty
    while the provider id stays owned — the write strips the deployment's
    rule and the provider goes unreferenced."""
    model = _model()
    inactive = _target(1)
    inactive.state = TargetStateEnum.UNAVAILABLE

    captured = await _run_sync(monkeypatch, model, [inactive])

    assert captured["kwargs"]["expected_match_rules"] == []
    assert captured["kwargs"]["expected_providers"] == []
    assert captured["kwargs"]["owned_provider_ids"] == {"gpustack-model-5"}


@pytest.mark.asyncio
async def test_native_anthropic_api_comes_from_the_deployment(monkeypatch):
    """The selector is read straight off the Model -- no lookup, and each
    deployment answers for itself."""
    model = _model()
    model.native_anthropic_api = True

    captured = await _run_sync(monkeypatch, model, [_target(1)])

    (provider,) = captured["kwargs"]["expected_providers"]
    assert "anthropic/v1/messages" in provider["capabilities"]


@pytest.mark.asyncio
async def test_notify_model_ai_proxy_change_publishes_model_events(monkeypatch):
    """Reference transitions enqueue the affected models: a live model
    gets a Model event (the Model controller owns the ai-proxy rebuild);
    a deleted model needs none — its own delete event strips the entry."""
    live_model = _model()
    publish = AsyncMock()

    async def fake_one_by_id(session, id, **kw):
        return live_model if id == MODEL_ID else None

    with (
        patch(
            "gpustack.server.controllers.Model.one_by_id",
            new=fake_one_by_id,
        ),
        patch("gpustack.server.controllers.event_bus.publish", publish),
    ):
        await notify_model_ai_proxy_change(session=None, model_ids={MODEL_ID, 404})

    assert publish.await_count == 1
    topic, event = publish.await_args.args
    assert topic == "model"
    assert event.data.id == MODEL_ID


@pytest.mark.asyncio
async def test_all_zero_weight_targets_still_yield_destinations():
    """Weight 0 on every target is the LB scoring / round-robin mode, not
    "no traffic": the gateway plugin selects among candidates, but the
    route's ingress must still exist, so the registries flatten to an
    equal placeholder share instead of being dropped entirely (a plain
    flatten yields no destinations and the ingress is deleted)."""
    model = _model()

    with (
        patch(
            "gpustack.server.controllers.ModelRouteTarget.all_by_field",
            AsyncMock(return_value=[_target(1, weight=0), _target(2, weight=0)]),
        ),
        patch(
            "gpustack.server.controllers.Model.one_by_id",
            AsyncMock(return_value=model),
        ),
        patch(
            "gpustack.server.controllers.Cluster.one_by_id",
            AsyncMock(return_value=_cluster()),
        ),
        patch(
            "gpustack.server.controllers.ModelInstance.all_by_field",
            AsyncMock(return_value=[_running_instance(1), _running_instance(2)]),
        ),
        patch(
            "gpustack.server.controllers.Worker.all_by_fields",
            AsyncMock(return_value=[]),
        ),
    ):
        destinations, _ = await calculate_destinations(
            session=None, model_route=ModelRoute(id=1, name="base")
        )

    # Both targets share the deployment's two instances (4 registry
    # entries in all): an equal 25% split, not an empty list.
    assert [weight for weight, _, _ in destinations] == [25, 25, 25, 25]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changed_field, should_notify",
    [
        ("native_anthropic_api", True),
        ("replicas", True),
        ("description", False),
        ("worker_proxy_address", True),
        ("worker_proxy_mode", True),
    ],
)
async def test_native_anthropic_api_edit_reaches_the_route(
    changed_field, should_notify
):
    """``native_anthropic_api`` is consumed where the route's provider entry is
    built, and that only runs off a route event -- so the Model controller has
    to treat it as route-affecting or flipping the selector would change nothing
    until the deployment happened to scale."""
    model = new_model(MODEL_ID, "base", huggingface_repo_id="repo/base")
    model.cluster_id = CLUSTER_ID
    hydrated = new_model(MODEL_ID, "base", huggingface_repo_id="repo/base")

    publish = AsyncMock()
    with (
        patch(
            "gpustack.server.controllers.Model.one_by_id",
            AsyncMock(return_value=hydrated),
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
                changed_fields={changed_field: (None, "x")},
            ),
        )

    assert publish.await_count == (1 if should_notify else 0)


class _NullSession:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *args):
        return False


@pytest.mark.asyncio
@pytest.mark.parametrize('field', ['proxy_address', 'state', 'unreachable', 'labels'])
async def test_worker_recovery_fanout_wakes_models_and_rebuilds_tunnel_routes(field):
    """A tunnel worker's proxy address coming back is the last event of the
    outage ordering: the instances are RUNNING again and ``ready_replicas``
    has already settled, so the fan-out from ``_notify_relatives`` is the
    only trigger left that can rebuild the gateway route. The
    route-target notification is made directly rather than riding the model
    event, whose ``changed_fields`` the coordinator drops on the wire -- a
    bare event is filtered out by ``notify_model_route_target`` and the
    route stays missing indefinitely."""
    worker = Worker(
        id=1,
        name="tunnel-worker",
        state=WorkerStateEnum.READY,
        hostname="host-b",
        ip="10.0.0.2",
        ifname="eth0",
        port=10128,
        worker_uuid="uuid-b",
        status=WorkerStatus.get_default_status(),
        proxy_mode=ModelInstanceProxyModeEnum.TUNNEL,
        proxy_address="http://10.0.0.2:10129",
    )
    instance = new_model_instance(1, "instance-1", MODEL_ID, worker_id=1)
    instance.model = _model()

    publish = AsyncMock()
    with (
        patch(
            "gpustack.server.controllers.async_session",
            return_value=_NullSession(),
        ),
        patch(
            "gpustack.server.controllers.ModelInstance.all_by_fields",
            AsyncMock(return_value=[instance]),
        ),
        patch(
            'gpustack.server.controllers.Model.all_by_fields',
            AsyncMock(return_value=[_model()]),
        ),
        patch(
            "gpustack.server.controllers.Model.one_by_id",
            AsyncMock(return_value=_model()),
        ),
        patch(
            "gpustack.server.controllers.ModelRouteTarget.all_by_fields",
            AsyncMock(return_value=[_target(1)]),
        ),
        patch("gpustack.server.controllers.event_bus.publish", publish),
    ):
        controller = WorkerController.__new__(WorkerController)
        await controller._notify_relatives(
            Event(
                type=EventType.UPDATED,
                data=worker,
                changed_fields={
                    field: {
                        'proxy_address': (None, worker.proxy_address),
                        'state': (WorkerStateEnum.UNREACHABLE, WorkerStateEnum.READY),
                        'unreachable': (True, False),
                        'labels': ({}, {'env': 'test'}),
                    }[field]
                },
            )
        )

    # One bare model event per model on the worker, plus the direct
    # route-target notification that does not depend on ``changed_fields``
    # surviving the event bus.
    assert publish.await_count == (
        2 if field == 'proxy_address' else 0 if field == 'labels' else 1
    )
    if field != 'labels':
        assert publish.await_args_list[0].args[1].data.id == MODEL_ID
    if field == 'proxy_address':
        assert "model" in publish.await_args_list[1].args[1].changed_fields


@pytest.mark.asyncio
async def test_pd_group_weights_only_the_router():
    """A role-bearing group splits no traffic onto its prefill or decode.

    The registry side (``_ensure_model_mcp_bridge``) already narrows to the
    router, so a weight naming a GPU member points at an upstream Envoy has no
    cluster for. Measured on a 1P1D Ascend group: the annotation read
    ``34% router / 33% prefill / 33% decode``, the router's third answered
    correctly and the other two never returned at all.
    """
    model = new_model(MODEL_ID, "asc-pd", model_scope_model_id="Qwen/Qwen3-30B-A3B")
    model.cluster_id = CLUSTER_ID
    model.roles = [
        RoleSpec(name=RoleNameEnum.PREFILL.value, replicas=1),
        RoleSpec(name=RoleNameEnum.DECODE.value, replicas=1),
        RoleSpec(name=RoleNameEnum.ROUTER.value, replicas=1),
    ]
    prefill = _running_instance(1)
    prefill.role = RoleNameEnum.PREFILL.value
    decode = _running_instance(2)
    decode.role = RoleNameEnum.DECODE.value
    router = _running_instance(3)
    router.role = RoleNameEnum.ROUTER.value

    with (
        patch(
            "gpustack.server.controllers.ModelRouteTarget.all_by_field",
            AsyncMock(return_value=[_target(1)]),
        ),
        patch(
            "gpustack.server.controllers.Model.one_by_id",
            AsyncMock(return_value=model),
        ),
        patch(
            "gpustack.server.controllers.Cluster.one_by_id",
            AsyncMock(return_value=_cluster()),
        ),
        patch(
            "gpustack.server.controllers.ModelInstance.all_by_field",
            AsyncMock(return_value=[prefill, decode, router]),
        ),
        patch(
            "gpustack.server.controllers.Worker.all_by_fields",
            AsyncMock(return_value=[]),
        ),
    ):
        destinations, _ = await calculate_destinations(
            session=None, model_route=ModelRoute(id=1, name="asc-pd")
        )

    assert [registry.get_service_name() for _, _, registry in destinations] == [
        f"model-{MODEL_ID}-3.static"
    ]


@pytest.mark.asyncio
async def test_group_without_a_running_router_weights_nothing():
    """No router running means no destination, not a fall back to the members.

    There is no member of a group that answers a whole request on its own, so
    an empty destination list is the honest answer and the route reports the
    deployment as unavailable.
    """
    model = new_model(MODEL_ID, "asc-pd", model_scope_model_id="Qwen/Qwen3-30B-A3B")
    model.cluster_id = CLUSTER_ID
    model.roles = [
        RoleSpec(name=RoleNameEnum.PREFILL.value, replicas=1),
        RoleSpec(name=RoleNameEnum.DECODE.value, replicas=1),
        RoleSpec(name=RoleNameEnum.ROUTER.value, replicas=1),
    ]
    prefill = _running_instance(1)
    prefill.role = RoleNameEnum.PREFILL.value
    decode = _running_instance(2)
    decode.role = RoleNameEnum.DECODE.value

    with (
        patch(
            "gpustack.server.controllers.ModelRouteTarget.all_by_field",
            AsyncMock(return_value=[_target(1)]),
        ),
        patch(
            "gpustack.server.controllers.Model.one_by_id",
            AsyncMock(return_value=model),
        ),
        patch(
            "gpustack.server.controllers.Cluster.one_by_id",
            AsyncMock(return_value=_cluster()),
        ),
        patch(
            "gpustack.server.controllers.ModelInstance.all_by_field",
            AsyncMock(return_value=[prefill, decode]),
        ),
        patch(
            "gpustack.server.controllers.Worker.all_by_fields",
            AsyncMock(return_value=[]),
        ),
    ):
        destinations, fallback = await calculate_destinations(
            session=None, model_route=ModelRoute(id=1, name="asc-pd")
        )

    assert destinations == []
    assert fallback == []


@pytest.mark.asyncio
@pytest.mark.parametrize('deleted', [False, True])
async def test_model_declares_smg_scope_before_registry_publication(
    monkeypatch, deleted
):
    from unittest.mock import MagicMock
    from gpustack.server import controllers

    model = _model()
    monkeypatch.setattr(
        controllers,
        '_model_gateway_registry_inputs',
        AsyncMock(return_value=([], {}, [])),
    )
    monkeypatch.setattr(
        controllers.mcp_handler,
        'sglang_smg_service_names',
        lambda *args: {'smg.static'},
    )
    order = []
    declare = AsyncMock(side_effect=lambda **kwargs: order.append('declare'))
    monkeypatch.setattr(
        controllers.mcp_handler, 'ensure_sglang_smg_transformer', declare
    )
    monkeypatch.setattr(
        controllers.mcp_handler,
        'ensure_model_mcp_bridge',
        AsyncMock(side_effect=lambda **kwargs: order.append('registry')),
    )
    controller = controllers.ModelController.__new__(controllers.ModelController)
    controller._disable_gateway = False
    controller._config = MagicMock()
    controller._higress_extension_api = object()
    controller._higress_network_api = object()
    monkeypatch.setattr(controllers, 'notify_model_route_target', AsyncMock())
    monkeypatch.setattr(controllers, 'sync_categories_and_meta', AsyncMock())
    controller._sync_model_ai_proxy = AsyncMock()
    await controller._sync_model_gateway(
        None,
        model,
        Event(type=EventType.DELETED if deleted else EventType.UPDATED, data=model),
    )
    assert order == ['declare', 'registry']
    assert declare.call_args.kwargs['service_names'] == (
        set() if deleted else {'smg.static'}
    )


@pytest.mark.asyncio
async def test_route_declares_only_local_smg_models_with_their_complete_services(
    monkeypatch,
):
    from gpustack.server import controllers

    model = _model(model_id=1)
    monkeypatch.setattr(controllers.Model, 'one_by_id', AsyncMock(return_value=model))
    inputs = AsyncMock(return_value=([], {}, []))
    monkeypatch.setattr(controllers, '_model_gateway_registry_inputs', inputs)
    monkeypatch.setattr(
        controllers.mcp_handler,
        'sglang_smg_service_names',
        lambda *args: {'model-1-7-r3-smg-a0.static', 'model-1-7-r3-smg-a0-l123.static'},
    )
    declare = AsyncMock()
    monkeypatch.setattr(
        controllers.mcp_handler, 'ensure_sglang_smg_transformer', declare
    )
    destinations = [
        (1, 'alias', _Registry(name))
        for name in (
            'model-1-7-r3-smg-a0.static',
            'model-1-7-r3-smg-a0-l123.static',
            'model-2-8-r0-dp0.static',
            'cluster-gateway.static',
            'model-1-7-r3-engine-dp0.static',
        )
    ]
    await controllers._ensure_route_smg_transformers(
        None, _Cfg(), object(), destinations
    )
    declare.assert_awaited_once()
    assert declare.call_args.kwargs['model_id'] == 1
    assert len(declare.call_args.kwargs['service_names']) == 2
    inputs.assert_awaited_once()


@pytest.mark.asyncio
async def test_smg_declaration_conflict_requeues_model(monkeypatch):
    import asyncio
    from unittest.mock import MagicMock
    from kubernetes_asyncio.client import ApiException
    from gpustack.server import controllers
    from gpustack.server.workqueue import WorkEvent, WorkEventType

    controller = controllers.ModelController.__new__(controllers.ModelController)
    controller._passes = asyncio.Semaphore(1)
    controller._queue = MagicMock()
    controller._inflight = {(1,): object()}
    controller._reconcile = AsyncMock(side_effect=ApiException(status=409))
    event = WorkEvent(
        keys=(1,),
        type=WorkEventType.MODIFIED,
        object=Event(type=EventType.UPDATED, data=_model(model_id=1)),
    )
    await controller._process(event)
    controller._queue.add_rate_limited.assert_called_once_with(event)
    controller._queue.forget.assert_not_called()
    controller._queue.done.assert_called_once_with((1,))


@pytest.mark.asyncio
async def test_sync_gateway_declares_smg_before_dispatching_candidates(monkeypatch):
    from unittest.mock import MagicMock
    from gpustack.server import controllers

    route = ModelRoute(id=1, name='r')
    destinations = [(1, 'm', _Registry('model-1-7-r3-smg-a0.static'))]
    monkeypatch.setattr(
        controllers.ModelRoute, 'one_by_id', AsyncMock(return_value=route)
    )
    monkeypatch.setattr(
        controllers,
        'calculate_destinations',
        AsyncMock(return_value=(destinations, [])),
    )
    monkeypatch.setattr(
        controllers.Principal, 'one_by_id', AsyncMock(return_value=None)
    )
    order = []
    declare = AsyncMock(side_effect=lambda *args: order.append('declare'))
    monkeypatch.setattr(controllers, '_ensure_route_smg_transformers', declare)
    monkeypatch.setattr(
        controllers,
        'dispatch_route_reconcile',
        AsyncMock(side_effect=lambda *args: order.append('candidates')),
    )
    monkeypatch.setattr(controllers.mcp_handler, 'ensure_model_ingress', AsyncMock())
    monkeypatch.setattr(
        controllers, 'ensure_route_generic_proxy_router_config', AsyncMock()
    )
    await controllers.sync_gateway(
        None,
        Event(type=EventType.UPDATED, data=route),
        MagicMock(),
        route,
        object(),
        object(),
        object(),
    )
    assert order == ['declare', 'candidates']
    assert declare.call_args.args[-1] == destinations


@pytest.mark.asyncio
async def test_model_controller_start_runs_orphan_cleanup_before_subscription(
    monkeypatch,
    config,
):
    from unittest.mock import MagicMock
    from gpustack.server import controllers

    order = []
    monkeypatch.setattr(controllers, 'async_session', lambda: _NullSession())
    monkeypatch.setattr(controllers, 'get_async_k8s_config', MagicMock())
    monkeypatch.setattr(controllers.k8s_client, 'ApiClient', MagicMock())
    monkeypatch.setattr(controllers, 'NetworkingHigressIoV1Api', MagicMock())
    monkeypatch.setattr(controllers, 'ExtensionsHigressIoV1Api', MagicMock())
    monkeypatch.setattr(
        controllers.Model, 'all_by_fields', AsyncMock(return_value=[_model(model_id=1)])
    )
    cleanup = AsyncMock(side_effect=lambda **kwargs: order.append('cleanup'))
    monkeypatch.setattr(
        controllers.mcp_handler, 'cleanup_sglang_smg_transformer', cleanup
    )

    async def subscribe(**kwargs):
        order.append('subscribe')
        if False:
            yield

    monkeypatch.setattr(controllers.Model, 'subscribe', subscribe)
    controller = controllers.ModelController(config)
    controller._disable_gateway = False
    controller._dispatch = AsyncMock()
    await controller.start()
    assert order == ['cleanup', 'subscribe']
    assert cleanup.call_args.kwargs['live_model_ids'] == {1}
    assert not controller._smg_cleanup_pending


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['model_gone', 'new_attempt'])
async def test_route_does_not_publish_smg_candidate_outside_current_declaration(
    monkeypatch, change
):
    from gpustack.server import controllers

    monkeypatch.setattr(
        controllers.Model,
        'one_by_id',
        AsyncMock(return_value=None if change == 'model_gone' else _model(model_id=1)),
    )
    monkeypatch.setattr(
        controllers,
        '_model_gateway_registry_inputs',
        AsyncMock(return_value=([], {}, [])),
    )
    monkeypatch.setattr(
        controllers.mcp_handler,
        'sglang_smg_service_names',
        lambda *args: {'model-1-7-r3-smg-a1.static'},
    )
    declare = AsyncMock()
    monkeypatch.setattr(
        controllers.mcp_handler, 'ensure_sglang_smg_transformer', declare
    )
    with pytest.raises(ValueError, match='changed while preparing route declarations'):
        await controllers._ensure_route_smg_transformers(
            None, _Cfg(), object(), [(1, 'm', _Registry('model-1-7-r3-smg-a0.static'))]
        )
    declare.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'publishing,orphan', [(False, False), (True, False), (False, True)]
)
async def test_smg_cleanup_failure_preserves_withdrawal_and_queue_retry(
    monkeypatch, publishing, orphan
):
    import asyncio
    from unittest.mock import MagicMock
    from kubernetes_asyncio.client import ApiException
    from gpustack.server import controllers
    from gpustack.server.workqueue import WorkEvent, WorkEventType

    model = _model(model_id=1)
    monkeypatch.setattr(
        controllers,
        '_model_gateway_registry_inputs',
        AsyncMock(return_value=([], {}, [])),
    )
    monkeypatch.setattr(
        controllers.mcp_handler,
        'sglang_smg_service_names',
        lambda *args: {'smg.static'} if publishing else set(),
    )
    failure = ApiException(status=409)
    monkeypatch.setattr(
        controllers.mcp_handler,
        'ensure_sglang_smg_transformer',
        AsyncMock(side_effect=failure if not orphan else None),
    )
    order = []
    monkeypatch.setattr(
        controllers.mcp_handler,
        'ensure_model_mcp_bridge',
        AsyncMock(side_effect=lambda **kwargs: order.append('withdraw_registry')),
    )
    monkeypatch.setattr(
        controllers,
        'notify_model_route_target',
        AsyncMock(side_effect=lambda **kwargs: order.append('notify')),
    )
    monkeypatch.setattr(controllers, 'sync_categories_and_meta', AsyncMock())
    controller = controllers.ModelController.__new__(controllers.ModelController)
    controller._disable_gateway = False
    controller._config = MagicMock()
    controller._higress_extension_api = object()
    controller._higress_network_api = object()
    controller._sync_model_ai_proxy = AsyncMock(
        side_effect=lambda *args: order.append('withdraw_ai')
    )
    if orphan:
        controller._smg_cleanup_pending = True
        controller._cleanup_smg_transformer_orphans = AsyncMock(return_value=failure)
    controller._passes = asyncio.Semaphore(1)
    controller._queue = MagicMock()
    controller._inflight = {(1,): object()}

    async def reconcile(event):
        await controller._sync_model_gateway(None, model, event)

    controller._reconcile = reconcile
    work = WorkEvent(
        keys=(1,),
        type=WorkEventType.MODIFIED,
        object=Event(type=EventType.UPDATED, data=model),
    )
    await controller._process(work)
    assert order == (
        [] if publishing else ['notify', 'withdraw_registry', 'withdraw_ai']
    )
    controller._queue.add_rate_limited.assert_called_once_with(work)
    controller._queue.forget.assert_not_called()


@pytest.mark.asyncio
async def test_legacy_registry_failure_does_not_block_existing_target_notification(
    monkeypatch,
):
    from unittest.mock import MagicMock
    from gpustack.server import controllers

    order = []
    monkeypatch.setattr(
        controllers,
        '_model_gateway_registry_inputs',
        AsyncMock(return_value=([], {}, [])),
    )
    monkeypatch.setattr(
        controllers.mcp_handler, 'ensure_sglang_smg_transformer', AsyncMock()
    )
    monkeypatch.setattr(
        controllers,
        'notify_model_route_target',
        AsyncMock(side_effect=lambda **kwargs: order.append('notify')),
    )
    monkeypatch.setattr(
        controllers,
        'sync_categories_and_meta',
        AsyncMock(side_effect=lambda *args: order.append('categories')),
    )

    async def registry(**kwargs):
        order.append('registry')
        raise RuntimeError('MCP unavailable')

    monkeypatch.setattr(controllers.mcp_handler, 'ensure_model_mcp_bridge', registry)
    controller = controllers.ModelController.__new__(controllers.ModelController)
    controller._disable_gateway = False
    controller._config = MagicMock()
    controller._higress_extension_api = object()
    controller._higress_network_api = object()
    controller._sync_model_ai_proxy = AsyncMock()
    model = _model()
    with pytest.raises(RuntimeError, match='MCP unavailable'):
        await controller._sync_model_gateway(
            None, model, Event(type=EventType.UPDATED, data=model)
        )
    assert order == ['notify', 'categories', 'registry']
    controllers._model_gateway_registry_inputs.assert_awaited_once()
    controller._sync_model_ai_proxy.assert_not_awaited()


@pytest.mark.asyncio
async def test_managed_rank_and_component_services_use_cluster_credentials(monkeypatch):
    services = [
        'model-5-1-r3-dp0.static',
        'model-5-1-r3-dp1.static',
        'model-5-2-r3-smg-a2.static',
    ]
    captured = await _run_sync(
        monkeypatch,
        _model(),
        [_target(1)],
        destinations_by_override={None: _destinations(*services)},
    )
    (provider,) = captured['kwargs']['expected_providers']
    (rule,) = captured['kwargs']['expected_match_rules']
    assert provider['apiTokens'] == ['cluster-token']
    assert sorted(rule.service) == sorted(services)
    assert rule.config['activeProviderId'] == provider['id']


@pytest.mark.asyncio
async def test_missing_cluster_token_warns_without_exposing_credentials(
    monkeypatch, caplog
):
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from gpustack.server.controllers import cluster_registration_tokens
    from gpustack.schemas.clusters import Cluster

    monkeypatch.setattr(
        Cluster,
        'one_by_id',
        AsyncMock(return_value=SimpleNamespace(registration_token=None)),
    )
    assert await cluster_registration_tokens(MagicMock(), 1) == []
    assert 'has no registration token' in caplog.text
