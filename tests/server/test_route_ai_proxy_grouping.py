from unittest.mock import AsyncMock, patch

import pytest

from gpustack.schemas.clusters import Cluster
from gpustack.schemas.model_routes import ModelRoute, ModelRouteTarget, TargetStateEnum
from gpustack.schemas.models import ModelInstanceStateEnum
from gpustack.server.bus import Event, EventType
from gpustack.server.controllers import (
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
    # the referencing routes' legacy per-route ids retire with this write
    assert captured["kwargs"]["owned_provider_ids"] == {
        "gpustack-model-5",
        "ai-route-route-1",
        "ai-route-route-2",
        "ai-route-route-3",
    }
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
    [("native_anthropic_api", True), ("replicas", True), ("description", False)],
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
    hydrated.model_route_targets = [_target(1)]

    publish = AsyncMock()
    with (
        patch(
            "gpustack.server.controllers.Model.one_by_id",
            AsyncMock(return_value=hydrated),
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
