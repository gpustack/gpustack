from unittest.mock import AsyncMock, patch

import pytest

from gpustack.gateway.utils import lora_registry_name_suffix
from gpustack.schemas.clusters import Cluster
from gpustack.schemas.model_routes import ModelRoute, ModelRouteTarget, TargetStateEnum
from gpustack.schemas.models import ModelInstanceStateEnum
from gpustack.server.bus import Event, EventType
from gpustack.server.controllers import (
    accumulate_model_ai_proxy_group,
    calculate_destinations,
    notify_model_route_target,
)
from tests.utils.model import new_model, new_model_instance

CLUSTER_ID = 3
MODEL_ID = 5


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


def _target(id: int, overridden_model_name=None, weight: int = 1):
    return ModelRouteTarget(
        id=id,
        route_id=1,
        model_id=MODEL_ID,
        overridden_model_name=overridden_model_name,
        weight=weight,
        state=TargetStateEnum.ACTIVE,
    )


@pytest.mark.asyncio
async def test_lora_targets_of_one_deployment_fold_into_a_single_group():
    """Several LoRA targets on one deployment share a single provider.

    Each LoRA target aliases the same instances under its own service name
    (see ``lora_registry_name_suffix``), so the group has to accumulate the
    services of every target instead of overwriting them — otherwise only the
    last adapter's rule would carry the deployment's token.
    """
    model = new_model(MODEL_ID, "base", huggingface_repo_id="repo/base")
    model.cluster_id = CLUSTER_ID
    instances = [_running_instance(1), _running_instance(2)]
    targets = [
        _target(1),
        _target(2, overridden_model_name="base:adapter-a"),
        _target(3, overridden_model_name="base:adapter-b"),
    ]

    with (
        patch(
            "gpustack.server.controllers.ModelRouteTarget.all_by_field",
            AsyncMock(return_value=targets),
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
            AsyncMock(return_value=instances),
        ),
        patch(
            "gpustack.server.controllers.Worker.all_by_fields",
            AsyncMock(return_value=[]),
        ),
    ):
        _, _, groups = await calculate_destinations(
            session=None, model_route=ModelRoute(id=1, name="base")
        )

    assert len(groups) == 1
    group = groups[0]
    assert group.provider_id() == "gpustack-model-5"
    assert group.api_tokens == ["cluster-token"]
    suffix_a = lora_registry_name_suffix("base:adapter-a")
    suffix_b = lora_registry_name_suffix("base:adapter-b")
    assert group.service_names == {
        "model-5-1.static",
        "model-5-2.static",
        f"model-5-1-{suffix_a}.static",
        f"model-5-2-{suffix_a}.static",
        f"model-5-1-{suffix_b}.static",
        f"model-5-2-{suffix_b}.static",
    }
    # No target declares fallback status codes, so the fallback ingress gets no
    # rule at all.
    assert group.fallback_service_names == set()


@pytest.mark.asyncio
async def test_fallback_target_services_land_on_both_ingresses():
    model = new_model(MODEL_ID, "base", huggingface_repo_id="repo/base")
    model.cluster_id = CLUSTER_ID
    fallback_target = _target(2, overridden_model_name="base:adapter-a")
    fallback_target.fallback_status_codes = ["5xx"]

    with (
        patch(
            "gpustack.server.controllers.ModelRouteTarget.all_by_field",
            AsyncMock(return_value=[_target(1), fallback_target]),
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
            AsyncMock(return_value=[_running_instance(1)]),
        ),
        patch(
            "gpustack.server.controllers.Worker.all_by_fields",
            AsyncMock(return_value=[]),
        ),
    ):
        _, _, groups = await calculate_destinations(
            session=None, model_route=ModelRoute(id=1, name="base")
        )

    (group,) = groups
    suffix_a = lora_registry_name_suffix("base:adapter-a")
    assert group.service_names == {
        "model-5-1.static",
        f"model-5-1-{suffix_a}.static",
    }
    # Only the fallback target's alias is on the fallback ingress; it is a
    # subset of the main ingress' services.
    assert group.fallback_service_names == {f"model-5-1-{suffix_a}.static"}
    assert group.fallback_service_names < group.service_names


@pytest.mark.asyncio
async def test_cluster_token_is_resolved_once_per_cluster():
    """Deployments of one cluster share the reconcile's token lookup."""
    one_by_id = AsyncMock(return_value=_cluster())
    model_groups = {}
    cluster_api_tokens = {}

    with patch("gpustack.server.controllers.Cluster.one_by_id", one_by_id):
        for model_id in (MODEL_ID, MODEL_ID + 1):
            model = new_model(model_id, f"m-{model_id}", huggingface_repo_id="repo/m")
            model.cluster_id = CLUSTER_ID
            await accumulate_model_ai_proxy_group(
                session=None,
                model_groups=model_groups,
                cluster_api_tokens=cluster_api_tokens,
                model=model,
                destinations=[],
                is_fallback_target=False,
            )

    assert one_by_id.await_count == 1
    assert {group.api_tokens[0] for group in model_groups.values()} == {"cluster-token"}


@pytest.mark.asyncio
async def test_native_anthropic_api_comes_from_the_deployment():
    """The selector is read straight off the Model -- no lookup, and each
    deployment answers for itself."""
    model_groups = {}
    cluster_api_tokens = {}

    with patch(
        "gpustack.server.controllers.Cluster.one_by_id",
        AsyncMock(return_value=_cluster()),
    ):
        for model_id, native in ((MODEL_ID, True), (MODEL_ID + 1, False)):
            model = new_model(model_id, f"m-{model_id}", huggingface_repo_id="repo/m")
            model.cluster_id = CLUSTER_ID
            model.native_anthropic_api = native
            await accumulate_model_ai_proxy_group(
                session=None,
                model_groups=model_groups,
                cluster_api_tokens=cluster_api_tokens,
                model=model,
                destinations=[],
                is_fallback_target=False,
            )

    assert model_groups[MODEL_ID].native_anthropic_api is True
    # One deployment's answer must not leak into another's provider entry.
    assert model_groups[MODEL_ID + 1].native_anthropic_api is False


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
