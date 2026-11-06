"""Raw route names must survive request and persistence validation."""

import pytest
from pydantic import ValidationError

from gpustack.schemas.model_routes import (
    ModelRouteCreate,
    ModelRouteTarget,
    ModelRouteTargetUpdateItem,
    ModelRouteUpdate,
)
from gpustack.schemas.models import ModelCreate, SourceEnum


@pytest.mark.parametrize(
    "name", ["qwen", "Qwen3-0.6B", "a", "Qwen3:sql", "model_name:adapter1"]
)
def test_route_name_survives_write_and_persistence_validation(name):
    assert ModelRouteCreate(name=name).name == name
    assert ModelRouteUpdate(name=name).name == name
    target = ModelRouteTarget(
        name=f"{name}-deployment", route_name=name, route_id=1, model_id=1
    )
    # ActiveRecord's before_flush hook validates a dumped relationship target.
    restored = ModelRouteTarget.model_validate(target.model_dump())
    assert restored.route_name == name


@pytest.mark.parametrize(
    "name",
    [
        "",
        "31231213",
        "3-model",
        "1",
        "org/model",
        "-model",
        "_model",
        "model-",
        "model_",
        "model.",
        "model:",
        "a b",
        "model\n",
    ],
)
def test_invalid_raw_route_name_is_rejected(name):
    with pytest.raises(ValidationError, match="name must start"):
        ModelRouteCreate(name=name)
    with pytest.raises(ValidationError, match="name must start"):
        ModelRouteUpdate(name=name)
    with pytest.raises(ValidationError, match="route_name must start"):
        ModelRouteTarget.model_validate(
            {"name": "target", "route_name": name, "route_id": 1, "model_id": 1}
        )


@pytest.mark.parametrize("name", ["Qwen3:sql", "a:2", "qwen-model:adapter"])
def test_letter_prefixed_local_model_name_accepts_lora_target(name):
    target = ModelRouteTargetUpdateItem(model_id=1, overridden_model_name=name)
    assert target.overridden_model_name == name


@pytest.mark.parametrize("name", ["31231213:sql", "1:2", "3-model:adapter"])
def test_numeric_local_model_name_rejects_lora_target(name):
    with pytest.raises(ValidationError, match="must start with a letter,"):
        ModelRouteTargetUpdateItem(model_id=1, overridden_model_name=name)


@pytest.mark.parametrize("enabled", [False, None])
@pytest.mark.parametrize("name", ["31231213", "3-model", "1", "org/model"])
def test_deployment_without_auto_route_does_not_apply_route_name_rules(enabled, name):
    model = ModelCreate(
        name=name,
        source=SourceEnum.LOCAL_PATH,
        local_path="/models/qwen",
        enable_model_route=enabled,
    )
    assert model.name == name
