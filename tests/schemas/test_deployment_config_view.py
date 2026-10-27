from copy import deepcopy

import pytest

from gpustack.schemas.deployment_document import (
    deployment_config_view,
    deployment_entry,
    entry_document_form,
)
from gpustack.schemas.models import Model, ModelCreate, RoleSpec, SpeculativeConfig


def test_view_cleans_typed_fields_without_losing_explicit_values():
    raw = {
        "roles": [
            {
                "name": "decode",
                "env": {},
                "backend_parameters": [],
                "image_name": None,
                "resources": {"cpu": 0, "memory": None},
                "legacy_setting": None,
            }
        ],
        "env": {"EXPLICIT_NULL": None, "EMPTY": ""},
        "cpu_offloading": False,
        "replicas": 0,
        "source": "retired-source",
        "backend_parameters": [],
        "worker_selector": {},
        "image_name": None,
        "run_command": "",
        "legacy_config": {"value": None},
    }
    before = deepcopy(raw)
    view = deployment_config_view(raw)
    assert list(view)[:2] == ["source", "replicas"]
    assert "image_name" not in view
    assert view["roles"] == [
        {
            "name": "decode",
            "env": {},
            "backend_parameters": [],
            "resources": {"cpu": 0},
            "legacy_setting": None,
        }
    ]
    for field in (
        "env",
        "cpu_offloading",
        "replicas",
        "backend_parameters",
        "worker_selector",
        "run_command",
        "legacy_config",
    ):
        assert view[field] == raw[field]
    assert "restart_on_error" not in view
    view["roles"][0]["env"]["NEW"] = "value"
    assert raw == before


@pytest.mark.parametrize(
    "field,override",
    [
        ("env", None),
        ("env", {}),
        ("backend_parameters", None),
        ("backend_parameters", []),
    ],
)
def test_view_distinguishes_role_inheritance_from_empty_overrides(field, override):
    raw = {"roles": [{"name": "decode", field: override}]}
    role = deployment_config_view(raw)["roles"][0]
    if override is None:
        assert field not in role
    else:
        assert role[field] == override


def test_yaml_projections_share_the_configuration_view():
    create = ModelCreate(
        name="qwen",
        source="huggingface",
        huggingface_repo_id="org/qwen",
        roles=[RoleSpec(name="decode", env={}, backend_parameters=[])],
        speculative_config=SpeculativeConfig(enabled=False),
    )
    expected = create.model_dump(mode="json", exclude_none=True)
    assert deployment_config_view(create.model_dump(mode="json")) == expected
    imported = entry_document_form(create, "cluster")
    exported = deployment_entry(
        Model(**create.model_dump()), bool(create.enable_model_route), "cluster"
    )
    assert imported == exported
    assert imported["roles"] == expected["roles"]
    assert imported["speculative_config"] == {"enabled": False}
    assert "image_name" not in imported
