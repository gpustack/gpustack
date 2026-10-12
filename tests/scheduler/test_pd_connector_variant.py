from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gpustack.scheduler import scheduler
from gpustack.schemas.models import DisaggregationSpec, RoleSpec
from tests.utils.model import new_model


def _model():
    return new_model(
        1,
        "DeepSeek-V4-display-only",
        huggingface_repo_id="org/model",
        backend="vLLM",
        backend_version="0.23.0",
        roles=[RoleSpec(name="prefill"), RoleSpec(name="decode")],
        disaggregation=DisaggregationSpec(mode="vllm-ascend-mooncake"),
        meta={"owner_note": "keep"},
    )


@pytest.mark.asyncio
async def test_evaluation_records_layout_even_with_existing_metadata(monkeypatch):
    model = _model()
    monkeypatch.setattr(
        scheduler, "evaluate_diffusion_model", AsyncMock(return_value=False)
    )
    monkeypatch.setattr(
        scheduler,
        "get_pretrained_config_with_workers",
        AsyncMock(
            return_value=SimpleNamespace(
                architectures=["Qwen3ForCausalLM"], model_type="qwen3"
            )
        ),
    )
    assert await scheduler.evaluate_pretrained_config(model)
    assert model.meta["pd_connector_variant"] == "standard"
    assert model.meta["owner_note"] == "keep"


@pytest.mark.asyncio
@pytest.mark.parametrize("override_scope", ["model", "roles"])
async def test_architecture_override_preserves_source_layout(
    monkeypatch, override_scope
):
    model = _model()
    parameters = ['--hf-overrides={"architectures":["Qwen3ForCausalLM"]}']
    if override_scope == "model":
        model.backend_parameters = parameters
    else:
        for role in model.roles:
            role.backend_parameters = parameters
    monkeypatch.setattr(
        scheduler, "evaluate_diffusion_model", AsyncMock(return_value=False)
    )
    config_loader = AsyncMock(
        return_value=SimpleNamespace(
            architectures=["Qwen3_5ForCausalLM"],
            layer_types=["full_attention", "linear_attention"],
        )
    )
    monkeypatch.setattr(scheduler, "get_pretrained_config_with_workers", config_loader)

    await scheduler.evaluate_pretrained_config(model)

    config_loader.assert_awaited_once()
    assert model.meta["pd_connector_variant"] == "hybrid"
    assert model.meta["owner_note"] == "keep"


@pytest.mark.asyncio
async def test_architecture_override_with_unavailable_layout_uses_fallback(monkeypatch):
    model = _model()
    model.backend_parameters = ['--hf-overrides={"architectures":["Qwen3ForCausalLM"]}']
    monkeypatch.setattr(
        scheduler, "evaluate_diffusion_model", AsyncMock(return_value=False)
    )
    monkeypatch.setattr(
        scheduler,
        "get_pretrained_config_with_workers",
        AsyncMock(side_effect=ValueError("unavailable model config")),
    )

    await scheduler.evaluate_pretrained_config(model)

    assert model.meta["pd_connector_variant"] == "hybrid"


@pytest.mark.parametrize(
    "parameters,expected",
    [
        (
            ['--hf_overrides={"layer_types":["full_attention","linear_attention"]}'],
            "hybrid",
        ),
        (
            [
                '--hf-overrides={"layer_types":["full_attention","linear_attention"]}',
                '--hf-overrides={"layer_types":["full_attention"]}',
            ],
            "standard",
        ),
        (
            [
                '--hf-overrides={"layer_types":["full_attention"]}',
                '--hf_overrides={"layer_types":["full_attention","linear_attention"]}',
            ],
            "hybrid",
        ),
    ],
)
def test_layout_overrides_follow_engine_parameter_precedence(parameters, expected):
    model = _model()
    model.backend_parameters = parameters

    scheduler._record_pd_connector_variant(
        model, {"layer_types": ["full_attention"]}, []
    )

    assert model.meta["pd_connector_variant"] == expected


def test_role_overrides_choose_layout_and_choice_stays_pinned_to_model_inputs():
    model = _model()
    for role in model.roles:
        role.backend_parameters = ['--hf-overrides={"compress_ratios":[4,128]}']
    config = {"model_type": "qwen3"}
    assert scheduler._record_pd_connector_variant(model, config, [])
    assert model.meta["pd_connector_variant"] == "hybrid"
    assert not scheduler._record_pd_connector_variant(
        model, {"model_type": "other"}, []
    )
    for role in model.roles:
        role.backend_parameters = []
    assert scheduler._record_pd_connector_variant(model, config, [])
    assert model.meta["pd_connector_variant"] == "standard"


def test_unavailable_layout_uses_catalog_fallback():
    model = _model()
    assert scheduler._record_pd_connector_variant(model, None, [])
    assert model.meta["pd_connector_variant"] == "hybrid"


@pytest.mark.parametrize(
    "recovered_config", [{"model_type": "qwen3"}, {"model_type": "deepseek_v4"}]
)
def test_layout_recovery_keeps_the_groups_conservative_connector(recovered_config):
    model = _model()
    assert scheduler._record_pd_connector_variant(model, None, [])
    assert not scheduler._record_pd_connector_variant(model, recovered_config, [])
    assert model.meta["pd_connector_variant"] == "hybrid"


@pytest.mark.parametrize(
    "override", ["bad-json", "model_type=qwen3", "[1]", "null", "1"]
)
def test_unreadable_layout_overrides_use_the_catalog_fallback(override):
    from gpustack.server.pd_mode_catalog import get_pd_mode

    model = _model()
    model.backend_parameters = ["--hf-overrides", override]
    assert scheduler._record_pd_connector_variant(model, {"model_type": "qwen3"}, [])
    mode = get_pd_mode(model.disaggregation.mode.value)
    assert model.meta["pd_connector_variant"] == mode.default_connector_variant
    assert model.meta["owner_note"] == "keep"


@pytest.mark.asyncio
@pytest.mark.parametrize("override", ["bad-json", "null", "[1]"])
async def test_unreadable_override_does_not_abort_other_model_metadata(
    monkeypatch, override
):
    model = _model()
    model.backend_parameters = ["--hf-overrides", override]
    monkeypatch.setattr(
        scheduler, "evaluate_diffusion_model", AsyncMock(return_value=False)
    )
    loader = AsyncMock(return_value=SimpleNamespace(architectures=["Qwen3ForCausalLM"]))
    monkeypatch.setattr(scheduler, "get_pretrained_config_with_workers", loader)
    await scheduler.evaluate_pretrained_config(model)
    loader.assert_awaited_once()
    assert model.categories
    assert model.meta["pd_connector_variant"]
