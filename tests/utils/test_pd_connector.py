from types import SimpleNamespace

import pytest

from gpustack.utils.pd_connector import pd_connector_variant


@pytest.mark.parametrize(
    "config,expected",
    [
        (None, None),
        ({"model_type": "qwen3"}, "standard"),
        ({"architectures": ["GlmMoeDsaForCausalLM"]}, "standard"),
        ({"model_type": "deepseek_v3"}, "standard"),
        ({"model_type": "deepseek_v4"}, "hybrid"),
        ({"compress_ratios": [0, 4, 128]}, "hybrid"),
        ({"layer_types": ["full_attention", "linear_attention"]}, "hybrid"),
        ({"text_config": {"compress_ratios": [4, 128]}}, "hybrid"),
        (SimpleNamespace(model_type="qwen3", name="DeepSeek-V4-Flash"), "standard"),
    ],
)
def test_variant_follows_model_config(config, expected):
    assert pd_connector_variant(config) == expected


@pytest.mark.parametrize(
    "ratios,expected",
    [
        (4, "hybrid"),
        (0.5, "hybrid"),
        (0, "standard"),
        (-1, "standard"),
        (None, "standard"),
        (True, "standard"),
        ("4", "standard"),
        ({"layer": 4}, "standard"),
        ([None, "4", {}, True], "standard"),
        ([None, "4", 128], "hybrid"),
        ((0, 4), "hybrid"),
    ],
)
def test_variant_handles_unusual_compression_metadata(ratios, expected):
    assert pd_connector_variant({"compress_ratios": ratios}) == expected


@pytest.mark.parametrize(
    "architectures,expected",
    [
        (1, "standard"),
        (0.5, "standard"),
        (None, "standard"),
        (True, "standard"),
        ("DeepseekV4ForCausalLM", "standard"),
        ([None, 1, {}], "standard"),
        ([None, "DeepseekV4ForCausalLM"], "hybrid"),
        (("DeepseekV4ForCausalLM",), "hybrid"),
    ],
)
def test_variant_handles_unusual_architecture_metadata(architectures, expected):
    config = SimpleNamespace(architectures=architectures)
    assert pd_connector_variant(config) == expected


@pytest.mark.parametrize(
    "nested_name", ["text_config", "llm_config", "language_config", "thinker_config"]
)
@pytest.mark.parametrize(
    "outer",
    [
        {"model_type": "deepseek_v4"},
        {"architectures": ["DeepseekV4ForCausalLM"]},
        {"compress_ratios": [4, 128]},
    ],
)
@pytest.mark.parametrize("as_object", [False, True])
def test_nested_standard_layout_preserves_outer_hybrid_signals(
    nested_name, outer, as_object
):
    config = {nested_name: {"model_type": "qwen3"}, **outer}
    if as_object:
        config = SimpleNamespace(**config)
    assert pd_connector_variant(config) == "hybrid"


def test_all_nested_language_configs_contribute_hybrid_signals():
    config = {
        "text_config": {"model_type": "qwen3"},
        "llm_config": {"compress_ratios": [4, 128]},
    }
    assert pd_connector_variant(config) == "hybrid"
