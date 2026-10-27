"""Recognizing hybrid-attention models from a pretrained config.

The verdict decides whether a deployment is warned about its extended KV
cache, so both directions matter: a recurrent-layer model must be caught
whichever way its family spells the composition out, and an ordinary
transformer — including one that alternates sliding and full attention —
must not be, or the warning sends users after settings that do nothing.
"""

import types

import pytest

from gpustack.utils.hybrid_attention import is_hybrid_attention


def _config(**fields):
    return types.SimpleNamespace(**fields)


# Shapes taken from the families' own config.json.
QWEN3_5 = _config(
    model_type="qwen3_5",
    architectures=["Qwen3_5ForConditionalGeneration"],
    layer_types=["linear_attention"] * 3 + ["full_attention"],
    linear_key_head_dim=128,
    linear_conv_kernel_dim=4,
)
GEMMA3 = _config(
    model_type="gemma3_text",
    architectures=["Gemma3ForCausalLM"],
    layer_types=["sliding_attention"] * 5 + ["full_attention"],
)
LLAMA = _config(
    model_type="llama",
    architectures=["LlamaForCausalLM"],
    num_hidden_layers=32,
)
JAMBA = _config(
    model_type="jamba",
    architectures=["JambaForCausalLM"],
    layers_block_type=["mamba", "mamba", "attention"],
)
NEMOTRON_H = _config(
    model_type="nemotron_h",
    architectures=["NemotronHForCausalLM"],
    hybrid_override_pattern="M-M-M*-",
)
BAMBA = _config(
    model_type="bamba",
    architectures=["BambaForCausalLM"],
    mamba_d_state=128,
    mamba_expand=2,
)


@pytest.mark.parametrize(
    "config",
    [QWEN3_5, JAMBA, NEMOTRON_H, BAMBA],
    ids=["layer_types", "layers_block_type", "override_pattern", "state_fields"],
)
def test_recurrent_layers_are_recognized(config):
    assert is_hybrid_attention(config) is True


@pytest.mark.parametrize(
    "config",
    [
        LLAMA,
        # Alternating window sizes still page one KV pair per token, so
        # none of the hybrid cache requirements apply.
        GEMMA3,
        None,
        {},
    ],
    ids=["plain", "sliding_window", "unreadable", "empty"],
)
def test_ordinary_attention_is_not_reported(config):
    assert is_hybrid_attention(config) is False


def test_raw_config_mapping_is_read_like_a_config_object():
    """The local-path fallback hands back the parsed config.json rather than
    a PretrainedConfig."""
    assert is_hybrid_attention(
        {"model_type": "qwen3_next", "layer_types": ["linear_attention"]}
    )


def test_multimodal_wrapper_is_read_through_its_text_config():
    wrapper = _config(
        architectures=["SomeVLForConditionalGeneration"],
        vision_config=_config(num_hidden_layers=12),
        text_config=QWEN3_5,
    )

    assert is_hybrid_attention(wrapper) is True


def test_multimodal_wrapper_over_plain_attention_is_not_reported():
    wrapper = _config(
        architectures=["SomeVLForConditionalGeneration"],
        vision_config=_config(num_hidden_layers=12),
        text_config=LLAMA,
    )

    assert is_hybrid_attention(wrapper) is False
