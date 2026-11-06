import json
import logging
from copy import deepcopy
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest
from transformers import AutoConfig, PretrainedConfig
from transformers.integrations.heterogeneity.configuration_utils import (
    AmbiguousGlobalPerLayerAttributeError,
)
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from gpustack.policies.candidate_selectors.base_candidate_selector import (
    ModelAttentionTypeEnum,
    ModelParameters,
)
from gpustack.schemas.models import Model
from gpustack.utils import model_config

FIXTURES = Path(__file__).parents[1] / "fixtures" / "model_configs"
CATALOG_CASES = json.loads((FIXTURES / "catalog" / "manifest.json").read_text())[
    "models"
]
# These checkpoints require the data-only fallback with the pinned Transformers.
CUSTOM_CONFIGS = {
    "DeepSeek-V4.1-Flash-1017",
    "MiMo-V2.6-Pro-MOPD",
    "Kimi-K3",
    "Nemotron-3-Nano-Omni-30B-A3B-Reasoning",
    "Qwen3-TTS-12Hz-1.7B-Base",
}


@pytest.fixture(autouse=True)
def offline_metadata_only(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", True)

    def forbidden(*args, **kwargs):
        pytest.fail("Configuration tests must not use the network or repository code")

    monkeypatch.setattr("requests.Session.request", forbidden)
    monkeypatch.setattr("httpx.Client.send", forbidden)
    monkeypatch.setattr(
        "transformers.dynamic_module_utils.get_class_from_dynamic_module", forbidden
    )
    monkeypatch.setattr(
        "transformers.models.auto.configuration_auto.get_class_from_dynamic_module",
        forbidden,
    )


def assert_equivalent_parameters(safe, native):
    model = Model(name="config-parity", categories=["llm"])
    actual, expected = ModelParameters(), ModelParameters()
    actual.from_model_pretrained_config(model, safe)
    expected.from_model_pretrained_config(model, native)
    for field in fields(ModelParameters):
        left, right = getattr(actual, field.name), getattr(expected, field.name)
        if isinstance(left, PretrainedConfig):
            left = left.to_dict()
        if isinstance(right, PretrainedConfig):
            right = right.to_dict()
        assert left == right, field.name


def test_auto_map_preserves_native_configuration_without_trust():
    path = FIXTURES / "deepseek_v3.json"
    raw = json.loads(path.read_text())
    assert raw["auto_map"]["AutoConfig"]
    native = AutoConfig.from_pretrained(
        str(path), trust_remote_code=False, local_files_only=True
    )
    parsed = model_config.safe_pretrained_config_from_dict(raw)
    assert type(parsed) is type(native) is CONFIG_MAPPING["deepseek_v3"]
    assert_equivalent_parameters(parsed, native)


@pytest.mark.parametrize(
    "case",
    [case for case in CATALOG_CASES if case["name"] not in CUSTOM_CONFIGS],
    ids=lambda case: case["name"],
)
def test_catalog_configs_match_native_safe_loading(case):
    path = FIXTURES / "catalog" / case["file"]
    native = AutoConfig.from_pretrained(
        str(path), trust_remote_code=False, local_files_only=True
    )
    raw = json.loads(path.read_text())
    original = deepcopy(raw)
    safe = model_config.safe_pretrained_config_from_dict(raw)
    assert type(safe) is type(native)
    assert_equivalent_parameters(safe, native)
    assert raw == original


@pytest.mark.parametrize("case", CATALOG_CASES, ids=lambda case: case["name"])
@pytest.mark.parametrize(
    "rope",
    [
        {"type": "video_rope", "factor": None},
        {"type": "linear", "factor": "custom"},
        {"type": "yarn", "factor": 4},
    ],
    ids=["unknown-type", "invalid-factor", "incomplete-yarn"],
)
def test_optional_rope_metadata_does_not_change_other_model_parameters(case, rope):
    raw = json.loads((FIXTURES / "catalog" / case["file"]).read_text())
    parsed = model_config.safe_pretrained_config_from_dict(raw)
    model = Model(name="metadata-independence", categories=["llm"])
    expected = ModelParameters()
    expected.from_model_pretrained_config(model, parsed)
    text_config = model_config.get_hf_text_config(parsed)
    text_config.rope_scaling = deepcopy(rope)
    actual = ModelParameters()
    actual.from_model_pretrained_config(model, parsed)
    for field in fields(ModelParameters):
        if field.name != "derived_max_seq_len":
            assert getattr(actual, field.name) == getattr(
                expected, field.name
            ), field.name


@pytest.mark.parametrize("alternating", [False, True])
def test_mistral_dispatch_matches_transformers(tmp_path, alternating):
    raw = {"model_type": "mistral", "num_hidden_layers": 2}
    if alternating:
        raw["layer_types"] = ["full_attention", "sliding_attention"]
    (tmp_path / "config.json").write_text(json.dumps(raw))
    native = AutoConfig.from_pretrained(
        str(tmp_path), trust_remote_code=False, local_files_only=True
    )
    safe = model_config.safe_pretrained_config_from_dict(raw)
    assert type(safe) is type(native)
    assert safe.model_type == native.model_type
    assert_equivalent_parameters(safe, native)


@pytest.mark.parametrize("multi_query,expected_kv_heads", [(True, 4), (False, 32)])
@pytest.mark.parametrize("explicit", [False, True])
def test_chatglm_aliases_preserve_explicit_fields(
    multi_query, expected_kv_heads, explicit
):
    raw = {
        "model_type": "chatglm",
        "num_layers": 40,
        "padded_vocab_size": 151552,
        "kv_channels": 128,
        "num_attention_heads": 32,
        "multi_query_attention": multi_query,
        "multi_query_group_num": 4,
        "seq_length": 1048576,
    }
    expected = {
        "num_hidden_layers": 40,
        "vocab_size": 151552,
        "head_dim": 128,
        "num_key_value_heads": expected_kv_heads,
    }
    if explicit:
        expected = {
            "num_hidden_layers": 28,
            "vocab_size": 65024,
            "head_dim": 64,
            "num_key_value_heads": 2,
        }
        raw.update(expected)
    original = deepcopy(raw)
    parsed = model_config.safe_pretrained_config_from_dict(raw)
    for key, value in expected.items():
        assert getattr(parsed, key) == value
    assert model_config.get_max_model_len(parsed, default=None) == 1048576
    assert raw == original


@pytest.mark.parametrize(
    "wrapper", ["flat", "text_config", "llm_config", "thinker_config"]
)
def test_incomplete_custom_metadata_preserves_unknown_fields(wrapper):
    text = {"model_type": "custom", "num_hidden_layers": 20, "hidden_size": 2048}
    raw = text
    if wrapper == "thinker_config":
        raw = {wrapper: {"text_config": text}}
    elif wrapper != "flat":
        raw = {wrapper: text}
    raw = {"architectures": ["LlamaForCausalLM"], **raw}
    parsed = model_config.safe_pretrained_config_from_dict(raw)
    params = ModelParameters()
    params.from_model_pretrained_config(
        Model(name="partial", categories=["llm"]), parsed
    )
    assert params.architectures == ["LlamaForCausalLM"]
    assert params.num_hidden_layers == 20
    assert params.hidden_size == 2048
    assert params.num_attention_heads is None
    assert params.num_key_value_heads is None
    assert params.derived_max_seq_len is None
    assert params.get_attention_type() == ModelAttentionTypeEnum.UNK


def test_explicit_context_fallback_is_not_reported_as_model_metadata():
    config = model_config.safe_pretrained_config_from_dict({"model_type": "custom"})
    assert model_config.get_max_model_len(config, default=None) is None
    assert model_config.get_max_model_len(config, default=4096) == 4096


@pytest.mark.parametrize("wrapper", ["llm_config", "text_config", "thinker_config"])
@pytest.mark.parametrize(
    "text_length,root_length,expected",
    [
        (None, 32768, 65536),
        (8192, 32768, 32768),
        ("invalid", 32768, 65536),
        (None, None, None),
    ],
)
def test_text_dimensions_preserve_root_length_fallback(
    wrapper, text_length, root_length, expected
):
    text = {
        "model_type": "custom_text",
        "num_attention_heads": 16,
        "hidden_size": 2048,
        "max_position_embeddings": text_length,
        "rope_scaling": {"type": "linear", "factor": 4},
    }
    raw = {
        "architectures": ["CustomMultimodal"],
        "max_position_embeddings": root_length,
        "rope_scaling": {"type": "linear", "factor": 2},
        wrapper: {"text_config": text} if wrapper == "thinker_config" else text,
    }
    original = deepcopy(raw)
    parsed = model_config.safe_pretrained_config_from_dict(raw)
    params = ModelParameters()
    params.from_model_pretrained_config(
        Model(name="split-metadata", categories=["llm"]), parsed
    )
    assert params.derived_max_seq_len == expected
    assert params.architectures == ["CustomMultimodal"]
    assert params.num_attention_heads == 16
    assert params.hidden_size == 2048
    assert raw == original


@pytest.mark.parametrize(
    "rope,expected",
    [
        ({"type": "linear", "factor": 2}, 16384),
        ({"rope_type": "dynamic", "factor": 1.5}, 12288),
        (
            {"type": "yarn", "factor": 4, "original_max_position_embeddings": 4096},
            16384,
        ),
        ({"type": "yarn", "factor": 4}, 8192),
        ({"type": "video_rope", "factor": None}, 8192),
        ({"type": "video_rope", "factor": 4}, 8192),
        ({"type": "llama3", "factor": 8}, 8192),
        ({"type": "longrope", "factor": 8}, 8192),
        ({"type": "su", "factor": 8}, 8192),
        ({"type": "default", "factor": 8}, 8192),
        ({"type": ["linear"], "factor": 8}, 8192),
        ({"full_attention": {"rope_type": "linear", "factor": 4}}, 8192),
        (None, 8192),
        ([], 8192),
    ],
)
def test_rope_scaling_requires_supported_semantics(rope, expected):
    config = SimpleNamespace(max_position_embeddings=8192, rope_scaling=rope)
    assert model_config.get_max_model_len(config, default=None) == expected


@pytest.mark.parametrize(
    "invalid", [None, True, False, "2", [], {}, 0, -1, float("nan"), float("inf")]
)
@pytest.mark.parametrize("rope_type", ["linear", "dynamic", "yarn"])
def test_invalid_rope_factor_preserves_declared_length(invalid, rope_type):
    config = SimpleNamespace(
        max_position_embeddings=8192,
        rope_scaling={
            "type": rope_type,
            "factor": invalid,
            "original_max_position_embeddings": 4096,
        },
    )
    assert model_config.get_max_model_len(config, default=None) == 8192


@pytest.mark.parametrize("invalid", [None, True, "4096", [], 0, -1, float("inf")])
def test_invalid_lengths_use_other_metadata_or_caller_default(invalid):
    config = SimpleNamespace(max_position_embeddings=invalid, seq_length=32768)
    assert model_config.get_max_model_len(config, default=None) == 32768
    del config.seq_length
    assert model_config.get_max_model_len(config, default=None) is None
    assert model_config.get_max_model_len(config, default=4096) == 4096
    config.max_position_embeddings = 8192
    config.rope_scaling = {
        "type": "yarn",
        "factor": 4,
        "original_max_position_embeddings": invalid,
    }
    assert model_config.get_max_model_len(config, default=None) == 8192


@pytest.mark.parametrize("factor", [1e308, 1e-308])
def test_unusable_scaled_length_preserves_declared_length(factor):
    config = SimpleNamespace(
        max_position_embeddings=8192, rope_scaling={"type": "linear", "factor": factor}
    )
    assert model_config.get_max_model_len(config, default=None) == 8192


@pytest.mark.parametrize("wrapper", ["llm_config", "text_config", "thinker_config"])
@pytest.mark.parametrize("root_architecture", [None, ["InternVLChatModel"]])
def test_language_parameters_preserve_root_architecture(wrapper, root_architecture):
    text = {
        "architectures": ["Qwen2ForCausalLM"],
        "hidden_size": 3584,
        "num_attention_heads": 28,
    }
    raw = {wrapper: {"text_config": text} if wrapper == "thinker_config" else text}
    if root_architecture:
        raw["architectures"] = root_architecture
    params = ModelParameters()
    params.from_model_pretrained_config(
        Model(name="multimodal", categories=["llm"]),
        model_config.safe_pretrained_config_from_dict(raw),
    )
    assert params.architectures == (root_architecture or text["architectures"])
    assert params.hidden_size == 3584
    assert params.num_attention_heads == 28


@pytest.mark.parametrize("nested", [False, True])
def test_per_layer_metadata_preserves_dimensions_in_composite_configs(nested):
    text = {
        "model_type": "llama",
        "num_hidden_layers": 2,
        "hidden_size": 4096,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "per_layer_config": {"1": {"num_attention_heads": 32}},
    }
    raw = {"model_type": "llava", "text_config": text} if nested else text
    original = deepcopy(raw)
    parsed = model_config.safe_pretrained_config_from_dict(raw)
    params = ModelParameters()
    params.from_model_pretrained_config(
        Model(name="per-layer", categories=["llm"]), parsed
    )
    assert params.num_attention_heads_per_layer == [16, 32]
    assert params.num_attention_heads == 32
    assert params.num_key_value_heads == 4
    assert params.head_dim == 256
    assert raw == original


@pytest.mark.parametrize("model_type", ["llama", "qwen2", "qwen3"])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("vary_query_heads", [False, True])
def test_per_layer_overrides_retain_native_architecture_defaults(
    model_type, nested, vary_query_heads
):
    text = {
        "model_type": model_type,
        "num_attention_heads": 32,
        "num_hidden_layers": 2,
        "per_layer_config": {"1": {"num_key_value_heads": 8}},
    }
    baseline = CONFIG_MAPPING[model_type].from_dict(
        {key: value for key, value in text.items() if key != "per_layer_config"}
    )
    if vary_query_heads:
        text["per_layer_config"]["1"]["num_attention_heads"] = 64
    raw = {"model_type": "llava", "text_config": text} if nested else text
    original = deepcopy(raw)
    parsed = model_config.safe_pretrained_config_from_dict(raw)
    language = model_config.get_hf_text_config(parsed)
    assert [layer.num_key_value_heads for layer in language.per_layer_config] == [
        baseline.num_key_value_heads,
        8,
    ]
    assert [layer.num_attention_heads for layer in language.per_layer_config] == [
        32,
        64 if vary_query_heads else 32,
    ]
    assert language.max_position_embeddings == baseline.max_position_embeddings
    params = ModelParameters()
    params.from_model_pretrained_config(
        Model(name="per-layer-defaults", categories=["llm"]), parsed
    )
    assert params.num_key_value_heads == baseline.num_key_value_heads
    assert params.head_dim == getattr(
        baseline, "head_dim", baseline.hidden_size // baseline.num_attention_heads
    )
    assert raw == original


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("field", ["intermediate_size", "num_heads", "n_groups"])
def test_attention_models_do_not_read_global_ssm_dimensions(nested, field):
    text = {
        "model_type": "llama",
        "num_hidden_layers": 2,
        "num_attention_heads": 32,
        field: 4096,
        "per_layer_config": {"1": {field: 8192}},
    }
    raw = {"model_type": "llava", "text_config": text} if nested else text
    original = deepcopy(raw)
    parsed = model_config.safe_pretrained_config_from_dict(raw)
    language = model_config.get_hf_text_config(parsed)
    assert [getattr(layer, field) for layer in language.per_layer_config] == [
        4096,
        8192,
    ]
    with pytest.raises(AmbiguousGlobalPerLayerAttributeError):
        getattr(language, field)
    params = ModelParameters()
    params.from_model_pretrained_config(Model(name="layer-widths"), parsed)
    assert params.num_attention_heads == 32
    assert params.intermediate_size is None
    assert params.num_heads is None
    assert params.n_groups is None
    assert raw == original


def test_native_parse_failure_reports_reason_without_checkpoint_content(
    monkeypatch, caplog
):
    def unavailable(*args, **kwargs):
        raise ValueError("sensitive-checkpoint-content")

    monkeypatch.setattr(CONFIG_MAPPING["llama"], "from_dict", unavailable)
    with caplog.at_level(logging.DEBUG, logger=model_config.__name__):
        config = model_config.safe_pretrained_config_from_dict(
            {
                "model_type": "llama",
                "num_attention_heads": 8,
                "max_position_embeddings": 8192,
            }
        )
    assert config.num_attention_heads == 8
    assert model_config.get_max_model_len(config, default=None) == 8192
    assert "installed config loading failed (ValueError)" in caplog.text
    assert "sensitive-checkpoint-content" not in caplog.text


def test_native_parse_failure_cannot_mutate_fallback_metadata(monkeypatch):
    raw = {
        "model_type": "llama",
        "num_attention_heads": 8,
        "quantization_config": {"quant_method": "gptq", "bits": 4},
    }
    original = deepcopy(raw)

    def mutating_constructor(config_dict):
        config_dict.pop("num_attention_heads")
        config_dict["quantization_config"]["bits"] = 8
        raise ValueError("unsupported native configuration")

    monkeypatch.setattr(CONFIG_MAPPING["llama"], "from_dict", mutating_constructor)
    parsed = model_config.safe_pretrained_config_from_dict(raw)
    assert parsed.num_attention_heads == 8
    assert parsed.quantization_config == original["quantization_config"]
    assert raw == original


@pytest.mark.parametrize("known", [False, True])
def test_logs_identify_config_source(known, caplog):
    with caplog.at_level(logging.DEBUG, logger=model_config.__name__):
        model_config.safe_pretrained_config_from_dict(
            {"model_type": "llama" if known else "custom"}
        )
    assert (
        "Using installed configuration class" if known else "no installed config class"
    ) in caplog.text
