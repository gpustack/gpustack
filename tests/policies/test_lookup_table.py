from types import SimpleNamespace
from copy import deepcopy
from itertools import combinations
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from gpustack.policies.candidate_selectors.vllm_resource_fit_selector import (
    VLLMResourceFitSelector,
)
from gpustack.policies.candidate_selectors.sglang_resource_fit_selector import (
    SGLangResourceFitSelector,
)
from gpustack.policies.lookup_table import (
    estimate_offloaded_lookup_bytes,
    get_offloaded_lookup_table_kind,
)
from gpustack.policies.resource_view import ResourceView
from gpustack.policies.utils import (
    estimate_model_vram,
    get_computed_ram_claim,
    get_model_ram_claim,
)
from gpustack.schemas.models import (
    BackendEnum,
    ExtendedKVCacheConfig,
    GPUSelector,
    Model,
    SourceEnum,
)
from gpustack.utils.model_config import safe_pretrained_config_from_dict
from tests.fixtures.workers.fixtures import (
    linux_nvidia_1_4090_24gx1,
    linux_nvidia_3_4090_24gx1,
    linux_nvidia_5_a100_80gx2,
    linux_nvidia_12_A40_48gx2,
)

GIB = 1024**3
ARCHITECTURES = ["DeepseekV41ForCausalLM"]
TABLE_BYTES = 33 * 2**28
TABLE_RAM = TABLE_BYTES


def lookup_config(architecture="DeepseekV41ForCausalLM"):
    return SimpleNamespace(
        architectures=[architecture],
        text_config=SimpleNamespace(
            num_attention_heads=8,
            max_position_embeddings=32768,
            dtype="bfloat16",
            engram_layer_ids=[1],
            engram_num_embeddings=[2**28],
            engram_head_dim=32,
            ple_layer_ids=[2],
            ple_embed_dim=256,
            ngram_vocab_size_base=33 * 2**20,
            ngram_size=3,
            heads_per_ngram=8,
        ),
        quantization_config={
            "quant_method": "fp8",
            "scale_fmt": "ue8m0",
            "weight_block_size": [32, 32],
        },
    )


def model(**kwargs):
    return Model(
        name="lookup",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="test/lookup",
        backend=BackendEnum.VLLM,
        categories=["llm"],
        **kwargs,
    )


@pytest.mark.parametrize(
    "env,parameters,version,image,enabled",
    [
        ({"VLLM_PLE_CPU_OFFLOAD": "1"}, [], None, None, True),
        ({"VLLM_PLE_CPU_OFFLOAD": "0"}, [], "0.30.0", None, False),
        (
            {"VLLM_PLE_CPU_OFFLOAD": "1"},
            ['--engram-config={"cpu_offload":false}'],
            "0.30.0",
            None,
            False,
        ),
        (
            {"VLLM_PLE_CPU_OFFLOAD": "0"},
            ['--engram-config', '{"cpu_offload":true}'],
            None,
            None,
            True,
        ),
        ({}, ['--engram-config.cpu_offload=true'], None, "custom:latest", True),
        ({}, [], "0.30.0", None, True),
        ({}, [], "0.29.0", None, False),
        ({}, [], None, None, True),
        ({}, [], "0.30.0-custom", None, True),
        ({}, [], "0.30.0", "custom:latest", True),
        ({"VLLM_PLE_CPU_OFFLOAD": "0"}, [], "custom", None, False),
        ({"VLLM_PLE_CPU_OFFLOAD": "1"}, [], "custom", None, True),
        (
            {"VLLM_PLE_CPU_OFFLOAD": "1"},
            ['--engram-config={"cpu_offload":false}'],
            "custom",
            None,
            False,
        ),
        ({}, ['--engram-config=invalid'], "0.30.0", None, False),
        ({}, ['--engram-config={"cpu_offload":"false"}'], "0.30.0", None, False),
        ({"VLLM_PLE_CPU_OFFLOAD": "invalid"}, [], "0.30.0", None, False),
    ],
)
@pytest.mark.parametrize(
    "architecture,kind",
    [("DeepseekV41ForCausalLM", "engram"), ("Qwen4ExpForConditionalGeneration", "ple")],
)
def test_offload_configuration_precedence(
    env, parameters, version, image, enabled, architecture, kind
):
    m = model(
        env=env,
        backend_parameters=parameters,
        backend_version=version,
        image_name=image,
    )
    assert get_offloaded_lookup_table_kind(m, [architecture]) == (
        kind if enabled else None
    )


@pytest.mark.parametrize("name", ["engram-config", "engram_config"])
@pytest.mark.parametrize("dotted", [False, True])
@pytest.mark.parametrize("inline", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_offload_parameter_spellings(name, dotted, inline, enabled):
    key = f"--{name}.cpu_offload" if dotted else f"--{name}"
    value = json.dumps(enabled if dotted else {"cpu_offload": enabled})
    parameters = [f"{key}={value}"] if inline else [key, value]
    m = model(backend_version="0.30.0", backend_parameters=parameters)
    assert estimate_offloaded_lookup_bytes(m, lookup_config()) == (
        TABLE_BYTES if enabled else 0
    )


@pytest.mark.parametrize("first_name", ["engram-config", "engram_config"])
@pytest.mark.parametrize(
    "first_dotted,last_dotted",
    [(False, False), (True, True), (False, True), (True, False)],
)
@pytest.mark.parametrize("enabled", [False, True])
def test_mixed_offload_parameter_aliases_preserve_precedence(
    first_name, first_dotted, last_dotted, enabled
):
    last_name = "engram_config" if first_name == "engram-config" else "engram-config"
    parameters = []
    for name, dotted, value in [
        (first_name, first_dotted, not enabled),
        (last_name, last_dotted, enabled),
    ]:
        key = f"--{name}.cpu_offload" if dotted else f"--{name}"
        parameters.append(
            f"{key}={json.dumps(value if dotted else {'cpu_offload': value})}"
        )
    m = model(backend_version="0.30.0", backend_parameters=parameters)
    # The dotted object replaces JSON; duplicate fields use the last value.
    expected_enabled = not enabled if first_dotted and not last_dotted else enabled
    assert estimate_offloaded_lookup_bytes(m, lookup_config()) == (
        TABLE_BYTES if expected_enabled else 0
    )


@pytest.mark.parametrize("json_name", ["engram-config", "engram_config"])
@pytest.mark.parametrize("dotted_name", ["engram-config", "engram_config"])
@pytest.mark.parametrize("json_last", [False, True])
@pytest.mark.parametrize("inline", [False, True])
@pytest.mark.parametrize("field", ["dp_shared_memory", "embedding_across_dp"])
@pytest.mark.parametrize(
    "version,legacy_env,json_offload,dotted_offload,enabled",
    [
        ("0.30.0", "0", True, None, False),
        ("0.30.0", "1", False, None, True),
        ("0.32.0", "0", False, None, True),
        ("0.32.0", "0", True, False, False),
        ("0.30.0", "0", False, True, True),
        ("0.30.0", "1", True, False, False),
    ],
)
def test_other_engram_dotted_fields_replace_json_offload(
    json_name,
    dotted_name,
    json_last,
    inline,
    field,
    version,
    legacy_env,
    json_offload,
    dotted_offload,
    enabled,
):
    json_args = [(f"--{json_name}", json.dumps({"cpu_offload": json_offload}))]
    dotted_args = [(f"--{dotted_name}.{field}", "false")]
    if dotted_offload is not None:
        dotted_args.append((f"--{dotted_name}.cpu_offload", json.dumps(dotted_offload)))
    pairs = dotted_args + json_args if json_last else json_args + dotted_args
    parameters = (
        [f"{key}={value}" for key, value in pairs]
        if inline
        else [arg for pair in pairs for arg in pair]
    )
    m = model(
        backend_version=version,
        env={"VLLM_PLE_CPU_OFFLOAD": legacy_env},
        backend_parameters=parameters,
    )
    assert estimate_offloaded_lookup_bytes(m, lookup_config()) == (
        TABLE_BYTES if enabled else 0
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("manual", [False, True])
async def test_mixed_engram_config_resident_weights_affect_admission(config, manual):
    worker = linux_nvidia_1_4090_24gx1(reserved=False)
    m = model(
        backend_version="0.30.0",
        env={"VLLM_PLE_CPU_OFFLOAD": "0"},
        backend_parameters=[
            '--engram-config={"cpu_offload":true}',
            "--engram-config.dp_shared_memory=false",
        ],
        gpu_selector=(
            GPUSelector(gpu_ids=[f"{worker.name}:cuda:0"], gpus_per_replica=1)
            if manual
            else None
        ),
    )
    selector = VLLMResourceFitSelector(config, m, [], resource_view=ResourceView([]))
    with (
        patch(
            "gpustack.policies.candidate_selectors.base_candidate_selector.get_pretrained_config_with_workers",
            new=AsyncMock(return_value=lookup_config()),
        ),
        patch("gpustack.policies.utils.get_model_weight_size", return_value=24 * GIB),
    ):
        candidates = await selector.select_candidates([worker])
    assert selector._vram_claim == int(24 * GIB * 1.2 + 2 * GIB)
    if not manual:
        assert candidates == []
        return
    assert len(candidates) == 1
    assert candidates[0].overcommit
    assert not candidates[0].computed_resource_claim.ram


@pytest.mark.parametrize(
    "version",
    [
        None,
        "",
        "0.30.0",
        "0.31.0",
        "0.32.0",
        "0.33.0",
        "v0.33.0",
        "0.33.1",
        "0.34.0",
        "0.100.0",
        "1.0.0",
    ],
)
@pytest.mark.parametrize("legacy_env", ["0", "1", "invalid"])
@pytest.mark.parametrize("explicit", [None, False, True])
@pytest.mark.parametrize(
    "architecture", ["DeepseekV41ForCausalLM", "Qwen4ExpForCausalLM"]
)
def test_offload_defaults_follow_backend_versions(
    version, legacy_env, explicit, architecture
):
    parameters = (
        []
        if explicit is None
        else [f"--engram_config.cpu_offload={json.dumps(explicit)}"]
    )
    m = model(
        backend_version=version,
        env={"VLLM_PLE_CPU_OFFLOAD": legacy_env},
        backend_parameters=parameters,
    )
    if explicit is not None:
        enabled = explicit
    elif version == "0.30.0":
        enabled = legacy_env == "1"
    else:
        enabled = True
    assert estimate_offloaded_lookup_bytes(m, lookup_config(architecture)) == (
        TABLE_BYTES if enabled else 0
    )


def test_offload_is_limited_to_supported_models_and_backends():
    m = model(env={"VLLM_PLE_CPU_OFFLOAD": "1"})
    assert get_offloaded_lookup_table_kind(m, ["Qwen3ForCausalLM"]) is None
    m.backend = BackendEnum.SGLANG
    assert get_offloaded_lookup_table_kind(m, ARCHITECTURES) is None


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("catalog/DeepSeek-V4.1-Flash-1017.json", 202758032400),
        # Table metadata from nvidia/DeepSeek-V4.1-Flash-NVFP4 config.json.
        ("deepseek_engram_mxfp4.json", 104451107600),
        ("qwen_ple.json", 51200000000),
    ],
)
@pytest.mark.parametrize("decoded", [False, True])
def test_published_model_dimensions(filename, expected, decoded):
    data = json.loads(
        (Path(__file__).parents[1] / "fixtures/model_configs" / filename).read_text()
    )
    config = safe_pretrained_config_from_dict(data) if decoded else data
    m = model(env={"VLLM_PLE_CPU_OFFLOAD": "1"})
    assert estimate_offloaded_lookup_bytes(m, config) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "version",
    [None, "", "0.32.0", "0.33.0", "v0.33.0", "0.34.0", "1.0.0", "0.33.0-custom"],
)
@pytest.mark.parametrize("image", [None, "custom:latest"])
async def test_backend_version_estimates_qwen_ple_resources(version, image):
    data = json.loads(
        (Path(__file__).parents[1] / "fixtures/model_configs/qwen_ple.json").read_text()
    )
    m = model(backend_version=version, image_name=image)
    offloaded = estimate_offloaded_lookup_bytes(m, data)
    with patch(
        "gpustack.policies.utils.get_model_weight_size", return_value=185523317458
    ):
        vram = await estimate_model_vram(m, offloaded_weights=offloaded)
    assert round(vram / GIB, 2) == 152.12
    assert get_model_ram_claim(m, offloaded) == 51200000000
    # Estimation preserves the runtime version selection, including Auto.
    assert m.backend_version == version


@pytest.mark.parametrize("nested", [False, True])
def test_text_only_and_multimodal_configurations(nested):
    root = lookup_config("Qwen4ExpForCausalLM")
    config = root if nested else root.text_config
    config.architectures = root.architectures
    config.quantization_config = root.quantization_config
    assert (
        estimate_offloaded_lookup_bytes(model(backend_version="0.30.0"), config)
        == TABLE_BYTES
    )


@pytest.mark.parametrize(
    "architecture", ["Qwen4ExpForConditionalGeneration", "DeepseekV41ForCausalLM"]
)
@pytest.mark.parametrize(
    "field,value",
    [
        ("quant_method", "gptq"),
        ("quant_method", "modelopt"),
        ("quant_method", "compressed-tensors"),
    ],
)
def test_unsupported_quantization_preserves_full_weight_estimate(
    architecture, field, value
):
    config = lookup_config(architecture)
    config.quantization_config[field] = value
    assert estimate_offloaded_lookup_bytes(model(backend_version="0.30.0"), config) == 0


@pytest.mark.parametrize("table_format,multiplier", [("fp8", 33), ("mxfp4", 17)])
@pytest.mark.parametrize("details", [False, True])
def test_engram_table_format_is_independent_of_expert_precision(
    table_format, multiplier, details
):
    config = lookup_config()
    config.quantization_config.update(
        expert_dtype="fp4", moe_quant_algo="NVFP4", engram_embed_quant=table_format
    )
    if details:
        config.quantization_config.update(
            engram_embed_block_size=32,
            engram_embed_scale_dtype="ue8m0",
            engram_embed_layers=[1],
        )
    assert estimate_offloaded_lookup_bytes(model(), config) == multiplier * 2**28


@pytest.mark.parametrize(
    "field,value",
    [
        ("engram_embed_quant", "nvfp4"),
        ("engram_embed_quant", "int4"),
        ("engram_embed_quant", "unknown"),
        ("engram_embed_quant", None),
        ("engram_embed_quant", {}),
        ("engram_embed_block_size", 16),
        ("engram_embed_block_size", "32"),
        ("engram_embed_scale_dtype", "float16"),
        ("engram_embed_layers", []),
        ("engram_embed_layers", [2]),
        ("engram_embed_layers", [1, 2]),
        ("engram_embed_layers", [1, 1]),
        ("engram_embed_layers", [True]),
    ],
)
def test_unsupported_engram_table_metadata_falls_back(field, value, caplog):
    config = lookup_config()
    config.quantization_config.update(engram_embed_quant="mxfp4")
    config.quantization_config[field] = value
    assert estimate_offloaded_lookup_bytes(model(), config) == 0
    assert "using the full weight size for VRAM" in caplog.text
    assert "table RAM is not estimated" in caplog.text


@pytest.mark.parametrize("location", ["root", "text", "quantization"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("engram_quant", "nvfp4"),
        ("engram_quant", "int4"),
        ("engram_quant_block_size", 16),
        ("engram_quant_global_scale", {"1": 0.004092}),
    ],
)
def test_custom_engram_markers_prevent_backbone_fp8_deduction(location, field, value):
    config = lookup_config()
    if location == "quantization":
        config.quantization_config[field] = value
    else:
        setattr(config if location == "root" else config.text_config, field, value)
    assert estimate_offloaded_lookup_bytes(model(), config) == 0


@pytest.mark.parametrize(
    "architecture,env",
    [
        ("DeepseekV41ForCausalLM", {"DSV41_ENGRAM_DTYPE": "int4"}),
        ("Qwen4ExpForCausalLM", {"VLLM_PLE_QUANT_DIR": "/tables"}),
        ("Qwen4ExpForCausalLM", {"VLLM_PLE_QUANT_RESIDENT": "1"}),
    ],
)
def test_custom_table_loaders_require_resource_overrides(architecture, env):
    config = lookup_config(architecture)
    assert estimate_offloaded_lookup_bytes(model(env=env), config) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("table_format", ["fp8", "mxfp4", "nvfp4", "int4"])
async def test_engram_table_storage_controls_vram_and_ram_admission(
    config, table_format
):
    worker = linux_nvidia_1_4090_24gx1()
    worker.status.memory.total = 5 * GIB
    worker.system_reserved.ram = 0
    metadata = lookup_config()
    metadata.quantization_config["engram_embed_quant"] = table_format
    selector = VLLMResourceFitSelector(
        config, model(), [], resource_view=ResourceView([])
    )
    with (
        patch(
            "gpustack.policies.candidate_selectors.base_candidate_selector.get_pretrained_config_with_workers",
            new=AsyncMock(return_value=metadata),
        ),
        patch("gpustack.policies.utils.get_model_weight_size", return_value=20 * GIB),
    ):
        candidates = await selector.select_candidates([worker])
    offloaded = {"fp8": TABLE_BYTES, "mxfp4": 17 * 2**28}.get(table_format, 0)
    assert selector._vram_claim == int((20 * GIB - offloaded) * 1.2) + 2 * GIB
    assert selector.get_resource_claim().ram == offloaded
    assert bool(candidates) == (table_format == "mxfp4")
    if candidates:
        assert candidates[0].computed_resource_claim.ram == offloaded
        assert not candidates[0].overcommit


@pytest.mark.parametrize(
    "field,value",
    [
        ("engram_num_embeddings", None),
        ("engram_num_embeddings", [1, 2]),
        ("engram_num_embeddings", [-1]),
        ("engram_num_embeddings", [True]),
        ("engram_head_dim", 0),
        ("engram_head_dim", 31),
        ("engram_layer_ids", [1, 1]),
    ],
)
def test_incomplete_engram_dimensions_fall_back(field, value):
    config = lookup_config()
    setattr(config.text_config, field, value)
    assert estimate_offloaded_lookup_bytes(model(backend_version="0.30.0"), config) == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("ple_embed_dim", None),
        ("ple_embed_dim", 0),
        ("ple_embed_dim", 255),
        ("ngram_vocab_size_base", "20000000"),
        ("ngram_vocab_size_base", -1),
        ("ngram_size", 1),
        ("heads_per_ngram", 0),
        ("ple_layer_ids", [0]),
    ],
)
def test_incomplete_ple_dimensions_fall_back(field, value):
    config = lookup_config("Qwen4ExpForConditionalGeneration")
    setattr(config.text_config, field, value)
    assert estimate_offloaded_lookup_bytes(model(backend_version="0.30.0"), config) == 0


@pytest.mark.parametrize(
    "architecture,field",
    [
        ("Qwen4ExpForCausalLM", "ple_layer_ids"),
        ("DeepseekV41ForCausalLM", "engram_layer_ids"),
    ],
)
def test_models_without_lookup_layers_need_no_deduction(architecture, field):
    config = lookup_config(architecture)
    setattr(config.text_config, field, [])
    assert estimate_offloaded_lookup_bytes(model(backend_version="0.30.0"), config) == 0


@pytest.mark.parametrize(
    "excluded,expected",
    [
        ([], TABLE_BYTES),
        (["model.language_model.layers.1.ple.key_proj"], TABLE_BYTES),
        (
            ["model.language_model.layers.1.ple.ple_embedding.ngram_embedding"],
            TABLE_BYTES * 2,
        ),
        (
            ["model.language_model.layers.*.ple.ple_embedding.ngram_embedding"],
            TABLE_BYTES,
        ),
        (["model.language_model.layers.1.ple"], TABLE_BYTES),
        (["ple.ple_embedding.ngram_embedding"], TABLE_BYTES),
        (
            [
                "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"
            ],
            TABLE_BYTES * 2,
        ),
        (["re:.*ple.*"], TABLE_BYTES),
    ],
)
@pytest.mark.parametrize("key", ["modules_to_not_convert", "ignored_layers"])
def test_ple_precision_respects_quantization_exclusions(key, excluded, expected):
    config = lookup_config("Qwen4ExpForConditionalGeneration")
    config.quantization_config[key] = excluded
    before = deepcopy(config)
    assert (
        estimate_offloaded_lookup_bytes(model(backend_version="0.30.0"), config)
        == expected
    )
    assert config == before


@pytest.mark.parametrize("ignored", [None, [], ["lm_head"]])
@pytest.mark.parametrize("shard", [False, True])
def test_ple_ignored_layers_take_precedence(ignored, shard):
    config = lookup_config("Qwen4ExpForConditionalGeneration")
    prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
    config.quantization_config.update(
        ignored_layers=ignored,
        modules_to_not_convert=[prefix + ".shard_0.weight" if shard else prefix],
    )
    assert estimate_offloaded_lookup_bytes(
        model(backend_version="0.30.0"), config
    ) == TABLE_BYTES * (1 if ignored else 2)


@pytest.mark.parametrize(
    "dtype,multiplier", [("bfloat16", 2), ("float16", 2), ("float32", 4), (None, 0)]
)
def test_unquantized_ple_uses_checkpoint_dtype(dtype, multiplier):
    config = lookup_config("Qwen4ExpForCausalLM")
    config.quantization_config = None
    config.text_config.dtype = dtype
    assert (
        estimate_offloaded_lookup_bytes(model(backend_version="0.30.0"), config)
        == TABLE_BYTES * multiplier
    )


def test_explicit_ple_fp8_dtype_overrides_backbone_precision():
    config = lookup_config("Qwen4ExpForCausalLM")
    config.quantization_config = None
    config.text_config.ple_embedding_dtype = "float8_e4m3fn"
    assert (
        estimate_offloaded_lookup_bytes(model(backend_version="0.30.0"), config)
        == TABLE_BYTES
    )


def test_per_layer_ple_precision_and_nested_quantization():
    config = lookup_config("Qwen4ExpForConditionalGeneration")
    config.text_config.ple_layer_ids = [2, 4]
    config.text_config.quantization_config = {
        "quant_method": "fp8",
        "ignored_layers": [
            "model.language_model.layers.3.ple.ple_embedding.ngram_embedding"
        ],
    }
    config.quantization_config = {"quant_method": "unsupported"}
    assert (
        estimate_offloaded_lookup_bytes(model(backend_version="0.30.0"), config)
        == TABLE_BYTES * 3
    )


@pytest.mark.parametrize(
    "parameters",
    [
        ["--quantization=awq"],
        ['--hf-overrides={"ngram_vocab_size_base":1}'],
        ["--dtype=float32"],
    ],
)
def test_incompatible_runtime_overrides_fall_back(parameters):
    config = lookup_config("Qwen4ExpForCausalLM")
    config.quantization_config = None
    m = model(backend_version="0.30.0", backend_parameters=parameters)
    assert estimate_offloaded_lookup_bytes(m, config) == 0


@pytest.mark.parametrize(
    "architecture", ["DeepseekV41ForCausalLM", "Qwen4ExpForCausalLM"]
)
@pytest.mark.parametrize("name", ["hf-overrides", "hf_overrides"])
@pytest.mark.parametrize("dotted", [False, True])
@pytest.mark.parametrize("syntax", ["equals", "separate", "combined"])
def test_hf_override_aliases_disable_lookup_deduction(
    architecture, name, dotted, syntax
):
    config = lookup_config(architecture)
    config.text_config.ple_embed_dim *= 2
    config.text_config.engram_num_embeddings = [2**29]
    m = model(backend_version="0.32.0")
    assert estimate_offloaded_lookup_bytes(m, config) == 2 * TABLE_BYTES
    option = f"--{name}.text_config.ple_embed_dim" if dotted else f"--{name}"
    value = "512" if dotted else '{"text_config":{"ple_embed_dim":512}}'
    m.backend_parameters = {
        "equals": [f"{option}={value}"],
        "separate": [option, value],
        "combined": [f"{option} {value}"],
    }[syntax]
    # Runtime architecture overrides invalidate the dimensions used for deduction.
    assert estimate_offloaded_lookup_bytes(m, config) == 0


def test_engram_requires_native_scale_format():
    config = lookup_config()
    config.quantization_config["scale_fmt"] = "float32"
    assert estimate_offloaded_lookup_bytes(model(backend_version="0.30.0"), config) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("offloaded,expected", [(0, 122), (60, 50), (110, 122)])
async def test_vram_formula_only_uses_resident_weights(offloaded, expected):
    with patch("gpustack.policies.utils.get_model_weight_size", return_value=100 * GIB):
        assert (
            await estimate_model_vram(model(), offloaded_weights=offloaded * GIB)
            == expected * GIB
        )


@pytest.mark.asyncio
async def test_declared_vram_has_priority():
    with patch("gpustack.policies.utils.get_model_weight_size") as read:
        assert (
            await estimate_model_vram(
                model(env={"GPUSTACK_MODEL_VRAM_CLAIM": "123"}),
                offloaded_weights=60 * GIB,
            )
            == 123
        )
        read.assert_not_called()


def test_declaring_both_resources_skips_config_estimation():
    m = model(
        env={"GPUSTACK_MODEL_VRAM_CLAIM": "123", "GPUSTACK_MODEL_RAM_CLAIM": "456"}
    )
    with patch(
        "gpustack.policies.lookup_table.get_offloaded_lookup_table_kind"
    ) as resolve:
        assert estimate_offloaded_lookup_bytes(m, None) == 0
        resolve.assert_not_called()


@pytest.mark.parametrize(
    "declared,expected", [(None, 12), ("3", 3), ("0", 0), ("-1", 12), ("invalid", 12)]
)
def test_ram_override_is_a_per_worker_total(declared, expected):
    env = {"GPUSTACK_MODEL_RAM_CLAIM": declared} if declared is not None else {}
    m = model(env=env)
    assert get_model_ram_claim(m, offloaded_ram=12) == expected
    assert get_computed_ram_claim(m, {0: 20}, offloaded_ram=12) == expected


@pytest.mark.parametrize(
    "cache,expected",
    [
        (ExtendedKVCacheConfig(enabled=True, ram_size=4), 16 * GIB),
        (ExtendedKVCacheConfig(enabled=True, ram_ratio=0.5), 22 * GIB),
        (
            ExtendedKVCacheConfig(enabled=True, mode="shared", cache_service_id=1),
            12 * GIB,
        ),
    ],
)
def test_offload_ram_adds_only_local_cache(cache, expected):
    assert (
        get_computed_ram_claim(
            model(extended_kv_cache=cache), {0: 20 * GIB}, offloaded_ram=12 * GIB
        )
        == expected
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "architecture,manual,ram,enabled,expected",
    [
        ("DeepseekV41ForCausalLM", False, 32, True, True),
        ("Qwen4ExpForConditionalGeneration", False, 32, True, True),
        ("DeepseekV41ForCausalLM", False, 8, True, False),
        ("DeepseekV41ForCausalLM", True, 8, True, True),
        ("DeepseekV41ForCausalLM", True, 32, True, True),
        ("DeepseekV41ForCausalLM", False, 32, False, False),
    ],
)
async def test_scheduler_places_offloaded_models_and_reserves_ram(
    config, architecture, manual, ram, enabled, expected
):
    worker = linux_nvidia_1_4090_24gx1()
    worker.status.memory.total = ram * GIB
    worker.system_reserved.ram = 0
    m = model(backend_version="0.30.0", env={"VLLM_PLE_CPU_OFFLOAD": str(int(enabled))})
    if manual:
        m.gpu_selector = GPUSelector(
            gpu_ids=[f"{worker.name}:cuda:0"], gpus_per_replica=1
        )
    selector = VLLMResourceFitSelector(config, m, [], resource_view=ResourceView([]))
    pretrained = lookup_config(architecture)
    with (
        patch(
            "gpustack.policies.candidate_selectors.base_candidate_selector.get_pretrained_config_with_workers",
            new=AsyncMock(return_value=pretrained),
        ),
        patch("gpustack.policies.utils.get_model_weight_size", return_value=20 * GIB),
    ):
        candidates = await selector.select_candidates([worker])
    assert bool(candidates) is expected
    if expected:
        assert bool(candidates[0].overcommit) is (manual and ram * GIB < TABLE_RAM)
        assert candidates[0].computed_resource_claim.ram == TABLE_RAM
        assert selector.get_resource_claim().ram == TABLE_RAM
        assert (
            ResourceView(
                [
                    SimpleNamespace(
                        worker_id=worker.id,
                        gpu_type="cuda",
                        gpu_indexes=candidates[0].gpu_indexes,
                        computed_resource_claim=candidates[0].computed_resource_claim,
                        distributed_servers=None,
                    )
                ]
            )
            .allocated(worker.id)
            .ram
            == TABLE_RAM
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("worker_count", [1, 2])
@pytest.mark.parametrize("gpus_per_worker", [1, 2])
async def test_ram_refusal_preserves_resource_summary_and_all_worker_reasons(
    config, worker_count, gpus_per_worker
):
    workers = [linux_nvidia_1_4090_24gx1(), linux_nvidia_3_4090_24gx1()][:worker_count]
    for worker in workers:
        worker.status.memory.total = 4 * GIB
        worker.system_reserved.ram = 0
        gpu = worker.status.gpu_devices[0]
        worker.status.gpu_devices = []
        for index in range(gpus_per_worker):
            device = deepcopy(gpu)
            device.index = index
            worker.status.gpu_devices.append(device)
    m = model(backend_parameters=["--tp=1"])
    selector = VLLMResourceFitSelector(config, m, [], resource_view=ResourceView([]))
    with (
        patch(
            "gpustack.policies.candidate_selectors.base_candidate_selector.get_pretrained_config_with_workers",
            new=AsyncMock(return_value=lookup_config()),
        ),
        patch("gpustack.policies.utils.get_model_weight_size", return_value=20 * GIB),
    ):
        assert await selector.select_candidates(workers) == []

    messages = selector.get_messages()
    assert len(messages) == 1
    message = messages[0]
    summary = "The model requires approximately 16.1 GiB of VRAM and 8.25 GiB of RAM."
    assert message.count(summary) == 1
    assert "at least 17.89 GiB of total VRAM" in message
    assert "each GPU needs 90% of allocatable VRAM" in message
    assert "The current available GPU only has" in message
    for worker in workers:
        reason = (
            f"Worker {worker.name} requires 8.25 GiB of RAM, "
            "but only 4.0 GiB is allocatable."
        )
        # Rejected GPU choices on the same worker must not duplicate its reason.
        assert message.count(reason) == 1
        assert message.index(summary) < message.index(reason)


@pytest.mark.asyncio
@pytest.mark.parametrize("manual", [False, True])
@pytest.mark.parametrize("ram", [1, 32])
async def test_multinode_reserves_ram_on_every_worker(config, manual, ram):
    workers = [linux_nvidia_1_4090_24gx1(), linux_nvidia_3_4090_24gx1()]
    for worker in workers:
        worker.status.memory.total = ram * GIB
        worker.system_reserved.ram = 0
    m = model(
        env={"VLLM_PLE_CPU_OFFLOAD": "1"}, distributed_inference_across_workers=True
    )
    if manual:
        m.gpu_selector = GPUSelector(
            gpu_ids=[f"{worker.name}:cuda:0" for worker in workers], gpus_per_replica=2
        )
    selector = VLLMResourceFitSelector(config, m, [], resource_view=ResourceView([]))
    with (
        patch(
            "gpustack.policies.candidate_selectors.base_candidate_selector.get_pretrained_config_with_workers",
            new=AsyncMock(return_value=lookup_config()),
        ),
        patch("gpustack.policies.utils.get_model_weight_size", return_value=40 * GIB),
    ):
        candidates = await selector.select_candidates(workers)
    if ram == 1 and not manual:
        assert candidates == []
        assert any("RAM" in message for message in selector.get_messages())
        return
    assert len(candidates) == 1
    assert bool(candidates[0].overcommit) is (ram == 1)
    assert candidates[0].computed_resource_claim.ram == TABLE_RAM
    assert candidates[0].subordinate_workers[0].computed_resource_claim.ram == TABLE_RAM
    instance = SimpleNamespace(
        worker_id=candidates[0].worker.id,
        gpu_type="cuda",
        gpu_indexes=candidates[0].gpu_indexes,
        computed_resource_claim=candidates[0].computed_resource_claim,
        distributed_servers=SimpleNamespace(
            subordinate_workers=candidates[0].subordinate_workers
        ),
    )
    resources = ResourceView([instance])
    for worker in workers:
        assert resources.allocated(worker.id).ram == TABLE_RAM
        assert resources.allocatable(worker).ram == max(0, ram * GIB - TABLE_RAM)


@pytest.mark.asyncio
@pytest.mark.parametrize("limited_worker_position", [0, 1, 2])
@pytest.mark.parametrize("limited_worker_ram", [8, 32])
@pytest.mark.parametrize("available_workers", [2, 3])
async def test_manual_pipeline_requires_full_gpu_count_regardless_of_table_ram(
    config, limited_worker_position, limited_worker_ram, available_workers
):
    workers = [linux_nvidia_1_4090_24gx1(), linux_nvidia_3_4090_24gx1()]
    worker = deepcopy(workers[0])
    worker.id = 100
    worker.name = "pipeline-worker"
    worker.ip = "192.168.100.100"
    workers.append(worker)
    for worker in workers:
        worker.status.memory.total = 32 * GIB
        worker.system_reserved.ram = 0
    workers[limited_worker_position].status.memory.total = limited_worker_ram * GIB
    m = model(
        env={"VLLM_PLE_CPU_OFFLOAD": "1"},
        distributed_inference_across_workers=True,
        backend_parameters=["--tensor-parallel-size=1", "--pipeline-parallel-size=3"],
        gpu_selector=GPUSelector(
            gpu_ids=[f"{worker.name}:cuda:0" for worker in workers],
            gpus_per_replica=3,
        ),
    )
    selector = VLLMResourceFitSelector(config, m, [], resource_view=ResourceView([]))
    with (
        patch(
            "gpustack.policies.candidate_selectors.base_candidate_selector.get_pretrained_config_with_workers",
            new=AsyncMock(return_value=lookup_config()),
        ),
        patch("gpustack.policies.utils.get_model_weight_size", return_value=40 * GIB),
    ):
        candidates = await selector.select_candidates(workers[:available_workers])
    if available_workers < 3:
        assert candidates == []
        return
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.overcommit is (limited_worker_ram == 8)
    if candidate.overcommit:
        assert any(
            workers[limited_worker_position].name in message and "RAM" in message
            for message in selector.get_messages()
        )
    claims = {candidate.worker.id: candidate.computed_resource_claim}
    claims.update(
        (worker.worker_id, worker.computed_resource_claim)
        for worker in candidate.subordinate_workers or []
    )
    assert sum(len(claim.vram) for claim in claims.values()) == 3
    assert claims.keys() == {worker.id for worker in workers}
    assert all(claim.ram == TABLE_RAM for claim in claims.values())


@pytest.mark.asyncio
@pytest.mark.parametrize("dual_worker_ram", [8, 32, 64])
@pytest.mark.parametrize("cache_ratio", [0, 1])
@pytest.mark.parametrize("reverse_order", [False, True])
async def test_manual_selection_keeps_single_worker_candidate_despite_ram_shortage(
    config, dual_worker_ram, cache_ratio, reverse_order
):
    workers = [linux_nvidia_1_4090_24gx1(), linux_nvidia_3_4090_24gx1()]
    for worker in workers:
        worker.status.memory.total = 32 * GIB
        worker.system_reserved.ram = 0
    dual_worker = deepcopy(workers[0])
    dual_worker.id = 100
    dual_worker.name = "dual-gpu-worker"
    dual_worker.ip = "192.168.100.100"
    dual_worker.status.memory.total = dual_worker_ram * GIB
    dual_worker.status.gpu_devices.append(deepcopy(dual_worker.status.gpu_devices[0]))
    dual_worker.status.gpu_devices[1].index = 1
    mixed_workers = [dual_worker, *workers]
    if reverse_order:
        mixed_workers.reverse()
    dual_fits = (
        TABLE_RAM
        + cache_ratio
        * sum(int(gpu.memory.total * 0.9) for gpu in dual_worker.status.gpu_devices)
        <= dual_worker_ram * GIB
    )
    for pool in [workers, mixed_workers]:
        m = model(
            env={"VLLM_PLE_CPU_OFFLOAD": "1"},
            distributed_inference_across_workers=True,
            gpu_selector=GPUSelector(
                gpu_ids=[
                    f"{worker.name}:cuda:{gpu.index}"
                    for worker in pool
                    for gpu in worker.status.gpu_devices
                ],
                gpus_per_replica=2,
            ),
            extended_kv_cache=(
                ExtendedKVCacheConfig(enabled=True, ram_ratio=cache_ratio)
                if cache_ratio
                else None
            ),
        )
        selector = VLLMResourceFitSelector(
            config, m, [], resource_view=ResourceView([])
        )
        with (
            patch(
                "gpustack.policies.candidate_selectors.base_candidate_selector.get_pretrained_config_with_workers",
                new=AsyncMock(return_value=lookup_config()),
            ),
            patch(
                "gpustack.policies.utils.get_model_weight_size", return_value=40 * GIB
            ),
        ):
            candidates = await selector.select_candidates(pool)
        assert len(candidates) == 1
        candidate = candidates[0]
        claims = {candidate.worker.id: candidate.computed_resource_claim}
        claims.update(
            (worker.worker_id, worker.computed_resource_claim)
            for worker in candidate.subordinate_workers or []
        )
        assert candidate.overcommit == (len(pool) == 3 and not dual_fits)
        if len(pool) == 3:
            assert claims.keys() == {dual_worker.id}
        else:
            assert claims.keys() == {worker.id for worker in workers}
        assert sum(len(claim.vram) for claim in claims.values()) == 2
        for worker in pool:
            if worker.id in claims:
                claim = claims[worker.id]
                assert claim.ram == TABLE_RAM + sum(claim.vram.values()) * cache_ratio


@pytest.mark.asyncio
@pytest.mark.parametrize("low_ram_position", [0, 1, 2])
@pytest.mark.parametrize("explicit_tp", [False, True])
@pytest.mark.parametrize(
    "cache",
    [
        ExtendedKVCacheConfig(enabled=True, ram_size=20),
        ExtendedKVCacheConfig(enabled=True, ram_ratio=1),
    ],
)
async def test_auto_multinode_preserves_cache_ram_admission(
    config, low_ram_position, explicit_tp, cache
):
    workers = [linux_nvidia_1_4090_24gx1(), linux_nvidia_3_4090_24gx1()]
    for worker in workers:
        worker.status.memory.total = 32 * GIB
        worker.system_reserved.ram = 0
    low_ram_worker = deepcopy(workers[0])
    low_ram_worker.id = 100
    low_ram_worker.name = "low-ram-worker"
    low_ram_worker.ip = "192.168.100.100"
    low_ram_worker.status.memory.total = 16 * GIB
    m = model(
        env={"VLLM_PLE_CPU_OFFLOAD": "1"},
        distributed_inference_across_workers=True,
        extended_kv_cache=cache,
        backend_parameters=["--tensor-parallel-size=2"] if explicit_tp else [],
    )
    mixed_workers = list(workers)
    mixed_workers.insert(low_ram_position, low_ram_worker)
    for pool in [workers, mixed_workers]:
        selector = VLLMResourceFitSelector(
            config, m, [], resource_view=ResourceView([])
        )
        with (
            patch(
                "gpustack.policies.candidate_selectors.base_candidate_selector.get_pretrained_config_with_workers",
                new=AsyncMock(return_value=lookup_config()),
            ),
            patch(
                "gpustack.policies.utils.get_model_weight_size", return_value=40 * GIB
            ),
        ):
            candidates = await selector.select_candidates(pool)
        assert len(candidates) == 1
        candidate = candidates[0]
        assert {candidate.worker.id} | {
            worker.worker_id for worker in candidate.subordinate_workers
        } == {worker.id for worker in workers}
        claims = [candidate.computed_resource_claim] + [
            worker.computed_resource_claim for worker in candidate.subordinate_workers
        ]
        for claim in claims:
            assert claim.ram == get_computed_ram_claim(
                m, claim.vram, offloaded_ram=TABLE_RAM
            )
            assert 16 * GIB < claim.ram <= 32 * GIB


@pytest.mark.asyncio
@pytest.mark.parametrize("small_worker_ram", [127, 128, 256])
@pytest.mark.parametrize("reverse_order", [False, True])
@pytest.mark.parametrize(
    "cache",
    [
        ExtendedKVCacheConfig(enabled=True, ram_size=128),
        ExtendedKVCacheConfig(enabled=True, ram_ratio=1),
    ],
)
async def test_single_worker_min_gpu_count_preserves_cache_ram_admission(
    config, small_worker_ram, reverse_order, cache
):
    small_worker = linux_nvidia_5_a100_80gx2(reserved=False)
    large_worker = linux_nvidia_12_A40_48gx2(reserved=False)
    large_worker.status.gpu_devices = [
        deepcopy(large_worker.status.gpu_devices[0]) for _ in range(4)
    ]
    for worker, ram, vram in [
        (small_worker, small_worker_ram, 80),
        (large_worker, 256, 48),
    ]:
        worker.status.memory.total = ram * GIB
        for index, gpu in enumerate(worker.status.gpu_devices):
            gpu.index = index
            gpu.memory.total = vram * GIB
    m = model(
        env={"VLLM_PLE_CPU_OFFLOAD": "1"},
        extended_kv_cache=cache,
    )
    workers = [small_worker, large_worker]
    if reverse_order:
        workers.reverse()
    expected_worker = small_worker if small_worker_ram == 256 else large_worker
    for pool, expected in [([large_worker], large_worker), (workers, expected_worker)]:
        selector = VLLMResourceFitSelector(
            config, m, [], resource_view=ResourceView([])
        )
        with (
            patch(
                "gpustack.policies.candidate_selectors.base_candidate_selector.get_pretrained_config_with_workers",
                new=AsyncMock(return_value=lookup_config()),
            ),
            patch(
                "gpustack.policies.utils.get_model_weight_size", return_value=120 * GIB
            ),
        ):
            candidates = await selector.select_candidates(pool)
        assert len(candidates) == 1
        candidate = candidates[0]
        assert candidate.worker.id == expected.id
        assert len(candidate.gpu_indexes) == len(expected.status.gpu_devices)
        claim = candidate.computed_resource_claim
        assert claim.ram == get_computed_ram_claim(
            m, claim.vram, offloaded_ram=TABLE_RAM
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("replicas", [1, 2])
@pytest.mark.parametrize("weight_size", [20, 40])
async def test_manual_vram_overcommit_preserved_with_insufficient_table_ram(
    config, replicas, weight_size
):
    worker = linux_nvidia_1_4090_24gx1()
    worker.status.memory.total = GIB
    worker.system_reserved.ram = 0
    m = model(
        replicas=replicas,
        env={"VLLM_PLE_CPU_OFFLOAD": "1"},
        gpu_selector=GPUSelector(gpu_ids=[f"{worker.name}:cuda:0"], gpus_per_replica=1),
    )
    selector = VLLMResourceFitSelector(config, m, [], resource_view=ResourceView([]))
    with (
        patch(
            "gpustack.policies.candidate_selectors.base_candidate_selector.get_pretrained_config_with_workers",
            new=AsyncMock(return_value=lookup_config()),
        ),
        patch(
            "gpustack.policies.utils.get_model_weight_size",
            return_value=weight_size * GIB,
        ),
    ):
        candidates = await selector.select_candidates([worker])
    if replicas > 1:
        assert candidates == []
        return
    assert len(candidates) == 1
    assert candidates[0].overcommit
    assert candidates[0].computed_resource_claim.ram == TABLE_RAM
    assert any(worker.name in msg and "RAM" in msg for msg in selector.get_messages())


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["cache", "override"])
@pytest.mark.parametrize("replicas", [1, 2])
@pytest.mark.parametrize(
    "selector_class,backend",
    [
        (VLLMResourceFitSelector, BackendEnum.VLLM),
        (SGLangResourceFitSelector, BackendEnum.SGLANG),
    ],
)
async def test_manual_ram_overcommit_without_offloaded_tables(
    config, source, replicas, selector_class, backend
):
    worker = linux_nvidia_1_4090_24gx1()
    worker.status.memory.total = GIB
    worker.system_reserved.ram = 0
    m = model(
        replicas=replicas,
        env={"GPUSTACK_MODEL_VRAM_CLAIM": str(GIB)},
        gpu_selector=GPUSelector(gpu_ids=[f"{worker.name}:cuda:0"], gpus_per_replica=1),
    )
    if source == "cache":
        m.extended_kv_cache = ExtendedKVCacheConfig(enabled=True, ram_size=2)
    else:
        m.env["GPUSTACK_MODEL_RAM_CLAIM"] = str(2 * GIB)
    m.backend = backend
    selector = selector_class(config, m, [], resource_view=ResourceView([]))
    with patch(
        "gpustack.policies.candidate_selectors.base_candidate_selector.get_pretrained_config_with_workers",
        new=AsyncMock(
            return_value=SimpleNamespace(
                architectures=["Qwen3ForCausalLM"], num_attention_heads=8
            )
        ),
    ):
        candidates = await selector.select_candidates([worker])
    if replicas > 1:
        assert candidates == []
        return
    assert len(candidates) == 1
    assert candidates[0].overcommit
    assert candidates[0].computed_resource_claim.ram == 2 * GIB
    assert any(worker.name in msg and "RAM" in msg for msg in selector.get_messages())


@pytest.mark.asyncio
@pytest.mark.parametrize("replicas", [1, 2])
async def test_manual_selection_does_not_redistribute_gpus_for_ram(config, replicas):
    workers = [linux_nvidia_1_4090_24gx1(), linux_nvidia_3_4090_24gx1()]
    for worker in workers:
        worker.status.memory.total = 32 * GIB
        worker.system_reserved.ram = 0
        worker.status.gpu_devices.append(deepcopy(worker.status.gpu_devices[0]))
        worker.status.gpu_devices[1].index = 1
    m = model(
        replicas=replicas,
        env={"VLLM_PLE_CPU_OFFLOAD": "1"},
        distributed_inference_across_workers=True,
        extended_kv_cache=ExtendedKVCacheConfig(enabled=True, ram_ratio=1),
        gpu_selector=GPUSelector(
            gpu_ids=[
                f"{worker.name}:cuda:{index}"
                for worker in workers
                for index in range(2)
            ],
            gpus_per_replica=2,
        ),
    )
    selector = VLLMResourceFitSelector(config, m, [], resource_view=ResourceView([]))
    with (
        patch(
            "gpustack.policies.candidate_selectors.base_candidate_selector.get_pretrained_config_with_workers",
            new=AsyncMock(return_value=lookup_config()),
        ),
        patch("gpustack.policies.utils.get_model_weight_size", return_value=40 * GIB),
    ):
        candidates = await selector.select_candidates(workers)
    if replicas > 1:
        assert candidates == []
        return
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.overcommit
    assert len(candidate.gpu_indexes) == 2
    assert not candidate.subordinate_workers
    claim = candidate.computed_resource_claim
    assert claim.ram == TABLE_RAM + sum(claim.vram.values())
    assert claim.ram > 32 * GIB


@pytest.mark.asyncio
@pytest.mark.parametrize("ram,expected", [(3, False), (4, False), (13, True)])
async def test_auto_single_gpu_preserves_cache_ram_admission(config, ram, expected):
    worker = linux_nvidia_1_4090_24gx1()
    worker.status.memory.total = ram * GIB
    worker.system_reserved.ram = 0
    m = model(
        env={"VLLM_PLE_CPU_OFFLOAD": "1"},
        extended_kv_cache=ExtendedKVCacheConfig(enabled=True, ram_size=4),
    )
    selector = VLLMResourceFitSelector(config, m, [], resource_view=ResourceView([]))
    with (
        patch(
            "gpustack.policies.candidate_selectors.base_candidate_selector.get_pretrained_config_with_workers",
            new=AsyncMock(return_value=lookup_config()),
        ),
        patch("gpustack.policies.utils.get_model_weight_size", return_value=20 * GIB),
    ):
        candidates = await selector.select_candidates([worker])
    assert bool(candidates) is expected
    if expected:
        assert candidates[0].computed_resource_claim.ram == TABLE_RAM + 4 * GIB


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "architecture", ["DeepseekV41ForCausalLM", "Qwen4ExpForCausalLM"]
)
@pytest.mark.parametrize("dp", [1, 2])
@pytest.mark.parametrize("cache_size", [0, 4])
@pytest.mark.parametrize("available_delta", [-1, 0])
async def test_data_parallel_table_ram_assumes_sharing_without_margin(
    config, architecture, dp, cache_size, available_delta
):
    worker = linux_nvidia_5_a100_80gx2()
    expected_ram = TABLE_BYTES + cache_size * GIB
    worker.status.memory.total = expected_ram + available_delta
    worker.system_reserved.ram = 0
    m = model(
        backend_version="0.32.0",
        env={"VLLM_PLE_CPU_OFFLOAD": "1"},
        backend_parameters=[f"--data-parallel-size={dp}"],
        extended_kv_cache=(
            ExtendedKVCacheConfig(enabled=True, ram_size=cache_size)
            if cache_size
            else None
        ),
    )
    selector = VLLMResourceFitSelector(config, m, [], resource_view=ResourceView([]))
    with (
        patch(
            "gpustack.policies.candidate_selectors.base_candidate_selector.get_pretrained_config_with_workers",
            new=AsyncMock(return_value=lookup_config(architecture)),
        ),
        patch("gpustack.policies.utils.get_model_weight_size", return_value=20 * GIB),
    ):
        candidates = await selector.select_candidates([worker])
    if available_delta < 0:
        assert candidates == []
        return
    assert len(candidates) == 2 // dp
    assert all(
        candidate.computed_resource_claim.ram == expected_ram
        for candidate in candidates
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "selector_class,backend",
    [
        (VLLMResourceFitSelector, BackendEnum.VLLM),
        (SGLangResourceFitSelector, BackendEnum.SGLANG),
    ],
)
@pytest.mark.parametrize("claim,expected", [(8, True), (128, False)])
async def test_declared_ram_is_enforced_by_selectors(
    config, selector_class, backend, claim, expected
):
    worker = linux_nvidia_1_4090_24gx1()
    worker.status.memory.total = 64 * GIB
    worker.system_reserved.ram = 0
    m = model(
        env={
            "GPUSTACK_MODEL_VRAM_CLAIM": str(2 * GIB),
            "GPUSTACK_MODEL_RAM_CLAIM": str(claim * GIB),
        }
    )
    m.backend = backend
    selector = selector_class(config, m, [], resource_view=ResourceView([]))
    with patch(
        "gpustack.policies.candidate_selectors.base_candidate_selector.get_pretrained_config_with_workers",
        new=AsyncMock(
            return_value=SimpleNamespace(
                architectures=["Qwen3ForCausalLM"], num_attention_heads=8
            )
        ),
    ):
        candidates = await selector.select_candidates([worker])
    assert bool(candidates) is expected
    if expected:
        assert candidates[0].computed_resource_claim.ram == claim * GIB


@pytest.mark.asyncio
@pytest.mark.parametrize("executor", ["mp", "ray"])
@pytest.mark.parametrize("selected_counts", [(4, 4), (3, 1), (2, 2)])
@pytest.mark.parametrize("reverse_order", [False, True])
@pytest.mark.parametrize("pcp", [1, 2])
@pytest.mark.parametrize("replicas", [1, 2])
async def test_manual_selection_leaves_topology_validation_to_backend(
    config, executor, selected_counts, reverse_order, pcp, replicas
):
    workers = [linux_nvidia_1_4090_24gx1(), linux_nvidia_3_4090_24gx1()]
    for worker, count in zip(workers, selected_counts):
        worker.status.memory.total = 80 * GIB + (pcp - 1) * TABLE_RAM
        worker.system_reserved.ram = 0
        worker.status.gpu_devices = [
            deepcopy(worker.status.gpu_devices[0]) for _ in range(count)
        ]
        for index, gpu in enumerate(worker.status.gpu_devices):
            gpu.index = index
            gpu.memory.total = 24 * GIB
    if reverse_order:
        workers.reverse()
    m = model(
        replicas=replicas,
        env={"VLLM_PLE_CPU_OFFLOAD": "1"},
        backend_parameters=[
            f"--tp={4 // pcp}",
            f"--pcp={pcp}",
            f"--distributed-executor-backend={executor}",
        ],
        distributed_inference_across_workers=True,
        extended_kv_cache=ExtendedKVCacheConfig(enabled=True, ram_ratio=1),
        gpu_selector=GPUSelector(
            gpu_ids=[
                f"{worker.name}:cuda:{gpu.index}"
                for worker in workers
                for gpu in worker.status.gpu_devices
            ],
            gpus_per_replica=4,
        ),
    )
    selector = VLLMResourceFitSelector(config, m, [], resource_view=ResourceView([]))
    with (
        patch(
            "gpustack.policies.candidate_selectors.base_candidate_selector.get_pretrained_config_with_workers",
            new=AsyncMock(return_value=lookup_config()),
        ),
        patch("gpustack.policies.utils.get_model_weight_size", return_value=40 * GIB),
    ):
        candidates = await selector.select_candidates(workers)
    if replicas > 1 and selected_counts == (4, 4):
        # Multi-replica manual scheduling still requires resources to fit.
        assert candidates == []
        return
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.overcommit == (selected_counts == (4, 4))
    expected_counts = [4] if selected_counts == (4, 4) else list(selected_counts)
    assert [len(candidate.gpu_indexes)] + [
        len(worker.gpu_indexes) for worker in candidate.subordinate_workers or []
    ] == expected_counts


@pytest.mark.asyncio
@pytest.mark.parametrize("manual", [False, True])
async def test_mp_topology_admission_only_applies_to_automatic_selection(
    config, manual
):
    workers = [linux_nvidia_1_4090_24gx1(), linux_nvidia_3_4090_24gx1()]
    for worker in workers:
        worker.status.memory.total = 80 * GIB
        worker.system_reserved.ram = 0
        worker.status.gpu_devices.append(deepcopy(worker.status.gpu_devices[0]))
        worker.status.gpu_devices[1].index = 1
    m = model(
        env={"VLLM_PLE_CPU_OFFLOAD": "1"},
        backend_parameters=[
            "--tp=2",
            "--pcp=2",
            "--dpl=2",
            "--distributed-executor-backend=mp",
        ],
        distributed_inference_across_workers=True,
        gpu_selector=(
            GPUSelector(
                gpu_ids=[
                    f"{worker.name}:cuda:{index}"
                    for worker in workers
                    for index in range(2)
                ],
                gpus_per_replica=4,
            )
            if manual
            else None
        ),
    )
    selector = VLLMResourceFitSelector(config, m, [], resource_view=ResourceView([]))
    with (
        patch(
            "gpustack.policies.candidate_selectors.base_candidate_selector.get_pretrained_config_with_workers",
            new=AsyncMock(return_value=lookup_config()),
        ),
        patch("gpustack.policies.utils.get_model_weight_size", return_value=40 * GIB),
    ):
        candidates = await selector.select_candidates(workers)
    # Cross-node TP/PCP requires DPL=1, but manual placement defers that check.
    assert bool(candidates) is manual
    if manual:
        assert len(candidates) == 1
        assert not candidates[0].overcommit
        assert candidates[0].computed_resource_claim.ram == TABLE_RAM
        assert (
            candidates[0].subordinate_workers[0].computed_resource_claim.ram
            == TABLE_RAM
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse_order", [False, True])
@pytest.mark.parametrize("replicas", [1, 2])
async def test_manual_single_worker_keeps_gpu_subset(config, reverse_order, replicas):
    worker = linux_nvidia_1_4090_24gx1()
    worker.status.memory.total = 64 * GIB
    worker.system_reserved.ram = 0
    worker.status.gpu_devices = [
        deepcopy(worker.status.gpu_devices[0]) for _ in range(4)
    ]
    for index, (gpu, size) in enumerate(
        zip(worker.status.gpu_devices, [80, 80, 24, 24])
    ):
        gpu.index = index
        gpu.memory.total = size * GIB
    if reverse_order:
        worker.status.gpu_devices.reverse()
    m = model(
        replicas=replicas,
        env={"VLLM_PLE_CPU_OFFLOAD": "1"},
        extended_kv_cache=ExtendedKVCacheConfig(enabled=True, ram_ratio=1),
        gpu_selector=GPUSelector(
            gpu_ids=[
                f"{worker.name}:cuda:{gpu.index}" for gpu in worker.status.gpu_devices
            ],
            gpus_per_replica=2,
        ),
    )
    selector = VLLMResourceFitSelector(config, m, [], resource_view=ResourceView([]))
    with (
        patch(
            "gpustack.policies.candidate_selectors.base_candidate_selector.get_pretrained_config_with_workers",
            new=AsyncMock(return_value=lookup_config()),
        ),
        patch("gpustack.policies.utils.get_model_weight_size", return_value=40 * GIB),
    ):
        candidates = await selector.select_candidates([worker])
    if replicas > 1 and not reverse_order:
        assert candidates == []
        return
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.overcommit == (not reverse_order)
    assert set(candidate.gpu_indexes) == ({2, 3} if reverse_order else {0, 1})
    assert not candidate.subordinate_workers
    claim = candidate.computed_resource_claim
    assert claim.ram == TABLE_RAM + sum(claim.vram.values())
    assert (claim.ram <= 64 * GIB) == reverse_order


@pytest.mark.asyncio
@pytest.mark.parametrize("worker_count", [16, 64])
@pytest.mark.parametrize("replicas", [1, 2])
@pytest.mark.parametrize(
    "scenario", ["small_gpus", "ram_limited", "large_gpus_later", "invalid_topology"]
)
async def test_manual_multi_worker_uses_one_greedy_allocation(
    config, worker_count, replicas, scenario
):
    workers = []
    for index in range(worker_count):
        worker = linux_nvidia_1_4090_24gx1()
        worker.id = index + 1
        worker.name = f"worker-{index}"
        worker.status.memory.total = 32 * GIB
        worker.system_reserved.ram = 0
        worker.status.gpu_devices[0].memory.total = 24 * GIB
        if scenario == "ram_limited":
            large_gpu = deepcopy(worker.status.gpu_devices[0])
            large_gpu.index = 1
            large_gpu.memory.total = 80 * GIB
            worker.status.gpu_devices.append(large_gpu)
        if scenario == "large_gpus_later" and index >= worker_count - 8:
            worker.status.gpu_devices[0].memory.total = 80 * GIB
            worker.status.memory.total = 256 * GIB
        workers.append(worker)
    vram = (10 if scenario == "invalid_topology" else 500) * GIB
    parameters = ["--tp=8", "--distributed-executor-backend=mp"]
    if scenario == "invalid_topology":
        parameters.append("--data-parallel-size-local=2")
    m = model(
        replicas=replicas,
        env={"GPUSTACK_MODEL_VRAM_CLAIM": str(vram), "VLLM_PLE_CPU_OFFLOAD": "1"},
        backend_parameters=parameters,
        distributed_inference_across_workers=True,
        extended_kv_cache=ExtendedKVCacheConfig(enabled=True, ram_ratio=1),
        gpu_selector=GPUSelector(
            gpu_ids=[
                f"{worker.name}:cuda:{gpu.index}"
                for worker in workers
                for gpu in worker.status.gpu_devices
            ],
            gpus_per_replica=8,
        ),
    )
    selector = VLLMResourceFitSelector(config, m, [], resource_view=ResourceView([]))
    get_claim = selector._get_worker_resource_claim
    claim_calls = 0

    def bounded_claim(*args, **kwargs):
        nonlocal claim_calls
        claim_calls += 1
        # Manual placement must not enumerate combinations from the selected pool.
        assert claim_calls <= 2 * worker_count
        return get_claim(*args, **kwargs)

    with (
        patch(
            "gpustack.policies.candidate_selectors.base_candidate_selector.get_pretrained_config_with_workers",
            new=AsyncMock(return_value=lookup_config()),
        ),
        patch("gpustack.policies.utils.get_model_weight_size", return_value=40 * GIB),
        patch.object(selector, "_get_worker_resource_claim", side_effect=bounded_claim),
        patch.object(
            selector,
            "_manual_select_multi_worker_multi_gpu_candidates",
            wraps=selector._manual_select_multi_worker_multi_gpu_candidates,
        ) as select_multi,
    ):
        candidates = await selector.select_candidates(workers)
    assert select_multi.call_count == 1
    if replicas > 1 and scenario != "invalid_topology":
        assert candidates == []
        return
    assert len(candidates) == 1
    candidate = candidates[0]
    assert bool(candidate.overcommit) == (scenario != "invalid_topology")
    members = [candidate.worker.name] + [
        worker.worker_name for worker in candidate.subordinate_workers
    ]
    assert members == [worker.name for worker in workers[: len(members)]]
    assert (
        len(candidate.gpu_indexes)
        + sum(len(worker.gpu_indexes) for worker in candidate.subordinate_workers)
        == 8
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("offload", [False, True])
@pytest.mark.parametrize("reverse_order", [False, True])
@pytest.mark.parametrize(
    "gpu_sizes,tp,vram,expected_indexes",
    [
        ([80, 24, 24], 2, 40, {1, 2}),
        ([80, 24, 24], None, 40, {1, 2}),
        ([80, 16, 16, 16, 16], None, 50, {1, 2, 3, 4}),
        ([80, 16, 16, 16, 16], 2, 50, None),
    ],
)
async def test_auto_single_worker_searches_ram_fit_gpu_subsets(
    config, offload, reverse_order, gpu_sizes, tp, vram, expected_indexes
):
    worker = linux_nvidia_1_4090_24gx1()
    worker.status.memory.total = 64 * GIB + (TABLE_RAM if offload else 0)
    worker.system_reserved.ram = 0
    worker.status.gpu_devices = [
        deepcopy(worker.status.gpu_devices[0]) for _ in gpu_sizes
    ]
    for index, (gpu, size) in enumerate(zip(worker.status.gpu_devices, gpu_sizes)):
        gpu.index = index
        gpu.memory.total = size * GIB
    if reverse_order:
        worker.status.gpu_devices.reverse()
    m = model(
        backend_version="0.30.0",
        env={
            "GPUSTACK_MODEL_VRAM_CLAIM": str(vram * GIB),
            "VLLM_PLE_CPU_OFFLOAD": str(int(offload)),
        },
        backend_parameters=[f"--tp={tp}"] if tp else [],
        extended_kv_cache=ExtendedKVCacheConfig(enabled=True, ram_ratio=1),
    )
    selector = VLLMResourceFitSelector(config, m, [], resource_view=ResourceView([]))
    with patch(
        "gpustack.policies.candidate_selectors.base_candidate_selector.get_pretrained_config_with_workers",
        new=AsyncMock(return_value=lookup_config("Qwen4ExpForCausalLM")),
    ):
        candidates = await selector.select_candidates([worker])
    if expected_indexes is None:
        assert candidates == []
        return
    assert len(candidates) == 1
    candidate = candidates[0]
    assert not candidate.overcommit
    assert set(candidate.gpu_indexes) == expected_indexes
    claim = candidate.computed_resource_claim
    assert sum(claim.vram.values()) >= vram * GIB
    assert claim.ram == sum(claim.vram.values()) + (TABLE_RAM if offload else 0)
    assert claim.ram <= worker.status.memory.total


@pytest.mark.asyncio
@pytest.mark.parametrize("gpu_count", [24, 32])
@pytest.mark.parametrize("offload", [False, True])
@pytest.mark.parametrize("ram_ratio", [0.5, 1.5])
@pytest.mark.parametrize("scenario", ["incompatible", "tight_fit", "search_limit"])
async def test_large_auto_gpu_pool_has_bounded_search(
    config, gpu_count, offload, ram_ratio, scenario
):
    worker = linux_nvidia_1_4090_24gx1()
    worker.system_reserved.ram = 0
    worker.status.gpu_devices = [
        deepcopy(worker.status.gpu_devices[0]) for _ in range(gpu_count)
    ]
    for index, gpu in enumerate(worker.status.gpu_devices):
        gpu.index = index
        gpu.memory.total = 4 * (20 + index) * GIB
    count = gpu_count // 2
    claims = [2 * (20 + index) * GIB for index in range(gpu_count)]
    minimum = sum(claims[:count])
    maximum = sum(claims[-count:])
    vram = minimum if scenario == "tight_fit" else (minimum + maximum) // 2
    if scenario == "search_limit":
        # Every subset sum is a multiple of 2 GiB; this narrow interval has no fit.
        vram += 1
    worker.status.memory.total = (
        int(vram * ram_ratio)
        + (TABLE_RAM if offload else 0)
        - (1 if scenario == "incompatible" else 0)
    )
    m = model(
        backend_version="0.30.0",
        env={
            "GPUSTACK_MODEL_VRAM_CLAIM": str(vram),
            "VLLM_PLE_CPU_OFFLOAD": str(int(offload)),
        },
        backend_parameters=[f"--tp={count}", "--gpu-memory-utilization=0.5"],
        extended_kv_cache=ExtendedKVCacheConfig(enabled=True, ram_ratio=ram_ratio),
    )
    metadata = lookup_config("Qwen4ExpForCausalLM")
    metadata.text_config.num_attention_heads = 96
    selector = VLLMResourceFitSelector(config, m, [], resource_view=ResourceView([]))
    search = selector._find_resource_fit_gpu_subset
    states = 0

    def bounded_search(*args):
        nonlocal states
        assert scenario != "incompatible", "Jointly infeasible bounds must skip search"
        indexes, visited = search(*args)
        states += visited
        assert states <= 4096
        return indexes, visited

    with (
        patch(
            "gpustack.policies.candidate_selectors.base_candidate_selector.get_pretrained_config_with_workers",
            new=AsyncMock(return_value=metadata),
        ),
        patch.object(
            selector, "_find_resource_fit_gpu_subset", side_effect=bounded_search
        ),
    ):
        candidates = await selector.select_candidates([worker])
    if scenario != "tight_fit":
        assert candidates == []
        if scenario == "search_limit":
            assert states == 4096
            assert any("search limit" in message for message in selector.get_messages())
        return
    assert len(candidates) == 1
    assert set(candidates[0].gpu_indexes) == set(range(count))
    assert candidates[0].computed_resource_claim.ram <= worker.status.memory.total
    assert states <= 4 * gpu_count


@pytest.mark.parametrize("sizes", [[8, 6, 4, 2], [8, 8, 4, 4, 2], [2, 5, 1, 7, 3]])
@pytest.mark.parametrize("count", [2, 3, 4])
@pytest.mark.parametrize("bounds", [(7, 7), (9, 12), (13, 16), (30, 40)])
def test_gpu_subset_pruning_preserves_feasible_combinations(sizes, count, bounds):
    claims = dict(enumerate(sizes))
    groups = {}
    for index, size in claims.items():
        groups.setdefault(size, []).append(index)
    expected = any(
        bounds[0] <= sum(subset) <= bounds[1] for subset in combinations(sizes, count)
    )
    indexes, _ = VLLMResourceFitSelector._find_resource_fit_gpu_subset(
        list(groups.values()), claims, count, bounds, 4096
    )
    assert (indexes is not None) == expected
    if indexes is not None:
        assert len(set(indexes)) == count
        assert bounds[0] <= sum(claims[index] for index in indexes) <= bounds[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["0.30.0", "0.31.0", "0.32.0", "0.33.0"])
@pytest.mark.parametrize("manual", [False, True])
@pytest.mark.parametrize(
    "settings,declared_ram,expected_ram",
    [
        ({}, None, TABLE_RAM),
        ({"dp_shared_memory": False}, None, TABLE_RAM),
        ({"embedding_across_dp": True}, None, TABLE_RAM),
        ({}, 10 * GIB, 10 * GIB),
        ({}, 2 * TABLE_RAM, 2 * TABLE_RAM),
    ],
)
@pytest.mark.parametrize(
    "architecture", ["DeepseekV41ForCausalLM", "Qwen4ExpForCausalLM"]
)
@pytest.mark.parametrize(
    "parallelism",
    [
        ["--dp=2"],
        ["--dp=2", "--dpl=2", "--data-parallel-hybrid-lb"],
        ["--dp=2", "--enable-elastic-ep"],
        ["--dp=2", "--enable_elastic_ep"],
    ],
)
async def test_dp_table_ram_assumes_sharing_unless_ram_is_overridden(
    config,
    version,
    manual,
    settings,
    declared_ram,
    expected_ram,
    architecture,
    parallelism,
):
    settings = {"cpu_offload": True, **settings}
    worker = linux_nvidia_5_a100_80gx2(reserved=False)
    worker.status.memory.total = 12 * GIB
    env = (
        {} if declared_ram is None else {"GPUSTACK_MODEL_RAM_CLAIM": str(declared_ram)}
    )
    m = model(
        backend_version=version,
        backend_parameters=[*parallelism, f"--engram-config={json.dumps(settings)}"],
        env=env,
        gpu_selector=(
            GPUSelector(
                gpu_ids=[
                    f"{worker.name}:cuda:{gpu.index}"
                    for gpu in worker.status.gpu_devices
                ],
                gpus_per_replica=2,
            )
            if manual
            else None
        ),
    )
    selector = VLLMResourceFitSelector(config, m, [], resource_view=ResourceView([]))
    with (
        patch(
            "gpustack.policies.candidate_selectors.base_candidate_selector.get_pretrained_config_with_workers",
            new=AsyncMock(return_value=lookup_config(architecture)),
        ),
        patch("gpustack.policies.utils.get_model_weight_size", return_value=20 * GIB),
    ):
        candidates = await selector.select_candidates([worker])
    fits = expected_ram <= 12 * GIB
    if not manual and not fits:
        assert candidates == []
        return
    assert len(candidates) == 1
    assert candidates[0].computed_resource_claim.ram == expected_ram
    assert bool(candidates[0].overcommit) == (not fits)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "parameters,copies,distributed",
    [
        (["--tp=1", "--dp=4"], 1, True),
        (["--tp=2", "--dp=2"], 1, True),
        (["--pp=2", "--dp=2"], 1, True),
        (["--pcp=2", "--dp=2"], 2, True),
        (["--dp=4", "--dpl=2", "--data-parallel-hybrid-lb"], 1, True),
    ],
)
async def test_lookup_ram_keeps_pcp_copies_and_assumes_dp_sharing(
    config, parameters, copies, distributed
):
    workers = [linux_nvidia_1_4090_24gx1(), linux_nvidia_3_4090_24gx1()]
    for worker in workers:
        worker.status.memory.total = copies * TABLE_RAM
        worker.system_reserved.ram = 0
        worker.status.gpu_devices.append(deepcopy(worker.status.gpu_devices[0]))
        worker.status.gpu_devices[1].index = 1
    m = model(
        backend_version="0.32.0",
        backend_parameters=parameters,
        distributed_inference_across_workers=True,
    )
    selector = VLLMResourceFitSelector(config, m, [], resource_view=ResourceView([]))
    with (
        patch(
            "gpustack.policies.candidate_selectors.base_candidate_selector.get_pretrained_config_with_workers",
            new=AsyncMock(return_value=lookup_config("Qwen4ExpForCausalLM")),
        ),
        patch("gpustack.policies.utils.get_model_weight_size", return_value=20 * GIB),
    ):
        candidates = await selector.select_candidates(workers)
    assert candidates
    for candidate in candidates:
        assert not candidate.overcommit
        assert bool(candidate.subordinate_workers) is distributed
        claims = [candidate.computed_resource_claim] + [
            worker.computed_resource_claim
            for worker in candidate.subordinate_workers or []
        ]
        assert all(claim.ram == copies * TABLE_RAM for claim in claims)


@pytest.mark.parametrize(
    "version,image,default_offload",
    [
        ("0.29.0", None, False),
        ("0.33.0rc1", None, False),
        ("0.34.0.dev1", None, False),
        ("1.0.0rc1", None, False),
        ("invalid", None, True),
        ("0.33.0-custom", None, True),
        ("0.33.0-cu128", None, True),
        ("vllm-v0.33.0", None, True),
        ("latest", None, True),
        ("nightly", None, True),
        ("0.32.0", "custom:latest", True),
        ("0.33.0", "custom:latest", True),
        ("1.0.0", "custom:latest", True),
        ("invalid", "custom:latest", True),
        (None, "custom:latest", True),
        ("", "custom:latest", True),
    ],
)
@pytest.mark.parametrize("dotted", [False, True])
@pytest.mark.parametrize("explicit", [None, False, True])
@pytest.mark.parametrize(
    "architecture", ["DeepseekV41ForCausalLM", "Qwen4ExpForCausalLM"]
)
def test_custom_backend_offload_defaults_and_overrides(
    version, image, default_offload, dotted, explicit, architecture
):
    m = model(backend_version=version, image_name=image)
    if explicit is not None:
        m.backend_parameters = (
            [f"--engram_config.cpu_offload={json.dumps(explicit)}"]
            if dotted
            else [f"--engram-config={json.dumps({'cpu_offload': explicit})}"]
        )
    enabled = default_offload if explicit is None else explicit
    assert estimate_offloaded_lookup_bytes(m, lookup_config(architecture)) == (
        TABLE_BYTES if enabled else 0
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["0.30.0", "0.31.0", "0.32.0"])
@pytest.mark.parametrize("replicas", [1, 2])
@pytest.mark.parametrize("ram", [8, 12])
@pytest.mark.parametrize("executor", ["mp", "ray"])
@pytest.mark.parametrize(
    "architecture,parameters,layout",
    [
        ("DeepseekV41ForCausalLM", ["--tp=1", "--dp=6"], [4, 2]),
        ("DeepseekV41ForCausalLM", ["--tp=1", "--dp=8"], [4, 4]),
        ("Qwen4ExpForCausalLM", [], [4, 2]),
        ("Qwen4ExpForCausalLM", [], [2, 2]),
    ],
)
async def test_manual_lookup_ram_assumes_dp_sharing_across_layouts(
    config, version, replicas, ram, executor, architecture, parameters, layout
):
    copies = [1] * len(layout)
    workers = [linux_nvidia_1_4090_24gx1(), linux_nvidia_3_4090_24gx1()]
    for worker, count in zip(workers, layout):
        gpu = worker.status.gpu_devices[0]
        worker.status.gpu_devices = [deepcopy(gpu) for _ in range(count)]
        for index, device in enumerate(worker.status.gpu_devices):
            device.index = index
        worker.status.memory.total = ram * GIB
        worker.system_reserved.ram = 0
    m = model(
        backend_version=version,
        backend_parameters=[f"--distributed-executor-backend={executor}", *parameters],
        replicas=replicas,
        distributed_inference_across_workers=True,
        gpu_selector=GPUSelector(
            gpu_ids=[
                f"{worker.name}:cuda:{gpu.index}"
                for worker in workers
                for gpu in worker.status.gpu_devices
            ],
            gpus_per_replica=sum(layout),
        ),
    )
    selector = VLLMResourceFitSelector(config, m, [], resource_view=ResourceView([]))
    with (
        patch(
            "gpustack.policies.candidate_selectors.base_candidate_selector.get_pretrained_config_with_workers",
            new=AsyncMock(return_value=lookup_config(architecture)),
        ),
        patch("gpustack.policies.utils.get_model_weight_size", return_value=20 * GIB),
    ):
        candidates = await selector.select_candidates(workers)
    overcommit = max(copies) * TABLE_RAM > ram * GIB
    if replicas > 1 and overcommit:
        assert candidates == []
        return
    assert len(candidates) == 1
    candidate = candidates[0]
    assert bool(candidate.overcommit) == overcommit
    claims = [candidate.computed_resource_claim] + [
        worker.computed_resource_claim for worker in candidate.subordinate_workers
    ]
    assert [claim.ram for claim in claims] == [count * TABLE_RAM for count in copies]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "architecture", ["DeepseekV41ForCausalLM", "Qwen4ExpForCausalLM"]
)
@pytest.mark.parametrize("embedding_across_dp", [False, True])
@pytest.mark.parametrize("dp", [1, 2])
@pytest.mark.parametrize("manual", [False, True])
@pytest.mark.parametrize("ram", [12, 64])
async def test_pcp_table_copies_survive_dp_sharing(
    config, architecture, embedding_across_dp, dp, manual, ram
):
    worker = linux_nvidia_5_a100_80gx2(reserved=False)
    gpu_count = 2 * dp
    worker.status.gpu_devices = [
        deepcopy(worker.status.gpu_devices[0]) for _ in range(gpu_count)
    ]
    for index, gpu in enumerate(worker.status.gpu_devices):
        gpu.index = index
    worker.status.memory.total = ram * GIB
    m = model(
        backend_version="0.32.0",
        backend_parameters=[
            "--tp=1",
            "--pcp=2",
            f"--dp={dp}",
            f'--engram-config={json.dumps({"embedding_across_dp": embedding_across_dp})}',
        ],
        gpu_selector=(
            GPUSelector(
                gpu_ids=[f"{worker.name}:cuda:{index}" for index in range(gpu_count)],
                gpus_per_replica=gpu_count,
            )
            if manual
            else None
        ),
    )
    selector = VLLMResourceFitSelector(config, m, [], resource_view=ResourceView([]))
    with (
        patch(
            "gpustack.policies.candidate_selectors.base_candidate_selector.get_pretrained_config_with_workers",
            new=AsyncMock(return_value=lookup_config(architecture)),
        ),
        patch("gpustack.policies.utils.get_model_weight_size", return_value=20 * GIB),
    ):
        candidates = await selector.select_candidates([worker])
    expected_ram = 2 * TABLE_RAM
    fits = expected_ram <= ram * GIB
    if not manual and not fits:
        assert candidates == []
        return
    assert len(candidates) == 1
    assert candidates[0].computed_resource_claim.ram == expected_ram
    assert bool(candidates[0].overcommit) == (not fits)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "architecture", ["DeepseekV41ForCausalLM", "Qwen4ExpForCausalLM"]
)
@pytest.mark.parametrize("gpu_count,pcp,dp", [(2, 2, 2), (2, 4, 1), (4, 2, 4)])
async def test_auto_multinode_table_ram_skips_insufficient_worker(
    config, architecture, gpu_count, pcp, dp
):
    workers = [linux_nvidia_1_4090_24gx1(), linux_nvidia_3_4090_24gx1()]
    workers.append(deepcopy(workers[1]))
    workers[2].id = 100
    workers[2].name = "lookup-worker-3"
    workers[2].ip = "192.168.100.100"
    copies = min(pcp, gpu_count)
    expected_ram = copies * TABLE_RAM + 4 * GIB
    for position, worker in enumerate(workers):
        worker.status.gpu_devices = [
            deepcopy(worker.status.gpu_devices[0]) for _ in range(gpu_count)
        ]
        for index, gpu in enumerate(worker.status.gpu_devices):
            gpu.index = index
        worker.status.memory.total = 12 * GIB if position == 0 else expected_ram
        worker.system_reserved.ram = 0
    m = model(
        backend_version="0.32.0",
        backend_parameters=[
            "--distributed-executor-backend=mp",
            "--tp=1",
            f"--pcp={pcp}",
            f"--dp={dp}",
        ],
        distributed_inference_across_workers=True,
        extended_kv_cache=ExtendedKVCacheConfig(enabled=True, ram_size=4),
    )
    selector = VLLMResourceFitSelector(config, m, [], resource_view=ResourceView([]))
    with (
        patch(
            "gpustack.policies.candidate_selectors.base_candidate_selector.get_pretrained_config_with_workers",
            new=AsyncMock(return_value=lookup_config(architecture)),
        ),
        patch("gpustack.policies.utils.get_model_weight_size", return_value=20 * GIB),
    ):
        candidates = await selector.select_candidates(workers)
    assert len(candidates) == 1
    candidate = candidates[0]
    assert not candidate.overcommit
    assert candidate.worker.id != workers[0].id
    assert candidate.computed_resource_claim.ram == expected_ram
    assert len(candidate.subordinate_workers) == 1
    subordinate = candidate.subordinate_workers[0]
    assert subordinate.worker_id != workers[0].id
    assert subordinate.computed_resource_claim.ram == expected_ram


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "version", [None, "", "0.30.0", "0.31.0", "0.32.0", "0.33.0", "0.34.0", "1.0.0"]
)
@pytest.mark.parametrize("ram", [4, 12])
async def test_removed_offload_env_does_not_suppress_ram_accounting(
    config, version, ram
):
    worker = linux_nvidia_1_4090_24gx1()
    worker.status.memory.total = ram * GIB
    worker.system_reserved.ram = 0
    m = model(backend_version=version, env={"VLLM_PLE_CPU_OFFLOAD": "0"})
    selector = VLLMResourceFitSelector(config, m, [], resource_view=ResourceView([]))
    with (
        patch(
            "gpustack.policies.candidate_selectors.base_candidate_selector.get_pretrained_config_with_workers",
            new=AsyncMock(return_value=lookup_config("Qwen4ExpForCausalLM")),
        ),
        patch("gpustack.policies.utils.get_model_weight_size", return_value=10 * GIB),
    ):
        candidates = await selector.select_candidates([worker])
    offloaded = 0 if version == "0.30.0" else TABLE_BYTES
    assert selector._vram_claim == int((10 * GIB - offloaded) * 1.2) + 2 * GIB
    assert bool(candidates) == (ram * GIB >= offloaded)
    if candidates:
        assert (candidates[0].computed_resource_claim.ram or 0) == offloaded


@pytest.mark.parametrize(
    "architecture,parameters,expected",
    [
        ("Qwen4ExpForCausalLM", [], TABLE_BYTES),
        ("Qwen4ExpForConditionalGeneration", [], TABLE_BYTES),
        ("Qwen4ExpForCausalLM", ["--no-ple-offload-embedding"], 0),
        ("Qwen4ExpForCausalLM", ["--ple-offload-embedding"], TABLE_BYTES),
        (
            "Qwen4ExpForConditionalGeneration",
            ["--ple-offload-embedding", "--ple-offload-backend=pinned"],
            TABLE_BYTES,
        ),
        (
            "Qwen4ExpForCausalLM",
            ["--ple-offload-embedding", "--ple-offload-backend=file"],
            TABLE_BYTES,
        ),
        ("Qwen4ExpForCausalLM", ["--ple-offload-backend=file"], TABLE_BYTES),
        (
            "Qwen4ExpForCausalLM",
            ["--ple-offload-embedding", "--no-ple-offload-embedding"],
            0,
        ),
        (
            "Qwen4ExpForCausalLM",
            ["--no-ple-offload-embedding", "--ple-offload-embedding"],
            TABLE_BYTES,
        ),
        (
            "Qwen4ExpForCausalLM",
            ["--ple-offload-embedding", "--ple-offload-backend=unknown"],
            0,
        ),
        ("DeepseekV41ForCausalLM", [], 0),
        (
            "DeepseekV41ForCausalLM",
            ['--engram-config={"placement":"cpu"}'],
            TABLE_BYTES,
        ),
        (
            "DeepseekV41ForCausalLM",
            ['--engram-config', '{"placement":"cpu","replication":"per_node"}'],
            TABLE_BYTES,
        ),
        ("DeepseekV41ForCausalLM", ['--engram-config={"placement":"gpu"}'], 0),
        ("DeepseekV41ForCausalLM", ['--engram-config={"cpu_offload":true}'], 0),
        ("DeepseekV41ForCausalLM", ['--engram-config=[]'], 0),
        ("DeepseekV41ForCausalLM", ['--engram-config=invalid'], 0),
        (
            "DeepseekV41ForCausalLM",
            ['--engram-config={"placement":"cpu"}', '--engram-config={}'],
            0,
        ),
    ],
)
def test_sglang_lookup_offload_settings(architecture, parameters, expected):
    m = model(backend_parameters=parameters)
    m.backend = BackendEnum.SGLANG
    assert estimate_offloaded_lookup_bytes(m, lookup_config(architecture)) == expected


@pytest.mark.parametrize("as_dict", [False, True])
@pytest.mark.parametrize(
    "text_dtype,root_dtype,dtype_args,offloaded",
    [
        ("bfloat16", None, [], True),
        ("torch.bfloat16", None, [], True),
        ("float16", None, [], False),
        ("half", None, [], False),
        ("float32", None, [], False),
        (None, None, [], False),
        ("unknown", None, [], False),
        (None, "bfloat16", [], True),
        ("float16", "bfloat16", [], False),
        ("bfloat16", "float16", [], True),
        ("bfloat16", None, ["--dtype=auto"], True),
        ("float32", None, ["--dtype=auto"], False),
        ("bfloat16", None, ["--dtype=half"], False),
        ("bfloat16", None, ["--dtype=float"], False),
        ("float16", None, ["--dtype", "bfloat16"], True),
        (None, None, ["--dtype=BFLOAT16"], True),
        ("bfloat16", None, ["--dtype=unknown"], False),
        ("bfloat16", None, ["--dtype=bfloat16", "--dtype=float16"], False),
        ("float16", None, ["--dtype=float16", "--dtype=bfloat16"], True),
    ],
)
def test_sglang_ple_default_uses_runtime_dtype(
    as_dict, text_dtype, root_dtype, dtype_args, offloaded
):
    m = model(backend_parameters=dtype_args)
    m.backend = BackendEnum.SGLANG
    metadata = lookup_config("Qwen4ExpForCausalLM")
    metadata.dtype = root_dtype
    metadata.text_config.dtype = text_dtype
    # FP8 table storage does not determine the runtime dtype or offload default.
    metadata.text_config.ple_embedding_dtype = "float8_e4m3fn"
    if as_dict:
        metadata = {**vars(metadata), "text_config": vars(metadata.text_config)}
    assert estimate_offloaded_lookup_bytes(m, metadata) == (
        TABLE_BYTES if offloaded else 0
    )


@pytest.mark.parametrize("nested", [False, True])
def test_sglang_ple_default_accepts_legacy_torch_dtype(nested):
    m = model()
    m.backend = BackendEnum.SGLANG
    metadata = lookup_config("Qwen4ExpForCausalLM")
    del metadata.text_config.dtype
    metadata.text_config.torch_dtype = "bfloat16"
    if not nested:
        metadata = {**vars(metadata), **vars(metadata.text_config)}
        del metadata["text_config"]
    assert estimate_offloaded_lookup_bytes(m, metadata) == TABLE_BYTES


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "dtype,flag,offloaded",
    [
        ("bfloat16", None, True),
        ("float16", None, False),
        ("bfloat16", "--no-ple-offload-embedding", False),
        ("float16", "--ple-offload-embedding", True),
    ],
)
async def test_sglang_ple_dtype_and_flags_control_scheduling(
    config, dtype, flag, offloaded
):
    worker = linux_nvidia_1_4090_24gx1()
    worker.status.memory.total = 64 * GIB
    worker.system_reserved.ram = 0
    parameters = [f"--dtype={dtype}", "--mem-fraction-static=0.9"]
    if flag:
        parameters.append(flag)
    m = model(backend_parameters=parameters)
    m.backend = BackendEnum.SGLANG
    selector = SGLangResourceFitSelector(config, m, [], resource_view=ResourceView([]))
    with (
        patch(
            "gpustack.policies.candidate_selectors.base_candidate_selector.get_pretrained_config_with_workers",
            new=AsyncMock(return_value=lookup_config("Qwen4ExpForCausalLM")),
        ),
        patch("gpustack.policies.utils.get_model_weight_size", return_value=20 * GIB),
    ):
        candidates = await selector.select_candidates([worker])
    table_bytes = TABLE_BYTES if offloaded else 0
    assert selector._vram_claim == int((20 * GIB - table_bytes) * 1.2 + 2 * GIB)
    assert selector._ram_claim == table_bytes
    assert bool(candidates) is offloaded
    if candidates:
        assert candidates[0].computed_resource_claim.ram == TABLE_BYTES


@pytest.mark.parametrize("override", ["{}", '{"ple_embed_dim":128}'])
def test_sglang_model_overrides_disable_lookup_estimation(override):
    m = model(
        backend_parameters=[
            "--ple-offload-embedding",
            "--json-model-override-args",
            override,
        ]
    )
    m.backend = BackendEnum.SGLANG
    assert estimate_offloaded_lookup_bytes(m, lookup_config("Qwen4ExpForCausalLM")) == 0


def test_sglang_engram_precision_must_match_runtime_storage():
    m = model(backend_parameters=['--engram-config={"placement":"cpu"}'])
    m.backend = BackendEnum.SGLANG
    metadata = lookup_config()
    metadata.quantization_config["engram_embed_quant"] = "mxfp4"
    assert estimate_offloaded_lookup_bytes(m, metadata) == 0


@pytest.mark.parametrize(
    "quantization,embedding_dtype,expected",
    [
        ({}, None, 2 * TABLE_BYTES),
        (
            {
                "quant_method": "fp8",
                "ignored_layers": ["model.layers.1.ple.ple_embedding.ngram_embedding"],
            },
            None,
            TABLE_BYTES,
        ),
        ({"quant_method": "modelopt"}, "float8_e4m3fn", TABLE_BYTES),
        ({"quant_method": "unknown"}, None, 0),
    ],
)
def test_sglang_ple_storage_precision(quantization, embedding_dtype, expected):
    m = model(backend_parameters=["--ple-offload-embedding"])
    m.backend = BackendEnum.SGLANG
    metadata = lookup_config("Qwen4ExpForCausalLM")
    metadata.quantization_config = quantization
    metadata.text_config.ple_embedding_dtype = embedding_dtype
    assert estimate_offloaded_lookup_bytes(m, metadata) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("storage", ["default", "pinned", "file", "engram"])
@pytest.mark.parametrize("layout", ["single_gpu", "single_worker", "multi_worker"])
@pytest.mark.parametrize("manual", [False, True])
@pytest.mark.parametrize("cache", [None, "size", "ratio"])
@pytest.mark.parametrize("shortage", [False, True])
async def test_sglang_lookup_ram_accounting_and_admission(
    config, storage, layout, manual, cache, shortage
):
    workers = [linux_nvidia_1_4090_24gx1()]
    if layout == "single_worker":
        workers = [linux_nvidia_5_a100_80gx2()]
    elif layout == "multi_worker":
        workers.append(linux_nvidia_3_4090_24gx1())
    for worker in workers:
        worker.system_reserved.ram = 0
        worker.system_reserved.vram = 0
        for gpu in worker.status.gpu_devices:
            gpu.memory.total = 24 * GIB
    gpu_count = 1 if layout == "single_gpu" else 2
    reserved_per_worker = int(24 * GIB * 0.9) * (2 if layout == "single_worker" else 1)
    expected_ram = TABLE_BYTES + (
        4 * GIB if cache == "size" else reserved_per_worker if cache == "ratio" else 0
    )
    for worker in workers:
        worker.status.memory.total = expected_ram - int(shortage)
    parameters = (
        ['--engram-config={"placement":"cpu"}']
        if storage == "engram"
        else (
            []
            if storage == "default"
            else ["--ple-offload-embedding", f"--ple-offload-backend={storage}"]
        )
    )
    m = model(
        backend_parameters=parameters
        + ["--mem-fraction-static=0.9", f"--tp={gpu_count}"],
        distributed_inference_across_workers=True,
        extended_kv_cache=(
            ExtendedKVCacheConfig(enabled=True, ram_size=4)
            if cache == "size"
            else (
                ExtendedKVCacheConfig(enabled=True, ram_ratio=1)
                if cache == "ratio"
                else None
            )
        ),
    )
    m.backend = BackendEnum.SGLANG
    if manual:
        m.gpu_selector = GPUSelector(
            gpu_ids=[
                f"{worker.name}:cuda:{gpu.index}"
                for worker in workers
                for gpu in worker.status.gpu_devices
            ],
            gpus_per_replica=gpu_count,
        )
    selector = SGLangResourceFitSelector(config, m, [], resource_view=ResourceView([]))
    architecture = (
        "DeepseekV41ForCausalLM" if storage == "engram" else "Qwen4ExpForCausalLM"
    )
    with (
        patch(
            "gpustack.policies.candidate_selectors.base_candidate_selector.get_pretrained_config_with_workers",
            new=AsyncMock(return_value=lookup_config(architecture)),
        ),
        patch(
            "gpustack.policies.utils.get_model_weight_size",
            return_value=20 * gpu_count * GIB,
        ),
    ):
        candidates = await selector.select_candidates(workers)
    assert selector._vram_claim == int(
        (20 * gpu_count * GIB - TABLE_BYTES) * 1.2 + 2 * GIB
    )
    if shortage and not manual:
        assert candidates == []
        assert "The model requires approximately" in selector.get_messages()[0]
        return
    assert len(candidates) == 1
    candidate = candidates[0]
    assert bool(candidate.overcommit) is (manual and shortage)
    claims = [candidate.computed_resource_claim] + [
        worker.computed_resource_claim for worker in candidate.subordinate_workers or []
    ]
    assert len(claims) == len(workers)
    assert all(claim.ram == expected_ram for claim in claims)


@pytest.mark.asyncio
@pytest.mark.parametrize("ram", [None, 0, 3 * GIB])
@pytest.mark.parametrize("vram", [None, 10 * GIB])
async def test_sglang_lookup_resource_overrides_are_independent(config, ram, vram):
    worker = linux_nvidia_1_4090_24gx1()
    worker.status.memory.total = 64 * GIB
    worker.system_reserved.ram = 0
    env = {}
    if ram is not None:
        env["GPUSTACK_MODEL_RAM_CLAIM"] = str(ram)
    if vram is not None:
        env["GPUSTACK_MODEL_VRAM_CLAIM"] = str(vram)
    m = model(
        backend_parameters=["--ple-offload-embedding", "--mem-fraction-static=0.9"],
        env=env,
        extended_kv_cache=ExtendedKVCacheConfig(enabled=True, ram_size=4),
    )
    m.backend = BackendEnum.SGLANG
    selector = SGLangResourceFitSelector(config, m, [], resource_view=ResourceView([]))
    with (
        patch(
            "gpustack.policies.candidate_selectors.base_candidate_selector.get_pretrained_config_with_workers",
            new=AsyncMock(return_value=lookup_config("Qwen4ExpForCausalLM")),
        ),
        patch("gpustack.policies.utils.get_model_weight_size", return_value=20 * GIB),
    ):
        candidates = await selector.select_candidates([worker])
    assert len(candidates) == 1
    assert candidates[0].computed_resource_claim.ram == (
        TABLE_BYTES + 4 * GIB if ram is None else ram
    )
    assert selector._vram_claim == (
        int((20 * GIB - TABLE_BYTES) * 1.2 + 2 * GIB) if vram is None else vram
    )
