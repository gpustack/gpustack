import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from gpustack.policies import utils as policy_utils
from gpustack.policies.candidate_selectors import (
    AscendMindIEResourceFitSelector,
    SGLangResourceFitSelector,
    VLLMResourceFitSelector,
)
from gpustack.policies.candidate_selectors.base_candidate_selector import (
    ModelParameters,
)
from gpustack.policies.candidate_selectors.custom_backend_resource_fit_selector import (
    CustomBackendResourceFitSelector,
)
from gpustack.policies.resource_view import ResourceView
from gpustack.scheduler import calculator, scheduler
from gpustack.schemas.models import BackendEnum, Model, SourceEnum
from tests.fixtures.workers.fixtures import (
    linux_ascend_1_910b_64gx8,
    linux_nvidia_22_H100_80gx8,
)


@pytest.fixture(autouse=True)
def offline_metadata_only(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", True)

    def forbidden(*args, **kwargs):
        pytest.fail("Configuration evaluation must not use network or repository code")

    monkeypatch.setattr("requests.Session.request", forbidden)
    monkeypatch.setattr("httpx.Client.send", forbidden)
    monkeypatch.setattr(
        "transformers.dynamic_module_utils.get_class_from_dynamic_module", forbidden
    )
    monkeypatch.setattr(
        "transformers.models.auto.configuration_auto.get_class_from_dynamic_module",
        forbidden,
    )


@pytest.fixture
def partial_model(tmp_path):
    raw = {
        "model_type": "custom_metadata",
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 2048,
        "num_hidden_layers": 16,
    }
    (tmp_path / "config.json").write_text(json.dumps(raw))
    return Model(
        name="partial-metadata",
        source=SourceEnum.LOCAL_PATH,
        local_path=str(tmp_path),
        backend=BackendEnum.VLLM,
    )


@pytest.mark.parametrize(
    "selector_class,backend",
    [
        (VLLMResourceFitSelector, BackendEnum.VLLM),
        (SGLangResourceFitSelector, BackendEnum.SGLANG),
    ],
)
@pytest.mark.parametrize("tp,weight_gib", [(None, 4), (2, 4), (None, 120)])
@pytest.mark.asyncio
async def test_missing_heads_allow_parallel_evaluation_with_warning(
    config, partial_model, monkeypatch, selector_class, backend, tp, weight_gib, caplog
):
    model = partial_model
    model.backend = backend
    await scheduler.evaluate_pretrained_config(model)
    assert model.categories == ["llm"]
    if tp:
        model.backend_parameters = [f"--tensor-parallel-size={tp}"]
    monkeypatch.setattr(
        policy_utils,
        "get_local_model_weight_size",
        AsyncMock(return_value=weight_gib * 1024**3),
    )
    selector = selector_class(config, model, [], resource_view=ResourceView([]))
    worker = linux_nvidia_22_H100_80gx8()
    candidates = await selector.select_candidates([worker])
    assert candidates
    if tp or weight_gib == 120:
        assert all(len(candidate.gpu_indexes) >= 2 for candidate in candidates)
    else:
        assert all(len(candidate.gpu_indexes) == 1 for candidate in candidates)
    assert "Skipping attention-head divisibility check" in caplog.text
    assert selector._model_params.num_key_value_heads is None
    assert selector._model_params.derived_max_seq_len is None


@pytest.mark.parametrize(
    "selector_class,backend",
    [
        (VLLMResourceFitSelector, BackendEnum.VLLM),
        (SGLangResourceFitSelector, BackendEnum.SGLANG),
    ],
)
@pytest.mark.asyncio
async def test_tp_does_not_require_unused_kv_or_context_fields(
    config, partial_model, tmp_path, monkeypatch, selector_class, backend
):
    raw = json.loads((tmp_path / "config.json").read_text())
    raw["num_attention_heads"] = 16
    (tmp_path / "config.json").write_text(json.dumps(raw))
    model = partial_model
    model.backend = backend
    model.backend_parameters = ["--tensor-parallel-size=2"]
    await scheduler.evaluate_pretrained_config(model)
    monkeypatch.setattr(
        policy_utils, "get_local_model_weight_size", AsyncMock(return_value=4 * 1024**3)
    )
    selector = selector_class(config, model, [], resource_view=ResourceView([]))
    assert await selector.select_candidates([linux_nvidia_22_H100_80gx8()])
    assert selector._model_params.num_key_value_heads is None
    assert selector._model_params.derived_max_seq_len is None


@pytest.mark.asyncio
async def test_mindie_requires_kv_estimation_dimensions(config, partial_model):
    model = partial_model
    model.backend = BackendEnum.ASCEND_MINDIE
    model.categories = ["llm"]
    selector = AscendMindIEResourceFitSelector(
        config, model, [], resource_view=ResourceView([])
    )
    await selector._init_model_parameters([])
    assert selector._model_params.derived_max_seq_len is None
    assert selector._serving_params.max_seq_len == 8192
    with pytest.raises(ValueError, match="num_attention_heads, head_dim"):
        await selector._estimate_usage([])
    with pytest.raises(ValueError, match="explicit resource claims"):
        await selector.select_candidates([linux_ascend_1_910b_64gx8()])
    model.env = {"GPUSTACK_MODEL_VRAM_CLAIM": "123456"}
    assert (await selector._estimate_usage([])).vram == 123456
    assert await selector.select_candidates([linux_ascend_1_910b_64gx8()])
    selector._serving_params.world_size = 2
    assert await selector._validate_parallelized()


@pytest.mark.parametrize("location", ["server", "worker"])
@pytest.mark.parametrize(
    "layer_overrides", [{"num_key_value_heads": 8}, {"intermediate_size": 8192}]
)
@pytest.mark.parametrize(
    "selector_class,backend",
    [
        (VLLMResourceFitSelector, BackendEnum.VLLM),
        (SGLangResourceFitSelector, BackendEnum.SGLANG),
    ],
)
@pytest.mark.asyncio
async def test_per_layer_overrides_allow_parallel_scheduling(
    config,
    partial_model,
    tmp_path,
    monkeypatch,
    location,
    layer_overrides,
    selector_class,
    backend,
):
    raw = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "num_attention_heads": 32,
        "num_hidden_layers": 2,
        "per_layer_config": {"1": layer_overrides},
    }
    (tmp_path / "config.json").write_text(json.dumps(raw))
    model = partial_model
    model.backend = backend
    model.categories = ["llm"]
    model.backend_parameters = ["--tensor-parallel-size=2"]
    if location == "worker":
        model.local_path = str(tmp_path / "worker-only")
        monkeypatch.setattr(
            calculator, "read_local_path_file_from_workers", AsyncMock(return_value=raw)
        )
    monkeypatch.setattr(
        policy_utils, "get_local_model_weight_size", AsyncMock(return_value=4 * 1024**3)
    )
    await scheduler.evaluate_pretrained_config(model)
    selector = selector_class(config, model, [], resource_view=ResourceView([]))
    candidates = await selector.select_candidates([linux_nvidia_22_H100_80gx8()])
    assert candidates
    assert all(len(candidate.gpu_indexes) == 2 for candidate in candidates)
    assert selector._model_params.num_key_value_heads == 32
    assert selector._model_params.head_dim == 128


@pytest.mark.asyncio
async def test_mindie_unknown_kv_layout_uses_full_multi_head_estimate(
    config, partial_model, tmp_path, monkeypatch, caplog
):
    raw = json.loads((tmp_path / "config.json").read_text())
    raw["num_attention_heads"] = 16
    (tmp_path / "config.json").write_text(json.dumps(raw))
    model = partial_model
    model.backend = BackendEnum.ASCEND_MINDIE
    model.categories = ["llm"]
    monkeypatch.setattr(
        "gpustack.policies.candidate_selectors.ascend_mindie_resource_fit_selector.get_local_model_weight_size",
        AsyncMock(return_value=4 * 1024**3),
    )
    selector = AscendMindIEResourceFitSelector(
        config, model, [], resource_view=ResourceView([])
    )
    await selector._init_model_parameters([])
    conservative = await selector._estimate_usage([])
    assert selector._model_params.num_key_value_heads is None
    assert "Unknown KV head layout" in caplog.text
    selector._model_params.num_key_value_heads = 16
    assert await selector._estimate_usage([]) == conservative
    selector._model_params.num_key_value_heads = 4
    assert (await selector._estimate_usage([])).vram < conservative.vram


@pytest.mark.asyncio
async def test_internvl_mindie_estimate_includes_multimodal_memory(
    config, partial_model, tmp_path, monkeypatch
):
    fixture = Path(__file__).parents[1] / "fixtures/model_configs/internvl.json"
    (tmp_path / "config.json").write_bytes(fixture.read_bytes())
    model = partial_model
    model.backend = BackendEnum.ASCEND_MINDIE
    model.backend_version = "2.2.rc1"
    model.categories = ["llm"]
    monkeypatch.setattr(
        "gpustack.policies.candidate_selectors.ascend_mindie_resource_fit_selector.get_local_model_weight_size",
        AsyncMock(return_value=8 * 1024**3),
    )
    selector = AscendMindIEResourceFitSelector(
        config, model, [], resource_view=ResourceView([])
    )
    await selector._init_model_parameters([])
    assert selector._model_params.architectures == ["InternVLChatModel"]
    assert selector._model_params.num_attention_heads == 28
    assert selector._model_params.hidden_size == 3584
    estimate = await selector._estimate_usage([])
    assert estimate.vram / 1024**3 == pytest.approx(32.984375)
    worker = linux_ascend_1_910b_64gx8()
    worker.status.gpu_devices = worker.status.gpu_devices[:1]
    worker.status.gpu_devices[0].memory.total = 24 * 1024**3
    assert not await selector.select_candidates([worker])


@pytest.mark.parametrize("location", ["server", "worker"])
@pytest.mark.asyncio
async def test_ming_custom_rope_does_not_block_custom_backend(
    config, partial_model, tmp_path, monkeypatch, location
):
    fixture = Path(__file__).parents[1] / "fixtures/model_configs/ming_flash_omni.json"
    raw = json.loads(fixture.read_text())
    (tmp_path / "config.json").write_bytes(fixture.read_bytes())
    model = partial_model
    model.backend = BackendEnum.CUSTOM
    model.categories = ["llm"]
    if location == "worker":
        model.local_path = str(tmp_path / "worker-only")
        monkeypatch.setattr(
            calculator, "read_local_path_file_from_workers", AsyncMock(return_value=raw)
        )
    monkeypatch.setattr(
        policy_utils, "get_local_model_weight_size", AsyncMock(return_value=4 * 1024**3)
    )
    selector = CustomBackendResourceFitSelector(
        config, model, [], resource_view=ResourceView([])
    )
    await selector._init_model_parameters([])
    assert selector._model_params.architectures == raw["architectures"]
    assert selector._model_params.num_attention_heads == 32
    assert selector._model_params.derived_max_seq_len == 32768
    assert await selector.select_candidates([linux_nvidia_22_H100_80gx8()])


@pytest.fixture(params=["mamba", "mamba2", "falcon_mamba"])
def state_space_model(request, partial_model, tmp_path):
    architectures = {
        "mamba": "MambaForCausalLM",
        "mamba2": "Mamba2ForCausalLM",
        "falcon_mamba": "FalconMambaForCausalLM",
    }
    raw = {
        "model_type": request.param,
        "architectures": [architectures[request.param]],
        "hidden_size": 256,
        "num_hidden_layers": 8,
        "vocab_size": 32768,
        "expand": 2,
    }
    if request.param == "mamba2":
        raw.update(num_heads=16, head_dim=32, n_groups=8)
    (tmp_path / "config.json").write_text(json.dumps(raw))
    return partial_model


@pytest.mark.parametrize(
    "selector_class,backend",
    [
        (VLLMResourceFitSelector, BackendEnum.VLLM),
        (SGLangResourceFitSelector, BackendEnum.SGLANG),
    ],
)
@pytest.mark.parametrize("explicit_tp", [True, False])
@pytest.mark.asyncio
async def test_state_space_models_support_explicit_and_automatic_tp(
    config, state_space_model, monkeypatch, selector_class, backend, explicit_tp, caplog
):
    model = state_space_model
    model.backend = backend
    if explicit_tp:
        model.backend_parameters = ["--tensor-parallel-size=2"]
    await scheduler.evaluate_pretrained_config(model)
    monkeypatch.setattr(
        policy_utils,
        "get_local_model_weight_size",
        AsyncMock(return_value=(4 if explicit_tp else 120) * 1024**3),
    )
    selector = selector_class(config, model, [], resource_view=ResourceView([]))
    candidates = await selector.select_candidates([linux_nvidia_22_H100_80gx8()])
    assert candidates
    assert all(len(candidate.gpu_indexes) >= 2 for candidate in candidates)
    if explicit_tp:
        assert all(len(candidate.gpu_indexes) == 2 for candidate in candidates)
    assert selector._model_params.num_attention_heads is None
    assert selector._model_params.intermediate_size == 512
    assert "Skipping attention-head divisibility check" not in caplog.text


@pytest.mark.parametrize(
    "model_type,dimensions,tp,error",
    [
        ("mamba", {"intermediate_size": 510}, 4, "intermediate_size"),
        ("falcon_mamba", {"intermediate_size": 510}, 4, "intermediate_size"),
        ("mamba2", {"num_heads": 6}, 4, "num_heads"),
        ("mamba2", {"n_groups": 6}, 4, "n_groups"),
        ("mamba2", {"n_groups": 1}, 4, None),
        ("mamba2", {"num_heads": None, "n_groups": None}, 4, None),
        ("custom_metadata", {"num_attention_heads": 6}, 4, "attention heads"),
        ("custom_metadata", {"vocab_size": 32767}, 2, "Vocabulary size"),
    ],
)
@pytest.mark.parametrize(
    "selector_class", [VLLMResourceFitSelector, SGLangResourceFitSelector]
)
def test_tp_rejects_known_incompatible_dimensions(
    config, partial_model, selector_class, model_type, dimensions, tp, error
):
    selector = selector_class(config, partial_model, [], resource_view=ResourceView([]))
    values = dict(
        model_type=model_type,
        intermediate_size=512,
        num_heads=16,
        n_groups=8,
        vocab_size=32768,
    )
    values.update(dimensions)
    selector._model_params = ModelParameters(**values)
    selector._num_attention_heads = selector._model_params.num_attention_heads
    message = selector._check_tp_size_divisibility(tp)
    if error:
        assert error in message
        assert not selector._is_tp_size_divisible(tp)
    else:
        assert message is None
        assert selector._is_tp_size_divisible(tp)
