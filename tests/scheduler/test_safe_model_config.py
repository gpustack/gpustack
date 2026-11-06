import json
import shutil
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from transformers import AutoConfig
from transformers import dynamic_module_utils
from transformers.models.auto import configuration_auto
from transformers.utils import hub as transformers_hub

from gpustack.policies.candidate_selectors import VLLMResourceFitSelector
from gpustack.policies.candidate_selectors.base_candidate_selector import (
    ModelParameters,
)
from gpustack.policies.resource_view import ResourceView
from gpustack.scheduler import calculator, scheduler
from gpustack.schemas.models import BackendEnum, CategoryEnum, Model, SourceEnum
from gpustack.utils import hub, model_config

FIXTURES = Path(__file__).parents[1] / "fixtures" / "model_configs"


def cache_hf_repository(config, repo_id, repository):
    cache = (
        Path(config.cache_dir)
        / "huggingface"
        / ("models--" + repo_id.replace("/", "--"))
    )
    revision = "a" * 40
    (cache / "refs").mkdir(parents=True, exist_ok=True)
    (cache / "refs" / "main").write_text(revision)
    snapshot = cache / "snapshots" / revision
    shutil.copytree(repository, snapshot, dirs_exist_ok=True)
    return snapshot


@pytest.fixture(autouse=True)
def forbid_repository_code(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", True)

    def forbidden(*args, **kwargs):
        pytest.fail("Metadata inspection must never resolve repository Python")

    monkeypatch.setattr(
        dynamic_module_utils, "get_class_from_dynamic_module", forbidden
    )
    monkeypatch.setattr(configuration_auto, "get_class_from_dynamic_module", forbidden)
    monkeypatch.setattr("requests.Session.request", forbidden)
    monkeypatch.setattr("httpx.Client.send", forbidden)


@pytest.mark.parametrize("trusted", [False, True], ids=["no-flag", "trust-flag"])
@pytest.mark.parametrize("model_type", ["custom_metadata", "llama"])
@pytest.mark.parametrize(
    "source",
    ["local", "hf-cache", "hf-download", "ms-cache", "ms-download", "worker"],
)
@pytest.mark.asyncio
async def test_repository_code_is_not_executed(
    tmp_path, config, monkeypatch, trusted, model_type, source
):
    marker = tmp_path / "executed"
    repository = tmp_path / "repository"
    repository.mkdir()
    config_dict = {
        "model_type": model_type,
        "architectures": ["LlamaForCausalLM"],
        "auto_map": {"AutoConfig": "configuration_probe.ProbeConfig"},
        "hidden_size": 2048,
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "max_position_embeddings": 8192,
    }
    (repository / "config.json").write_text(json.dumps(config_dict))
    (repository / "configuration_probe.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).touch()\n"
        "from transformers import PretrainedConfig\n"
        "class ProbeConfig(PretrainedConfig):\n"
        "    pass\n"
    )
    model = Model(
        name="metadata-probe",
        source=SourceEnum.LOCAL_PATH,
        local_path=str(repository),
        backend=BackendEnum.VLLM,
        backend_parameters=["--trust-remote-code"] if trusted else [],
    )
    monkeypatch.setattr(
        scheduler, "evaluate_diffusion_model", AsyncMock(return_value=False)
    )
    if source == "hf-cache":
        model.source = SourceEnum.HUGGING_FACE
        model.huggingface_repo_id = "metadata/probe"
        cache_hf_repository(config, model.huggingface_repo_id, repository)
    elif source == "hf-download":
        model.source = SourceEnum.HUGGING_FACE
        model.huggingface_repo_id = f"metadata/{tmp_path.name}"

        def hf_download(repo_id, filename, **kwargs):
            assert repo_id == model.huggingface_repo_id
            assert filename == "config.json"
            assert kwargs["cache_dir"] == str(Path(config.cache_dir) / "huggingface")
            assert kwargs["token"] == config.huggingface_token
            snapshot = cache_hf_repository(config, repo_id, repository)
            return str(snapshot / filename)

        monkeypatch.setattr(transformers_hub, "hf_hub_download", hf_download)
    elif source.startswith("ms-"):
        model.source = SourceEnum.MODEL_SCOPE
        model.model_scope_model_id = f"metadata/{tmp_path.name}"
        cache = Path(config.cache_dir) / "model_scope" / model.model_scope_model_id
        if source == "ms-cache":
            shutil.copytree(repository, cache)
        else:

            def download(**kwargs):
                assert kwargs["model_id"] == model.model_scope_model_id
                assert kwargs["allow_file_pattern"] == ["config.json", "config.*.json"]
                cache.mkdir(parents=True)
                shutil.copy(repository / "config.json", cache)
                return str(cache)

            monkeypatch.setattr(hub, "modelscope_snapshot_download", download)
    elif source == "worker":
        model.local_path = str(tmp_path / "worker-only")
        monkeypatch.setattr(
            calculator,
            "read_local_path_file_from_workers",
            AsyncMock(return_value=config_dict),
        )

    await scheduler.evaluate_pretrained_config(model)
    assert model.categories == [CategoryEnum.LLM]
    selector = VLLMResourceFitSelector(
        config, model, [], resource_view=ResourceView([])
    )
    await selector._init_model_parameters([])
    assert selector._num_attention_heads == 16
    assert selector._model_params.num_key_value_heads == 4
    assert selector._model_params.derived_max_seq_len == 8192
    parsed = await calculator.get_pretrained_config_with_workers(
        model, trust_remote_code=True
    )
    assert parsed.hidden_size == 2048
    if source != "worker":
        assert (
            hub.get_pretrained_config(model, trust_remote_code=True).hidden_size == 2048
        )
    assert not marker.exists()
    assert model.backend_parameters == (["--trust-remote-code"] if trusted else [])


@pytest.mark.parametrize(
    "fixture,layers,heads,kv_heads,head_dim,max_len,vision_heads,experts",
    [
        ("qwen2", 24, 14, 2, 64, 32768, None, None),
        ("qwen3", 36, 32, 8, 128, 40960, None, None),
        ("qwen3_moe", 48, 32, 4, 128, 40960, None, 128),
        ("qwen3_vl", 36, 32, 8, 128, 262144, 16, None),
        ("qwen3_5", 32, 16, 4, 256, 262144, 16, None),
        ("llama", 80, 64, 8, 128, 131072, None, None),
        ("deepseek_v3", 61, 128, 128, 64, 163840, None, 256),
        ("deepseek_v32", 61, 128, 128, 64, 163840, None, 256),
        ("chatglm", 40, 32, 4, 128, 1048576, None, None),
        ("phi4mm", 32, 24, 8, 128, 131072, None, None),
        ("internvl", 28, 28, 4, 128, 65536, 16, None),
        ("gemma", 34, 8, 4, 256, 131072, 16, None),
        ("mistral", 32, 32, 8, 128, 32768, None, None),
        ("bert", 24, 16, None, 64, 512, None, None),
        ("qwen_asr", 28, 16, 8, 128, 65536, None, None),
    ],
)
@pytest.mark.asyncio
async def test_mainstream_model_metadata(
    tmp_path, fixture, layers, heads, kv_heads, head_dim, max_len, vision_heads, experts
):
    shutil.copy(FIXTURES / f"{fixture}.json", tmp_path / "config.json")
    model = Model(
        name=fixture,
        source=SourceEnum.LOCAL_PATH,
        local_path=str(tmp_path),
        backend=BackendEnum.VLLM,
    )
    await scheduler.evaluate_pretrained_config(model)
    assert model.categories
    parsed = await calculator.get_pretrained_config_with_workers(model)
    params = ModelParameters()
    params.from_model_pretrained_config(model, parsed)
    assert params.num_hidden_layers == layers
    assert params.num_attention_heads == heads
    assert params.num_key_value_heads == kv_heads
    assert params.head_dim == head_dim
    assert params.derived_max_seq_len == max_len
    assert params.vision_num_attention_heads == vision_heads
    assert params.moe_num_experts == experts
    if fixture.startswith("deepseek"):
        assert params.kv_lora_rank == 512
        assert params.qk_rope_head_dim == 64
        assert params.quantization_config["quant_method"] == "fp8"


@pytest.mark.parametrize("model_type", ["llama", "qwen3", "gpt2", "mpt"])
def test_installed_config_defaults_and_aliases(model_type):
    data = {"model_type": model_type, "auto_map": {"AutoConfig": "missing.Config"}}
    original = deepcopy(data)
    parsed = model_config.safe_pretrained_config_from_dict(data)
    expected = AutoConfig.for_model(model_type)
    for attr in ("num_hidden_layers", "num_attention_heads", "hidden_size"):
        assert getattr(parsed, attr) == getattr(expected, attr)
    assert data == original


@pytest.mark.parametrize("data", [[], None, "invalid", 42])
def test_invalid_config_is_rejected(data):
    with pytest.raises(ValueError, match="JSON object"):
        model_config.safe_pretrained_config_from_dict(data)


@pytest.mark.asyncio
async def test_unknown_architecture_is_not_accepted_as_compatible(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "custom",
                "architectures": ["UnknownCustomModel"],
                "auto_map": {"AutoConfig": "missing.Config"},
            }
        )
    )
    model = Model(
        name="unsupported",
        source=SourceEnum.LOCAL_PATH,
        local_path=str(tmp_path),
        backend=BackendEnum.VLLM,
        backend_parameters=["--trust-remote-code"],
    )
    with pytest.raises(ValueError, match="Unsupported architecture"):
        await scheduler.evaluate_pretrained_config(model)
