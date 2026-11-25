from functools import lru_cache
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from gpustack.policies import utils as policy_utils
from gpustack.policies.candidate_selectors.base_candidate_selector import (
    ModelParameters,
)
from gpustack.policies.candidate_selectors.vllm_resource_fit_selector import (
    VLLMResourceFitSelector,
)
from gpustack.policies.resource_view import ResourceView
from gpustack.policies.worker_filters.backend_framework_filter import (
    BackendFrameworkFilter,
)
from gpustack.scheduler import calculator, scheduler
from gpustack.schemas.catalog_source import (
    KIND_MODEL_SET,
    build_catalog_entries,
    normalize_catalog_yaml,
)
from gpustack.schemas.inference_backend import (
    InferenceBackend,
    VersionConfig,
    VersionConfigDict,
)
from gpustack.schemas.model_sets import Catalog, ModelSet
from gpustack.schemas.models import CategoryEnum, Model, ModelCreate, SourceEnum
from gpustack.schemas.source import SourceContent, SourceTypeEnum
from gpustack.schemas.workers import GPUDeviceStatus
from gpustack.server.catalog import read_builtin_catalog_text
from gpustack.utils.compat_importlib import pkg_resources
from gpustack.utils.hybrid_attention import is_hybrid_attention
from gpustack.utils.model_catalog import filter_specs_by_gpu
from tests.fixtures.workers.fixtures import (
    linux_nvidia_4_4080_16gx4,
    linux_nvidia_22_H100_80gx8,
)

REPO_ROOT = Path(__file__).parents[2]

# Flat native HF Transformers Qwen3ASRForConditionalGeneration config.json
# from oruk/oruk-fourier-2b@8c2a8871f1304aa2f634da4fb95e24438c4945b9.
ORUK_FOURIER_2B_CONFIG = {
    "architectures": ["Qwen3ASRForConditionalGeneration"],
    "audio_config": {
        "d_model": 1024,
        "downsample_hidden_size": 480,
        "encoder_attention_heads": 16,
        "encoder_ffn_dim": 4096,
        "encoder_layers": 24,
        "max_position_embeddings": 13,
        "model_type": "qwen3_asr_encoder",
        "num_key_value_heads": 16,
        "num_mel_bins": 128,
        "output_dim": 2048,
    },
    "audio_token_id": 151676,
    "dtype": "bfloat16",
    "eos_token_id": [151643, 151645],
    "model_type": "qwen3_asr",
    "pad_token_id": 151645,
    "text_config": {
        "attention_bias": False,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "layer_types": ["full_attention"] * 28,
        "max_position_embeddings": 65536,
        "max_window_layers": 28,
        "model_type": "qwen3",
        "num_attention_heads": 16,
        "num_hidden_layers": 28,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [24, 20, 20],
            "rope_theta": 1000000,
            "rope_type": "default",
        },
        "tie_word_embeddings": True,
        "vocab_size": 151936,
    },
    "tie_word_embeddings": True,
}


@lru_cache(maxsize=1)
def _load_oruk_fourier_model_set() -> ModelSet:
    path = pkg_resources.files("gpustack.assets").joinpath("model-catalog.yaml")
    raw_text = path.read_text(encoding="utf-8")
    catalog = Catalog.model_validate(yaml.safe_load(raw_text))
    direct = next(m for m in catalog.model_sets if m.name == "Oruk-Fourier-2B")

    # Also verify through the full CatalogSource normalization pipeline.
    normalized = normalize_catalog_yaml(read_builtin_catalog_text(str(path)))
    entries = build_catalog_entries(
        [SourceContent("builtin", SourceTypeEnum.BUILTIN, normalized)]
    )
    via_pipeline = next(
        ModelSet(**e.payload)
        for e in entries
        if e.kind == KIND_MODEL_SET and e.name == "Oruk-Fourier-2B"
    )
    assert via_pipeline.name == direct.name
    return via_pipeline


def test_oruk_fourier_2b_catalog_metadata_and_icon():
    model_set = _load_oruk_fourier_model_set()
    assert model_set.home == "https://huggingface.co/oruk/oruk-fourier-2b"
    assert model_set.icon == "/static/catalog_icons/oruk.png"
    assert (REPO_ROOT / model_set.icon.lstrip("/")).is_file()
    assert model_set.size == 2
    assert model_set.categories == [CategoryEnum.SPEECH_TO_TEXT]
    assert model_set.licenses == ["cc-by-nc-4.0"]
    assert model_set.release_date.isoformat() == "2026-11-22"
    assert "English-only" in model_set.description
    assert "CC BY-NC 4.0" in model_set.description

    assert len(model_set.specs) == 1
    spec = model_set.specs[0]
    assert spec.mode == "standard"
    assert spec.quantization == "BF16"
    assert spec.source == SourceEnum.HUGGING_FACE
    assert spec.huggingface_repo_id == "oruk/oruk-fourier-2b"
    assert spec.backend == "vLLM"
    assert spec.backend_version == "0.34.0"
    assert spec.categories == [CategoryEnum.SPEECH_TO_TEXT]

    model_create = ModelCreate(**{**spec.model_dump(), "name": "oruk-fourier-2b"})
    assert model_create.huggingface_repo_id == "oruk/oruk-fourier-2b"
    assert model_create.backend == "vLLM"
    assert model_create.backend_version == "0.34.0"


@pytest.mark.parametrize(
    "vendor,expected_count",
    [
        ("nvidia", 1),
        ("amd", 1),
        ("ascend", 0),
        ("hygon", 0),
        ("mthreads", 0),
    ],
)
def test_oruk_fourier_2b_gpu_vendor_filtering(vendor, expected_count):
    model_set = _load_oruk_fourier_model_set()
    specs = filter_specs_by_gpu(
        [GPUDeviceStatus(vendor=vendor)],
        model_set.specs,
    )
    assert len(specs) == expected_count


@pytest.mark.asyncio
async def test_oruk_fourier_2b_flat_config_evaluation_and_candidate_selection(
    config, tmp_path, monkeypatch
):
    model_set = _load_oruk_fourier_model_set()
    spec = model_set.specs[0].model_dump()
    spec.update(
        name="Oruk-Fourier-2B",
        source=SourceEnum.LOCAL_PATH,
        local_path=str(tmp_path),
    )
    (tmp_path / "config.json").write_text(
        json.dumps(ORUK_FOURIER_2B_CONFIG), encoding="utf-8"
    )
    model = Model(**spec)

    await scheduler.evaluate_pretrained_config(model)
    assert model.categories == [CategoryEnum.SPEECH_TO_TEXT]

    parsed = await calculator.get_pretrained_config_with_workers(model)
    assert parsed.architectures == ["Qwen3ASRForConditionalGeneration"]
    assert is_hybrid_attention(parsed) is False

    params = ModelParameters()
    params.from_model_pretrained_config(model, parsed)
    assert params.architectures == ["Qwen3ASRForConditionalGeneration"]
    assert params.num_hidden_layers == 28
    assert params.hidden_size == 2048
    assert params.num_attention_heads == 16
    assert params.num_key_value_heads == 8
    assert params.head_dim == 128
    assert params.derived_max_seq_len == 65536

    worker = linux_nvidia_22_H100_80gx8()
    monkeypatch.setattr(
        policy_utils,
        "get_local_model_weight_size",
        AsyncMock(return_value=4_076_193_080),
    )
    selector = VLLMResourceFitSelector(
        config, model, [], resource_view=ResourceView([])
    )
    candidates = await selector.select_candidates([worker])
    assert candidates, selector._messages
    assert all(
        all(v > 0 for v in c.computed_resource_claim.vram.values()) for c in candidates
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "available_version,expect_match",
    [
        ("0.33.0", False),
        ("0.34.0", True),
    ],
)
async def test_oruk_fourier_2b_requires_vllm_0_34_0(available_version, expect_match):
    model_set = _load_oruk_fourier_model_set()
    spec = model_set.specs[0].model_dump()
    spec.update(id=1, name="oruk-fourier-2b", replicas=1, ready_replicas=0)
    model = Model(**spec)
    worker = linux_nvidia_4_4080_16gx4()

    backend = InferenceBackend(
        id=1,
        backend_name="vLLM",
        version_configs=VersionConfigDict(
            root={
                available_version: VersionConfig(
                    image_name=f"vllm/vllm-openai:v{available_version}",
                    built_in_frameworks=["cuda", "rocm"],
                    custom_framework="",
                )
            }
        ),
        default_version=available_version,
        is_built_in=True,
    )

    mock_session = AsyncMock()
    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_ctx.__aexit__ = AsyncMock(return_value=None)

    with (
        patch(
            "gpustack.policies.worker_filters.backend_framework_filter.async_session",
            return_value=mock_ctx,
        ),
        patch(
            "gpustack.policies.worker_filters.backend_framework_filter.InferenceBackend.all",
            new=AsyncMock(return_value=[backend]),
        ),
        patch(
            "gpustack.policies.worker_filters.backend_framework_filter.RunnerOverrideEntry.all",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "gpustack.policies.worker_filters.backend_framework_filter.merged_service_runners",
            return_value=[],
        ),
    ):
        filtered, messages = await BackendFrameworkFilter(model).filter([worker])

    assert (len(filtered) == 1) is expect_match
    if not expect_match:
        assert any("0.34.0" in m for m in messages)
