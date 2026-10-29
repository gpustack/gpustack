import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from transformers import PretrainedConfig
from gpustack.policies.candidate_selectors import VLLMResourceFitSelector
from gpustack.policies.utils import should_skip_gpu_count_check
from gpustack.scheduler.evaluator import evaluate_model_metadata
from gpustack.scheduler import evaluator, scheduler
from tests.utils.model import make_model, new_model
from gpustack.scheduler.scheduler import (
    UNSUPPORTED_MODEL_REMEDY,
    evaluate_pretrained_config,
    set_model_gpus_per_replica,
)
from gpustack.schemas.models import CategoryEnum, BackendEnum, ComputedResourceClaim
from gpustack.schemas.model_evaluations import ModelSpec


@pytest.fixture(autouse=True)
def offline_model_metadata(monkeypatch):
    fixtures = Path(__file__).parents[1] / "fixtures" / "model_configs"
    cases = json.loads((fixtures / "scheduler_cases.json").read_text())
    snapshots = {
        "microsoft/Phi-4-multimodal-instruct": "phi4mm.json",
        "Qwen/Qwen2-0.5B-Instruct": "qwen2.json",
        "deepseek-ai/DeepSeek-R1": "deepseek_v3.json",
    }
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", True)

    def forbidden(*args, **kwargs):
        pytest.fail("Scheduler tests must not use network or repository Python")

    def get_config_dict(repo_id, **kwargs):
        if repo_id in snapshots:
            return json.loads((fixtures / snapshots[repo_id]).read_text()), {}
        assert repo_id in cases, f"Missing configuration fixture: {repo_id}"
        return cases[repo_id], {}

    def read_repo_file_content(model, file_path, **kwargs):
        assert file_path == "model_index.json"
        if model.huggingface_repo_id == "Tongyi-MAI/Z-Image-Turbo":
            return cases[model.huggingface_repo_id]
        return None

    monkeypatch.setattr(
        PretrainedConfig, "get_config_dict", staticmethod(get_config_dict)
    )
    monkeypatch.setattr(
        "gpustack.utils.hub.read_repo_file_content", read_repo_file_content
    )
    monkeypatch.setattr(
        "huggingface_hub.HfApi.auth_check", lambda *args, **kwargs: None
    )
    monkeypatch.setattr("requests.Session.request", forbidden)
    monkeypatch.setattr("httpx.Client.send", forbidden)
    monkeypatch.setattr(
        "transformers.dynamic_module_utils.get_class_from_dynamic_module", forbidden
    )
    monkeypatch.setattr(
        "transformers.models.auto.configuration_auto.get_class_from_dynamic_module",
        forbidden,
    )


@pytest.mark.parametrize(
    "backend,version,image,compatible",
    [
        (None, None, None, True),
        (BackendEnum.VLLM, None, None, False),
        (BackendEnum.CUSTOM, None, None, True),
        (BackendEnum.VLLM, "custom-version", None, True),
        (BackendEnum.VLLM, None, "example/custom-vllm:latest", True),
    ],
)
@pytest.mark.asyncio
async def test_unknown_architecture_only_rejects_explicit_builtin_vllm(
    config, backend, version, image, compatible
):
    model = ModelSpec(
        source="huggingface",
        huggingface_repo_id="LiquidAI/LFM3-1.7B",
        backend=backend,
        backend_version=version,
        image_name=image,
    )
    actual, messages = await evaluate_model_metadata(config, model, [])
    assert actual == compatible
    if not compatible:
        assert "Unsupported architecture: ['Lfm3ForCausalLM']" in messages[0]
    else:
        assert messages == []


@pytest.mark.asyncio
async def test_unselected_backend_routes_gguf_to_gguf_evaluation(config, monkeypatch):
    model = ModelSpec(
        source="huggingface",
        huggingface_repo_id="LiquidAI/LFM3-1.7B-GGUF",
        huggingface_filename="model.gguf",
    )
    evaluate_gguf = AsyncMock()
    monkeypatch.setattr(scheduler, "evaluate_gguf_model", evaluate_gguf)
    assert await evaluate_model_metadata(config, model, []) == (True, [])
    evaluate_gguf.assert_awaited_once_with(model, workers=[])
    assert model.backend is None


@pytest.mark.parametrize("backend", [None, BackendEnum.VLLM])
@pytest.mark.asyncio
async def test_diffusion_detection_precedes_text_architecture_check(config, backend):
    model = ModelSpec(
        source="huggingface",
        huggingface_repo_id="Tongyi-MAI/Z-Image-Turbo",
        backend=backend,
    )
    assert await evaluate_model_metadata(config, model, []) == (True, [])
    assert model.categories == [CategoryEnum.IMAGE]


@pytest.mark.parametrize(
    "backend",
    [
        None,
        BackendEnum.VLLM,
        BackendEnum.SGLANG,
        BackendEnum.ASCEND_MINDIE,
        BackendEnum.CUSTOM,
    ],
)
@pytest.mark.parametrize(
    "repo_id", ["microsoft/Phi-4-multimodal-instruct", "deepseek-ai/DeepSeek-R1"]
)
@pytest.mark.parametrize("trusted", [False, True])
@pytest.mark.asyncio
async def test_auto_map_does_not_determine_backend_compatibility(
    config, backend, repo_id, trusted
):
    parameters = ["--trust-remote-code"] if trusted else []
    model = ModelSpec(
        source="huggingface",
        huggingface_repo_id=repo_id,
        backend=backend,
        backend_parameters=parameters.copy(),
    )
    assert await evaluate_model_metadata(config, model, []) == (True, [])
    assert model.backend_parameters == parameters
    assert model.categories == [CategoryEnum.LLM]


@pytest.mark.parametrize("failure", [None, "environment", "kv_cache", "resources"])
@pytest.mark.parametrize("trusted", [False, True])
@pytest.mark.parametrize(
    "repo_id", ["microsoft/Phi-4-multimodal-instruct", "deepseek-ai/DeepSeek-R1"]
)
@pytest.mark.asyncio
async def test_auto_map_preserves_full_evaluation(
    config, monkeypatch, failure, trusted, repo_id
):
    parameters = ["--trust-remote-code"] if trusted else []
    model = ModelSpec(
        source="huggingface",
        huggingface_repo_id=repo_id,
        backend=BackendEnum.VLLM,
        backend_parameters=parameters.copy(),
    )
    worker = SimpleNamespace(id=1, cluster_id=1)
    candidate = SimpleNamespace(
        overcommit=False,
        subordinate_workers=None,
        computed_resource_claim=ComputedResourceClaim(ram=1024, vram={0: 4096}),
    )
    environment = AsyncMock(
        return_value=(
            (False, ["Environment incompatible"])
            if failure == "environment"
            else (True, [])
        )
    )
    kv_cache = AsyncMock(
        return_value=(
            (False, ["KV cache incompatible"]) if failure == "kv_cache" else (True, [])
        )
    )
    resources = AsyncMock(
        return_value=(
            (None, ["Not enough VRAM"]) if failure == "resources" else (candidate, [])
        )
    )
    monkeypatch.setattr(evaluator, "set_default_spec", AsyncMock(return_value=False))
    monkeypatch.setattr(evaluator, "set_gguf_model_file_path", AsyncMock())
    monkeypatch.setattr(
        evaluator, "evaluate_model_input", AsyncMock(return_value=(True, []))
    )
    monkeypatch.setattr(evaluator, "evaluate_environment", environment)
    monkeypatch.setattr(evaluator, "evaluate_hybrid_model_kv_cache", kv_cache)
    monkeypatch.setattr(scheduler, "find_candidate", resources)
    result = await evaluator.evaluate_model(config, None, model, [worker], [])
    assert result.compatible == (failure is None)
    assert not result.error
    environment.assert_awaited_once()
    assert kv_cache.await_count == (0 if failure == "environment" else 1)
    assert resources.await_count == (0 if failure in {"environment", "kv_cache"} else 1)
    if failure is None:
        assert result.resource_claim.model_dump() == {"ram": 1024, "vram": 4096}
        assert result.resource_claim_by_cluster_id == {1: result.resource_claim}
    else:
        assert result.resource_claim is None
        expected = {
            "environment": "Environment incompatible",
            "kv_cache": "KV cache incompatible",
            "resources": "Unable to find a schedulable worker for the model.",
        }
        assert result.compatibility_messages[0] == expected[failure]
    assert result.scheduling_messages == (
        ["Not enough VRAM"] if failure == "resources" else []
    )
    assert len(result.compatibility_messages) == int(failure is not None)
    assert model.backend_parameters == parameters


@pytest.mark.asyncio
async def test_model_without_remote_code_declaration_has_no_hint(config):
    model = ModelSpec(
        source="huggingface",
        huggingface_repo_id="Qwen/Qwen2-0.5B-Instruct",
        backend=BackendEnum.VLLM,
    )
    assert await evaluate_model_metadata(config, model, []) == (True, [])


@pytest.mark.parametrize(
    "case_name, model, expect_error, expect_error_match, expect_categories",
    [
        (
            # Metadata inspection does not require executing custom model code.
            "custom_code_without_trust_remote_code",
            new_model(
                1,
                "test_name",
                1,
                huggingface_repo_id="microsoft/Phi-4-multimodal-instruct",
                backend=BackendEnum.VLLM,
                backend_parameters=[],
            ),
            None,
            None,
            ["LLM"],
        ),
        (
            # Inference trust flags do not change metadata inspection.
            "custom_code_with_trust_remote_code",
            new_model(
                1,
                "test_name",
                1,
                huggingface_repo_id="microsoft/Phi-4-multimodal-instruct",
                backend=BackendEnum.VLLM,
                backend_parameters=["--trust-remote-code"],
            ),
            None,
            None,
            ["LLM"],
        ),
        (
            # Checkpoint:
            # The model is of an unsupported architecture.
            # This should raise a ValueError with a specific message.
            "unsupported_architecture",
            new_model(
                1,
                "test_name",
                1,
                huggingface_repo_id="google-t5/t5-base",
                backend=BackendEnum.VLLM,
                backend_parameters=[],
            ),
            ValueError,
            "Unsupported architecture:",
            None,
        ),
        (
            # Checkpoint:
            # The model is of an unsupported architecture using custom backend.
            # This should pass without errors.
            "pass_unsupported_architecture_custom_backend",
            new_model(
                1,
                "test_name",
                1,
                huggingface_repo_id="google-t5/t5-base",
                backend=BackendEnum.CUSTOM,
                backend_parameters=[],
            ),
            None,
            None,
            None,
        ),
        (
            # Checkpoint:
            # The model is of an unsupported architecture using custom backend version.
            # This should pass without errors.
            "pass_unsupported_architecture_custom_backend_version",
            new_model(
                1,
                "test_name",
                1,
                huggingface_repo_id="google-t5/t5-base",
                backend=BackendEnum.VLLM,
                backend_version="custom_version",
                backend_parameters=[],
            ),
            None,
            None,
            None,
        ),
        (
            # Checkpoint:
            # The model is of an unsupported architecture using a custom image.
            # This should pass without errors.
            "pass_unsupported_architecture_custom_image",
            new_model(
                1,
                "test_name",
                1,
                huggingface_repo_id="google-t5/t5-base",
                backend=BackendEnum.VLLM,
                image_name="vllm/vllm-openai:nightly",
                backend_parameters=[],
            ),
            None,
            None,
            None,
        ),
        (
            # Checkpoint:
            # The model is of a supported architecture.
            # This should pass without errors.
            "supported_architecture",
            new_model(
                1,
                "test_name",
                1,
                huggingface_repo_id="Qwen/Qwen2-0.5B-Instruct",
                backend=BackendEnum.VLLM,
                backend_parameters=[],
            ),
            None,
            None,
            ["LLM"],
        ),
        (
            # Custom configuration metadata supports classification without imports.
            "custom_architecture_without_repository_imports",
            new_model(
                1,
                "test_name",
                1,
                huggingface_repo_id="deepseek-ai/DeepSeek-OCR",
                backend=BackendEnum.VLLM,
                backend_parameters=["--trust-remote-code"],
            ),
            None,
            None,
            ["LLM"],
        ),
        (
            # Checkpoint:
            # Image model.
            # This should pass without errors.
            "pass_image_model",
            new_model(
                1,
                "test_name",
                1,
                huggingface_repo_id="Tongyi-MAI/Z-Image-Turbo",
                backend=BackendEnum.SGLANG,
                backend_parameters=[],
                categories=[CategoryEnum.IMAGE],
            ),
            None,
            None,
            ["IMAGE"],
        ),
    ],
)
@pytest.mark.asyncio
async def test_evaluate_pretrained_config(
    config, case_name, model, expect_error, expect_error_match, expect_categories
):
    try:
        if expect_error:
            with pytest.raises(expect_error, match=expect_error_match):
                await evaluate_pretrained_config(model)
        else:
            await evaluate_pretrained_config(model)
            if expect_categories:
                assert model.categories == [CategoryEnum[c] for c in expect_categories]
    except AssertionError as e:
        raise AssertionError(f"Test case '{case_name}' failed: {e}") from e


@pytest.mark.parametrize(
    "case_name, model, expect_compatible, expect_error_match",
    [
        (
            # Checkpoint:
            # The model is of an unsupported architecture.
            # This should raise a ValueError with a specific message.
            "unsupported_architecture",
            new_model(
                1,
                "test_name",
                1,
                huggingface_repo_id="google-t5/t5-base",
                backend=BackendEnum.VLLM,
                backend_parameters=[],
            ),
            False,
            [
                f"Unsupported architecture: ['T5ForConditionalGeneration']. {UNSUPPORTED_MODEL_REMEDY}"
            ],
        ),
        (
            # Checkpoint:
            # The model is of an unsupported architecture but config environment variable set to skip evaluation.
            # This should return compatible.
            "pass_evaluation_skip",
            new_model(
                1,
                "test_name",
                1,
                huggingface_repo_id="google-t5/t5-base",
                backend=BackendEnum.VLLM,
                backend_parameters=[],
                env={"GPUSTACK_SKIP_MODEL_EVALUATION": "1"},
            ),
            True,
            [],
        ),
    ],
)
@pytest.mark.asyncio
async def test_evaluate_model_metadata(
    config, case_name, model, expect_compatible, expect_error_match
):
    try:
        actual_compatible, actual_error = await evaluate_model_metadata(
            config, model, []
        )
        assert (
            actual_compatible == expect_compatible
        ), f"Expected compatibility: {expect_compatible}, but got: {actual_compatible}. Error: {actual_error}"
        assert (
            expect_error_match == actual_error
        ), f"Expected error message: {expect_error_match}, but got: {actual_error}"
    except AssertionError as e:
        raise AssertionError(f"Test case '{case_name}' failed: {e}") from e


# --- issue #5089: GPUSTACK_SKIP_GPU_COUNT_CHECK bypass ---------------------

_MANUAL_8_GPU_IDS = [f"w22:cuda:{i}" for i in range(8)]
_TP8_DP2_DPL1 = [
    "--tensor-parallel-size=8",
    "--data-parallel-size=2",
    "--data-parallel-size-local=1",
]


@pytest.mark.parametrize(
    "case_name, manual_select, env, expected",
    [
        (
            "no_gpu_selector_env_set",
            False,
            {"GPUSTACK_SKIP_GPU_COUNT_CHECK": "1"},
            False,
        ),
        ("manual_no_env", True, None, False),
        ("manual_env_1", True, {"GPUSTACK_SKIP_GPU_COUNT_CHECK": "1"}, True),
        ("manual_env_true", True, {"GPUSTACK_SKIP_GPU_COUNT_CHECK": "TRUE"}, True),
        ("manual_env_yes", True, {"GPUSTACK_SKIP_GPU_COUNT_CHECK": "yes"}, True),
        ("manual_env_on", True, {"GPUSTACK_SKIP_GPU_COUNT_CHECK": "on"}, True),
        ("manual_env_0", True, {"GPUSTACK_SKIP_GPU_COUNT_CHECK": "0"}, False),
        ("manual_env_false", True, {"GPUSTACK_SKIP_GPU_COUNT_CHECK": "false"}, False),
        ("manual_env_empty", True, {"GPUSTACK_SKIP_GPU_COUNT_CHECK": ""}, False),
    ],
)
def test_should_skip_gpu_count_check(case_name, manual_select, env, expected):
    model = make_model(
        gpus_per_replica=8,
        gpu_ids=_MANUAL_8_GPU_IDS if manual_select else None,
        backend=BackendEnum.VLLM.value,
        env=env,
    )
    assert should_skip_gpu_count_check(model) is expected, f"case '{case_name}' failed"


@pytest.mark.parametrize(
    "case_name, env, expected",
    [
        # world_size = tp*dp = 16 takes precedence by default.
        ("default", None, 16),
        # Bypass on: honor the manual selection (8 GPUs / 1 replica).
        ("skip", {"GPUSTACK_SKIP_GPU_COUNT_CHECK": "1"}, 8),
    ],
)
def test_set_model_gpus_per_replica_skip_gpu_count_check(case_name, env, expected):
    model = make_model(
        gpus_per_replica=None,
        gpu_ids=_MANUAL_8_GPU_IDS,
        repo_id="deepseek-ai/DeepSeek-R1",
        backend=BackendEnum.VLLM.value,
        backend_parameters=_TP8_DP2_DPL1,
        env=env,
    )
    set_model_gpus_per_replica(model)
    assert model.gpu_selector.gpus_per_replica == expected, f"case '{case_name}' failed"


def _build_selector_for_set_gpu_count(model) -> VLLMResourceFitSelector:
    # Bypass __init__ to exercise _set_gpu_count in isolation.
    selector = VLLMResourceFitSelector.__new__(VLLMResourceFitSelector)
    selector._model = model
    selector._gpu_count = 0
    selector._selected_gpu_workers = []
    selector._selected_gpu_indexes_by_gpu_type_and_worker = {}
    return selector


def test_set_gpu_count_mismatch_raises_by_default():
    model = make_model(
        gpus_per_replica=8,
        gpu_ids=_MANUAL_8_GPU_IDS,
        backend=BackendEnum.VLLM.value,
    )
    selector = _build_selector_for_set_gpu_count(model)
    with pytest.raises(ValueError, match="does not match the world size"):
        selector._set_gpu_count(world_size=16, strategies=["tp", "dp"])


def test_set_gpu_count_mismatch_bypassed():
    model = make_model(
        gpus_per_replica=8,
        gpu_ids=_MANUAL_8_GPU_IDS,
        backend=BackendEnum.VLLM.value,
        env={"GPUSTACK_SKIP_GPU_COUNT_CHECK": "1"},
    )
    selector = _build_selector_for_set_gpu_count(model)
    selector._set_gpu_count(world_size=16, strategies=["tp", "dp"])
    # Keeps the manual selection count instead of overwriting with world_size.
    assert selector._gpu_count == 8
