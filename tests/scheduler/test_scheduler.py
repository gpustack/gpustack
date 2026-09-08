import pytest
from gpustack.policies.candidate_selectors import VLLMResourceFitSelector
from gpustack.policies.utils import (
    manual_distributed_from_env,
    should_skip_gpu_count_check,
)
from gpustack.scheduler.evaluator import evaluate_model_metadata
from tests.utils.model import make_model, new_model
from gpustack.scheduler.scheduler import (
    evaluate_pretrained_config,
    set_model_gpus_per_replica,
)
from gpustack.schemas.models import CategoryEnum, BackendEnum


@pytest.mark.parametrize(
    "case_name, model, expect_error, expect_error_match, expect_categories",
    [
        (
            # Checkpoint:
            # The model contains custom code but `--trust-remote-code` is not provided.
            # This should raise a ValueError with a specific message.
            "custom_code_without_trust_remote_code",
            new_model(
                1,
                "test_name",
                1,
                huggingface_repo_id="microsoft/Phi-4-multimodal-instruct",
                backend=BackendEnum.VLLM,
                backend_parameters=[],
            ),
            ValueError,
            "The model contains custom code that must be executed to load correctly. If you trust the source, please pass the backend parameter `--trust-remote-code` to allow custom code to be run.",
            None,
        ),
        (
            # Checkpoint:
            # The model contains custom code and `--trust-remote-code` is provided.
            # This should pass without errors and set the model category to LLM.
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
            # The model is of a supported architecture.
            # This should pass without errors.
            "supported_architecture",
            new_model(
                1,
                "test_name",
                1,
                huggingface_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
                backend=BackendEnum.VLLM,
                backend_parameters=[],
            ),
            None,
            None,
            ["LLM"],
        ),
        (
            # Checkpoint:
            # The model could run with vllm backend but get import error while get pretrained config.
            # This should pass without errors.
            "pass_import_error_in_pretrained_config",
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
                "Unsupported architecture: ['T5ForConditionalGeneration']. To proceed with deployment, ensure the model is supported by backend, or deploy it using a custom backend version or custom backend."
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
        # GPUSTACK_MANUAL_DISTRIBUTED implies the bypass, but still only when
        # GPUs are manually selected (the gpu_selector guard stays in front).
        ("manual_distributed_on", True, {"GPUSTACK_MANUAL_DISTRIBUTED": "1"}, True),
        (
            "manual_distributed_no_selector",
            False,
            {"GPUSTACK_MANUAL_DISTRIBUTED": "1"},
            False,
        ),
        ("manual_distributed_off", True, {"GPUSTACK_MANUAL_DISTRIBUTED": "0"}, False),
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
        ("no_env", None, False),
        ("on_1", {"GPUSTACK_MANUAL_DISTRIBUTED": "1"}, True),
        ("on_true", {"GPUSTACK_MANUAL_DISTRIBUTED": "TRUE"}, True),
        ("on_yes", {"GPUSTACK_MANUAL_DISTRIBUTED": "yes"}, True),
        ("off_0", {"GPUSTACK_MANUAL_DISTRIBUTED": "0"}, False),
        ("off_empty", {"GPUSTACK_MANUAL_DISTRIBUTED": ""}, False),
        ("other_key", {"OTHER": "1"}, False),
    ],
)
def test_manual_distributed_from_env(case_name, env, expected):
    assert manual_distributed_from_env(env) is expected, f"case '{case_name}' failed"


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


def test_should_schedule_dp_node_per_instance_gates_members():
    """DP-node-per-instance: only the rank-0 coordinator enqueues for scheduling. Members (dp_rank > 0)
    are gated out — they get their worker from the coordinator's fan-out, not by
    racing for their own — while rank 0 and other instances schedule normally."""
    from datetime import datetime, timezone
    from gpustack.scheduler.scheduler import Scheduler
    from gpustack.schemas.models import ModelInstance, ModelInstanceStateEnum

    scheduler = object.__new__(Scheduler)
    now = datetime.now(timezone.utc)

    def instance(dp_rank):
        mi = ModelInstance(
            name="m",
            model_id=1,
            model_name="m",
            dp_rank=dp_rank,
            worker_id=None,
            state=ModelInstanceStateEnum.PENDING,
        )
        mi.created_at = now
        mi.updated_at = now
        return mi

    assert scheduler._should_schedule(instance(2)) is False  # member gated out
    assert scheduler._should_schedule(instance(0)) is True  # coordinator schedules
    assert (
        scheduler._should_schedule(instance(None)) is True
    )  # other instances unaffected
