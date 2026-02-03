import pytest
from gpustack.scheduler.evaluator import evaluate_model_metadata
from tests.utils.model import new_model
from gpustack.scheduler.scheduler import evaluate_pretrained_config
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
