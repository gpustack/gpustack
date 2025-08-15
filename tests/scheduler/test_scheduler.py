import pytest
from tests.utils.model import new_model
from gpustack.scheduler.scheduler import evaluate_pretrained_config
from gpustack.schemas.models import CategoryEnum, BackendEnum


@pytest.mark.asyncio
async def test_evaluate_pretrained_config(config):
    Phi_4_multimodal = new_model(
        1,
        "test_name",
        1,
        huggingface_repo_id="microsoft/Phi-4-multimodal-instruct",
        backend=BackendEnum.VLLM,
        backend_parameters=[],
    )

    # First run,will cache all file and automatic carry --trust-remote-code.
    await evaluate_pretrained_config(Phi_4_multimodal)

    # Custom code without --trust-remote-code, should raise ValueError
    with pytest.raises(
        ValueError,
        match="The model contains custom code that must be executed to load correctly. If you trust the source, please pass the backend parameter `--trust-remote-code` to allow custom code to be run.",
    ):
        await evaluate_pretrained_config(Phi_4_multimodal)

    # Custom code with --trust-remote-code, should not raise ValueError
    Phi_4_multimodal.backend_parameters = ["--trust-remote-code"]
    await evaluate_pretrained_config(Phi_4_multimodal)
    assert Phi_4_multimodal.categories == [CategoryEnum.LLM]

    t5 = new_model(
        1,
        "test_name",
        1,
        huggingface_repo_id="google-t5/t5-base",
        backend=BackendEnum.VLLM,
        backend_parameters=[],
    )

    # Model architecture not supported, should raise ValueError
    with pytest.raises(ValueError, match="Not a supported model"):
        await evaluate_pretrained_config(t5)

    qwen = new_model(
        1,
        "test_name",
        1,
        huggingface_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        backend=BackendEnum.VLLM,
        backend_parameters=[],
    )

    # Model architecture supported, should not raise ValueError
    await evaluate_pretrained_config(qwen)
    assert qwen.categories == [CategoryEnum.LLM]
