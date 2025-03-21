from tenacity import retry, stop_after_attempt, wait_fixed
from gpustack.utils.hub import (
    get_model_weight_size,
)
from gpustack.schemas.models import (
    Model,
    SourceEnum,
)


def test_get_hub_model_weight_size():
    model_to_weight_sizes = [
        (
            Model(
                source=SourceEnum.HUGGING_FACE,
                huggingface_repo_id="Qwen/Qwen2-0.5B-Instruct",
            ),
            988_097_824,
        ),
        (
            Model(
                source=SourceEnum.HUGGING_FACE,
                huggingface_repo_id="Qwen/Qwen2-VL-7B-Instruct",
            ),
            16_582_831_200,
        ),
        (
            Model(
                source=SourceEnum.HUGGING_FACE,
                huggingface_repo_id="Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4",
            ),
            41_621_048_632,
        ),
        (
            Model(
                source=SourceEnum.HUGGING_FACE,
                huggingface_repo_id="unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
            ),
            39_518_238_055,
        ),
        (
            Model(
                source=SourceEnum.HUGGING_FACE,
                huggingface_repo_id="deepseek-ai/DeepSeek-R1",
            ),
            688_586_727_753,
        ),
        (
            Model(
                source=SourceEnum.HUGGING_FACE,
                huggingface_repo_id="Systran/faster-whisper-large-v3",
            ),
            3_087_284_237,
        ),
        (
            Model(
                source=SourceEnum.HUGGING_FACE,
                huggingface_repo_id="FunAudioLLM/CosyVoice2-0.5B",
            ),
            3_545_354_370,
        ),
        (
            Model(
                source=SourceEnum.MODEL_SCOPE,
                model_scope_model_id="Qwen/Qwen2-0.5B-Instruct",
            ),
            988_097_824,
        ),
        (
            Model(
                source=SourceEnum.MODEL_SCOPE,
                model_scope_model_id="Qwen/Qwen2-VL-7B-Instruct",
            ),
            16_582_831_200,
        ),
        (
            Model(
                source=SourceEnum.MODEL_SCOPE,
                model_scope_model_id="Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4",
            ),
            41_621_048_632,
        ),
        (
            Model(
                source=SourceEnum.MODEL_SCOPE,
                model_scope_model_id="unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
            ),
            39_518_238_055,
        ),
        (
            Model(
                source=SourceEnum.MODEL_SCOPE,
                model_scope_model_id="deepseek-ai/DeepSeek-R1",
            ),
            688_586_727_753,
        ),
        (
            Model(
                source=SourceEnum.MODEL_SCOPE,
                model_scope_model_id="gpustack/faster-whisper-large-v3",
            ),
            3_087_284_237,
        ),
        (
            Model(
                source=SourceEnum.MODEL_SCOPE,
                model_scope_model_id="iic/CosyVoice2-0.5B",
            ),
            3_545_354_370,
        ),
    ]

    for model, expected_weight_size in model_to_weight_sizes:
        computed = get_hub_model_weight_size_with_retry(model)
        assert (
            computed == expected_weight_size
        ), f"weight_size mismatch for {model}, computed: {computed}, expected: {expected_weight_size}"


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def get_hub_model_weight_size_with_retry(model: Model) -> int:
    return get_model_weight_size(model)
