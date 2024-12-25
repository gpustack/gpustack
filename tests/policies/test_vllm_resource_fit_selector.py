from gpustack.config.config import set_global_config, Config
from gpustack.policies.candidate_selectors.vllm_resource_fit_selector import (
    get_hub_model_weight_size,
)
from gpustack.schemas.models import Model, SourceEnum


def test_get_hub_model_weight_size():
    model_to_weight_sizes = [
        (
            Model(
                source=SourceEnum.HUGGING_FACE,
                huggingface_repo_id="Qwen/Qwen2-0.5B-Instruct",
            ),
            1_000_000_000,
        ),
        (
            Model(
                source=SourceEnum.HUGGING_FACE,
                huggingface_repo_id="Qwen/Qwen2-VL-7B-Instruct",
            ),
            14_000_000_000,
        ),
        (
            Model(
                source=SourceEnum.HUGGING_FACE,
                huggingface_repo_id="Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4",
            ),
            36_000_000_000,
        ),
        (
            Model(
                source=SourceEnum.HUGGING_FACE,
                huggingface_repo_id="unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
            ),
            35_000_000_000,
        ),
        (
            Model(
                source=SourceEnum.MODEL_SCOPE,
                model_scope_model_id="Qwen/Qwen2-0.5B-Instruct",
            ),
            1_000_000_000,
        ),
        (
            Model(
                source=SourceEnum.MODEL_SCOPE,
                model_scope_model_id="Qwen/Qwen2-VL-7B-Instruct",
            ),
            14_000_000_000,
        ),
        (
            Model(
                source=SourceEnum.MODEL_SCOPE,
                model_scope_model_id="Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4",
            ),
            36_000_000_000,
        ),
        (
            Model(
                source=SourceEnum.MODEL_SCOPE,
                model_scope_model_id="unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
            ),
            35_000_000_000,
        ),
    ]

    set_global_config(Config(data_dir="/tmp/test_data_dir"))

    for model, expected_weight_size in model_to_weight_sizes:
        computed = get_hub_model_weight_size(model)
        assert (
            computed == expected_weight_size
        ), f"weight_size mismatch for {model}, computed: {computed}, expected: {expected_weight_size}"
