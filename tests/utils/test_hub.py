import pytest
from tenacity import retry, stop_after_attempt, wait_fixed
from gpustack.utils.hub import (
    get_hugging_face_model_min_gguf_path,
    get_model_scope_model_min_gguf_path,
    get_model_weight_size,
    match_hugging_face_files,
    match_model_scope_file_paths,
    read_repo_file_content,
)
from gpustack.schemas.models import (
    Model,
    SourceEnum,
)
from tests.utils.model import new_model


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
                model_scope_model_id="gpustack/CosyVoice2-0.5B",
            ),
            2_557_256_546,
            # The CosyVoice2-0.5B repository contains a subdirectory named CosyVoice-BlankEN,
            # which is optional and should be excluded from weight calculations.
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


def test_get_hf_min_gguf_file():
    model_to_gguf_file_path = [
        (
            "Qwen/Qwen2-0.5B-Instruct-GGUF",
            "qwen2-0_5b-instruct-q2_k.gguf",
        ),
        (
            "bartowski/Qwen2-VL-7B-Instruct-GGUF",
            "Qwen2-VL-7B-Instruct-IQ2_M.gguf",
        ),
        (
            "Qwen/Qwen2.5-72B-Instruct-GGUF",
            "qwen2.5-72b-instruct-q2_k-00001-of-00007.gguf",
        ),
        (
            "unsloth/Llama-3.3-70B-Instruct-GGUF",
            "Llama-3.3-70B-Instruct-UD-IQ1_M.gguf",
        ),
        (
            "unsloth/DeepSeek-R1-GGUF",
            "DeepSeek-R1-UD-IQ1_M/DeepSeek-R1-UD-IQ1_M-00001-of-00004.gguf",
        ),
    ]

    for model, expected_file_path in model_to_gguf_file_path:
        got = get_hugging_face_model_min_gguf_path(model)
        assert (
            got == expected_file_path
        ), f"min GGUF file path mismatch for huggingface model {model}, got: {got}, expected: {expected_file_path}"


def test_get_ms_min_gguf_file():
    model_to_gguf_file_path = [
        (
            "Qwen/Qwen2-0.5B-Instruct-GGUF",
            "qwen2-0_5b-instruct-q2_k.gguf",
        ),
        (
            "bartowski/Qwen2-VL-7B-Instruct-GGUF",
            "Qwen2-VL-7B-Instruct-IQ2_M.gguf",
        ),
        (
            "Qwen/Qwen2.5-72B-Instruct-GGUF",
            "qwen2.5-72b-instruct-q2_k-00001-of-00007.gguf",
        ),
        (
            "unsloth/Llama-3.3-70B-Instruct-GGUF",
            "Llama-3.3-70B-Instruct-UD-IQ1_M.gguf",
        ),
        (
            "unsloth/DeepSeek-R1-GGUF",
            "DeepSeek-R1-UD-IQ1_M/DeepSeek-R1-UD-IQ1_M-00001-of-00004.gguf",
        ),
    ]

    for model, expected_file_path in model_to_gguf_file_path:
        got = get_model_scope_model_min_gguf_path(model)
        assert (
            got == expected_file_path
        ), f"min GGUF file path mismatch for modelscope model {model}, got: {got}, expected: {expected_file_path}"


@pytest.mark.parametrize(
    "m, file, token, predicate",
    [
        (
            new_model(
                id=1,
                name="test_name",
                huggingface_repo_id="Qwen/Qwen3-0.6B",
            ),
            "config.json",
            None,
            lambda content: "Qwen3ForCausalLM" in content.get("architectures", []),
        ),
        (
            new_model(id=2, name="test_name2", model_scope_model_id="Qwen/Qwen3-0.6B"),
            "config.json",
            None,
            lambda content: "Qwen3ForCausalLM" in content.get("architectures", []),
        ),
    ],
)
def test_read_repo_file_content(m, file, token, predicate):
    config_dict = read_repo_file_content(m, file, token)
    assert predicate(config_dict)


def test_match_files_with_mmproj_at_root():
    repo_id = "unsloth/Qwen3.5-4B-GGUF"

    hf_matched = match_hugging_face_files(
        repo_id=repo_id,
        filename="Qwen3.5-4B-Q4_K_S.gguf",
        extra_filename="*mmproj*.gguf",
    )

    assert hf_matched == [
        "Qwen3.5-4B-Q4_K_S.gguf",
        "mmproj-F32.gguf",
    ]

    ms_matched = match_model_scope_file_paths(
        model_id=repo_id,
        file_path="Qwen3.5-4B-Q4_K_S.gguf",
        extra_file_path="*mmproj*.gguf",
    )

    assert ms_matched == [
        "Qwen3.5-4B-Q4_K_S.gguf",
        "mmproj-F32.gguf",
    ]


def test_match_file_paths_in_subdir_and_mmproj_at_root():
    repo_id = "unsloth/Qwen3.5-397B-A17B-GGUF"

    expected = [
        "UD-Q6_K_XL/Qwen3.5-397B-A17B-UD-Q6_K_XL-00001-of-00009.gguf",
        "UD-Q6_K_XL/Qwen3.5-397B-A17B-UD-Q6_K_XL-00002-of-00009.gguf",
        "UD-Q6_K_XL/Qwen3.5-397B-A17B-UD-Q6_K_XL-00003-of-00009.gguf",
        "UD-Q6_K_XL/Qwen3.5-397B-A17B-UD-Q6_K_XL-00004-of-00009.gguf",
        "UD-Q6_K_XL/Qwen3.5-397B-A17B-UD-Q6_K_XL-00005-of-00009.gguf",
        "UD-Q6_K_XL/Qwen3.5-397B-A17B-UD-Q6_K_XL-00006-of-00009.gguf",
        "UD-Q6_K_XL/Qwen3.5-397B-A17B-UD-Q6_K_XL-00007-of-00009.gguf",
        "UD-Q6_K_XL/Qwen3.5-397B-A17B-UD-Q6_K_XL-00008-of-00009.gguf",
        "UD-Q6_K_XL/Qwen3.5-397B-A17B-UD-Q6_K_XL-00009-of-00009.gguf",
        "mmproj-F32.gguf",
    ]

    hf_matched = match_hugging_face_files(
        repo_id=repo_id,
        filename="UD-Q6_K_XL/*.gguf",
        extra_filename="*mmproj*.gguf",
    )
    assert hf_matched == expected

    ms_matched = match_model_scope_file_paths(
        model_id=repo_id,
        file_path="UD-Q6_K_XL/*.gguf",
        extra_file_path="*mmproj*.gguf",
    )
    assert ms_matched == expected
