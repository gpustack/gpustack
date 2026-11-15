import pytest
from tenacity import retry, stop_after_attempt, wait_fixed
from transformers import PretrainedConfig
from gpustack.utils import hub, model_config
from gpustack.utils.hub import (
    FileEntry,
    get_hugging_face_model_min_gguf_path,
    get_model_scope_model_min_gguf_path,
    get_model_weight_size,
    match_hugging_face_files,
    match_model_scope_file_paths,
    read_repo_file_content,
)
from gpustack.worker.downloaders import select_model_scope_file_info
from gpustack.schemas.models import (
    Model,
    SourceEnum,
)
from tests.utils.model import new_model


def test_get_hf_text_config_with_heterogeneous_attention_heads():
    config = PretrainedConfig(
        text_config={
            "num_hidden_layers": 2,
            "num_attention_heads": 64,
            "per_layer_config": {1: {"num_attention_heads": 96}},
        },
    )

    text_config = model_config.get_hf_text_config(config)

    assert text_config.per_layer_config[0].num_attention_heads == 64
    assert text_config.per_layer_config[1].num_attention_heads == 96
    assert not text_config.allow_global_per_layer_attribute_access


@pytest.mark.parametrize('status', [401, 403])
def test_hub_auth_errors_report_restricted_access(monkeypatch, status):
    from unittest.mock import MagicMock
    from huggingface_hub import hf_api
    from huggingface_hub.utils import httpx

    response = httpx.Response(
        status,
        request=httpx.Request(
            'GET', 'https://huggingface.co/api/models/test/model/auth-check'
        ),
    )
    session = MagicMock()
    session.get.return_value = response
    monkeypatch.setattr(hf_api, 'get_session', lambda: session)

    with pytest.raises(Exception, match='Access to the model is restricted'):
        hub.auth_check(
            Model(source=SourceEnum.HUGGING_FACE, huggingface_repo_id='test/model')
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


def test_get_hf_min_gguf_file_skips_mmproj(monkeypatch):
    # mmproj files sort before the main weights alphabetically; the main
    # model must be selected regardless of ordering or quantization match.
    files = [
        "mmproj-vllm-d3-mini-Q8_0.gguf",
        "vllm-d3-mini-Q8_0.gguf",
    ]
    monkeypatch.setattr(
        'gpustack.utils.hub.HfApi.list_repo_files', lambda self, model_id: files
    )

    got = get_hugging_face_model_min_gguf_path('ggml-org/vllm-d3-mini-GGUF')
    assert got == "vllm-d3-mini-Q8_0.gguf"


@pytest.mark.parametrize(
    "path,expected",
    [
        ("mmproj-vllm-d3-mini-Q8_0.gguf", True),
        # Case-insensitive: fnmatch is case-sensitive on Linux.
        ("MMPROJ-vllm-d3-mini-Q8_0.GGUF", True),
        # File-name matching only: a cache directory whose name contains
        # "mmproj" must not disqualify main weights.
        ("/srv/mmproj-cache/huggingface/org/model/model-Q8_0.gguf", False),
        (
            "/var/lib/gpustack/cache/model_scope/ggml-org/m/vllm-d3-mini-Q8_0.gguf",
            False,
        ),
    ],
)
def test_is_mmproj_file(path, expected):
    assert hub.is_mmproj_file(path) is expected


@pytest.mark.parametrize(
    "path,expected",
    [
        # Repo-relative matching also catches a projector that lives in an
        # "mmproj/" subfolder with a plain file name.
        ("mmproj/vllm-d3-mini-Q8_0.gguf", True),
        ("MMProj/vllm-d3-mini-Q8_0.gguf", True),
        ("mmproj-vllm-d3-mini-Q8_0.gguf", True),
        ("vllm-d3-mini-Q8_0.gguf", False),
        ("subdir/vllm-d3-mini-Q8_0.gguf", False),
    ],
)
def test_is_mmproj_repo_path(path, expected):
    assert hub.is_mmproj_repo_path(path) is expected


def test_get_ms_min_gguf_file_skips_mmproj(monkeypatch):
    files = [
        "mmproj-vllm-d3-mini-Q8_0.gguf",
        "vllm-d3-mini-Q8_0.gguf",
    ]
    monkeypatch.setattr(
        'gpustack.utils.hub.HubApi.get_model_files',
        lambda self, model_id, recursive=True: [{"Path": f} for f in files],
    )

    got = get_model_scope_model_min_gguf_path('ggml-org/vllm-d3-mini-GGUF')
    assert got == "vllm-d3-mini-Q8_0.gguf"


def _ms_source(file_path):
    from gpustack.schemas.models import ModelSource, SourceEnum

    return ModelSource(
        source=SourceEnum.MODEL_SCOPE,
        model_scope_model_id='ggml-org/vllm-d3-mini-GGUF',
        model_scope_file_path=file_path,
    )


def test_select_model_scope_file_info_keeps_main_file_first():
    # The first entry feeds resolved_path; the projector must never sort
    # ahead of the main weights, even when the user pattern also matches it.
    files = [
        FileEntry('mmproj-vllm-d3-mini-Q8_0.gguf', 366894720),
        FileEntry('vllm-d3-mini-Q8_0.gguf', 5157835264),
    ]

    got = select_model_scope_file_info(files, _ms_source('*Q8_0.gguf'))
    assert [f.rfilename for f in got] == [
        'vllm-d3-mini-Q8_0.gguf',
        'mmproj-vllm-d3-mini-Q8_0.gguf',
    ]


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


def test_modelscope_match_requires_main_file_before_mmproj(monkeypatch):
    class FakeHubApi:
        def get_model_files(self, *args, **kwargs):
            return [{"Path": "mmproj-F32.gguf"}]

    monkeypatch.setattr(hub, "HubApi", FakeHubApi)

    assert (
        match_model_scope_file_paths(
            model_id="test/model",
            file_path="missing.gguf",
            extra_file_path="*mmproj*.gguf",
        )
        == []
    )


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
