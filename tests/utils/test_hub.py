import json
import struct
import pytest
from tenacity import retry, stop_after_attempt, wait_fixed
from gpustack.utils import hub
from gpustack.utils.hub import (
    get_hugging_face_model_min_gguf_path,
    get_model_scope_model_min_gguf_path,
    get_model_weight_size,
    match_hugging_face_files,
    match_model_scope_file_paths,
    read_repo_file_content,
    read_safetensors_header_keys,
    detect_voice_cloning_support,
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


def _safetensors_bytes(keys) -> bytes:
    header = {
        key: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]} for key in keys
    }
    header["__metadata__"] = {"format": "pt"}
    payload = json.dumps(header).encode()
    return struct.pack("<Q", len(payload)) + payload


def test_read_safetensors_header_keys(tmp_path):
    (tmp_path / "consolidated.safetensors").write_bytes(
        _safetensors_bytes(["audio_tokenizer.decoder_blocks.0.w", "lm_head.weight"])
    )
    # A corrupt header claiming to be gigabytes long must not be read into memory.
    (tmp_path / "oversized.safetensors").write_bytes(
        struct.pack("<Q", hub._MAX_SAFETENSORS_HEADER_BYTES + 1)
    )
    # Valid JSON that is not an object: iterating it would yield non-string keys
    # that crash the caller's .startswith() outside its try block.
    array_header = b'[1, 2]'
    (tmp_path / "array.safetensors").write_bytes(
        struct.pack("<Q", len(array_header)) + array_header
    )

    assert read_safetensors_header_keys(str(tmp_path), "consolidated.safetensors") == {
        "audio_tokenizer.decoder_blocks.0.w",
        "lm_head.weight",
    }
    assert read_safetensors_header_keys(str(tmp_path), "oversized.safetensors") is None
    assert read_safetensors_header_keys(str(tmp_path), "array.safetensors") is None
    assert read_safetensors_header_keys(str(tmp_path), "missing.safetensors") is None


@pytest.mark.parametrize(
    "keys, expected",
    [
        # Voxtral-TTS family with encoder weights -> voice cloning supported.
        (
            {"audio_tokenizer.decoder_blocks.0.w", "audio_tokenizer.input_proj.weight"},
            True,
        ),
        # Voxtral-TTS family without encoder weights (open checkpoint) -> unsupported.
        ({"audio_tokenizer.decoder_blocks.0.w", "audio_tokenizer.quantizer.x"}, False),
        # Not a Voxtral-TTS-family checkpoint -> no claim.
        ({"model.layers.0.mlp.weight"}, None),
        # Manifest unreadable -> no claim.
        (None, None),
    ],
)
def test_detect_voice_cloning_support(monkeypatch, keys, expected):
    monkeypatch.setattr(hub, "_load_weight_keys", lambda model, **_: keys)
    m = new_model(
        1,
        "tts",
        huggingface_repo_id="mistralai/Voxtral-4B-TTS-2603",
        categories=["text_to_speech"],
    )

    assert detect_voice_cloning_support(m) is expected


def test_detect_voice_cloning_support_prefers_the_sharded_index(tmp_path):
    # The index lists every weight, so the shard headers are never opened - the
    # unreadable shard would otherwise force an inconclusive verdict.
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"audio_tokenizer.input_proj.w": "shard"}})
    )
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"garbage")
    m = Model(source=SourceEnum.LOCAL_PATH, local_path=str(tmp_path))

    assert detect_voice_cloning_support(m) is True


@pytest.mark.parametrize(
    "keys, expected",
    [
        # Local Voxtral-TTS checkpoint missing the encoder weights -> unsupported.
        (["audio_tokenizer.decoder_blocks.0.w"], False),
        # Local checkpoint shipping the encoder weights -> supported.
        (["audio_tokenizer.decoder_blocks.0.w", "audio_tokenizer.input_proj.w"], True),
        # Not a Voxtral-TTS-family checkpoint -> no claim.
        (["model.layers.0.mlp.weight"], None),
    ],
)
def test_detect_voice_cloning_support_local_path(tmp_path, keys, expected):
    # The worker running the instance holds these files, so the header is read
    # straight off disk — no network, no token.
    (tmp_path / "consolidated.safetensors").write_bytes(_safetensors_bytes(keys))
    (tmp_path / "params.json").write_text("{}")
    m = Model(source=SourceEnum.LOCAL_PATH, local_path=str(tmp_path))

    assert detect_voice_cloning_support(m) is expected


def test_detect_voice_cloning_support_reads_local_dir_for_any_source(
    tmp_path, monkeypatch
):
    # Detection never goes remote: a ModelScope checkpoint - which has no cheap
    # range read - is read from the copy the worker downloaded, addressed by the
    # instance's resolved_path. Without a local copy there is no fallback.
    (tmp_path / "consolidated.safetensors").write_bytes(
        _safetensors_bytes(["audio_tokenizer.decoder_blocks.0.w"])
    )
    monkeypatch.setattr(
        hub, "list_repo", lambda *a, **k: pytest.fail("must not hit the remote")
    )
    m = new_model(1, "tts", model_scope_model_id="org/repo")

    assert detect_voice_cloning_support(m, local_dir=str(tmp_path)) is False
    assert detect_voice_cloning_support(m, local_dir=str(tmp_path / "nope")) is None


def test_detect_voice_cloning_support_unions_shards_without_index(tmp_path):
    # Mistral-native sharded layout ships no index.json. Reading only the first
    # shard would miss the encoder weights and wrongly reject a model that does
    # support voice cloning.
    (tmp_path / "consolidated-00001-of-00002.safetensors").write_bytes(
        _safetensors_bytes(["audio_tokenizer.decoder_blocks.0.w"])
    )
    (tmp_path / "consolidated-00002-of-00002.safetensors").write_bytes(
        _safetensors_bytes(["audio_tokenizer.input_proj.weight"])
    )
    m = Model(source=SourceEnum.LOCAL_PATH, local_path=str(tmp_path))

    assert detect_voice_cloning_support(m) is True


def test_detect_voice_cloning_support_unreadable_shard_is_inconclusive(tmp_path):
    # A shard we cannot parse may be the one holding the encoder weights, so the
    # verdict must be "unknown" rather than a partial-manifest False.
    (tmp_path / "consolidated-00001-of-00002.safetensors").write_bytes(
        _safetensors_bytes(["audio_tokenizer.decoder_blocks.0.w"])
    )
    (tmp_path / "consolidated-00002-of-00002.safetensors").write_bytes(b"garbage")

    assert (
        detect_voice_cloning_support(
            Model(source=SourceEnum.LOCAL_PATH, local_path=str(tmp_path))
        )
        is None
    )


def test_detect_voice_cloning_support_local_path_unreadable(tmp_path):
    missing = Model(source=SourceEnum.LOCAL_PATH, local_path=str(tmp_path / "nope"))
    no_weights = Model(source=SourceEnum.LOCAL_PATH, local_path=str(tmp_path))

    # A missing directory or one without safetensors makes no claim.
    assert detect_voice_cloning_support(missing) is None
    assert detect_voice_cloning_support(no_weights) is None
