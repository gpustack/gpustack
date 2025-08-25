import unittest.mock as mock
from tenacity import retry, stop_after_attempt, wait_fixed
from gpustack.utils.hub import (
    get_hugging_face_model_min_gguf_path,
    get_model_scope_model_min_gguf_path,
    get_model_weight_size,
    list_repo,
    filter_model_files,
    match_model_scope_file_paths,
)
from gpustack.schemas.models import (
    Model,
    SourceEnum,
)
from gpustack.utils.hub import FilterPurpose


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


def test_list_repo_subdirectory_filtering():
    """Test list_repo with subdirectory filtering based on model environment variables."""
    # Mock data for testing - using correct repo path format
    mock_file_data = [
        {"name": "test/model/model.safetensors", "size": 1000},
        {"name": "test/model/config.json", "size": 100},
        {"name": "test/model/subdirectory/quantized.bin", "size": 500},
        {"name": "test/model/subdirectory/special.gguf", "size": 300},
    ]

    expected_root_only = [
        {"name": "model.safetensors", "size": 1000},
        {"name": "config.json", "size": 100},
    ]

    expected_with_subdirs = [
        {"name": "model.safetensors", "size": 1000},
        {"name": "config.json", "size": 100},
        {"name": "subdirectory/quantized.bin", "size": 500},
        {"name": "subdirectory/special.gguf", "size": 300},
    ]

    # Test HuggingFace source
    with mock.patch('gpustack.utils.hub.HfFileSystem') as mock_hffs:
        mock_fs_instance = mock.MagicMock()
        mock_hffs.return_value = mock_fs_instance
        mock_fs_instance.ls.return_value = mock_file_data

        with (
            mock.patch('gpustack.utils.hub.validate_repo_id'),
            mock.patch('gpustack.utils.hub.save_cache', return_value=True),
            mock.patch('gpustack.utils.hub.load_cache', return_value=(None, False)),
        ):

            # Test with model that excludes subdirectories
            model_exclude_subdirs = Model(
                source=SourceEnum.HUGGING_FACE,
                huggingface_repo_id="test/model",
                env={"GPUSTACK_MODEL_INCLUDE_SUBDIRS": "false"},
            )
            result = list_repo(
                "test/model", SourceEnum.HUGGING_FACE, model=model_exclude_subdirs
            )
            assert len(result) == 2
            assert result == expected_root_only

            # Test with model that includes subdirectories
            model_include_subdirs = Model(
                source=SourceEnum.HUGGING_FACE,
                huggingface_repo_id="test/model",
                env={"GPUSTACK_MODEL_INCLUDE_SUBDIRS": "true"},
            )
            result = list_repo(
                "test/model", SourceEnum.HUGGING_FACE, model=model_include_subdirs
            )
            assert len(result) == 4
            assert result == expected_with_subdirs

            # Test with no model (default behavior - without subdirectories)
            result = list_repo("test/model", SourceEnum.HUGGING_FACE, model=None)
            assert len(result) == 2
            assert result == expected_root_only

    # Test ModelScope source with same logic
    mock_ms_file_data_with_subdirs = [
        {"Path": "model.safetensors", "Size": 1000},
        {"Path": "config.json", "Size": 100},
        {"Path": "subdirectory/quantized.bin", "Size": 500},
        {"Path": "subdirectory/special.gguf", "Size": 300},
    ]

    mock_ms_file_data_root_only = [
        {"Path": "model.safetensors", "Size": 1000},
        {"Path": "config.json", "Size": 100},
    ]

    with mock.patch('gpustack.utils.hub.HubApi') as mock_hubapi:
        mock_api_instance = mock.MagicMock()
        mock_hubapi.return_value = mock_api_instance

        with (
            mock.patch('gpustack.utils.hub.save_cache', return_value=True),
            mock.patch('gpustack.utils.hub.load_cache', return_value=(None, False)),
        ):

            # Test ModelScope with subdirectory exclusion
            model_ms_exclude = Model(
                source=SourceEnum.MODEL_SCOPE,
                model_scope_model_id="test/model",
                env={"GPUSTACK_MODEL_INCLUDE_SUBDIRS": "false"},
            )
            # Mock API to return only root files when recursive=False
            mock_api_instance.get_model_files.return_value = mock_ms_file_data_root_only
            result = list_repo(
                "test/model", SourceEnum.MODEL_SCOPE, model=model_ms_exclude
            )
            assert len(result) == 2
            expected_ms_root = [
                {"name": "model.safetensors", "size": 1000},
                {"name": "config.json", "size": 100},
            ]
            assert result == expected_ms_root

            # Test ModelScope with subdirectory inclusion
            model_ms_include = Model(
                source=SourceEnum.MODEL_SCOPE,
                model_scope_model_id="test/model",
                env={"GPUSTACK_MODEL_INCLUDE_SUBDIRS": "true"},
            )
            # Mock API to return all files when recursive=True
            mock_api_instance.get_model_files.return_value = (
                mock_ms_file_data_with_subdirs
            )
            result = list_repo(
                "test/model", SourceEnum.MODEL_SCOPE, model=model_ms_include
            )
            assert len(result) == 4
            expected_ms_with_subdirs = [
                {"name": "model.safetensors", "size": 1000},
                {"name": "config.json", "size": 100},
                {"name": "subdirectory/quantized.bin", "size": 500},
                {"name": "subdirectory/special.gguf", "size": 300},
            ]
            assert result == expected_ms_with_subdirs


def test_filter_model_files_format_preference():
    """Test filter_model_files with different format preferences."""
    from gpustack.utils.hub import FileEntry

    # Mock file data with different formats
    mock_files = [
        FileEntry("model.safetensors", 1000),
        FileEntry("model.bin", 1000),
        FileEntry("model.gguf", 800),
        FileEntry("config.json", 50),
    ]

    # Test default preference (safetensors first)
    model_default = Model(
        source=SourceEnum.HUGGING_FACE, huggingface_repo_id="test/model"
    )
    result = filter_model_files(mock_files, model_default, FilterPurpose.EVALUATE)
    # Should select only safetensors for inference
    assert len(result) == 1
    assert result[0].rfilename == "model.safetensors"

    # Test custom preference via environment
    model_custom = Model(
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="test/model",
        env={"GPUSTACK_MODEL_FORMAT_PREFERENCE": "gguf,bin,safetensors"},
    )
    result = filter_model_files(mock_files, model_custom, FilterPurpose.EVALUATE)
    # Should select gguf first based on custom preference
    assert len(result) == 1
    assert result[0].rfilename == "model.gguf"


def test_filter_model_files_version_preference():
    """Test filter_model_files with different version preferences."""
    from gpustack.utils.hub import FileEntry

    # Mock files with consolidated and sharded versions
    mock_files = [
        FileEntry("model.safetensors", 2000),  # consolidated
        FileEntry("model-00001-of-00002.safetensors", 1000),  # sharded
        FileEntry("model-00002-of-00002.safetensors", 1000),  # sharded
    ]

    # Test consolidated preference (default)
    model_consolidated = Model(
        source=SourceEnum.HUGGING_FACE, huggingface_repo_id="test/model"
    )
    result = filter_model_files(mock_files, model_consolidated, FilterPurpose.EVALUATE)
    assert len(result) == 1
    assert result[0].rfilename == "model.safetensors"

    # Test sharded preference
    model_sharded = Model(
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="test/model",
        env={"GPUSTACK_MODEL_VERSION_PREFERENCE": "sharded"},
    )
    result = filter_model_files(mock_files, model_sharded, FilterPurpose.EVALUATE)
    assert len(result) == 2
    assert all(
        "-00001-of-00002" in f.rfilename or "-00002-of-00002" in f.rfilename
        for f in result
    )

    # Test both preference
    model_both = Model(
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="test/model",
        env={"GPUSTACK_MODEL_VERSION_PREFERENCE": "both"},
    )
    result = filter_model_files(mock_files, model_both, FilterPurpose.EVALUATE)
    assert len(result) == 3


def test_match_model_scope_file_paths_with_model_filtering():
    """Test match_model_scope_file_paths with model parameter for intelligent filtering."""
    # Mock ModelScope API response
    mock_ms_files = [
        {"Path": "model.safetensors", "Size": 2000},
        {"Path": "model.bin", "Size": 2000},
        {"Path": "model.gguf", "Size": 1500},
        {"Path": "config.json", "Size": 100},
        {"Path": "tokenizer.json", "Size": 200},
        {"Path": "model-00001-of-00002.safetensors", "Size": 1000},
        {"Path": "model-00002-of-00002.safetensors", "Size": 1000},
    ]

    with mock.patch('gpustack.utils.hub.HubApi') as mock_hubapi:
        mock_api_instance = mock.MagicMock()
        mock_hubapi.return_value = mock_api_instance
        mock_api_instance.get_model_files.return_value = mock_ms_files

        # Test with model parameter - should use filter_model_files for intelligent selection
        model = Model(
            source=SourceEnum.MODEL_SCOPE,
            model_scope_model_id="test/model",
        )

        # Test matching all model files with intelligent filtering
        result = match_model_scope_file_paths(
            "test/model", "*.safetensors", model=model
        )

        # Should prefer consolidated over sharded files by default
        assert len(result) == 1
        assert "model.safetensors" in result[0]

        # Test with custom format preference - match all formats then filter
        model_gguf_pref = Model(
            source=SourceEnum.MODEL_SCOPE,
            model_scope_model_id="test/model",
            env={"GPUSTACK_MODEL_FORMAT_PREFERENCE": "gguf,safetensors,bin"},
        )

        result = match_model_scope_file_paths(
            "test/model", "model.gguf", model=model_gguf_pref
        )

        # Should select GGUF format when specifically matched
        assert len(result) == 1
        assert "model.gguf" in result[0]

        # Test intelligent filtering with wildcard pattern
        result_wildcard = match_model_scope_file_paths(
            "test/model", "model.*", model=model_gguf_pref
        )

        # Should prefer GGUF format based on custom preference when using wildcard
        assert len(result_wildcard) == 1
        assert "model.gguf" in result_wildcard[0]

        # Test with sharded preference
        model_sharded_pref = Model(
            source=SourceEnum.MODEL_SCOPE,
            model_scope_model_id="test/model",
            env={"GPUSTACK_MODEL_VERSION_PREFERENCE": "sharded"},
        )

        result = match_model_scope_file_paths(
            "test/model", "*.safetensors", model=model_sharded_pref
        )

        # Should prefer sharded files
        assert len(result) == 2
        assert all("-00001-of-00002" in f or "-00002-of-00002" in f for f in result)

        # Test without model parameter - should use basic fnmatch filtering
        result_no_model = match_model_scope_file_paths("test/model", "*.safetensors")

        # Should return all matching files without intelligent filtering
        assert len(result_no_model) == 3
        expected_files = [
            "model.safetensors",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ]
        for expected in expected_files:
            assert any(expected in f for f in result_no_model)


def test_match_model_scope_file_paths_extra_file_selection():
    """Test match_model_scope_file_paths with extra file selection using model parameter."""
    mock_ms_files = [
        {"Path": "model.safetensors", "Size": 2000},
        {"Path": "config.json", "Size": 100},
        {"Path": "tokenizer.json", "Size": 200},
        {"Path": "tokenizer_config.json", "Size": 150},
        {"Path": "special_tokens_map.json", "Size": 50},
        {"Path": "generation_config.json", "Size": 80},
    ]

    with mock.patch('gpustack.utils.hub.HubApi') as mock_hubapi:
        mock_api_instance = mock.MagicMock()
        mock_hubapi.return_value = mock_api_instance
        mock_api_instance.get_model_files.return_value = mock_ms_files

        model = Model(
            source=SourceEnum.MODEL_SCOPE,
            model_scope_model_id="test/model",
        )

        # Test with extra file path - should intelligently select the most suitable extra file
        result = match_model_scope_file_paths(
            "test/model", "*.safetensors", "*.json", model
        )

        # Should include the main model file and the most suitable extra file
        assert len(result) >= 2
        assert any("model.safetensors" in f for f in result)

        # Should include config.json as it's typically the most important extra file
        assert any("config.json" in f for f in result)
