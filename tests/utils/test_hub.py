import unittest.mock as mock
from tenacity import retry, stop_after_attempt, wait_fixed
from gpustack.utils.hub import (
    get_hugging_face_model_min_gguf_path,
    get_model_scope_model_min_gguf_path,
    get_model_weight_size,
    list_repo,
    ModelFileFilter,
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
    from gpustack.utils.hub import FileEntry

    # Mock data for testing - using correct repo path format
    mock_file_data = [
        {"name": "test/model/model.safetensors", "size": 1000},
        {"name": "test/model/config.json", "size": 100},
        {"name": "test/model/subdirectory/quantized.bin", "size": 500},
        {"name": "test/model/subdirectory/special.gguf", "size": 300},
    ]

    expected_root_only = [
        FileEntry("model.safetensors", 1000),
        FileEntry("config.json", 100),
    ]

    expected_with_subdirs = [
        FileEntry("model.safetensors", 1000),
        FileEntry("config.json", 100),
        FileEntry("subdirectory/quantized.bin", 500),
        FileEntry("subdirectory/special.gguf", 300),
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
            # Test excluding subdirectories
            result = list_repo(
                "test/model", SourceEnum.HUGGING_FACE, include_subdirs=False
            )
            assert len(result) == 2
            assert result == expected_root_only

            # Test including subdirectories
            result = list_repo(
                "test/model", SourceEnum.HUGGING_FACE, include_subdirs=True
            )
            assert len(result) == 4
            assert result == expected_with_subdirs

            # Test default behavior (includes subdirectories)
            result = list_repo("test/model", SourceEnum.HUGGING_FACE)
            assert len(result) == 4
            assert result == expected_with_subdirs

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
            # Mock API to return only root files when recursive=False
            mock_api_instance.get_model_files.return_value = mock_ms_file_data_root_only
            result = list_repo(
                "test/model", SourceEnum.MODEL_SCOPE, include_subdirs=False
            )
            assert len(result) == 2
            expected_ms_root = [
                FileEntry("model.safetensors", 1000),
                FileEntry("config.json", 100),
            ]
            assert result == expected_ms_root

            # Test ModelScope with subdirectory inclusion
            # Mock API to return all files when recursive=True
            mock_api_instance.get_model_files.return_value = (
                mock_ms_file_data_with_subdirs
            )
            result = list_repo(
                "test/model", SourceEnum.MODEL_SCOPE, include_subdirs=True
            )
            assert len(result) == 4
            expected_ms_with_subdirs = [
                FileEntry("model.safetensors", 1000),
                FileEntry("config.json", 100),
                FileEntry("subdirectory/quantized.bin", 500),
                FileEntry("subdirectory/special.gguf", 300),
            ]
            assert result == expected_ms_with_subdirs


def test_model_file_filter_format_preference():
    """Test ModelFileFilter with different format preferences."""
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
    filter_instance = ModelFileFilter(model_default, FilterPurpose.EVALUATE)
    result = filter_instance.filter_files(mock_files)
    # Should select only safetensors for inference
    assert len(result) == 1
    assert result[0].rfilename == "model.safetensors"

    # Test custom preference via environment
    model_custom = Model(
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="test/model",
        env={"GPUSTACK_MODEL_FORMAT_PREF": "gguf,bin,safetensors"},
    )
    filter_custom = ModelFileFilter(model_custom, FilterPurpose.EVALUATE)
    result = filter_custom.filter_files(mock_files)
    # Should select gguf first based on custom preference
    assert len(result) == 1
    assert result[0].rfilename == "model.gguf"


def test_model_file_filter_sharding_detection():
    """Test ModelFileFilter with automatic sharding detection via .index.json files."""
    from gpustack.utils.hub import FileEntry

    # Test consolidated model (no index file)
    consolidated_files = [
        FileEntry("model.safetensors", 2000),  # consolidated
        FileEntry("config.json", 100),
    ]

    model = Model(source=SourceEnum.HUGGING_FACE, huggingface_repo_id="test/model")
    filter_instance = ModelFileFilter(model, FilterPurpose.EVALUATE)
    result = filter_instance.filter_files(consolidated_files)
    assert len(result) == 1
    assert result[0].rfilename == "model.safetensors"

    # Test sharded model (with index file)
    sharded_files = [
        FileEntry("model.safetensors.index.json", 10),  # index file indicates sharding
        FileEntry("model-00001-of-00002.safetensors", 1000),  # sharded
        FileEntry("model-00002-of-00002.safetensors", 1000),  # sharded
        FileEntry("config.json", 100),
    ]

    filter_instance = ModelFileFilter(model, FilterPurpose.EVALUATE)
    result = filter_instance.filter_files(sharded_files)
    # Should detect sharding and select shard files (fallback to all safetensors files)
    safetensors_files = [f for f in result if f.rfilename.endswith('.safetensors')]
    assert (
        len(safetensors_files) >= 1
    )  # At least some safetensors files should be selected

    # Test download purpose - should include index file
    filter_download = ModelFileFilter(model, FilterPurpose.DOWNLOAD)
    result_download = filter_download.filter_files(sharded_files)
    index_files = [f for f in result_download if f.rfilename.endswith('.index.json')]
    assert len(index_files) >= 0  # May include index file for download


def test_sharded_files_from_index():
    """Test ModelFileFilter._get_sharded_files_from_index method with various scenarios."""
    from gpustack.utils.hub import ModelFileFilter, FileEntry

    # Create a dummy model for testing
    model = Model(source=SourceEnum.HUGGING_FACE, huggingface_repo_id="test/model")
    filter_instance = ModelFileFilter(model)

    # Test safetensors files with index
    safetensors_files = [
        FileEntry("model.safetensors.index.json", 1000),
        FileEntry("model-00001-of-00002.safetensors", 5000),
        FileEntry("model-00002-of-00002.safetensors", 5000),
        FileEntry("config.json", 500),
    ]

    sharded_files = filter_instance._get_sharded_files_from_index(
        safetensors_files, 'safetensors'
    )
    assert len(sharded_files) == 2
    assert all(f.rfilename.endswith('.safetensors') for f in sharded_files)
    assert all(not f.rfilename.endswith('.index.json') for f in sharded_files)

    # Test pytorch files with index
    pytorch_files = [
        FileEntry("pytorch_model.bin.index.json", 800),
        FileEntry("pytorch_model-00001-of-00003.bin", 3000),
        FileEntry("pytorch_model-00002-of-00003.bin", 3000),
        FileEntry("pytorch_model-00003-of-00003.bin", 3000),
        FileEntry("config.json", 500),
    ]

    sharded_files = filter_instance._get_sharded_files_from_index(pytorch_files, 'bin')
    assert len(sharded_files) == 3
    assert all(f.rfilename.endswith('.bin') for f in sharded_files)
    assert all(not f.rfilename.endswith('.index.json') for f in sharded_files)

    # Test mixed files (should only return the specified format)
    mixed_files = [
        FileEntry("model.safetensors.index.json", 1000),
        FileEntry("model-00001-of-00002.safetensors", 5000),
        FileEntry("model-00002-of-00002.safetensors", 5000),
        FileEntry("pytorch_model.bin", 4000),
        FileEntry("config.json", 500),
    ]

    safetensors_only = filter_instance._get_sharded_files_from_index(
        mixed_files, 'safetensors'
    )
    assert len(safetensors_only) == 2
    assert all('.safetensors' in f.rfilename for f in safetensors_only)


def test_model_file_filter_class():
    """Test the ModelFileFilter class directly."""
    from gpustack.utils.hub import ModelFileFilter, FileEntry, FilterPurpose

    # Test files
    test_files = [
        FileEntry("model.safetensors", 2000),
        FileEntry("pytorch_model.bin", 2000),
        FileEntry("config.json", 100),
        FileEntry("subdir/model.gguf", 1000),
    ]

    # Test with default preferences
    model = Model(source=SourceEnum.HUGGING_FACE, huggingface_repo_id="test/model")
    filter_instance = ModelFileFilter(model, FilterPurpose.EVALUATE)

    # Test properties
    assert filter_instance.model == model
    assert filter_instance.purpose == FilterPurpose.EVALUATE
    assert filter_instance.format_preference == [
        'safetensors',
        'bin',
        'pt',
        'pth',
        'gguf',
    ]
    assert filter_instance.subdir_preference is False
    assert filter_instance.repo_id == "test/model"
    assert filter_instance.source == SourceEnum.HUGGING_FACE

    # Test filtering
    result = filter_instance.filter_files(test_files)
    assert len(result) == 1
    assert result[0].rfilename == "model.safetensors"

    # Test with custom preferences
    model_custom = Model(
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="test/model",
        env={
            "GPUSTACK_MODEL_FORMAT_PREF": "gguf,bin,safetensors",
            "GPUSTACK_MODEL_INCLUDE_SUBDIRS": "true",
        },
    )
    filter_custom = ModelFileFilter(model_custom, FilterPurpose.EVALUATE)

    # Test custom properties
    assert filter_custom.format_preference == ['gguf', 'bin', 'safetensors']
    assert filter_custom.subdir_preference is True

    # Test filtering with custom preferences
    result_custom = filter_custom.filter_files(test_files)
    assert len(result_custom) == 1
    assert result_custom[0].rfilename == "subdir/model.gguf"


def test_model_file_filter_download_purpose():
    """Test ModelFileFilter with download purpose."""
    from gpustack.utils.hub import ModelFileFilter, FileEntry, FilterPurpose

    test_files = [
        FileEntry("model.safetensors", 2000),
        FileEntry("pytorch_model.bin", 2000),
        FileEntry("config.json", 100),
        FileEntry("subdir/model.gguf", 1000),
    ]

    model = Model(source=SourceEnum.HUGGING_FACE, huggingface_repo_id="test/model")
    filter_instance = ModelFileFilter(model, FilterPurpose.DOWNLOAD)

    # Download should include all formats and subdirectories
    result = filter_instance.filter_files(test_files)
    assert len(result) >= 2  # Should include multiple formats

    # Should include files from subdirectories for download
    subdir_files = [f for f in result if '/' in f.rfilename]
    assert len(subdir_files) >= 1


def test_model_file_filter_utility_functions():
    """Test utility functions and ModelFileFilter methods."""
    from gpustack.utils.hub import ModelFileFilter, FileEntry

    model = Model(
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="test/model",
        env={"GPUSTACK_MODEL_FORMAT_PREF": "bin,safetensors"},
    )

    filter_instance = ModelFileFilter(model)

    # Test ModelFileFilter methods
    assert filter_instance._get_format_preference() == ['bin', 'safetensors']
    assert filter_instance._get_subdir_preference() is False

    # Test utility function
    assert (
        filter_instance._get_index_filename('safetensors')
        == 'model.safetensors.index.json'
    )

    test_files = [
        FileEntry("model.safetensors", 2000),
        FileEntry("pytorch_model.bin", 2000),
        FileEntry("config.json", 100),
    ]

    categorized = filter_instance._categorize_files_by_format(test_files)
    assert 'safetensors' in categorized
    assert 'bin' in categorized
    assert len(categorized['safetensors']) == 1
    assert len(categorized['bin']) == 1

    filter_instance.format_preference = ['safetensors']
    consolidated = filter_instance._select_files_with_index_detection(categorized)
    assert len(consolidated) == 1
    assert consolidated[0].rfilename == "model.safetensors"
