from unittest.mock import patch, AsyncMock, MagicMock

import pytest

from gpustack.policies.worker_filters.backend_framework_filter import (
    BackendFrameworkFilter,
)
from gpustack.schemas.models import Model, BackendEnum
from gpustack.schemas.inference_backend import InferenceBackend, VersionConfig
from tests.fixtures.workers.fixtures import (
    linux_nvidia_4_4080_16gx4,
    linux_cpu_1,
    linux_ascend_1_910b_64gx8,
    linux_nvidia_3_4090_24gx1,
)


def create_model(backend="vLLM", backend_version=None, **kwargs) -> Model:
    """Create a test model with specified backend configuration."""
    return Model(
        id=1,
        name="test-model",
        replicas=1,
        ready_replicas=0,
        source="huggingface",
        huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct",
        backend=backend,
        backend_version=backend_version,
        **kwargs,
    )


def create_inference_backend(
    backend_name="vLLM",
    version_configs=None,
    is_built_in=False,
):
    """Create a test inference backend."""
    if version_configs is None:
        version_configs = {
            "0.11.0": VersionConfig(
                image_name="test:0.11.0",
                built_in_frameworks=["cuda", "rocm"],
                custom_framework="",
            )
        }

    backend = InferenceBackend(
        id=1,
        backend_name=backend_name,
        version_configs=version_configs,
        default_version="0.11.0",
        is_built_in=is_built_in,
    )
    # Simulate database deserialization - version_configs should be a VersionConfigDict with root attribute
    from gpustack.schemas.inference_backend import VersionConfigDict

    backend.version_configs = VersionConfigDict(root=version_configs)
    return backend


@pytest.mark.asyncio
async def test_cuda_gpu_worker_passes():
    """
    Test 1: Worker with CUDA GPU should pass the filter (basic functionality).
    """
    model = create_model(backend="vLLM", backend_version=None)
    workers = [linux_nvidia_4_4080_16gx4()]

    backend = create_inference_backend(
        backend_name="vLLM",
        version_configs={
            "0.11.2": VersionConfig(
                image_name="test:0.11.2",
                built_in_frameworks=["cuda"],
                custom_framework="",
            )
        },
        is_built_in=True,
    )

    filter_instance = BackendFrameworkFilter(model)

    # Mock only the database query, not _has_supported_runners
    async def mock_session_exec(statement):
        mock_result = MagicMock()
        mock_result.first.return_value = backend
        return mock_result

    with patch(
        'gpustack.policies.worker_filters.backend_framework_filter.async_session'
    ) as mock_async_session:
        # Create a mock session that returns the backend
        mock_session = AsyncMock()
        mock_session.exec = mock_session_exec
        mock_async_session.return_value.__aenter__.return_value = mock_session

        filtered_workers, messages = await filter_instance.filter(workers)

        assert len(filtered_workers) == 1
        assert filtered_workers[0].name == "host-4-4080"
        assert len(messages) == 0


@pytest.mark.asyncio
async def test_cuda_gpu_worker_with_specified_version_passes():
    """
    Test 2: Worker with CUDA GPU should pass when backend version is specified.
    """
    model = create_model(backend="vLLM", backend_version="0.11.0")
    workers = [linux_nvidia_4_4080_16gx4()]

    backend = create_inference_backend(
        backend_name="vLLM",
        version_configs={
            "0.11.0": VersionConfig(
                image_name="test:0.11.0",
                built_in_frameworks=["cuda"],
                custom_framework="",
            )
        },
        is_built_in=True,
    )

    filter_instance = BackendFrameworkFilter(model)

    async def mock_session_exec(statement):
        mock_result = MagicMock()
        mock_result.first.return_value = backend
        return mock_result

    with patch(
        'gpustack.policies.worker_filters.backend_framework_filter.async_session'
    ) as mock_async_session:
        mock_session = AsyncMock()
        mock_session.exec = mock_session_exec
        mock_async_session.return_value.__aenter__.return_value = mock_session

        filtered_workers, messages = await filter_instance.filter(workers)

        assert len(filtered_workers) == 1
        assert filtered_workers[0].name == "host-4-4080"
        assert len(messages) == 0


@pytest.mark.asyncio
async def test_cpu_only_worker_filtered_out():
    """
    Test 3: Worker with only CPU should be filtered out when CPU is not in supported frameworks.
    """
    model = create_model(backend="vLLM", backend_version=None)
    workers = [linux_cpu_1()]

    backend = create_inference_backend(
        backend_name="vLLM",
        version_configs={
            "0.11.0": VersionConfig(
                image_name="test:0.11.0",
                built_in_frameworks=["cuda"],  # Only CUDA is supported, not CPU
                custom_framework="",
            )
        },
        is_built_in=True,
    )

    filter_instance = BackendFrameworkFilter(model)

    async def mock_session_exec(statement):
        mock_result = MagicMock()
        mock_result.first.return_value = backend
        return mock_result

    with patch(
        'gpustack.policies.worker_filters.backend_framework_filter.async_session'
    ) as mock_async_session:
        mock_session = AsyncMock()
        mock_session.exec = mock_session_exec
        mock_async_session.return_value.__aenter__.return_value = mock_session

        with patch(
            'gpustack.policies.worker_filters.backend_framework_filter.list_service_runners',
            return_value=[],
        ):
            filtered_workers, messages = await filter_instance.filter(workers)

            # CPU worker should be filtered out since CPU is not in supported frameworks
            assert len(filtered_workers) == 0
            assert len(messages) == 1
            assert "host-cpu-1" in messages[0]
            assert "filtered out" in messages[0]


@pytest.mark.asyncio
async def test_custom_backend_with_cpu_support():
    """
    Test 4: Custom backend with CPU framework support allows CPU worker to pass.
    """
    model = create_model(backend="vLLM", backend_version="1.0.0")
    workers = [linux_cpu_1()]

    backend = create_inference_backend(
        backend_name="vLLM",
        version_configs={
            "1.0.0": VersionConfig(
                image_name="test:1.0.0",
                built_in_frameworks=None,
                custom_framework="cpu",
            )
        },
        is_built_in=False,
    )

    filter_instance = BackendFrameworkFilter(model)

    async def mock_session_exec(statement):
        mock_result = MagicMock()
        mock_result.first.return_value = backend
        mock_result.all.return_value = [backend]
        return mock_result

    with patch(
        'gpustack.policies.worker_filters.backend_framework_filter.async_session'
    ) as mock_async_session:
        mock_session = AsyncMock()
        mock_session.exec = mock_session_exec
        mock_async_session.return_value.__aenter__.return_value = mock_session

        with patch(
            'gpustack.policies.worker_filters.backend_framework_filter.list_service_runners',
            return_value=[],
        ):
            filtered_workers, messages = await filter_instance.filter(workers)

            assert len(filtered_workers) == 1
            assert filtered_workers[0].name == "host-cpu-1"
            assert len(messages) == 0


@pytest.mark.asyncio
async def test_mixed_gpu_types():
    """
    Test 5: Worker with mixed GPU types should pass if at least one type is supported.
    """
    model = create_model(backend="vLLM", backend_version=None)

    # Create a worker with mixed GPU types (simulate by modifying the worker)
    worker = linux_nvidia_4_4080_16gx4()
    # Modify first GPU to simulate ROCm
    if worker.status and worker.status.gpu_devices:
        worker.status.gpu_devices[0].type = "rocm"
        worker.status.gpu_devices[0].vendor = "amd"

    workers = [worker]

    backend = create_inference_backend(
        backend_name="vLLM",
        version_configs={
            "0.11.0": VersionConfig(
                image_name="test:0.11.0",
                built_in_frameworks=["cuda", "rocm"],
                custom_framework="",
            )
        },
        is_built_in=True,
    )

    filter_instance = BackendFrameworkFilter(model)

    async def mock_session_exec(statement):
        mock_result = MagicMock()
        mock_result.first.return_value = backend
        return mock_result

    with patch(
        'gpustack.policies.worker_filters.backend_framework_filter.async_session'
    ) as mock_async_session:
        mock_session = AsyncMock()
        mock_session.exec = mock_session_exec
        mock_async_session.return_value.__aenter__.return_value = mock_session

        filtered_workers, messages = await filter_instance.filter(workers)

        assert len(filtered_workers) == 1
        assert len(messages) == 0


@pytest.mark.asyncio
async def test_version_mismatch():
    """
    Test 6: Worker with CUDA GPU should be filtered out when version doesn't match.
    """
    model = create_model(backend="vLLM", backend_version="11.8")
    workers = [linux_nvidia_4_4080_16gx4()]

    backend = create_inference_backend(
        backend_name="vLLM",
        version_configs={
            "12.4": VersionConfig(
                image_name="test:12.4",
                built_in_frameworks=["cuda"],
                custom_framework="",
            )
        },
        is_built_in=True,
    )

    filter_instance = BackendFrameworkFilter(model)

    async def mock_session_exec(statement):
        mock_result = MagicMock()
        mock_result.first.return_value = backend
        return mock_result

    with patch(
        'gpustack.policies.worker_filters.backend_framework_filter.async_session'
    ) as mock_async_session:
        mock_session = AsyncMock()
        mock_session.exec = mock_session_exec
        mock_async_session.return_value.__aenter__.return_value = mock_session

        with patch(
            'gpustack.policies.worker_filters.backend_framework_filter.list_service_runners',
            return_value=[],
        ):
            filtered_workers, messages = await filter_instance.filter(workers)

            assert len(filtered_workers) == 0
            assert len(messages) == 1
            assert "host-4-4080" in messages[0]
            assert "11.8" in messages[0] or "backend version" in messages[0]


@pytest.mark.asyncio
async def test_ascend_gpu_variant_handling():
    """
    Test 7: Worker with Ascend GPU should handle variant (CANN version) correctly.
    """
    model = create_model(backend="MindIE", backend_version=None)
    workers = [linux_ascend_1_910b_64gx8()]

    backend = create_inference_backend(
        backend_name="MindIE",
        version_configs={
            "1.0.0": VersionConfig(
                image_name="test:1.0.0",
                built_in_frameworks=["cann"],
                custom_framework="",
            )
        },
        is_built_in=True,
    )

    filter_instance = BackendFrameworkFilter(model)

    async def mock_session_exec(statement):
        mock_result = MagicMock()
        mock_result.first.return_value = backend
        return mock_result

    with patch(
        'gpustack.policies.worker_filters.backend_framework_filter.async_session'
    ) as mock_async_session:
        mock_session = AsyncMock()
        mock_session.exec = mock_session_exec
        mock_async_session.return_value.__aenter__.return_value = mock_session

        filtered_workers, messages = await filter_instance.filter(workers)

        assert len(filtered_workers) == 1
        assert filtered_workers[0].name == "ascend_0"
        assert len(messages) == 0


@pytest.mark.asyncio
async def test_empty_gpu_list_treated_as_cpu():
    """
    Test 8: Worker with empty GPU list should be treated as CPU worker - should be filtered out when CPU is not in supported frameworks.
    """
    model = create_model(backend="vLLM", backend_version=None)

    # Create a worker with empty GPU list
    worker = linux_cpu_1()
    worker.status.gpu_devices = []

    workers = [worker]

    backend = create_inference_backend(
        backend_name="vLLM",
        version_configs={
            "0.11.0": VersionConfig(
                image_name="test:0.11.0",
                built_in_frameworks=["cuda"],  # Only CUDA is supported, not CPU
                custom_framework="",
            )
        },
        is_built_in=True,
    )

    filter_instance = BackendFrameworkFilter(model)

    async def mock_session_exec(statement):
        mock_result = MagicMock()
        mock_result.first.return_value = backend
        return mock_result

    with patch(
        'gpustack.policies.worker_filters.backend_framework_filter.async_session'
    ) as mock_async_session:
        mock_session = AsyncMock()
        mock_session.exec = mock_session_exec
        mock_async_session.return_value.__aenter__.return_value = mock_session

        with patch(
            'gpustack.policies.worker_filters.backend_framework_filter.list_service_runners',
            return_value=[],
        ):
            filtered_workers, messages = await filter_instance.filter(workers)

            # CPU worker should be filtered out since CPU is not in supported frameworks
            assert len(filtered_workers) == 0
            assert len(messages) == 1


@pytest.mark.asyncio
async def test_custom_backend_skip_filter():
    """
    Test 9: Custom backend should skip framework compatibility filter.
    """
    model = create_model(backend=BackendEnum.CUSTOM, backend_version=None)
    workers = [linux_cpu_1(), linux_nvidia_4_4080_16gx4()]

    filter_instance = BackendFrameworkFilter(model)

    # No mocking needed for custom backend
    filtered_workers, messages = await filter_instance.filter(workers)

    # All workers should pass
    assert len(filtered_workers) == 2
    assert len(messages) == 0


@pytest.mark.asyncio
async def test_multiple_workers_filtering():
    """
    Test 10: Multiple workers should be filtered correctly - only CUDA worker passes when CPU is not in supported frameworks.
    """
    model = create_model(backend="vLLM", backend_version="0.11.0")
    workers = [
        linux_nvidia_4_4080_16gx4(),
        linux_cpu_1(),
        linux_nvidia_3_4090_24gx1(),
    ]

    backend = create_inference_backend(
        backend_name="vLLM",
        version_configs={
            "0.11.0": VersionConfig(
                image_name="test:0.11.0",
                built_in_frameworks=["cuda"],  # Only CUDA is supported
                custom_framework="",
            )
        },
        is_built_in=True,
    )

    filter_instance = BackendFrameworkFilter(model)

    async def mock_session_exec(statement):
        mock_result = MagicMock()
        mock_result.first.return_value = backend
        return mock_result

    with patch(
        'gpustack.policies.worker_filters.backend_framework_filter.async_session'
    ) as mock_async_session:
        mock_session = AsyncMock()
        mock_session.exec = mock_session_exec
        mock_async_session.return_value.__aenter__.return_value = mock_session

        filtered_workers, messages = await filter_instance.filter(workers)

        # Only CUDA worker should pass
        assert len(filtered_workers) == 2
        assert filtered_workers[0].name == "host-4-4080"
        # CPU and Ascend workers should be filtered out
        assert len(messages) == 1


@pytest.mark.asyncio
async def test_no_backend_name_skips_filter():
    """
    Test 11: When backend name cannot be determined, filter should skip.
    """
    # Create a model that will result in backend_name being None
    # We need to bypass get_backend's default behavior by directly setting backend_name
    model = create_model(backend=None, backend_version=None)

    # Create filter instance and manually set backend_name to None to test the skip logic
    filter_instance = BackendFrameworkFilter(model)
    filter_instance.backend_name = None

    workers = [linux_nvidia_4_4080_16gx4(), linux_cpu_1()]

    # No mocking needed - filter should skip
    filtered_workers, messages = await filter_instance.filter(workers)

    # All workers should pass (filter skipped)
    assert len(filtered_workers) == 2
    assert len(messages) == 0


@pytest.mark.asyncio
async def test_auto_matching_mode():
    """
    Test 12: Auto matching mode (no backend_version specified) should check for any available version.
    """
    model = create_model(backend="vLLM", backend_version=None)
    workers = [linux_nvidia_4_4080_16gx4()]

    backend = create_inference_backend(
        backend_name="vLLM",
        version_configs={
            "0.11.0": VersionConfig(
                image_name="test:0.11.0",
                built_in_frameworks=["cuda"],
                custom_framework="",
            )
        },
        is_built_in=True,
    )

    filter_instance = BackendFrameworkFilter(model)

    async def mock_session_exec(statement):
        mock_result = MagicMock()
        mock_result.first.return_value = backend
        return mock_result

    with patch(
        'gpustack.policies.worker_filters.backend_framework_filter.async_session'
    ) as mock_async_session:
        mock_session = AsyncMock()
        mock_session.exec = mock_session_exec
        mock_async_session.return_value.__aenter__.return_value = mock_session

        filtered_workers, messages = await filter_instance.filter(workers)

        assert len(filtered_workers) == 1
        assert len(messages) == 0


@pytest.mark.asyncio
async def test_has_supported_runners_with_list_service_runners():
    """
    Test 13: Test _has_supported_runners method when using list_service_runners.
    """
    model = create_model(backend="vLLM", backend_version=None)
    workers = [linux_nvidia_4_4080_16gx4()]

    filter_instance = BackendFrameworkFilter(model)

    async def mock_session_exec(statement):
        # Return None for backend (no version configs in database)
        mock_result = MagicMock()
        mock_result.first.return_value = None
        return mock_result

    with patch(
        'gpustack.policies.worker_filters.backend_framework_filter.async_session'
    ) as mock_async_session:
        mock_session = AsyncMock()
        mock_session.exec = mock_session_exec
        mock_async_session.return_value.__aenter__.return_value = mock_session

        # Mock list_service_runners to return a runner
        with patch(
            'gpustack.policies.worker_filters.backend_framework_filter.list_service_runners'
        ) as mock_list:
            mock_list.return_value = [
                {
                    "version": "0.11.0",
                    "backend": "cuda",
                }
            ]

            filtered_workers, messages = await filter_instance.filter(workers)

            # Worker should pass because list_service_runners returned a runner
            assert len(filtered_workers) == 1
            assert filtered_workers[0].name == "host-4-4080"
            assert len(messages) == 0

            # Verify list_service_runners was called with correct parameters
            mock_list.assert_called_once()
            call_kwargs = mock_list.call_args[1]
            assert call_kwargs["backend"] == "cuda"
            assert call_kwargs["service"] == "vllm"
            assert call_kwargs["with_deprecated"] is False


@pytest.mark.asyncio
async def test_cpu_worker_with_built_in_cpu_support():
    """
    Test 13: CPU worker should pass when CPU is in built_in_frameworks.
    """
    model = create_model(backend="llama.cpp", backend_version=None)
    workers = [linux_nvidia_4_4080_16gx4()]

    backend = create_inference_backend(
        backend_name="llama.cpp",
        version_configs={
            "cpu-version": VersionConfig(
                image_name="test:cpu",
                custom_framework="cpu",
            )
        },
        is_built_in=True,
    )

    filter_instance = BackendFrameworkFilter(model)

    async def mock_session_exec(statement):
        mock_result = MagicMock()
        mock_result.first.return_value = backend
        mock_result.all.return_value = [backend]
        return mock_result

    with patch(
        'gpustack.policies.worker_filters.backend_framework_filter.async_session'
    ) as mock_async_session:
        mock_session = AsyncMock()
        mock_session.exec = mock_session_exec
        mock_async_session.return_value.__aenter__.return_value = mock_session

        with patch(
            'gpustack.policies.worker_filters.backend_framework_filter.list_service_runners',
            return_value=[],
        ):
            filtered_workers, messages = await filter_instance.filter(workers)

            # CPU worker should pass since CPU is in built_in_frameworks
            assert len(filtered_workers) == 1
            assert filtered_workers[0].name == "host-4-4080"
            assert len(messages) == 0


@pytest.mark.asyncio
async def test_cuda_version_incompatibility():
    """
    Test 14: Worker with CUDA 12.4 should be filtered out when runner only supports CUDA 12.8.
    """
    model = create_model(backend="vLLM", backend_version="0.13.0")

    # Create a worker with CUDA 12.4 runtime
    worker = linux_nvidia_4_4080_16gx4()
    if worker.status and worker.status.gpu_devices:
        for gpu in worker.status.gpu_devices:
            gpu.runtime_version = "12.4"

    workers = [worker]

    filter_instance = BackendFrameworkFilter(model)

    async def mock_session_exec(statement):
        # Return None for backend (no version configs in database)
        mock_result = MagicMock()
        mock_result.first.return_value = None
        return mock_result

    with patch(
        'gpustack.policies.worker_filters.backend_framework_filter.async_session'
    ) as mock_async_session:
        mock_session = AsyncMock()
        mock_session.exec = mock_session_exec
        mock_async_session.return_value.__aenter__.return_value = mock_session

        with patch(
            'gpustack.policies.worker_filters.backend_framework_filter.list_service_runners',
            return_value=[],
        ):
            filtered_workers, messages = await filter_instance.filter(workers)

        # Worker should be filtered out because available runners don't support CUDA 12.4
        # (real list_service_runners will return runners with 12.4 and 12.8,
        # but 12.8 > 12.4 means no backward compatible runner)
        assert len(filtered_workers) == 0
        assert len(messages) == 1
        assert "host-4-4080" in messages[0]


@pytest.mark.asyncio
async def test_cpu_offloading_with_incompatible_gpu():
    """
    Test 15: Worker with incompatible GPU should pass when model has cpu_offloading=True and CPU runner exists.

    This test verifies that when a model has cpu_offloading enabled, a worker with an incompatible GPU
    can still be considered a valid candidate if a CPU runner is available.
    """
    model = create_model(backend="vLLM", backend_version="0.11.0", cpu_offloading=True)

    # Create a worker with ROCm GPU (incompatible with vLLM in this test scenario)
    worker = linux_nvidia_4_4080_16gx4()
    if worker.status and worker.status.gpu_devices:
        for gpu in worker.status.gpu_devices:
            gpu.type = "rocm"
            gpu.vendor = "amd"

    workers = [worker]

    # Backend only supports CUDA, not ROCm
    backend = create_inference_backend(
        backend_name="vLLM",
        version_configs={
            "0.11.0": VersionConfig(
                image_name="test:0.11.0",
                built_in_frameworks=[
                    "cuda",
                    "cpu",
                ],  # Supports CUDA and CPU, but not ROCm
                custom_framework="",
            )
        },
        is_built_in=True,
    )

    filter_instance = BackendFrameworkFilter(model)

    async def mock_session_exec(statement):
        mock_result = MagicMock()
        mock_result.first.return_value = backend
        return mock_result

    with patch(
        'gpustack.policies.worker_filters.backend_framework_filter.async_session'
    ) as mock_async_session:
        mock_session = AsyncMock()
        mock_session.exec = mock_session_exec
        mock_async_session.return_value.__aenter__.return_value = mock_session

        filtered_workers, messages = await filter_instance.filter(workers)

        # Worker should pass because cpu_offloading=True adds CPU as a query condition,
        # and CPU is in the built_in_frameworks
        assert len(filtered_workers) == 1
        assert filtered_workers[0].name == "host-4-4080"
        assert len(messages) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
