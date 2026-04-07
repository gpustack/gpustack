"""
Test for evaluate_runtime_version functionality

Test scenarios:
- vLLM backend requires CUDA 12.4+ (exclusive)
- Worker has CUDA version 12.4
- Expected: Should return incompatible with a message suggesting upgrade to at least 12.6
"""

import pytest
from gpustack.scheduler.evaluator import evaluate_runtime_version
from gpustack.schemas.model_evaluations import ModelSpec
from gpustack.schemas.workers import Worker, WorkerStatus, GPUDeviceStatus


def create_cuda_worker(cuda_version: str, worker_name: str = "test-worker") -> Worker:
    """Create a worker with specified CUDA version."""
    gpu_device = GPUDeviceStatus(
        index=0,
        name="NVIDIA RTX 4080",
        type="cuda",
        vendor="nvidia",
        runtime_version=cuda_version,
        memory_total=16 * 1024 * 1024 * 1024,  # 16GB
        memory_allocated=0,
    )

    worker_status = WorkerStatus(
        gpu_devices=[gpu_device],
    )

    worker = Worker(
        id=1,
        name=worker_name,
        hostname="test-host",
        ip="192.168.1.100",
        status=worker_status,
    )

    return worker


def create_vllm_model() -> ModelSpec:
    """Create a vLLM model spec."""
    return ModelSpec(
        name="test-vllm-model",
        replicas=1,
        source="huggingface",
        huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct",
        backend="vLLM",
    )


@pytest.mark.asyncio
async def test_cuda_12_4_should_be_incompatible():
    """
    Test scenario 1: CUDA 12.4 should be incompatible.

    Expected:
    - compatible should be False
    - messages should contain upgrade hint
    """
    model = create_vllm_model()
    workers = [create_cuda_worker("12.4", "worker-cuda-12.4")]

    compatible, messages = await evaluate_runtime_version(model, workers)

    # Assertions
    assert compatible is False, "CUDA 12.4 should be incompatible"
    assert len(messages) == 1, "Should have one error message"
    assert "12.4" in messages[0], "Message should contain current version 12.4"
    assert "12.6" in messages[0], "Message should contain minimum version 12.6"
    assert "12.8" in messages[0], "Message should contain recommended version 12.8"


@pytest.mark.asyncio
async def test_cuda_12_6_should_pass():
    """
    Test scenario 2: CUDA 12.6 should pass.

    Expected:
    - compatible should be True
    - messages should be empty
    """
    model = create_vllm_model()
    workers = [create_cuda_worker("12.6", "worker-cuda-12.6")]

    compatible, messages = await evaluate_runtime_version(model, workers)

    # Assertions
    assert compatible is True, "CUDA 12.6 should be compatible"
    assert len(messages) == 0, "Should have no error messages"


@pytest.mark.asyncio
async def test_mixed_workers():
    """
    Test scenario 3: Mixed version workers.

    Expected:
    - Highest version is 12.6, which meets the requirement
    - Should be compatible
    """
    model = create_vllm_model()
    workers = [
        create_cuda_worker("12.2", "worker-cuda-12.2"),
        create_cuda_worker("12.4", "worker-cuda-12.4"),
        create_cuda_worker("12.6", "worker-cuda-12.6"),
    ]

    compatible, messages = await evaluate_runtime_version(model, workers)

    # Assertions
    assert (
        compatible is True
    ), "Should be compatible (highest version 12.6 meets requirement)"
    assert len(messages) == 0, "Should have no error messages"


@pytest.mark.asyncio
async def test_all_workers_below_requirement():
    """
    Test scenario 4: All workers below requirement.

    Expected:
    - Should be incompatible
    - Should have clear upgrade hint
    """
    model = create_vllm_model()
    workers = [
        create_cuda_worker("12.2", "worker-cuda-12.2"),
        create_cuda_worker("12.4", "worker-cuda-12.4"),
    ]

    compatible, messages = await evaluate_runtime_version(model, workers)

    # Assertions
    assert compatible is False, "Should be incompatible"
    assert len(messages) == 1, "Should have one error message"
    assert "12.4" in messages[0], "Message should contain highest version 12.4"
    assert "12.6" in messages[0], "Message should contain minimum version 12.6"


@pytest.mark.asyncio
async def test_built_in_custom_backend_version_skips_runtime_check():
    """
    User-defined versions on built-in backends (suffix -custom) use their own
    images; gpustack-runner runtime matrix does not apply.
    """
    model = create_vllm_model().model_copy(update={"backend_version": "0.0.1-custom"})
    workers = [create_cuda_worker("12.2", "worker-below-default-req")]

    compatible, messages = await evaluate_runtime_version(model, workers)

    assert compatible is True
    assert messages == []


@pytest.mark.asyncio
async def test_built_in_explicit_image_skips_runtime_check():
    model = create_vllm_model().model_copy(
        update={"image_name": "example.com/my-vllm:tag"}
    )
    workers = [create_cuda_worker("12.2", "worker-below-default-req")]

    compatible, messages = await evaluate_runtime_version(model, workers)

    assert compatible is True
    assert messages == []
