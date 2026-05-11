"""
Test for evaluate_runtime_version functionality

Test scenarios:
- vLLM backend requires CUDA 12.6+ (inclusive)
- Worker has CUDA version 12.4
- Expected: Should return incompatible with a message suggesting upgrade to at least 12.6

"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from gpustack.scheduler.evaluator import evaluate_runtime_version
from gpustack.schemas.model_evaluations import ModelSpec
from gpustack.schemas.workers import Worker, WorkerStatus, GPUDeviceStatus


_PINNED_VLLM_CUDA_VERSIONS = ["12.6", "12.8", "12.9", "13.0"]


def _fake_list_backend_runners(**kwargs):
    """Stand-in for ``gpustack_runner.list_backend_runners``.

    Covers both call shapes used by ``_check_runtime_version``:

    - With ``backend_version`` set: exact-match supportability check;
      returns a non-empty list iff the version is in the pinned matrix.
    - Without ``backend_version`` (after the caller pops it): enumeration
      of all supported versions; always returns one fake runner whose
      ``versions`` mirrors the pinned matrix.

    Requests for unrelated backend/service combos return empty so the
    fake stays scoped to vLLM+CUDA.
    """
    if kwargs.get("backend") != "cuda" or kwargs.get("service") != "vllm":
        return []
    fake_runner = SimpleNamespace(
        backend="cuda",
        versions=[SimpleNamespace(version=v) for v in _PINNED_VLLM_CUDA_VERSIONS],
    )
    backend_version = kwargs.get("backend_version")
    if backend_version is not None:
        return [fake_runner] if backend_version in _PINNED_VLLM_CUDA_VERSIONS else []
    return [fake_runner]


@pytest.fixture(autouse=True)
def pin_runner_matrix():
    """Freeze the vLLM/CUDA runner matrix for the whole module.

    Without this, an upstream ``gpustack_runner`` bump (e.g., adding CUDA
    13.0 or marking 12.6 deprecated) would shift min/recommended versions
    and break the literal-version assertions below.
    """
    with patch(
        "gpustack.scheduler.evaluator.list_backend_runners",
        side_effect=_fake_list_backend_runners,
    ):
        yield


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
    assert (
        _PINNED_VLLM_CUDA_VERSIONS[-1] in messages[0]
    ), f"Message should contain recommended version {_PINNED_VLLM_CUDA_VERSIONS[-1]}"


@pytest.mark.asyncio
async def test_cuda_12_8_should_pass():
    """
    Test scenario 2: CUDA 12.8 should pass.

    Expected:
    - compatible should be True
    - messages should be empty
    """
    model = create_vllm_model()
    workers = [create_cuda_worker("12.8", "worker-cuda-12.8")]

    compatible, messages = await evaluate_runtime_version(model, workers)

    # Assertions
    assert compatible is True, "CUDA 12.8 should be compatible"
    assert len(messages) == 0, "Should have no error messages"


@pytest.mark.asyncio
async def test_mixed_workers():
    """
    Test scenario 3: Mixed version workers.

    Expected:
    - Highest version is 12.8, which meets the requirement
    - Should be compatible
    """
    model = create_vllm_model()
    workers = [
        create_cuda_worker("12.2", "worker-cuda-12.2"),
        create_cuda_worker("12.4", "worker-cuda-12.4"),
        create_cuda_worker("12.8", "worker-cuda-12.8"),
    ]

    compatible, messages = await evaluate_runtime_version(model, workers)

    # Assertions
    assert (
        compatible is True
    ), "Should be compatible (highest version 12.8 meets requirement)"
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
    assert (
        _PINNED_VLLM_CUDA_VERSIONS[0] in messages[0]
    ), f"Message should contain lowest supported version {_PINNED_VLLM_CUDA_VERSIONS[0]}"
    assert (
        _PINNED_VLLM_CUDA_VERSIONS[-1] in messages[0]
    ), f"Message should contain recommended version {_PINNED_VLLM_CUDA_VERSIONS[-1]}"


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
