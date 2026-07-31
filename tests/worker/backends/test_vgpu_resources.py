"""
Tests for the vGPU (gpu_type_selector) translation in the worker backend:

- ``_get_configured_resources`` emits the operator-recognized exclusive /
  sliced / partitioned resource keys instead of the whole-card device key.
- ``_transform_workload_plan`` hands the selector's InstanceType name to the
  runtime via ``WorkloadPlan.instance_type`` (the runtime owns Kueue queue
  admission; gpustack carries no Kueue code).
- Non-vGPU deployments stay bit-identical.
"""

import types

import pytest

from gpustack_runtime.deployer import WorkloadPlan

import gpustack.worker.backends.base as base_module
from gpustack.schemas.models import GPUTypeSelector
from gpustack.worker.backends.custom import CustomServer

# An operator-style InstanceType name (see docs/user-guide/gpuservice-instance-types.md).
TYPE_NAME = "gpustack--generic-ln-x64-12c-46g--nvidia-a10g-1d"


def _gpu_device(index=0, device_type="cuda"):
    return types.SimpleNamespace(
        index=index,
        type=device_type,
        runtime_version="12.8",
        arch_family=None,
    )


def _backend(
    gpu_type_selector=None,
    gpu_type="cuda",
    gpu_indexes=None,
    gpu_devices=None,
):
    backend = CustomServer.__new__(CustomServer)
    backend._model = types.SimpleNamespace(gpu_type_selector=gpu_type_selector)
    backend._model_instance = types.SimpleNamespace(
        worker_id=1,
        gpu_indexes=gpu_indexes if gpu_indexes is not None else [0],
        gpu_type=gpu_type,
        distributed_servers=None,
    )
    if gpu_devices is None:
        gpu_devices = [_gpu_device(device_type=gpu_type or "cuda")]
    backend._worker = types.SimpleNamespace(
        id=1,
        status=types.SimpleNamespace(gpu_devices=gpu_devices),
    )
    return backend


# --- vGPU resource translation --- #


def test_vgpu_soft_slice_resources():
    backend = _backend(
        gpu_type_selector=GPUTypeSelector(
            type=TYPE_NAME,
            accelerator_sliced_memory_percentage=40,
            accelerator_sliced_cores_percentage=40,
        ),
        gpu_indexes=[],
    )

    resources = backend._get_configured_resources()

    assert dict(resources) == {
        "nvidia.com/gpu.sliced": "1",
        "nvidia.com/gpu.sliced.memory-percentage": "40",
        "nvidia.com/gpu.sliced.cores-percentage": "40",
    }


def test_vgpu_unset_cores_percentage_defaults_to_100():
    # The schema mirrors the operator webhook defaulting rule: an unset cores
    # percentage defaults to 100 (a full compute budget).
    backend = _backend(
        gpu_type_selector=GPUTypeSelector(
            type=TYPE_NAME, accelerator_sliced_memory_percentage=30
        ),
        gpu_indexes=[],
    )

    resources = backend._get_configured_resources()

    assert dict(resources) == {
        "nvidia.com/gpu.sliced": "1",
        "nvidia.com/gpu.sliced.memory-percentage": "30",
        "nvidia.com/gpu.sliced.cores-percentage": "100",
    }


def test_vgpu_zero_percentage_is_exclusive_whole_card():
    # Both 0 means whole-card exclusive: the bare base resource, requested
    # through the type-based path (never the whole-card device key, and no
    # slicing keys — the operator webhook rejects percentage budgets of 0).
    backend = _backend(
        gpu_type_selector=GPUTypeSelector(
            type=TYPE_NAME,
            accelerator_sliced_memory_percentage=0,
            accelerator_sliced_cores_percentage=0,
        ),
        gpu_indexes=[],
    )

    resources = backend._get_configured_resources()

    assert dict(resources) == {"nvidia.com/gpu": "1"}


def test_vgpu_whole_card_exclusive_works_on_amd():
    # Exclusive also works on pools without slicing capability (e.g. AMD).
    backend = _backend(
        gpu_type_selector=GPUTypeSelector(
            type=TYPE_NAME,
            accelerator_sliced_memory_percentage=0,
            accelerator_sliced_cores_percentage=0,
        ),
        gpu_type="rocm",
        gpu_indexes=[],
    )

    resources = backend._get_configured_resources()

    assert dict(resources) == {"amd.com/gpu": "1"}


def test_vgpu_partitioned_profile_resources():
    backend = _backend(
        gpu_type_selector=GPUTypeSelector(
            type=TYPE_NAME, accelerator_partitioned_profile="1g.5gb"
        ),
        gpu_indexes=[],
    )

    resources = backend._get_configured_resources()

    assert dict(resources) == {
        "nvidia.com/gpu.partitioned": "1",
        "nvidia.com/gpu.partitioned.mig-1g.5gb": "1",
    }


def test_vgpu_amd_resource_base():
    backend = _backend(
        gpu_type_selector=GPUTypeSelector(
            type=TYPE_NAME,
            accelerator_sliced_memory_percentage=50,
        ),
        gpu_type="rocm",
        gpu_indexes=[],
    )

    resources = backend._get_configured_resources()

    assert dict(resources) == {
        "amd.com/gpu.sliced": "1",
        "amd.com/gpu.sliced.memory-percentage": "50",
        "amd.com/gpu.sliced.cores-percentage": "100",
    }


def test_vgpu_resource_base_falls_back_to_worker_device_type():
    # gpu_type not stamped on the instance: fall back to the worker's device.
    backend = _backend(
        gpu_type_selector=GPUTypeSelector(
            type=TYPE_NAME,
            accelerator_sliced_memory_percentage=25,
        ),
        gpu_type=None,
        gpu_indexes=[],
        gpu_devices=[_gpu_device(device_type="cuda")],
    )

    resources = backend._get_configured_resources()

    assert "nvidia.com/gpu.sliced" in resources


def test_vgpu_unresolvable_resource_base_fails_closed():
    # No GPU type anywhere: refuse to build the request rather than deploy
    # without isolation.
    backend = _backend(
        gpu_type_selector=GPUTypeSelector(
            type=TYPE_NAME,
            accelerator_sliced_memory_percentage=40,
        ),
        gpu_type=None,
        gpu_indexes=[],
        gpu_devices=[],
    )

    with pytest.raises(RuntimeError, match="gpu_type_selector"):
        backend._get_configured_resources()


def test_vgpu_unknown_gpu_type_fails_closed():
    backend = _backend(
        gpu_type_selector=GPUTypeSelector(
            type=TYPE_NAME,
            accelerator_sliced_memory_percentage=40,
        ),
        gpu_type="not-a-backend",
        gpu_indexes=[],
        gpu_devices=[_gpu_device(device_type="not-a-backend")],
    )

    with pytest.raises(RuntimeError, match="gpu_type_selector"):
        backend._get_configured_resources()


# --- Non-vGPU stays bit-identical --- #


def test_non_vgpu_resources_unchanged():
    backend = _backend(
        gpu_indexes=[0, 1],
        gpu_devices=[_gpu_device(index=0), _gpu_device(index=1)],
    )

    resources = backend._get_configured_resources()

    assert dict(resources) == {"nvidia.com/devices": "0,1"}


# --- WorkloadPlan instance_type handoff --- #


@pytest.fixture
def passthrough_transform(monkeypatch):
    # Keep _transform_workload_plan deployer-agnostic: the docker transform is
    # covered elsewhere and irrelevant to the instance_type handoff.
    monkeypatch.setattr(
        base_module, "transform_workload_plan", lambda _c, w, _f=None: w
    )


def _plan_backend(gpu_type_selector):
    backend = _backend(gpu_type_selector=gpu_type_selector)
    backend._config = types.SimpleNamespace(system_default_container_registry=None)
    backend._fallback_registry = None
    return backend


def test_vgpu_workload_carries_instance_type(passthrough_transform):
    backend = _plan_backend(
        GPUTypeSelector(type=TYPE_NAME, accelerator_sliced_memory_percentage=40)
    )
    plan = WorkloadPlan(name="w1")

    backend._transform_workload_plan(plan)

    assert plan.instance_type == TYPE_NAME


def test_vgpu_whole_card_exclusive_carries_instance_type(passthrough_transform):
    backend = _plan_backend(
        GPUTypeSelector(
            type=TYPE_NAME,
            accelerator_sliced_memory_percentage=0,
            accelerator_sliced_cores_percentage=0,
        )
    )
    plan = WorkloadPlan(name="w1")

    backend._transform_workload_plan(plan)

    assert plan.instance_type == TYPE_NAME


def test_non_vgpu_workload_instance_type_untouched(passthrough_transform):
    backend = _plan_backend(None)
    plan = WorkloadPlan(name="w1")

    backend._transform_workload_plan(plan)

    assert plan.instance_type is None
    assert plan.labels is None
