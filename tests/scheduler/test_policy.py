import pytest
from gpustack.scheduler.policy import ResourceFitPolicy
from gpustack.schemas.models import (
    ComputedResourceClaim,
    ModelInstance,
    ModelInstanceStateEnum,
)
from tests.scheduler.fixtures.fixtures import (
    llama3_8b_estimate_claim,
    llama3_70b_estimate_claim,
    worker_macos_metal,
    worker_linux_nvidia_multi_gpu,
    worker_linux_nvidia_single_gpu,
    system_reserved,
)

from unittest.mock import patch


@pytest.mark.asyncio
async def test_filter():
    workers = [
        worker_macos_metal(),
        worker_linux_nvidia_single_gpu(),
        worker_linux_nvidia_multi_gpu(),
    ]

    reserved = system_reserved()
    claim = llama3_8b_estimate_claim()

    policy = ResourceFitPolicy(estimate=claim.estimate, system_reserved=reserved)

    with patch.object(policy, '_get_worker_model_instances', return_value=[]):
        filtered_workers = await policy.filter(workers)
        assert len(filtered_workers) == 4

        # uma
        assert filtered_workers[0].computed_resource_claim.offload_layers == 33
        assert filtered_workers[0].computed_resource_claim.is_unified_memory
        assert filtered_workers[0].computed_resource_claim.memory == 88206296
        assert filtered_workers[0].computed_resource_claim.gpu_memory == 5000658944

        # non uma
        assert filtered_workers[1].computed_resource_claim.offload_layers == 33
        assert not filtered_workers[1].computed_resource_claim.is_unified_memory
        assert filtered_workers[1].computed_resource_claim.memory == 245492696
        assert filtered_workers[1].computed_resource_claim.gpu_memory == 5964816384

        assert filtered_workers[2].computed_resource_claim.offload_layers == 33
        assert not filtered_workers[2].computed_resource_claim.is_unified_memory
        assert filtered_workers[2].computed_resource_claim.memory == 245492696
        assert filtered_workers[2].computed_resource_claim.gpu_memory == 5964816384

        assert filtered_workers[2].computed_resource_claim.offload_layers == 33
        assert not filtered_workers[2].computed_resource_claim.is_unified_memory
        assert filtered_workers[2].computed_resource_claim.memory == 245492696
        assert filtered_workers[2].computed_resource_claim.gpu_memory == 5964816384


@pytest.mark.asyncio
async def test_filter_with_cannot_full_offload_model():
    workers = [
        worker_macos_metal(),
        worker_linux_nvidia_single_gpu(),
        worker_linux_nvidia_multi_gpu(),
    ]

    reserved = system_reserved()
    estimate_claim = llama3_70b_estimate_claim()

    policy = ResourceFitPolicy(
        estimate=estimate_claim.estimate, system_reserved=reserved
    )

    with patch.object(policy, '_get_worker_model_instances', return_value=[]):
        filtered_workers = await policy.filter(workers)
        assert len(filtered_workers) == 4

        # uma
        assert filtered_workers[0].computed_resource_claim.offload_layers == 44
        assert filtered_workers[0].computed_resource_claim.is_unified_memory
        assert filtered_workers[0].computed_resource_claim.memory == 1896060312
        assert filtered_workers[0].computed_resource_claim.gpu_memory == 22656319488

        # non uma
        assert filtered_workers[1].computed_resource_claim.offload_layers == 47
        assert not filtered_workers[1].computed_resource_claim.is_unified_memory
        assert filtered_workers[1].computed_resource_claim.memory == 1952683416
        assert filtered_workers[1].computed_resource_claim.gpu_memory == 25604571136

        assert filtered_workers[2].computed_resource_claim.offload_layers == 30
        assert not filtered_workers[2].computed_resource_claim.is_unified_memory
        assert filtered_workers[2].computed_resource_claim.memory == 2523108760
        assert filtered_workers[2].computed_resource_claim.gpu_memory == 16850993152

        assert filtered_workers[3].computed_resource_claim.offload_layers == 30
        assert not filtered_workers[3].computed_resource_claim.is_unified_memory
        assert filtered_workers[3].computed_resource_claim.memory == 2523108760
        assert filtered_workers[3].computed_resource_claim.gpu_memory == 16850993152


@pytest.mark.asyncio
async def test_filter_with_system_reserved_and_existed_model_instances():
    workers = [
        worker_macos_metal(),
        worker_linux_nvidia_single_gpu(),
        worker_linux_nvidia_multi_gpu(),
    ]

    reserved = system_reserved(memory=1, gpu_memory=1)
    estimate_claim = llama3_70b_estimate_claim()

    policy = ResourceFitPolicy(
        estimate=estimate_claim.estimate, system_reserved=reserved
    )

    with patch.object(
        policy,
        '_get_worker_model_instances',
        return_value=[
            ModelInstance(
                name="test",
                worker_id=1,
                gpu_index=0,
                model_id=1,
                model_name="test",
                state=ModelInstanceStateEnum.running,
                computed_resource_claim=ComputedResourceClaim(
                    offload_layers=30,
                    is_unified_memory=True,
                    memory=15 * 1024 * 1024 * 1024,
                    gpu_memory=15 * 1024 * 1024 * 1024,
                ),
            )
        ],
    ):

        filtered_workers = await policy.filter(workers)
        assert len(filtered_workers) == 3

        assert filtered_workers[0].computed_resource_claim.offload_layers == 47
        assert not filtered_workers[0].computed_resource_claim.is_unified_memory
        assert filtered_workers[0].computed_resource_claim.memory == 1952683416
        assert filtered_workers[0].computed_resource_claim.gpu_memory == 25604571136

        assert filtered_workers[1].computed_resource_claim.offload_layers == 30
        assert not filtered_workers[1].computed_resource_claim.is_unified_memory
        assert filtered_workers[1].computed_resource_claim.memory == 2523108760
        assert filtered_workers[1].computed_resource_claim.gpu_memory == 16850993152

        assert filtered_workers[2].computed_resource_claim.offload_layers == 30
        assert not filtered_workers[2].computed_resource_claim.is_unified_memory
        assert filtered_workers[2].computed_resource_claim.memory == 2523108760
        assert filtered_workers[2].computed_resource_claim.gpu_memory == 16850993152
