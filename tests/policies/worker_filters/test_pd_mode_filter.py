import pytest

from gpustack.policies.worker_filters.pd_mode_filter import PDModeRuntimeFilter
from gpustack.schemas.models import DisaggregationSpec, Model, PDModeEnum
from tests.fixtures.workers.fixtures import (
    linux_ascend_1_910b_64gx8,
    linux_cpu_1,
    linux_nvidia_4_4080_16gx4,
)


def create_model(mode=None) -> Model:
    return Model(
        id=1,
        name="test-model",
        replicas=1,
        ready_replicas=0,
        source="huggingface",
        huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct",
        backend="vLLM",
        disaggregation=DisaggregationSpec(mode=mode) if mode else None,
    )


@pytest.mark.asyncio
async def test_no_disaggregation_is_a_noop():
    workers = [linux_nvidia_4_4080_16gx4(), linux_ascend_1_910b_64gx8()]
    candidates, messages = await PDModeRuntimeFilter(create_model()).filter(workers)
    assert candidates == workers
    assert messages == []


@pytest.mark.asyncio
async def test_custom_mode_is_never_narrowed():
    """The load-bearing case: an engine × accelerator pair with no built-in
    recipe must still be able to run PD by hand. `custom` declares no
    `gpu_filters`, so this filter has to stay a no-op for it on every
    accelerator -- otherwise "no built-in recipe" silently becomes "no PD"."""
    workers = [
        linux_nvidia_4_4080_16gx4(),
        linux_ascend_1_910b_64gx8(),
        linux_cpu_1(),
    ]
    candidates, messages = await PDModeRuntimeFilter(
        create_model(PDModeEnum.CUSTOM)
    ).filter(workers)
    assert candidates == workers
    assert messages == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [PDModeEnum.VLLM_NIXL, PDModeEnum.SGLANG_MOONCAKE, PDModeEnum.SGLANG_NIXL],
)
async def test_nvidia_modes_drop_non_nvidia_workers(mode):
    """These three modes declare no runtime of their own, so only
    `gpu_filters` keeps them off Ascend and AMD workers."""
    nvidia = linux_nvidia_4_4080_16gx4()
    ascend = linux_ascend_1_910b_64gx8()
    candidates, messages = await PDModeRuntimeFilter(create_model(mode)).filter(
        [nvidia, ascend]
    )

    assert candidates == [nvidia]
    assert len(messages) == 1
    assert "nvidia" in messages[0]


@pytest.mark.asyncio
async def test_ascend_mode_drops_non_ascend_workers():
    nvidia = linux_nvidia_4_4080_16gx4()
    ascend = linux_ascend_1_910b_64gx8()
    candidates, messages = await PDModeRuntimeFilter(
        create_model(PDModeEnum.VLLM_ASCEND_MOONCAKE)
    ).filter([nvidia, ascend])

    assert candidates == [ascend]
    assert len(messages) == 1
    assert "ascend" in messages[0]


@pytest.mark.asyncio
async def test_unrunnable_pair_leaves_nothing():
    """The whole point of the filter: the combination cannot work, so it must
    produce an empty candidate list rather than a worker that will fail inside
    the connector."""
    candidates, messages = await PDModeRuntimeFilter(
        create_model(PDModeEnum.VLLM_ASCEND_MOONCAKE)
    ).filter([linux_nvidia_4_4080_16gx4()])

    assert candidates == []
    assert len(messages) == 1


@pytest.mark.asyncio
async def test_worker_without_reported_devices_is_not_a_match():
    """A CPU-only worker reports no accelerator, so it cannot satisfy a vendor
    requirement -- absence is not a wildcard."""
    candidates, _ = await PDModeRuntimeFilter(
        create_model(PDModeEnum.VLLM_ASCEND_MOONCAKE)
    ).filter([linux_cpu_1()])

    assert candidates == []
