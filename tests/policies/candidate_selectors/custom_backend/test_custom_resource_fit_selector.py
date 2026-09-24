from unittest.mock import patch

from tests.utils.mock import mock_async_session

import pytest

from gpustack.policies.candidate_selectors.custom_backend_resource_fit_selector import (
    CustomBackendResourceFitSelector,
)
from gpustack.policies.scorers.placement_scorer import PlacementScorer
from tests.fixtures.workers.fixtures import (
    linux_nvidia_0_4090_24gx1,
    linux_nvidia_4_4080_16gx4,
    linux_cpu_1,
)
from tests.policies.candidate_selectors.vllm.test_vllm_resource_fit_selector import (
    make_model,
)


@pytest.mark.parametrize(
    "index, workers, model, expect_msg",
    [
        (
            1,
            [linux_nvidia_4_4080_16gx4()],
            make_model(1, None, "Qwen/Qwen2.5-Omni-7B"),
            [
                '- The model requires approximately 26.99 GiB of VRAM and 2.7 GiB of RAM.\n'
                '- The current available GPU only has 15.99 GiB allocatable VRAM.'
            ],
        ),
        (
            2,
            [linux_nvidia_4_4080_16gx4()],
            make_model(
                1,
                [
                    "host-4-4080:cuda:0",
                    "host-4-4080:cuda:1",
                    "host-4-4080:cuda:2",
                ],
                "Qwen/Qwen3-8B",
            ),
            [
                '- The model requires approximately 20.31 GiB of VRAM and 2.03 GiB of RAM.\n'
            ],
        ),
    ],
)
@pytest.mark.asyncio
async def test_schedule_single_work_multi_gpu(
    config, index, workers, model, expect_msg
):
    m = model

    mis = []

    resource_fit_selector = CustomBackendResourceFitSelector(config, m, mis)
    placement_scorer = PlacementScorer(m, mis)

    if index == 1:
        # Simulate a scenario where the model's num_attention_heads cannot be evenly divided by the gpu_count through auto-scheduling.
        resource_fit_selector._num_attention_heads = 25

    with (
        patch(
            'gpustack.schemas.workers.Worker.all',
            return_value=workers,
        ),
        patch(
            'gpustack.policies.worker_filters.backend_framework_filter.async_session',
            return_value=mock_async_session(),
        ),
        patch(
            'gpustack.policies.scorers.placement_scorer.async_session',
            return_value=mock_async_session(),
        ),
    ):

        candidates = await resource_fit_selector.select_candidates(workers)
        _ = await placement_scorer.score(candidates)

        assert resource_fit_selector._messages == expect_msg


@pytest.mark.parametrize(
    "workers, model, env_overrides, expect_msg",
    [
        (
            [linux_nvidia_4_4080_16gx4()],
            make_model(1, None, "Qwen/Qwen2.5-Omni-7B"),
            {"GPUSTACK_MODEL_VRAM_CLAIM": str(27 * 1024**3)},
            [
                '- The model requires approximately 27.0 GiB of VRAM and 2.7 GiB of RAM.\n'
                '- The current available GPU only has 15.99 GiB allocatable VRAM.'
            ],
        ),
        (
            [linux_cpu_1()],
            make_model(1, None, "Qwen/Qwen2.5-7B-Instruct", cpu_offloading=True),
            {"GPUSTACK_MODEL_VRAM_CLAIM": str(700 * 1024**3)},
            [
                '- The model requires approximately 700.0 GiB of VRAM and 70.0 GiB of RAM.\n'
                '- CPU-only inference is supported. Requires at least 70.0 GiB RAM.'
            ],
        ),
        (
            # Single-GPU worker that cannot fit the model: the multi-GPU
            # diagnostic must not appear (no worker has >=2 GPUs), otherwise
            # the user sees a misleading "0.0 GiB across 0 GPUs" message.
            [linux_nvidia_0_4090_24gx1()],
            make_model(1, None, "Qwen/Qwen-Image-Edit"),
            {"GPUSTACK_MODEL_VRAM_CLAIM": str(54 * 1024**3)},
            [
                '- The model requires approximately 54.0 GiB of VRAM and 5.4 GiB of RAM.\n'
                '- The current available GPU only has 24.23 GiB allocatable VRAM.'
            ],
        ),
    ],
)
@pytest.mark.asyncio
async def test_failed_cases_auto_schedule(
    config, workers, model, env_overrides, expect_msg
):
    model.env = env_overrides

    mis = []

    resource_fit_selector = CustomBackendResourceFitSelector(config, model, mis)
    placement_scorer = PlacementScorer(model, mis)

    with (
        patch(
            'gpustack.schemas.workers.Worker.all',
            return_value=workers,
        ),
        patch(
            'gpustack.policies.worker_filters.backend_framework_filter.async_session',
            return_value=mock_async_session(),
        ),
        patch(
            'gpustack.policies.scorers.placement_scorer.async_session',
            return_value=mock_async_session(),
        ),
    ):
        candidates = await resource_fit_selector.select_candidates(workers)
        _ = await placement_scorer.score(candidates)

        assert resource_fit_selector._messages == expect_msg


# --- an engine that seizes a fraction of the card ------------------------- #
#
# A custom backend booked at what its weights need is under-booked: vLLM behind
# it seizes its configured fraction of the card regardless. The ledger then
# reports free memory the engine has already taken, admits the group, and the
# member dies of `torch.OutOfMemoryError`.


def _selector(config, params):
    model = make_model(1, None, "Qwen/Qwen3-8B")
    model.backend_parameters = params
    return CustomBackendResourceFitSelector(config, model, [])


class _Card:
    def __init__(self, total):
        self.memory = type("Mem", (), {"total": total})()


_24G = 24 * 1024**3


def test_no_fraction_declared_leaves_the_claim_exactly_as_it_was(config):
    """The bound on this change. A custom backend is by definition an engine
    GPUStack does not model; most pre-allocate nothing, and assuming vLLM's
    default would make every one of them ask for most of a card."""
    selector = _selector(config, ["--some-unrelated-flag=1"])
    selector._vram_claim = 7 * 1024**3
    assert selector._whole_card_fraction is None
    assert selector._vram_claim_on(_Card(_24G)) == 7 * 1024**3


def test_a_declared_fraction_raises_the_claim_to_what_the_engine_will_seize(config):
    selector = _selector(config, ["--gpu-memory-utilization=0.9"])
    selector._vram_claim = 7 * 1024**3
    assert selector._whole_card_fraction == 0.9
    assert selector._vram_claim_on(_Card(_24G)) == int(_24G * 0.9)


def test_a_repeated_flag_reserves_from_the_value_the_engine_will_use(config):
    """argparse keeps the last occurrence, so the reservation must too.

    Reading the first one books less than the engine seizes, and the member is
    admitted onto a card it does not fit.
    """
    selector = _selector(
        config, ["--gpu-memory-utilization=0.5", "--gpu-memory-utilization=0.95"]
    )
    assert selector._whole_card_fraction == 0.95
    assert selector._vram_claim_on(_Card(_24G)) == int(_24G * 0.95)


def test_sglang_says_the_same_thing_with_another_name(config):
    selector = _selector(config, ["--mem-fraction-static=0.75"])
    assert selector._whole_card_fraction == 0.75
    assert selector._vram_claim_on(_Card(_24G)) == int(_24G * 0.75)


def test_the_weights_still_win_when_they_need_more_than_the_fraction(config):
    """The maximum, not the fraction alone: a fraction below what the weights
    need would under-book a model that does not fit — the same failure this
    fixes, pointed the other way."""
    selector = _selector(config, ["--gpu-memory-utilization=0.2"])
    selector._vram_claim = 20 * 1024**3
    assert selector._vram_claim_on(_Card(_24G)) == 20 * 1024**3


@pytest.mark.parametrize("raw", ["not-a-number", "0", "-0.5", "1.5"])
def test_a_value_the_engine_would_reject_is_not_acted_on(config, raw):
    """Clamping would book a number the engine will never honour, and the
    engine is the one that gets to reject its own argument."""
    assert (
        _selector(config, [f"--gpu-memory-utilization={raw}"])._whole_card_fraction
        is None
    )


# --- what the multi-GPU path books, per card ------------------------------- #


def _distribute(selector, worker, free_per_gpu):
    """The per-card VRAM the multi-GPU path would book on this worker."""
    from types import SimpleNamespace

    allocatable = SimpleNamespace(
        vram={i: free for i, free in enumerate(free_per_gpu)}, ram=64 * 1024**3
    )
    with patch(
        "gpustack.policies.candidate_selectors."
        "custom_backend_resource_fit_selector.get_worker_allocatable_resource",
        return_value=allocatable,
    ):
        candidates = selector.find_single_worker_multi_gpu_candidates([worker])
    if not candidates:
        return None
    return candidates[0].computed_resource_claim.vram


def test_without_a_fraction_the_weights_are_split_not_repeated(config):
    """The claim a custom backend has always made: weights divided across the
    cards it lands on. Booking the whole estimate on each card would reserve it
    once per GPU and starve everything placed after it."""
    selector = _selector(config, ["--some-unrelated-flag=1"])
    selector._vram_claim = 8 * 1024**3
    worker = linux_nvidia_4_4080_16gx4()
    free = [16 * 1024**3] * 4

    vram = _distribute(selector, worker, free)

    assert vram is not None
    # 8 GiB of weights over four cards, not 8 GiB on each of them.
    assert vram == {i: 2 * 1024**3 for i in range(4)}, vram


def test_a_card_that_cannot_hold_its_own_fraction_is_left_out(config):
    """The sum is not the test. One roomy card covering a nearly full one gets
    the member admitted, and the distribution then books the full per-card
    fraction against memory the ledger says is taken — an OOM at startup, not a
    scheduling near-miss."""
    selector = _selector(config, ["--gpu-memory-utilization=0.5"])
    selector._vram_claim = 4 * 1024**3
    worker = linux_nvidia_4_4080_16gx4()
    # Three cards free, one with 1 GiB left — it cannot hold its 8 GiB share.
    free = [16 * 1024**3, 16 * 1024**3, 16 * 1024**3, 1024**3]

    vram = _distribute(selector, worker, free)

    assert vram is not None, "the three roomy cards should still form a member"
    # The full card is gone from the candidate, and the three that remain each
    # carry the same per-card fraction rather than a share of the weights.
    assert 3 not in vram, vram
    assert len(set(vram.values())) == 1, vram
    assert all(v > 4 * 1024**3 for v in vram.values()), vram
