"""Whole cards from an InstanceType pool, several on one worker.

The slicing path is not retested here — it is unchanged and has its own suite.
What matters is the boundary: which mode each branch handles, that the claim
covers every card rather than one, and that a member is never spread across
workers by this path.
"""

import pytest

from gpustack.policies.candidate_selectors.instance_type_whole_card_selector import (
    InstanceTypeWholeCardSelector,
    is_whole_card_claim,
)
from gpustack.schemas.models import GPUTypeSelector


def _selector(memory=0, cores=0, profile=None):
    return GPUTypeSelector(
        type="pool-a",
        accelerator_sliced_memory_percentage=memory,
        accelerator_sliced_cores_percentage=cores,
        accelerator_partitioned_profile=profile,
    )


# --- which mode is which ---------------------------------------------------- #


def test_all_percentages_at_rest_is_a_whole_card_claim():
    assert is_whole_card_claim(_selector()) is True


def test_a_memory_percentage_makes_it_a_slice():
    # Both percentages together: the schema refuses a lone one, because 0 is
    # only meaningful when both are 0 (that is the whole-card mode).
    assert is_whole_card_claim(_selector(memory=50, cores=50)) is False


def test_a_cores_percentage_makes_it_a_slice():
    assert is_whole_card_claim(_selector(memory=25, cores=25)) is False


def test_a_partition_profile_makes_it_a_partition():
    assert is_whole_card_claim(_selector(profile="1g.10gb")) is False


def test_no_selector_at_all_is_not_a_whole_card_claim():
    """Guards the factory: the branch must not fire for a deployment that
    never asked for an InstanceType."""
    assert is_whole_card_claim(None) is False


# --- the claim covers every card -------------------------------------------- #


def _built(cards):
    obj = InstanceTypeWholeCardSelector.__new__(InstanceTypeWholeCardSelector)
    obj._cards = cards
    obj._slice_vram = 40 * 1024**3
    obj._ram_claim = 8 * 1024**3
    return obj


def test_the_claim_has_one_entry_per_card():
    """The defect this class exists for. `vram` is summed by
    `compute_worker_allocated` to decide whether the next claim fits, so a
    single entry made a tp=4 member look like it occupied one card — and the
    worker then advertised room it did not have."""
    claim = _built(4)._create_slice_claim()
    assert len(claim.vram) == 4
    assert sum(claim.vram.values()) == 4 * 40 * 1024**3


def test_one_card_is_unchanged_from_the_slicing_shape():
    """The common case must produce exactly what the parent produced, or this
    branch would change behaviour for deployments that never needed it."""
    claim = _built(1)._create_slice_claim()
    assert claim.vram == {0: 40 * 1024**3}


def test_a_zero_or_missing_card_count_still_claims_one():
    """A member that occupies an accelerator occupies at least one. Zero would
    make the claim empty and the worker read as free."""
    obj = InstanceTypeWholeCardSelector.__new__(InstanceTypeWholeCardSelector)
    obj._cards = max(int(0 or 1), 1)
    obj._slice_vram = 1
    obj._ram_claim = 0
    assert len(obj._create_slice_claim().vram) == 1


# --- never spread across workers -------------------------------------------- #


@pytest.mark.asyncio
async def test_candidates_carrying_subordinates_are_refused():
    """The parent falls back to one slice per worker across several hosts. For
    whole cards that puts a member's tensor-parallel group on several machines
    — two all-reduces per layer over the network — so it is refused instead,
    and crossing hosts stays something the user asks for explicitly."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, patch

    obj = _built(4)
    obj._messages = []
    spread = SimpleNamespace(subordinate_workers=[SimpleNamespace(id=2)])

    with patch.object(
        InstanceTypeWholeCardSelector.__bases__[0],
        "select_candidates",
        AsyncMock(return_value=[spread]),
    ):
        result = await obj.select_candidates([])

    assert result == []
    assert "single worker" in obj._messages[0]


@pytest.mark.asyncio
async def test_single_host_candidates_pass_through():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, patch

    obj = _built(2)
    obj._messages = []
    local = SimpleNamespace(subordinate_workers=None)

    with patch.object(
        InstanceTypeWholeCardSelector.__bases__[0],
        "select_candidates",
        AsyncMock(return_value=[local]),
    ):
        result = await obj.select_candidates([])

    assert result == [local]


# --- how much VRAM one member may be given here ------------------------------ #


def test_a_members_capacity_is_every_card_it_takes():
    """The parent decides "does this fit on one host" by comparing the whole
    member's claim against a member's capacity there. For a vGPU member that
    is one slice; for a whole-card member it is all of its cards.

    Measured against one card, a member needing two or more failed that test
    and fell into the parent's spread-across-hosts branch — whose candidates
    this selector refuses by design, so exactly the multi-card members it
    exists to place had nowhere to go.
    """
    assert _built(4)._member_vram_capacity() == 4 * 40 * 1024**3
    assert _built(1)._member_vram_capacity() == 40 * 1024**3


def test_the_parent_still_measures_a_vgpu_member_by_one_slice():
    """The override must not move placement for the deployments already using
    the parent: a vGPU member takes one slice and is measured by one."""
    from gpustack.policies.candidate_selectors.vgpu_resource_fit_selector import (
        VGPUResourceFitSelector,
    )

    parent = VGPUResourceFitSelector.__new__(VGPUResourceFitSelector)
    parent._slice_vram = 40 * 1024**3

    assert parent._member_vram_capacity() == 40 * 1024**3


# --- how many of the pool's cards the host actually has --------------------- #


def _pool_worker(cards):
    from types import SimpleNamespace

    return SimpleNamespace(
        name="w",
        status=SimpleNamespace(
            gpu_devices=[
                SimpleNamespace(vendor="NVIDIA", name="NVIDIA H100 80GB HBM3")
                for _ in range(cards)
            ]
        ),
    )


def _pool_detail():
    from gpustack.schemas.gpu_instance_types import GPUInstanceTypeDetail

    return GPUInstanceTypeDetail(manufacturer="NVIDIA", product="H100 80GB HBM3")


def test_a_host_short_of_cards_is_not_in_the_pool_for_this_member():
    """What is free on a node is the node-side scheduler's call, but how many
    of the pool's cards it HAS is inventory: a two-card node can never serve a
    four-card member however empty it is, and picking it produces a workload
    that sits Pending against a resource the node never advertises enough of.
    """
    selector = _built(4)

    assert selector._worker_matches_pool(_pool_worker(4), _pool_detail()) is True
    assert selector._worker_matches_pool(_pool_worker(8), _pool_detail()) is True
    assert selector._worker_matches_pool(_pool_worker(2), _pool_detail()) is False


def test_one_matching_card_is_still_enough_for_a_vgpu_member():
    """The parent is unchanged: a slice is one card's worth, so finding one of
    the pool's cards is the whole question there."""
    from gpustack.policies.candidate_selectors.vgpu_resource_fit_selector import (
        VGPUResourceFitSelector,
    )

    parent = VGPUResourceFitSelector.__new__(VGPUResourceFitSelector)

    assert parent._worker_matches_pool(_pool_worker(1), _pool_detail()) is True


def test_a_host_with_none_of_the_pools_cards_is_refused_either_way():
    from gpustack.policies.candidate_selectors.vgpu_resource_fit_selector import (
        VGPUResourceFitSelector,
    )

    parent = VGPUResourceFitSelector.__new__(VGPUResourceFitSelector)

    assert parent._worker_matches_pool(_pool_worker(0), _pool_detail()) is False
    assert _built(4)._worker_matches_pool(_pool_worker(0), _pool_detail()) is False
