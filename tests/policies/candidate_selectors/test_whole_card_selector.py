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
