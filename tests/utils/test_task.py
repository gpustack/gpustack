"""first_successful must not leave losing tasks running after it has already
returned a result to its caller."""

import asyncio

import pytest

from gpustack.utils.task import first_successful


@pytest.mark.asyncio
async def test_first_successful_cancels_and_cleans_up_losers():
    closed = []

    async def candidate(name: str, delay: float, result):
        try:
            await asyncio.sleep(delay)
            return result
        finally:
            closed.append(name)

    before = asyncio.all_tasks()
    result = await first_successful(
        [
            candidate("winner", 0, "ok"),
            candidate("loser-1", 90, "late"),
            candidate("loser-2", 90, "late"),
        ]
    )
    spawned = asyncio.all_tasks() - before

    assert result == "ok"
    assert all(task.done() for task in spawned)
    assert sorted(closed) == ["loser-1", "loser-2", "winner"]


@pytest.mark.asyncio
async def test_first_successful_honors_predicate_and_exhaustion():
    async def candidate(result):
        return result

    # A falsy-but-valid result is accepted when the predicate allows it.
    assert (
        await first_successful(
            [candidate(None), candidate(0)],
            is_success=lambda size: size is not None,
        )
        == 0
    )

    # Every candidate rejected yields None instead of hanging.
    assert await first_successful([candidate(None), candidate(None)]) is None
