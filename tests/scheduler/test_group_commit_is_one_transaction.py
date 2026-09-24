"""A group's members land together or not at all.

`solve_then_commit` exists so a group is never half-placed: the search proves
every member fits before anything is written. That guarantee ends at the write
if the rows go in one commit each — a failure partway through leaves some
members on workers and the rest PENDING, which the next reconcile reads as a
scale-out against a stale picture rather than as a group still forming.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gpustack.schemas.models import (
    ComputedResourceClaim,
    Model,
    ModelInstance,
    SourceEnum,
)
from gpustack.scheduler.scheduler import apply_candidate_to_instance


def _model():
    model = Model(
        name="llm", source=SourceEnum.HUGGING_FACE, huggingface_repo_id="org/repo"
    )
    model.id = 1
    return model


def _instance():
    return ModelInstance(
        id=1,
        name="llm-0",
        model_id=1,
        model_name="llm",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/repo",
    )


def _candidate():
    return SimpleNamespace(
        worker=SimpleNamespace(
            id=7,
            name="w7",
            ip="10.0.0.7",
            advertise_address=None,
            ifname="eth0",
        ),
        gpu_indexes=[0],
        gpu_addresses=None,
        gpu_type="cuda",
        computed_resource_claim=ComputedResourceClaim(vram={0: 1024}, ram=0),
        subordinate_workers=None,
    )


@pytest.mark.asyncio
async def test_the_single_instance_path_still_commits_its_own_row():
    """Unchanged: one instance is its own transaction, and nothing waits on it."""
    instance = _instance()
    with patch("gpustack.scheduler.scheduler.ModelInstanceService") as service:
        service.return_value.update = AsyncMock()
        await apply_candidate_to_instance(None, _model(), instance, _candidate())

    service.return_value.update.assert_awaited_once()
    assert instance.worker_id == 7


@pytest.mark.asyncio
async def test_a_group_member_is_written_but_not_committed_on_its_own():
    """The row is filled in exactly as before — only the commit is deferred, so
    the caller can put the whole group in one transaction."""
    instance = _instance()
    with patch("gpustack.scheduler.scheduler.ModelInstanceService") as service:
        service.return_value.update = AsyncMock()
        await apply_candidate_to_instance(
            None, _model(), instance, _candidate(), commit=False
        )

    service.return_value.update.assert_not_awaited()
    # Everything the committing caller needs is already on the row.
    assert instance.worker_id == 7
    assert instance.gpu_indexes == [0]
    assert instance.computed_resource_claim.vram == {0: 1024}
