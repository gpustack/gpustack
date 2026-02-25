from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gpustack.policies.scorers.model_file_locality_scorer import (
    ModelFileLocalityScorer,
)
from gpustack.policies.base import ModelInstanceScheduleCandidate
from gpustack.schemas.model_files import ModelFileStateEnum
from gpustack.schemas.models import (
    ComputedResourceClaim,
    Model,
    ModelInstanceSubordinateWorker,
    ModelSource,
    SourceEnum,
)
from tests.fixtures.workers.fixtures import linux_cpu_1, linux_cpu_2


def _mock_model_file(worker_id, state=ModelFileStateEnum.READY):
    return SimpleNamespace(worker_id=worker_id, state=state, resolved_paths=[])


def _make_candidate(worker, subordinate_workers=None, score=None):
    return ModelInstanceScheduleCandidate(
        worker=worker,
        gpu_indexes=[],
        computed_resource_claim=ComputedResourceClaim(ram=0, vram={}),
        score=score,
        subordinate_workers=subordinate_workers,
    )


@pytest.mark.asyncio
async def test_locality_score_no_ready_files():
    model = Model(name="m", source=SourceEnum.HUGGING_FACE, huggingface_repo_id="a/b")
    candidates = [_make_candidate(linux_cpu_1())]

    mock_session = AsyncMock()
    mock_async_session = AsyncMock()
    mock_async_session.__aenter__.return_value = mock_session

    with (
        patch(
            "gpustack.policies.scorers.model_file_locality_scorer.async_session",
            return_value=mock_async_session,
        ),
        patch(
            "gpustack.server.services.ModelFileService.get_by_source_index",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "gpustack.server.services.ModelFileService.get_by_resolved_path",
            new=AsyncMock(return_value=[]),
        ),
    ):
        scorer = ModelFileLocalityScorer(model, max_score=5.0)
        scored = await scorer.score(candidates)

    assert scored[0].score == 0.0


@pytest.mark.asyncio
async def test_locality_score_partial_ready_main_only():
    worker1 = linux_cpu_1()
    worker2 = linux_cpu_2()
    model = Model(name="m", source=SourceEnum.HUGGING_FACE, huggingface_repo_id="a/b")

    candidates = [
        _make_candidate(worker1),
        _make_candidate(worker2),
    ]

    mock_session = AsyncMock()
    mock_async_session = AsyncMock()
    mock_async_session.__aenter__.return_value = mock_session

    with (
        patch(
            "gpustack.policies.scorers.model_file_locality_scorer.async_session",
            return_value=mock_async_session,
        ),
        patch(
            "gpustack.server.services.ModelFileService.get_by_source_index",
            new=AsyncMock(return_value=[_mock_model_file(worker1.id)]),
        ),
        patch(
            "gpustack.server.services.ModelFileService.get_by_resolved_path",
            new=AsyncMock(return_value=[]),
        ),
    ):
        scorer = ModelFileLocalityScorer(model, max_score=5.0)
        scored = await scorer.score(candidates)

    assert scored[0].score == 5.0
    assert scored[1].score == 0.0


@pytest.mark.asyncio
async def test_locality_score_with_subordinate_workers():
    worker1 = linux_cpu_1()
    worker2 = linux_cpu_2()
    model = Model(name="m", source=SourceEnum.HUGGING_FACE, huggingface_repo_id="a/b")

    subordinate = ModelInstanceSubordinateWorker(worker_id=worker2.id)
    candidates = [_make_candidate(worker1, subordinate_workers=[subordinate])]

    mock_session = AsyncMock()
    mock_async_session = AsyncMock()
    mock_async_session.__aenter__.return_value = mock_session

    with (
        patch(
            "gpustack.policies.scorers.model_file_locality_scorer.async_session",
            return_value=mock_async_session,
        ),
        patch(
            "gpustack.server.services.ModelFileService.get_by_source_index",
            new=AsyncMock(return_value=[_mock_model_file(worker1.id)]),
        ),
        patch(
            "gpustack.server.services.ModelFileService.get_by_resolved_path",
            new=AsyncMock(return_value=[]),
        ),
    ):
        scorer = ModelFileLocalityScorer(model, max_score=5.0)
        scored = await scorer.score(candidates)

    assert scored[0].score == pytest.approx(2.5)


@pytest.mark.asyncio
async def test_locality_score_main_and_draft_average():
    worker1 = linux_cpu_1()
    worker2 = linux_cpu_2()
    model = Model(name="m", source=SourceEnum.HUGGING_FACE, huggingface_repo_id="a/b")
    draft = ModelSource(source=SourceEnum.HUGGING_FACE, huggingface_repo_id="c/d")

    subordinate = ModelInstanceSubordinateWorker(worker_id=worker2.id)
    candidates = [_make_candidate(worker1, subordinate_workers=[subordinate])]

    mock_session = AsyncMock()
    mock_async_session = AsyncMock()
    mock_async_session.__aenter__.return_value = mock_session

    with (
        patch(
            "gpustack.policies.scorers.model_file_locality_scorer.async_session",
            return_value=mock_async_session,
        ),
        patch(
            "gpustack.server.services.ModelFileService.get_by_source_index",
            new=AsyncMock(
                side_effect=[
                    [_mock_model_file(worker1.id)],  # main
                    [_mock_model_file(worker2.id)],  # draft
                ]
            ),
        ),
        patch(
            "gpustack.server.services.ModelFileService.get_by_resolved_path",
            new=AsyncMock(return_value=[]),
        ),
    ):
        scorer = ModelFileLocalityScorer(model, draft_model_source=draft, max_score=5.0)
        scored = await scorer.score(candidates)

    assert scored[0].score == pytest.approx(2.5)


@pytest.mark.asyncio
async def test_locality_score_main_preferred_over_draft():
    worker1 = linux_cpu_1()
    model = Model(name="m", source=SourceEnum.HUGGING_FACE, huggingface_repo_id="a/b")
    draft = ModelSource(source=SourceEnum.HUGGING_FACE, huggingface_repo_id="c/d")

    mock_session = AsyncMock()
    mock_async_session = AsyncMock()
    mock_async_session.__aenter__.return_value = mock_session

    with (
        patch(
            "gpustack.policies.scorers.model_file_locality_scorer.async_session",
            return_value=mock_async_session,
        ),
        patch(
            "gpustack.server.services.ModelFileService.get_by_source_index",
            new=AsyncMock(
                side_effect=[
                    [_mock_model_file(worker1.id)],  # main ready
                    [_mock_model_file(worker1.id)],  # draft ready
                ]
            ),
        ),
        patch(
            "gpustack.server.services.ModelFileService.get_by_resolved_path",
            new=AsyncMock(return_value=[]),
        ),
    ):
        scorer = ModelFileLocalityScorer(model, draft_model_source=draft, max_score=5.0)
        scored = await scorer.score([_make_candidate(worker1)])

    main_score = scored[0].score

    with (
        patch(
            "gpustack.policies.scorers.model_file_locality_scorer.async_session",
            return_value=mock_async_session,
        ),
        patch(
            "gpustack.server.services.ModelFileService.get_by_source_index",
            new=AsyncMock(
                side_effect=[
                    [],  # main not ready
                    [_mock_model_file(worker1.id)],  # draft ready
                ]
            ),
        ),
        patch(
            "gpustack.server.services.ModelFileService.get_by_resolved_path",
            new=AsyncMock(return_value=[]),
        ),
    ):
        scorer = ModelFileLocalityScorer(model, draft_model_source=draft, max_score=5.0)
        scored = await scorer.score([_make_candidate(worker1)])

    draft_only_score = scored[0].score

    assert main_score > draft_only_score
