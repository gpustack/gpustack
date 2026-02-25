from types import SimpleNamespace
from typing import Dict, List
from unittest.mock import AsyncMock, patch

import pytest

from gpustack import envs
from gpustack.policies.base import (
    Allocatable,
    ModelInstanceScore,
    ModelInstanceScheduleCandidate,
    ModelInstanceScorer,
    ScheduleCandidatesScorer,
)
from gpustack.policies.scorers.model_file_locality_scorer import (
    ModelFileLocalityScorer,
)
from gpustack.policies.scorers.offload_layer_scorer import OffloadLayerScorer
from gpustack.policies.scorers.placement_scorer import (
    PlacementScorer,
    ScaleTypeEnum,
)
from gpustack.policies.scorers.score_chain import (
    CandidateScoreChain,
    ModelInstanceScoreChain,
)
from gpustack.policies.scorers.status_scorer import StatusScorer
from gpustack.schemas.model_files import ModelFileStateEnum
from gpustack.schemas.models import (
    ComputedResourceClaim,
    ModelInstanceStateEnum,
    PlacementStrategyEnum,
)
from gpustack.schemas.workers import WorkerStateEnum
from tests.fixtures.workers.fixtures import (
    linux_cpu_1,
    linux_cpu_2,
    linux_nvidia_19_4090_24gx2,
    linux_nvidia_2_4080_16gx2,
)
from tests.utils.model import new_model, new_model_instance


class ListCandidateScorer(ScheduleCandidatesScorer):
    def __init__(self, scores: List[float]):
        self._scores = scores

    async def score(self, candidates: List[ModelInstanceScheduleCandidate]):
        for idx, candidate in enumerate(candidates):
            candidate.score = self._scores[idx] if idx < len(self._scores) else None
        return candidates


class DummyInstanceScorer(ModelInstanceScorer):
    def __init__(self, scores: List[float], max_score: float):
        self._scores = scores
        self._max_score = max_score
        self.called = False

    async def score_instances(self, instances):
        self.called = True
        results = []
        for idx, instance in enumerate(instances):
            score = self._scores[idx] if idx < len(self._scores) else 0
            results.append(ModelInstanceScore(model_instance=instance, score=score))
        return results


def _make_candidate(worker, ram: int = 1, vram=None, gpu_indexes=None):
    return ModelInstanceScheduleCandidate(
        worker=worker,
        gpu_indexes=gpu_indexes,
        computed_resource_claim=ComputedResourceClaim(ram=ram, vram=vram or {}),
        score=None,
    )


def _build_candidates(worker1, worker2, ram: int = 1):
    return [
        _make_candidate(worker1, ram=ram),
        _make_candidate(worker2, ram=ram),
    ]


def _build_gpu_candidates(worker1, worker2, vram: int = 50, ram: int = 0):
    return [
        _make_candidate(worker1, ram=ram, vram={0: vram}, gpu_indexes=[0]),
        _make_candidate(worker2, ram=ram, vram={0: vram}, gpu_indexes=[0]),
    ]


def _mock_model_file(worker_id, state=ModelFileStateEnum.READY):
    return SimpleNamespace(worker_id=worker_id, state=state, resolved_paths=[])


def _scores_by_instance(
    scores: List[ModelInstanceScore],
) -> Dict[int, float]:
    return {item.model_instance.id: item.score for item in scores}


@pytest.mark.asyncio
async def test_candidate_score_chain_sums_scores():
    candidates = [
        _make_candidate(linux_cpu_1()),
        _make_candidate(linux_cpu_2()),
    ]
    chain = CandidateScoreChain(
        scorers=[
            ListCandidateScorer([10, 20]),
            ListCandidateScorer([1.5, None]),
        ]
    )

    scored = await chain.score(candidates)

    assert scored[0].score == 11.5
    assert scored[1].score == 20.0


@pytest.mark.asyncio
async def test_candidate_score_chain_handles_all_none():
    candidates = [
        _make_candidate(linux_cpu_1()),
        _make_candidate(linux_cpu_2()),
    ]
    chain = CandidateScoreChain(
        scorers=[
            ListCandidateScorer([None, None]),
            ListCandidateScorer([None, None]),
        ]
    )

    scored = await chain.score(candidates)

    assert scored[0].score == 0.0
    assert scored[1].score == 0.0


@pytest.mark.asyncio
async def test_instance_score_chain_skips_zero_max_score():
    instances = [new_model_instance(1, "i1", 1, worker_id=1)]
    zero_scorer = DummyInstanceScorer([100], max_score=0)
    valid_scorer = DummyInstanceScorer(
        [5], max_score=envs.SCHEDULER_SCALE_DOWN_OFFLOAD_MAX_SCORE
    )

    chain = ModelInstanceScoreChain(
        scorers=[zero_scorer, valid_scorer],
        total_max_score=None,
    )

    scored = await chain.score(instances)

    assert zero_scorer.called is False
    assert valid_scorer.called is True
    assert scored[0].score == 5.0


@pytest.mark.asyncio
async def test_candidate_score_chain_spread_locality():
    worker1 = linux_nvidia_19_4090_24gx2()
    worker2 = linux_nvidia_2_4080_16gx2()
    model = new_model(
        1,
        "test",
        placement_strategy=PlacementStrategyEnum.SPREAD,
        huggingface_repo_id="a/b",
    )
    model_instances = [
        new_model_instance(1, "m1", model.id, worker_id=worker1.id),
        new_model_instance(2, "m2", model.id, worker_id=worker2.id),
    ]

    placement_scorer = PlacementScorer(
        model,
        model_instances,
        max_score=envs.SCHEDULER_SCALE_UP_PLACEMENT_MAX_SCORE,
    )

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
            new=AsyncMock(return_value=[_mock_model_file(worker2.id)]),
        ),
        patch(
            "gpustack.server.services.ModelFileService.get_by_resolved_path",
            new=AsyncMock(return_value=[]),
        ),
    ):
        locality_scorer = ModelFileLocalityScorer(
            model,
            draft_model_source=None,
            max_score=envs.SCHEDULER_SCALE_UP_LOCALITY_MAX_SCORE,
        )

        placement_scored = await placement_scorer.score(
            _build_gpu_candidates(worker1, worker2, vram=1)
        )
        chain_scored = await CandidateScoreChain(
            [placement_scorer, locality_scorer]
        ).score(_build_gpu_candidates(worker1, worker2, vram=1))

    placement_scores = {c.worker.id: c.score for c in placement_scored}
    chain_scores = {c.worker.id: c.score for c in chain_scored}

    assert placement_scores[worker1.id] == pytest.approx(91.5)
    assert placement_scores[worker2.id] == pytest.approx(91.5)
    assert chain_scores[worker1.id] == pytest.approx(91.5)
    assert chain_scores[worker2.id] == pytest.approx(96.5)

    placement_pick = max(placement_scored, key=lambda c: c.score).worker.id
    chain_pick = max(chain_scored, key=lambda c: c.score).worker.id
    assert placement_pick == worker1.id
    assert chain_pick == worker2.id


@pytest.mark.asyncio
async def test_candidate_score_chain_binpack_and_locality():
    worker1 = linux_nvidia_19_4090_24gx2()
    worker2 = linux_nvidia_2_4080_16gx2()
    model = new_model(
        1,
        "test",
        placement_strategy=PlacementStrategyEnum.BINPACK,
        huggingface_repo_id="a/b",
    )

    placement_scorer = PlacementScorer(
        model, [], max_score=envs.SCHEDULER_SCALE_UP_PLACEMENT_MAX_SCORE
    )

    mock_session = AsyncMock()
    mock_async_session = AsyncMock()
    mock_async_session.__aenter__.return_value = mock_session

    def allocatable_side_effect(_, worker, gpu_type=None):
        return Allocatable(ram=0, vram={0: 100})

    with (
        patch(
            "gpustack.policies.scorers.placement_scorer.get_worker_allocatable_resource",
            side_effect=allocatable_side_effect,
        ),
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
        locality_scorer = ModelFileLocalityScorer(
            model,
            draft_model_source=None,
            max_score=envs.SCHEDULER_SCALE_UP_LOCALITY_MAX_SCORE,
        )

        placement_scored = await placement_scorer.score(
            _build_gpu_candidates(worker1, worker2, vram=50)
        )
        chain_scored = await CandidateScoreChain(
            [placement_scorer, locality_scorer]
        ).score(_build_gpu_candidates(worker1, worker2, vram=50))

    placement_scores = {c.worker.id: c.score for c in placement_scored}
    chain_scores = {c.worker.id: c.score for c in chain_scored}

    assert placement_scores[worker1.id] == pytest.approx(33.3333333)
    assert placement_scores[worker2.id] == pytest.approx(33.3333333)
    assert chain_scores[worker1.id] == pytest.approx(38.3333333)
    assert chain_scores[worker2.id] == pytest.approx(33.3333333)


@pytest.mark.asyncio
async def test_candidate_score_chain_binpack_locality_changes_pick():
    worker1 = linux_nvidia_19_4090_24gx2()
    worker2 = linux_nvidia_2_4080_16gx2()
    model = new_model(
        1,
        "test",
        placement_strategy=PlacementStrategyEnum.BINPACK,
        huggingface_repo_id="a/b",
    )

    placement_scorer = PlacementScorer(
        model, [], max_score=envs.SCHEDULER_SCALE_UP_PLACEMENT_MAX_SCORE
    )

    mock_session = AsyncMock()
    mock_async_session = AsyncMock()
    mock_async_session.__aenter__.return_value = mock_session

    def allocatable_side_effect(_, worker, gpu_type=None):
        if worker.id == worker1.id:
            return Allocatable(ram=0, vram={0: 100})
        return Allocatable(ram=0, vram={0: 110})

    with (
        patch(
            "gpustack.policies.scorers.placement_scorer.get_worker_allocatable_resource",
            side_effect=allocatable_side_effect,
        ),
        patch(
            "gpustack.policies.scorers.model_file_locality_scorer.async_session",
            return_value=mock_async_session,
        ),
        patch(
            "gpustack.server.services.ModelFileService.get_by_source_index",
            new=AsyncMock(return_value=[_mock_model_file(worker2.id)]),
        ),
        patch(
            "gpustack.server.services.ModelFileService.get_by_resolved_path",
            new=AsyncMock(return_value=[]),
        ),
    ):
        locality_scorer = ModelFileLocalityScorer(
            model,
            draft_model_source=None,
            max_score=envs.SCHEDULER_SCALE_UP_LOCALITY_MAX_SCORE,
        )

        placement_scored = await placement_scorer.score(
            _build_gpu_candidates(worker1, worker2, vram=50)
        )
        chain_scored = await CandidateScoreChain(
            [placement_scorer, locality_scorer]
        ).score(_build_gpu_candidates(worker1, worker2, vram=50))

    placement_scores = {c.worker.id: c.score for c in placement_scored}
    chain_scores = {c.worker.id: c.score for c in chain_scored}

    assert placement_scores[worker1.id] == pytest.approx(33.3333333)
    assert placement_scores[worker2.id] == pytest.approx(30.3030303)
    assert chain_scores[worker1.id] == pytest.approx(33.3333333)
    assert chain_scores[worker2.id] == pytest.approx(35.3030303)

    placement_pick = max(placement_scored, key=lambda c: c.score).worker.id
    chain_pick = max(chain_scored, key=lambda c: c.score).worker.id
    assert placement_pick == worker1.id
    assert chain_pick == worker2.id


@pytest.mark.asyncio
async def test_instance_score_chain_scales_with_total_max_score():
    instances = [
        new_model_instance(1, "i1", 1, worker_id=1),
        new_model_instance(2, "i2", 1, worker_id=2),
    ]
    scorer_a = DummyInstanceScorer(
        [10, 5], max_score=envs.SCHEDULER_SCALE_DOWN_STATUS_MAX_SCORE
    )
    scorer_b = DummyInstanceScorer(
        [30, 0], max_score=envs.SCHEDULER_SCALE_DOWN_OFFLOAD_MAX_SCORE
    )

    chain = ModelInstanceScoreChain(
        scorers=[scorer_a, scorer_b],
        total_max_score=20,
    )

    scored = await chain.score(instances)

    assert scored[0].score == pytest.approx(7.2727273)
    assert scored[1].score == pytest.approx(0.9090909)


@pytest.mark.asyncio
async def test_instance_score_chain_with_real_scorers():
    worker1 = linux_nvidia_19_4090_24gx2()
    worker2 = linux_nvidia_2_4080_16gx2()
    worker1.state = WorkerStateEnum.READY
    worker2.state = WorkerStateEnum.NOT_READY

    model = new_model(
        1,
        "test",
        placement_strategy=PlacementStrategyEnum.BINPACK,
        huggingface_repo_id="a/b",
    )

    instances = [
        new_model_instance(
            1,
            "i1",
            model.id,
            worker_id=worker1.id,
            state=ModelInstanceStateEnum.RUNNING,
            computed_resource_claim=ComputedResourceClaim(
                ram=100,
                vram={},
                offload_layers=10,
                total_layers=10,
            ),
        ),
        new_model_instance(
            2,
            "i2",
            model.id,
            worker_id=worker2.id,
            state=ModelInstanceStateEnum.RUNNING,
            computed_resource_claim=ComputedResourceClaim(
                ram=50,
                vram={},
                offload_layers=0,
                total_layers=10,
            ),
        ),
    ]

    mock_session = AsyncMock()
    mock_async_session = AsyncMock()
    mock_async_session.__aenter__.return_value = mock_session

    def allocatable_side_effect(_, worker, gpu_type=None):
        if worker.id == worker1.id:
            return Allocatable(ram=1000, vram={})
        return Allocatable(ram=2000, vram={})

    with (
        patch(
            "gpustack.policies.scorers.status_scorer.async_session",
            return_value=mock_async_session,
        ),
        patch(
            "gpustack.policies.scorers.placement_scorer.get_worker_allocatable_resource",
            side_effect=allocatable_side_effect,
        ),
        patch(
            "gpustack.policies.scorers.placement_scorer.async_session",
            return_value=mock_async_session,
        ),
        patch(
            "gpustack.policies.scorers.status_scorer.Worker.all",
            new=AsyncMock(return_value=[worker1, worker2]),
        ),
        patch(
            "gpustack.policies.scorers.placement_scorer.Worker.all",
            new=AsyncMock(return_value=[worker1, worker2]),
        ),
    ):
        status_scorer = StatusScorer(
            model, max_score=envs.SCHEDULER_SCALE_DOWN_STATUS_MAX_SCORE
        )
        offload_scorer = OffloadLayerScorer(
            model, max_score=envs.SCHEDULER_SCALE_DOWN_OFFLOAD_MAX_SCORE
        )
        placement_scorer = PlacementScorer(
            model,
            instances,
            scale_type=ScaleTypeEnum.SCALE_DOWN,
            max_score=envs.SCHEDULER_SCALE_DOWN_PLACEMENT_MAX_SCORE,
        )

        chain_scores = await ModelInstanceScoreChain(
            scorers=[status_scorer, offload_scorer, placement_scorer],
            total_max_score=None,
        ).score(instances)

    chain_map = _scores_by_instance(chain_scores)

    assert chain_map[instances[0].id] == pytest.approx(110.0909091)
    assert chain_map[instances[1].id] == pytest.approx(0.0243902439)
