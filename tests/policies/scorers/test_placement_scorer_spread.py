import pytest

from gpustack.policies.base import ModelInstanceScheduleCandidate
from gpustack.policies.scorers.placement_scorer import PlacementScorer
from gpustack.schemas.models import ComputedResourceClaim, PlacementStrategyEnum
from tests.fixtures.workers.fixtures import (
    linux_nvidia_1_4090_24gx1,
    linux_nvidia_2_4080_16gx2,
)
from tests.utils.model import new_model, new_model_instance


def make_candidate(worker, gpu_indexes):
    vram = {gpu_indexes[0]: 1} if gpu_indexes else None
    claim = ComputedResourceClaim(ram=1, vram=vram)
    return ModelInstanceScheduleCandidate(
        worker=worker,
        gpu_indexes=gpu_indexes,
        computed_resource_claim=claim,
    )


@pytest.mark.asyncio
async def test_spread_prefers_worker_with_zero_current_instances():
    w1 = linux_nvidia_1_4090_24gx1()
    w2 = linux_nvidia_2_4080_16gx2()

    model = new_model(
        1,
        "m",
        placement_strategy=PlacementStrategyEnum.SPREAD,
        huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct",
    )
    # One current model instance already on w2
    mis = [new_model_instance(1, "mi1", model.id, worker_id=w2.id, gpu_indexes=[0])]

    scorer = PlacementScorer(model, mis)
    candidates = [make_candidate(w1, [0]), make_candidate(w2, [0])]
    scored = await scorer.score(candidates)

    assert scored[0].score > scored[1].score


@pytest.mark.asyncio
async def test_spread_prefers_worker_with_fewer_current_instances_when_all_have_current():
    w1 = linux_nvidia_1_4090_24gx1()
    w2 = linux_nvidia_2_4080_16gx2()

    model = new_model(
        2,
        "m2",
        placement_strategy=PlacementStrategyEnum.SPREAD,
        huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct",
    )
    mis = [
        new_model_instance(1, "mi1", model.id, worker_id=w1.id, gpu_indexes=[0]),
        new_model_instance(2, "mi2", model.id, worker_id=w2.id, gpu_indexes=[0]),
        new_model_instance(3, "mi3", model.id, worker_id=w2.id, gpu_indexes=[1]),
    ]

    scorer = PlacementScorer(model, mis)
    candidates = [make_candidate(w1, [0]), make_candidate(w2, [0])]
    scored = await scorer.score(candidates)

    assert scored[0].score > scored[1].score


@pytest.mark.asyncio
async def test_spread_considers_other_model_instances_as_secondary_weight():
    w1 = linux_nvidia_1_4090_24gx1()
    w2 = linux_nvidia_2_4080_16gx2()

    model = new_model(
        3,
        "m3",
        placement_strategy=PlacementStrategyEnum.SPREAD,
        huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct",
    )
    other_model_id = 999

    mis = [
        new_model_instance(1, "mi1", model.id, worker_id=w1.id, gpu_indexes=[0]),
        new_model_instance(2, "mi2", model.id, worker_id=w2.id, gpu_indexes=[0]),
        new_model_instance(
            3,
            "mi_other1",
            other_model_id,
            worker_id=w2.id,
            gpu_indexes=[1],
        ),
        new_model_instance(
            4,
            "mi_other2",
            other_model_id,
            worker_id=w2.id,
            gpu_indexes=[1],
        ),
    ]

    scorer = PlacementScorer(model, mis)
    candidates = [make_candidate(w1, [0]), make_candidate(w2, [0])]
    scored = await scorer.score(candidates)

    assert scored[0].score > scored[1].score


@pytest.mark.asyncio
async def test_spread_prefers_gpu_with_fewer_instances_on_same_worker():
    w1 = linux_nvidia_2_4080_16gx2()

    model = new_model(
        4,
        "m4",
        placement_strategy=PlacementStrategyEnum.SPREAD,
        huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct",
    )
    mis = [
        new_model_instance(1, "mi1", model.id, worker_id=w1.id, gpu_indexes=[0]),
        new_model_instance(2, "mi2", model.id, worker_id=w1.id, gpu_indexes=[0]),
    ]

    scorer = PlacementScorer(model, mis)
    candidates = [make_candidate(w1, [0]), make_candidate(w1, [1])]
    scored = await scorer.score(candidates)

    assert scored[1].score > scored[0].score
