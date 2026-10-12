from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gpustack.scheduler import evaluator
from gpustack.schemas.model_sets import ModelSpec
from gpustack.schemas.models import SourceEnum


@pytest.mark.asyncio
async def test_cached_evaluation_uses_the_same_reservations_as_its_cache_key(
    monkeypatch,
):
    first = [SimpleNamespace(worker_id=1, computed_resource_claim={"ram": 10})]
    changed = [SimpleNamespace(worker_id=1, computed_resource_claim={"ram": 20})]
    model = ModelSpec(source=SourceEnum.LOCAL_PATH, local_path="/models/model")
    evaluate = AsyncMock(return_value=evaluator.ModelEvaluationResult())
    monkeypatch.setattr(evaluator, "evaluate_cache", {})
    monkeypatch.setattr(
        evaluator, "visible_cache_service", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        evaluator,
        "cache_instances_in",
        AsyncMock(
            side_effect=[first, first, changed],
        ),
    )
    monkeypatch.setattr(evaluator, "evaluate_model", evaluate)

    workers = [SimpleNamespace(id=1, model_dump=lambda **kwargs: {"id": 1})]
    for _ in range(3):
        await evaluator.evaluate_model_with_cache(
            SimpleNamespace(), None, model, workers, []
        )

    assert evaluator.cache_instances_in.call_args.kwargs["worker_ids"] == [1]
    assert evaluate.await_count == 2
    for call, rows, ram in zip(evaluate.await_args_list, [first, changed], [10, 20]):
        assert call.kwargs["cache_instances"] is rows
        assert call.kwargs["resource_view"].allocated(1).ram == ram


@pytest.mark.asyncio
@pytest.mark.parametrize("cluster_id", [None, 3])
async def test_cache_reservations_are_scoped_to_evaluation_workers(
    monkeypatch, cluster_id
):
    from gpustack.scheduler.group_schedule import cache_instances_in
    from gpustack.schemas.cache_services import CacheServiceInstance

    rows = AsyncMock(return_value=[])
    monkeypatch.setattr(CacheServiceInstance, "all_by_fields", rows)
    await cache_instances_in(object(), cluster_id, worker_ids=[1, 2])
    rows.assert_awaited_once()
    assert rows.call_args.args[1] == ({"cluster_id": 3} if cluster_id else {})
    condition = rows.call_args.kwargs["extra_conditions"][0]
    assert next(iter(condition.compile().params.values())) == [1, 2]
    rows.reset_mock()
    assert await cache_instances_in(object(), cluster_id, worker_ids=[]) == []
    rows.assert_not_awaited()
