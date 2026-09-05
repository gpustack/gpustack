import pytest

from gpustack.http_proxy.strategies import RoundRobinStrategy
from gpustack.schemas.models import ModelInstance


def make_instances(ids, model_id=10, restart_count=0):
    return [
        ModelInstance(
            id=i,
            model_id=model_id,
            worker_ip=f"10.0.0.{i}",
            port=8000,
            restart_count=restart_count,
        )
        for i in ids
    ]


@pytest.mark.asyncio
async def test_cycles_when_each_call_passes_fresh_objects():
    strategy = RoundRobinStrategy()
    selected = []
    for call in range(6):
        inst = await strategy.select_instance(make_instances([1, 2, 3], restart_count=call))
        selected.append(inst.id)
    assert selected == [1, 2, 3, 1, 2, 3]


@pytest.mark.asyncio
async def test_returns_object_from_current_list():
    strategy = RoundRobinStrategy()
    instances = make_instances([1, 2])
    first = await strategy.select_instance(instances)
    assert first is instances[0]
    instances = make_instances([1, 2])
    second = await strategy.select_instance(instances)
    assert second is instances[1]


@pytest.mark.asyncio
async def test_resets_when_membership_changes():
    strategy = RoundRobinStrategy()
    assert (await strategy.select_instance(make_instances([1, 2]))).id == 1
    assert (await strategy.select_instance(make_instances([1, 2, 3]))).id == 1
    assert (await strategy.select_instance(make_instances([1, 2, 3]))).id == 2
    assert (await strategy.select_instance(make_instances([2, 3]))).id == 2


@pytest.mark.asyncio
async def test_models_are_tracked_separately():
    strategy = RoundRobinStrategy()
    assert (await strategy.select_instance(make_instances([1, 2], model_id=10))).id == 1
    assert (await strategy.select_instance(make_instances([7, 8], model_id=20))).id == 7
    assert (await strategy.select_instance(make_instances([1, 2], model_id=10))).id == 2
    assert (await strategy.select_instance(make_instances([7, 8], model_id=20))).id == 8


@pytest.mark.asyncio
async def test_single_instance():
    strategy = RoundRobinStrategy()
    for _ in range(3):
        assert (await strategy.select_instance(make_instances([1]))).id == 1


@pytest.mark.asyncio
async def test_empty_list_raises():
    strategy = RoundRobinStrategy()
    with pytest.raises(Exception, match="No instances available"):
        await strategy.select_instance([])
