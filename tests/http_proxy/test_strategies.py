import pytest

from gpustack.http_proxy.strategies import RoundRobinStrategy
from gpustack.schemas.models import ModelInstance


@pytest.fixture
def strategy():
    return RoundRobinStrategy()


@pytest.fixture
def instances():
    return [
        ModelInstance(id=1, model_id=10, worker_ip="10.0.0.1", port=8000),
        ModelInstance(id=2, model_id=10, worker_ip="10.0.0.2", port=8000),
        ModelInstance(id=3, model_id=10, worker_ip="10.0.0.3", port=8000),
    ]


class TestRoundRobinStrategy:

    @pytest.mark.asyncio
    async def test_distributes_requests_across_instances(self, strategy, instances):
        """Three consecutive requests should hit three different instances."""
        selected = []
        for _ in range(3):
            inst = await strategy.select_instance(instances)
            selected.append(inst.id)

        assert len(set(selected)) == 3

    @pytest.mark.asyncio
    async def test_round_robin_order(self, strategy, instances):
        """Requests should cycle through instances in order."""
        selected_ids = []
        for _ in range(6):
            inst = await strategy.select_instance(instances)
            selected_ids.append(inst.id)

        assert selected_ids == [1, 2, 3, 1, 2, 3]

    @pytest.mark.asyncio
    async def test_single_instance(self, strategy):
        """With one instance, all requests go to it."""
        instances = [
            ModelInstance(id=1, model_id=10, worker_ip="10.0.0.1", port=8000),
        ]
        for _ in range(3):
            inst = await strategy.select_instance(instances)
            assert inst.id == 1

    @pytest.mark.asyncio
    async def test_empty_instances_raises(self, strategy):
        """Empty list should raise an exception."""
        with pytest.raises(Exception, match="No instances available"):
            await strategy.select_instance([])

    @pytest.mark.asyncio
    async def test_iterator_resets_on_instance_list_change(self, strategy):
        """When the instance list changes, the iterator should be recreated."""
        instances_v1 = [
            ModelInstance(id=1, model_id=10, worker_ip="10.0.0.1", port=8000),
            ModelInstance(id=2, model_id=10, worker_ip="10.0.0.2", port=8000),
        ]

        # Consume one, next should be id=2
        inst = await strategy.select_instance(instances_v1)
        assert inst.id == 1

        # Now change the list (add a new instance)
        instances_v2 = [
            ModelInstance(id=1, model_id=10, worker_ip="10.0.0.1", port=8000),
            ModelInstance(id=2, model_id=10, worker_ip="10.0.0.2", port=8000),
            ModelInstance(id=3, model_id=10, worker_ip="10.0.0.3", port=8000),
        ]

        # Iterator should restart from the beginning
        inst = await strategy.select_instance(instances_v2)
        assert inst.id == 1

    @pytest.mark.asyncio
    async def test_iterator_resets_on_instance_removal(self, strategy):
        """When an instance is removed, the iterator should be recreated."""
        instances_v1 = [
            ModelInstance(id=1, model_id=10, worker_ip="10.0.0.1", port=8000),
            ModelInstance(id=2, model_id=10, worker_ip="10.0.0.2", port=8000),
            ModelInstance(id=3, model_id=10, worker_ip="10.0.0.3", port=8000),
        ]

        inst = await strategy.select_instance(instances_v1)
        assert inst.id == 1

        # Remove instance id=3
        instances_v2 = [
            ModelInstance(id=1, model_id=10, worker_ip="10.0.0.1", port=8000),
            ModelInstance(id=2, model_id=10, worker_ip="10.0.0.2", port=8000),
        ]

        inst = await strategy.select_instance(instances_v2)
        assert inst.id == 1

    @pytest.mark.asyncio
    async def test_different_models_have_separate_iterators(self, strategy):
        """Different model_ids should maintain independent round-robin state."""
        instances_model_a = [
            ModelInstance(id=1, model_id=10, worker_ip="10.0.0.1", port=8000),
            ModelInstance(id=2, model_id=10, worker_ip="10.0.0.2", port=8000),
        ]
        instances_model_b = [
            ModelInstance(id=3, model_id=20, worker_ip="10.0.0.3", port=8000),
            ModelInstance(id=4, model_id=20, worker_ip="10.0.0.4", port=8000),
        ]

        # Request for model A
        inst_a1 = await strategy.select_instance(instances_model_a)
        assert inst_a1.id == 1

        # Request for model B (should start independently)
        inst_b1 = await strategy.select_instance(instances_model_b)
        assert inst_b1.id == 3

        # Next request for model A
        inst_a2 = await strategy.select_instance(instances_model_a)
        assert inst_a2.id == 2

        # Next request for model B
        inst_b2 = await strategy.select_instance(instances_model_b)
        assert inst_b2.id == 4

    @pytest.mark.asyncio
    async def test_same_ids_different_objects_no_reset(self, strategy):
        """
        Passing a new list with the same instance IDs should NOT reset the
        iterator. This is the key regression test for the original bug where
        comparing ModelInstance objects caused the iterator to reset every call.
        """
        instances_v1 = [
            ModelInstance(id=1, model_id=10, worker_ip="10.0.0.1", port=8000),
            ModelInstance(id=2, model_id=10, worker_ip="10.0.0.2", port=8000),
        ]

        inst = await strategy.select_instance(instances_v1)
        assert inst.id == 1

        # Create a new list with the same IDs (different object instances)
        instances_v2 = [
            ModelInstance(id=1, model_id=10, worker_ip="10.0.0.1", port=8000),
            ModelInstance(id=2, model_id=10, worker_ip="10.0.0.2", port=8000),
        ]

        # Iterator should NOT reset; next should be id=2
        inst = await strategy.select_instance(instances_v2)
        assert inst.id == 2
