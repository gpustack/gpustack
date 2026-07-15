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

    @pytest.mark.asyncio
    async def test_returns_fresh_instance_on_updated_attributes(self, strategy):
        """
        When instance attributes change (e.g. port, worker_ip), the strategy
        should return the fresh object from the input list, not a stale cached
        one. This is the regression test for the Gemini review comment.
        """
        instances_v1 = [
            ModelInstance(id=1, model_id=10, worker_ip="10.0.0.1", port=8000),
            ModelInstance(id=2, model_id=10, worker_ip="10.0.0.2", port=8000),
        ]

        inst = await strategy.select_instance(instances_v1)
        assert inst.id == 1
        assert inst.port == 8000

        # Create a new list with the same IDs but updated port
        instances_v2 = [
            ModelInstance(id=1, model_id=10, worker_ip="10.0.0.1", port=8001),
            ModelInstance(id=2, model_id=10, worker_ip="10.0.0.2", port=8001),
        ]

        # Next round-robin pick should be id=2 with the updated port
        inst = await strategy.select_instance(instances_v2)
        assert inst.id == 2
        assert inst.port == 8001

    @pytest.mark.asyncio
    async def test_reordered_instance_list_does_not_reset_iterator(self, strategy):
        """
        When the same instances are passed in a different order (e.g. due to
        no ORDER BY in the DB query), the iterator should NOT reset.
        This is the regression test for the DB ordering issue.
        """
        instances_v1 = [
            ModelInstance(id=1, model_id=10, worker_ip="10.0.0.1", port=8000),
            ModelInstance(id=2, model_id=10, worker_ip="10.0.0.2", port=8000),
            ModelInstance(id=3, model_id=10, worker_ip="10.0.0.3", port=8000),
        ]

        # First call: should return id=1 (sorted order)
        inst = await strategy.select_instance(instances_v1)
        assert inst.id == 1

        # Second call with instances in reversed order — should NOT reset
        instances_reversed = [
            ModelInstance(id=3, model_id=10, worker_ip="10.0.0.3", port=8000),
            ModelInstance(id=2, model_id=10, worker_ip="10.0.0.2", port=8000),
            ModelInstance(id=1, model_id=10, worker_ip="10.0.0.1", port=8000),
        ]
        inst = await strategy.select_instance(instances_reversed)
        assert inst.id == 2

        # Third call with original order again
        inst = await strategy.select_instance(instances_v1)
        assert inst.id == 3

    @pytest.mark.asyncio
    async def test_round_robin_survives_random_ordering(self, strategy):
        """
        Simulate realistic DB behavior where instances come back in arbitrary
        order. Round-robin should still cycle through all instances correctly.
        """

        def make_instances(order):
            base = {
                1: ("10.0.0.1", 8000),
                2: ("10.0.0.2", 8000),
                3: ("10.0.0.3", 8000),
            }
            return [
                ModelInstance(
                    id=idx, model_id=10, worker_ip=base[idx][0], port=base[idx][1]
                )
                for idx in order
            ]

        selected_ids = []
        orders = [
            [1, 2, 3],
            [3, 1, 2],
            [2, 3, 1],
            [1, 3, 2],
            [3, 2, 1],
            [2, 1, 3],
        ]
        for order in orders:
            inst = await strategy.select_instance(make_instances(order))
            selected_ids.append(inst.id)

        # Should cycle 1, 2, 3 regardless of input order
        assert selected_ids == [1, 2, 3, 1, 2, 3]
