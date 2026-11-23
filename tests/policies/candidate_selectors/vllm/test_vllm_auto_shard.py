"""Automatic MP placement shards one replica when DP does not fit a node."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from gpustack.policies.base import Allocatable, ModelInstanceScheduleCandidate
from gpustack.policies.candidate_selectors.vllm_resource_fit_selector import (
    VLLMResourceFitSelector,
)
from gpustack.policies.resource_view import ResourceView
from gpustack.schemas.models import (
    ComputedResourceClaim,
    ModelInstanceSubordinateWorker,
)

_GIB = 1024**3


def _selector():
    selector = VLLMResourceFitSelector.__new__(VLLMResourceFitSelector)
    selector._model = SimpleNamespace(
        name="qwen3-vl-8b-instruct",
        backend_parameters=[],
        backend_version=None,
        image_name="example/runner:cuda",
        env=None,
        extended_kv_cache=None,
    )
    selector._event_collector = MagicMock()
    selector._is_tp_size_divisible = lambda size: size in (1, 2)
    selector._gpu_memory_utilization = 0.9
    return selector


def test_selector_records_shard_on_the_candidate():
    selector = _selector()
    selector._vram_claim = 16 * _GIB
    candidate = ModelInstanceScheduleCandidate(
        worker=MagicMock(),
        gpu_indexes=[0],
        computed_resource_claim=ComputedResourceClaim(vram={0: 10 * _GIB}),
        subordinate_workers=[
            ModelInstanceSubordinateWorker(
                gpu_indexes=[0],
                computed_resource_claim=ComputedResourceClaim(vram={0: 10 * _GIB}),
            )
        ],
    )
    kept = selector._shard_when_dp_replica_does_not_fit([candidate])
    assert kept == [candidate]
    assert candidate.auto_parallelism.tp == 2
    assert candidate.computed_resource_claim.vram == {0: 10 * _GIB}
    assert candidate.auto_parallelism.pp is None


def test_selector_shards_when_non_llm_reservation_is_zero():
    """A zero reservation is not a capacity. Non-LLM admission uses the card."""
    selector = _selector()
    selector._model.name = "embedding"
    selector._vram_claim = 16 * _GIB
    selector._gpu_memory_utilization = 0
    selector._workers_by_id = {2: SimpleNamespace(id=2)}
    selector.get_worker_allocatable_resource = lambda worker, gpu_type=None: (
        Allocatable(ram=0, vram={0: 12 * _GIB})
    )
    candidate = ModelInstanceScheduleCandidate(
        worker=SimpleNamespace(id=1),
        gpu_indexes=[0],
        gpu_type="cuda",
        computed_resource_claim=ComputedResourceClaim(vram={0: 0}),
        subordinate_workers=[
            ModelInstanceSubordinateWorker(
                worker_id=2,
                gpu_indexes=[0],
                gpu_type="cuda",
                computed_resource_claim=ComputedResourceClaim(vram={0: 0}),
            )
        ],
    )
    kept = selector._shard_when_dp_replica_does_not_fit([candidate])
    assert kept == [candidate]
    assert candidate.auto_parallelism.tp == 2
    assert candidate.computed_resource_claim.vram == {0: 8 * _GIB}
    assert candidate.subordinate_workers[0].computed_resource_claim.vram == {
        0: 8 * _GIB
    }


def _one_gpu_worker(name, identity, total):
    return SimpleNamespace(
        id=identity,
        name=name,
        ip=f"10.0.0.{identity}",
        ifname="eth0",
        status=SimpleNamespace(
            memory=SimpleNamespace(total=64 * _GIB, is_unified_memory=False),
            gpu_devices=[
                SimpleNamespace(
                    index=0, type="cuda", memory=SimpleNamespace(total=total)
                )
            ],
        ),
        system_reserved=SimpleNamespace(ram=0, vram=0),
    )


def test_selector_replaces_a_prefix_whose_smallest_gpu_misses_its_share():
    """The first pair can sum past the claim and still lose to a later pair.

    Workers of 8, 24 and 24 GiB, model 30 GiB, TP width 2. 8+24 covers the
    sum and fails the 15 GiB share. The two 24 GiB workers hold that share.
    """
    selector = _selector()
    selector._gpu_memory_utilization = 1
    selector._gpu_count = None
    selector._vram_claim = 30 * _GIB
    selector._worker_ram_fits = lambda *args, **kwargs: True
    small = _one_gpu_worker("small", 1, 8 * _GIB)
    first = _one_gpu_worker("large-a", 2, 24 * _GIB)
    second = _one_gpu_worker("large-b", 3, 24 * _GIB)
    workers = [small, first, second]
    selector._workers_by_id = {worker.id: worker for worker in workers}
    selector.get_worker_allocatable_resource = lambda worker, gpu_type=None: (
        Allocatable(ram=0, vram={0: worker.status.gpu_devices[0].memory.total})
    )

    found = selector.auto_select_multi_worker_multi_gpu_candidates(workers, "cuda")
    assert len(found) == 1
    assert found[0].worker.name == "large-a"
    assert found[0].subordinate_workers[0].worker_name == "large-b"

    kept = selector._shard_when_dp_replica_does_not_fit(found)
    assert kept == found
    assert found[0].auto_parallelism.tp == 2


def test_selector_keeps_workers_for_a_wider_shard():
    """A worker rejected by TP2 remains eligible when TP4 lowers its share."""
    selector = _selector()
    selector._is_tp_size_divisible = lambda size: size in (1, 4)
    selector._gpu_memory_utilization = 1
    selector._gpu_count = None
    selector._vram_claim = 36 * _GIB
    selector._worker_ram_fits = lambda *args, **kwargs: True
    workers = [
        _one_gpu_worker("large", 1, 24 * _GIB),
        _one_gpu_worker("medium-a", 2, 16 * _GIB),
        _one_gpu_worker("medium-b", 3, 16 * _GIB),
        _one_gpu_worker("medium-c", 4, 16 * _GIB),
    ]
    selector._workers_by_id = {worker.id: worker for worker in workers}
    selector.get_worker_allocatable_resource = lambda worker, gpu_type=None: (
        Allocatable(ram=0, vram={0: worker.status.gpu_devices[0].memory.total})
    )

    found = selector.auto_select_multi_worker_multi_gpu_candidates(workers, "cuda")

    assert len(found) == 1
    assert found[0].worker.name == "large"
    assert len(found[0].subordinate_workers) == 3
    kept = selector._shard_when_dp_replica_does_not_fit(found)
    assert kept == found
    assert found[0].auto_parallelism.tp == 4


def test_non_llm_shard_claims_reduce_both_workers_allocatable_vram():
    """The persisted shard share is visible to the next scheduling view."""
    selector = _selector()
    selector._model.name = "embedding"
    selector._vram_claim = 24 * _GIB
    selector._gpu_memory_utilization = 0
    leader = _one_gpu_worker("leader", 1, 16 * _GIB)
    follower = _one_gpu_worker("follower", 2, 16 * _GIB)
    selector._workers_by_id = {2: follower}
    selector.get_worker_allocatable_resource = lambda worker, gpu_type=None: (
        Allocatable(ram=0, vram={0: 16 * _GIB})
    )
    candidate = ModelInstanceScheduleCandidate(
        worker=leader,
        gpu_indexes=[0],
        gpu_type="cuda",
        computed_resource_claim=ComputedResourceClaim(vram={0: 0}),
        subordinate_workers=[
            ModelInstanceSubordinateWorker(
                worker_id=2,
                gpu_indexes=[0],
                gpu_type="cuda",
                computed_resource_claim=ComputedResourceClaim(vram={0: 0}),
            )
        ],
    )

    assert selector._shard_when_dp_replica_does_not_fit([candidate]) == [candidate]
    instance = SimpleNamespace(
        worker_id=1,
        computed_resource_claim=candidate.computed_resource_claim,
        gpu_indexes=[0],
        gpu_type="cuda",
        distributed_servers=SimpleNamespace(
            subordinate_workers=candidate.subordinate_workers
        ),
    )
    view = ResourceView([instance])

    assert view.allocatable(leader, "cuda").vram == {0: 4 * _GIB}
    assert view.allocatable(follower, "cuda").vram == {0: 4 * _GIB}
