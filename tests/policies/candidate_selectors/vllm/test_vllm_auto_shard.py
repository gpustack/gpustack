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
    ExtendedKVCacheConfig,
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
    selector._offloaded_ram = 0
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


def _two_gpu_worker(name, identity):
    worker = _one_gpu_worker(name, identity, 16 * _GIB)
    worker.status.gpu_devices.append(
        SimpleNamespace(index=1, type="cuda", memory=SimpleNamespace(total=16 * _GIB))
    )
    return worker


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


def test_fixed_gpu_count_does_not_fall_back_to_one_worker():
    """Multi-worker fallback requires at least two workers."""
    selector = _selector()
    selector._gpu_memory_utilization = 0
    selector._gpu_count = 2
    selector._vram_claim = 24 * _GIB
    selector._worker_ram_fits = lambda *args, **kwargs: True
    workers = [_two_gpu_worker("first", 1), _two_gpu_worker("second", 2)]
    selector._workers_by_id = {worker.id: worker for worker in workers}
    selector.get_worker_allocatable_resource = lambda worker, gpu_type=None: (
        Allocatable(ram=64 * _GIB, vram={0: 10 * _GIB, 1: 10 * _GIB})
    )

    found = selector.auto_select_multi_worker_multi_gpu_candidates(workers, "cuda")

    assert found == []


def test_non_llm_shard_requires_every_gpu_to_hold_its_share():
    """Spare VRAM on one GPU cannot cover a shortfall on another GPU."""
    selector = _selector()
    selector._is_tp_size_divisible = lambda size: size in (1, 2, 4)
    selector._gpu_memory_utilization = 0
    selector._gpu_count = None
    selector._vram_claim = 24 * _GIB
    selector._worker_ram_fits = lambda *args, **kwargs: True
    workers = [_two_gpu_worker("first", 1), _two_gpu_worker("second", 2)]
    selector._workers_by_id = {worker.id: worker for worker in workers}
    selector.get_worker_allocatable_resource = lambda worker, gpu_type=None: (
        Allocatable(ram=64 * _GIB, vram={0: 15 * _GIB, 1: 1 * _GIB})
    )

    found = selector.auto_select_multi_worker_multi_gpu_candidates(workers, "cuda")

    assert found == []


def test_non_llm_shard_checks_ratio_cache_ram_from_final_vram_claim():
    """A ratio-sized local cache is admitted from the persisted shard claim."""
    selector = _selector()
    selector._model.name = "embedding"
    selector._model.extended_kv_cache = ExtendedKVCacheConfig(enabled=True, ram_ratio=1)
    selector._gpu_memory_utilization = 0
    selector._vram_claim = 24 * _GIB
    leader = _one_gpu_worker("leader", 1, 16 * _GIB)
    follower = _one_gpu_worker("follower", 2, 16 * _GIB)
    selector._workers_by_id = {2: follower}
    selector.get_worker_allocatable_resource = lambda worker, gpu_type=None: (
        Allocatable(ram=8 * _GIB, vram={0: 16 * _GIB})
    )
    candidate = ModelInstanceScheduleCandidate(
        worker=leader,
        gpu_indexes=[0],
        gpu_type="cuda",
        computed_resource_claim=ComputedResourceClaim(vram={0: 0}),
        subordinate_workers=[
            ModelInstanceSubordinateWorker(
                worker_id=2,
                worker_name="follower",
                gpu_indexes=[0],
                gpu_type="cuda",
                computed_resource_claim=ComputedResourceClaim(vram={0: 0}),
            )
        ],
    )

    assert selector._shard_when_dp_replica_does_not_fit([candidate]) == []
    assert candidate.computed_resource_claim.ram == 12 * _GIB


def test_selector_skips_high_total_worker_with_one_undersized_gpu():
    """Each width can choose a feasible subset without its highest-total worker."""
    selector = _selector()
    selector._is_tp_size_divisible = lambda size: size in (1, 2, 4, 6)
    selector._gpu_memory_utilization = 0
    selector._gpu_count = None
    selector._vram_claim = 24 * _GIB
    selector._worker_ram_fits = lambda *args, **kwargs: True
    high_total = _two_gpu_worker("uneven", 1)
    first = _two_gpu_worker("balanced-a", 2)
    second = _two_gpu_worker("balanced-b", 3)
    workers = [high_total, first, second]
    selector._workers_by_id = {worker.id: worker for worker in workers}
    budgets = {
        1: {0: 22 * _GIB, 1: 1 * _GIB},
        2: {0: 10 * _GIB, 1: 10 * _GIB},
        3: {0: 10 * _GIB, 1: 10 * _GIB},
    }
    selector.get_worker_allocatable_resource = lambda worker, gpu_type=None: (
        Allocatable(ram=64 * _GIB, vram=budgets[worker.id])
    )

    found = selector.auto_select_multi_worker_multi_gpu_candidates(workers, "cuda")

    assert len(found) == 1
    assert found[0].worker.name == "balanced-a"
    assert found[0].subordinate_workers[0].worker_name == "balanced-b"


def test_manual_single_replica_shard_preserves_ram_overcommit():
    """Final shard RAM can overcommit when a manual single replica permits it."""
    selector = _selector()
    selector._model.name = "embedding"
    selector._model.replicas = 1
    selector._model.env = {"GPUSTACK_MODEL_RAM_CLAIM": str(20 * _GIB)}
    selector._gpu_memory_utilization = 0
    selector._vram_claim = 24 * _GIB
    leader = _one_gpu_worker("leader", 1, 16 * _GIB)
    follower = _one_gpu_worker("follower", 2, 16 * _GIB)
    selector._selected_gpu_workers = [leader, follower]
    selector._workers_by_id = {2: follower}
    selector.get_worker_allocatable_resource = lambda worker, gpu_type=None: (
        Allocatable(ram=18 * _GIB, vram={0: 16 * _GIB})
    )
    candidate = ModelInstanceScheduleCandidate(
        worker=leader,
        gpu_indexes=[0],
        gpu_type="cuda",
        computed_resource_claim=ComputedResourceClaim(vram={0: 0}),
        subordinate_workers=[
            ModelInstanceSubordinateWorker(
                worker_id=2,
                worker_name="follower",
                gpu_indexes=[0],
                gpu_type="cuda",
                computed_resource_claim=ComputedResourceClaim(vram={0: 0}),
            )
        ],
        overcommit=True,
    )

    kept = selector._shard_when_dp_replica_does_not_fit([candidate])

    assert kept == [candidate]
    assert candidate.auto_parallelism.tp == 2
    assert candidate.computed_resource_claim.ram == 20 * _GIB
    assert candidate.subordinate_workers[0].computed_resource_claim.ram == 20 * _GIB
    assert candidate.overcommit is True


def test_explicit_pipeline_keeps_its_uneven_worker_budgets():
    """Pinned PP reaches topology validation without automatic TP share filters."""
    selector = _selector()
    selector._model.backend_parameters = ["--pipeline-parallel-size=2"]
    selector._gpu_memory_utilization = 0
    selector._gpu_count = 2
    selector._vram_claim = 24 * _GIB
    selector._worker_ram_fits = lambda *args, **kwargs: True
    first = _one_gpu_worker("stage-a", 1, 20 * _GIB)
    second = _one_gpu_worker("stage-b", 2, 8 * _GIB)
    workers = [first, second]
    selector._workers_by_id = {worker.id: worker for worker in workers}
    budgets = {1: {0: 20 * _GIB}, 2: {0: 8 * _GIB}}
    selector.get_worker_allocatable_resource = lambda worker, gpu_type=None: (
        Allocatable(ram=64 * _GIB, vram=budgets[worker.id])
    )

    found = selector.auto_select_multi_worker_multi_gpu_candidates(workers, "cuda")

    assert len(found) == 1
    assert found[0].worker.name == "stage-a"
    assert found[0].subordinate_workers[0].worker_name == "stage-b"
    assert found[0].auto_parallelism is None


def test_automatic_shard_reports_projected_cache_ram_shortage():
    """RAM filtered before candidate creation remains visible in scheduling output."""
    selector = _selector()
    selector._model.name = "embedding"
    selector._model.extended_kv_cache = ExtendedKVCacheConfig(enabled=True, ram_ratio=1)
    selector._gpu_memory_utilization = 0
    selector._gpu_count = None
    selector._vram_claim = 24 * _GIB
    selector._worker_ram_fits = lambda *args, **kwargs: True
    workers = [
        _one_gpu_worker("first", 1, 16 * _GIB),
        _one_gpu_worker("second", 2, 16 * _GIB),
    ]
    selector._workers_by_id = {worker.id: worker for worker in workers}
    selector.get_worker_allocatable_resource = lambda worker, gpu_type=None: (
        Allocatable(ram=8 * _GIB, vram={0: 16 * _GIB})
    )

    found = selector.auto_select_multi_worker_multi_gpu_candidates(workers, "cuda")

    assert found == []
    messages = " ".join(
        str(call.args[-1]) for call in selector._event_collector.add.call_args_list
    )
    assert "requires 12.0 GiB of RAM" in messages
    assert "only 8.0 GiB is allocatable" in messages


def test_ray_local_dp_hint_uses_a_wider_pipeline_for_available_vram():
    """A DPL hint does not let installed VRAM admit an undersized Ray pipeline."""
    selector = _selector()
    selector._model.backend_parameters = ["--data-parallel-size-local=1"]
    selector._model.image_name = None
    selector._gpu_memory_utilization = 0
    selector._gpu_count = None
    selector._vram_claim = 42 * _GIB
    selector._worker_ram_fits = lambda *args, **kwargs: True
    workers = [
        _one_gpu_worker("first", 1, 20 * _GIB),
        _one_gpu_worker("second", 2, 20 * _GIB),
        _one_gpu_worker("third", 3, 32 * _GIB),
    ]
    selector._workers_by_id = {worker.id: worker for worker in workers}
    budgets = {
        1: {0: 15 * _GIB},
        2: {0: 15 * _GIB},
        3: {0: 16 * _GIB},
    }
    selector.get_worker_allocatable_resource = lambda worker, gpu_type=None: (
        Allocatable(ram=64 * _GIB, vram=budgets[worker.id])
    )

    found = selector.auto_select_multi_worker_multi_gpu_candidates(workers, "cuda")

    assert len(found) == 1
    assert found[0].worker.name == "third"
    assert len(found[0].subordinate_workers) == 2
    assert {worker.worker_name for worker in found[0].subordinate_workers} == {
        "first",
        "second",
    }


def test_pinned_tensor_parallelism_still_requires_each_gpu_share():
    """Only explicit PP may use unequal stage budgets."""
    selector = _selector()
    selector._model.backend_parameters = ["--tensor-parallel-size=2"]
    selector._gpu_memory_utilization = 0
    selector._gpu_count = 2
    selector._vram_claim = 24 * _GIB
    selector._worker_ram_fits = lambda *args, **kwargs: True
    workers = [
        _one_gpu_worker("large", 1, 32 * _GIB),
        _one_gpu_worker("small", 2, 32 * _GIB),
    ]
    selector._workers_by_id = {worker.id: worker for worker in workers}
    budgets = {1: {0: 20 * _GIB}, 2: {0: 8 * _GIB}}
    selector.get_worker_allocatable_resource = lambda worker, gpu_type=None: (
        Allocatable(ram=64 * _GIB, vram=budgets[worker.id])
    )

    found = selector.auto_select_multi_worker_multi_gpu_candidates(workers, "cuda")

    assert found == []


def test_fixed_tp_retries_with_a_later_worker_at_the_same_width():
    """Requested TP width constrains the subset without closing its worker pool."""
    selector = _selector()
    selector._model.backend_parameters = ["--tensor-parallel-size=2"]
    selector._gpu_memory_utilization = 0.9
    selector._gpu_count = 2
    selector._vram_claim = 36 * _GIB
    selector._worker_ram_fits = lambda *args, **kwargs: True
    workers = [
        _one_gpu_worker("large-a", 1, 32 * _GIB),
        _one_gpu_worker("small", 2, 16 * _GIB),
        _one_gpu_worker("large-b", 3, 32 * _GIB),
    ]
    selector._workers_by_id = {worker.id: worker for worker in workers}
    selector.get_worker_allocatable_resource = lambda worker, gpu_type=None: (
        Allocatable(
            ram=64 * _GIB,
            vram={0: worker.status.gpu_devices[0].memory.total},
        )
    )

    found = selector.auto_select_multi_worker_multi_gpu_candidates(workers, "cuda")

    assert len(found) == 1
    assert found[0].worker.name == "large-a"
    assert found[0].subordinate_workers[0].worker_name == "large-b"


def test_ray_inferred_pipeline_requires_each_stage_allowance():
    """Ray cannot use aggregate VRAM to compensate an undersized PP stage."""
    selector = _selector()
    selector._model.backend_parameters = ["--data-parallel-size-local=1"]
    selector._model.image_name = None
    selector._gpu_memory_utilization = 0
    selector._gpu_count = None
    selector._vram_claim = 24 * _GIB
    selector._worker_ram_fits = lambda *args, **kwargs: True
    first = _one_gpu_worker("stage-a", 1, 20 * _GIB)
    second = _one_gpu_worker("stage-b", 2, 8 * _GIB)
    workers = [first, second]
    selector._workers_by_id = {worker.id: worker for worker in workers}
    budgets = {1: {0: 20 * _GIB}, 2: {0: 8 * _GIB}}
    selector.get_worker_allocatable_resource = lambda worker, gpu_type=None: (
        Allocatable(ram=64 * _GIB, vram=budgets[worker.id])
    )

    found = selector.auto_select_multi_worker_multi_gpu_candidates(workers, "cuda")

    assert found == []


def test_fixed_width_failure_message_uses_only_the_allowed_subset():
    """Capacity shown for TP2 excludes extra workers scanned as replacements."""
    selector = _selector()
    selector._model.backend_parameters = ["--tensor-parallel-size=2"]
    selector._gpu_memory_utilization = 0.9
    selector._gpu_count = 2
    selector._vram_claim = 36 * _GIB
    selector._worker_ram_fits = lambda *args, **kwargs: True
    workers = [
        _one_gpu_worker("large", 1, 32 * _GIB),
        _one_gpu_worker("small-a", 2, 16 * _GIB),
        _one_gpu_worker("small-b", 3, 16 * _GIB),
    ]
    selector._workers_by_id = {worker.id: worker for worker in workers}
    selector.get_worker_allocatable_resource = lambda worker, gpu_type=None: (
        Allocatable(
            ram=64 * _GIB,
            vram={0: worker.status.gpu_devices[0].memory.total},
        )
    )

    found = selector.auto_select_multi_worker_multi_gpu_candidates(workers, "cuda")

    assert found == []
    messages = " ".join(
        str(call.args[-1]) for call in selector._event_collector.add.call_args_list
    )
    assert "['large', 'small-a'] provides 43.2 GiB" in messages
    assert "small-b" not in messages
    assert "fail to meet requirements" not in messages


def test_zero_utilization_failure_capacity_uses_available_vram():
    """A freer replacement cannot reduce the capacity shown to the user."""
    selector = _selector()
    selector._gpu_memory_utilization = 0
    selector._gpu_count = 2
    selector._vram_claim = 200 * _GIB
    selector._worker_ram_fits = lambda *args, **kwargs: True
    workers = [
        _one_gpu_worker("large-a", 1, 80 * _GIB),
        _one_gpu_worker("large-b", 2, 80 * _GIB),
        _one_gpu_worker("small-free", 3, 24 * _GIB),
    ]
    selector._workers_by_id = {worker.id: worker for worker in workers}
    budgets = {1: 10 * _GIB, 2: 9 * _GIB, 3: 20 * _GIB}
    selector.get_worker_allocatable_resource = lambda worker, gpu_type=None: (
        Allocatable(ram=64 * _GIB, vram={0: budgets[worker.id]})
    )

    found = selector.auto_select_multi_worker_multi_gpu_candidates(workers, "cuda")
    messages = " ".join(
        str(call.args[-1]) for call in selector._event_collector.add.call_args_list
    )

    assert found == []
    assert "largest available worker has 30.0 GiB" in messages
    assert "104.0 GiB" not in messages
