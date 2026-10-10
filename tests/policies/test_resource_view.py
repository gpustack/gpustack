from types import SimpleNamespace

import pytest

from gpustack.policies.candidate_selectors import (
    AscendMindIEResourceFitSelector,
    GGUFResourceFitSelector,
    SGLangResourceFitSelector,
    VGPUResourceFitSelector,
    VLLMResourceFitSelector,
)
from gpustack.policies.candidate_selectors.custom_backend_resource_fit_selector import (
    CustomBackendResourceFitSelector,
)
from gpustack.policies.candidate_selectors.instance_type_whole_card_selector import (
    InstanceTypeWholeCardSelector,
)
from gpustack.policies.base import Allocated
from gpustack.policies.resource_view import ResourceView
from gpustack.schemas.models import (
    ComputedResourceClaim,
    Model,
    SourceEnum,
    PlacementStrategyEnum,
)
from gpustack.schemas.workers import MemoryInfo, SystemReserved, Worker, WorkerStatus


def _instance(instance_id, ram=10, vram=20):
    return SimpleNamespace(
        id=instance_id,
        worker_id=1,
        gpu_type="cuda",
        gpu_indexes=[0],
        computed_resource_claim=ComputedResourceClaim(ram=ram, vram={0: vram}),
        distributed_servers=None,
    )


def _worker():
    return Worker(
        id=1,
        name="worker",
        hostname="worker",
        ip="127.0.0.1",
        status=WorkerStatus(memory=MemoryInfo(total=100)),
        system_reserved=SystemReserved(ram=10, vram=0),
    )


def _model():
    return Model(
        name="router",
        source=SourceEnum.LOCAL_PATH,
        local_path="/models/router",
        backend="custom",
        placement_strategy=PlacementStrategyEnum.BINPACK,
    )


def test_retry_excludes_main_and_subordinate_claims_but_retains_other_workloads():
    retry = _instance(1)
    retry.distributed_servers = SimpleNamespace(
        subordinate_workers=[
            SimpleNamespace(
                worker_id=2,
                computed_resource_claim=ComputedResourceClaim(
                    ram=0,
                    vram={1: 30},
                ),
            ),
        ]
    )
    view = ResourceView([retry, _instance(2)], {1: Allocated(ram=40, vram={})})
    placement = view.with_model_instances([_instance(2)])

    assert placement.allocated(1) == Allocated(ram=50, vram={0: 20})
    assert placement.allocated(2) == Allocated(ram=0, vram={})
    assert view.allocated(1) == Allocated(ram=60, vram={0: 40})
    assert view.allocated(2) == Allocated(ram=0, vram={1: 30})


def test_subordinate_claim_with_no_gpu_type_counts_against_every_gpu_type():
    # GGUF never sets gpu_type on its instances or their subordinate workers,
    # since llama-box places by raw VRAM rather than by a named GPU type. A
    # rpc-server's claim must still be visible when some other backend (which
    # does pass a concrete gpu_type, e.g. "cuda") asks what's free on that
    # worker — otherwise it schedules on top of memory the rpc-server holds.
    gguf = _instance(1)
    gguf.gpu_type = None
    gguf.distributed_servers = SimpleNamespace(
        subordinate_workers=[
            SimpleNamespace(
                worker_id=2,
                gpu_type=None,
                computed_resource_claim=ComputedResourceClaim(ram=0, vram={0: 30}),
            ),
        ]
    )
    view = ResourceView([gguf])

    assert view.allocated(2, "cuda") == Allocated(ram=0, vram={0: 30})


def test_simulated_bindings_share_reservations_without_mutating_the_base_view():
    models = [_instance(1)]
    reservations = {1: Allocated(ram=40, vram={})}
    view = ResourceView(models, reservations)
    simulated = view.with_model_instances([*models, _instance(2)])
    reservations[1].ram = 90
    assert len(models) == 1

    assert view.allocatable(_worker()).ram == 40
    assert simulated.allocatable(_worker()).ram == 30
    simulated.allocated(1).vram[0] = 999
    assert view.allocated(1).vram == {0: 20}
    assert simulated.allocated(1).vram == {0: 40}


@pytest.mark.asyncio
async def test_selector_and_scorer_use_the_same_resource_view():
    from gpustack.scheduler.scheduler import build_candidate_selector
    from gpustack.policies.scorers.placement_scorer import PlacementScorer

    model, worker = _model(), _worker()
    view = ResourceView([], {1: Allocated(ram=50, vram={})})
    selector = build_candidate_selector(
        SimpleNamespace(),
        model,
        [],
        cpu_only=True,
        ram_claim=20,
        resource_view=view,
    )
    candidates = await selector.select_candidates([worker])
    scored = await PlacementScorer(model, [], max_score=100, resource_view=view).score(
        candidates
    )

    assert len(scored) == 1
    assert scored[0].score == pytest.approx(50)
    assert selector.get_worker_allocatable_resource(worker).ram == 40

    full = build_candidate_selector(
        SimpleNamespace(),
        model,
        [],
        cpu_only=True,
        ram_claim=20,
        resource_view=ResourceView([], {1: Allocated(ram=80, vram={})}),
    )
    assert full.get_worker_allocatable_resource(worker).ram == 10
    assert await full.select_candidates([worker]) == []
    assert selector.get_worker_allocatable_resource(worker).ram == 40


@pytest.mark.asyncio
async def test_group_slot_simulation_retains_cache_reservations():
    from gpustack.scheduler.group_capacity import GroupCapacity
    from gpustack.scheduler.offer_slot import count_offer_slots

    model, worker = _model(), _worker()
    capacity = GroupCapacity(
        SimpleNamespace(),
        model,
        [worker],
        [],
        cache_instances=[
            SimpleNamespace(
                worker_id=1,
                computed_resource_claim={"ram": 40},
                deleted_at=None,
            )
        ],
        resource_view=ResourceView([], {1: Allocated(ram=40, vram={})}),
    )
    offer = await count_offer_slots(
        make_selector=lambda instances: capacity._selector(model, instances, True, 20),
        workers=[worker],
        model_instances=[],
        limit=5,
    )

    assert offer.slots == 2
    assert len(offer.placements) == 2
    assert capacity._resource_view.allocatable(worker).ram == 50


@pytest.mark.parametrize("ram_claim, expected_groups", [(40, 1), (50, 0)])
def test_gpu_grouping_uses_shared_memory_after_reservations(ram_claim, expected_groups):
    from gpustack.policies.utils import group_worker_gpu_by_memory

    worker = _worker()
    worker.status = WorkerStatus(
        memory=MemoryInfo(total=100, is_unified_memory=True),
        gpu_devices=[{"index": 0, "type": "mps", "memory": {"total": 100}}],
    )
    resources = ResourceView([], {1: Allocated(ram=45, vram={})})
    groups = group_worker_gpu_by_memory([worker], resources, ram_claim, "mps")

    assert len(groups) == expected_groups
    if groups:
        assert groups[0][0].allocatable_vram == 45


def test_scheduling_policies_require_an_explicit_resource_view():
    from gpustack.scheduler.scheduler import build_candidate_selector
    from gpustack.policies.scorers.placement_scorer import PlacementScorer

    with pytest.raises(TypeError, match="resource_view"):
        build_candidate_selector(SimpleNamespace(), _model(), [])
    with pytest.raises(TypeError, match="resource_view"):
        PlacementScorer(_model(), [])


@pytest.mark.parametrize(
    "selector_type",
    [
        AscendMindIEResourceFitSelector,
        GGUFResourceFitSelector,
        SGLangResourceFitSelector,
        VGPUResourceFitSelector,
        VLLMResourceFitSelector,
        CustomBackendResourceFitSelector,
        InstanceTypeWholeCardSelector,
    ],
)
def test_every_backend_uses_reservations_from_its_constructor(selector_type):
    instances = [_instance(1)]
    resource_view = ResourceView(instances, {1: Allocated(ram=30, vram={})})
    if selector_type is GGUFResourceFitSelector:
        selector = selector_type(_model(), instances, resource_view=resource_view)
    else:
        selector = selector_type(
            SimpleNamespace(), _model(), instances, resource_view=resource_view
        )

    assert selector.get_worker_allocatable_resource(_worker()).ram == 50


@pytest.mark.parametrize("worker_id", [1, 2])
def test_distributed_deployment_limit_uses_model_bindings(worker_id):
    from gpustack.schemas.models import BackendEnum

    model = _model()
    model.backend = BackendEnum.VLLM
    existing = _instance(1)
    existing.name = "distributed-model"
    existing.model = SimpleNamespace(backend=BackendEnum.VLLM)
    existing.distributed_servers = SimpleNamespace(
        subordinate_workers=[SimpleNamespace(worker_id=2)]
    )
    instances = [existing]
    selector = VLLMResourceFitSelector(
        SimpleNamespace(), model, instances, resource_view=ResourceView(instances)
    )
    worker = _worker()
    worker.id = worker_id

    assert not selector._validate_distributed_limit_per_worker(worker)
    assert "distributed-model" in selector.get_messages()[0]
