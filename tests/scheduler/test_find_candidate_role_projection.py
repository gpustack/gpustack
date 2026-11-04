"""`find_candidate` projects the role before building anything.

The design's claim is that the scheduler's filters, selectors and scorers —
dozens of files reading Model-level fields directly — need no changes for
multi-role deployments, because the Model they are handed has already had the
role's overrides applied. These tests hold that claim to account: they check
what the filters and the selector actually receive, and that the selector
*branch* is chosen from the role's backend rather than the model's.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gpustack.policies.resource_view import ResourceView

from gpustack.scheduler import scheduler
from gpustack.schemas.models import (
    BackendEnum,
    DisaggregationSpec,
    Model,
    PDModeEnum,
    RoleSpec,
)


def _model(**kwargs):
    base = dict(
        id=1,
        name="pd",
        source="local_path",
        local_path="/models/m",
        backend=BackendEnum.VLLM.value,
        backend_parameters=["--model-level"],
        replicas=1,
    )
    base.update(kwargs)
    return Model(**base)


def _worker():
    return SimpleNamespace(id=7, name="node-a")


def _pd_model():
    return _model(
        roles=[
            RoleSpec(name="prefill", replicas=1),
            RoleSpec(name="decode", replicas=1),
            RoleSpec(name="router", replicas=1),
        ],
        disaggregation=DisaggregationSpec(mode=PDModeEnum.VLLM_NIXL),
    )


class _RecordingSelector:
    """Stands in for a ScheduleCandidatesSelector and records its model."""

    seen = []

    def __init__(self, *args, **kwargs):
        # Selectors receive the model and its complete resource view.
        self.model = next(a for a in args if hasattr(a, "backend_parameters"))
        # Only the custom selector takes it; recorded so a test can tell "the
        # custom selector was chosen" from "it was chosen for a router".
        self.cpu_only = kwargs.get("cpu_only", False)
        self.resource_view = kwargs["resource_view"]
        self.instances = next(a for a in args if isinstance(a, (list, tuple)))
        type(self).seen.append(self)

    async def select_candidates(self, workers):
        return []

    def get_messages(self):
        return []


@pytest.fixture
def harness():
    """Let workers through, record what each selector was built with, and
    keep scoring out of the way."""
    selectors = {}
    for name in (
        "VGPUResourceFitSelector",
        "GGUFResourceFitSelector",
        "AscendMindIEResourceFitSelector",
        "VLLMResourceFitSelector",
        "SGLangResourceFitSelector",
        "CustomBackendResourceFitSelector",
    ):
        selectors[name] = type(name, (_RecordingSelector,), {"seen": []})

    filters_seen = []

    class _FilterChain:
        def __init__(self, filters):
            filters_seen.append(filters)

        async def filter(self, workers):
            return workers, []

    class _ScoreChain:
        def __init__(self, scorers):
            pass

        async def score(self, candidates):
            return candidates

    with patch.multiple(
        scheduler,
        load_resource_view=AsyncMock(
            side_effect=lambda session, instances, **kwargs: ResourceView(instances)
        ),
        WorkerFilterChain=_FilterChain,
        CandidateScoreChain=_ScoreChain,
        pick_highest_score_candidate=lambda candidates: None,
        **selectors,
    ):
        yield SimpleNamespace(selectors=selectors, filters_seen=filters_seen)


async def _run(model, role=None):
    # `session` is unused on this path — the filters and selectors are stubbed —
    # so None is what the sibling selector tests pass too.
    return await scheduler.find_candidate(
        None, SimpleNamespace(cache_dir=None), model, [_worker()], [], role=role
    )


def _selected(harness):
    """The one selector that got built, and the model it was built with."""
    built = [(name, cls.seen[0]) for name, cls in harness.selectors.items() if cls.seen]
    assert len(built) == 1, f"expected one selector, got {[n for n, _ in built]}"
    return built[0]


@pytest.mark.asyncio
async def test_a_role_less_model_reaches_the_selector_unprojected(harness):
    model = _model()
    await _run(model)

    name, selector = _selected(harness)
    assert name == "VLLMResourceFitSelector"
    assert selector.model is model


@pytest.mark.asyncio
@pytest.mark.parametrize('reserve_gib', [2, 3])
async def test_managed_smg_ram_refusal_uses_the_reservation_constant(
    harness, monkeypatch, reserve_gib
):
    from gpustack.policies.candidate_selectors import sglang_resource_fit_selector

    selector = SimpleNamespace(
        select_candidates=AsyncMock(return_value=[object()]), get_messages=lambda: []
    )
    monkeypatch.setattr(
        scheduler, 'build_candidate_selector', lambda *a, **kw: selector
    )
    monkeypatch.setattr(scheduler, '_incompatible_cohort_message', lambda *a: None)
    monkeypatch.setattr(scheduler, 'sglang_smg_mode', lambda model: True)
    monkeypatch.setattr(
        scheduler, 'ROUTER_DEFAULT_MEMORY', reserve_gib * 1024**3, raising=False
    )
    monkeypatch.setattr(
        sglang_resource_fit_selector, 'reserve_sglang_smg_ram', lambda *a: []
    )

    candidate, messages = await scheduler.find_candidate(
        None,
        SimpleNamespace(cache_dir=None),
        _model(),
        [_worker()],
        [],
        managed_sglang=True,
    )

    assert candidate is None
    assert messages == [
        f'The primary Worker requires an additional {reserve_gib} GiB of RAM for SMG.',
        'No workers meet the resource requirements.',
    ]


@pytest.mark.asyncio
async def test_the_role_s_parameters_reach_the_selector(harness):
    model = _model(
        roles=[
            RoleSpec(name="prefill", replicas=3, backend_parameters=["--role-level"]),
            RoleSpec(name="decode"),
        ]
    )
    await _run(model, role="prefill")

    _, selector = _selected(harness)
    assert selector.model.backend_parameters == ["--role-level"]
    # And the role's count, not the 0/1 deployment switch: this is what the
    # multi-replica overcommit rule and gpus-per-replica read.
    assert selector.model.replicas == 3
    assert model.replicas == 1


@pytest.mark.asyncio
async def test_a_sibling_role_gets_its_own_values(harness):
    model = _model(
        roles=[
            RoleSpec(name="prefill", backend_parameters=["--p"]),
            RoleSpec(name="decode", backend_parameters=["--d"]),
        ]
    )
    await _run(model, role="decode")

    _, selector = _selected(harness)
    assert selector.model.backend_parameters == ["--d"]


@pytest.mark.asyncio
async def test_the_selector_branch_follows_the_role_s_backend(harness):
    # The branch is `model.backend == BackendEnum.SGLANG`, untouched code.
    # It picks SGLang only because the projection happened first.
    model = _model(
        roles=[
            RoleSpec(name="prefill"),
            RoleSpec(name="decode", backend=BackendEnum.SGLANG.value),
        ],
        disaggregation=DisaggregationSpec(mode=PDModeEnum.CUSTOM),
    )
    await _run(model, role="decode")

    name, _ = _selected(harness)
    assert name == "SGLangResourceFitSelector"


@pytest.mark.asyncio
async def test_the_worker_filters_see_the_projection_too(harness):
    model = _model(
        roles=[RoleSpec(name="prefill", worker_selector={"zone": "a"})],
    )
    await _run(model, role="prefill")

    (filters,) = harness.filters_seen
    # Every filter is constructed from the model; check the ones that hold on
    # to it got the projection rather than the Model itself.
    seen = [
        m
        for m in (
            getattr(f, "_model", None) or getattr(f, "model", None) for f in filters
        )
        if m is not None
    ]
    assert seen, "no filter kept a reference to the model"
    assert all(m is not model for m in seen)
    assert all(m.worker_selector == {"zone": "a"} for m in seen)


@pytest.mark.asyncio
async def test_find_candidate_does_not_mutate_the_model(harness):
    model = _model(
        roles=[RoleSpec(name="prefill", replicas=3, backend_parameters=["--role"])]
    )
    await _run(model, role="prefill")

    assert model.backend_parameters == ["--model-level"]
    assert model.replicas == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('managed', [False, True])
@pytest.mark.parametrize('mode', ['external', 'hybrid', 'multi-port-external'])
@pytest.mark.parametrize('port_range', ['40000-40003', '40000-40063'])
async def test_pending_instance_keeps_its_port_mode_after_manual_dp_edit(
    harness, monkeypatch, managed, mode, port_range
):
    from uuid import uuid4
    from gpustack.policies.base import ModelInstanceScheduleCandidate
    from gpustack.scheduler import port_budget
    from gpustack.schemas.models import DistributedServers
    from gpustack.schemas.port_reservations import WorkerPortAvailability
    from tests.utils.model import new_model_instance

    env = {'GPUSTACK_MANUAL_DP': '1' if managed else '0', 'OTHER': 'preserved'}
    model = _model(
        backend_parameters=[f'--data-parallel-{mode}-lb', '--tp=2', '--dp=2'],
        env=env.copy(),
    )
    pending = new_model_instance(10, 'pending', model.id)
    if managed:
        pending.distributed_servers = DistributedServers(workloads=[])
    worker = _admission_worker()
    candidate = ModelInstanceScheduleCandidate(worker, [0, 1, 2, 3], None)
    monkeypatch.setattr(
        harness.selectors['VLLMResourceFitSelector'],
        'select_candidates',
        AsyncMock(return_value=[candidate]),
    )
    monkeypatch.setattr(
        scheduler, 'pick_highest_score_candidate', lambda found: next(iter(found), None)
    )
    snapshots = AsyncMock(
        return_value={
            worker.id: WorkerPortAvailability(
                worker_id=worker.id, generation=uuid4(), port_range=port_range
            )
        }
    )
    monkeypatch.setattr(port_budget, 'worker_port_snapshots', snapshots)

    selected, messages = await scheduler.find_candidate(
        None,
        SimpleNamespace(cache_dir=None),
        model,
        [worker],
        [pending],
        exclude_instance_id=pending.id,
    )

    if managed and mode != 'hybrid' and port_range == '40000-40003':
        assert selected is None
        assert any('service ports' in message for message in messages)
    else:
        assert selected is candidate
        assert messages == []
    if managed:
        snapshots.assert_awaited_once_with([worker], pending.id)
    else:
        snapshots.assert_not_awaited()
    assert model.env == env
    assert _selected(harness)[1].model.env['OTHER'] == 'preserved'
    assert pending.state == 'pending'
    assert (pending.distributed_servers is not None) == managed


@pytest.mark.asyncio
@pytest.mark.parametrize('managed', [False, True])
@pytest.mark.parametrize('skip_count_check', [False, True])
async def test_pending_instance_keeps_gpu_count_validation_after_manual_dp_edit(
    harness, monkeypatch, managed, skip_count_check
):
    from gpustack.policies.base import ModelInstanceScheduleCandidate
    from gpustack.policies.candidate_selectors import vllm_resource_fit_selector
    from gpustack.schemas.models import (
        ComputedResourceClaim,
        DistributedServers,
        GPUSelector,
    )
    from tests.utils.model import new_model_instance

    selector_class = vllm_resource_fit_selector.VLLMResourceFitSelector
    model = _model(
        backend_parameters=['--data-parallel-external-lb', '--tp=2', '--dp=2'],
        gpu_selector=GPUSelector(gpu_ids=['worker-7:cuda:0', 'worker-7:cuda:1']),
        env={
            'GPUSTACK_MANUAL_DP': '1' if managed else '0',
            'GPUSTACK_SKIP_GPU_COUNT_CHECK': '1' if skip_count_check else '0',
        },
    )
    pending = new_model_instance(10, 'pending', model.id)
    if managed:
        pending.distributed_servers = DistributedServers(workloads=[])
    worker = _admission_worker()
    candidate = ModelInstanceScheduleCandidate(
        worker, [0, 1], ComputedResourceClaim(vram={0: 1, 1: 1}, ram=0)
    )
    monkeypatch.setattr(scheduler, 'VLLMResourceFitSelector', selector_class)

    async def init_parameters(selector, workers):
        selector._gpu_memory_utilization = 0

    monkeypatch.setattr(selector_class, '_init_model_parameters', init_parameters)
    monkeypatch.setattr(
        selector_class, 'find_manual_gpu_selection_candidates', lambda *_: [candidate]
    )
    monkeypatch.setattr(
        vllm_resource_fit_selector, 'estimate_model_vram', AsyncMock(return_value=0)
    )
    monkeypatch.setattr(
        vllm_resource_fit_selector, 'estimate_offloaded_lookup_bytes', lambda *_: 0
    )
    monkeypatch.setattr(
        scheduler, 'pick_highest_score_candidate', lambda found: next(iter(found), None)
    )

    selected, messages = await scheduler.find_candidate(
        None,
        SimpleNamespace(cache_dir=None, huggingface_token=None),
        model,
        [worker],
        [pending],
        exclude_instance_id=pending.id,
    )

    if managed:
        assert selected is None
        assert messages
        if not skip_count_check:
            assert any(
                'does not match the world size' in message for message in messages
            )
    else:
        assert selected is candidate
        assert messages == []


@pytest.mark.asyncio
@pytest.mark.parametrize('manual', ['0', '1'])
async def test_preview_uses_desired_manual_dp_without_an_instance(harness, manual):
    model = _model(
        backend_parameters=['--data-parallel-external-lb', '--dp=2'],
        env={'GPUSTACK_MANUAL_DP': manual},
    )

    await _run(model)

    assert _selected(harness)[1].model is model


# --- the router takes no accelerator --------------------------------------- #


def test_the_answer_is_read_off_the_role_not_off_the_projection():
    """`role_takes_no_accelerator` asks the ROLE, and the projection flattens
    the role's overrides onto the model — after which there is no role left to
    ask. So `find_candidate` has to read it first, which is what this pins.

    There is deliberately no `cpu_only` field to check alongside it. A boolean
    cannot stand in for a quantity: the scheduler needs a VRAM claim, and such
    a flag could only decide whether to go ask `estimate_model_vram`, which
    sizes the model's WEIGHTS — so its `False` branch would book a proxy at the
    whole model, 164 GiB for a 72B, with no per-role way to override it."""
    from gpustack.schemas.models import _ROLE_OVERRIDE_FIELDS, _ROLE_OWN_FIELDS

    assert "cpu_only" not in _ROLE_OWN_FIELDS
    assert "cpu_only" not in _ROLE_OVERRIDE_FIELDS
    assert "resources" in _ROLE_OWN_FIELDS


@pytest.mark.asyncio
async def test_the_router_uses_the_cpu_only_selector(harness):
    """Observed live: the router inherited the group's vLLM backend, so a vLLM
    selector sized the model's weights for a process that never loads them —
    and it sat unschedulable on a host whose cards its own peers had filled."""
    await _run(_pd_model(), role="router")

    custom = harness.selectors["CustomBackendResourceFitSelector"]
    assert len(custom.seen) == 1
    assert custom.seen[0].cpu_only is True
    assert not harness.selectors["VLLMResourceFitSelector"].seen


@pytest.mark.asyncio
async def test_a_gpu_role_keeps_its_engines_selector(harness):
    await _run(_pd_model(), role="prefill")

    assert len(harness.selectors["VLLMResourceFitSelector"].seen) == 1
    assert not harness.selectors["CustomBackendResourceFitSelector"].seen


@pytest.mark.asyncio
async def test_a_role_less_model_never_takes_the_accelerator_free_path(harness):
    await _run(_model())

    assert len(harness.selectors["VLLMResourceFitSelector"].seen) == 1
    assert not harness.selectors["CustomBackendResourceFitSelector"].seen


def _admission_worker(identity=7, supported=True):
    return SimpleNamespace(
        id=identity,
        name=f'worker-{identity}',
        cluster_id=None,
        deleted_at=None,
        state='ready',
        system_principal_id=identity + 100 if supported else None,
        worker_version='2.3.0',
    )


@pytest.mark.asyncio
async def test_identity_filter_keeps_target_marker_before_exclusion_and_role_projection(
    harness,
):
    from gpustack.schemas.models import DistributedServers
    from gpustack.policies.worker_filters.status_filter import StatusFilter
    from tests.utils.model import new_model_instance

    model = _pd_model()
    pending = new_model_instance(10, 'new', model.id)
    pending.distributed_servers = DistributedServers(workloads=[])
    await scheduler.find_candidate(
        None,
        SimpleNamespace(cache_dir=None),
        model,
        [_admission_worker()],
        [pending],
        role='router',
        exclude_instance_id=10,
    )
    status = next(f for f in harness.filters_seen[0] if isinstance(f, StatusFilter))
    kept, messages = await status.filter([_admission_worker(supported=False)])
    assert kept == []
    assert any('identity' in message for message in messages)
    assert status._model is not model


@pytest.mark.asyncio
@pytest.mark.parametrize('bound', [False, True])
async def test_initial_identity_filter_admits_bound_workers_regardless_of_version(
    harness, monkeypatch, bound
):
    from gpustack.schemas.models import DistributedServers
    from gpustack.policies.worker_filters.status_filter import StatusFilter
    from tests.utils.model import new_model_instance

    monkeypatch.setattr(
        'gpustack.extension.resolve_version_info', lambda: ('2.3.1-ent1', 'test')
    )
    model = _model()
    pending = new_model_instance(10, 'new', model.id)
    pending.distributed_servers = DistributedServers(workloads=[])

    class IdentityChain:
        def __init__(self, filters):
            self.filters = filters

        async def filter(self, workers):
            return await next(
                f for f in self.filters if isinstance(f, StatusFilter)
            ).filter(workers)

    monkeypatch.setattr(scheduler, 'WorkerFilterChain', IdentityChain)
    candidate, messages = await scheduler.find_candidate(
        None,
        SimpleNamespace(cache_dir=None),
        model,
        [_admission_worker(supported=bound)],
        [pending],
        exclude_instance_id=10,
    )
    assert candidate is None
    if bound:
        assert harness.selectors['VLLMResourceFitSelector'].seen
    else:
        assert any('identity' in message for message in messages)
        assert all(not cls.seen for cls in harness.selectors.values())


@pytest.mark.asyncio
async def test_pending_managed_model_cannot_join_unbound_assigned_legacy_router(
    harness,
):
    from gpustack.schemas.models import DistributedServers
    from tests.utils.model import new_model_instance

    model = _pd_model()
    pending = new_model_instance(10, 'new', model.id)
    pending.distributed_servers = DistributedServers(workloads=[])
    router = new_model_instance(11, 'router', model.id, worker_id=9)
    router.role = 'router'
    candidate, messages = await scheduler.find_candidate(
        None,
        SimpleNamespace(cache_dir=None),
        model,
        [_admission_worker(), _admission_worker(9, False)],
        [pending, router],
        exclude_instance_id=10,
    )
    assert candidate is None
    assert any('re-register' in message for message in messages)
    assert all(not cls.seen for cls in harness.selectors.values())


@pytest.mark.asyncio
async def test_cohort_identity_precheck_does_not_require_existing_sibling_ready():
    from gpustack.schemas.models import DistributedServers
    from tests.utils.model import new_model_instance

    model = _model()
    pending = new_model_instance(10, 'new', model.id)
    pending.distributed_servers = DistributedServers(workloads=[])
    sibling = new_model_instance(11, 'existing', model.id, worker_id=9)
    worker = _admission_worker(9)
    worker.state = 'maintenance'
    assert (
        scheduler._incompatible_cohort_message(
            model, [worker], [pending, sibling], True
        )
        is None
    )
    worker.system_principal_id = None
    assert 'bound' in scheduler._incompatible_cohort_message(
        model, [worker], [pending, sibling], True
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('role', [None, 'prefill'])
async def test_port_admission_filters_candidates_before_the_score_chain(
    harness, monkeypatch, role
):
    from gpustack.scheduler import port_budget

    candidates = [object(), object()]
    model = _pd_model() if role else _model()
    for selector in harness.selectors.values():
        monkeypatch.setattr(
            selector, 'select_candidates', AsyncMock(return_value=candidates)
        )
    order = []

    async def admit(
        session, effective, requested_role, found, workers, instances, excluded
    ):
        assert found == candidates and requested_role == role
        order.append('ports')
        return [candidates[1]], ['port shortage']

    class ScoreChain:
        def __init__(self, scorers):
            pass

        async def score(self, found):
            assert found == [candidates[1]]
            order.append('score')
            return found

    monkeypatch.setattr(port_budget, 'filter_candidates', admit)
    monkeypatch.setattr(scheduler, 'CandidateScoreChain', ScoreChain)
    _, messages = await _run(model, role)
    assert order == ['ports', 'score']
    assert 'port shortage' in messages
