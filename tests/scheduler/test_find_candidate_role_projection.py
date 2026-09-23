"""`find_candidate` projects the role before building anything.

The design's claim is that the scheduler's filters, selectors and scorers —
dozens of files reading Model-level fields directly — need no changes for
multi-role deployments, because the Model they are handed has already had the
role's overrides applied. These tests hold that claim to account: they check
what the filters and the selector actually receive, and that the selector
*branch* is chosen from the role's backend rather than the model's.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

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
        # The selectors take either (config, model, instances) or
        # (model, instances, cache_dir); the model is the first Model-ish arg.
        self.model = next(a for a in args if hasattr(a, "backend_parameters"))
        # Only the custom selector takes it; recorded so a test can tell "the
        # custom selector was chosen" from "it was chosen for a router".
        self.cpu_only = kwargs.get("cpu_only", False)
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
