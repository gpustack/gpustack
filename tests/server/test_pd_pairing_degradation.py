"""The two markers that say what the pairing pre-check could not.

Admission refuses a prefill/decode mismatch it can prove, and the proof needs
both sides' effective values. Until now "prefill wrote it, decode did not" was
read as nothing to compare, which is how the ordinary misconfiguration — edit
one role, leave the other — passed a check the user believes ran.

Substituting the engines' defaults was the wrong repair, and `server.pd_pairing`
records why for each factor. So the gap is reported in two places instead:

* `pairing_unverified` on the spec, saying a factor was declared on one side
  and left silent on the other;
* `pairing_tp_misplaced` after placement, where the cards the members
  actually got are the tensor parallelism and the recipe's declared direction
  can finally be applied.

Both are markers. The first describes deployments that are usually correct, and
the second describes members that are already running.
"""

from types import SimpleNamespace

from gpustack.schemas.models import (
    DegradationReasonEnum,
    DisaggregationSpec,
    GPUSelector,
    ModelInstanceSubordinateWorker,
    PDModeEnum,
    RoleSpec,
)
from gpustack.server.controllers import (
    _pairing_tp_misplaced,
    _pairing_unverified,
)

NIXL = PDModeEnum.VLLM_NIXL
MOONCAKE = PDModeEnum.VLLM_ASCEND_MOONCAKE
CUSTOM = PDModeEnum.CUSTOM


def _model(mode=NIXL, backend_parameters=None, prefill=None, decode=None):
    roles = [
        prefill or RoleSpec(name="prefill", replicas=1),
        decode or RoleSpec(name="decode", replicas=1),
        RoleSpec(name="router", replicas=1),
    ]
    return SimpleNamespace(
        disaggregation=(DisaggregationSpec(mode=mode) if mode is not None else None),
        backend_parameters=backend_parameters,
        roles=roles,
    )


def _role(name, parameters=None, gpu_ids=None, replicas=1):
    return RoleSpec(
        name=name,
        replicas=replicas,
        backend_parameters=parameters,
        gpu_selector=GPUSelector(gpu_ids=list(gpu_ids)) if gpu_ids else None,
    )


def _instance(role, cards, worker_id=1, subordinate=False):
    servers = None
    if subordinate:
        servers = SimpleNamespace(
            subordinate_workers=[
                ModelInstanceSubordinateWorker(worker_id=2, gpu_indexes=[0])
            ]
        )
    return SimpleNamespace(
        role=role,
        worker_id=worker_id,
        gpu_indexes=list(range(cards)),
        distributed_servers=servers,
    )


# --- the spec-only marker --------------------------------------------------- #


def test_a_window_on_one_role_only_is_unverified():
    """The measured case the whole pre-check exists for, arriving in the shape
    it could not judge: decode's window is whatever the model config says, and
    whether that is 8192 is not knowable without the checkpoint."""
    model = _model(prefill=_role("prefill", ["--max-model-len=8192"]))
    assert _pairing_unverified(model) is True


def test_a_window_on_both_roles_is_not():
    model = _model(
        prefill=_role("prefill", ["--max-model-len=8192"]),
        decode=_role("decode", ["--max-model-len=8192"]),
    )
    assert _pairing_unverified(model) is False


def test_two_silent_roles_are_not_marked():
    """Not "unknown, assume bad". Both roles take the same default from the
    same engine on the same model, so whatever it resolves to it resolves to it
    twice — and marking this would put the badge on the ordinary way a group is
    deployed, which is how a marker gets ignored."""
    assert _pairing_unverified(_model()) is False


def test_a_model_level_value_both_roles_inherit_is_not_marked():
    assert _pairing_unverified(_model(backend_parameters=["--dtype=float16"])) is False


def test_clearing_one_roles_parameters_is_that_role_falling_silent():
    """An empty override deliberately does not inherit, so prefill runs the
    engine's defaults while decode runs the model's 8192. Same hole, reached
    from the other side."""
    model = _model(
        backend_parameters=["--max-model-len=8192"],
        prefill=_role("prefill", []),
    )
    assert _pairing_unverified(model) is True


def test_an_explicit_auto_against_silence_is_verified():
    """`auto` is what the silent side runs. Marking this would report a pair
    that provably agrees."""
    model = _model(prefill=_role("prefill", ["--dtype=auto"]))
    assert _pairing_unverified(model) is False


def test_a_concrete_dtype_against_silence_is_unverified():
    model = _model(prefill=_role("prefill", ["--dtype=float16"]))
    assert _pairing_unverified(model) is True


def test_an_explicit_auto_against_a_concrete_dtype_is_unverified_too():
    """Writing `auto` down does not make the pair decidable. It resolves to
    the checkpoint's dtype, so whether it equals float16 is a question about a
    file nothing on this path opens — which is why this is a marker rather than
    the 400 that two unequal concrete strings earn."""
    model = _model(
        prefill=_role("prefill", ["--dtype=auto"]),
        decode=_role("decode", ["--dtype=float16"]),
    )
    assert _pairing_unverified(model) is True


def test_two_concrete_dtypes_never_reach_this_marker():
    """They are refused at admission, so a model row carrying them does not
    exist. Asserted from the other end: the factor is a *mismatch*, not an
    absence, and this marker only ever reports absences."""
    model = _model(
        prefill=_role("prefill", ["--dtype=float16"]),
        decode=_role("decode", ["--dtype=bfloat16"]),
    )
    assert _pairing_unverified(model) is False


def test_a_tp_against_a_pinned_peer_is_verified_not_marked():
    """Both sides determinable — decode's cards are pinned, so its tensor
    parallelism is known and the pair was actually judged. (That this
    particular pair is judged *badly* is admission's business; a group with
    this shape never reaches a controller.)"""
    model = _model(
        prefill=_role("prefill", ["--tensor-parallel-size=2"]),
        decode=_role("decode", gpu_ids=["w1:npu:0", "w1:npu:1"]),
    )
    assert _pairing_unverified(model) is False


def test_a_tp_against_an_unpinned_peer_is_unverified():
    model = _model(prefill=_role("prefill", ["--tensor-parallel-size=2"]))
    assert _pairing_unverified(model) is True


def test_a_model_without_disaggregation_is_untouched():
    assert _pairing_unverified(_model(mode=None)) is False


def test_a_group_missing_a_side_is_untouched():
    """Not every multi-role deployment is a P/D pair, and a group without both
    has no pairing to leave unverified."""
    model = SimpleNamespace(
        disaggregation=DisaggregationSpec(mode=NIXL),
        backend_parameters=["--dtype=float16"],
        roles=[RoleSpec(name="prefill", replicas=1)],
    )
    assert _pairing_unverified(model) is False


def test_the_marker_is_the_value_the_api_publishes():
    assert DegradationReasonEnum.PAIRING_UNVERIFIED.value == "pairing_unverified"


# --- the placement marker --------------------------------------------------- #


def test_a_placed_decode_narrower_than_its_prefill_is_marked():
    """Neither role writes a tp, so admission had nothing to compare. The
    scheduler gave prefill four cards and decode one, both engines derive tp
    from exactly that, and under NIXL the first transfer raises an IndexError
    inside decode."""
    model = _model()
    instances = [_instance("prefill", 4), _instance("decode", 1)]
    assert _pairing_tp_misplaced(model, instances) is True


def test_a_placed_decode_at_least_as_wide_is_not():
    model = _model()
    instances = [_instance("prefill", 2), _instance("decode", 4)]
    assert _pairing_tp_misplaced(model, instances) is False


def test_the_direction_is_still_the_recipes():
    """Huawei's reference deployment — prefill TP4 / decode TP1 — is what
    Mooncake is for, and it is the exact shape NIXL forbids. The same placement
    is a fault under one recipe and the intended layout under the other."""
    instances = [_instance("prefill", 4), _instance("decode", 1)]
    assert _pairing_tp_misplaced(_model(mode=MOONCAKE), instances) is False
    assert (
        _pairing_tp_misplaced(
            _model(mode=MOONCAKE),
            [_instance("prefill", 1), _instance("decode", 4)],
        )
        is True
    )


def test_custom_declares_no_direction_so_nothing_is_marked():
    instances = [_instance("prefill", 4), _instance("decode", 1)]
    assert _pairing_tp_misplaced(_model(mode=CUSTOM), instances) is False


def test_a_declared_tp_beats_the_cards_the_member_got():
    """The engine runs what the role wrote, so a decode declaring TP4 on two
    cards is a decode at TP4 — a launch failure of its own, and not this
    marker's business."""
    model = _model(
        prefill=_role("prefill", ["--tensor-parallel-size=4"]),
        decode=_role("decode", ["--tensor-parallel-size=4"]),
    )
    instances = [_instance("prefill", 4), _instance("decode", 2)]
    assert _pairing_tp_misplaced(model, instances) is False


def test_an_unplaced_member_contributes_nothing():
    """`gpu_indexes` on a row the scheduler has not answered for is not a
    width, and a group mid-formation must not wear a marker about its shape."""
    model = _model()
    instances = [
        _instance("prefill", 4),
        _instance("decode", 0, worker_id=None),
    ]
    assert _pairing_tp_misplaced(model, instances) is False


def test_a_member_spanning_workers_contributes_nothing():
    """Its world size is split into tp and pp much further down the vLLM path,
    so its card count is not its tensor parallelism."""
    model = _model()
    instances = [
        _instance("prefill", 4),
        _instance("decode", 1, subordinate=True),
    ]
    assert _pairing_tp_misplaced(model, instances) is False


def test_a_role_writing_dp_without_tp_contributes_nothing():
    """Any parallelism flag turns GPUStack's tp injection off wholesale, so the
    member runs the engine's own default rather than its cards."""
    model = _model(decode=_role("decode", ["--data-parallel-size=2"]))
    instances = [_instance("prefill", 4), _instance("decode", 1)]
    assert _pairing_tp_misplaced(model, instances) is False


def test_the_narrowest_decode_is_what_counts():
    """The router pairs at random, so one decode narrow enough is enough: a
    transfer will land on it."""
    model = _model()
    instances = [
        _instance("prefill", 2),
        _instance("decode", 2),
        _instance("decode", 1),
    ]
    assert _pairing_tp_misplaced(model, instances) is True


def test_a_model_without_disaggregation_is_untouched_by_placement_too():
    assert _pairing_tp_misplaced(_model(mode=None), []) is False


def test_the_placement_marker_is_the_value_the_api_publishes():
    assert DegradationReasonEnum.PAIRING_TP_MISPLACED.value == "pairing_tp_misplaced"
