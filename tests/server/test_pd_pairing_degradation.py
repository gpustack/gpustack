"""Report pairing factors with unresolved effective values."""

from types import SimpleNamespace

from gpustack.schemas.models import (
    DegradationReasonEnum,
    DisaggregationSpec,
    PDModeEnum,
    RoleSpec,
)
from gpustack.server.controllers import _pairing_unverified

NIXL = PDModeEnum.VLLM_NIXL


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


def _role(name, parameters=None):
    return RoleSpec(name=name, replicas=1, backend_parameters=parameters)


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


def test_tp_on_one_role_is_not_a_pairing_warning():
    model = _model(prefill=_role("prefill", ["--tensor-parallel-size=2"]))
    assert _pairing_unverified(model) is False


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


def test_the_legacy_placement_marker_can_still_be_read():
    assert (
        DegradationReasonEnum("pairing_tp_misplaced")
        is DegradationReasonEnum.PAIRING_TP_MISPLACED
    )


# --- a window written in a form the comparison cannot read ------------------ #


def test_two_unreadable_windows_are_not_taken_for_silence():
    """vLLM accepts `--max-model-len 128k`, and an integer parse answers None
    for that exactly as it does for a key nobody wrote. Reading both alike put
    a prefill at 128k beside a decode at 32k in the branch that means "neither
    said anything, so both take the window from the model config" — and called
    them agreed. That is the mismatch this comparison exists for."""
    from gpustack.validation.pd.parameters import (
        AGREE,
        UNDECIDABLE,
        compare_max_model_len,
    )

    verdict, prefill, decode = compare_max_model_len(
        ["--max-model-len=128k"], ["--max-model-len=32k"]
    )

    assert verdict == UNDECIDABLE
    assert (prefill, decode) == (None, None)

    # The same spelling on both sides is still agreement: whatever the engine
    # makes of it, it makes the same thing of it twice.
    same, _, _ = compare_max_model_len(
        ["--max-model-len=128k"], ["--max-model-len=128k"]
    )
    assert same == AGREE


def test_one_unreadable_window_beside_a_silent_role_is_not_agreement():
    """A role that wrote `128k` and one that wrote nothing are not the same
    deployment: the second takes whatever the model config says, and nothing
    here knows that the first resolves to it."""
    from gpustack.validation.pd.parameters import UNDECIDABLE, compare_max_model_len

    verdict, _, _ = compare_max_model_len(["--max-model-len=128k"], [])

    assert verdict == UNDECIDABLE


def test_both_silent_still_agree():
    """The ordinary case, unchanged: neither role names a window, so both take
    it from the same model config."""
    from gpustack.validation.pd.parameters import AGREE, compare_max_model_len

    assert compare_max_model_len([], [])[0] == AGREE
