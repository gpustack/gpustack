"""Prefill/decode pairing pre-checks.

The division of labour with the engine is what these tests encode. Most
handshake factors are hashed by the connector and rejected on contact, so
checking them here buys attribution rather than safety. Two are different, and
they are why this validation exists at all:

* `max_model_len` is checked by nothing. With prefill's window wider than
  decode's the handshake passes, KV transfers, short prompts answer normally,
  and only a prompt above decode's window fails — at decode, after prefill has
  already computed it. The user believes the deployment serves prefill's
  window.
* a decode narrower than its prefill is asserted at run time, but surfaces as
  an `IndexError` inside decode rather than as a configuration error. That is
  NIXL's rule; the direction belongs to the connector and is read off the
  recipe (`PDMode.pairing`), because vllm-ascend's Mooncake wants the
  opposite and `custom` injects no connector at all.
"""

from contextlib import contextmanager

import pytest

from gpustack.api.exceptions import BadRequestException
from gpustack.routes.models import validate_role_pairing
from gpustack.schemas.models import (
    DisaggregationSpec,
    GPUSelector,
    Model,
    ModelCreate,
    PDModeEnum,
    RoleSpec,
    SourceEnum,
)
from gpustack.server.pd_pairing import undecidable_factors


@contextmanager
def rejects(fragment):
    """The API's HTTPException carries its text on ``.message``, not on
    ``str()``, so ``pytest.raises(match=...)`` would match the empty string."""
    with pytest.raises(BadRequestException) as excinfo:
        yield
    assert fragment in excinfo.value.message, excinfo.value.message


def _model_in(
    roles=None,
    disaggregation=True,
    backend_parameters=None,
    mode=PDModeEnum.VLLM_NIXL,
):
    return ModelCreate(
        name="m",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/repo",
        backend="vLLM",
        backend_parameters=backend_parameters,
        roles=roles,
        disaggregation=(DisaggregationSpec(mode=mode) if disaggregation else None),
    )


def _roles(prefill_params=None, decode_params=None):
    return [
        RoleSpec(name="prefill", replicas=1, backend_parameters=prefill_params),
        RoleSpec(name="decode", replicas=1, backend_parameters=decode_params),
        RoleSpec(name="router", replicas=1, cpu_only=True),
    ]


# --- the one nothing else checks ------------------------------------------- #


def test_mismatched_context_length_is_refused():
    with rejects("context lengths"):
        validate_role_pairing(
            _model_in(
                _roles(
                    prefill_params=["--max-model-len=8192"],
                    decode_params=["--max-model-len=4096"],
                )
            )
        )


def test_the_refusal_names_the_window_that_would_break():
    """The message has to carry the number, because the symptom the user would
    otherwise see is a 400 on a long prompt with nothing pointing here."""
    with rejects("4096"):
        validate_role_pairing(
            _model_in(
                _roles(
                    prefill_params=["--max-model-len", "8192"],
                    decode_params=["--max-model-len", "4096"],
                )
            )
        )


def test_sglangs_spelling_of_the_same_thing_is_caught():
    with rejects("context lengths"):
        validate_role_pairing(
            _model_in(
                _roles(
                    prefill_params=["--context-length=8192"],
                    decode_params=["--context-length=4096"],
                )
            )
        )


def test_matching_context_lengths_pass():
    validate_role_pairing(
        _model_in(
            _roles(
                prefill_params=["--max-model-len=8192"],
                decode_params=["--max-model-len=8192"],
            )
        )
    )


# --- tensor parallelism ---------------------------------------------------- #


def test_a_decode_narrower_than_its_prefill_is_refused():
    with rejects("tensor parallelism"):
        validate_role_pairing(
            _model_in(
                _roles(
                    prefill_params=["--tensor-parallel-size=8"],
                    decode_params=["--tensor-parallel-size=4"],
                )
            )
        )


def test_a_wider_decode_is_allowed():
    """The constraint is one-directional: decode must be at least prefill."""
    validate_role_pairing(
        _model_in(
            _roles(
                prefill_params=["-tp", "4"],
                decode_params=["-tp", "8"],
            )
        )
    )


def test_the_direction_is_the_connectors_not_pds():
    """vllm-ascend's Mooncake gathers a decode rank's KV from several prefill
    ranks; Huawei's reference deployment is prefill TP4 / decode TP1 — the
    exact shape NIXL's rule forbids. Holding every recipe to NIXL's rule
    rejected a working deployment, so the recipe declares its own."""
    validate_role_pairing(
        _model_in(
            _roles(
                prefill_params=["--tensor-parallel-size=4"],
                decode_params=["--tensor-parallel-size=1"],
            ),
            mode=PDModeEnum.VLLM_ASCEND_MOONCAKE,
        )
    )


def test_the_ascend_recipe_is_held_to_its_own_direction():
    with rejects("prefill runs tensor parallelism 1, below decode's 4"):
        validate_role_pairing(
            _model_in(
                _roles(
                    prefill_params=["--tensor-parallel-size=1"],
                    decode_params=["--tensor-parallel-size=4"],
                ),
                mode=PDModeEnum.VLLM_ASCEND_MOONCAKE,
            )
        )


def test_custom_injects_no_connector_so_no_direction_is_enforced():
    """The user's engine is the judge: GPUStack knows nothing about the
    connector they wrote into `--kv-transfer-config`."""
    for prefill, decode in (("8", "4"), ("4", "8")):
        validate_role_pairing(
            _model_in(
                _roles(prefill_params=["-tp", prefill], decode_params=["-tp", decode]),
                mode=PDModeEnum.CUSTOM,
            )
        )


def test_an_unresolvable_mode_falls_back_to_the_nixl_rule(monkeypatch):
    """What every recipe was held to before the direction became declarable,
    so a catalog that cannot answer changes nothing."""
    from gpustack.routes import models as routes_models

    monkeypatch.setattr(routes_models, "get_pd_mode", lambda name: None)
    with rejects("decode runs tensor parallelism 4, below prefill's 8"):
        validate_role_pairing(
            _model_in(_roles(prefill_params=["-tp", "8"], decode_params=["-tp", "4"]))
        )


# --- factors the engine also checks, caught here for attribution ----------- #


@pytest.mark.parametrize(
    "flag,label",
    [
        ("--dtype", "dtype"),
        ("--kv-cache-dtype", "KV cache dtype"),
        ("--block-size", "block size"),
        ("--kv-cache-layout", "KV cache layout"),
    ],
)
def test_disagreeing_handshake_factors_are_refused(flag, label):
    with rejects(label):
        validate_role_pairing(
            _model_in(
                _roles(
                    prefill_params=[f"{flag}=a"],
                    decode_params=[f"{flag}=b"],
                )
            )
        )


# --- inheritance ----------------------------------------------------------- #


def test_roles_inheriting_the_model_parameters_agree_by_construction():
    """`None` inherits, so two roles that override nothing cannot disagree."""
    validate_role_pairing(
        _model_in(_roles(), backend_parameters=["--max-model-len=8192"])
    )


def test_one_role_overriding_is_compared_against_the_inherited_value():
    """The dangerous shape: the user edits only decode and never sees the
    model-level value they are now contradicting."""
    with rejects("context lengths"):
        validate_role_pairing(
            _model_in(
                _roles(decode_params=["--max-model-len=4096"]),
                backend_parameters=["--max-model-len=8192"],
            )
        )


def test_an_empty_override_does_not_inherit():
    """A role that deliberately clears the model's parameters must not get
    them back — so nothing is compared, rather than the model's value being
    compared against itself.

    Still not a refusal, and now not silent either: clearing the parameters is
    one side falling silent, so the factors the model declared become
    undecidable and the deployment says so. See
    `test_clearing_the_parameters_is_a_side_falling_silent` below."""
    validate_role_pairing(
        _model_in(
            _roles(prefill_params=[]),
            backend_parameters=["--max-model-len=8192"],
        )
    )


# --- scope ----------------------------------------------------------------- #


def test_a_role_less_model_is_untouched():
    validate_role_pairing(_model_in(roles=None, disaggregation=False))


def test_plain_multi_role_without_disaggregation_is_untouched():
    """`roles` without `disaggregation` is not PD, so there is no pairing to
    check — nothing transfers KV between them."""
    validate_role_pairing(
        _model_in(
            _roles(
                prefill_params=["--max-model-len=8192"],
                decode_params=["--max-model-len=4096"],
            ),
            disaggregation=False,
        )
    )


def test_a_sparse_update_is_judged_against_the_stored_roles():
    """A PUT that changes only the model-level parameters would otherwise be
    read as a role-less model, and the contradiction it creates with the
    stored decode override would be accepted by not mentioning it."""
    from gpustack.schemas.models import ModelUpdate

    stored = Model(
        id=1,
        name="m",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/repo",
        owner_principal_id=1,
        roles=_roles(decode_params=["--max-model-len=4096"]),
        disaggregation=DisaggregationSpec(mode=PDModeEnum.VLLM_NIXL),
    )
    update = ModelUpdate(
        name="m",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/repo",
        backend_parameters=["--max-model-len=8192"],
    )

    with rejects("context lengths"):
        validate_role_pairing(update, stored=stored)


# --- the hybrid KV cache manager ------------------------------------------- #

_HMA_ENABLE = "--no-disable-hybrid-kv-cache-manager"
_HMA_DISABLE = "--disable-hybrid-kv-cache-manager"


def test_one_role_re_enabling_hma_is_rejected():
    """A KV connector disables it on its own, so both sides agree by default.
    Turning it back on for one of them is a real divergence, and it is one of
    the factors the connector hashes — so the pair is refused on contact and
    the group never serves."""
    with rejects("hybrid KV cache manager"):
        validate_role_pairing(
            _model_in(
                roles=[
                    RoleSpec(name="prefill", backend_parameters=[_HMA_ENABLE]),
                    RoleSpec(name="decode"),
                ],
            )
        )


def test_both_roles_re_enabling_it_is_allowed():
    validate_role_pairing(
        _model_in(
            roles=[
                RoleSpec(name="prefill", backend_parameters=[_HMA_ENABLE]),
                RoleSpec(name="decode", backend_parameters=[_HMA_ENABLE]),
            ],
        )
    )


def test_the_disabling_spelling_on_one_side_only_is_not_a_divergence():
    """It restates what the connector already does, so the effective values
    still match — rejecting it would refuse a configuration that works, which
    is exactly what a value-equality check would have done here."""
    validate_role_pairing(
        _model_in(
            roles=[
                RoleSpec(name="prefill", backend_parameters=[_HMA_DISABLE]),
                RoleSpec(name="decode"),
            ],
        )
    )


def test_the_last_spelling_wins_as_argparse_reads_it():
    """A role carrying both is read the way the engine reads it, not the way
    the list happens to be ordered."""
    validate_role_pairing(
        _model_in(
            roles=[
                RoleSpec(
                    name="prefill", backend_parameters=[_HMA_ENABLE, _HMA_DISABLE]
                ),
                RoleSpec(name="decode"),
            ],
        )
    )


# --- one side silent -------------------------------------------------------- #
#
# The rules above must also hold when only one role writes the parameter down,
# because that is the ordinary way a group is misconfigured: edit prefill,
# leave decode alone. Substituting the engines' defaults is not the answer —
# an unwritten tp is the member's card count and not 1, an unwritten dtype is
# `auto` and needs the checkpoint to resolve, and block size and KV cache
# layout belong to the platform and the attention backend. So the test is "are
# both effective values determinable", and what is not determinable is
# reported instead of refused.


def _selector(*gpu_ids):
    return GPUSelector(gpu_ids=list(gpu_ids))


def _silent_side_roles(prefill=None, decode=None):
    return [
        prefill or RoleSpec(name="prefill", replicas=1),
        decode or RoleSpec(name="decode", replicas=1),
        RoleSpec(name="router", replicas=1),
    ]


def test_a_decode_pinned_to_one_card_is_narrower_than_a_prefill_at_tp2():
    """The shape a both-sides-written check lets straight through: decode
    writes no tp, but it pins a single card, GPUStack injects the card count
    for a single-worker member, and a decode that runs TP1 under a prefill at
    TP2 is exactly the inversion NIXL reports as an IndexError."""
    with rejects("decode runs tensor parallelism 1, below prefill's 2"):
        validate_role_pairing(
            _model_in(
                _silent_side_roles(
                    prefill=RoleSpec(
                        name="prefill",
                        replicas=1,
                        backend_parameters=["--tensor-parallel-size=2"],
                    ),
                    decode=RoleSpec(
                        name="decode",
                        replicas=1,
                        gpu_selector=_selector("w1:npu:0"),
                    ),
                )
            )
        )


def test_a_decode_that_pins_nothing_is_reported_rather_than_refused():
    """The same prefill against a decode that says nothing at all. Its cards
    are the scheduler's to choose, so its tensor parallelism is not knowable
    from the spec — and holding it to 1 would refuse the two-card decode that
    would have paired perfectly. Accepted, and marked."""
    model_in = _model_in(
        _silent_side_roles(
            prefill=RoleSpec(
                name="prefill",
                replicas=1,
                backend_parameters=["--tensor-parallel-size=2"],
            ),
        )
    )
    validate_role_pairing(model_in)
    assert "tensor parallelism" in undecidable_factors(
        model_in.roles[0], model_in.roles[1], model_in.backend_parameters
    )


def test_a_pin_spread_over_two_workers_says_nothing_about_tp():
    """A member spanning workers has its world size split into tp and pp
    further down the vLLM path, so the pinned count is not the tensor
    parallelism and cannot refuse anything."""
    model_in = _model_in(
        _silent_side_roles(
            prefill=RoleSpec(
                name="prefill",
                replicas=1,
                backend_parameters=["--tensor-parallel-size=4"],
            ),
            decode=RoleSpec(
                name="decode",
                replicas=1,
                gpu_selector=_selector("w1:npu:0", "w2:npu:0"),
            ),
        )
    )
    validate_role_pairing(model_in)


def test_a_role_that_writes_dp_but_not_tp_is_not_read_as_its_card_count():
    """Writing any parallelism flag turns GPUStack's injection off wholesale,
    so a decode with `--data-parallel-size 2` on two pinned cards runs the
    engine's own tp default rather than 2. Which default that is belongs to the
    engine, so nothing here refuses."""
    model_in = _model_in(
        _silent_side_roles(
            prefill=RoleSpec(
                name="prefill",
                replicas=1,
                backend_parameters=["--tensor-parallel-size=4"],
            ),
            decode=RoleSpec(
                name="decode",
                replicas=1,
                backend_parameters=["--data-parallel-size=2"],
                gpu_selector=_selector("w1:npu:0", "w1:npu:1"),
            ),
        )
    )
    validate_role_pairing(model_in)


def test_an_explicit_auto_and_a_silent_side_are_the_same_declaration():
    """`auto` is what the silent side runs, so writing it on one role is not a
    divergence from the other role writing nothing."""
    validate_role_pairing(
        _model_in(_roles(prefill_params=["--dtype=auto"])),
    )
    validate_role_pairing(
        _model_in(_roles(prefill_params=["--kv-cache-dtype", "auto"])),
    )


def test_auto_against_a_concrete_dtype_is_reported_rather_than_refused():
    """Writing `auto` down does not turn a question into an answer. `auto` is
    the checkpoint's dtype, not a third one, so on a float16 checkpoint these
    two are the same run — and telling them apart needs a config file admission
    cannot open. Refusing them with a 400 purely because the two strings
    differ would be a guess dressed as a verdict."""
    for prefill, decode in (
        (["--dtype=auto"], ["--dtype=float16"]),
        (["--dtype=float16"], ["--dtype=auto"]),
        (["--kv-cache-dtype=auto"], ["--kv-cache-dtype=fp8"]),
    ):
        model_in = _model_in(_roles(prefill_params=prefill, decode_params=decode))
        validate_role_pairing(model_in)
        assert undecidable_factors(
            model_in.roles[0], model_in.roles[1], model_in.backend_parameters
        )


def test_two_concrete_values_that_differ_are_still_refused():
    """The line the downgrade above does not cross: with no `auto` on either
    side the spec settles it by itself, and nothing has to be opened to know
    the two roles run different dtypes."""
    with rejects("dtype"):
        validate_role_pairing(
            _model_in(
                _roles(
                    prefill_params=["--dtype=float16"],
                    decode_params=["--dtype=bfloat16"],
                )
            )
        )


def test_a_concrete_dtype_against_silence_is_reported_rather_than_refused():
    """Silence is `auto`, so this is the same undecidable pair as the explicit
    one above, reached without anyone typing the word."""
    model_in = _model_in(_roles(prefill_params=["--dtype=float16"]))
    validate_role_pairing(model_in)
    assert "dtype" in undecidable_factors(
        model_in.roles[0], model_in.roles[1], model_in.backend_parameters
    )


def test_the_underscore_spelling_is_the_same_flag():
    """vLLM's FlexibleArgumentParser takes `--kv_cache_dtype` as readily as
    `--kv-cache-dtype`. A table listing only one of them let a prefill on fp8
    pair with a decode on fp8_e5m2 without ever comparing them."""
    with rejects("KV cache dtype"):
        validate_role_pairing(
            _model_in(
                _roles(
                    prefill_params=["--kv-cache-dtype=fp8"],
                    decode_params=["--kv_cache_dtype=fp8_e5m2"],
                )
            )
        )


def test_the_underscore_spelling_of_the_context_window_is_too():
    with rejects("context lengths"):
        validate_role_pairing(
            _model_in(
                _roles(
                    prefill_params=["--max_model_len=8192"],
                    decode_params=["--max-model-len=4096"],
                )
            )
        )


def test_the_underscore_spelling_of_tp_is_too():
    with rejects("decode runs tensor parallelism 4, below prefill's 8"):
        validate_role_pairing(
            _model_in(
                _roles(
                    prefill_params=["--tp_size=8"],
                    decode_params=["--tensor_parallel_size=4"],
                )
            )
        )


def test_a_flag_written_twice_is_compared_as_argparse_reads_it():
    """Last wins. A prefill carrying `--dtype bfloat16 --dtype float16` runs
    float16, so comparing the first value would clear a pair that diverges —
    and the same read in reverse would refuse one that does not."""
    with rejects("dtype"):
        validate_role_pairing(
            _model_in(
                _roles(
                    prefill_params=["--dtype=bfloat16", "--dtype=float16"],
                    decode_params=["--dtype=bfloat16"],
                )
            )
        )
    validate_role_pairing(
        _model_in(
            _roles(
                prefill_params=["--dtype=float16", "--dtype=bfloat16"],
                decode_params=["--dtype=bfloat16"],
            )
        )
    )


def test_a_context_window_written_twice_is_read_the_same_way():
    with rejects("context lengths"):
        validate_role_pairing(
            _model_in(
                _roles(
                    prefill_params=["--max-model-len=4096", "--max-model-len=8192"],
                    decode_params=["--max-model-len=4096"],
                )
            )
        )


def test_clearing_the_parameters_is_a_side_falling_silent():
    """The decision `test_an_empty_override_does_not_inherit` left open. An
    empty override is not "no opinion" — the role runs the engine's defaults
    while its sibling runs the model's 8192, which is the same hole this
    section exists to close. Still not a 400, because the defaults are not
    resolvable here; reported instead."""
    model_in = _model_in(
        _roles(prefill_params=[]),
        backend_parameters=["--max-model-len=8192"],
    )
    validate_role_pairing(model_in)
    assert "context length" in undecidable_factors(
        model_in.roles[0], model_in.roles[1], model_in.backend_parameters
    )


def test_two_silent_roles_are_not_reported():
    """Both sides take the same default from the same engine on the same
    model, so whatever it resolves to, it resolves to it twice. Marking this
    would put the badge on the ordinary way a group is deployed."""
    model_in = _model_in(_roles())
    validate_role_pairing(model_in)
    assert (
        undecidable_factors(
            model_in.roles[0], model_in.roles[1], model_in.backend_parameters
        )
        == []
    )


def test_a_model_level_value_both_roles_inherit_is_not_reported():
    model_in = _model_in(_roles(), backend_parameters=["--dtype=float16"])
    validate_role_pairing(model_in)
    assert (
        undecidable_factors(
            model_in.roles[0], model_in.roles[1], model_in.backend_parameters
        )
        == []
    )


def test_a_multi_replica_pin_is_not_divided_by_its_replica_count():
    """`_set_gpu_count` reads a role's selector whole — the division by
    `replicas` belongs to `set_model_gpus_per_replica`, which runs on the Model
    and deliberately never reaches a role's projection. Reproducing that
    arithmetic here would have read this decode as TP1 and refused a pair that
    runs."""
    model_in = _model_in(
        _silent_side_roles(
            prefill=RoleSpec(
                name="prefill",
                replicas=1,
                backend_parameters=["--tensor-parallel-size=2"],
            ),
            decode=RoleSpec(
                name="decode",
                replicas=2,
                gpu_selector=_selector("w1:npu:0", "w1:npu:1"),
            ),
        )
    )
    validate_role_pairing(model_in)
    assert "tensor parallelism" in undecidable_factors(
        model_in.roles[0], model_in.roles[1], model_in.backend_parameters
    )


def test_an_explicit_gpus_per_replica_is_the_authority():
    """The field `_set_gpu_count` reads first, so when it is set the pin is
    unambiguous however many replicas the role runs."""
    decode = RoleSpec(name="decode", replicas=2)
    decode.gpu_selector = GPUSelector(
        gpu_ids=["w1:npu:0", "w1:npu:1"], gpus_per_replica=1
    )
    with rejects("decode runs tensor parallelism 1, below prefill's 2"):
        validate_role_pairing(
            _model_in(
                _silent_side_roles(
                    prefill=RoleSpec(
                        name="prefill",
                        replicas=1,
                        backend_parameters=["--tensor-parallel-size=2"],
                    ),
                    decode=decode,
                )
            )
        )
