"""What prefill and decode have to agree on, and when the answer is knowable.

"Did both roles write this parameter?" is a different question from "do the
two roles run the same value", because a role that writes nothing does not run
nothing -- it runs the engine's default, which is a value like any other. Only
the second question catches the ordinary way a group is misconfigured: edit
prefill, leave decode alone.

Substituting the defaults is the obvious repair and is wrong for every
parameter below, which is why this module exists instead.

* `--tensor-parallel-size` unwritten is **not 1**. GPUStack injects the
  member's own card count when a role is silent and lands on one worker
  (`get_auto_parallelism_arguments`, in both the vLLM and the SGLang backend),
  so a prefill at TP2 beside a silent two-card decode is a correct deployment
  that a static 1 would refuse.
* `--dtype` and `--kv-cache-dtype` default to `auto`, whose meaning is "follow
  the checkpoint". Resolving it needs the model's config, and this runs on the
  admission path with no session to fetch one.
* `--block-size` and `--kv-cache-layout` are decided by the platform and the
  attention backend, `VLLM_KV_CACHE_LAYOUT` included. Any constant written
  here would drift into a false alarm on some future vLLM release.

So the question this module answers is the third one: is each side's effective
value *determinable from the spec*? Both determinable is a judgement, and a
mismatch there is refused at admission. One side silent with a default nobody
here can resolve is not a judgement, and saying so is what the
`pairing_unverified` degradation is for. The one thing never done is
guessing, because a guess on this path becomes a red compatibility error in
the deploy form (`evaluate_model_input` turns the refusal into one) on a
deployment that would have run.
"""

from typing import Dict, List, Optional, Sequence, Tuple

from gpustack.schemas.pd_modes import PDTensorParallelPairingEnum
from gpustack.utils.command import find_last_int_parameter, find_last_parameter

# The parameters that must agree between prefill and decode, with every
# spelling the engines accept for them.
#
# Both the hyphen and the underscore form, always. vLLM's
# `FlexibleArgumentParser` accepts `--kv_cache_dtype` as readily as
# `--kv-cache-dtype`, so a table listing one of them lets a prefill on
# `--kv-cache-dtype fp8` pair with a decode on `--kv_cache_dtype fp8_e5m2` --
# two different values that never get compared. The backends' own alias lists
# (`vllm.py`) carry both for the same reason.
#
# The split is not stylistic: the first entry is the one the engines do NOT
# check, and the rest are ones they do -- checked here anyway so the report
# names the role rather than surfacing as a geometry assertion inside a
# container.
PAIRING_MAX_LEN = [
    "max-model-len",
    "max_model_len",
    "context-length",
    "context_length",
]
PAIRING_TP = [
    "tensor-parallel-size",
    "tensor_parallel_size",
    "tp",
    "tp-size",
    "tp_size",
]
PAIRING_MUST_MATCH: Dict[str, List[str]] = {
    "dtype": ["dtype"],
    "KV cache dtype": ["kv-cache-dtype", "kv_cache_dtype"],
    "block size": ["block-size", "block_size", "page-size", "page_size"],
    "KV cache layout": ["kv-cache-layout", "kv_cache_layout"],
}

PAIRING_FOLLOWS_THE_MODEL = frozenset({"dtype", "KV cache dtype"})
"""The factors whose default is `auto`, i.e. "whatever the checkpoint says".

Two consequences, and they pull in opposite directions.

Silence and an explicit `auto` are the *same declaration*, so a prefill writing
`--dtype auto` beside a decode writing nothing is a pair that agrees rather
than one that cannot be judged.

And `auto` on either side against a concrete value is **undecidable, not a
mismatch**. `auto` is not a third dtype -- it resolves to the checkpoint's, and
on a float16 checkpoint `--dtype auto` and `--dtype float16` are the same run.
Refusing that pair needs the model's config, which this path has no session to
fetch, so the refusal would be a guess dressed as a verdict -- and on the
deploy form it would show up as a red compatibility error on a deployment that
works. Two *concrete* values that differ are still refused: there the spec
alone settles it.
"""

_FOLLOWS_THE_MODEL_DEFAULT = "auto"

# Every parallelism flag, not just tp. Writing any one of them turns off
# GPUStack's injection wholesale (`get_auto_parallelism_arguments` returns
# early on the first one it finds), so a role that writes only `--dp` runs the
# engine's own tp default rather than its card count -- and which default that
# is belongs to the engine.
PAIRING_ANY_PARALLELISM = PAIRING_TP + [
    "data-parallel-size",
    "data_parallel_size",
    "dp",
    "dp-size",
    "dp_size",
    "pipeline-parallel-size",
    "pipeline_parallel_size",
    "pp",
    "pp-size",
    "pp_size",
]

AGREE = "agree"
DIFFER = "differ"
UNDECIDABLE = "undecidable"


def role_parameters(role, model_parameters) -> List[str]:
    """A role's effective engine parameters.

    `None` inherits, an empty list does not -- a role that deliberately clears
    the model's parameters must not silently get them back. Note that clearing
    them is itself a side falling silent, so the factors the model declared
    become undecidable rather than agreed.
    """
    if role is None:
        return list(model_parameters or [])
    if getattr(role, "backend_parameters", None) is None:
        return list(model_parameters or [])
    return list(role.backend_parameters)


def compare_must_match(
    label: str,
    names: Sequence[str],
    prefill_params: Sequence[str],
    decode_params: Sequence[str],
) -> str:
    """Whether the two sides agree on one must-match factor, or cannot be told.

    Values are read last-wins, as argparse reads them, and compared case- and
    whitespace-insensitively: the engines' own choice lists are lowercase, so
    two spellings of one choice are one value and refusing them would be an
    invented divergence.
    """
    prefill_value = find_last_parameter(prefill_params, names)
    decode_value = find_last_parameter(decode_params, names)

    if label in PAIRING_FOLLOWS_THE_MODEL:
        # Silence is `auto` here, so normalising both sides first makes
        # "one wrote auto, the other wrote nothing" the agreement it is.
        prefill_value = prefill_value or _FOLLOWS_THE_MODEL_DEFAULT
        decode_value = decode_value or _FOLLOWS_THE_MODEL_DEFAULT
        if _same(prefill_value, decode_value):
            return AGREE
        if _same(prefill_value, _FOLLOWS_THE_MODEL_DEFAULT) or _same(
            decode_value, _FOLLOWS_THE_MODEL_DEFAULT
        ):
            # `auto` is the checkpoint's dtype, not a third one, so whether it
            # equals the other side's concrete value is a question about a file
            # this path cannot open.
            return UNDECIDABLE
        return DIFFER

    if prefill_value is not None and decode_value is not None:
        return AGREE if _same(prefill_value, decode_value) else DIFFER
    if prefill_value is None and decode_value is None:
        # Both take the same default from the same engine on the same model.
        # Whatever it resolves to, it resolves to it twice.
        return AGREE
    return UNDECIDABLE


def compare_max_model_len(
    prefill_params: Sequence[str], decode_params: Sequence[str]
) -> Tuple[str, Optional[int], Optional[int]]:
    """The context-length verdict, with the two numbers the message needs.

    Presence is read separately from the number, because an integer parse
    answers None for two different things: the key was never written, and the
    key was written in a form this cannot read. vLLM takes `--max-model-len
    128k`, so the second is reachable -- and reading both as "never written"
    put a prefill at `128k` beside a decode at `32k` in the branch that means
    "neither said anything, so both take the window from the model config",
    and called them agreed. That is the mismatch this comparison exists for,
    and the one that surfaces as a long prompt failing after prefill has
    already been paid for.
    """
    prefill_raw = find_last_parameter(prefill_params, PAIRING_MAX_LEN)
    decode_raw = find_last_parameter(decode_params, PAIRING_MAX_LEN)
    prefill_len = find_last_int_parameter(prefill_params, PAIRING_MAX_LEN)
    decode_len = find_last_int_parameter(decode_params, PAIRING_MAX_LEN)
    if prefill_len is not None and decode_len is not None:
        verdict = AGREE if prefill_len == decode_len else DIFFER
    elif prefill_raw is None and decode_raw is None:
        # Both roles take the window out of the same model config.
        verdict = AGREE
    elif prefill_raw is not None and prefill_raw == decode_raw:
        # Written the same way on both sides. Whatever the engine makes of a
        # spelling this cannot parse, it makes the same thing of it twice.
        verdict = AGREE
    else:
        verdict = UNDECIDABLE
    return verdict, prefill_len, decode_len


def selector_spans_workers(role) -> bool:
    """Whether this role's own pin puts one replica on more than one machine.

    Told apart from "not pinned at all", which `selector_cards_per_replica`
    cannot do -- both answer None there, and a caller that reads None as
    "nothing was said" will happily substitute a number of its own. For a
    distributed member that number is wrong by construction: the world size is
    split into tp and pp much further down the vLLM path, and a guess made here
    lands in a descriptor the engine then contradicts.

    False for an unpinned role, which is the ordinary case and says nothing
    either way.
    """
    selector = getattr(role, "gpu_selector", None)
    gpu_ids = getattr(selector, "gpu_ids", None) if selector is not None else None
    if not gpu_ids:
        return False
    # "worker_name:device:index" -- more than one worker name and the member is
    # distributed, so the card count is not the world size.
    return len({str(gpu_id).rsplit(":", 2)[0] for gpu_id in gpu_ids}) > 1


def selector_cards_per_replica(role) -> Optional[int]:
    """How many cards one replica of this role is pinned to, when the pin is
    unambiguous about it.

    Read off the role's **own** `gpu_selector` and never the Model's. A
    model-level selector is the pool the whole deployment draws from; dividing
    it by one role's replica count would invent a per-role number, and invent
    the same one for the other role.

    Deliberately silent unless the count is exact, because everything this
    feeds is a refusal.

    `gpus_per_replica` is the authority when it is set, because that is the
    field `_set_gpu_count` reads. When it is not, the count is `len(gpu_ids)`
    and *not* `len(gpu_ids) // replicas`: the division belongs to
    `set_model_gpus_per_replica`, which runs on the Model and deliberately does
    not reach a role's projection, so a role's selector is read whole. Rather
    than reproduce that asymmetry this answers only for a single-replica role,
    where the two readings coincide and no arithmetic is being guessed at.

    A pin spread over several workers is silent for a different reason: that is
    a distributed member, and `cal_distributed_parallelism_arguments` splits its
    world size into tp and pp much further down the vLLM path.
    """
    if selector_spans_workers(role):
        return None

    selector = getattr(role, "gpu_selector", None)
    gpu_ids = getattr(selector, "gpu_ids", None) if selector is not None else None
    if not gpu_ids:
        return None

    per_replica = getattr(selector, "gpus_per_replica", None)
    if not per_replica:
        if (getattr(role, "replicas", None) or 1) != 1:
            return None
        per_replica = len(gpu_ids)
    return per_replica if per_replica >= 1 else None


def effective_tensor_parallelism(role, parameters: Sequence[str]) -> Optional[int]:
    """The tensor parallelism this role will actually run, or None when the
    spec cannot say.

    Three cases, and the middle one is the whole reason this is not a
    `find_int_parameter` call:

    * the role writes a tp -- that is the answer, whatever its cards say;
    * the role writes no parallelism at all and pins its own cards -- GPUStack
      injects `--tensor-parallel-size <cards>` for a single-worker member, so
      the pinned count is the answer, and a role pinned to a single card really
      does run TP1. That is the mismatch this makes catchable: prefill at TP2
      beside a decode pinned to one card is refused today by nothing at all;
    * anything else -- no pin, or a role writing dp or pp but not tp, or a pin
      spread over workers -- has no answer here, and the caller marks the
      deployment rather than refusing it.
    """
    declared = find_last_int_parameter(parameters, PAIRING_TP)
    if declared is not None:
        return declared
    if find_last_parameter(parameters, PAIRING_ANY_PARALLELISM) is not None:
        return None
    return selector_cards_per_replica(role)


def tensor_parallel_rule(mode) -> PDTensorParallelPairingEnum:
    """The direction a resolved recipe imposes on prefill vs decode.

    The direction belongs to the KV connector rather than to PD, which is why
    it is read off the mode at all: NIXL needs decode at least as wide as
    prefill, vllm-ascend's Mooncake needs the opposite (Huawei's reference
    deployment is prefill TP4 / decode TP1), and `custom` injects no connector
    GPUStack knows. A mode the catalog could not resolve is held to the NIXL
    rule, which is what every mode was held to before the rule became
    declarable.
    """
    if mode is None:
        return PDTensorParallelPairingEnum.DECODE_GE_PREFILL
    return mode.pairing.tensor_parallel


def violates_tensor_parallel_direction(
    rule: PDTensorParallelPairingEnum, *, prefill_tp: int, decode_tp: int
) -> bool:
    """Whether these two widths break the rule. One place, because admission
    and placement have to answer it the same way."""
    if rule == PDTensorParallelPairingEnum.DECODE_GE_PREFILL:
        return decode_tp < prefill_tp
    if rule == PDTensorParallelPairingEnum.PREFILL_GE_DECODE:
        return prefill_tp < decode_tp
    return False


def undecidable_factors(prefill, decode, model_parameters) -> List[str]:
    """The pairing factors where one side declared and the other went silent.

    Not a list of mismatches -- a mismatch between two determinable values is
    refused at admission and never reaches here. This is the residue: the
    places where GPUStack knows one side's value, cannot resolve the other's,
    and therefore did not check. Naming that is the point, because the
    alternative the user experiences is a check they believe happened.

    Both sides silent is deliberately not on the list. Two roles taking the
    same default from the same engine on the same model agree whatever it
    resolves to, and marking them would put the badge on the ordinary way a
    group is deployed.
    """
    prefill_params = role_parameters(prefill, model_parameters)
    decode_params = role_parameters(decode, model_parameters)

    factors: List[str] = []
    if compare_max_model_len(prefill_params, decode_params)[0] == UNDECIDABLE:
        factors.append("context length")

    prefill_tp = effective_tensor_parallelism(prefill, prefill_params)
    decode_tp = effective_tensor_parallelism(decode, decode_params)
    if (prefill_tp is None) != (decode_tp is None):
        factors.append("tensor parallelism")

    for label, names in PAIRING_MUST_MATCH.items():
        if (
            compare_must_match(label, names, prefill_params, decode_params)
            == UNDECIDABLE
        ):
            factors.append(label)

    return factors


def _same(left: Optional[str], right: Optional[str]) -> bool:
    if left is None or right is None:
        return False
    return left.strip().lower() == right.strip().lower()
