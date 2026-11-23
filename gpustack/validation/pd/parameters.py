"""What prefill and decode have to agree on, and when the answer is knowable.

"Did both roles write this parameter?" is a different question from "do the
two roles run the same value", because a role that writes nothing does not run
nothing -- it runs the engine's default, which is a value like any other. Only
the second question catches the ordinary way a group is misconfigured: edit
prefill, leave decode alone.

Substituting the defaults is the obvious repair and is wrong for every
parameter below, which is why this module exists instead.

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


def _same(left: Optional[str], right: Optional[str]) -> bool:
    if left is None or right is None:
        return False
    return left.strip().lower() == right.strip().lower()
