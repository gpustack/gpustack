import re
import sys
import sysconfig
from os.path import dirname, abspath, join
import shutil
from typing import List, Literal, Optional, Tuple, Union
import shlex

from gpustack_runtime.deployer.__utils__ import compare_versions


_TRUTHY_VALUES = frozenset({"1", "true", "yes", "on", "t", "y"})
_FALSY_VALUES = frozenset({"0", "false", "no", "off", "f", "n"})


def is_command_available(command_name):
    """
    Use `shutil.which` to determine whether a command is available.

    Args:
    command_name (str): The name of the command to check.

    Returns:
    bool: True if the command is available, False otherwise.
    """

    return shutil.which(command_name) is not None


def _iter_param_pairs(parameters):
    """
    Yield ``(key_without_dashes, value_or_None)`` over the argv stream produced
    by :func:`flatten_to_argv`.

    Bare flags yield ``(key, None)``. Multi-value parameters (``--lora-modules
    v1 v2``) yield ``(key, v1)`` only — present consumers only need to detect
    a key's presence or read its first value; nobody needs the value list.
    """
    tokens = flatten_to_argv(parameters)
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if not is_parameter_key(tok):
            i += 1
            continue
        if "=" in tok:
            key, _, value = tok.partition("=")
            yield key.lstrip("-"), value
            i += 1
            continue
        # Bare --key. The next token is its value iff it is not itself a key.
        if i + 1 < n and not is_parameter_key(tokens[i + 1]):
            yield tok.lstrip("-"), tokens[i + 1]
            i += 2
        else:
            yield tok.lstrip("-"), None
            i += 1


def find_parameter(parameters: List[str], param_names: List[str]) -> Optional[str]:
    """
    Return the value of the first parameter whose key (without leading dashes)
    is in ``param_names``. ``None`` if not found or the key is flag-only.
    """
    if parameters is None:
        return None
    for key, value in _iter_param_pairs(parameters):
        if key in param_names and value is not None:
            return value
    return None


def find_int_parameter(parameters: List[str], param_names: List[str]) -> Optional[int]:
    """
    Find specified integer parameter by name from the parameters.
    Return the integer value of the parameter if found, otherwise return None.
    """
    value = find_parameter(parameters, param_names)
    if value is not None:
        try:
            return int(value)
        except ValueError:
            return None
    return None


def find_bool_parameter(parameters: List[str], param_names: List[str]) -> bool:
    """
    Return whether any parameter whose key is in ``param_names`` is "enabled".

    Enabling is interpreted permissively:
    - bare flag (``--foo``) -> True
    - common truthy values (``--foo true`` / ``--foo=1`` / ``--foo yes`` ...) -> True
    - common falsy values (``--foo false`` / ``--foo=0`` / ``--foo no`` ...) -> False
    - any other value -> True (the key was explicitly declared; the backend
      validates the value itself, GPUStack should not silently treat unrecognized
      values as "off")

    Values are matched case-insensitively. The first occurrence wins.
    """
    if parameters is None:
        return False
    for key, value in _iter_param_pairs(parameters):
        if key not in param_names:
            continue
        if value is None:
            return True
        if value.lower() in _FALSY_VALUES:
            return False
        return True
    return False


def subordinates_serve_api(backend_parameters: Optional[List[str]]) -> bool:
    """Whether each subordinate worker runs its own API server (non-headless) and
    must be registered as an independent gateway backend. True for any non-internal
    load-balance mode (hybrid-LB / external-LB; see
    ``resolve_data_parallel_load_balance_mode``); internal-LB followers stay
    ``--headless`` so only the leader serves.
    """
    return resolve_data_parallel_load_balance_mode(backend_parameters) != "internal"


def resolve_data_parallel_load_balance_mode(
    backend_parameters: Optional[List[str]],
) -> str:
    """Resolve the effective vLLM data-parallel load-balance mode from the
    user-supplied backend parameters.

    ``--data-parallel-hybrid-lb`` -> ``"hybrid"``; ``--data-parallel-external-lb``
    or a pinned ``--data-parallel-rank`` -> ``"external"`` (vLLM implies
    external-LB then). Defaults to ``"internal"``.
    """
    if find_bool_parameter(backend_parameters, ["data-parallel-hybrid-lb"]):
        return "hybrid"
    if (
        find_bool_parameter(backend_parameters, ["data-parallel-external-lb"])
        or find_parameter(backend_parameters, ["data-parallel-rank"]) is not None
    ):
        return "external"
    return "internal"


def get_versioned_command(command_name: str, version: str) -> str:
    """
    Get the versioned command name.
    """
    if command_name.endswith(".exe"):
        return f"{command_name[:-4]}_{version}.exe"

    return f"{command_name}_{version}"


def get_command_path(command_name: str) -> str:
    """
    Return the full path of sepcified command. Supports both frozen and python base environments.
    """
    base_path = (
        dirname(sys.executable)
        if getattr(sys, 'frozen', False)
        else sysconfig.get_path("scripts")
    )
    return abspath(join(base_path, command_name))


def extend_args_no_exist(
    arguments: List[str], *args: Union[str, Tuple[str, str]]
) -> None:
    """
    Extend arguments list with key-value pairs only if the key is not already present.

    This function prevents duplicate parameters when user-defined backend parameters
    may conflict with system-generated ones.

    Args:
        arguments: The list of arguments to extend (modified in place)
        *args: Variable number of arguments, each can be:
               - A tuple of (key, value) like ("--host", "127.0.0.1")
               - A single string key like "--enable-metrics" (flag without value)

    Examples:
        extend_args_no_exist(args, ("--host", "127.0.0.1"), ("--port", "8080"))
        extend_args_no_exist(args, "--enable-metrics")
    """
    for arg in args:
        if isinstance(arg, tuple):
            key, value = arg
            if not any(
                existing_arg == key or existing_arg.startswith(f"{key}=")
                for existing_arg in arguments
            ):
                arguments.extend([key, value])
        else:
            # Single flag without value
            if arg not in arguments:
                arguments.append(arg)


def format_backend_parameters(parameters: Optional[List[str]]) -> List[str]:
    """
    Format flattened command-line tokens as backend parameter entries.

    Examples:
        ["--max-model-len", "8192", "--enable-prefix-caching"]
            -> ["--max-model-len=8192", "--enable-prefix-caching"]
        ["--dtype=float16"] -> ["--dtype=float16"]
        ["--lora-modules", "{a}", "{b}", "{c}"]
            -> ["--lora-modules", "{a}", "{b}", "{c}"]
    """
    if not parameters:
        return []

    formatted = []
    index = 0
    while index < len(parameters):
        parameter = parameters[index]
        if not parameter.startswith("-") or "=" in parameter:
            formatted.append(parameter)
            index += 1
            continue

        # Collect every consecutive non-flag token as a value of this flag.
        values: List[str] = []
        j = index + 1
        while j < len(parameters) and not _looks_like_parameter(parameters[j]):
            values.append(parameters[j])
            j += 1

        if len(values) == 0:
            formatted.append(parameter)
        elif len(values) == 1:
            formatted.append(f"{parameter}={values[0]}")
        else:
            # Multi-value flag: keep flag + values as separate tokens
            # (space-separated on the command line) so none get orphaned.
            formatted.append(parameter)
            formatted.extend(values)
        index = j

    return formatted


def _looks_like_parameter(value: str) -> bool:
    if not value.startswith("-"):
        return False

    # Negative numeric values are parameter values, not flags.
    try:
        float(value)
        return False
    except ValueError:
        return True


def _mask_json_segments(text: str):
    """
    Replace JSON segments with placeholders before shlex.split.

    This prevents shlex from incorrectly splitting JSON values that contain
    spaces or special characters. For example:
        --config={"key": "value with spaces"}
    Without masking, shlex would split on the spaces inside the JSON.
    """

    result = []
    mapping = {}

    i = 0
    n = len(text)
    json_id = 0

    while i < n:
        # Look for pattern: '=' followed by JSON start ('{' or '[')
        if text[i] == "=" and i + 1 < n and text[i + 1] in "{[":
            start = i + 1
            j = start

            # Track state for proper JSON parsing
            depth = 0  # Bracket nesting level
            in_string = False  # Whether we're inside a quoted string
            escape = False  # Whether previous char was backslash

            # Manually parse to find the matching closing bracket
            while j < n:
                c = text[j]

                # Handle escape sequences (e.g., \" inside strings)
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                # Toggle string state on unescaped quotes
                elif c == '"':
                    in_string = not in_string
                # Only count brackets outside of strings
                elif not in_string:
                    if c in "{[":
                        depth += 1
                    elif c in "}]":
                        depth -= 1
                        # Found matching closing bracket
                        if depth == 0:
                            j += 1
                            break

                j += 1

            # Extract the JSON segment
            json_text = text[start:j]

            # Wrap the id in ASCII Unit Separator (\x1f) so the placeholder
            # cannot collide with any legitimate user input — \x1f is a control
            # character that does not appear in CLI parameter strings, while
            # shlex still preserves it intact within a token.
            placeholder = f"\x1fJSON_{json_id}\x1f"
            json_id += 1

            # Store mapping for later restoration
            mapping[placeholder] = json_text

            # Replace JSON with placeholder in result
            result.append("=" + placeholder)
            i = j
            continue

        # Copy non-JSON characters as-is
        result.append(text[i])
        i += 1

    return "".join(result), mapping


def _unmask_json(tokens, mapping):
    """Restore JSON placeholders back to original JSON text."""
    restored = []
    for t in tokens:
        # Replace all placeholders in this token with original JSON
        for k, v in mapping.items():
            if k in t:
                t = t.replace(k, v)
        restored.append(t)
    return restored


def _normalize_continuations(text: str) -> str:
    """
    Tolerate shell line-continuation backslashes that appear when users paste
    multi-line command snippets (e.g. vLLM recipes) into the parameters field.

    Without this, shlex.split raises ``ValueError: No escaped character`` on a
    trailing ``\\``.
    """
    text = re.sub(r'\\\r?\n', ' ', text)

    match = re.search(r'(\\+)\s*$', text)
    if match and len(match.group(1)) % 2 == 1:
        text = text[: match.start()] + match.group(1)[:-1]

    return text


def safe_split(expr: str):
    """
    Split parameter string using shlex while preserving JSON values.

    Process:
    1. Mask JSON segments with placeholders
    2. Normalize shell line-continuations and stray trailing backslashes
    3. Use shlex.split (which respects quotes and escapes)
    4. Restore original JSON text
    """
    masked, mapping = _mask_json_segments(expr)
    masked = _normalize_continuations(masked)
    tokens = shlex.split(masked)
    return _unmask_json(tokens, mapping)


def is_parameter_key(token: str) -> bool:
    """
    Return True when ``token`` looks like a CLI flag/option key (e.g. ``--foo``).

    Numeric tokens like ``-1``, ``-0.5``, ``-.5`` are treated as values, not
    keys. The POSIX ``--`` end-of-options marker is also not a key.
    """
    if not token.startswith("-") or len(token) <= 1:
        return False
    if token == "--":
        return False
    try:
        float(token)
        return False
    except ValueError:
        return True


def flatten_to_argv(parameters: Optional[List[str]]) -> List[str]:
    """
    Reduce a ``backend_parameters`` list to its argv-equivalent flat token list.

    ``backend_parameters`` is semantically a concatenated argv. Each element is
    one of:

    - A bare argv token: ``"--host"`` / ``"0.0.0.0"`` / a JSON blob / a path
      with spaces. Treated as a single token; never re-split.
    - A full ``--key [value]`` / ``--key=value`` string or a whole pasted
      command line. Re-tokenized via :func:`safe_split` (shlex + JSON masking
      + line-continuation handling).

    The distinction is made by inspecting the element's leading word: if it
    looks like a CLI key, the element is re-tokenized; otherwise it passes
    through verbatim. This preserves opaque value tokens (multi-value lora
    JSON, paths with spaces, ``-0.5`` negatives) while still expanding pasted
    command lines.
    """
    if not parameters:
        return []

    tokens: List[str] = []
    for entry in parameters:
        if entry is None:
            continue
        stripped = entry.strip()
        if not stripped:
            continue
        first_word = stripped.split(None, 1)[0]
        if is_parameter_key(first_word):
            tokens.extend(safe_split(stripped))
        else:
            tokens.append(stripped)
    return tokens


ExecutorBackend = Literal["ray", "mp"]

# vLLM v0.18.0 removed Ray from its default dependencies. Only relevant for
# user-supplied custom images (see should_default_to_ray for the rationale).
_VLLM_RAY_DROPPED_FROM_DEFAULTS = "0.18.0"


def should_default_to_ray(backend_version: Optional[str]) -> bool:
    """
    Whether GPUStack should default to the Ray executor backend when the user
    does not explicitly choose one.

    gpustack-runner images bundle Ray themselves regardless of vLLM version,
    so Ray is always available there. The only case where Ray may be absent
    is user-supplied custom images — identified by a ``-custom`` suffix in
    ``backend_version`` — running vLLM >= 0.18.0, where upstream dropped Ray
    from default dependencies. In that case we default to ``mp`` to avoid a
    startup failure.
    """
    if not backend_version or "-custom" not in backend_version:
        return True
    try:
        return compare_versions(backend_version, _VLLM_RAY_DROPPED_FROM_DEFAULTS) < 0
    except Exception:
        return True


def resolve_executor_backend(
    backend_parameters: Optional[List[str]],
    backend_version: Optional[str],
) -> ExecutorBackend:
    """
    Resolve the dispatch branch for vLLM distributed execution.

    Returns ``"ray"`` if GPUStack should take the Ray sidecar path (legacy),
    ``"mp"`` for the MP multi-node headless path (current).

    Precedence:
    1. User-supplied ``--distributed-executor-backend`` wins. Any explicit value
       other than ``"mp"`` is routed to the ray branch — GPUStack does not inject
       its own MP topology arguments and leaves the choice to vLLM.
    2. Otherwise the default depends on ``backend_version`` via
       :func:`should_default_to_ray`.
    """
    user_value = find_parameter(backend_parameters, ["distributed-executor-backend"])
    if user_value is not None:
        return "mp" if user_value == "mp" else "ray"

    return "ray" if should_default_to_ray(backend_version) else "mp"
