"""`{{name}}` substitution, with no backend gating.

One implementation, two callers: the run-command rewrite in
`InferenceBackend.replace_command_param` and the env-value rewrite in
`InferenceServer._get_configured_env`.

The env-value case is why this module exists. `replace_command_param` is
reached only when a version config has a `run_command` and *no*
`built_in_frameworks` — and "has built_in_frameworks" is the definition of a
built-in backend — so substitution living there alone never reaches the
built-in vLLM and SGLang paths, and `VLLM_NIXL_SIDE_CHANNEL_HOST={{worker_ip}}`
reaches the container verbatim: measured as `ZMQError: No such device
(addr='tcp://{{worker_ip}}:5600')`. Rendering has to happen for every backend,
so it cannot be a method on a backend.

Variable names may contain dots — `ports.kv_side_channel.count`,
`roles.prefill.tensor_parallel_size` — and the context is a flat mapping keyed
by the whole dotted name. Structure is the caller's business; this module only
substitutes.
"""

import logging
import re
from typing import Dict, Iterable, List, Mapping, Optional

logger = logging.getLogger(__name__)

# No whitespace inside the braces. `{{ worker_ip }}` is not a placeholder and
# is not silently treated as one: a value that fails to render has to look
# different from one that rendered to something, or it becomes the literal
# that reached the engine above.
_PLACEHOLDER = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_.]*)\}\}")


def placeholders(text: Optional[str]) -> List[str]:
    """Every placeholder name in `text`, in order, with duplicates kept."""
    if not text:
        return []
    return _PLACEHOLDER.findall(text)


def render(
    text: Optional[str],
    variables: Mapping[str, object],
    *,
    context: str = "",
) -> Optional[str]:
    """Substitute `{{name}}` from `variables`.

    An unknown name is left as it is rather than blanked, because a run
    command may legitimately reference something resolved later, and blanking
    would turn a missing value into a plausible-looking wrong one. Unknown
    names are logged with `context` so they stop being invisible.
    """
    if not text:
        return text

    missing = []

    def _substitute(match: "re.Match[str]") -> str:
        name = match.group(1)
        if name not in variables:
            missing.append(name)
            return match.group(0)
        value = variables[name]
        return "" if value is None else str(value)

    rendered = _PLACEHOLDER.sub(_substitute, text)
    if missing:
        where = f" in {context}" if context else ""
        logger.warning(
            "Left %s unresolved%s: no value for %s. "
            "The placeholder reaches the process verbatim.",
            "a placeholder" if len(missing) == 1 else "placeholders",
            where,
            ", ".join(f"{{{{{name}}}}}" for name in dict.fromkeys(missing)),
        )
    return rendered


def render_values(
    env: Optional[Dict[str, str]],
    variables: Mapping[str, object],
) -> Optional[Dict[str, str]]:
    """Render every *value* of `env`; keys are left alone.

    Values do not see each other. Allowing one env value to reference another
    would make the result depend on iteration order and admit cycles, and
    nothing needs it: the connector parameters that drove this are all
    expressed in terms of the deployment, not of each other.
    """
    if not env:
        return env
    return {
        key: render(value, variables, context=f"env {key}")
        for key, value in env.items()
    }


def deployment_variables(
    *,
    model_path: Optional[str] = None,
    port: Optional[int] = None,
    worker_ip: Optional[str] = None,
    model_name: Optional[str] = None,
    gpu_count: Optional[int] = None,
    gpu_ids: Optional[Iterable[int]] = None,
    role: Optional[str] = None,
    group_id: Optional[str] = None,
) -> Dict[str, object]:
    """The variables every backend can resolve from a scheduled instance.

    The six original names are always present, and an absent one renders to
    "" (`port` to `str(None)`), because that is exactly what
    `replace_command_param` has always done and the run-command path must not
    change behaviour. `role` and `group_id` are omitted when there is no role,
    so a non-PD deployment referencing them gets the unresolved warning rather
    than a silent empty string.

    Names outside this set are resolved by their owners and merged in — that
    is where "unresolved" is a real signal: `ports.<name>` and
    `ports.<name>.count` once the named-port bands are allocated, `net_device`
    from the worker's interface, `peers.<role>` when the router config is
    rendered, and `roles.<role>.<field>` for the cross-role references a
    disaggregated engine config needs.
    """
    variables: Dict[str, object] = {
        "model_path": model_path or "",
        "port": port,
        "worker_ip": worker_ip or "",
        "model_name": model_name or "",
        "gpu_count": "" if gpu_count is None else gpu_count,
        "gpu_ids": ",".join(str(i) for i in gpu_ids) if gpu_ids else "",
    }
    if role is not None:
        variables["role"] = role
    if group_id is not None:
        variables["group_id"] = group_id
    return variables
