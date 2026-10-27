"""Create-time ``{{generated_*}}`` placeholder substitution for GPU instance specs.

Templates may carry placeholders (mustache-style, aligned with the
inference-backend ``{{var_name}}`` convention) that the backend resolves once
at create time; the concrete values are persisted in the spec. Unlike the
inference backends — which store placeholders in the DB and resolve at launch
time — GPU instances must resolve at create time: the UI reads generated
values back (e.g. to build access links), and stop/start replays the stored
spec, so regeneration on every apply would change the credential.
"""

import re
from typing import Callable, Dict

from gpustack.schemas.gpu_instances import GPUInstanceSpec
from gpustack.security import generate_secret_key

_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")

# Placeholder name -> value provider. Providers are called lazily and at most
# once per spec: every occurrence of the same placeholder resolves to the same
# value. Unknown placeholders are left unchanged, mirroring the
# inference-backend ``_resolve_env_vars`` semantics.
_PLACEHOLDER_PROVIDERS: Dict[str, Callable[[], str]] = {
    "generated_token": generate_secret_key,
}


def substitute_generated_placeholders(spec: GPUInstanceSpec) -> None:
    """Resolve ``{{generated_*}}`` placeholders in ``spec`` in place.

    Scans ``spec.command`` items and ``spec.ports[].access_params`` values. A
    spec without placeholders is left semantically untouched.
    """
    values: Dict[str, str] = {}

    def replace(match: re.Match) -> str:
        name = match.group(1)
        provider = _PLACEHOLDER_PROVIDERS.get(name)
        if provider is None:
            return match.group(0)
        if name not in values:
            values[name] = provider()
        return values[name]

    if spec.command:
        spec.command = [_PLACEHOLDER_RE.sub(replace, item) for item in spec.command]
    for port in spec.ports or []:
        if port.access_params:
            port.access_params = {
                key: _PLACEHOLDER_RE.sub(replace, value)
                for key, value in port.access_params.items()
            }
