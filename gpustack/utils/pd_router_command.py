"""Restricted Router command templates, parsed without executing template code."""

import re
import shlex
import uuid
from typing import Dict, List, Literal, Mapping, Optional, Set

from jinja2 import Environment, TemplateSyntaxError, nodes

_PEER_NAMES = frozenset({"prefill", "decode", "prefill_urls", "decode_urls"})
_PEER_REFERENCE = re.compile(
    r"\{\{[^{}]*\b(?:" + "|".join(sorted(_PEER_NAMES)) + r")\b"
)
_SCALAR_NAMES = (
    "worker_ip",
    "port",
    "model_path",
    "model_name",
    "gpu_count",
    "gpu_ids",
    "role",
    "group_id",
    "runner_image",
    "net_device",
    "intra_node_data_parallel_size",
)
_PEER_SCHEMA = {"ip": None, "port": None, "url": None, "ports": {"*": None}}
_SCHEMA = {
    **dict.fromkeys(_SCALAR_NAMES),
    "ports": {"*": None},
    "prefill": [_PEER_SCHEMA],
    "decode": [_PEER_SCHEMA],
    "prefill_urls": [None],
    "decode_urls": [None],
}
_MAX_SOURCE = 65536
_MAX_OUTPUT = 1048576
_MAX_VALUES = 8192


def is_router_command_template(command: Optional[str]) -> bool:
    """Whether the command needs placement-time template rendering."""
    return bool(command and ("{{" in command or "{%" in command))


def references_router_peers(command: Optional[str]) -> bool:
    """Recognize peer names even in expressions that cannot render as argv."""
    if not command:
        return False
    if len(command) <= _MAX_SOURCE:
        try:
            syntax = Environment().parse(command)
            return any(node.name in _PEER_NAMES for node in syntax.find_all(nodes.Name))
        except (TemplateSyntaxError, RecursionError):
            pass
    # Invalid or oversized expressions still cannot send peer placeholders to
    # an engine. Scan their text without an unbounded template parse.
    return bool(_PEER_REFERENCE.search(command))


def router_environment(
    model, defaults: Optional[Mapping[str, str]] = None
) -> Dict[str, str]:
    """Resolve the effective Router env, preserving whole-map role overrides."""
    from gpustack.schemas.models import find_role

    role = find_role(model, "router")
    role_env = getattr(role, "env", None)
    env = role_env if role_env is not None else getattr(model, "env", None)
    return {**(defaults or {}), **(env or {})}


def router_command_management(
    command: Optional[str],
    *,
    env: Optional[Mapping[str, str]] = None,
) -> Literal["catalog", "restart", "manual"]:
    """Peer-dependent user commands refresh by restart, never by inferred API."""
    if not command or not command.strip():
        return "catalog"
    if not is_router_command_template(command):
        return "manual"
    try:
        names = _CommandTemplate(command, env=env).names
    except ValueError:
        # Admission reports malformed templates; policy must not guess an API.
        return "manual"
    return "restart" if names & _PEER_NAMES else "manual"


def validate_router_command_template(
    command: str, *, env: Optional[Mapping[str, str]] = None
) -> None:
    """Validate syntax and variable paths without requiring scheduled peers."""
    template = _CommandTemplate(command, env=env)
    template.arguments(None)


def _environment_values(env: Optional[Mapping[str, str]]) -> Dict[str, str]:
    """Keep the same unreserved env names in validation and runtime contexts."""
    return {
        key: value
        for key, value in (env or {}).items()
        if key.partition(".")[0] not in _SCHEMA
    }


def render_router_command_template(
    command: str,
    variables: Mapping[str, object],
    peers: Mapping[str, object],
    *,
    env: Optional[Mapping[str, str]] = None,
) -> List[str]:
    """Expand scalar values, finite peer loops and URL joins into atomic argv.

    Only Jinja's parser is used. No user template is compiled or evaluated by
    Jinja; the interpreter accepts a small AST and only reads primitive maps.
    Values become opaque markers before lexical splitting, then replace those
    markers once. Quotes, spaces and shell metacharacters in data stay data.
    """
    template = _CommandTemplate(command, env=env)
    context = _environment_values(env)
    context.update(
        {name: variables[name] for name in _SCALAR_NAMES if name in variables}
    )
    context["ports"] = {
        name.removeprefix("ports."): value
        for name, value in variables.items()
        if name.startswith("ports.") and name.count(".") == 1
    }
    for role in ("prefill", "decode"):
        entries = []
        addresses = peers.get(role, [])
        if len(addresses) > _MAX_VALUES:
            raise ValueError("Router template has too many peer endpoints.")
        for peer in addresses:
            host = (
                f"[{peer.ip}]"
                if ":" in peer.ip and not peer.ip.startswith("[")
                else peer.ip
            )
            entries.append(
                {
                    "ip": peer.ip,
                    "port": peer.port,
                    "url": f"http://{host}:{peer.port}",
                    "ports": dict(peer.ports),
                }
            )
        if template.names & {role, f"{role}_urls"} and not entries:
            raise ValueError(f"Router template has no ready {role} endpoints.")
        context[role] = entries
        context[f"{role}_urls"] = [entry["url"] for entry in entries]
    return template.arguments(context)


def _path(node) -> List[str]:
    parts = []
    while isinstance(node, nodes.Getattr) and not node.attr.startswith("_"):
        parts.append(node.attr)
        node = node.node
        if len(parts) > 2:
            raise ValueError(
                "Router template field paths are at most three levels deep."
            )
    if isinstance(node, nodes.Name):
        return [node.name, *reversed(parts)]
    raise ValueError(
        "Router templates support named values and public field paths only."
    )


def _lookup(path, context):
    if path[0] not in context and ".".join(path) in context:
        path = [".".join(path)]
    value = context
    for part in path:
        if not isinstance(value, dict) or part not in value or value[part] is None:
            raise ValueError(
                f"Router template value '{'.'.join(path)}' is unavailable."
            )
        value = value[part]
    return value


class _CommandTemplate:
    def __init__(self, command: str, *, env: Optional[Mapping[str, str]] = None):
        if len(command) > _MAX_SOURCE:
            raise ValueError("Router command template exceeds 64 KiB.")
        try:
            self.body = Environment().parse(command).body
        except (TemplateSyntaxError, RecursionError) as exc:
            raise ValueError(f"Invalid Router command template: {exc}") from exc
        self.names: Set[str] = set()
        self._validate_body(
            self.body, {**dict.fromkeys(_environment_values(env)), **_SCHEMA}
        )

    def _schema(self, node, scope):
        path = _path(node)
        lookup_path = (
            [".".join(path)]
            if path[0] not in scope and ".".join(path) in scope
            else path
        )
        value = scope
        for part in lookup_path:
            if not isinstance(value, dict) or (part not in value and "*" not in value):
                raise ValueError(f"Unknown Router template value '{'.'.join(path)}'.")
            value = value.get(part, value.get("*"))
        if path[0] in _SCHEMA:
            self.names.add(path[0])
        return value

    def _validate_expression(self, node, scope):
        if isinstance(node, nodes.Filter):
            if (
                node.name != "join"
                or node.kwargs
                or node.dyn_args
                or node.dyn_kwargs
                or len(node.args) > 1
                or (
                    node.args
                    and (
                        not isinstance(node.args[0], nodes.Const)
                        or not isinstance(node.args[0].value, str)
                        or len(node.args[0].value) > 128
                    )
                )
                or self._schema(node.node, scope) != [None]
            ):
                raise ValueError(
                    "Router templates only support URL lists with join('separator')."
                )
        elif self._schema(node, scope) is not None:
            raise ValueError(
                "Router template lists need a for loop or join; objects need a field path."
            )

    def _validate_body(self, body, scope, in_loop=False):
        for node in body:
            if isinstance(node, nodes.Output):
                for child in node.nodes:
                    if not isinstance(child, nodes.TemplateData):
                        self._validate_expression(child, scope)
            elif isinstance(node, nodes.For):
                if (
                    in_loop
                    or node.recursive
                    or node.else_
                    or node.test
                    or not isinstance(node.target, nodes.Name)
                    or node.target.name in _SCHEMA
                    or node.target.name.startswith("_")
                    or not isinstance(node.iter, nodes.Name)
                    or node.iter.name not in _PEER_NAMES
                ):
                    raise ValueError(
                        "Router templates support non-nested for loops over prefill/decode lists only."
                    )
                item_schema = self._schema(node.iter, scope)[0]
                self._validate_body(
                    node.body, {**scope, node.target.name: item_schema}, True
                )
            else:
                raise ValueError(
                    "Router templates support values, for loops and join only."
                )

    def arguments(self, context: Optional[Dict]) -> List[str]:
        prefix = f"GPUSTACK_VALUE_{uuid.uuid4().hex}_"
        values = []
        chunks = []
        length = 0
        value_length = 0

        def append(value):
            nonlocal length
            length += len(value)
            if length > _MAX_OUTPUT:
                raise ValueError("Rendered Router command exceeds 1 MiB.")
            chunks.append(value)

        def output(node, scope):
            nonlocal value_length
            if len(values) >= _MAX_VALUES:
                raise ValueError("Router template expands too many values.")
            value = self._value(node, scope) if context is not None else "value"
            if not isinstance(value, (str, int, float)) or "\x00" in str(value):
                raise ValueError(
                    "Router template output must be a scalar without NUL bytes."
                )
            value_length += len(str(value))
            if value_length > _MAX_OUTPUT:
                raise ValueError("Rendered Router values exceed 1 MiB.")
            values.append(str(value))
            append(f"{prefix}{len(values) - 1}END")

        def walk(body, scope):
            for node in body:
                if isinstance(node, nodes.Output):
                    for child in node.nodes:
                        if isinstance(child, nodes.TemplateData):
                            append(child.data)
                        else:
                            output(child, scope)
                else:
                    entries = (
                        _lookup([node.iter.name], scope)
                        if context is not None
                        else [None]
                    )
                    if len(entries) > _MAX_VALUES:
                        raise ValueError("Router template has too many peer endpoints.")
                    for entry in entries:
                        walk(node.body, {**scope, node.target.name: entry})

        walk(self.body, context or {})
        return _restore_arguments("".join(chunks), prefix, values)

    @staticmethod
    def _value(node, scope):
        if isinstance(node, nodes.Filter):
            values = _lookup(_path(node.node), scope)
            separator = node.args[0].value if node.args else ""
            return separator.join(values)
        return _lookup(_path(node), scope)


def _restore_arguments(layout: str, prefix: str, values: List[str]) -> List[str]:
    try:
        tokens = shlex.split(layout)
    except ValueError as exc:
        raise ValueError(f"Invalid Router command quoting: {exc}") from exc
    marker = re.compile(re.escape(prefix) + r"(\d+)END")
    result = [
        marker.sub(lambda match: values[int(match[1])], token) for token in tokens
    ]
    if (
        not result
        or not result[0]
        or any("\x00" in token for token in result)
        or sum(map(len, result)) > _MAX_OUTPUT
    ):
        raise ValueError(
            "Router command must contain an executable and fit within 1 MiB."
        )
    return result
