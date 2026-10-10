"""Tenant routes must not bypass the authorization-aware watch entry point."""

import ast
from pathlib import Path

import gpustack.routes


def test_raw_resource_streaming_is_limited_to_platform_admin_routes():
    routes_dir = Path(gpustack.routes.__file__).parent
    # These two routers are mounted exclusively behind get_admin_user.
    platform_routes = {"users.py", "organizations.py"}
    violations = []
    for path in routes_dir.rglob("*.py"):
        if str(path.relative_to(routes_dir)) in platform_routes:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "streaming"
            ):
                violations.append(f"{path.relative_to(routes_dir)}:{node.lineno}")
    assert violations == []


def test_tenant_streaming_calls_explicitly_pass_context():
    routes_dir = Path(gpustack.routes.__file__).parent
    violations = []
    for path in routes_dir.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "tenant_streaming"
            ):
                continue
            ctx = (
                node.args[1]
                if len(node.args) > 1
                else next((kw.value for kw in node.keywords if kw.arg == "ctx"), None)
            )
            if not isinstance(ctx, ast.Name) or ctx.id != "ctx":
                violations.append(f"{path.relative_to(routes_dir)}:{node.lineno}")
    assert violations == []
