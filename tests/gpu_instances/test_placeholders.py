"""Create-time ``{{generated_*}}`` placeholder substitution for GPU instance specs.

Templates (e.g. the built-in JupyterLab ones) carry ``{{generated_token}}`` in
``spec.command`` and ``spec.ports[].accessParams``; the backend resolves every
occurrence to one per-instance value at create time and persists the concrete
spec, so stop/start replays and UI link building always see the same token.
"""

import re

from gpustack.gpu_instances.placeholders import substitute_generated_placeholders
from gpustack.schemas.gpu_instances import (
    GPUInstancePort,
    GPUInstanceSpec,
)

_HEX_TOKEN_RE = re.compile(r"^[0-9a-f]{32}$")


def _spec() -> GPUInstanceSpec:
    return GPUInstanceSpec(
        type_="gpu",
        image="jupyterlab",
        command=[
            "jupyter",
            "lab",
            "--ServerApp.token={{generated_token}}",
            "--IdentityProvider.token={{generated_token}}",
        ],
        ports=[
            GPUInstancePort(
                name="JUPYTER",
                port=8888,
                access_params={"token": "{{generated_token}}", "fixed": "literal"},
            ),
            GPUInstancePort(name="SSH", port=22),
        ],
    )


def test_all_occurrences_share_one_token():
    spec = _spec()

    substitute_generated_placeholders(spec)

    token = spec.command[2].split("=", 1)[1]
    assert _HEX_TOKEN_RE.match(token)
    # Every occurrence — across command items and accessParams values — is the
    # same per-instance value.
    assert spec.command[3] == f"--IdentityProvider.token={token}"
    assert spec.ports[0].access_params["token"] == token
    # Non-placeholder values pass through untouched.
    assert spec.ports[0].access_params["fixed"] == "literal"
    assert spec.ports[1].access_params is None


def test_no_placeholder_leaves_spec_byte_identical():
    spec = GPUInstanceSpec(
        type_="gpu",
        image="busybox",
        command=["sleep", "infinity"],
        ports=[GPUInstancePort(port=8080)],
    )
    before = spec.model_dump(by_alias=True, exclude_none=True)

    substitute_generated_placeholders(spec)

    assert spec.model_dump(by_alias=True, exclude_none=True) == before


def test_unknown_placeholder_left_unchanged():
    # Mirrors the inference-backend _resolve_env_vars semantics: unknown names
    # are not an error and stay as written.
    spec = GPUInstanceSpec(
        type_="gpu",
        image="busybox",
        command=["--token={{not_a_generated_value}}"],
        ports=[GPUInstancePort(port=8888, access_params={"token": "{{other}}"})],
    )

    substitute_generated_placeholders(spec)

    assert spec.command[0] == "--token={{not_a_generated_value}}"
    assert spec.ports[0].access_params["token"] == "{{other}}"


def test_distinct_specs_get_distinct_tokens():
    spec_a = _spec()
    spec_b = _spec()

    substitute_generated_placeholders(spec_a)
    substitute_generated_placeholders(spec_b)

    assert spec_a.command[2] != spec_b.command[2]
