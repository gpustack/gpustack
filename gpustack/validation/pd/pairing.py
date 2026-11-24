"""Validate parameter and environment compatibility of a prefill/decode pair."""

import json

from typing import Any, List, Optional, Union

from gpustack.api.exceptions import BadRequestException
from gpustack.schemas.models import (
    BackendEnum,
    Model,
    ModelCreate,
    ModelSpecBase,
    ModelUpdate,
    RoleNameEnum,
)
from gpustack.schemas.pd_modes import iter_placeholders
from gpustack.utils.command import find_last_parameter, flatten_to_argv
from gpustack.validation.pd.common import model_field
from gpustack.validation.pd.environment import (
    environment_pairing_factors,
    modern_nixl_hma,
)
from gpustack.validation.pd.parameters import (
    DIFFER,
    UNDECIDABLE,
    PAIRING_MUST_MATCH,
    compare_max_model_len,
    compare_must_match,
    role_parameters,
)


def _validate_connector_json(parameters: List[str], role: str) -> None:
    """Validate JSON structure while leaving templated scalar values unresolved."""
    raw = find_last_parameter(parameters, ["kv-transfer-config", "kv_transfer_config"])
    if raw is None:
        return
    # Numeric launch placeholders may appear outside JSON strings.
    for placeholder in iter_placeholders(raw):
        raw = raw.replace(placeholder, "0")
    try:
        descriptor = json.loads(raw)
    except (TypeError, ValueError):
        descriptor = None
    if not isinstance(descriptor, dict):
        raise BadRequestException(
            message=f"--kv-transfer-config on role '{role}' must be a JSON object."
        )


def _toggle_enabled(toggle, parameters: List[str]) -> bool:
    """Where a role's parameters leave one boolean engine switch.

    Last spelling wins, matching argparse, so a role carrying both flags is
    read the way the engine would read it rather than the way the list happens
    to be ordered.
    """
    enabled = toggle.default_enabled
    for token in flatten_to_argv(parameters):
        name = token.split("=", 1)[0].replace("_", "-")
        if name == toggle.enable_flag:
            enabled = True
        elif name == toggle.disable_flag:
            enabled = False
    return enabled


def _check_pairing_toggles(
    field, prefill_params, decode_params, *, resolve_hma: bool = False
) -> None:
    """Refuse a pair whose two roles land a declared boolean switch differently.

    The switches are declared per backend in `pd-modes.yaml` under
    `pairing_toggles` rather than written here, for the same reason the
    value-carrying parameters live in `validation/pd/parameters.py`'s tables: which
    settings must match is a fact about the engine, and a fact kept in code is
    one nobody finds when adding the next engine.
    """
    from gpustack.server.pd_mode_catalog import get_pairing_toggles

    # A deployment that names no backend runs vLLM, the same default
    # `get_backend` applies once the row exists.
    backend = field("backend") or BackendEnum.VLLM
    for toggle in get_pairing_toggles(backend):
        if resolve_hma and toggle.key == "hybrid_kv_cache_manager":
            # HMA-capable NIXL runtimes resolve this from flags, env and model shape.
            continue
        if _toggle_enabled(toggle, prefill_params) == _toggle_enabled(
            toggle, decode_params
        ):
            continue
        detail = toggle.description or f"the two roles disagree on {toggle.key}"
        raise BadRequestException(
            message=(
                f"prefill and decode disagree on {toggle.key}: {detail}. The "
                f"pair is then rejected on contact and the group never serves. "
                f"Note that a KV connector may set this on its own — the "
                f"divergence comes from one role carrying "
                f"{toggle.enable_flag} and the other not."
            )
        )


def validate_role_pairing(  # noqa: C901
    model_in: Union[ModelCreate, ModelUpdate, ModelSpecBase],
    stored: Optional[Model] = None,
    *,
    pretrained_config: Optional[Any] = None,
    skip_compatibility: bool = False,
) -> None:
    """Reject provably incompatible parameter and environment configurations.

    Context lengths must agree even when the engine handshake accepts them.
    The engine and its connector check the parallel topology at startup or
    during KV transfer.
    Unknown effective values of pairing factors are reported through
    ``pairing_unverified``.

    Args:
        model_in: The submitted deployment specification.
        stored: The stored deployment supplying omitted update fields.
        pretrained_config: Source configuration for model-dependent rules.
        skip_compatibility: Limit validation to connector configuration structure.
    """

    def field(name: str):
        return model_field(model_in, stored, name)

    roles = field("roles")
    if not roles or not field("disaggregation"):
        return

    model_parameters = field("backend_parameters")
    prefill = next((r for r in roles if r.name == RoleNameEnum.PREFILL.value), None)
    decode = next((r for r in roles if r.name == RoleNameEnum.DECODE.value), None)
    if prefill is None or decode is None:
        return

    prefill_params = role_parameters(prefill, model_parameters)
    decode_params = role_parameters(decode, model_parameters)

    backend = field("backend") or BackendEnum.VLLM.value
    for role, parameters in ((prefill, prefill_params), (decode, decode_params)):
        if (role.backend or backend) == BackendEnum.VLLM.value:
            _validate_connector_json(parameters, role.name)

    if skip_compatibility:
        return

    verdict, prefill_len, decode_len = compare_max_model_len(
        prefill_params, decode_params
    )
    if verdict == DIFFER:
        raise BadRequestException(
            message=(
                f"prefill and decode declare different context lengths "
                f"({prefill_len} vs {decode_len}). No engine checks this: "
                f"the pair handshakes, transfers KV and answers short "
                f"prompts, and a prompt above "
                f"{min(prefill_len, decode_len)} tokens fails at decode "
                f"after prefill has already computed it. Give both roles "
                f"the same context length."
            )
        )

    _check_pairing_toggles(
        field,
        prefill_params,
        decode_params,
        resolve_hma=modern_nixl_hma(model_in, stored=stored),
    )

    for label, names in PAIRING_MUST_MATCH.items():
        if compare_must_match(label, names, prefill_params, decode_params) != DIFFER:
            continue
        prefill_value = find_last_parameter(prefill_params, names)
        decode_value = find_last_parameter(decode_params, names)
        raise BadRequestException(
            message=(
                f"prefill and decode declare different {label} "
                f"('{prefill_value}' vs '{decode_value}'). The connector "
                f"rejects the pair on contact, so the group would never "
                f"serve; the roles must agree."
            )
        )

    for factor in environment_pairing_factors(
        model_in, stored=stored, pretrained_config=pretrained_config
    ):
        if factor.verdict == DIFFER:
            raise BadRequestException(message=factor.message)


def undecidable_factors(
    prefill, decode, model_parameters, *, model: Optional[Any] = None
) -> List[str]:
    """The pairing factors where one side declared and the other went silent.

    These are the factors where GPUStack knows one side's value and cannot
    resolve the other's. Reporting them tells callers which comparisons
    require more information.

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

    for label, names in PAIRING_MUST_MATCH.items():
        if (
            compare_must_match(label, names, prefill_params, decode_params)
            == UNDECIDABLE
        ):
            factors.append(label)

    if model is not None:
        for factor in environment_pairing_factors(model):
            if factor.verdict == UNDECIDABLE and factor.label not in factors:
                factors.append(factor.label)

    return factors
