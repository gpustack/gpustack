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
from gpustack.schemas.pd_modes import PDTensorParallelPairingEnum, iter_placeholders
from gpustack.server.pd_mode_catalog import get_pd_mode
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
    effective_tensor_parallelism,
    role_parameters,
    tensor_parallel_rule,
    violates_tensor_parallel_direction,
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


def _check_tensor_parallel_pairing(disaggregation, *, prefill_tp, decode_tp) -> None:
    """Apply the recipe's declared tensor-parallel direction.

    The direction belongs to the KV connector, so it is read off the mode
    (`PDMode.pairing.tensor_parallel`) rather than written here: NIXL needs
    decode at least as wide as prefill, vllm-ascend's Mooncake needs the
    opposite (Huawei's reference deployment is prefill TP4 / decode TP1), and
    `custom` injects no connector GPUStack knows. A mode the catalog cannot
    resolve is held to the NIXL rule, which is what every mode was held to
    before the rule became declarable.
    """
    mode_name = disaggregation.mode.value
    rule = tensor_parallel_rule(get_pd_mode(mode_name))
    if not violates_tensor_parallel_direction(
        rule, prefill_tp=prefill_tp, decode_tp=decode_tp
    ):
        return

    if rule == PDTensorParallelPairingEnum.DECODE_GE_PREFILL:
        raise BadRequestException(
            message=(
                f"decode runs tensor parallelism {decode_tp}, below prefill's "
                f"{prefill_tp}. A decode narrower than its prefill cannot "
                f"receive that prefill's KV layout, and the engine reports it "
                f"as an IndexError inside decode rather than as a "
                f"configuration error. decode's tensor parallelism must be at "
                f"least prefill's."
            )
        )
    raise BadRequestException(
        message=(
            f"prefill runs tensor parallelism {prefill_tp}, below decode's "
            f"{decode_tp}. pd mode '{mode_name}' gathers each decode rank's "
            f"KV from prefill ranks, which needs prefill's tensor "
            f"parallelism to be at least decode's."
        )
    )


def validate_role_pairing(  # noqa: C901
    model_in: Union[ModelCreate, ModelUpdate, ModelSpecBase],
    stored: Optional[Model] = None,
    *,
    pretrained_config: Optional[Any] = None,
) -> None:
    """Reject provably incompatible parameter and environment configurations.

    Context lengths must agree even when the engine handshake accepts them.
    Tensor-parallel direction is connector-specific; other factors describe KV
    representation and transfer compatibility. Unknown effective values stay
    undecidable and are reported through ``pairing_unverified``.

    Args:
        model_in: The submitted deployment specification.
        stored: The stored deployment supplying omitted update fields.
        pretrained_config: Source configuration for model-dependent rules.
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

    # Effective, not declared. A role that writes no tp is not a role with an
    # unknown one when it pins its own cards: the backend injects the card
    # count for a single-worker member, so a decode pinned to one card beside a
    # prefill at TP2 is a real inversion and worth catching. A role that pins
    # nothing stays unknown and is reported as unverified rather than
    # held to a 1 it was never going to run.
    prefill_tp = effective_tensor_parallelism(prefill, prefill_params)
    decode_tp = effective_tensor_parallelism(decode, decode_params)
    if prefill_tp is not None and decode_tp is not None:
        _check_tensor_parallel_pairing(
            field("disaggregation"), prefill_tp=prefill_tp, decode_tp=decode_tp
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

    if model is not None:
        for factor in environment_pairing_factors(model):
            if factor.verdict == UNDECIDABLE and factor.label not in factors:
                factors.append(factor.label)

    return factors
