"""Resolve environment-dependent vLLM/NIXL pairing factors without guessing."""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from packaging.version import InvalidVersion, Version

from gpustack.schemas.models import Model, ModelSpecBase, RoleSpec
from gpustack.validation.pd.common import model_field
from gpustack.validation.pd.parameters import (
    AGREE,
    DIFFER,
    UNDECIDABLE,
    role_parameters,
)
from gpustack.utils.command import find_last_parameter, flatten_to_argv
from gpustack.utils.hybrid_attention import is_hybrid_attention

_LAYOUT_ALIASES = {"NHD": "LBNHC", "HND": "LBHNC"}
_LAYOUTS = frozenset({"LBNHC", "LBHNC", "LHBNC", "BLHNC", "BLNHC", "BHLNC"})
_ATTENTION_BACKENDS = frozenset(
    {
        "FLASH_ATTN",
        "FLASHINFER",
        "TRITON_ATTN",
        "FLEX_ATTENTION",
        "FLASH_ATTN_MLA",
        "FLASHINFER_MLA",
        "TRITON_MLA",
        "CUTLASS_MLA",
        "FLASHMLA",
    }
)
_HMA_ENV = "VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE"
_NESTED_CONFIGS = ("text_config", "llm_config", "language_config", "thinker_config")
_NIXL_STATE_RULES_VERSION = Version("0.27.1")
"""The bundled NVIDIA recipe's verified HMA and DS state-transfer contract."""


class PDPairingError(ValueError):
    """Resolved model metadata proves the prefill/decode pair incompatible."""


@dataclass
class EnvironmentPairingFactor:
    """One resolved compatibility verdict and its safe admission message."""

    label: str
    verdict: str
    message: str = ""


@dataclass
class _RoleEnvironment:
    name: str
    parameters: List[str]
    env: Dict[str, str]
    version: Optional[Version]
    speculative: bool
    cached: bool


def effective_role_environment(
    role: RoleSpec, model_env: Optional[Dict[str, str]]
) -> Dict[str, str]:
    """Copy the dictionary the role uses; a non-null override replaces it.

    Args:
        role: The role whose environment is being resolved.
        model_env: The deployment-level environment inherited by a null override.

    Returns:
        A copy of the effective environment dictionary.
    """
    return dict(model_env or {}) if role.env is None else dict(role.env)


def _roles(
    model: ModelSpecBase, stored: Optional[Model]
) -> Optional[Tuple[_RoleEnvironment, _RoleEnvironment]]:
    disaggregation = model_field(model, stored, "disaggregation")
    if not disaggregation or disaggregation.mode != "vllm-nixl":
        return None
    roles = model_field(model, stored, "roles") or []
    prefill = next((role for role in roles if role.name == "prefill"), None)
    decode = next((role for role in roles if role.name == "decode"), None)
    if prefill is None or decode is None:
        return None
    backend = model_field(model, stored, "backend") or "vLLM"
    if any((role.backend or backend) != "vLLM" for role in (prefill, decode)):
        return None
    parameters = model_field(model, stored, "backend_parameters")
    env = model_field(model, stored, "env")
    return tuple(
        _RoleEnvironment(
            role.name,
            role_parameters(role, parameters),
            effective_role_environment(role, env),
            _version(role, model, stored),
            _enabled_override(role, model, stored, "speculative_config")
            or find_last_parameter(
                role_parameters(role, parameters),
                ["speculative-config", "speculative_config"],
            )
            is not None,
            _enabled_override(role, model, stored, "extended_kv_cache"),
        )
        for role in (prefill, decode)
    )


def _enabled_override(role, model, stored, name) -> bool:
    value = getattr(role, name)
    if value is None:
        value = model_field(model, stored, name)
    return bool(value and value.enabled)


def _version(
    role: RoleSpec, model: ModelSpecBase, stored: Optional[Model]
) -> Optional[Version]:
    image = role.image_name
    if image is None:
        image = model_field(model, stored, "image_name")
    if image:
        return None
    value = role.backend_version
    if value is None:
        value = model_field(model, stored, "backend_version")
    try:
        return Version(value) if value else None
    except InvalidVersion:
        return None


def _config_field(config: Any, name: str) -> Any:
    return config.get(name) if isinstance(config, dict) else getattr(config, name, None)


def _text_config(config: Any) -> Any:
    for name in _NESTED_CONFIGS:
        nested = _config_field(config, name)
        if nested is not None:
            return _text_config(nested)
    return config


def _role_config(role: _RoleEnvironment, config: Any) -> Any:
    """Apply explicit config overrides without changing the source configuration."""
    if config is None or (isinstance(config, dict) and not config):
        return None
    override = find_last_parameter(role.parameters, ["hf-overrides", "hf_overrides"])
    if override:
        try:
            values = json.loads(override)
        except (TypeError, ValueError):
            return None
        if not isinstance(values, dict) or config is None:
            return None
        base = config if isinstance(config, dict) else vars(config)
        config = {**base, **values}
    return _text_config(config)


def _boolean(value: Optional[str]) -> Optional[bool]:
    """Match vLLM's bool(int(value)) environment readers, including unset values."""
    try:
        return bool(int(value)) if value is not None else None
    except ValueError:
        return None


def _connector(role: _RoleEnvironment) -> Optional[Dict[str, Any]]:
    """Read the user-owned descriptor; None means the connector is not knowable."""
    raw = find_last_parameter(
        role.parameters, ["kv-transfer-config", "kv_transfer_config"]
    )
    if raw is None:
        return {"kv_connector": "NixlConnector"}
    try:
        descriptor = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(descriptor, dict) or descriptor.get("kv_connector_module_path"):
        return None
    if descriptor.get("kv_connector") != "NixlConnector":
        return None
    return descriptor


def _permutation(role: _RoleEnvironment) -> Optional[bool]:
    connector = _connector(role)
    if connector is None:
        return None
    value = connector.get("enable_permute_local_kv", False)
    if isinstance(value, bool):
        return value
    if str(value).lower() in {"true", "1"}:
        return True
    if str(value).lower() in {"false", "0"}:
        return False
    return None


def _layout(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip().upper()
    value = _LAYOUT_ALIASES.get(value, value)
    return value if value in _LAYOUTS else None


def _compare(left: Any, right: Any) -> str:
    if left is None or right is None:
        return UNDECIDABLE
    return AGREE if left == right else DIFFER


def _layout_factor(prefill, decode, configs) -> Optional[EnvironmentPairingFactor]:
    name = "VLLM_KV_CACHE_LAYOUT"
    if name not in prefill.env and name not in decode.env:
        return None
    left, right = _layout(prefill.env.get(name)), _layout(decode.env.get(name))
    verdict = _layout_verdict(prefill, decode, left, right, configs)
    return EnvironmentPairingFactor(
        "KV cache layout",
        verdict,
        f"prefill and decode declare different {name} "
        f"({left} vs {right}). The NIXL connector cannot transfer "
        "between these KV cache layouts with the selected permutation settings. "
        "Give both roles compatible layouts.",
    )


def _layout_verdict(prefill, decode, left, right, configs) -> str:
    connectors = [_connector(role) for role in (prefill, decode)]
    if any(connector is None for connector in connectors):
        return UNDECIDABLE
    if any(connector.get("kv_buffer_device") == "cpu" for connector in connectors):
        # Host staging can have a different transfer layout from the device cache.
        return UNDECIDABLE
    verdict = _compare(left, right)
    if verdict != DIFFER:
        return verdict
    if any(config is None for config in configs):
        return UNDECIDABLE
    if all(_config_field(config, "kv_lora_rank") for config in configs):
        disabled = [
            _boolean(role.env.get("VLLM_MLA_DISABLE")) for role in (prefill, decode)
        ]
        if disabled == [False, False]:
            # MLA transfers latent vectors and does not compare attention KV layouts.
            return AGREE
        if None in disabled:
            return UNDECIDABLE
    if (left, right) != ("LBHNC", "LBNHC"):
        return DIFFER
    permute = _permutation(decode)
    if permute is None:
        return UNDECIDABLE
    if not permute:
        return DIFFER
    return _permutation_hma_verdict(prefill, decode, configs)


def _permutation_hma_verdict(prefill, decode, configs) -> str:
    for role, config in zip((prefill, decode), configs):
        if is_hybrid_attention(config):
            return DIFFER
        kinds = _config_field(config, "layer_types") or []
        non_full = bool(set(kinds) - {"attention", "full_attention"})
        sliding = _config_field(config, "sliding_window") is not None
        if not (non_full or sliding):
            continue
        enabled = _resolved_hma(role, config)
        if enabled is True:
            return DIFFER
        if enabled is None:
            return UNDECIDABLE
    return AGREE


def _mla_factor(prefill, decode, configs) -> Optional[EnvironmentPairingFactor]:
    name = "VLLM_MLA_DISABLE"
    if name not in prefill.env and name not in decode.env:
        return None
    verdict = _compare(_boolean(prefill.env.get(name)), _boolean(decode.env.get(name)))
    if verdict != AGREE:
        if any(config is None for config in configs):
            verdict = UNDECIDABLE
        elif not any(_config_field(config, "kv_lora_rank") for config in configs):
            verdict = AGREE
    return EnvironmentPairingFactor(
        "MLA representation",
        verdict,
        f"prefill and decode disagree on {name}. An MLA model must use "
        "the same latent KV representation on both roles.",
    )


def _ssm_factor(prefill, decode, configs) -> Optional[EnvironmentPairingFactor]:
    name = "VLLM_SSM_CONV_STATE_LAYOUT"
    if name not in prefill.env and name not in decode.env:
        return None
    values = [role.env.get(name) for role in (prefill, decode)]
    if values == ["DS", "DS"]:
        verdict = AGREE
    elif any(config is None for config in configs):
        verdict = UNDECIDABLE
    elif not any(map(is_hybrid_attention, configs)):
        verdict = AGREE
    elif any(
        role.version is None or role.version < _NIXL_STATE_RULES_VERSION
        for role in (prefill, decode)
    ):
        verdict = UNDECIDABLE
    else:
        verdict = DIFFER if "SD" in values else UNDECIDABLE
    return EnvironmentPairingFactor(
        "SSM convolution state layout",
        verdict,
        f"NIXL transfer of recurrent model state requires {name}=DS on "
        "both prefill and decode; SD convolution state cannot be transferred.",
    )


def _hma_flag(role: _RoleEnvironment) -> Optional[bool]:
    enabled = None
    for token in flatten_to_argv(role.parameters):
        name = token.split("=", 1)[0].replace("_", "-")
        if name == "--no-disable-hybrid-kv-cache-manager":
            enabled = True
        elif name == "--disable-hybrid-kv-cache-manager":
            enabled = False
    return enabled


def _hma_value(role: _RoleEnvironment) -> Optional[bool]:
    explicit = _hma_flag(role)
    if explicit is False:
        return False
    if role.speculative or role.cached or _connector(role) is None:
        return None
    if role.version is None or role.version < _NIXL_STATE_RULES_VERSION:
        return None
    return _boolean(role.env.get(_HMA_ENV))


def modern_nixl_hma(model: ModelSpecBase, stored: Optional[Model] = None) -> bool:
    """Whether both managed roles use NIXL's HMA-capable runtime contract."""
    roles = _roles(model, stored)
    return bool(
        roles
        and all(
            role.version and role.version >= _NIXL_STATE_RULES_VERSION for role in roles
        )
    )


def _resolved_hma(role, config) -> Optional[bool]:
    flag = _hma_flag(role)
    if flag is False:
        return False
    if config is None:
        return None
    if role.speculative or role.cached or _connector(role) is None:
        return None
    if role.version is None or role.version < _NIXL_STATE_RULES_VERSION:
        return None
    if _config_field(config, "attention_chunk_size") is None:
        return True
    return _hma_value(role)


def _hma_factor(prefill, decode, configs) -> Optional[EnvironmentPairingFactor]:
    flags = [_hma_flag(role) for role in (prefill, decode)]
    modern = all(
        role.version and role.version >= _NIXL_STATE_RULES_VERSION
        for role in (prefill, decode)
    )
    env_declared = _HMA_ENV in prefill.env or _HMA_ENV in decode.env
    if not env_declared and (not modern or flags == [None, None]):
        return None
    if flags == [False, False]:
        verdict = AGREE
    elif flags[0] is not None and flags[1] is not None and flags[0] != flags[1]:
        verdict = DIFFER
    elif any(config is None for config in configs):
        verdict = UNDECIDABLE
    elif (
        env_declared
        and flags == [None, None]
        and not any(
            _config_field(config, "attention_chunk_size") is not None
            for config in configs
        )
    ):
        verdict = AGREE
    else:
        values = [
            _resolved_hma(role, config)
            for role, config in zip((prefill, decode), configs)
        ]
        verdict = _compare(*values)
        if any(flag is True and value is False for flag, value in zip(flags, values)):
            verdict = DIFFER
    return EnvironmentPairingFactor(
        "hybrid KV cache manager",
        verdict,
        f"prefill and decode run incompatible hybrid KV cache manager "
        f"settings. Check the HMA flags and {_HMA_ENV}; both roles must "
        "use compatible HMA settings supported by their model.",
    )


def _attention_backend(role: _RoleEnvironment) -> Optional[str]:
    value = find_last_parameter(
        role.parameters, ["attention-backend", "attention_backend"]
    )
    if value is None:
        return None
    value = value.strip().upper()
    return value if value in _ATTENTION_BACKENDS else None


def _attention_factor(prefill, decode) -> Optional[EnvironmentPairingFactor]:
    values = [_attention_backend(role) for role in (prefill, decode)]
    declared = any(
        find_last_parameter(role.parameters, ["attention-backend", "attention_backend"])
        is not None
        for role in (prefill, decode)
    )
    if not declared:
        batch = [
            _boolean(role.env.get("VLLM_BATCH_INVARIANT")) for role in (prefill, decode)
        ]
        if batch[0] == batch[1]:
            return None
        return EnvironmentPairingFactor("attention backend", UNDECIDABLE)
    verdict = _compare(*values)
    if _connector(prefill) is None or _connector(decode) is None:
        verdict = UNDECIDABLE
    for role in (prefill, decode):
        connector = _connector(role) or {}
        extra = connector.get("kv_connector_extra_config") or {}
        if isinstance(extra, dict) and extra.get("enforce_handshake_compat") is False:
            verdict = UNDECIDABLE
    return EnvironmentPairingFactor(
        "attention backend",
        verdict,
        "prefill and decode declare different attention backends. NIXL's "
        "compatibility hash requires matching resolved attention backends.",
    )


def environment_pairing_factors(
    model: ModelSpecBase,
    stored: Optional[Model] = None,
    *,
    pretrained_config: Optional[Any] = None,
) -> List[EnvironmentPairingFactor]:
    """Compare managed NIXL roles using their merged deployment specification.

    Unknown or templated values cannot justify an admission refusal. Network
    endpoints and performance settings are role-specific and are not compared.

    Args:
        model: The submitted or stored deployment specification.
        stored: The row supplying omitted fields of a sparse update.
        pretrained_config: Source configuration for model-dependent rules.

    Returns:
        Applicable factors, each with an agreement, mismatch or undecidable verdict.
    """
    roles = _roles(model, stored)
    if roles is None:
        return []
    prefill, decode = roles
    configs = [_role_config(role, pretrained_config) for role in roles]
    factors = [
        _layout_factor(prefill, decode, configs),
        _mla_factor(prefill, decode, configs),
        _ssm_factor(prefill, decode, configs),
        _hma_factor(prefill, decode, configs),
        _attention_factor(prefill, decode),
    ]
    result = [factor for factor in factors if factor is not None]
    if any(_connector(role) is None for role in roles):
        for factor in result:
            factor.verdict = UNDECIDABLE
    return result


def environment_pairing_needs_config(
    model: ModelSpecBase, stored: Optional[Model] = None
) -> bool:
    """Whether reading model metadata can settle a conditional pairing rule."""
    labels = {
        "MLA representation",
        "SSM convolution state layout",
        "hybrid KV cache manager",
    }
    factors = environment_pairing_factors(model, stored=stored)
    roles = _roles(model, stored)
    if roles is None or any(_connector(role) is None for role in roles):
        return False
    if any(
        factor.verdict == UNDECIDABLE and factor.label in labels for factor in factors
    ):
        return True
    return bool(
        all(_layout(role.env.get("VLLM_KV_CACHE_LAYOUT")) is not None for role in roles)
        and not any(
            (_connector(role) or {}).get("kv_buffer_device") == "cpu" for role in roles
        )
        and any(
            factor.verdict == UNDECIDABLE and factor.label == "KV cache layout"
            for factor in factors
        )
    )
