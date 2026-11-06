"""Interpret model metadata without importing code from model repositories.

Family adapters are reviewed, local data transformations keyed by model_type.
Add an adapter with an upstream reference and an offline regression fixture.
Installed Transformers classes own architecture defaults; generic metadata keeps
unknown fields unknown so each caller can validate only what its check requires.
"""

import logging
import math
from copy import deepcopy
from typing import Any, Callable, Dict, Optional

from transformers import PretrainedConfig
from transformers.integrations.heterogeneity.configuration_utils import (
    AmbiguousGlobalPerLayerAttributeError,
)
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

logger = logging.getLogger(__name__)


def _adapt_mistral(config_dict: Dict) -> None:
    # Match AutoConfig dispatch for alternating-attention checkpoints labeled mistral.
    # https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/auto/configuration_auto.py#L420-L425
    if "layer_types" in config_dict:
        config_dict["model_type"] = "ministral"


def _adapt_chatglm(config_dict: Dict) -> None:
    # ChatGLM serializes dimensions under family-specific names. Only supplied
    # values are mapped; missing dimensions must not acquire unrelated defaults.
    # https://github.com/vllm-project/vllm/blob/e8ae84bf7e992c63570905fce3073fd7a7de2412/vllm/transformers_utils/configs/chatglm.py
    for target, source in {
        "num_hidden_layers": "num_layers",
        "vocab_size": "padded_vocab_size",
        "head_dim": "kv_channels",
    }.items():
        if source in config_dict:
            config_dict.setdefault(target, config_dict[source])
    kv_heads = (
        config_dict.get("multi_query_group_num")
        if config_dict.get("multi_query_attention")
        else config_dict.get("num_attention_heads")
    )
    if kv_heads is not None:
        config_dict.setdefault("num_key_value_heads", kv_heads)


_CONFIG_ADAPTERS: Dict[str, Callable[[Dict], None]] = {
    "mistral": _adapt_mistral,
    "chatglm": _adapt_chatglm,
}

# Quantization and RoPE dictionaries are data, not nested model configurations.
_TEXT_CONFIG_PATHS = (
    ("llm_config",),
    ("text_config",),
    ("thinker_config", "text_config"),
)
_NESTED_CONFIG_KEYS = tuple(
    sorted({key for path in _TEXT_CONFIG_PATHS for key in path} | {"vision_config"})
)


def safe_pretrained_config_from_dict(config_dict: Dict) -> PretrainedConfig:
    """Interpret JSON with installed config classes or a data-only fallback.

    Args:
        config_dict: Decoded model configuration. The input is not modified.

    Returns:
        Config with architecture defaults when supported, otherwise serialized
        metadata. Repository auto_map entries never select executable classes.
    """
    if not isinstance(config_dict, dict):
        raise ValueError("Model config.json must contain a JSON object.")

    config_dict = deepcopy(config_dict)
    model_type = config_dict.get("model_type")
    if isinstance(model_type, str) and model_type in _CONFIG_ADAPTERS:
        _CONFIG_ADAPTERS[model_type](config_dict)
        logger.debug("Applied local metadata adapter for model_type=%s", model_type)
    model_type = config_dict.get("model_type")
    config = _installed_config_from_dict(config_dict, model_type)
    return config if config is not None else _metadata_config_from_dict(config_dict)


def _installed_config_from_dict(
    config_dict: Dict, model_type: Any
) -> Optional[PretrainedConfig]:
    if not isinstance(model_type, str) or model_type not in CONFIG_MAPPING:
        logger.debug(
            "Using JSON metadata: no installed config class for %s", model_type
        )
        return None
    try:
        config_class = CONFIG_MAPPING[model_type]
        # Constructors can mutate nested dictionaries before failing; keep the
        # JSON fallback's metadata independent of the native parsing attempt.
        config = config_class.from_dict(deepcopy(config_dict))
    except (
        ImportError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        AmbiguousGlobalPerLayerAttributeError,
    ) as error:
        # Exception messages can contain checkpoint content; log the error type only.
        logger.warning(
            "Using JSON metadata for %s: installed config loading failed (%s)",
            model_type,
            type(error).__name__,
        )
        return None
    logger.debug(
        "Using installed configuration class %s for %s",
        config_class.__name__,
        model_type,
    )
    return config


def _metadata_config_from_dict(config_dict: Dict) -> PretrainedConfig:
    if config_dict.get("per_layer_config"):
        # Native validation may read ambiguous global attributes. Resolve known
        # architecture defaults without overrides, then retain all explicit values
        # and let the generic config apply the per-layer overrides independently.
        base_dict = {
            key: value
            for key, value in config_dict.items()
            if key != "per_layer_config"
        }
        base = _installed_config_from_dict(base_dict, config_dict.get("model_type"))
        if base is not None:
            config_dict = {**base.to_dict(), **config_dict}
    # Generic configs must receive legacy RoPE metadata after their attributes exist.
    rope_scaling = config_dict.pop("rope_scaling", None)
    config = PretrainedConfig.from_dict(config_dict)
    if rope_scaling is not None:
        config.rope_scaling = rope_scaling
    for key in _NESTED_CONFIG_KEYS:
        value = getattr(config, key, None)
        if isinstance(value, dict) and value:
            setattr(config, key, safe_pretrained_config_from_dict(value))
    return config


# Simplified from vllm.config._get_and_verify_max_len
# Keep in our codebase to avoid dependency on vllm's internal
# APIs which may change unexpectedly.
# https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/config.py#L2453
def get_max_model_len(
    pretrained_config: Any,
    default: Optional[int] = 2048,
    *,
    fallback_config: Any = None,
) -> Optional[int]:
    """Read valid length metadata without requiring optional RoPE interpretation.

    Unknown RoPE semantics or unusable scaling values preserve the declared
    length. Missing or invalid length fields use fallback_config, then the
    caller's explicit default. RoPE and length always come from the same config.
    """
    derived_max_model_len = float("inf")
    possible_keys = [
        # OPT
        "max_position_embeddings",
        # GPT-2
        "n_positions",
        # MPT
        "max_seq_len",
        # ChatGLM2
        "seq_length",
        # Command-R
        "model_max_length",
        # Whisper
        "max_target_positions",
        # Others
        "max_sequence_length",
        "max_seq_length",
        "seq_len",
    ]
    # Choose the smallest "max_length" from the possible keys.
    for key in possible_keys:
        max_len = getattr(pretrained_config, key, None)
        if _is_positive_finite_number(max_len) and max_len >= 1:
            derived_max_model_len = min(derived_max_model_len, max_len)

    # If none of the keys were found in the config, use a default and
    # log a warning.
    if derived_max_model_len == float("inf"):
        if fallback_config is not None and fallback_config is not pretrained_config:
            return get_max_model_len(fallback_config, default=default)
        logger.warning(
            "The model's config.json does not contain a valid length in any of "
            "the following keys to determine the original maximum length: "
            "%s. Using caller fallback %s.",
            possible_keys,
            default,
        )
        return default

    derived_max_model_len = _apply_rope_length_scaling(
        int(derived_max_model_len), getattr(pretrained_config, "rope_scaling", None)
    )

    logger.debug(f"Derived max model length: {derived_max_model_len}")
    return int(derived_max_model_len)


def _is_positive_finite_number(value: Any) -> bool:
    # Booleans are JSON flags, not numeric dimensions or scaling factors.
    return (type(value) is int and value > 0) or (
        type(value) is float and math.isfinite(value) and value > 0
    )


def _apply_rope_length_scaling(declared_length: int, rope_scaling: Any) -> int:
    """Apply only understood scalar scaling; preserve other metadata unchanged."""
    if not isinstance(rope_scaling, dict):
        return declared_length
    rope_type = rope_scaling.get("type", rope_scaling.get("rope_type"))
    # Other types can already declare a scaled limit or use non-scalar semantics.
    # Recognizing a RoPE type in Transformers does not imply this length formula.
    if not isinstance(rope_type, str) or rope_type not in {"linear", "dynamic", "yarn"}:
        return declared_length

    factor = rope_scaling.get("factor")
    base_length = (
        rope_scaling.get("original_max_position_embeddings")
        if rope_type == "yarn"
        else declared_length
    )
    if not _is_positive_finite_number(factor) or not _is_positive_finite_number(
        base_length
    ):
        logger.warning(
            "Ignoring incomplete or invalid RoPE scaling; using declared length"
        )
        return declared_length
    try:
        scaled_length = base_length * factor
    except OverflowError:
        scaled_length = None
    if not _is_positive_finite_number(scaled_length) or scaled_length < 1:
        logger.warning("Ignoring unusable RoPE scaling result; using declared length")
        return declared_length
    return int(scaled_length)


def get_hf_text_config(
    config: Optional[PretrainedConfig],
) -> Optional[PretrainedConfig]:
    """Select the text subconfiguration, preserving partially supplied metadata."""
    for path in _TEXT_CONFIG_PATHS:
        candidate = config
        for key in path:
            candidate = (
                candidate.get(key)
                if isinstance(candidate, dict)
                else getattr(candidate, key, None)
            )
        if candidate is None or (isinstance(candidate, dict) and not candidate):
            continue
        if isinstance(candidate, dict):
            return safe_pretrained_config_from_dict(candidate)
        return candidate
    return config
