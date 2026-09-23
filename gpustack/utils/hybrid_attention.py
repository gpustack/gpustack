"""Recognize hybrid-attention models from their pretrained config.

A hybrid model interleaves recurrent layers — Mamba, or Gated-DeltaNet
linear attention — with full-attention ones. Their recurrent layers hold a
fixed-size state instead of one key/value pair per token, so a KV cache
connector either declares hybrid support to the engine and then constrains
how cache chunks align, or cannot serve the model at all. Both outcomes are
worth telling a user about before they deploy.

The verdict is read off the config's own fields rather than an architecture
name list: the field names below are shared by the families that have them
and survive a release, while a name list ages out with every new model.
"""

from typing import Any, Optional, Set

HYBRID_MODEL_DOC_URL = (
    "https://docs.gpustack.ai/latest/user-guide/cache-service-management/"
    "#running-hybrid-attention-models-with-lmcache"
)
"""Where a compatibility message sends the user: what to configure, and how
to read the block size a hybrid model needs it configured for."""

STANDARD_ATTENTION_LAYER_TYPES = {
    "attention",
    "full_attention",
    "sliding_attention",
    "chunked_attention",
}
"""Layer kinds that page like ordinary attention. Gemma 3's sliding window
and gpt-oss's alternation make those models hybrid in the loose sense and
need none of this: every layer still holds one KV pair per token."""

_LAYER_TYPE_ATTRS = ("layer_types", "layers_block_type")
"""Where a config spells out its per-layer composition. The two names are
the same list under different families' spellings."""

_RECURRENT_STATE_ATTRS = (
    "linear_conv_kernel_dim",
    "linear_key_head_dim",
    "linear_num_key_heads",
    "mamba_d_conv",
    "mamba_d_state",
    "mamba_expand",
    "mamba_num_heads",
    "ssm_state_size",
)
"""Fields only a model with recurrent layers declares — the fallback for
configs written before their family spelled out ``layer_types``."""

_NESTED_CONFIG_ATTRS = (
    "text_config",
    "llm_config",
    "language_config",
    "thinker_config",
)
"""Where a multimodal wrapper keeps its language model's config: the
recurrent layers are declared there, not beside the vision tower."""


def _field(config: Any, name: str) -> Any:
    """One config field, whether the config arrived as a PretrainedConfig or
    as the raw config.json mapping (the local-path fallback returns a dict)."""
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


def _layer_kinds(config: Any) -> Set[str]:
    """The per-layer composition, lowercased. Empty when the config does not
    spell one out."""
    for attr in _LAYER_TYPE_ATTRS:
        value = _field(config, attr)
        if isinstance(value, (list, tuple)) and value:
            return {str(item).lower() for item in value}
    return set()


def is_hybrid_attention(pretrained_config: Optional[Any]) -> bool:
    """Whether the model interleaves recurrent layers with attention ones.

    False for anything unrecognized, including a config that could not be
    read: the verdict only raises a compatibility warning, and telling a
    user that a plain transformer is hybrid sends them after settings that
    change nothing.

    Args:
        pretrained_config: The model's Hugging Face config, as a
            PretrainedConfig or the raw mapping.

    Returns:
        True when the model carries recurrent layers.
    """
    if not pretrained_config:
        return False

    if _layer_kinds(pretrained_config) - STANDARD_ATTENTION_LAYER_TYPES:
        return True

    # Nemotron-H states its composition as a pattern string ("M-M-M*-"),
    # where M is a Mamba layer.
    pattern = _field(pretrained_config, "hybrid_override_pattern")
    if isinstance(pattern, str) and "m" in pattern.lower():
        return True

    if any(
        _field(pretrained_config, attr) is not None for attr in _RECURRENT_STATE_ATTRS
    ):
        return True

    return any(
        is_hybrid_attention(_field(pretrained_config, attr))
        for attr in _NESTED_CONFIG_ATTRS
    )
