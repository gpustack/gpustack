"""Select PD connectors from pretrained KV layout metadata."""

from typing import Any, Optional

from gpustack.utils.hybrid_attention import is_hybrid_attention


def pd_connector_variant(config: Optional[Any]) -> Optional[str]:
    """Classify standard and hybrid KV layouts, including compressed MLA.

    Args:
        config: The effective pretrained language-model configuration.

    Returns:
        A catalog variant, or None when no configuration is available.
    """
    if config is None:
        return None

    def field(name):
        return (
            config.get(name)
            if isinstance(config, dict)
            else getattr(config, name, None)
        )

    for name in ("text_config", "llm_config", "language_config", "thinker_config"):
        nested = field(name)
        if nested is not None and pd_connector_variant(nested) == "hybrid":
            return "hybrid"
    ratios = field("compress_ratios")
    if not isinstance(ratios, (list, tuple)):
        ratios = (ratios,)
    if is_hybrid_attention(config) or any(
        isinstance(ratio, (int, float)) and not isinstance(ratio, bool) and ratio > 0
        for ratio in ratios
    ):
        return "hybrid"
    model_type = str(field("model_type") or "").lower()
    architectures = field("architectures")
    if not isinstance(architectures, (list, tuple)):
        architectures = []
    # Architecture metadata also identifies compressed layouts when configs
    # omit their per-layer ratios; deployment/repository names are never read.
    if model_type == "deepseek_v4" or any(
        isinstance(architecture, str) and architecture.startswith("DeepseekV4")
        for architecture in architectures
    ):
        return "hybrid"
    return "standard"
