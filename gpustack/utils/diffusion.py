"""Classify Diffusers artifacts without loading model code or weights."""

from typing import Any, Dict, Optional

from gpustack.schemas.models import CategoryEnum

# These identifiers name video pipelines, not repositories or model aliases.
VIDEO_PIPELINES = frozenset(
    {
        "WanPipeline",
        "WanImageToVideoPipeline",
        "WanVideoToVideoPipeline",
        "WanVACEPipeline",
        "WanAnimatePipeline",
        "CogVideoXPipeline",
        "CogVideoXImageToVideoPipeline",
        "CogVideoXVideoToVideoPipeline",
        "CogVideoXFunControlPipeline",
        "HunyuanVideoPipeline",
        "LTXPipeline",
        "LTXImageToVideoPipeline",
        "LTXConditionPipeline",
        "LTXI2VLongMultiPromptPipeline",
    }
)


def normalize_diffusers_model_index(data: Any) -> Optional[Dict[str, Any]]:
    """Return a Diffusers index from either supported JSON shape."""
    entries = data if isinstance(data, list) else [data]
    return next(
        (
            entry
            for entry in entries
            if isinstance(entry, dict) and "_diffusers_version" in entry
        ),
        None,
    )


def diffusion_model_category(data: Dict[str, Any]) -> CategoryEnum:
    """Recognize known video pipelines and retain image as the fallback."""
    pipeline = data.get("_class_name")
    if isinstance(pipeline, str) and pipeline in VIDEO_PIPELINES:
        return CategoryEnum.VIDEO
    return CategoryEnum.IMAGE
