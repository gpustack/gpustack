"""Describe video model capabilities independently of deployment aliases."""

from typing import Any, Dict, Optional

from gpustack.schemas.models import BackendEnum, Model, get_backend
from gpustack.utils.command import find_last_parameter

MINIMAX_H3_PIPELINES = frozenset({"MiniMaxH3ModularPipeline", "MiniMaxH3Pipeline"})
MINIMAX_H3_REPOSITORIES = frozenset({"minimaxai/minimax-h3", "minimax/minimax-h3"})


def get_video_capabilities(
    model: Model, index: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """Resolve H3's request contract from the artifact and selected runtime.

    A stored artifact identity allows local-path models to reuse inspected
    metadata without performing filesystem or Hub requests during inference.
    """
    stored = (getattr(model, "meta", None) or {}).get("video") or {}
    if not isinstance(stored, dict):
        stored = {}
    source = getattr(model, "model_source_index", None)
    pipeline = (index or {}).get("_class_name")
    recognized = (
        getattr(model, "model_source_key", "").lower() in MINIMAX_H3_REPOSITORIES
        or isinstance(pipeline, str)
        and pipeline in MINIMAX_H3_PIPELINES
        or stored.get("family") == "minimax_h3"
        and source is not None
        and stored.get("source_index") == source
    )
    if not recognized:
        return None
    backend = get_backend(model)
    if backend not in (BackendEnum.SGLANG, BackendEnum.VLLM):
        return None
    parameters = model.backend_parameters or []
    partition = find_last_parameter(
        parameters, ["model-variant" if backend == BackendEnum.SGLANG else "task-type"]
    )
    if not partition:
        leaf = model.model_source_key.rstrip("/").rsplit("/", 1)[-1].lower()
        partition = leaf if leaf in ("fl2va", "ref2va") else None
    partition = str(
        partition or ("fl2va" if backend == BackendEnum.SGLANG else "combined")
    ).lower()
    tasks = {
        "fl2va": ["t2va", "fl2va"],
        "ref2va": ["ref2va"],
        "combined": ["t2va", "fl2va", "ref2va"],
        "hybrid": ["t2va", "fl2va", "ref2va"],
    }.get(partition, [])
    return {
        "family": "minimax_h3",
        "source_index": source,
        "request_profile": (
            "sglang_h3" if backend == BackendEnum.SGLANG else "vllm_omni_h3"
        ),
        "tasks": tasks,
        # Reference tasks require model-specific media controls in the playground.
        "playground_tasks": [task for task in tasks if task == "t2va"],
        "default_task": "t2va" if "t2va" in tasks else None,
        "default_size": "768x1344",
        "resolutions": [
            "768x1344",
            "1344x768",
            "768x768",
            "1024x768",
            "768x1024",
            "1792x768",
        ],
        "duration": {"min": 4, "max": 15, "default": 4},
    }


def update_video_metadata(model: Model, index: Optional[Dict[str, Any]] = None) -> bool:
    """Merge derived capabilities while retaining other model metadata."""
    capabilities = get_video_capabilities(model, index)
    if capabilities is None:
        meta = model.meta or {}
        stored = meta.get("video")
        if (
            isinstance(stored, dict)
            and stored.get("family") == "minimax_h3"
            and stored.get("source_index") is not None
            and stored.get("source_index") != model.model_source_index
        ):
            model.meta = {key: value for key, value in meta.items() if key != "video"}
            return True
        return False
    meta = model.meta or {}
    stored = meta.get("video")
    video = {**(stored if isinstance(stored, dict) else {}), **capabilities}
    updated = {**meta, "video": video}
    if updated == meta:
        return False
    model.meta = updated
    return True
