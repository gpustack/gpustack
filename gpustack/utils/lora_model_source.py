"""Build ModelSource rows for LoRA adapters from Model.lora_list entries."""

import logging
from typing import List

from gpustack.schemas.models import (
    LoraListEntry,
    Model,
    ModelSource,
    SourceEnum,
)

logger = logging.getLogger(__name__)


def lora_route_name_for(base_model_name: str, lora_name: str) -> str:
    prefix = f"{base_model_name}:"
    return lora_name if lora_name.startswith(prefix) else f"{prefix}{lora_name}"


def model_base_descriptor(model: Model) -> str:
    """Human/catalog string used to match adapter_config.base_model_name_or_path."""
    if model.source == SourceEnum.HUGGING_FACE:
        parts = [model.huggingface_repo_id or ""]
        if model.huggingface_filename:
            parts.append(model.huggingface_filename)
        return "/".join(p for p in parts if p)
    if model.source == SourceEnum.MODEL_SCOPE:
        parts = [model.model_scope_model_id or ""]
        if model.model_scope_file_path:
            parts.append(model.model_scope_file_path)
        return "/".join(p for p in parts if p)
    return model.local_path or ""


def lora_entry_to_model_source(entry: LoraListEntry) -> ModelSource:
    src = (entry.source or SourceEnum.HUGGING_FACE.value).lower()
    if src == SourceEnum.HUGGING_FACE.value:
        repo = entry.lora_repo_name or entry.lora_name
        if not repo:
            raise ValueError("lora_repo_name is required for huggingface LoRA entries")
        return ModelSource(
            source=SourceEnum.HUGGING_FACE,
            huggingface_repo_id=repo,
            huggingface_filename=entry.huggingface_filename,
        )
    if src == SourceEnum.MODEL_SCOPE.value:
        mid = entry.lora_repo_name or entry.lora_name
        if not mid:
            raise ValueError("lora_repo_name is required for model_scope LoRA entries")
        return ModelSource(
            source=SourceEnum.MODEL_SCOPE,
            model_scope_model_id=mid,
            model_scope_file_path=entry.model_scope_file_path,
        )
    if src == SourceEnum.LOCAL_PATH.value:
        # Fall back to lora_repo_name to tolerate UI form bindings that store the
        # absolute path in lora_repo_name when source=local_path.
        path = entry.local_path or entry.lora_repo_name
        if not path:
            raise ValueError(
                "local_path (or lora_repo_name as path) is required for local_path LoRA entries"
            )
        return ModelSource(source=SourceEnum.LOCAL_PATH, local_path=path)
    raise ValueError(f"Unsupported LoRA source: {src}")


def normalized_lora_list(model: Model) -> List[LoraListEntry]:
    raw = model.lora_list
    if not raw:
        return []
    if not isinstance(raw, list):
        return []
    out: List[LoraListEntry] = []
    for item in raw:
        if isinstance(item, LoraListEntry):
            if item.lora_name:
                out.append(item)
            continue
        if isinstance(item, dict):
            try:
                e = LoraListEntry.model_validate(item)
                if e.lora_name:
                    out.append(e)
            except Exception as ex:
                logger.debug(f"Skip invalid lora_list item {item!r}: {ex}")
            continue
    return out
