from typing import Optional

from gpustack.schemas.models import Model, ModelSource, SourceEnum
from gpustack.server.catalog import get_catalog_draft_models


def get_draft_model_source(model: Model) -> Optional[ModelSource]:
    """
    Get the model source for the draft model.
    First check the catalog for the draft model.
    If not found, get the model source empirically to support custom draft models.
    """
    if model.speculative_config is None or not model.speculative_config.draft_model:
        return None

    draft_model = model.speculative_config.draft_model
    catalog_draft_models = get_catalog_draft_models()
    for catalog_draft_model in catalog_draft_models:
        if catalog_draft_model.name == draft_model:
            return catalog_draft_model

    # If draft_model looks like a path, assume it's a local path.
    if draft_model.startswith("/"):
        return ModelSource(source=SourceEnum.LOCAL_PATH, local_path=draft_model)

    # Otherwise, assume it comes from the same source as the main model.
    if model.source == SourceEnum.HUGGING_FACE:
        return ModelSource(
            source=SourceEnum.HUGGING_FACE,
            huggingface_repo_id=draft_model,
        )
    elif model.source == SourceEnum.MODEL_SCOPE:
        return ModelSource(
            source=SourceEnum.MODEL_SCOPE,
            model_scope_model_id=draft_model,
        )
    return None
