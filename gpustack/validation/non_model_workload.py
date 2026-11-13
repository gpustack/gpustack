"""Validate explicit accelerator-free services without changing model inference."""

from gpustack.schemas.inference_backend import is_custom_backend
from gpustack.schemas.models import ModelSpecBase
from gpustack.utils.model_resources import (
    NON_MODEL_WORKLOAD_ENV,
    non_model_workload_enabled,
)


def validate_non_model_workload(model: ModelSpecBase) -> None:
    """Reject unsupported deployment shapes and conflicting GPU declarations.

    Args:
        model: The effective deployment, including stored fields on updates.

    Raises:
        ValueError: The non-model workload declaration conflicts with deployment
            settings. Disabled declarations leave existing behavior unchanged.
    """
    roles = getattr(model, "roles", None) or []
    enabled = non_model_workload_enabled(model)
    if not enabled and not any(non_model_workload_enabled(role) for role in roles):
        return
    if roles or getattr(model, "disaggregation", None) is not None:
        raise ValueError(
            f"{NON_MODEL_WORKLOAD_ENV} is supported only for standalone deployments, "
            "not roles or disaggregation."
        )
    if not is_custom_backend(model.backend):
        raise ValueError(
            f"{NON_MODEL_WORKLOAD_ENV} requires a custom backend or a Custom "
            "deployment with an explicit image and command."
        )
    gpu_selector = getattr(model, "gpu_selector", None)
    if gpu_selector and gpu_selector.gpu_ids:
        raise ValueError(
            f"Manual GPU selection is not supported with {NON_MODEL_WORKLOAD_ENV}. "
            "Remove gpu_selector.gpu_ids or disable the non-model workload setting."
        )
    if getattr(model, "gpu_type_selector", None) is not None:
        raise ValueError(
            f"gpu_type_selector is not supported with {NON_MODEL_WORKLOAD_ENV}. "
            "Remove it or disable the non-model workload setting."
        )
