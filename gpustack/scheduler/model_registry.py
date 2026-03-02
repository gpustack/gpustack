# Synced with https://github.com/vllm-project/vllm/blob/v0.16.0/vllm/model_executor/models/registry.py
# Update these when the builtin vLLM is updated
# List of supported model architectures for the default version of the vLLM backend
# TODO version-aware support list
from typing import Dict, List

from gpustack.schemas.models import BackendEnum, CategoryEnum

from gpustack.scheduler import model_architectures


def _get_backend_models(backend: BackendEnum) -> Dict[str, List[str]]:
    """
    Get model lists for a specific backend.

    Args:
        backend: BackendEnum (e.g., BackendEnum.VLLM, BackendEnum.SGLANG)

    Returns:
        Dictionary containing model lists for each category
    """
    prefix = backend.value.upper()

    return {
        "llm": getattr(model_architectures, f"{prefix}_LLM_MODELS", []),
        "multimodal": getattr(model_architectures, f"{prefix}_MULTIMODAL_MODELS", []),
        "embedding": getattr(model_architectures, f"{prefix}_EMBEDDING_MODELS", []),
        "reranker": getattr(model_architectures, f"{prefix}_RERANKER_MODELS", []),
        "speech_to_text": getattr(
            model_architectures, f"{prefix}_SPEECH_TO_TEXT_MODELS", []
        ),
        "text_to_speech": getattr(
            model_architectures, f"{prefix}_TEXT_TO_SPEECH_MODELS", []
        ),
    }


def detect_model_type(
    architectures: List[str], backend: BackendEnum = BackendEnum.VLLM
) -> CategoryEnum:
    """
    Detect the model type based on the architectures.

    Args:
        architectures: List of model architecture names.
        backend: BackendEnum or backend name. Defaults to BackendEnum.VLLM.

    Returns:
        The detected model category.
    """
    if backend == BackendEnum.CUSTOM:
        # Default to vLLM for custom backend
        backend = BackendEnum.VLLM

    models = _get_backend_models(backend)

    for architecture in architectures or []:
        # Check multimodal first (multimodal models are also LLM)
        if architecture in models.get("multimodal", []):
            return CategoryEnum.LLM

        for category in CategoryEnum:
            if category == CategoryEnum.UNKNOWN:
                continue
            if architecture in models.get(category.value, []):
                return category

    return CategoryEnum.UNKNOWN


def is_multimodal_model(
    architectures: List[str], backend: BackendEnum = BackendEnum.VLLM
) -> bool:
    """
    Check if the model is a multimodal model based on the architectures.

    Args:
        architectures: List of model architecture names.
        backend: BackendEnum or backend name. Defaults to BackendEnum.VLLM.

    Returns:
        True if the model is multimodal, False otherwise.
    """
    models = _get_backend_models(backend)

    for architecture in architectures or []:
        if architecture in models["multimodal"]:
            return True
    return False
