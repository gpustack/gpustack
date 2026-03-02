# TODO version-aware support list
from typing import List

from gpustack.schemas.models import BackendEnum, CategoryEnum

from gpustack.scheduler import model_architectures


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

    models = model_architectures.get_backend_models_dict(backend or BackendEnum.VLLM)

    for architecture in architectures or []:
        # Check multimodal first (multimodal models are also LLM)
        if architecture in models.get("multimodal", []):
            return CategoryEnum.LLM

        # The order of checking categories is important to correctly classify models
        # that might fit into multiple categories (e.g., embedding and llm).
        # The previous implementation had a specific check order, which should be preserved.
        check_order = [
            CategoryEnum.EMBEDDING,
            CategoryEnum.RERANKER,
            CategoryEnum.SPEECH_TO_TEXT,
            CategoryEnum.TEXT_TO_SPEECH,
            CategoryEnum.IMAGE,
            CategoryEnum.LLM,
        ]
        for category in check_order:
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
    models = model_architectures.get_backend_models_dict(backend or BackendEnum.VLLM)

    for architecture in architectures or []:
        if architecture in models["multimodal"]:
            return True
    return False
