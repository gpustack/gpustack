# Model architectures loader
# Loads model architectures from YAML configuration file
from typing import Dict, List

import yaml

from gpustack.schemas.models import CategoryEnum

# Path to the YAML configuration file
_YAML_FILE = "model_architectures.yaml"


def _load_architectures() -> Dict:
    """Load model architectures from YAML file."""
    import os

    # Get the directory of this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, _YAML_FILE)

    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


# Load architectures at module import
_ARCHITECTURES = _load_architectures()


# vLLM backend model architectures
VLLM_LLM_MODELS: List[str] = _ARCHITECTURES.get("vLLM", {}).get("llm", [])
VLLM_MULTIMODAL_MODELS: List[str] = _ARCHITECTURES.get("vLLM", {}).get("multimodal", [])
VLLM_EMBEDDING_MODELS: List[str] = _ARCHITECTURES.get("vLLM", {}).get("embedding", [])
VLLM_RERANKER_MODELS: List[str] = _ARCHITECTURES.get("vLLM", {}).get("reranker", [])
VLLM_SPEECH_TO_TEXT_MODELS: List[str] = _ARCHITECTURES.get("vLLM", {}).get(
    "speech_to_text", []
)
VLLM_TEXT_TO_SPEECH_MODELS: List[str] = _ARCHITECTURES.get("vLLM", {}).get(
    "text_to_speech", []
)

# SGLang backend model architectures
SGLANG_LLM_MODELS: List[str] = _ARCHITECTURES.get("SGLang", {}).get("llm", [])
SGLANG_MULTIMODAL_MODELS: List[str] = _ARCHITECTURES.get("SGLang", {}).get(
    "multimodal", []
)
SGLANG_EMBEDDING_MODELS: List[str] = _ARCHITECTURES.get("SGLang", {}).get(
    "embedding", []
)
SGLANG_RERANKER_MODELS: List[str] = _ARCHITECTURES.get("SGLang", {}).get("reranker", [])
SGLANG_SPEECH_TO_TEXT_MODELS: List[str] = _ARCHITECTURES.get("SGLang", {}).get(
    "speech_to_text", []
)
SGLANG_TEXT_TO_SPEECH_MODELS: List[str] = _ARCHITECTURES.get("SGLang", {}).get(
    "text_to_speech", []
)

# MindIE backend model architectures
MINDIE_LLM_MODELS: List[str] = _ARCHITECTURES.get("MindIE", {}).get("llm", [])


def get_models_by_category(backend: str, category: CategoryEnum) -> List[str]:
    """
    Get model architectures by backend and category.

    Args:
        backend: Backend name (e.g., "vllm", "sglang")
        category: Model category from CategoryEnum

    Returns:
        List of model architecture names
    """
    category_key = category.value
    return _ARCHITECTURES.get(backend, {}).get(category_key, [])


def get_all_models_by_backend(backend: str) -> List[str]:
    """
    Get all model architectures for a backend.

    Args:
        backend: Backend name (e.g., "vllm", "sglang")

    Returns:
        List of all model architecture names
    """
    backend_data = _ARCHITECTURES.get(backend, {})
    all_models = []
    for models in backend_data.values():
        all_models.extend(models)
    return all_models
