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


def get_backend_models_dict(backend: str) -> Dict[str, List[str]]:
    """
    Get all model architectures for a backend organized by category.

    Args:
        backend: Backend name (e.g., "vLLM", "SGLang", "MindIE")

    Returns:
        Dictionary mapping category names to model architecture lists.
        Supports both CategoryEnum values and string keys including "multimodal".
    """
    backend_data = _ARCHITECTURES.get(backend, {})

    return {
        CategoryEnum.LLM: backend_data.get("llm", []),
        CategoryEnum.EMBEDDING: backend_data.get("embedding", []),
        CategoryEnum.RERANKER: backend_data.get("reranker", []),
        CategoryEnum.SPEECH_TO_TEXT: backend_data.get("speech_to_text", []),
        CategoryEnum.TEXT_TO_SPEECH: backend_data.get("text_to_speech", []),
        # Keep string keys for backward compatibility
        "multimodal": backend_data.get("multimodal", []),
    }
