import pytest
from gpustack.routes.model_provider import determine_model_category
from gpustack.schemas.model_provider import ModelProviderTypeEnum
from gpustack.schemas.models import CategoryEnum


doubao_llm_data = {
    "created": 1767587883,
    "domain": "VLM",
    "features": {
        "batch": {"batch_chat": False, "batch_job": False},
        "cache": {"prefix_cache": False, "session_cache": False},
        "structured_outputs": {"json_object": True, "json_schema": True},
        "tools": {"function_calling": True},
    },
    "id": "doubao-seed-1-8-251228",
    "modalities": {
        "input_modalities": ["text", "image", "video"],
        "output_modalities": ["text"],
    },
    "name": "doubao-seed-1-8",
    "object": "model",
    "task_type": ["VisualQuestionAnswering", "TextGeneration"],
    "token_limits": {
        "context_window": 262144,
        "max_input_token_length": 229376,
        "max_output_token_length": 65536,
        "max_reasoning_token_length": 32768,
    },
    "version": "251228",
}

doubao_llm_data2 = {
    "created": 1736337657,
    "domain": "LLM",
    "features": {
        "cache": {
            "prefix_cache": True,
            "session_cache": True,
        },
        "structured_outputs": {
            "json_object": False,
            "json_schema": False,
        },
        "tools": {
            "function_calling": True,
        },
    },
    "id": "doubao-1-5-lite-32k-250115",
    "modalities": {"input_modalities": ["text"], "output_modalities": ["text"]},
    "name": "doubao-1-5-lite-32k",
    "object": "model",
    "task_type": ["TextGeneration"],
    "token_limits": {"context_window": 32768, "max_output_token_length": 12288},
    "version": "250115",
}

doubao_embedding_data = {
    "created": 1715588483,
    "domain": "Embedding",
    "features": {},
    "id": "doubao-embedding-text-240515",
    "modalities": {"input_modalities": ["text"]},
    "name": "doubao-embedding",
    "object": "model",
    "status": "Retiring",
    "task_type": ["TextEmbedding"],
    "token_limits": {},
    "version": "text-240515",
}

qwen_llm_data = {
    "id": "qwen3-max-2026-01-23",
    "object": "model",
    "created": 1769481796,
    "owned_by": "system",
}

qwen_image_data = {
    "id": "qwen-image-edit-max",
    "object": "model",
    "created": 1768570977,
    "owned_by": "system",
}

qwen_llm_split_name_data = {
    "id": "siliconflow/deepseek-v3.2",
    "object": "model",
    "created": 1769611475,
    "owned_by": "system",
}


@pytest.mark.parametrize(
    "provider_type,model,expected",
    [
        # actual data from doubao
        (
            ModelProviderTypeEnum.DOUBAO,
            doubao_llm_data,
            [CategoryEnum.LLM.value],
        ),
        (
            ModelProviderTypeEnum.DOUBAO,
            doubao_llm_data2,
            [CategoryEnum.LLM.value],
        ),
        (
            ModelProviderTypeEnum.DOUBAO,
            doubao_embedding_data,
            [CategoryEnum.EMBEDDING.value],
        ),
        # actual data from qwen
        (
            ModelProviderTypeEnum.QWEN,
            qwen_image_data,
            [CategoryEnum.IMAGE.value],
        ),
        (
            ModelProviderTypeEnum.QWEN,
            qwen_llm_data,
            [CategoryEnum.LLM.value],
        ),
        (
            ModelProviderTypeEnum.QWEN,
            qwen_llm_split_name_data,
            [CategoryEnum.LLM.value],
        ),
        # actual data from deepseek
        (
            ModelProviderTypeEnum.DEEPSEEK,
            {
                "id": "deepseek-chat",
                "object": "model",
                "owned_by": "deepseek",
            },
            [CategoryEnum.LLM.value],
        ),
        (
            ModelProviderTypeEnum.DEEPSEEK,
            {
                "id": "deepseek-reasoner",
                "object": "model",
                "owned_by": "deepseek",
            },
            [CategoryEnum.LLM.value],
        ),
    ],
)
def test_determine_model_category(provider_type, model, expected):
    assert determine_model_category(provider_type, model) == expected
