import logging
import requests
from typing import Any, Dict
from gpustack.schemas.models import (
    BackendEnum,
    ModelInstance,
    Model,
    CategoryEnum,
)

logger = logging.getLogger(__name__)


def get_meta_from_running_instance(
    mi: ModelInstance, backend: str, model: Model
) -> Dict[str, Any]:
    """
    Get the meta information from the running instance (synchronous version).
    """

    if backend == BackendEnum.SGLANG and CategoryEnum.IMAGE in model.categories:
        # SGLang Diffusion does not provide metadata endpoints at the moment.
        return {}

    meta_path = "/v1/models"
    if backend == BackendEnum.ASCEND_MINDIE:
        # Ref: https://www.hiascend.com/document/detail/zh/mindie/21RC2/mindieservice/servicedev/mindie_service0066.html
        meta_path = "/info"

    try:
        url = f"http://{mi.worker_ip}:{mi.port}{meta_path}"
        response = requests.get(url, timeout=1)
        response.raise_for_status()

        response_json = response.json()

        if backend == BackendEnum.ASCEND_MINDIE:
            model_meta = parse_tgi_info_meta(response_json)
        else:
            model_meta = parse_v1_models_meta(response_json)

        mutate_model_meta(model, model_meta)
        return model_meta
    except Exception as e:
        logger.warning(f"Failed to get meta from running instance {mi.name}: {e}")
        return {}


def mutate_model_meta(model: Model, meta: Dict[str, Any]) -> None:
    """
    Mutate the model's meta information with the provided meta dictionary.
    """
    readable_source = model.readable_source.lower()
    if "qwen3-tts" in readable_source:
        # Special handling for Qwen3-TTS model
        # Refs:
        # - https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/examples/online_serving/qwen3_tts/
        # - https://github.com/QwenLM/Qwen3-TTS
        qwen3_tts_meta = {
            "voices": [
                "Vivian",
                "Serena",
                "Uncle_Fu",
                "Dylan",
                "Eric",
                "Ryan",
                "Aiden",
                "Ono_Anna",
                "Sohee",
            ],
            "languages": ["Auto", "Chinese", "English", "Japanese", "Korean"],
        }
        QWEN3_TTS_TASK_TYPES = ["CustomVoice", "VoiceDesign", "Base"]
        for task_type in QWEN3_TTS_TASK_TYPES:
            if task_type.lower() in readable_source:
                qwen3_tts_meta["task_type"] = task_type
                break
        meta.update(qwen3_tts_meta)


def parse_v1_models_meta(response_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse the meta information from the /v1/models response.
    """
    if "data" not in response_json or not response_json["data"]:
        return {}

    first_model = response_json["data"][0]
    meta_info = first_model.get("meta", {})

    # Optional keys from different backends
    optional_keys = [
        "voices",
        "max_model_len",
    ]
    for key in optional_keys:
        if key in first_model:
            meta_info[key] = first_model[key]

    return meta_info


def parse_tgi_info_meta(response_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse the meta information from the TGI-like /info response.

    Example:
    {
        "docker_label": null,
        "max_batch_total_tokens": 8192,
        "max_best_of": 1,
        "max_concurrent_requests": 200,
        "max_stop_sequences": null,
        "max_waiting_tokens": null,
        "sha": null,
        "validation_workers": null,
        "version": "1.0.0",
        "waiting_served_ratio": null,
        "models": [
            {
                "model_device_type": "npu",
                "model_dtype": "float16",
                "model_id": "deepseek",
                "model_pipeline_tag": "text-generation",
                "model_sha": null,
                "max_total_tokens": 2560
            }
        ],
        "max_input_length": 2048
    }
    """
    meta_info = {}

    if "models" in response_json and response_json["models"]:
        first_model = response_json["models"][0]
        meta_info.update(first_model)

    # Optional keys from TGI-like backends
    optional_keys = [
        "max_batch_total_tokens",
        "max_best_of",
        "max_concurrent_requests",
        "max_stop_sequences",
        "max_waiting_tokens",
        "max_input_length",
    ]
    for key in optional_keys:
        if key in response_json:
            meta_info[key] = response_json[key]

    return meta_info
