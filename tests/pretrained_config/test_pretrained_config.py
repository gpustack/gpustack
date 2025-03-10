from gpustack.worker.backends.vllm import get_hf_text_config, get_max_model_len
from transformers import AutoConfig


def test_get_max_model_len():
    hf_model_lengths = {
        "Qwen/Qwen2-0.5B-Instruct": 32768,
        "Qwen/Qwen2-VL-7B-Instruct": 32768,
        "tiiuae/falcon-7b": 2048,
        "microsoft/Phi-3.5-mini-instruct": 131072,
        "llava-hf/llava-v1.6-mistral-7b-hf": 32768,
        "unsloth/Llama-3.2-11B-Vision-Instruct": 131072,
        "THUDM/glm-4-9b-chat-1m": 1048576,
    }

    for model_name, expected_max_model_len in hf_model_lengths.items():
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        pretrained_or_hf_text_config = get_hf_text_config(config)
        assert (
            get_max_model_len(pretrained_or_hf_text_config) == expected_max_model_len
        ), f"max_model_len mismatch for {model_name}"
