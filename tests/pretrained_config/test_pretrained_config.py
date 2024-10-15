from gpustack.worker.backends.vllm import get_max_model_len
from transformers import AutoConfig


def test_get_max_model_len():
    hf_model_lengths = {
        "Qwen/Qwen2-0.5B-Instruct": 32768,
        "Qwen/Qwen2-VL-7B-Instruct": 32768,
        "tiiuae/falcon-7b": 2048,
        "microsoft/Phi-3.5-mini-instruct": 131072,
        "liuhaotian/llava-v1.6-34b": 4096,
    }

    for model_name, expected_max_model_len in hf_model_lengths.items():
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        assert (
            get_max_model_len(config) == expected_max_model_len
        ), f"max_model_len mismatch for {model_name}"
