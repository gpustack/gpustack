"""
Shared fixtures for candidate selector tests.

Mock DeepSeek metadata with values mirrored from public ``config.json`` so
VRAM/attention-shape estimates stay stable without a hub download. Other repos
use the configuration loader.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gpustack.scheduler.calculator import (
    get_pretrained_config_with_workers as _orig_get_pretrained_config_with_workers,
)


_MOCK_PRETRAINED_DEEPSEEK_R1 = SimpleNamespace(
    architectures=["DeepseekV3ForCausalLM"],
    num_hidden_layers=61,
    hidden_size=7168,
    vocab_size=129280,
    num_attention_heads=128,
    num_key_value_heads=128,
    n_group=8,
    q_lora_rank=1536,
    kv_lora_rank=512,
    qk_rope_head_dim=64,
    qk_nope_head_dim=128,
    v_head_dim=128,
    torch_dtype="bfloat16",
    quantization_config={
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
    },
    moe_intermediate_size=2048,
    n_routed_experts=256,
    n_shared_experts=1,
    max_position_embeddings=163840,
    rope_scaling={
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 40,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 4096,
        "type": "yarn",
    },
)


_MOCK_PRETRAINED_DEEPSEEK_V32 = SimpleNamespace(
    architectures=["DeepseekV32ForCausalLM"],
    num_hidden_layers=61,
    hidden_size=7168,
    vocab_size=129280,
    num_attention_heads=128,
    num_key_value_heads=128,
    n_group=8,
    q_lora_rank=1536,
    kv_lora_rank=512,
    qk_rope_head_dim=64,
    qk_nope_head_dim=128,
    v_head_dim=128,
    torch_dtype="bfloat16",
    quantization_config={
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "scale_fmt": "ue8m0",
        "weight_block_size": [128, 128],
    },
    moe_intermediate_size=2048,
    n_routed_experts=256,
    n_shared_experts=1,
    max_position_embeddings=163840,
    rope_scaling={
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 40,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 4096,
        "type": "yarn",
    },
)


_MOCK_PRETRAINED_DEEPSEEK_V2_CHAT = SimpleNamespace(
    architectures=["DeepseekV2ForCausalLM"],
    num_hidden_layers=60,
    hidden_size=5120,
    vocab_size=102400,
    num_attention_heads=128,
    num_key_value_heads=128,
    n_group=8,
    q_lora_rank=1536,
    kv_lora_rank=512,
    qk_rope_head_dim=64,
    qk_nope_head_dim=128,
    v_head_dim=128,
    torch_dtype="bfloat16",
    moe_intermediate_size=1536,
    n_routed_experts=160,
    n_shared_experts=2,
    max_position_embeddings=163840,
    rope_scaling={
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 40,
        "mscale": 0.707,
        "mscale_all_dim": 0.707,
        "original_max_position_embeddings": 4096,
        "type": "yarn",
    },
)


_MOCK_PRETRAINED_BY_REPO_ID = {
    "deepseek-ai/DeepSeek-R1": _MOCK_PRETRAINED_DEEPSEEK_R1,
    "deepseek-ai/DeepSeek-V3.2": _MOCK_PRETRAINED_DEEPSEEK_V32,
    "deepseek-ai/DeepSeek-V2-Chat": _MOCK_PRETRAINED_DEEPSEEK_V2_CHAT,
}


async def _mock_or_real_get_pretrained_config_with_workers(
    model, workers=None, trust_remote_code=False
):
    repo_id = model.huggingface_repo_id or model.model_scope_model_id
    mocked = _MOCK_PRETRAINED_BY_REPO_ID.get(repo_id)
    if mocked is not None:
        return mocked
    return await _orig_get_pretrained_config_with_workers(
        model, workers, trust_remote_code=trust_remote_code
    )


@pytest.fixture(autouse=True)
def _mock_deepseek_pretrained_configs():
    with patch(
        "gpustack.policies.candidate_selectors.base_candidate_selector.get_pretrained_config_with_workers",
        new=_mock_or_real_get_pretrained_config_with_workers,
    ):
        yield
