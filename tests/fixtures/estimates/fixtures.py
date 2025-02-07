import os
from gpustack.scheduler.calculator import modelResoruceClaim

# The data structure is compatible with the gguf-parser-go (v0.9.2), data values are designed according to test needs


def llama3_8b_partial_offload_estimate_claim():
    '''
    gguf-parser --ol-model llama3:8b \
    --gpu-layers=-1 --gpu-layers-step=1 --ctx-size 8192 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_8b_partial_offload_estimate_claim.json
    '''
    return load_model_estimate_claim_from_file(
        "llama3_8b_partial_offload_estimate_claim.json"
    )


def llama3_8b_full_offload_estimate_claim():
    '''
    gguf-parser --ol-model llama3:8b \
    --gpu-layers=-1 --ctx-size 8192 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_8b_full_offload_estimate_claim.json
    '''
    return load_model_estimate_claim_from_file(
        "llama3_8b_full_offload_estimate_claim.json"
    )


def llama3_8b_disable_offload_estimate_claim():
    '''
    gguf-parser --ol-model llama3:8b \
    --gpu-layers=0 --ctx-size=8192  \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_8b_disable_offload_estimate_claim.json
    '''
    return load_model_estimate_claim_from_file(
        "llama3_8b_disable_offload_estimate_claim.json"
    )


def llama3_70b_partial_offload_estimate_claim():
    '''
    gguf-parser --ol-model llama3:70b \
    --gpu-layers=-1 --gpu-layers-step=1 --ctx-size 8192 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_70b_partial_offload_estimate_claim.json
    '''
    return load_model_estimate_claim_from_file(
        "llama3_70b_partial_offload_estimate_claim.json"
    )


def llama3_70b_full_offload_estimate_claim():
    '''
    gguf-parser --ol-model llama3:70b \
    --gpu-layers=-1 --ctx-size 8192 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_70b_full_offload_estimate_claim.json
    '''
    return load_model_estimate_claim_from_file(
        "llama3_70b_full_offload_estimate_claim.json"
    )


def llama3_70b_disable_offload_estimate_claim():
    '''
    gguf-parser --ol-model llama3:70b \
    --gpu-layers=0 --ctx-size 8192 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_70b_disable_offload_estimate_claim.json
    '''
    return load_model_estimate_claim_from_file(
        "llama3_70b_disable_offload_estimate_claim.json"
    )


def llama3_70b_full_offload_split_2_4080_estimate_claim():
    '''
    gguf-parser --ol-model llama3:70b \
    --gpu-layers=-1 --ctx-size 8192 \
    --tensor-split=17171480576,17171480576 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_70b_full_offload_split_2_4080_estimate_claim.json
    '''
    return load_model_estimate_claim_from_file(
        "llama3_70b_full_offload_split_2_4080_estimate_claim.json"
    )


def llama3_70b_full_offload_split_2_4080_estimate_claim_2():
    '''
    gguf-parser --ol-model llama3:70b \
    --gpu-layers=-1 --ctx-size 8192 \
    --tensor-split=17171480576,16647192576 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_70b_full_offload_split_2_4090_estimate_claim_2.json
    or
    gguf-parser --ol-model llama3:70b \
    --gpu-layers=-1 --ctx-size 8192 \
    --tensor-split=17171480576,16542334976 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_70b_full_offload_split_2_4090_estimate_claim_2.json
    '''
    return load_model_estimate_claim_from_file(
        "llama3_70b_full_offload_split_2_4090_estimate_claim_2.json"
    )


def llama3_70b_full_offload_split_2_4080_estimate_claim_3():
    '''
    gguf-parser --ol-model llama3:70b \
    --gpu-layers=-1 --ctx-size 8192 \
    --tensor-split=16647192576,16542334976 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_70b_full_offload_split_2_4090_estimate_claim_3.json
    '''
    return load_model_estimate_claim_from_file(
        "llama3_70b_full_offload_split_2_4090_estimate_claim_3.json"
    )


def llama3_70b_full_offload_split_3_4080_estimate_claim():
    '''
    gguf-parser --ol-model llama3:70b \
    --gpu-layers=-1 --ctx-size 8192 \
    --tensor-split=17171480576,17171480576,16647192576 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_70b_full_offload_split_3_4080_estimate_claim.json
    or
    gguf-parser --ol-model llama3:70b \
    --gpu-layers=-1 --ctx-size 8192 \
    --tensor-split=17171480576,17171480576,16542334976 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_70b_full_offload_split_3_4080_estimate_claim.json
    '''
    return load_model_estimate_claim_from_file(
        "llama3_70b_full_offload_split_3_4080_estimate_claim.json"
    )


def llama3_70b_full_offload_split_3_4080_estimate_claim_2():
    '''
    gguf-parser --ol-model llama3:70b \
    --gpu-layers=-1 --ctx-size 8192 \
    --tensor-split=17171480576,16647192576,16542334976 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_70b_full_offload_split_3_4080_estimate_claim_2.json
    '''
    return load_model_estimate_claim_from_file(
        "llama3_70b_full_offload_split_3_4080_estimate_claim_2.json"
    )


def llama3_70b_full_offload_split_3_4080_estimate_claim_3():
    '''
    gguf-parser --ol-model llama3:70b \
    --gpu-layers=-1 --ctx-size 8192 \
    --tensor-split=17171480576,17171480576,17171480576 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_70b_full_offload_split_3_4080_estimate_claim_3.json
    '''
    return load_model_estimate_claim_from_file(
        "llama3_70b_full_offload_split_3_4080_estimate_claim_3.json"
    )


def llama3_70b_full_offload_split_4_4080_estimate_claim():
    '''
    gguf-parser --ol-model llama3:70b \
    --gpu-layers=-1 --ctx-size 8192 \
    --tensor-split=17171480576,17171480576,17171480576,17171480576 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_70b_full_offload_split_4_4080_estimate_claim.json
    '''
    return load_model_estimate_claim_from_file(
        "llama3_70b_full_offload_split_4_4080_estimate_claim.json"
    )


def llama3_70b_full_offload_split_2_4090_estimate_claim():
    '''
    gguf-parser --ol-model llama3:70b \
    --gpu-layers=-1 --ctx-size 8192 \
    --tensor-split=26015170560,26015170560 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_70b_full_offload_split_2_4090_estimate_claim.json
    '''
    return load_model_estimate_claim_from_file(
        "llama3_70b_full_offload_split_2_4090_estimate_claim.json"
    )


def llama3_70b_partial_offload_split_2_4080_estimate_claim():
    '''
    gguf-parser --ol-model llama3:70b \
    --gpu-layers=-1 --gpu-layers-step=1 --ctx-size 8192 \
    --tensor-split=17171480576,17171480576 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_70b_partial_offload_split_2_4080_estimate_claim.json
    '''
    return load_model_estimate_claim_from_file(
        "llama3_70b_partial_offload_split_2_4080_estimate_claim.json"
    )


def llama3_70b_partial_offload_split_2_4080_4090_estimate_claim():
    '''
    gguf-parser --ol-model llama3:70b \
    --gpu-layers=-1 --gpu-layers-step=1 --ctx-size 8192 \
    --tensor-split=17171480576,26015170560 --rpc=host:50020 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_70b_partial_offload_split_2_4080_4090_estimate_claim.json
    '''
    return load_model_estimate_claim_from_file(
        "llama3_70b_partial_offload_split_2_4080_4090_estimate_claim.json"
    )


def llama3_70b_partial_offload_split_3_4080_4090_estimate_claim():
    '''
    gguf-parser --ol-model llama3:70b \
    --gpu-layers=-1 --gpu-layers-step=1 --ctx-size 8192 \
    --tensor-split=26015170560,17171480576,17171480576 --rpc=host:50020 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_70b_partial_offload_split_3_4080_4090_estimate_claim.json
    '''
    return load_model_estimate_claim_from_file(
        "llama3_70b_partial_offload_split_3_4080_4090_estimate_claim.json"
    )


def llama3_70b_partial_offload_split_3_4080_4090_estimate_claim_2():
    '''
    gguf-parser --ol-model llama3:70b \
    --gpu-layers=-1 --gpu-layers-step=1 --ctx-size 8192 \
    --tensor-split=17171480576,17171480576,26015170560 --rpc=host:50020,host:50021 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_70b_partial_offload_split_3_4080_4090_estimate_claim_2.json
    '''
    return load_model_estimate_claim_from_file(
        "llama3_70b_partial_offload_split_3_4080_4090_estimate_claim_2.json"
    )


def deepseek_r1_q4_k_m_full_offload_estimate_claim():
    # gguf-parser 0.13.10
    '''
    gguf-parser --in-max-ctx-size --skip-tokenizer --skip-architecture --skip-metadata \
    --image-vae-tiling --cache-expiration 168h0m0s --no-mmap \
    --ctx-size 2048 --cache-path /opt/models/cache/gguf-parser --gpu-layers -1 \
    -hf-repo unsloth/DeepSeek-R1-GGUF -hf-file DeepSeek-R1-Q4_K_M/DeepSeek-R1-Q4_K_M-00001-of-00009.gguf --json >> deepseek_r1_q4_k_m_full_offload_estimate_claim.json
    '''
    return load_model_estimate_claim_from_file(
        "deepseek_r1_q4_k_m_full_offload_estimate_claim.json"
    )


def deepseek_r1_q4_k_m_partial_offload_split_6_estimate_claim():
    # gguf-parser 0.13.10
    '''
    gguf-parser --in-max-ctx-size --skip-tokenizer --skip-architecture --skip-metadata \
    --image-vae-tiling --cache-expiration 168h0m0s --no-mmap \
    --ctx-size 2048 --cache-path /opt/models/cache/gguf-parser --gpu-layers-step 1 \
    --tensor-split 81920,81920,81920,81920,81920,81920 --rpc llm02-A100:50053,llm02-A100:50054,llm03-A100:50055,llm03-A100:50056 \
    -hf-repo unsloth/DeepSeek-R1-GGUF -hf-file DeepSeek-R1-Q4_K_M/DeepSeek-R1-Q4_K_M-00001-of-00009.gguf --json >> deepseek_r1_q4_k_m_patial_offload_split_6_estimate_claim.json
    '''
    return load_model_estimate_claim_from_file(
        "deepseek_r1_q4_k_m_patial_offload_split_6_estimate_claim.json"
    )


def deepseek_r1_q4_k_m_partial_offload_estimate_claim():
    # gguf-parser 0.13.10
    '''
    gguf-parser --in-max-ctx-size --skip-tokenizer --skip-architecture --skip-metadata \
    --image-vae-tiling --cache-expiration 168h0m0s --no-mmap \
    --ctx-size 2048 --cache-path /opt/models/cache/gguf-parser --gpu-layers-step 1 \
    -hf-repo unsloth/DeepSeek-R1-GGUF -hf-file DeepSeek-R1-Q4_K_M/DeepSeek-R1-Q4_K_M-00001-of-00009.gguf --json >> deepseek_r1_q4_k_m_patial_offload_estimate_claim.json
    '''
    return load_model_estimate_claim_from_file(
        "deepseek_r1_q4_k_m_patial_offload_estimate_claim.json"
    )


def deepseek_r1_ud_iq2_xxs_full_offload_estimate_claim():
    # gguf-parser 0.13.10
    '''
    gguf-parser --in-max-ctx-size --skip-tokenizer --skip-architecture --skip-metadata \
    --image-vae-tiling --cache-expiration 168h0m0s --no-mmap \
    --ctx-size 2048 --cache-path /opt/models/cache/gguf-parser --gpu-layers -1 \
    -hf-repo unsloth/DeepSeek-R1-GGUF -hf-file DeepSeek-R1-UD-IQ2_XXS/DeepSeek-R1-UD-IQ2_XXS-00001-of-00004.gguf --json >> deepseek_r1_ud_iq2_xxs_full_offload_estimate_claim.json
    '''
    return load_model_estimate_claim_from_file(
        "deepseek_r1_ud_iq2_xxs_full_offload_estimate_claim.json"
    )


def deepseek_r1_ud_iq2_xxs_full_offload_split_2_estimate_claim():
    # gguf-parser 0.13.10
    '''
    gguf-parser --in-max-ctx-size --skip-tokenizer --skip-architecture --skip-metadata \
    --image-vae-tiling --cache-expiration 168h0m0s --no-mmap \
    --ctx-size 2048 --cache-path /opt/models/cache/gguf-parser --gpu-layers -1 \
    --tensor-split 24576,24576 \
    -hf-repo unsloth/DeepSeek-R1-GGUF -hf-file DeepSeek-R1-UD-IQ2_XXS/DeepSeek-R1-UD-IQ2_XXS-00001-of-00004.gguf --json >> deepseek_r1_ud_iq2_xxs_full_offload_split_2_estimate_claim.json
    '''
    return load_model_estimate_claim_from_file(
        "deepseek_r1_ud_iq2_xxs_full_offload_split_2_estimate_claim.json"
    )


def deepseek_r1_ud_iq2_xxs_full_offload_split_3_estimate_claim():
    # gguf-parser 0.13.10
    '''
    gguf-parser --in-max-ctx-size --skip-tokenizer --skip-architecture --skip-metadata \
    --image-vae-tiling --cache-expiration 168h0m0s --no-mmap \
    --ctx-size 2048 --cache-path /opt/models/cache/gguf-parser --gpu-layers -1 \
    --tensor-split 24576,24576,24576 \
    -hf-repo unsloth/DeepSeek-R1-GGUF -hf-file DeepSeek-R1-UD-IQ2_XXS/DeepSeek-R1-UD-IQ2_XXS-00001-of-00004.gguf --json >> deepseek_r1_ud_iq2_xxs_full_offload_split_3_estimate_claim.json
    '''
    return load_model_estimate_claim_from_file(
        "deepseek_r1_ud_iq2_xxs_full_offload_split_3_estimate_claim.json"
    )


def deepseek_r1_ud_iq2_xxs_full_offload_split_4_estimate_claim():
    # gguf-parser 0.13.10
    '''
    gguf-parser --in-max-ctx-size --skip-tokenizer --skip-architecture --skip-metadata \
    --image-vae-tiling --cache-expiration 168h0m0s --no-mmap \
    --ctx-size 2048 --cache-path /opt/models/cache/gguf-parser --gpu-layers -1 \
    --tensor-split 24576,24576,24576,24576 \
    -hf-repo unsloth/DeepSeek-R1-GGUF -hf-file DeepSeek-R1-UD-IQ2_XXS/DeepSeek-R1-UD-IQ2_XXS-00001-of-00004.gguf --json >> deepseek_r1_ud_iq2_xxs_full_offload_split_4_estimate_claim.json
    '''
    return load_model_estimate_claim_from_file(
        "deepseek_r1_ud_iq2_xxs_full_offload_split_4_estimate_claim.json"
    )


def deepseek_r1_ud_iq2_xxs_full_offload_split_5_estimate_claim():
    # gguf-parser 0.13.10
    '''
    gguf-parser --in-max-ctx-size --skip-tokenizer --skip-architecture --skip-metadata \
    --image-vae-tiling --cache-expiration 168h0m0s --no-mmap \
    --ctx-size 2048 --cache-path /opt/models/cache/gguf-parser --gpu-layers -1 \
    --tensor-split 24576,24576,24576,24576,24576 \
    -hf-repo unsloth/DeepSeek-R1-GGUF -hf-file DeepSeek-R1-UD-IQ2_XXS/DeepSeek-R1-UD-IQ2_XXS-00001-of-00004.gguf --json >> deepseek_r1_ud_iq2_xxs_full_offload_split_5_estimate_claim.json
    '''
    return load_model_estimate_claim_from_file(
        "deepseek_r1_ud_iq2_xxs_full_offload_split_5_estimate_claim.json"
    )


def deepseek_r1_ud_iq2_xxs_full_offload_split_6_estimate_claim():
    # gguf-parser 0.13.10
    '''
    gguf-parser --in-max-ctx-size --skip-tokenizer --skip-architecture --skip-metadata \
    --image-vae-tiling --cache-expiration 168h0m0s --no-mmap \
    --ctx-size 2048 --cache-path /opt/models/cache/gguf-parser --gpu-layers -1 \
    --tensor-split 24576,24576,24576,24576,24576,24576 \
    -hf-repo unsloth/DeepSeek-R1-GGUF -hf-file DeepSeek-R1-UD-IQ2_XXS/DeepSeek-R1-UD-IQ2_XXS-00001-of-00004.gguf --json >> deepseek_r1_ud_iq2_xxs_full_offload_split_6_estimate_claim.json
    '''
    return load_model_estimate_claim_from_file(
        "deepseek_r1_ud_iq2_xxs_full_offload_split_6_estimate_claim.json"
    )


def deepseek_r1_ud_iq2_xxs_full_offload_split_7_estimate_claim():
    # gguf-parser 0.13.10
    '''
    gguf-parser --in-max-ctx-size --skip-tokenizer --skip-architecture --skip-metadata \
    --image-vae-tiling --cache-expiration 168h0m0s --no-mmap \
    --ctx-size 2048 --cache-path /opt/models/cache/gguf-parser --gpu-layers -1 \
    --tensor-split 24576,24576,24576,24576,24576,24576,24576 \
    -hf-repo unsloth/DeepSeek-R1-GGUF -hf-file DeepSeek-R1-UD-IQ2_XXS/DeepSeek-R1-UD-IQ2_XXS-00001-of-00004.gguf --json >> deepseek_r1_ud_iq2_xxs_full_offload_split_7_estimate_claim.json
    '''
    return load_model_estimate_claim_from_file(
        "deepseek_r1_ud_iq2_xxs_full_offload_split_7_estimate_claim.json"
    )


def deepseek_r1_ud_iq2_xxs_partial_offload_estimate_claim():
    # gguf-parser 0.13.10
    '''
    gguf-parser --in-max-ctx-size --skip-tokenizer --skip-architecture --skip-metadata \
    --image-vae-tiling --cache-expiration 168h0m0s --no-mmap \
    --ctx-size 2048 --cache-path /opt/models/cache/gguf-parser --gpu-layers-step 1 \
    -hf-repo unsloth/DeepSeek-R1-GGUF -hf-file DeepSeek-R1-UD-IQ2_XXS/DeepSeek-R1-UD-IQ2_XXS-00001-of-00004.gguf --json >> deepseek_r1_ud_iq2_xxs_partial_offload_estimate_claim.json
    '''
    return load_model_estimate_claim_from_file(
        "deepseek_r1_ud_iq2_xxs_partial_offload_estimate_claim.json"
    )


def deepseek_r1_ud_iq2_xxs_full_offload_split_8_estimate_claim():
    # gguf-parser 0.13.10
    '''
    gguf-parser --in-max-ctx-size --skip-tokenizer --skip-architecture --skip-metadata \
    --image-vae-tiling --cache-expiration 168h0m0s --no-mmap \
    --ctx-size 2048 --cache-path /opt/models/cache/gguf-parser --gpu-layers -1 \
    --tensor-split 24576,24576,24576,24576,24576,24576,24576,24576 \
    -hf-repo unsloth/DeepSeek-R1-GGUF -hf-file DeepSeek-R1-UD-IQ2_XXS/DeepSeek-R1-UD-IQ2_XXS-00001-of-00004.gguf --json >> deepseek_r1_ud_iq2_xxs_full_offload_split_8_estimate_claim.json
    '''
    return load_model_estimate_claim_from_file(
        "deepseek_r1_ud_iq2_xxs_full_offload_split_8_estimate_claim.json"
    )


def deepseek_r1_ud_iq2_xxs_partial_offload_split_2_estimate_claim():
    # gguf-parser 0.13.10
    '''
    gguf-parser --in-max-ctx-size --skip-tokenizer --skip-architecture --skip-metadata \
    --image-vae-tiling --cache-expiration 168h0m0s --no-mmap \
    --ctx-size 2048 --cache-path /opt/models/cache/gguf-parser --gpu-layers-step 1 \
    --tensor-split 24576,24576 \
    -hf-repo unsloth/DeepSeek-R1-GGUF -hf-file DeepSeek-R1-UD-IQ2_XXS/DeepSeek-R1-UD-IQ2_XXS-00001-of-00004.gguf --json >> deepseek_r1_ud_iq2_xxs_partial_offload_split_2_estimate_claim.json
    '''
    return load_model_estimate_claim_from_file(
        "deepseek_r1_ud_iq2_xxs_partial_offload_split_2_estimate_claim.json"
    )


def deepseek_r1_ud_iq2_xxs_partial_offload_split_3_estimate_claim():
    # gguf-parser 0.13.10
    '''
    gguf-parser --in-max-ctx-size --skip-tokenizer --skip-architecture --skip-metadata \
    --image-vae-tiling --cache-expiration 168h0m0s --no-mmap \
    --ctx-size 2048 --cache-path /opt/models/cache/gguf-parser --gpu-layers-step 1 \
    --tensor-split 24576,24576,24576 \
    -hf-repo unsloth/DeepSeek-R1-GGUF -hf-file DeepSeek-R1-UD-IQ2_XXS/DeepSeek-R1-UD-IQ2_XXS-00001-of-00004.gguf --json >> deepseek_r1_ud_iq2_xxs_partial_offload_split_3_estimate_claim.json
    '''
    return load_model_estimate_claim_from_file(
        "deepseek_r1_ud_iq2_xxs_partial_offload_split_3_estimate_claim.json"
    )


def deepseek_r1_ud_iq2_xxs_partial_offload_split_4_estimate_claim():
    # gguf-parser 0.13.10
    '''
    gguf-parser --in-max-ctx-size --skip-tokenizer --skip-architecture --skip-metadata \
    --image-vae-tiling --cache-expiration 168h0m0s --no-mmap \
    --ctx-size 2048 --cache-path /opt/models/cache/gguf-parser --gpu-layers-step 1 \
    --tensor-split 24576,24576,24576,24576 \
    -hf-repo unsloth/DeepSeek-R1-GGUF -hf-file DeepSeek-R1-UD-IQ2_XXS/DeepSeek-R1-UD-IQ2_XXS-00001-of-00004.gguf --json >> deepseek_r1_ud_iq2_xxs_partial_offload_split_4_estimate_claim.json
    '''
    return load_model_estimate_claim_from_file(
        "deepseek_r1_ud_iq2_xxs_partial_offload_split_4_estimate_claim.json"
    )


def deepseek_r1_ud_iq2_xxs_partial_offload_split_5_estimate_claim():
    # gguf-parser 0.13.10
    '''
    gguf-parser --in-max-ctx-size --skip-tokenizer --skip-architecture --skip-metadata \
    --image-vae-tiling --cache-expiration 168h0m0s --no-mmap \
    --ctx-size 2048 --cache-path /opt/models/cache/gguf-parser --gpu-layers-step 1 \
    --tensor-split 24576,24576,24576,24576,24576 \
    -hf-repo unsloth/DeepSeek-R1-GGUF -hf-file DeepSeek-R1-UD-IQ2_XXS/DeepSeek-R1-UD-IQ2_XXS-00001-of-00004.gguf --json >> deepseek_r1_ud_iq2_xxs_partial_offload_split_5_estimate_claim.json
    '''
    return load_model_estimate_claim_from_file(
        "deepseek_r1_ud_iq2_xxs_partial_offload_split_5_estimate_claim.json"
    )


def deepseek_r1_ud_iq2_xxs_partial_offload_split_6_estimate_claim():
    # gguf-parser 0.13.10
    '''
    gguf-parser --in-max-ctx-size --skip-tokenizer --skip-architecture --skip-metadata \
    --image-vae-tiling --cache-expiration 168h0m0s --no-mmap \
    --ctx-size 2048 --cache-path /opt/models/cache/gguf-parser --gpu-layers-step 1 \
    --tensor-split 24576,24576,24576,24576,24576,24576 \
    -hf-repo unsloth/DeepSeek-R1-GGUF -hf-file DeepSeek-R1-UD-IQ2_XXS/DeepSeek-R1-UD-IQ2_XXS-00001-of-00004.gguf --json >> deepseek_r1_ud_iq2_xxs_partial_offload_split_6_estimate_claim.json
    '''
    return load_model_estimate_claim_from_file(
        "deepseek_r1_ud_iq2_xxs_partial_offload_split_6_estimate_claim.json"
    )


def deepseek_r1_ud_iq2_xxs_partial_offload_split_7_estimate_claim():
    # gguf-parser 0.13.10
    '''
    gguf-parser --in-max-ctx-size --skip-tokenizer --skip-architecture --skip-metadata \
    --image-vae-tiling --cache-expiration 168h0m0s --no-mmap \
    --ctx-size 2048 --cache-path /opt/models/cache/gguf-parser --gpu-layers-step 1 \
    --tensor-split 24576,24576,24576,24576,24576,24576,24576 \
    -hf-repo unsloth/DeepSeek-R1-GGUF -hf-file DeepSeek-R1-UD-IQ2_XXS/DeepSeek-R1-UD-IQ2_XXS-00001-of-00004.gguf --json >> deepseek_r1_ud_iq2_xxs_partial_offload_split_7_estimate_claim.json
    '''
    return load_model_estimate_claim_from_file(
        "deepseek_r1_ud_iq2_xxs_partial_offload_split_7_estimate_claim.json"
    )


def deepseek_r1_ud_iq2_xxs_partial_offload_split_8_estimate_claim():
    # gguf-parser 0.13.10
    '''
    gguf-parser --in-max-ctx-size --skip-tokenizer --skip-architecture --skip-metadata \
    --image-vae-tiling --cache-expiration 168h0m0s --no-mmap \
    --ctx-size 2048 --cache-path /opt/models/cache/gguf-parser --gpu-layers-step 1 \
    --tensor-split 24576,24576,24576,24576,24576,24576,24576,24576 \
    -hf-repo unsloth/DeepSeek-R1-GGUF -hf-file DeepSeek-R1-UD-IQ2_XXS/DeepSeek-R1-UD-IQ2_XXS-00001-of-00004.gguf --json >> deepseek_r1_ud_iq2_xxs_partial_offload_split_8_estimate_claim.json
    '''
    return load_model_estimate_claim_from_file(
        "deepseek_r1_ud_iq2_xxs_partial_offload_split_8_estimate_claim.json"
    )


def deepseek_r1_distill_qwen_32b_bf16_partial_offload_split_3_1_estimate_claim():
    # gguf-parser 0.13.10
    '''
    gguf-parser --in-max-ctx-size --skip-tokenizer --skip-architecture --skip-metadata \
    --image-vae-tiling --cache-expiration 168h0m0s --no-mmap \
    --ctx-size 32768 --cache-path /opt/models/cache/gguf-parser --gpu-layers-step 1 \
    --tensor-split 16368,15360,23540 --rpc=host:50020,host:50020 \
    -hf-repo bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF -hf-file DeepSeek-R1-Distill-Qwen-32B-bf16/DeepSeek-R1-Distill-Qwen-32B-bf16-00001-of-00002.gguf \
    --json >> deepseek_r1_distill_qwen_32b_bf16_partial_offload_split_3_1_estimate_claim.json
    '''
    return load_model_estimate_claim_from_file(
        "deepseek_r1_distill_qwen_32b_bf16_partial_offload_split_3_1_estimate_claim.json"
    )


def deepseek_r1_distill_qwen_32b_bf16_partial_offload_split_3_2_estimate_claim():
    # gguf-parser 0.13.10
    '''
    gguf-parser --in-max-ctx-size --skip-tokenizer --skip-architecture --skip-metadata \
    --image-vae-tiling --cache-expiration 168h0m0s --no-mmap \
    --ctx-size 32768 --cache-path /opt/models/cache/gguf-parser --gpu-layers-step 1 \
    --tensor-split 23540,15360,16368 --rpc=host:50020,host:50020 \
    -hf-repo bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF -hf-file DeepSeek-R1-Distill-Qwen-32B-bf16/DeepSeek-R1-Distill-Qwen-32B-bf16-00001-of-00002.gguf \
    --json >> deepseek_r1_distill_qwen_32b_bf16_partial_offload_split_3_2_estimate_claim.json
    '''
    return load_model_estimate_claim_from_file(
        "deepseek_r1_distill_qwen_32b_bf16_partial_offload_split_3_2_estimate_claim.json"
    )


def deepseek_r1_distill_qwen_32b_bf16_partial_offload_split_3_3_estimate_claim():
    # gguf-parser 0.13.10
    '''
    gguf-parser --in-max-ctx-size --skip-tokenizer --skip-architecture --skip-metadata \
    --image-vae-tiling --cache-expiration 168h0m0s --no-mmap \
    --ctx-size 32768 --cache-path /opt/models/cache/gguf-parser --gpu-layers-step 1 \
    --tensor-split 23540,16368,15360 --rpc=host:50020,host:50020 \
    -hf-repo bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF -hf-file DeepSeek-R1-Distill-Qwen-32B-bf16/DeepSeek-R1-Distill-Qwen-32B-bf16-00001-of-00002.gguf \
    --json >> deepseek_r1_distill_qwen_32b_bf16_partial_offload_split_3_3_estimate_claim.json
    '''
    return load_model_estimate_claim_from_file(
        "deepseek_r1_distill_qwen_32b_bf16_partial_offload_split_3_3_estimate_claim.json"
    )


def load_model_estimate_claim_from_file(file_name) -> modelResoruceClaim:
    dir = os.path.dirname(__file__)
    file_path = os.path.join(dir, file_name)
    with open(file_path, 'r') as file:
        model_estimate_claim = modelResoruceClaim.from_json(file.read())
    return model_estimate_claim
