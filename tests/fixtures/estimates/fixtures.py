import os
from gpustack.scheduler.calculator import GGUFParserOutput


def llama3_70b_full_offload():
    # gguf-parser 0.9.2
    '''
    gguf-parser --ol-model llama3:70b \
    --gpu-layers=-1 --gpu-layers=999 --ctx-size 8192 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_70b_full_offload.json
    '''
    return load_model_claim_from_file("llama3_70b_full_offload.json")


def llama3_70b_full_offload_split_2_4080():
    # gguf-parser 0.9.2
    '''
    gguf-parser --ol-model llama3:70b \
    --gpu-layers=999 --gpu-layers-step=1 --ctx-size 8192 \
    --tensor-split=17171480576,17171480576 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_70b_full_offload_split_2_4080.json
    '''
    return load_model_claim_from_file("llama3_70b_full_offload_split_2_4080.json")


def deepseek_r1_ud_iq2_xxs_full_offload():
    # gguf-parser 0.13.10
    '''
    gguf-parser --in-max-ctx-size --skip-tokenizer --skip-architecture --skip-metadata \
    --image-vae-tiling --cache-expiration 168h0m0s --no-mmap \
    --ctx-size 2048 --cache-path /opt/models/cache/gguf-parser --gpu-layers 999 \
    -hf-repo unsloth/DeepSeek-R1-GGUF -hf-file DeepSeek-R1-UD-IQ2_XXS/DeepSeek-R1-UD-IQ2_XXS-00001-of-00004.gguf \
    --json >> deepseek_r1_ud_iq2_xxs_full_offload.json
    '''
    return load_model_claim_from_file("deepseek_r1_ud_iq2_xxs_full_offload.json")


def deepseek_r1_ud_iq2_xxs_full_offload_split_8():
    # gguf-parser 0.13.10
    '''
    gguf-parser --in-max-ctx-size --skip-tokenizer --skip-architecture --skip-metadata \
    --image-vae-tiling --cache-expiration 168h0m0s --no-mmap \
    --ctx-size 2048 --cache-path /opt/models/cache/gguf-parser --gpu-layers 999 \
    --tensor-split 24576,24576,24576,24576,24576,24576,24576,24576 \
    -hf-repo unsloth/DeepSeek-R1-GGUF -hf-file DeepSeek-R1-UD-IQ2_XXS/DeepSeek-R1-UD-IQ2_XXS-00001-of-00004.gguf \
    --json >> deepseek_r1_ud_iq2_xxs_full_offload_split_8.json
    '''
    return load_model_claim_from_file(
        "deepseek_r1_ud_iq2_xxs_full_offload_split_8.json"
    )


def llama3_8b_partial_offload():
    # gguf-parser 0.9.2
    '''
    gguf-parser --ol-model llama3:8b \
    --gpu-layers=-1 --gpu-layers-step=1 --ctx-size 8192 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_8b_partial_offload.json
    '''
    return load_model_claim_from_file("llama3_8b_partial_offload.json")


def llama3_8b_partial_offload_split_1main_1rpc():
    # gguf-parser 0.13.10
    '''
    gguf-parser --ol-model llama3:8b \
    --gpu-layers=-1 --gpu-layers-step=1 --ctx-size 8192 \
    --skip-tokenizer --skip-architecture --skip-metadata \
    --tensor-split 1,1 --rpc host:port \
    --json >> llama3_8b_partial_offload_split_1main_1rpc.json
    '''
    return load_model_claim_from_file("llama3_8b_partial_offload_split_1main_1rpc.json")


def llama3_8b_disable_offload():
    # gguf-parser 0.9.2
    '''
    gguf-parser --ol-model llama3:8b \
    --gpu-layers=0 --ctx-size=8192  \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_8b_disable_offload.json
    '''
    return load_model_claim_from_file("llama3_8b_disable_offload.json")


def llama3_70b_partial_offload():
    # gguf-parser 0.9.2
    '''
    gguf-parser --ol-model llama3:70b \
    --gpu-layers=-1 --gpu-layers-step=1 --ctx-size 8192 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_70b_partial_offload.json
    '''
    return load_model_claim_from_file("llama3_70b_partial_offload.json")


def llama3_70b_disable_offload():
    # gguf-parser 0.9.2
    '''
    gguf-parser --ol-model llama3:70b \
    --gpu-layers=0 --ctx-size 8192 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_70b_disable_offload.json
    '''
    return load_model_claim_from_file("llama3_70b_disable_offload.json")


def llama3_70b_partial_offload_split_2_4080():
    # gguf-parser 0.9.2
    '''
    gguf-parser --ol-model llama3:70b \
    --gpu-layers=-1 --gpu-layers-step=1 --ctx-size 8192 \
    --tensor-split=17171480576,17171480576 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_70b_partial_offload_split_2_4080.json
    '''
    return load_model_claim_from_file("llama3_70b_partial_offload_split_2_4080.json")


def llama3_70b_partial_offload_split_2_4090():
    # gguf-parser 0.9.2
    '''
    gguf-parser --ol-model llama3:70b \
    --gpu-layers=-1 --gpu-layers-step=1 --ctx-size 8192 \
    --tensor-split=26015170560,26015170560 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_70b_partial_offload_split_2_4090.json
    '''
    return load_model_claim_from_file("llama3_70b_partial_offload_split_2_4090.json")


def llama3_70b_partial_offload_split_3_4080():
    # gguf-parser 0.13.10
    '''
    gguf-parser --ol-model llama3:70b \
    --gpu-layers=-1 --gpu-layers-step=1 --ctx-size 8192 \
    --tensor-split=17171480576,17171480576,17171480576 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_70b_partial_offload_split_3_4080.json
    '''
    return load_model_claim_from_file("llama3_70b_partial_offload_split_3_4080.json")


def llama3_70b_partial_offload_split_2_4080_4090():
    # gguf-parser 0.9.2
    '''
    gguf-parser --ol-model llama3:70b \
    --gpu-layers=-1 --gpu-layers-step=1 --ctx-size 8192 \
    --tensor-split=17171480576,26015170560 --rpc=host:50020 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_70b_partial_offload_split_2_4080_4090.json
    '''
    return load_model_claim_from_file(
        "llama3_70b_partial_offload_split_2_4080_4090.json"
    )


def llama3_70b_partial_offload_split_3_4080_4090():
    # gguf-parser 0.9.2
    '''
    gguf-parser --ol-model llama3:70b \
    --gpu-layers=-1 --gpu-layers-step=1 --ctx-size 8192 \
    --tensor-split=26015170560,17171480576,17171480576 --rpc=host:50020 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_70b_partial_offload_split_3_4080_4090.json
    '''
    return load_model_claim_from_file(
        "llama3_70b_partial_offload_split_3_4080_4090.json"
    )


def llama3_70b_partial_offload_split_3_4080_2():
    # gguf-parser 0.9.2
    '''
    gguf-parser --ol-model llama3:70b \
    --gpu-layers=-1 --gpu-layers-step=1 --ctx-size 8192 \
    --tensor-split=17171480576,17171480576,16647192576 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_70b_partial_offload_split_3_4080_2.json
    '''
    return load_model_claim_from_file("llama3_70b_partial_offload_split_3_4080_2.json")


def llama3_70b_partial_offload_split_3_4080_3():
    # gguf-parser 0.9.2
    '''
    gguf-parser --ol-model llama3:70b \
    --gpu-layers=-1 --gpu-layers-step=1 --ctx-size 8192 \
    --tensor-split=17171480576,17171480576,16542334976 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_70b_partial_offload_split_3_4080_3.json
    '''
    return load_model_claim_from_file("llama3_70b_partial_offload_split_3_4080_3.json")


def llama3_70b_partial_offload_split_3_4080_4():
    # gguf-parser 0.9.2
    '''
    gguf-parser --ol-model llama3:70b \
    --gpu-layers=-1 --gpu-layers-step=1 --ctx-size 8192 \
    --tensor-split=17171480576,16647192576,16647192576 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_70b_partial_offload_split_3_4080_4.json
    '''
    return load_model_claim_from_file("llama3_70b_partial_offload_split_3_4080_4.json")


def llama3_70b_partial_offload_split_3_4080_4090_2():
    # gguf-parser 0.9.2
    '''
    gguf-parser --ol-model llama3:70b \
    --gpu-layers=-1 --gpu-layers-step=1 --ctx-size 8192 \
    --tensor-split=17171480576,17171480576,26015170560 --rpc=host:50020,host:50021 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_70b_partial_offload_split_3_4080_4090_2.json
    '''
    return load_model_claim_from_file(
        "llama3_70b_partial_offload_split_3_4080_4090_2.json"
    )


def deepseek_r1_q4_k_m_partial_offload_split_1main_1rpc():
    # gguf-parser 0.13.10
    '''
    gguf-parser --in-max-ctx-size --skip-tokenizer --skip-architecture --skip-metadata \
    --image-vae-tiling --cache-expiration 168h0m0s --no-mmap \
    --ctx-size 2048 --cache-path /opt/models/cache/gguf-parser --gpu-layers-step 1 \
    --tensor-split 1,1 --rpc llm02-A100:50053 \
    -hf-repo unsloth/DeepSeek-R1-GGUF -hf-file DeepSeek-R1-Q4_K_M/DeepSeek-R1-Q4_K_M-00001-of-00009.gguf \
    --json >> deepseek_r1_q4_k_m_partial_offload_split_1main_1rpc.json
    '''
    return load_model_claim_from_file(
        "deepseek_r1_q4_k_m_partial_offload_split_1main_1rpc.json"
    )


def deepseek_r1_q4_k_m_partial_offload_split_6():
    # gguf-parser 0.13.10
    '''
    gguf-parser --in-max-ctx-size --skip-tokenizer --skip-architecture --skip-metadata \
    --image-vae-tiling --cache-expiration 168h0m0s --no-mmap \
    --ctx-size 2048 --cache-path /opt/models/cache/gguf-parser --gpu-layers-step 1 \
    --tensor-split 81920,81920,81920,81920,81920,81920 --rpc llm02-A100:50053,llm02-A100:50054,llm03-A100:50055,llm03-A100:50056 \
    -hf-repo unsloth/DeepSeek-R1-GGUF -hf-file DeepSeek-R1-Q4_K_M/DeepSeek-R1-Q4_K_M-00001-of-00009.gguf \
    --json >> deepseek_r1_q4_k_m_patial_offload_split_6.json
    '''
    return load_model_claim_from_file("deepseek_r1_q4_k_m_patial_offload_split_6.json")


def deepseek_r1_q4_k_m_partial_offload():
    # gguf-parser 0.13.10
    '''
    gguf-parser --in-max-ctx-size --skip-tokenizer --skip-architecture --skip-metadata \
    --image-vae-tiling --cache-expiration 168h0m0s --no-mmap \
    --ctx-size 2048 --cache-path /opt/models/cache/gguf-parser --gpu-layers-step 1 \
    -hf-repo unsloth/DeepSeek-R1-GGUF -hf-file DeepSeek-R1-Q4_K_M/DeepSeek-R1-Q4_K_M-00001-of-00009.gguf \
    --json >> deepseek_r1_q4_k_m_patial_offload.json
    '''
    return load_model_claim_from_file("deepseek_r1_q4_k_m_patial_offload.json")


def deepseek_r1_ud_iq2_xxs_partial_offload():
    # gguf-parser 0.13.10
    '''
    gguf-parser --in-max-ctx-size --skip-tokenizer --skip-architecture --skip-metadata \
    --image-vae-tiling --cache-expiration 168h0m0s --no-mmap \
    --ctx-size 2048 --cache-path /opt/models/cache/gguf-parser --gpu-layers-step 1 \
    -hf-repo unsloth/DeepSeek-R1-GGUF -hf-file DeepSeek-R1-UD-IQ2_XXS/DeepSeek-R1-UD-IQ2_XXS-00001-of-00004.gguf \
    --json >> deepseek_r1_ud_iq2_xxs_partial_offload.json
    '''
    return load_model_claim_from_file("deepseek_r1_ud_iq2_xxs_partial_offload.json")


def deepseek_r1_ud_iq2_xxs_partial_offload_split_2():
    # gguf-parser 0.13.10
    '''
    gguf-parser --in-max-ctx-size --skip-tokenizer --skip-architecture --skip-metadata \
    --image-vae-tiling --cache-expiration 168h0m0s --no-mmap \
    --ctx-size 2048 --cache-path /opt/models/cache/gguf-parser --gpu-layers-step 1 \
    --tensor-split 24576,24576 \
    -hf-repo unsloth/DeepSeek-R1-GGUF -hf-file DeepSeek-R1-UD-IQ2_XXS/DeepSeek-R1-UD-IQ2_XXS-00001-of-00004.gguf \
    --json >> deepseek_r1_ud_iq2_xxs_partial_offload_split_2.json
    '''
    return load_model_claim_from_file(
        "deepseek_r1_ud_iq2_xxs_partial_offload_split_2.json"
    )


def deepseek_r1_ud_iq2_xxs_partial_offload_split_3():
    # gguf-parser 0.13.10
    '''
    gguf-parser --in-max-ctx-size --skip-tokenizer --skip-architecture --skip-metadata \
    --image-vae-tiling --cache-expiration 168h0m0s --no-mmap \
    --ctx-size 2048 --cache-path /opt/models/cache/gguf-parser --gpu-layers-step 1 \
    --tensor-split 24576,24576,24576 \
    -hf-repo unsloth/DeepSeek-R1-GGUF -hf-file DeepSeek-R1-UD-IQ2_XXS/DeepSeek-R1-UD-IQ2_XXS-00001-of-00004.gguf \
    --json >> deepseek_r1_ud_iq2_xxs_partial_offload_split_3.json
    '''
    return load_model_claim_from_file(
        "deepseek_r1_ud_iq2_xxs_partial_offload_split_3.json"
    )


def deepseek_r1_ud_iq2_xxs_partial_offload_split_4():
    # gguf-parser 0.13.10
    '''
    gguf-parser --in-max-ctx-size --skip-tokenizer --skip-architecture --skip-metadata \
    --image-vae-tiling --cache-expiration 168h0m0s --no-mmap \
    --ctx-size 2048 --cache-path /opt/models/cache/gguf-parser --gpu-layers-step 1 \
    --tensor-split 24576,24576,24576,24576 \
    -hf-repo unsloth/DeepSeek-R1-GGUF -hf-file DeepSeek-R1-UD-IQ2_XXS/DeepSeek-R1-UD-IQ2_XXS-00001-of-00004.gguf \
    --json >> deepseek_r1_ud_iq2_xxs_partial_offload_split_4.json
    '''
    return load_model_claim_from_file(
        "deepseek_r1_ud_iq2_xxs_partial_offload_split_4.json"
    )


def deepseek_r1_ud_iq2_xxs_partial_offload_split_5():
    # gguf-parser 0.13.10
    '''
    gguf-parser --in-max-ctx-size --skip-tokenizer --skip-architecture --skip-metadata \
    --image-vae-tiling --cache-expiration 168h0m0s --no-mmap \
    --ctx-size 2048 --cache-path /opt/models/cache/gguf-parser --gpu-layers-step 1 \
    --tensor-split 24576,24576,24576,24576,24576 \
    -hf-repo unsloth/DeepSeek-R1-GGUF -hf-file DeepSeek-R1-UD-IQ2_XXS/DeepSeek-R1-UD-IQ2_XXS-00001-of-00004.gguf \
    --json >> deepseek_r1_ud_iq2_xxs_partial_offload_split_5.json
    '''
    return load_model_claim_from_file(
        "deepseek_r1_ud_iq2_xxs_partial_offload_split_5.json"
    )


def deepseek_r1_ud_iq2_xxs_partial_offload_split_6():
    # gguf-parser 0.13.10
    '''
    gguf-parser --in-max-ctx-size --skip-tokenizer --skip-architecture --skip-metadata \
    --image-vae-tiling --cache-expiration 168h0m0s --no-mmap \
    --ctx-size 2048 --cache-path /opt/models/cache/gguf-parser --gpu-layers-step 1 \
    --tensor-split 24576,24576,24576,24576,24576,24576 \
    -hf-repo unsloth/DeepSeek-R1-GGUF -hf-file DeepSeek-R1-UD-IQ2_XXS/DeepSeek-R1-UD-IQ2_XXS-00001-of-00004.gguf \
    --json >> deepseek_r1_ud_iq2_xxs_partial_offload_split_6.json
    '''
    return load_model_claim_from_file(
        "deepseek_r1_ud_iq2_xxs_partial_offload_split_6.json"
    )


def deepseek_r1_ud_iq2_xxs_partial_offload_split_7():
    # gguf-parser 0.13.10
    '''
    gguf-parser --in-max-ctx-size --skip-tokenizer --skip-architecture --skip-metadata \
    --image-vae-tiling --cache-expiration 168h0m0s --no-mmap \
    --ctx-size 2048 --cache-path /opt/models/cache/gguf-parser --gpu-layers-step 1 \
    --tensor-split 24576,24576,24576,24576,24576,24576,24576 \
    -hf-repo unsloth/DeepSeek-R1-GGUF -hf-file DeepSeek-R1-UD-IQ2_XXS/DeepSeek-R1-UD-IQ2_XXS-00001-of-00004.gguf \
    --json >> deepseek_r1_ud_iq2_xxs_partial_offload_split_7.json
    '''
    return load_model_claim_from_file(
        "deepseek_r1_ud_iq2_xxs_partial_offload_split_7.json"
    )


def deepseek_r1_ud_iq2_xxs_partial_offload_split_8():
    # gguf-parser 0.13.10
    '''
    gguf-parser --in-max-ctx-size --skip-tokenizer --skip-architecture --skip-metadata \
    --image-vae-tiling --cache-expiration 168h0m0s --no-mmap \
    --ctx-size 2048 --cache-path /opt/models/cache/gguf-parser --gpu-layers-step 1 \
    --tensor-split 24576,24576,24576,24576,24576,24576,24576,24576 \
    -hf-repo unsloth/DeepSeek-R1-GGUF -hf-file DeepSeek-R1-UD-IQ2_XXS/DeepSeek-R1-UD-IQ2_XXS-00001-of-00004.gguf \
    --json >> deepseek_r1_ud_iq2_xxs_partial_offload_split_8.json
    '''
    return load_model_claim_from_file(
        "deepseek_r1_ud_iq2_xxs_partial_offload_split_8.json"
    )


def deepseek_r1_distill_qwen_32b_bf16_partial_offload():
    # gguf-parser 0.13.10
    '''
    gguf-parser --in-max-ctx-size --skip-tokenizer --skip-architecture --skip-metadata \
    --image-vae-tiling --cache-expiration 168h0m0s --no-mmap \
    --ctx-size 32768 --cache-path /opt/models/cache/gguf-parser --gpu-layers-step 1 \
    -hf-repo bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF -hf-file DeepSeek-R1-Distill-Qwen-32B-bf16/DeepSeek-R1-Distill-Qwen-32B-bf16-00001-of-00002.gguf \
    --json >> deepseek_r1_distill_qwen_32b_bf16_partial_offload.json
    '''
    return load_model_claim_from_file(
        "deepseek_r1_distill_qwen_32b_bf16_partial_offload.json"
    )


def deepseek_r1_distill_qwen_32b_bf16_partial_offload_split_1main_1rpc():
    # gguf-parser 0.13.10
    '''
    gguf-parser --in-max-ctx-size --skip-tokenizer --skip-architecture --skip-metadata \
    --image-vae-tiling --cache-expiration 168h0m0s --no-mmap \
    --ctx-size 32768 --cache-path /opt/models/cache/gguf-parser --gpu-layers-step 1 \
    --tensor-split 1,1 --rpc=host:50020 \
    -hf-repo bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF -hf-file DeepSeek-R1-Distill-Qwen-32B-bf16/DeepSeek-R1-Distill-Qwen-32B-bf16-00001-of-00002.gguf \
    --json >> deepseek_r1_distill_qwen_32b_bf16_partial_offload_split_1main_1rpc.json
    '''
    return load_model_claim_from_file(
        "deepseek_r1_distill_qwen_32b_bf16_partial_offload_split_1main_1rpc.json"
    )


def deepseek_r1_distill_qwen_32b_bf16_partial_offload_split_3_1():
    # gguf-parser 0.13.10
    '''
    gguf-parser --in-max-ctx-size --skip-tokenizer --skip-architecture --skip-metadata \
    --image-vae-tiling --cache-expiration 168h0m0s --no-mmap \
    --ctx-size 32768 --cache-path /opt/models/cache/gguf-parser --gpu-layers-step 1 \
    --tensor-split 16368,15360,23540 --rpc=host:50020,host:50020 \
    -hf-repo bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF -hf-file DeepSeek-R1-Distill-Qwen-32B-bf16/DeepSeek-R1-Distill-Qwen-32B-bf16-00001-of-00002.gguf \
    --json >> deepseek_r1_distill_qwen_32b_bf16_partial_offload_split_3_1.json
    '''
    return load_model_claim_from_file(
        "deepseek_r1_distill_qwen_32b_bf16_partial_offload_split_3_1.json"
    )


def deepseek_r1_distill_qwen_32b_bf16_partial_offload_split_3_2():
    # gguf-parser 0.13.10
    '''
    gguf-parser --in-max-ctx-size --skip-tokenizer --skip-architecture --skip-metadata \
    --image-vae-tiling --cache-expiration 168h0m0s --no-mmap \
    --ctx-size 32768 --cache-path /opt/models/cache/gguf-parser --gpu-layers-step 1 \
    --tensor-split 23540,15360,16368 --rpc=host:50020,host:50020 \
    -hf-repo bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF -hf-file DeepSeek-R1-Distill-Qwen-32B-bf16/DeepSeek-R1-Distill-Qwen-32B-bf16-00001-of-00002.gguf \
    --json >> deepseek_r1_distill_qwen_32b_bf16_partial_offload_split_3_2.json
    '''
    return load_model_claim_from_file(
        "deepseek_r1_distill_qwen_32b_bf16_partial_offload_split_3_2.json"
    )


def deepseek_r1_distill_qwen_32b_bf16_partial_offload_split_3_3():
    # gguf-parser 0.13.10
    '''
    gguf-parser --in-max-ctx-size --skip-tokenizer --skip-architecture --skip-metadata \
    --image-vae-tiling --cache-expiration 168h0m0s --no-mmap \
    --ctx-size 32768 --cache-path /opt/models/cache/gguf-parser --gpu-layers-step 1 \
    --tensor-split 23540,16368,15360 --rpc=host:50020,host:50020 \
    -hf-repo bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF -hf-file DeepSeek-R1-Distill-Qwen-32B-bf16/DeepSeek-R1-Distill-Qwen-32B-bf16-00001-of-00002.gguf \
    --json >> deepseek_r1_distill_qwen_32b_bf16_partial_offload_split_3_3.json
    '''
    return load_model_claim_from_file(
        "deepseek_r1_distill_qwen_32b_bf16_partial_offload_split_3_3.json"
    )


def deepseek_v3_0324_ud_iq1_s_disable_offload():
    # gguf-parser 0.21.0
    """
    gguf-parser --estimate --no-mmap --parallel 4 --ctx-size 8192 --gpu-layers 0 \
    -hf-repo unsloth/DeepSeek-V3-0324-GGUF \
    -hf-file UD-IQ1_S/DeepSeek-V3-0324-UD-IQ1_S-00001-of-00004.gguf \
    --json > deepseek_v3_0324_ud_iq1_s_disable_offload.json
    """
    return load_model_claim_from_file("deepseek_v3_0324_ud_iq1_s_disable_offload.json")


def deepseek_v3_0324_ud_iq1_s_full_offload():
    # gguf-parser 0.21.0
    """
    gguf-parser --estimate --no-mmap --parallel 4 --ctx-size 8192 --gpu-layers 999 \
    -hf-repo unsloth/DeepSeek-V3-0324-GGUF \
    -hf-file UD-IQ1_S/DeepSeek-V3-0324-UD-IQ1_S-00001-of-00004.gguf \
    --json > deepseek_v3_0324_ud_iq1_s_full_offload.json
    """
    return load_model_claim_from_file("deepseek_v3_0324_ud_iq1_s_full_offload.json")


def deepseek_v3_0324_ud_iq1_s_partial_offload():
    # gguf-parser 0.21.0
    """
    gguf-parser --estimate --no-mmap --parallel 4 --ctx-size 8192 --gpu-layers-step 1 \
    -hf-repo unsloth/DeepSeek-V3-0324-GGUF \
    -hf-file UD-IQ1_S/DeepSeek-V3-0324-UD-IQ1_S-00001-of-00004.gguf \
    --json > deepseek_v3_0324_ud_iq1_s_partial_offload.json
    """
    return load_model_claim_from_file("deepseek_v3_0324_ud_iq1_s_partial_offload.json")


def load_model_claim_from_file(file_name) -> GGUFParserOutput:
    dir = os.path.dirname(__file__)
    file_path = os.path.join(dir, file_name)
    with open(file_path, 'r') as file:
        model_estimate_claim = GGUFParserOutput.from_json(file.read())
    return model_estimate_claim
