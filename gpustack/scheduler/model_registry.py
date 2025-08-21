# Synced with https://github.com/vllm-project/vllm/blob/v0.10.1.1/vllm/model_executor/models/registry.py
# Update these when the builtin vLLM is updated

_TEXT_GENERATION_MODELS = [
    # [Decoder-only]
    "AquilaModel",
    "AquilaForCausalLM",
    "ArceeForCausalLM",
    "ArcticForCausalLM",
    "MiniMaxForCausalLM",
    "MiniMaxText01ForCausalLM",
    "MiniMaxM1ForCausalLM",
    "BaiChuanForCausalLM",
    "BaichuanForCausalLM",
    "BailingMoeForCausalLM",
    "BambaForCausalLM",
    "BloomForCausalLM",
    "ChatGLMModel",
    "ChatGLMForConditionalGeneration",
    "CohereForCausalLM",
    "Cohere2ForCausalLM",
    "DbrxForCausalLM",
    "DeepseekForCausalLM",
    "DeepseekV2ForCausalLM",
    "DeepseekV3ForCausalLM",
    "Dots1ForCausalLM",
    "Ernie4_5_ForCausalLM",
    "Ernie4_5ForCausalLM",  # Note: New class for "Ernie4_5_ForCausalLM"
    "Ernie4_5_MoeForCausalLM",
    "ExaoneForCausalLM",
    "Exaone4ForCausalLM",
    "FalconForCausalLM",
    "Fairseq2LlamaForCausalLM",
    "GemmaForCausalLM",
    "Gemma2ForCausalLM",
    "Gemma3ForCausalLM",
    "Gemma3nForCausalLM",
    "GlmForCausalLM",
    "Glm4ForCausalLM",
    "Glm4MoeForCausalLM",
    "GptOssForCausalLM",
    "GPT2LMHeadModel",
    "GPTBigCodeForCausalLM",
    "GPTJForCausalLM",
    "GPTNeoXForCausalLM",
    "GraniteForCausalLM",
    "GraniteMoeForCausalLM",
    "GraniteMoeHybridForCausalLM",
    "GraniteMoeSharedForCausalLM",
    "GritLM",
    "Grok1ModelForCausalLM",
    "HunYuanMoEV1ForCausalLM",
    "HunYuanDenseV1ForCausalLM",
    "HCXVisionForCausalLM",
    "InternLMForCausalLM",
    "InternLM2ForCausalLM",
    "InternLM2VEForCausalLM",
    "InternLM3ForCausalLM",
    "JAISLMHeadModel",
    "JambaForCausalLM",
    "LlamaForCausalLM",
    "LLaMAForCausalLM",
    "Llama4ForCausalLM",
    "MambaForCausalLM",
    "FalconMambaForCausalLM",
    "FalconH1ForCausalLM",
    "Mamba2ForCausalLM",
    "MiniCPMForCausalLM",
    "MiniCPM3ForCausalLM",
    "MistralForCausalLM",
    "MixtralForCausalLM",
    "QuantMixtralForCausalLM",
    "MptForCausalLM",
    "MPTForCausalLM",
    "MiMoForCausalLM",
    "NemotronForCausalLM",
    "NemotronHForCausalLM",
    "OlmoForCausalLM",
    "Olmo2ForCausalLM",
    "OlmoeForCausalLM",
    "OPTForCausalLM",
    "OrionForCausalLM",
    "PersimmonForCausalLM",
    "PhiForCausalLM",
    "Phi3ForCausalLM",
    "Phi3SmallForCausalLM",
    "PhiMoEForCausalLM",
    "Phi4FlashForCausalLM",
    "Plamo2ForCausalLM",
    "QWenLMHeadModel",
    "Qwen2ForCausalLM",
    "Qwen2MoeForCausalLM",
    "Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM",
    "RWForCausalLM",
    "Step3TextForCausalLM",
    "StableLMEpochForCausalLM",
    "StableLmForCausalLM",
    "Starcoder2ForCausalLM",
    "SolarForCausalLM",
    "TeleChat2ForCausalLM",
    "TeleFLMForCausalLM",
    "XverseForCausalLM",
    "Zamba2ForCausalLM",
    # [Encoder-decoder]
    "BartModel",
    "BartForConditionalGeneration",
    "MBartForConditionalGeneration",
]

_EMBEDDING_MODELS = [
    # [Text-only]
    "BertModel",
    "DeciLMForCausalLM",
    "Gemma2Model",
    # "GlmForCausalLM", # Registered in _TEXT_GENERATION_MODELS
    "GPT2ForSequenceClassification",
    # "GritLM",
    "GteModel",
    "GteNewModel",
    "InternLM2ForRewardModel",
    "JambaForSequenceClassification",
    "LlamaModel",
    # "AquilaModel", # Registered in _TEXT_GENERATION_MODELS
    # "AquilaForCausalLM", # Registered in _TEXT_GENERATION_MODELS
    # "InternLMForCausalLM", # Registered in _TEXT_GENERATION_MODELS
    # "InternLM3ForCausalLM", # Registered in _TEXT_GENERATION_MODELS
    # "LlamaForCausalLM", # Registered in _TEXT_GENERATION_MODELS
    # "LLaMAForCausalLM", # Registered in _TEXT_GENERATION_MODELS
    # "MistralForCausalLM", # Registered in _TEXT_GENERATION_MODELS
    # "XverseForCausalLM", # Registered in _TEXT_GENERATION_MODELS
    "MistralModel",
    "ModernBertModel",
    "NomicBertModel",
    # "Phi3ForCausalLM", # Registered in _TEXT_GENERATION_MODELS
    "Qwen2Model",
    # "Qwen2ForCausalLM", # Registered in _TEXT_GENERATION_MODELS
    "Qwen2ForRewardModel",
    "Qwen2ForProcessRewardModel",
    "RobertaForMaskedLM",
    "RobertaModel",
    # "TeleChat2ForCausalLM", # Registered in _TEXT_GENERATION_MODELS
    "XLMRobertaModel",
    # [Multimodal]
    # "LlavaNextForConditionalGeneration", # Registered in _TEXT_GENERATION_MODELS
    # "Phi3VForCausalLM",
    # "Qwen2VLForConditionalGeneration", # Registered in _TEXT_GENERATION_MODELS
    "PrithviGeoSpatialMAE",
]

_CROSS_ENCODER_MODELS = [
    "BertForSequenceClassification",
    "RobertaForSequenceClassification",
    "XLMRobertaForSequenceClassification",
    "ModernBertForSequenceClassification",
    # [Auto-converted]
    "JinaVLForRanking",
]

_MULTIMODAL_MODELS = [
    # [Decoder-only]
    "AriaForConditionalGeneration",
    "AyaVisionForConditionalGeneration",
    "Blip2ForConditionalGeneration",
    "ChameleonForConditionalGeneration",
    "Cohere2VisionForConditionalGeneration",
    "DeepseekVLV2ForCausalLM",
    "FuyuForCausalLM",
    "Gemma3ForConditionalGeneration",
    "Gemma3nForConditionalGeneration",
    "GLM4VForCausalLM",
    "Glm4vForConditionalGeneration",
    "Glm4v_moeForConditionalGeneration",
    "Glm4vMoeForConditionalGeneration",  # Note: New class for "Glm4v_moeForConditionalGeneration"
    "GraniteSpeechForConditionalGeneration",
    "H2OVLChatModel",
    "InternVLChatModel",
    "InternS1ForConditionalGeneration",
    "Idefics3ForConditionalGeneration",
    "SmolVLMForConditionalGeneration",
    "KeyeForConditionalGeneration",
    "KimiVLForConditionalGeneration",
    "Llama_Nemotron_Nano_VL",
    "LlavaForConditionalGeneration",
    "LlavaNextForConditionalGeneration",
    "LlavaNextVideoForConditionalGeneration",
    "LlavaOnevisionForConditionalGeneration",
    "MantisForConditionalGeneration",
    "MiniMaxVL01ForConditionalGeneration",
    "MiniCPMO",
    "MiniCPMV",
    "Mistral3ForConditionalGeneration",
    "MolmoForCausalLM",
    "NVLM_D",
    "Ovis",
    "PaliGemmaForConditionalGeneration",
    "Phi3VForCausalLM",
    "Phi4MMForCausalLM",
    "Phi4MultimodalForCausalLM",
    "PixtralForConditionalGeneration",
    "QwenVLForConditionalGeneration",
    "Qwen2VLForConditionalGeneration",
    "Qwen2_5_VLForConditionalGeneration",
    "Qwen2AudioForConditionalGeneration",
    "Qwen2_5OmniModel",
    "Qwen2_5OmniForConditionalGeneration",
    "UltravoxModel",
    "Step3VLForConditionalGeneration",
    "TarsierForConditionalGeneration",
    "Tarsier2ForConditionalGeneration",
    "VoxtralForConditionalGeneration",
    # [Encoder-decoder]
    "Florence2ForConditionalGeneration",
    "MllamaForConditionalGeneration",
    "Llama4ForConditionalGeneration",
    "SkyworkR1VChatModel",
    "WhisperForConditionalGeneration",
]

_TRANSFORMERS_SUPPORTED_MODELS = [
    # Text generation models
    "SmolLM3ForCausalLM",
    # Multimodal models
    "Emu3ForConditionalGeneration",
]

_TRANSFORMERS_BACKEND_MODELS = [
    "TransformersModel",
    "TransformersForCausalLM",
    "TransformersForMultimodalLM",
]

vllm_supported_embedding_architectures = _EMBEDDING_MODELS

vllm_supported_reranker_architectures = _CROSS_ENCODER_MODELS

vllm_supported_llm_architectures = (
    _TEXT_GENERATION_MODELS
    + _MULTIMODAL_MODELS
    + _TRANSFORMERS_SUPPORTED_MODELS
    + _TRANSFORMERS_BACKEND_MODELS
)
