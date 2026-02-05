# Synced with https://github.com/vllm-project/vllm/blob/v0.15.0/vllm/model_executor/models/registry.py
# Update these when the builtin vLLM is updated
# List of supported model architectures for the default version of the vLLM backend
# TODO version-aware support list
from typing import List

from gpustack.schemas.models import CategoryEnum

_TEXT_GENERATION_MODELS = [
    # [Decoder-only]
    "ApertusForCausalLM",
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
    "BailingMoeV2ForCausalLM",
    "BambaForCausalLM",
    "BloomForCausalLM",
    "ChatGLMModel",
    "ChatGLMForConditionalGeneration",
    "CohereForCausalLM",
    "Cohere2ForCausalLM",
    "CwmForCausalLM",
    "DbrxForCausalLM",
    "DeepseekForCausalLM",
    "DeepseekV2ForCausalLM",
    "DeepseekV3ForCausalLM",
    "DeepseekV32ForCausalLM",
    "Dots1ForCausalLM",
    "Ernie4_5_ForCausalLM",
    "Ernie4_5ForCausalLM",  # Note: New class for "Ernie4_5_ForCausalLM"
    "Ernie4_5_MoeForCausalLM",
    "ExaoneForCausalLM",
    "Exaone4ForCausalLM",
    "ExaoneMoEForCausalLM",
    "Fairseq2LlamaForCausalLM",
    "FalconForCausalLM",
    "FalconMambaForCausalLM",
    "FalconH1ForCausalLM",
    "FlexOlmoForCausalLM",
    "GemmaForCausalLM",
    "Gemma2ForCausalLM",
    "Gemma3ForCausalLM",
    "Gemma3nForCausalLM",
    "Qwen3NextForCausalLM",
    "GlmForCausalLM",
    "Glm4ForCausalLM",
    "Glm4MoeForCausalLM",
    "Glm4MoeLiteForCausalLM",
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
    "Grok1ForCausalLM",
    "HunYuanMoEV1ForCausalLM",
    "HunYuanDenseV1ForCausalLM",
    "HCXVisionForCausalLM",
    "InternLMForCausalLM",
    "InternLM2ForCausalLM",
    "InternLM2VEForCausalLM",
    "InternLM3ForCausalLM",
    "IQuestCoderForCausalLM",
    "IQuestLoopCoderForCausalLM",
    "JAISLMHeadModel",
    "JambaForCausalLM",
    "KimiLinearForCausalLM",
    "Lfm2ForCausalLM",
    "Lfm2MoeForCausalLM",
    "LlamaForCausalLM",
    "LLaMAForCausalLM",
    "Llama4ForCausalLM",
    "LongcatFlashForCausalLM",
    "MambaForCausalLM",
    "Mamba2ForCausalLM",
    "MiniCPMForCausalLM",
    "MiniCPM3ForCausalLM",
    "MiniMaxForCausalLM",
    "MiniMaxText01ForCausalLM",
    "MiniMaxM1ForCausalLM",
    "MiniMaxM2ForCausalLM",
    "MistralForCausalLM",
    "MixtralForCausalLM",
    "MotifForCausalLM",
    "QuantMixtralForCausalLM",
    "MptForCausalLM",
    "MPTForCausalLM",
    "MiMoForCausalLM",
    "MiMoV2FlashForCausalLM",
    "NemotronForCausalLM",
    "NemotronHForCausalLM",
    "OlmoForCausalLM",
    "Olmo2ForCausalLM",
    "Olmo3ForCausalLM",
    "OlmoeForCausalLM",
    "OPTForCausalLM",
    "OrionForCausalLM",
    "OuroForCausalLM",
    "PanguEmbeddedForCausalLM",
    "PanguProMoEV2ForCausalLM",
    "PanguUltraMoEForCausalLM",
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
    "SeedOssForCausalLM",
    "Step1ForCausalLM",
    "Step3TextForCausalLM",
    "Step3p5ForCausalLM",
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
    "BertSpladeSparseEmbeddingModel",
    "DeciLMForCausalLM",
    "Gemma2Model",
    "Gemma3TextModel",
    # "GlmForCausalLM", # Registered in _TEXT_GENERATION_MODELS
    "GPT2ForSequenceClassification",
    # "GritLM",
    "GteModel",
    "GteNewModel",
    "InternLM2ForRewardModel",
    "JambaForSequenceClassification",
    "LlamaBidirectionalModel",
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
    "BgeM3EmbeddingModel",
    # [Multimodal]
    # "LlavaNextForConditionalGeneration", # Registered in _TEXT_GENERATION_MODELS
    # "Phi3VForCausalLM",
    # "Qwen2VLForConditionalGeneration", # Registered in _TEXT_GENERATION_MODELS
    "CLIPModel",
    "PrithviGeoSpatialMAE",
    "Terratorch",
]

_CROSS_ENCODER_MODELS = [
    "BertForSequenceClassification",
    "BertForTokenClassification",
    "GteNewForSequenceClassification",
    "RobertaForSequenceClassification",
    "XLMRobertaForSequenceClassification",
    "ModernBertForSequenceClassification",
    "ModernBertForTokenClassification",
    "JinaVLForRanking",
    "LlamaBidirectionalForSequenceClassification",
]

_MULTIMODAL_MODELS = [
    # [Decoder-only]
    "AriaForConditionalGeneration",
    "AyaVisionForConditionalGeneration",
    "BeeForConditionalGeneration",
    "Blip2ForConditionalGeneration",
    "ChameleonForConditionalGeneration",
    "Cohere2VisionForConditionalGeneration",
    "DeepseekVLV2ForCausalLM",
    "DeepseekOCRForCausalLM",
    "DotsOCRForCausalLM",
    "Eagle2_5_VLForConditionalGeneration",
    "Ernie4_5_VLMoeForConditionalGeneration",
    "FuyuForCausalLM",
    "Gemma3ForConditionalGeneration",
    "Gemma3nForConditionalGeneration",
    "GLM4VForCausalLM",
    "Glm4vForConditionalGeneration",
    "Glm4v_moeForConditionalGeneration",
    "Glm4vMoeForConditionalGeneration",  # Note: New class for "Glm4v_moeForConditionalGeneration"
    "H2OVLChatModel",
    "HunYuanVLForConditionalGeneration",
    "StepVLForConditionalGeneration",
    "InternVLChatModel",
    "NemotronH_Nano_VL_V2",
    "InternS1ForConditionalGeneration",
    "InternVLForConditionalGeneration",
    "Idefics3ForConditionalGeneration",
    "IsaacForConditionalGeneration",
    "SmolVLMForConditionalGeneration",
    "KananaVForConditionalGeneration",
    "KeyeForConditionalGeneration",
    "KeyeVL1_5ForConditionalGeneration",
    "RForConditionalGeneration",
    "KimiVLForConditionalGeneration",
    "KimiK25ForConditionalGeneration",
    "LightOnOCRForConditionalGeneration",
    "Lfm2VlForConditionalGeneration",
    "Llama_Nemotron_Nano_VL",
    "Llama4ForConditionalGeneration",
    "LlavaForConditionalGeneration",
    "LlavaNextForConditionalGeneration",
    "LlavaNextVideoForConditionalGeneration",
    "LlavaOnevisionForConditionalGeneration",
    "MantisForConditionalGeneration",
    "MiDashengLMModel",
    "MiniMaxVL01ForConditionalGeneration",
    "MiniCPMO",
    "MiniCPMV",
    "Mistral3ForConditionalGeneration",
    "MolmoForCausalLM",
    "Molmo2ForConditionalGeneration",
    "NVLM_D",
    "Ovis",
    "Ovis2_5",
    "PaddleOCRVLForConditionalGeneration",
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
    "Qwen3OmniMoeForConditionalGeneration",
    "Qwen3VLForConditionalGeneration",
    "Qwen3VLMoeForConditionalGeneration",
    "UltravoxModel",
    "SkyworkR1VChatModel",
    "Step3VLForConditionalGeneration",
    "TarsierForConditionalGeneration",
    "Tarsier2ForConditionalGeneration",
    "VoxtralStreamingGeneration",
    # [Encoder-decoder]
    "Florence2ForConditionalGeneration",
    "MllamaForConditionalGeneration",
    "Llama4ForConditionalGeneration",
    "SkyworkR1VChatModel",
    "NemotronParseForConditionalGeneration",
]

_SPEECH_TO_TEXT_MODELS = [
    "GlmAsrForConditionalGeneration",
    "GraniteSpeechForConditionalGeneration",
    "VoxtralForConditionalGeneration",
    "WhisperForConditionalGeneration",
]

_TEXT_TO_SPEECH_MODELS = [
    "Qwen3TTSForConditionalGeneration",
]

_TRANSFORMERS_SUPPORTED_MODELS = [
    # Text generation models
    "SmolLM3ForCausalLM",
    # Multimodal models
    "Emu3ForConditionalGeneration",
]


_TRANSFORMERS_BACKEND_TEXT_GENERATION_MODELS = [
    "TransformersModel",
    "TransformersForCausalLM",
    "TransformersMoEForCausalLM",
]

_TRANSFORMERS_BACKEND_MULTIMODAL_MODELS = [
    "TransformersForMultimodalLM",
    "TransformersMultiModalForCausalLM",
    "TransformersMultiModalMoEForCausalLM",
]

_TRANSFORMERS_BACKEND_EMBEDDING_MODELS = [
    "TransformersEmbeddingModel",
    "TransformersMoEEmbeddingModel",
    "TransformersMultiModalEmbeddingModel",
]

_TRANSFORMERS_BACKEND_CROSS_ENCODER_MODELS = [
    "TransformersForSequenceClassification",
    "TransformersMoEForSequenceClassification",
    "TransformersMultiModalForSequenceClassification",
]

_LLM_MODELS = (
    _TEXT_GENERATION_MODELS
    + _MULTIMODAL_MODELS
    + _TRANSFORMERS_SUPPORTED_MODELS
    + _TRANSFORMERS_BACKEND_TEXT_GENERATION_MODELS
    + _TRANSFORMERS_BACKEND_MULTIMODAL_MODELS
)

_EMBEDDING_MODELS = _EMBEDDING_MODELS + _TRANSFORMERS_BACKEND_EMBEDDING_MODELS

_RERANKER_MODELS = _CROSS_ENCODER_MODELS + _TRANSFORMERS_BACKEND_CROSS_ENCODER_MODELS


def detect_model_type(architectures: List[str]) -> CategoryEnum:
    """
    Detect the model type based on the architectures.

    Args:
        architectures: List of model architecture names.

    Returns:
        The detected model category.
    """
    for architecture in architectures or []:
        if architecture in _EMBEDDING_MODELS:
            return CategoryEnum.EMBEDDING
        if architecture in _RERANKER_MODELS:
            return CategoryEnum.RERANKER
        if architecture in _SPEECH_TO_TEXT_MODELS:
            return CategoryEnum.SPEECH_TO_TEXT
        if architecture in _TEXT_TO_SPEECH_MODELS:
            return CategoryEnum.TEXT_TO_SPEECH
        if architecture in _LLM_MODELS:
            return CategoryEnum.LLM
    return CategoryEnum.UNKNOWN


def is_multimodal_model(architectures: List[str]) -> bool:
    """
    Check if the model is a multimodal model based on the architectures.

    Args:
        architectures: List of model architecture names.

    Returns:
        True if the model is multimodal, False otherwise.
    """
    for architecture in architectures or []:
        if architecture in _MULTIMODAL_MODELS:
            return True
    return False
