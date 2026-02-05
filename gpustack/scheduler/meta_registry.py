from typing import Dict, Optional
from transformers import PretrainedConfig

# Languages supported by Voxtral models
# https://github.com/vllm-project/vllm/blob/db6f71d4c9efc4679b05311c9a8fcc594b187c06/vllm/model_executor/models/voxtral.py#L69
VOXTRAL_SUPPORTED_LANGS = {
    "en": "English",
    "ar": "Arabic",
    "nl": "Dutch",
    "fr": "French",
    "de": "German",
    "hi": "Hindi",
    "it": "Italian",
    "pt": "Portuguese",
    "es": "Spanish",
}

# Languages supported by Granite-Speech models
# https://github.com/vllm-project/vllm/blob/6abb0454adb531de0b081bbf65ccf907e4bd560d/vllm/model_executor/models/granite_speech.py#L80C1-L86C2
GRANITE_SUPPORTED_LANGS = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "es": "Spanish",
}

# Languages supported by Whisper, GLMASR, Qwen3ASR
# https://github.com/vllm-project/vllm/blob/6abb0454adb531de0b081bbf65ccf907e4bd560d/vllm/model_executor/models/whisper_utils.py#L6
ISO639_1_SUPPORTED_LANGS = {
    "en": "English",
    "zh": "Chinese",
    "af": "Afrikaans",
    "ar": "Arabic",
    "hy": "Armenian",
    "az": "Azerbaijani",
    "be": "Belarusian",
    "bs": "Bosnian",
    "bg": "Bulgarian",
    "ca": "Catalan",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "gl": "Galician",
    "de": "German",
    "el": "Greek",
    "he": "Hebrew",
    "hi": "Hindi",
    "hu": "Hungarian",
    "is": "Icelandic",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "kn": "Kannada",
    "kk": "Kazakh",
    "ko": "Korean",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "mk": "Macedonian",
    "ms": "Malay",
    "mr": "Marathi",
    "mi": "Maori",
    "ne": "Nepali",
    "no": "Norwegian",
    "fa": "Persian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sr": "Serbian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "es": "Spanish",
    "sw": "Swahili",
    "sv": "Swedish",
    "tl": "Tagalog",
    "ta": "Tamil",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "cy": "Welsh",
}

# Languages supported by Qwen3-TTS
# https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice/blob/main/config.json#L111
QWEN3_TTS_SUPPORTED_LANGS = {
    "auto": "Auto",
    "zh": "Chinese",
    "en": "English",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "es": "Spanish",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "ru": "Russian",
}


# Voices supported by Qwen3-TTS
# https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice/blob/main/config.json#L129
QWEN3_TTS_SUPPORTED_VOICES = [
    "Vivian",
    "Serena",
    "Uncle_Fu",
    "Dylan",
    "Eric",
    "Ryan",
    "Aiden",
    "Ono_Anna",
    "Sohee",
]


def get_model_meta(pretrained_config: PretrainedConfig) -> Optional[Dict[str, any]]:
    """
    Get model meta information based on the model architectures.
    """
    if not pretrained_config:
        return None

    architectures = getattr(pretrained_config, "architectures", []) or []
    if not architectures:
        return None

    model_meta: dict[str, any] = {}

    arch_set = set(architectures)
    if "VoxtralForConditionalGeneration" in arch_set:
        model_meta["languages"] = list(VOXTRAL_SUPPORTED_LANGS.keys())
    elif "GraniteSpeechForConditionalGeneration" in arch_set:
        model_meta["languages"] = list(GRANITE_SUPPORTED_LANGS.keys())
    elif any(
        arch
        in {
            "WhisperForConditionalGeneration",
            "GlmAsrForConditionalGeneration",
            "Qwen3ASRForConditionalGeneration",
        }
        for arch in arch_set
    ):
        model_meta["languages"] = list(ISO639_1_SUPPORTED_LANGS.keys())
    elif "Qwen3TTSForConditionalGeneration" in arch_set:
        model_meta["languages"] = list(
            QWEN3_TTS_SUPPORTED_LANGS.values()
        )  # Qwen3-TTS uses full language names
        model_meta["voices"] = QWEN3_TTS_SUPPORTED_VOICES
        tts_model_type = getattr(pretrained_config, "tts_model_type", "") or ""
        if tts_model_type:
            # Options: CustomVoice, VoiceDesign, Base. Convert snake_case to CamelCase. e.g., custom_voice -> CustomVoice.
            model_meta["task_type"] = "".join(
                word.capitalize() for word in tts_model_type.split("_")
            )

    return model_meta
