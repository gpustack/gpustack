"""Architecture classification for the vLLM v0.27.1 sync."""

import pytest

from gpustack.scheduler.model_registry import detect_model_type, is_multimodal_model
from gpustack.schemas.models import CategoryEnum


@pytest.mark.parametrize(
    "architecture, expected_category",
    [
        ("InklingForCausalLM", CategoryEnum.LLM),
        ("InklingForConditionalGeneration", CategoryEnum.LLM),
        ("LongcatFlashNgramForCausalLM", CategoryEnum.LLM),
        ("Qwen3_5ForCausalLM", CategoryEnum.LLM),
        ("Qwen3_5MoeForCausalLM", CategoryEnum.LLM),
        ("Cosmos3EdgeForConditionalGeneration", CategoryEnum.LLM),
        ("KimiK3ForConditionalGeneration", CategoryEnum.LLM),
        ("VaultGemmaForCausalLM", CategoryEnum.LLM),
        ("BertForMaskedLM", CategoryEnum.EMBEDDING),
        ("RobertaForTokenClassification", CategoryEnum.RERANKER),
        ("XLMRobertaForTokenClassification", CategoryEnum.RERANKER),
        ("VibeVoiceAsrForConditionalGeneration", CategoryEnum.SPEECH_TO_TEXT),
    ],
)
def test_detect_model_type(architecture: str, expected_category: CategoryEnum):
    """An UNKNOWN category blocks deployment outright: evaluate_pretrained_config
    raises "Unsupported architecture" for a vLLM model without an explicit
    backend_version."""
    assert detect_model_type([architecture]) == expected_category


def test_multimodal_and_exaone_rename():
    assert is_multimodal_model(["KimiK3ForConditionalGeneration"]) is True
    assert is_multimodal_model(["Cosmos3EdgeForConditionalGeneration"]) is True

    # vLLM renamed ExaoneMoE to ExaoneMoe in v0.27.1; both spellings must work.
    assert detect_model_type(["ExaoneMoEForCausalLM"]) == CategoryEnum.LLM
    assert detect_model_type(["ExaoneMoeForCausalLM"]) == CategoryEnum.LLM

    # Draft models are never deployed standalone, so vLLM's speculative
    # decoding group stays out of the list.
    for architecture in [
        "Gemma4DSparkModel",
        "K3DSparkModel",
        "InklingMTPModel",
        "KimiK3MTPModel",
    ]:
        assert detect_model_type([architecture]) == CategoryEnum.UNKNOWN
