"""Architecture classification for supported inference models."""

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


@pytest.mark.parametrize(
    "architecture,category,multimodal",
    [
        ("BailingMoeV3ForCausalLM", CategoryEnum.LLM, False),
        ("GraniteMoeSWAForCausalLM", CategoryEnum.LLM, False),
        ("GraniteSWAForCausalLM", CategoryEnum.LLM, False),
        ("HYV4ForCausalLM", CategoryEnum.LLM, False),
        ("MuseGlimmerForCausalLM", CategoryEnum.LLM, False),
        ("Qwen4ExpForCausalLM", CategoryEnum.LLM, False),
        ("DeepseekV3BidirectionalModel", CategoryEnum.EMBEDDING, False),
        ("Dots3NoteForCausalLM", CategoryEnum.LLM, True),
        ("InternS2MobiusForConditionalGeneration", CategoryEnum.LLM, True),
        ("MuseGlimmerForConditionalGeneration", CategoryEnum.LLM, True),
        ("NemotronH_Omni_Reasoning_V3", CategoryEnum.LLM, True),
        ("Qwen4ExpForConditionalGeneration", CategoryEnum.LLM, True),
    ],
)
def test_vllm_029_model_categories(architecture, category, multimodal):
    assert detect_model_type([architecture]) == category
    assert is_multimodal_model([architecture]) is multimodal


@pytest.mark.parametrize(
    "architecture,multimodal",
    [
        ("DeepseekV41ForCausalLM", True),
        ("Glm5NextForCausalLM", False),
        ("Glm5NextForConditionalGeneration", True),
        ("Qwen3_5ForConditionalGeneration", True),
    ],
)
def test_catalog_model_classification(architecture, multimodal):
    architectures = [architecture]
    assert detect_model_type(architectures) == CategoryEnum.LLM
    assert is_multimodal_model(architectures) is multimodal


@pytest.mark.parametrize(
    "architecture",
    [
        "BailingMoeV3MTPModel",
        "DFlash2DraftModel",
        "DFlashMuseGlimmerAssistantModel",
        "Dots3NoteMTPModel",
        "Glm5NextMTPModel",
        "HYV4MTPModel",
        "InternS2MobiusMTP",
        "MuseGlimmerAssistantModel",
        "Qwen4ExpMTP",
    ],
)
def test_draft_architectures_are_not_standalone_models(architecture):
    assert detect_model_type([architecture]) == CategoryEnum.UNKNOWN


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
