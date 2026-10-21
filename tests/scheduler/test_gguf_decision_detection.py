from unittest.mock import AsyncMock, patch

import pytest

from gpustack.scheduler.calculator import (
    Architecture,
    Estimate,
    GGUFParserOutput,
    ModelResourceClaim,
)
from gpustack.scheduler.scheduler import evaluate_gguf_model
from gpustack.schemas.models import CategoryEnum
from tests.utils.model import new_model


def _claim(model, reranking: bool, decision: bool = False) -> ModelResourceClaim:
    return ModelResourceClaim(
        model=model,
        # The parser's verdict for a modern-bert GGUF: the architecture the
        # decision backbones share with actual cross-encoders. The parser
        # suppresses reranking for decision files, but the ladder stays
        # decision-first so estimates from older parsers or caches cannot
        # misfile a decision model as a reranker.
        resource_claim_estimate=Estimate(
            items=[], architecture="modern-bert", reranking=reranking, decision=decision
        ),
        resource_architecture=Architecture(),
    )


def _model(**kwargs) -> object:
    return new_model(
        1,
        "laya",
        huggingface_repo_id="ggml-org/Laya-GGUF",
        huggingface_filename="Laya-Q8_0.gguf",
        **kwargs,
    )


@pytest.mark.asyncio
async def test_decision_flag_outranks_reranker_guess():
    model = _model(categories=[])
    with patch(
        "gpustack.scheduler.scheduler.calculate_gguf_model_resource_claim",
        AsyncMock(return_value=_claim(model, reranking=True, decision=True)),
    ):
        changed = await evaluate_gguf_model(model)

    assert changed is True
    assert model.categories == [CategoryEnum.DECISION]


@pytest.mark.asyncio
async def test_no_decision_flag_falls_back_to_derived_ladder():
    model = _model(categories=[])
    with patch(
        "gpustack.scheduler.scheduler.calculate_gguf_model_resource_claim",
        AsyncMock(return_value=_claim(model, reranking=True, decision=False)),
    ):
        changed = await evaluate_gguf_model(model)

    assert changed is True
    assert model.categories == [CategoryEnum.RERANKER]


@pytest.mark.asyncio
async def test_decision_flag_reclassifies_stale_ladder_guess():
    # A decision GGUF evaluated under the pre-0.27.0 parser keeps the
    # reranker guess that ladder wrote; the flag's reclassification must
    # reach it on the next evaluation, or the deployment misses the
    # decision probe skip and fails its /v1/chat/completions health check.
    model = _model(categories=[CategoryEnum.RERANKER])
    with patch(
        "gpustack.scheduler.scheduler.calculate_gguf_model_resource_claim",
        AsyncMock(return_value=_claim(model, reranking=True, decision=True)),
    ):
        changed = await evaluate_gguf_model(model)

    assert changed is True
    assert model.categories == [CategoryEnum.DECISION]


@pytest.mark.asyncio
async def test_multi_category_declaration_survives_the_decision_flag():
    # Multiple categories are a declaration (the ladder writes exactly
    # one); the flag does not rewrite them.
    model = _model(categories=[CategoryEnum.LLM, CategoryEnum.EMBEDDING])
    with patch(
        "gpustack.scheduler.scheduler.calculate_gguf_model_resource_claim",
        AsyncMock(return_value=_claim(model, reranking=True, decision=True)),
    ):
        changed = await evaluate_gguf_model(model)

    assert changed is False
    assert model.categories == [CategoryEnum.LLM, CategoryEnum.EMBEDDING]


@pytest.mark.asyncio
async def test_declared_categories_survive_the_estimate_flags():
    model = _model(categories=[CategoryEnum.DECISION])
    with patch(
        "gpustack.scheduler.scheduler.calculate_gguf_model_resource_claim",
        AsyncMock(return_value=_claim(model, reranking=True, decision=True)),
    ):
        changed = await evaluate_gguf_model(model)

    # A declaration is final: neither the estimate's decision flag nor the
    # derived ladder may rewrite it.
    assert changed is False
    assert model.categories == [CategoryEnum.DECISION]


def test_parser_output_exposes_decision_flag():
    # The JSON gguf-parser v0.27.0 emits for a decision GGUF under the
    # scheduler's flag set (``--skip-tokenizer --skip-metadata --json``):
    # a presence-based ``decision`` on the estimate, independent of the
    # suppressed reranking heuristic.
    output = GGUFParserOutput.from_json(
        """
        {
          "architecture": {"type": "model", "architecture": "modern-bert"},
          "estimate": {
            "items": [],
            "type": "model",
            "architecture": "modern-bert",
            "contextSize": 8192,
            "embeddingOnly": true,
            "reranking": false,
            "decision": true,
            "distributable": true
          }
        }
        """
    )

    assert output.estimate.decision is True
    assert output.estimate.reranking is False


def test_parser_output_defaults_decision_to_false():
    # Estimates from parser versions before v0.27.0, and cached ones, carry
    # no ``decision`` key; the claim must read as a plain non-decision model.
    output = GGUFParserOutput.from_json(
        """
        {
          "architecture": {"type": "model", "architecture": "modern-bert"},
          "estimate": {
            "items": [],
            "type": "model",
            "architecture": "modern-bert",
            "reranking": true
          }
        }
        """
    )

    assert output.estimate.decision is False
