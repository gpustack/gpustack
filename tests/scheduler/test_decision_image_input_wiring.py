"""The decision-image-input meta sync runs on every evaluation path.

evaluate_pretrained_config's core has early returns (diffusion
classification, skipped architecture check) and GGUF models take
evaluate_gguf_model instead — the contract must be published or cleared
on all of them, or a reclassified model keeps publishing a stale
decision_image_input key to its route.
"""

from unittest.mock import AsyncMock, patch

import pytest

from gpustack.schemas.models import CategoryEnum
from gpustack.scheduler import scheduler
from tests.scheduler.test_gguf_decision_detection import _claim, _model
from tests.utils.model import new_model


def _pretrained_model(**kwargs):
    return new_model(
        2,
        "d1",
        huggingface_repo_id="LiquidAI/d1-3B",
        **kwargs,
    )


@pytest.mark.asyncio
async def test_early_return_path_still_clears_a_stale_key():
    # The diffusion early return reclassifies the model away from decision:
    # the wrapper must still run the meta sync and drop the stale key.
    model = _pretrained_model(
        categories=[CategoryEnum.IMAGE],
        backend="vLLM",
        meta={"decision_image_input": "images"},
    )
    with patch.object(
        scheduler,
        "_evaluate_pretrained_config_core",
        AsyncMock(return_value=True),
    ):
        changed = await scheduler.evaluate_pretrained_config(model)

    assert changed is True
    assert "decision_image_input" not in (model.meta or {})


@pytest.mark.asyncio
async def test_pretrained_path_publishes_the_key():
    model = _pretrained_model(categories=[CategoryEnum.DECISION], backend="vLLM")
    with patch.object(
        scheduler,
        "_evaluate_pretrained_config_core",
        AsyncMock(return_value=False),
    ):
        changed = await scheduler.evaluate_pretrained_config(model)

    assert changed is True
    assert model.meta["decision_image_input"] == "images"


@pytest.mark.asyncio
async def test_gguf_decision_model_publishes_the_key():
    # A GGUF served by an explicitly selected vLLM backend resolves like any
    # vLLM deployment: the decision GGUF must reach its route with the key.
    model = _model(categories=[], backend="vLLM")
    with patch(
        "gpustack.scheduler.scheduler.calculate_gguf_model_resource_claim",
        AsyncMock(return_value=_claim(model, reranking=True, decision=True)),
    ):
        changed = await scheduler.evaluate_gguf_model(model)

    assert changed is True
    assert model.categories == [CategoryEnum.DECISION]
    assert model.meta["decision_image_input"] == "images"


@pytest.mark.asyncio
async def test_gguf_undeclared_backend_writes_no_key():
    # A GGUF left on the CUSTOM default backend carries no declaration:
    # absence keeps the client's current request construction.
    model = _model(categories=[])
    with patch(
        "gpustack.scheduler.scheduler.calculate_gguf_model_resource_claim",
        AsyncMock(return_value=_claim(model, reranking=False, decision=True)),
    ):
        changed = await scheduler.evaluate_gguf_model(model)

    assert changed is True
    assert model.categories == [CategoryEnum.DECISION]
    assert "decision_image_input" not in (model.meta or {})
