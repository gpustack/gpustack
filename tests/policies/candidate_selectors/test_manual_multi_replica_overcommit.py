from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gpustack.policies.candidate_selectors import VLLMResourceFitSelector
from gpustack.policies.resource_view import ResourceView
from gpustack.schemas.models import BackendEnum, GPUSelector, Model
from tests.fixtures.workers.fixtures import linux_nvidia_1_4090_24gx1

GIB = 1024**3

# A pinned 4090 reserves 90% of its card, roughly 21.8 GiB, so a claim above
# that is the per-GPU constraint the selector compares every candidate
# against. The stub config keeps the attention-shape estimate independent of a
# hub download; GPUSTACK_MODEL_VRAM_CLAIM short-circuits the weight-size lookup.
PRETRAINED_CONFIG = SimpleNamespace(
    architectures=["Qwen2ForCausalLM"],
    num_hidden_layers=2,
    hidden_size=1024,
    vocab_size=32768,
    num_attention_heads=8,
    num_key_value_heads=2,
    max_position_embeddings=2048,
    torch_dtype="bfloat16",
)

OVERCOMMITTING_VRAM_CLAIM = 40 * GIB
FITTING_VRAM_CLAIM = 10 * GIB


def deployment(replicas, vram_claim):
    return Model(
        name="manual-multi-replica",
        source="huggingface",
        huggingface_repo_id="test/manual-multi-replica",
        backend=BackendEnum.VLLM,
        categories=["llm"],
        replicas=replicas,
        env={"GPUSTACK_MODEL_VRAM_CLAIM": str(vram_claim)},
        gpu_selector=GPUSelector(gpu_ids=["host4090:cuda:0"], gpus_per_replica=1),
    )


async def select_candidates(config, replicas, vram_claim, workers=None):
    if workers is None:
        workers = [linux_nvidia_1_4090_24gx1()]
    selector = VLLMResourceFitSelector(
        config,
        deployment(replicas, vram_claim),
        [],
        resource_view=ResourceView([]),
    )
    with patch(
        "gpustack.policies.candidate_selectors.base_candidate_selector.get_pretrained_config_with_workers",
        new=AsyncMock(return_value=PRETRAINED_CONFIG),
    ):
        candidates = await selector.select_candidates(workers)
    return selector, candidates


@pytest.mark.asyncio
async def test_multi_replica_manual_selection_is_admitted_on_overcommit(config):
    selector, candidates = await select_candidates(
        config, replicas=2, vram_claim=OVERCOMMITTING_VRAM_CLAIM
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.overcommit is True
    assert candidate.gpu_indexes == [0]

    messages = "\n".join(selector.get_messages())
    assert "Manual GPU selection resulted in resource overcommit" in messages
    assert "may still fail to start" in messages

    # Every rendered line carries exactly one list bullet. The overcommit text
    # arrives already bullet-prefixed, so nesting it in another list builder
    # would render the nested bullets as "- - ".
    lines = messages.splitlines()
    assert not any(line.startswith("- -") for line in lines)
    admission_lines = [line for line in lines if "may still fail to start" in line]
    assert len(admission_lines) == 1
    assert admission_lines[0].startswith("- The deployment is admitted anyway")


@pytest.mark.asyncio
async def test_single_replica_manual_selection_is_unchanged_on_overcommit(config):
    _, candidates = await select_candidates(
        config, replicas=1, vram_claim=OVERCOMMITTING_VRAM_CLAIM
    )

    assert len(candidates) == 1
    assert candidates[0].overcommit is True
    assert candidates[0].gpu_indexes == [0]


@pytest.mark.asyncio
async def test_multi_replica_manual_selection_admits_a_candidate_that_fits(config):
    _, candidates = await select_candidates(
        config, replicas=2, vram_claim=FITTING_VRAM_CLAIM
    )

    assert len(candidates) == 1
    assert not candidates[0].overcommit
    assert candidates[0].gpu_indexes == [0]


@pytest.mark.asyncio
async def test_multi_replica_manual_selection_without_candidates_stays_refused(config):
    _, candidates = await select_candidates(
        config,
        replicas=2,
        vram_claim=OVERCOMMITTING_VRAM_CLAIM,
        workers=[],
    )

    assert candidates == []
