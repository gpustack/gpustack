"""Model metadata settles conditional NIXL admission factors before placement."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gpustack.scheduler import scheduler
from gpustack.schemas.models import (
    DisaggregationSpec,
    Model,
    PDModeEnum,
    RoleSpec,
    SourceEnum,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "variable,prefill,decode,config,message",
    [
        ("VLLM_MLA_DISABLE", "0", "1", {"kv_lora_rank": 512}, "VLLM_MLA_DISABLE"),
        (
            "VLLM_SSM_CONV_STATE_LAYOUT",
            "SD",
            "SD",
            {"layer_types": ["linear_attention", "full_attention"]},
            "VLLM_SSM_CONV_STATE_LAYOUT",
        ),
        (
            "VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE",
            "0",
            "1",
            {"attention_chunk_size": 8192},
            "hybrid KV cache manager",
        ),
    ],
)
async def test_metadata_rejects_incompatible_role_environments(
    monkeypatch, variable, prefill, decode, config, message
):
    pretrained = SimpleNamespace(architectures=["Qwen3ForCausalLM"], **config)
    monkeypatch.setattr(
        scheduler, "evaluate_diffusion_model", AsyncMock(return_value=False)
    )
    monkeypatch.setattr(
        scheduler,
        "get_pretrained_config_with_workers",
        AsyncMock(return_value=pretrained),
    )
    model = Model(
        name="pairing",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/model",
        backend="vLLM",
        backend_version="0.29.0",
        disaggregation=DisaggregationSpec(mode=PDModeEnum.VLLM_NIXL),
        roles=[
            RoleSpec(name="prefill", env={variable: prefill}),
            RoleSpec(name="decode", env={variable: decode}),
        ],
    )

    with pytest.raises(ValueError, match=message):
        await scheduler.evaluate_pretrained_config(model)


@pytest.mark.asyncio
@pytest.mark.parametrize("pairing_failure", [True, False])
async def test_evaluation_blocks_incompatible_pairs_but_preserves_metadata_fallback(
    monkeypatch, pairing_failure
):
    from datetime import datetime, timezone
    from unittest.mock import MagicMock
    from gpustack.schemas.models import ModelInstanceStateEnum

    now = datetime.now(timezone.utc)
    instance = SimpleNamespace(
        id=1,
        model_id=1,
        name="pd-prefill",
        worker_id=None,
        state=ModelInstanceStateEnum.PENDING,
        created_at=now,
        updated_at=now,
    )
    model = Model(
        name="pd",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/model",
        backend="vLLM",
        backend_version="0.29.0",
        disaggregation=DisaggregationSpec(mode=PDModeEnum.VLLM_NIXL),
        roles=[
            RoleSpec(name="prefill", env={"VLLM_MLA_DISABLE": "0"}),
            RoleSpec(name="decode", env={"VLLM_MLA_DISABLE": "1"}),
        ],
    )
    context = MagicMock()
    monkeypatch.setattr(scheduler, "async_session", lambda: context)
    monkeypatch.setattr(
        scheduler.ModelInstance, "one_by_id", AsyncMock(return_value=instance)
    )
    monkeypatch.setattr(scheduler.Model, "one_by_id", AsyncMock(return_value=model))
    monkeypatch.setattr(scheduler.Worker, "all", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        scheduler, "prioritize_workers_with_model_files", AsyncMock(return_value=[])
    )
    updates = AsyncMock()
    monkeypatch.setattr(
        scheduler,
        "ModelInstanceService",
        lambda session: SimpleNamespace(update=updates),
    )
    monkeypatch.setattr(
        scheduler, "ModelService", lambda session: SimpleNamespace(update=AsyncMock())
    )
    monkeypatch.setattr(
        scheduler, "evaluate_diffusion_model", AsyncMock(return_value=False)
    )
    loader = AsyncMock(
        return_value=SimpleNamespace(
            architectures=["Qwen3ForCausalLM"], kv_lora_rank=512
        )
    )
    if not pairing_failure:
        loader.side_effect = ValueError("metadata unavailable")
    monkeypatch.setattr(scheduler, "get_pretrained_config_with_workers", loader)
    runner = scheduler.Scheduler.__new__(scheduler.Scheduler)
    runner._queue = SimpleNamespace(put=AsyncMock())
    await runner._evaluate(instance)
    if pairing_failure:
        runner._queue.put.assert_not_awaited()
        assert instance.state == ModelInstanceStateEnum.ERROR
        assert "VLLM_MLA_DISABLE" in instance.state_message
    else:
        runner._queue.put.assert_awaited_once_with(instance)
        assert instance.state == ModelInstanceStateEnum.ANALYZING
