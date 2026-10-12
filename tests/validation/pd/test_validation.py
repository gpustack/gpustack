"""Every PD deployment passes through the same configuration validation gate."""

import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from gpustack.api.exceptions import BadRequestException
from gpustack.schemas.models import (
    DisaggregationSpec,
    ModelCreate,
    PDModeEnum,
    RoleSpec,
    SourceEnum,
    GatherSpec,
)
from gpustack.schemas.clusters import Cluster, GatherStrategyEnum


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "roles,mode,policy,message",
    [
        ([RoleSpec(name="decode")], PDModeEnum.VLLM_NIXL, "fail", "prefill role"),
        (
            [
                RoleSpec(name="prefill", backend_parameters=["--max-model-len=8192"]),
                RoleSpec(name="decode", backend_parameters=["--max-model-len=4096"]),
            ],
            PDModeEnum.VLLM_NIXL,
            "fail",
            "context lengths",
        ),
        (
            [
                RoleSpec(name="prefill", env={"VLLM_SSM_CONV_STATE_LAYOUT": "SD"}),
                RoleSpec(name="decode", env={"VLLM_SSM_CONV_STATE_LAYOUT": "DS"}),
            ],
            PDModeEnum.VLLM_NIXL,
            "fail",
            "VLLM_SSM_CONV_STATE_LAYOUT",
        ),
        (
            [RoleSpec(name="prefill"), RoleSpec(name="decode")],
            PDModeEnum.VLLM_ASCEND_MOONCAKE,
            "recompute",
            "kv_load_failure_policy",
        ),
    ],
)
async def test_unified_entry_rejects_incompatible_pd_configuration(
    monkeypatch, roles, mode, policy, message
):
    from gpustack.scheduler import calculator
    from gpustack.validation.pd import validate_pd_config

    monkeypatch.setattr(
        calculator,
        "get_pretrained_config_with_workers",
        AsyncMock(return_value={"layer_types": ["linear_attention", "full_attention"]}),
    )
    model = ModelCreate(
        name="pd",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/model",
        backend="vLLM",
        backend_version="0.29.0",
        backend_parameters=["--tensor-parallel-size=1", "--max-model-len=4096"],
        roles=roles,
        disaggregation=DisaggregationSpec(mode=mode, kv_load_failure_policy=policy),
    )

    with pytest.raises(BadRequestException) as error:
        await validate_pd_config(None, model)
    assert error.value.status_code == 400
    assert message in error.value.message


@pytest.mark.asyncio
@pytest.mark.parametrize("reset_backend", [False, True])
async def test_runtime_backend_reset_uses_default_instead_of_stored_backend(
    monkeypatch, reset_backend
):
    from gpustack.schemas.models import Model, ModelUpdate
    from gpustack.validation.pd import runtime

    monkeypatch.setattr(runtime, "cluster_vendors", AsyncMock(return_value={"nvidia"}))
    stored = Model(
        name="pd",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/model",
        backend="SGLang",
        disaggregation=DisaggregationSpec(mode=PDModeEnum.SGLANG_NIXL),
    )
    update = ModelUpdate(
        name="pd",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/model",
        **({"backend": None} if reset_backend else {}),
    )
    if reset_backend:
        with pytest.raises(BadRequestException) as error:
            await runtime.validate_pd_mode_runtime(
                None, update, stored=stored, cluster_id=1
            )
        assert "selected engine is vLLM" in error.value.message
    else:
        await runtime.validate_pd_mode_runtime(
            None, update, stored=stored, cluster_id=1
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "vendors,gather,message",
    [
        ({"ascend"}, None, "cannot run here"),
        (
            {"nvidia"},
            GatherSpec(strategy=GatherStrategyEnum.MUST_GATHER, layer="missing-layer"),
            "not a layer of this cluster",
        ),
    ],
)
async def test_unified_entry_enforces_cluster_constraints(
    monkeypatch, vendors, gather, message
):
    from gpustack.validation.pd import validate_pd_config, runtime

    monkeypatch.setattr(runtime, "cluster_vendors", AsyncMock(return_value=vendors))
    monkeypatch.setattr(
        Cluster,
        "one_by_id",
        AsyncMock(return_value=SimpleNamespace(topology=None)),
    )
    model = ModelCreate(
        name="pd",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/model",
        backend="vLLM",
        backend_version="0.29.0",
        backend_parameters=["--tensor-parallel-size=1", "--max-model-len=4096"],
        roles=[RoleSpec(name="prefill"), RoleSpec(name="decode")],
        disaggregation=DisaggregationSpec(mode=PDModeEnum.VLLM_NIXL),
        gather=gather,
    )
    with pytest.raises(BadRequestException) as error:
        await validate_pd_config(None, model, cluster_id=1)
    assert error.value.status_code == 400
    assert message in error.value.message
