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


@pytest.mark.parametrize(
    'mode,lb,allowed,message',
    [
        ('vllm-mooncake', 'external', True, ''),
        ('vllm-nixl', 'multi-port-external', True, ''),
        ('vllm-mooncake', 'multi-port-external', False, 'bootstrap'),
        ('vllm-ascend-mooncake', 'multi-port-external', False, 'Ascend'),
        ('custom', 'multi-port-external', False, 'not supported'),
    ],
)
def test_managed_pd_lb_admission_is_specific_to_the_transport(
    mode, lb, allowed, message
):
    from gpustack.validation.pd.roles import validate_external_pd_membership

    parameters = [f'--data-parallel-{lb}-lb', '--dp=4', '--dpl=2']
    if allowed:
        validate_external_pd_membership(mode, 'vLLM', parameters)
    else:
        with pytest.raises(BadRequestException) as error:
            validate_external_pd_membership(mode, 'vLLM', parameters)
        assert message in error.value.message
        if mode == 'vllm-mooncake':
            assert '51531' in error.value.message


@pytest.mark.parametrize(
    'changes',
    [
        dict(description='updated'),
        dict(backend_parameters=['--data-parallel-external-lb', '--dp=4', '--tp=1']),
    ],
)
def test_static_external_mooncake_pd_accepts_new_and_updated_deployments(changes):
    from gpustack.schemas.models import Model, ModelUpdate
    from gpustack.validation.pd.roles import validate_roles

    values = dict(
        name='pd',
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id='org/model',
        backend='vLLM',
        backend_parameters=['--data-parallel-external-lb', '--dp=2', '--tp=1'],
        roles=[RoleSpec(name=n, replicas=1) for n in ['prefill', 'decode', 'router']],
        disaggregation=DisaggregationSpec(mode='vllm-mooncake'),
    )
    validate_roles(ModelCreate(**values))
    validate_roles(
        ModelUpdate(
            name='pd',
            source=SourceEnum.HUGGING_FACE,
            huggingface_repo_id='org/model',
            **changes,
        ),
        stored=Model(id=1, **values),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('role_name', [None, 'prefill', 'decode'])
@pytest.mark.parametrize(
    'other', ['--data-parallel-multi-port-external-lb', '--data-parallel-hybrid-lb']
)
async def test_pd_conflicting_lb_flags_are_bad_requests(role_name, other):
    from gpustack.routes.models import validate_model_in

    parameters = ['--data-parallel-external-lb', other]
    model = ModelCreate(
        name='pd',
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id='org/model',
        backend='vLLM',
        backend_parameters=parameters if role_name is None else [],
        roles=[
            RoleSpec(
                name=name,
                backend_parameters=parameters if name == role_name else None,
            )
            for name in ['prefill', 'decode', 'router']
        ],
        disaggregation=DisaggregationSpec(mode=PDModeEnum.VLLM_NIXL),
    )
    with pytest.raises(BadRequestException) as error:
        await validate_model_in(None, model)
    assert error.value.status_code == 400
    assert 'load balancing' in error.value.message


def test_non_vllm_pd_does_not_validate_vllm_lb_flags():
    from gpustack.validation.pd.roles import validate_external_pd_membership

    validate_external_pd_membership(
        'custom', 'SGLang', ['--data-parallel-multi-port-external-lb']
    )
