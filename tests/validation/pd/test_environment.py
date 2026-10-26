"""Admission uses each role's effective environment, including inheritance."""

import json
from unittest.mock import AsyncMock

import pytest

from gpustack.api.exceptions import BadRequestException
from gpustack.validation.pd.pairing import validate_role_pairing
from gpustack.routes import models as model_routes
from gpustack.validation.pd import validate_pd_config, placement, runtime
from gpustack.scheduler import calculator
from gpustack.schemas.models import (
    DisaggregationSpec,
    Model,
    ModelCreate,
    ModelUpdate,
    PDModeEnum,
    RoleSpec,
    SourceEnum,
)
from gpustack.server.controllers import _pairing_unverified
from gpustack.validation.pd.parameters import UNDECIDABLE
from gpustack.validation.pd.environment import environment_pairing_factors


def model_input(prefill_env=None, decode_env=None, model_env=None, **kwargs):
    return ModelCreate(
        name="pairing",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/model",
        backend="vLLM",
        backend_version="0.29.0",
        backend_parameters=["--tensor-parallel-size=1", "--max-model-len=4096"],
        env=model_env,
        roles=[
            RoleSpec(name="prefill", env=prefill_env),
            RoleSpec(name="decode", env=decode_env),
            RoleSpec(name="router"),
        ],
        disaggregation=DisaggregationSpec(mode=PDModeEnum.VLLM_NIXL),
    ).model_copy(update=kwargs)


def assert_rejected(model, fragment, **kwargs):
    with pytest.raises(BadRequestException) as error:
        validate_role_pairing(model, pretrained_config=_PLAIN_CONFIG, **kwargs)
    assert error.value.status_code == 400
    assert fragment in error.value.message


def test_different_layout_environments_are_rejected():
    model = model_input(
        {"VLLM_KV_CACHE_LAYOUT": "NHD"}, {"VLLM_KV_CACHE_LAYOUT": "HND"}
    )
    assert_rejected(model, "VLLM_KV_CACHE_LAYOUT")


def test_role_layout_is_compared_with_inherited_model_environment():
    model = model_input(
        decode_env={"VLLM_KV_CACHE_LAYOUT": "HND"},
        model_env={"VLLM_KV_CACHE_LAYOUT": "NHD"},
    )
    assert_rejected(model, "VLLM_KV_CACHE_LAYOUT")


@pytest.mark.parametrize(
    "prefill,decode", [("NHD", "LBNHC"), ("HND", "LBHNC"), ("HND", "HND")]
)
def test_equivalent_layout_names_are_accepted(prefill, decode):
    model = model_input(
        {"VLLM_KV_CACHE_LAYOUT": prefill}, {"VLLM_KV_CACHE_LAYOUT": decode}
    )
    validate_role_pairing(model)
    assert _pairing_unverified(model) is False


@pytest.mark.parametrize("role_env", [{}, {"OMP_NUM_THREADS": "2"}])
def test_role_environment_replaces_the_model_dictionary(role_env):
    model = model_input(decode_env=role_env, model_env={"VLLM_KV_CACHE_LAYOUT": "NHD"})
    validate_role_pairing(model)
    assert _pairing_unverified(model) is True


def test_an_unknown_layout_is_reported_instead_of_guessed():
    model = model_input(
        {"VLLM_KV_CACHE_LAYOUT": "{{worker_ip}}"},
        {"VLLM_KV_CACHE_LAYOUT": "HND"},
    )
    validate_role_pairing(model)
    assert _pairing_unverified(model) is True


def test_an_inherited_layout_is_verified():
    model = model_input(model_env={"VLLM_KV_CACHE_LAYOUT": "HND"})
    validate_role_pairing(model)
    assert _pairing_unverified(model) is False


def test_sparse_model_environment_update_uses_stored_roles():
    original = model_input(decode_env={"VLLM_KV_CACHE_LAYOUT": "HND"})
    stored = Model.model_validate(original.model_dump())
    update = ModelUpdate(
        name=original.name,
        source=original.source,
        huggingface_repo_id=original.huggingface_repo_id,
        env={"VLLM_KV_CACHE_LAYOUT": "NHD"},
    )
    assert_rejected(update, "VLLM_KV_CACHE_LAYOUT", stored=stored)


@pytest.mark.parametrize("mode", [PDModeEnum.CUSTOM, PDModeEnum.SGLANG_NIXL, None])
def test_vllm_environment_rules_are_scoped_to_the_managed_nixl_mode(mode):
    model = model_input(
        {"VLLM_KV_CACHE_LAYOUT": "NHD"},
        {"VLLM_KV_CACHE_LAYOUT": "HND"},
        disaggregation=DisaggregationSpec(mode=mode) if mode else None,
    )
    validate_role_pairing(model)
    assert _pairing_unverified(model) is False


def test_instance_network_and_performance_settings_can_differ():
    model = model_input(
        {
            "VLLM_NIXL_SIDE_CHANNEL_HOST": "192.0.2.1",
            "VLLM_NIXL_SIDE_CHANNEL_PORT": "5601",
            "UCX_NET_DEVICES": "mlx5_0:1",
            "UCX_TLS": "rc,ud",
            "OMP_NUM_THREADS": "2",
            "VLLM_BATCH_INVARIANT": "0",
        },
        {
            "VLLM_NIXL_SIDE_CHANNEL_HOST": "192.0.2.2",
            "VLLM_NIXL_SIDE_CHANNEL_PORT": "5602",
            "UCX_NET_DEVICES": "mlx5_1:1",
            "UCX_TLS": "all",
            "OMP_NUM_THREADS": "4",
            "VLLM_BATCH_INVARIANT": "1",
        },
    )
    validate_role_pairing(model)


@pytest.fixture
def admission(monkeypatch):
    monkeypatch.setattr(runtime, "validate_pd_mode_runtime", AsyncMock())
    monkeypatch.setattr(placement, "validate_gather_layer", AsyncMock())

    async def validate(model, config):
        fetch = AsyncMock(return_value=config)
        monkeypatch.setattr(calculator, "get_pretrained_config_with_workers", fetch)
        await model_routes.validate_model_in(None, model)
        return fetch

    return validate


_MLA_CONFIG = {"model_type": "deepseek_v3", "kv_lora_rank": 512}
_SSM_CONFIG = {"layer_types": ["linear_attention", "full_attention"]}
_PLAIN_CONFIG = {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}
_CHUNKED_CONFIG = {"model_type": "llama4", "attention_chunk_size": 8192}


@pytest.mark.asyncio
async def test_mla_model_cannot_use_different_kv_representations(admission):
    model = model_input({"VLLM_MLA_DISABLE": "0"}, {"VLLM_MLA_DISABLE": "1"})
    with pytest.raises(BadRequestException) as error:
        await admission(model, _MLA_CONFIG)
    assert "VLLM_MLA_DISABLE" in error.value.message


@pytest.mark.asyncio
async def test_plain_model_is_unaffected_by_the_mla_switch(admission):
    model = model_input({"VLLM_MLA_DISABLE": "0"}, {"VLLM_MLA_DISABLE": "1"})
    await admission(model, _PLAIN_CONFIG)


def test_model_dependent_disagreement_is_unverified_without_configuration():
    model = model_input({"VLLM_MLA_DISABLE": "0"}, {"VLLM_MLA_DISABLE": "1"})
    validate_role_pairing(model)
    assert _pairing_unverified(model) is True


@pytest.mark.asyncio
@pytest.mark.parametrize("decode", ["DS", "SD"])
async def test_nixl_ssm_requires_ds_even_when_both_roles_agree(admission, decode):
    model = model_input(
        {"VLLM_SSM_CONV_STATE_LAYOUT": "SD"},
        {"VLLM_SSM_CONV_STATE_LAYOUT": decode},
    )
    with pytest.raises(BadRequestException) as error:
        await admission(model, _SSM_CONFIG)
    assert "VLLM_SSM_CONV_STATE_LAYOUT" in error.value.message
    assert "DS" in error.value.message


@pytest.mark.asyncio
async def test_plain_model_can_ignore_ssm_layout_differences(admission):
    model = model_input(
        {"VLLM_SSM_CONV_STATE_LAYOUT": "SD"},
        {"VLLM_SSM_CONV_STATE_LAYOUT": "DS"},
    )
    await admission(model, _PLAIN_CONFIG)


@pytest.mark.asyncio
async def test_nixl_ssm_accepts_ds_on_both_roles(admission):
    model = model_input(model_env={"VLLM_SSM_CONV_STATE_LAYOUT": "DS"})
    await admission(model, _SSM_CONFIG)


@pytest.mark.asyncio
async def test_chunked_attention_environment_can_make_hma_incompatible(admission):
    name = "VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE"
    model = model_input({name: "0"}, {name: "1"})
    with pytest.raises(BadRequestException) as error:
        await admission(model, _CHUNKED_CONFIG)
    assert "hybrid KV cache manager" in error.value.message


@pytest.mark.asyncio
async def test_explicitly_disabled_hma_allows_different_environment(admission):
    name = "VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE"
    model = model_input(
        {name: "0"},
        {name: "1"},
        backend_parameters=["--disable-hybrid-kv-cache-manager"],
    )
    await admission(model, _CHUNKED_CONFIG)


@pytest.mark.asyncio
async def test_plain_attention_allows_different_hma_environment(admission):
    name = "VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE"
    await admission(model_input({name: "0"}, {name: "1"}), _PLAIN_CONFIG)


def test_explicit_attention_backends_are_compared():
    model = model_input()
    model.roles[0].backend_parameters = model.backend_parameters + [
        "--attention-backend=FLASH_ATTN"
    ]
    model.roles[1].backend_parameters = model.backend_parameters + [
        "--attention_backend",
        "FLASHINFER",
    ]
    assert_rejected(model, "attention backend")


def test_auto_attention_backend_is_unverified_against_an_explicit_backend():
    model = model_input()
    model.roles[0].backend_parameters = model.backend_parameters + [
        "--attention-backend=FLASH_ATTN"
    ]
    validate_role_pairing(model)
    assert _pairing_unverified(model) is True


def test_removed_attention_environment_is_ignored_on_current_vllm():
    model = model_input(
        {"VLLM_ATTENTION_BACKEND": "FLASH_ATTN"},
        {"VLLM_ATTENTION_BACKEND": "FLASHINFER"},
    )
    validate_role_pairing(model)
    assert _pairing_unverified(model) is False


def test_rocm_layout_switch_is_not_applied_to_the_nvidia_recipe():
    model = model_input(
        {"VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT": "0"},
        {"VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT": "1"},
    )
    validate_role_pairing(model)
    assert _pairing_unverified(model) is False


def set_decode_connector(model, **config):
    descriptor = {"kv_connector": "NixlConnector", "kv_role": "kv_consumer", **config}
    model.roles[1].backend_parameters = model.backend_parameters + [
        "--kv-transfer-config",
        json.dumps(descriptor),
    ]


@pytest.mark.asyncio
async def test_supported_layout_permutation_is_allowed(admission):
    model = model_input(
        {"VLLM_KV_CACHE_LAYOUT": "HND"}, {"VLLM_KV_CACHE_LAYOUT": "NHD"}
    )
    set_decode_connector(model, enable_permute_local_kv=True)
    await admission(model, _PLAIN_CONFIG)


def test_permutation_does_not_allow_the_unsupported_direction():
    model = model_input(
        {"VLLM_KV_CACHE_LAYOUT": "NHD"}, {"VLLM_KV_CACHE_LAYOUT": "HND"}
    )
    set_decode_connector(model, enable_permute_local_kv=True)
    assert_rejected(model, "VLLM_KV_CACHE_LAYOUT")


@pytest.mark.asyncio
async def test_permutation_is_not_supported_with_recurrent_cache_groups(admission):
    model = model_input(
        {"VLLM_KV_CACHE_LAYOUT": "HND"}, {"VLLM_KV_CACHE_LAYOUT": "NHD"}
    )
    set_decode_connector(model, enable_permute_local_kv=True)
    with pytest.raises(BadRequestException) as error:
        await admission(model, _SSM_CONFIG)
    assert "VLLM_KV_CACHE_LAYOUT" in error.value.message


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["0.27.1", "0.29.0"])
async def test_modern_nixl_default_hma_matches_explicit_enable(admission, version):
    model = model_input(backend_version=version)
    model.roles[0].backend_parameters = model.backend_parameters + [
        "--no-disable-hybrid-kv-cache-manager"
    ]
    await admission(model, _PLAIN_CONFIG)


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["0.27.1", "0.29.0"])
async def test_modern_nixl_default_hma_conflicts_with_explicit_disable(
    admission, version
):
    model = model_input(backend_version=version)
    model.roles[0].backend_parameters = model.backend_parameters + [
        "--disable-hybrid-kv-cache-manager"
    ]
    with pytest.raises(BadRequestException) as error:
        await admission(model, _PLAIN_CONFIG)
    assert "hybrid KV cache manager" in error.value.message


def test_attention_hash_escape_hatch_is_not_overruled():
    model = model_input()
    model.roles[0].backend_parameters = model.backend_parameters + [
        "--attention-backend=FLASH_ATTN"
    ]
    set_decode_connector(
        model, kv_connector_extra_config={"enforce_handshake_compat": False}
    )
    model.roles[1].backend_parameters.append("--attention-backend=FLASHINFER")
    validate_role_pairing(model)
    assert _pairing_unverified(model) is True


def test_different_batch_invariance_can_leave_backend_selection_unverified():
    model = model_input({"VLLM_BATCH_INVARIANT": "0"}, {"VLLM_BATCH_INVARIANT": "1"})
    validate_role_pairing(model)
    assert _pairing_unverified(model) is True


@pytest.mark.asyncio
async def test_unreadable_model_metadata_preserves_the_unverified_verdict(monkeypatch):
    model = model_input({"VLLM_MLA_DISABLE": "0"}, {"VLLM_MLA_DISABLE": "1"})
    fetch = AsyncMock(side_effect=ValueError("unavailable source"))
    monkeypatch.setattr(calculator, "get_pretrained_config_with_workers", fetch)
    await validate_pd_config(None, model)
    assert _pairing_unverified(model) is True


@pytest.mark.asyncio
async def test_conditional_sparse_update_fetches_the_merged_source(monkeypatch):
    original = model_input(decode_env={"VLLM_MLA_DISABLE": "1"})
    stored = Model.model_validate(original.model_dump())
    update = ModelUpdate(
        name=original.name,
        source=original.source,
        huggingface_repo_id=original.huggingface_repo_id,
        env={"VLLM_MLA_DISABLE": "0"},
    )
    fetch = AsyncMock(return_value=_MLA_CONFIG)
    monkeypatch.setattr(calculator, "get_pretrained_config_with_workers", fetch)
    with pytest.raises(BadRequestException) as error:
        await validate_pd_config(None, update, stored=stored)
    assert "VLLM_MLA_DISABLE" in error.value.message
    effective = fetch.call_args.args[0]
    assert effective.roles == stored.roles
    assert effective.backend_version == stored.backend_version
    assert stored.env is None


@pytest.mark.asyncio
async def test_conditional_config_overrides_are_applied_without_mutating_source(
    admission,
):
    model = model_input({"VLLM_MLA_DISABLE": "0"}, {"VLLM_MLA_DISABLE": "1"})
    model.backend_parameters = ["--hf-overrides", '{"kv_lora_rank": 512}']
    config = dict(_PLAIN_CONFIG)
    with pytest.raises(BadRequestException) as error:
        await admission(model, config)
    assert "VLLM_MLA_DISABLE" in error.value.message
    assert "kv_lora_rank" not in config


@pytest.mark.asyncio
async def test_layout_mismatch_is_rejected_for_the_issue_model(admission):
    model = model_input(
        {"VLLM_KV_CACHE_LAYOUT": "NHD"}, {"VLLM_KV_CACHE_LAYOUT": "HND"}
    )
    with pytest.raises(BadRequestException) as error:
        await admission(model, _PLAIN_CONFIG)
    assert error.value.status_code == 400
    assert "VLLM_KV_CACHE_LAYOUT" in error.value.message


@pytest.mark.asyncio
async def test_mla_transfer_does_not_require_matching_attention_layout_names(admission):
    model = model_input(
        {"VLLM_KV_CACHE_LAYOUT": "NHD", "VLLM_MLA_DISABLE": "0"},
        {"VLLM_KV_CACHE_LAYOUT": "HND", "VLLM_MLA_DISABLE": "0"},
    )
    await admission(model, _MLA_CONFIG)


def test_host_staging_layout_is_not_guessed_from_device_environment():
    model = model_input(
        {"VLLM_KV_CACHE_LAYOUT": "NHD"}, {"VLLM_KV_CACHE_LAYOUT": "HND"}
    )
    model.roles[0].backend_parameters = model.backend_parameters + [
        "--kv-transfer-config",
        json.dumps(
            {
                "kv_connector": "NixlConnector",
                "kv_role": "kv_producer",
                "kv_buffer_device": "cpu",
                "enable_permute_local_kv": True,
            }
        ),
    ]
    validate_role_pairing(model)
    assert _pairing_unverified(model) is True


@pytest.mark.asyncio
async def test_user_owned_connector_does_not_inherit_nixl_ssm_restrictions(admission):
    model = model_input(model_env={"VLLM_SSM_CONV_STATE_LAYOUT": "SD"})
    for role in model.roles[:2]:
        role.backend_parameters = model.backend_parameters + [
            "--kv-transfer-config",
            '{"kv_connector":"UserConnector","kv_role":"kv_both"}',
        ]
    await admission(model, _SSM_CONFIG)
    assert _pairing_unverified(model) is True


def test_empty_configuration_cannot_verify_a_conditional_factor():
    model = model_input({"VLLM_MLA_DISABLE": "0"}, {"VLLM_MLA_DISABLE": "1"})
    factors = environment_pairing_factors(model, pretrained_config={})
    assert any(factor.verdict == UNDECIDABLE for factor in factors)


@pytest.mark.asyncio
async def test_permutation_allows_mixed_attention_with_hma_explicitly_disabled(
    admission,
):
    model = model_input(
        {"VLLM_KV_CACHE_LAYOUT": "HND"},
        {"VLLM_KV_CACHE_LAYOUT": "NHD"},
        backend_parameters=["--disable-hybrid-kv-cache-manager"],
    )
    set_decode_connector(model, enable_permute_local_kv=True)
    await admission(model, {"layer_types": ["sliding_attention", "full_attention"]})


@pytest.mark.parametrize(
    "version,image", [(None, None), ("unknown", None), (None, "org/vllm:custom")]
)
@pytest.mark.asyncio
async def test_unknown_runtime_does_not_inherit_a_version_specific_ssm_rule(
    admission, version, image
):
    model = model_input(
        model_env={"VLLM_SSM_CONV_STATE_LAYOUT": "SD"},
        backend_version=version,
        image_name=image,
    )
    await admission(model, _SSM_CONFIG)
    assert _pairing_unverified(model) is True


@pytest.mark.asyncio
async def test_empty_role_version_does_not_inherit_the_pinned_model_version(admission):
    model = model_input(model_env={"VLLM_SSM_CONV_STATE_LAYOUT": "SD"})
    model.roles[0].backend_version = ""
    await admission(model, _SSM_CONFIG)
    assert _pairing_unverified(model) is True


@pytest.mark.asyncio
async def test_verified_environment_does_not_fetch_model_metadata(admission):
    fetch = await admission(
        model_input(model_env={"VLLM_KV_CACHE_LAYOUT": "HND"}), _PLAIN_CONFIG
    )
    fetch.assert_not_awaited()


@pytest.mark.parametrize("raw", ["not-json", "[]", "null", "42"])
def test_invalid_connector_json_is_rejected_at_admission(raw):
    model = model_input()
    model.roles[0].backend_parameters = ["--kv-transfer-config", raw]
    with pytest.raises(BadRequestException) as error:
        validate_role_pairing(model, pretrained_config=_PLAIN_CONFIG)
    assert "kv-transfer-config" in error.value.message
    assert "JSON object" in error.value.message


def test_connector_json_may_contain_templated_numeric_values():
    model = model_input()
    model.roles[0].backend_parameters = [
        "--kv-transfer-config",
        '{"kv_connector":"NixlConnector","kv_port":{{ports.kv_port}}}',
    ]
    validate_role_pairing(model, pretrained_config=_PLAIN_CONFIG)
    assert _pairing_unverified(model)
