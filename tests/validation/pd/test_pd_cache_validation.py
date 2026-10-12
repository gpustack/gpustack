"""PD cache composition follows the mode's provider and role declarations."""

import pytest

from gpustack.api.exceptions import BadRequestException
from gpustack.schemas.models import DisaggregationSpec, PDModeEnum


@pytest.mark.parametrize(
    "mode,provider,role",
    [
        (PDModeEnum.VLLM_NIXL, "Mooncake", "prefill"),
        (PDModeEnum.VLLM_NIXL, "LMCache", "router"),
        (PDModeEnum.VLLM_MOONCAKE, "LMCache", "decode"),
    ],
)
def test_unsupported_pd_cache_combination_is_refused(mode, provider, role):
    from gpustack.validation.pd.roles import validate_pd_cache_provider

    with pytest.raises(BadRequestException) as error:
        validate_pd_cache_provider(DisaggregationSpec(mode=mode), provider, role=role)
    assert "does not support" in error.value.message
    assert provider in error.value.message
    assert role in error.value.message
