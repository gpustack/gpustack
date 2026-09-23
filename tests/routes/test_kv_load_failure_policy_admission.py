"""A policy the chosen mode's connector cannot read.

`kv_load_failure_policy` is rendered by exactly one of the four built-in
recipes. On the other three the value would be accepted, stored
and read back while nothing reaches the engine — and this is the one field the
design notes single out as needing the user to weigh a real trade-off, `fail`
returning a 500 against `recompute` degrading silently. Someone who weighed it
and got neither is worse off than someone told the mode cannot honour it.
"""

import pytest

from gpustack.routes.models import _reject_a_policy_the_mode_cannot_apply
from gpustack.schemas.models import DisaggregationSpec, PDModeEnum
from gpustack.api.exceptions import BadRequestException


def _spec(mode, policy):
    return DisaggregationSpec(mode=mode, kv_load_failure_policy=policy)


def test_the_one_mode_that_renders_it_accepts_it():
    _reject_a_policy_the_mode_cannot_apply(_spec(PDModeEnum.VLLM_NIXL, "recompute"))


@pytest.mark.parametrize(
    "mode",
    [
        PDModeEnum.SGLANG_MOONCAKE,
        PDModeEnum.SGLANG_NIXL,
        PDModeEnum.VLLM_ASCEND_MOONCAKE,
    ],
)
def test_a_mode_whose_connector_has_no_such_setting_refuses(mode):
    """SGLang has no equivalent concept — its KV lifecycle is a bootstrap
    timeout that aborts, not a load that can fail and be retried — and
    Mooncake's connector does not read the key."""
    with pytest.raises(BadRequestException) as caught:
        _reject_a_policy_the_mode_cannot_apply(_spec(mode, "recompute"))
    assert "never reach the engine" in caught.value.message


@pytest.mark.parametrize("mode", list(PDModeEnum))
def test_the_default_is_never_refused(mode):
    """`fail` is what an engine that never sees the setting does anyway, so
    refusing it would break every group on those modes to no purpose. Only the
    deliberate choice is worth a refusal."""
    _reject_a_policy_the_mode_cannot_apply(_spec(mode, "fail"))


def test_custom_is_told_where_the_setting_actually_lives():
    """`custom` injects nothing by contract, so the remedy is not "pick
    another mode" — the connector configuration is already the user's to
    write, and the key belongs in it."""
    with pytest.raises(BadRequestException) as caught:
        _reject_a_policy_the_mode_cannot_apply(_spec(PDModeEnum.CUSTOM, "recompute"))
    assert "--kv-transfer-config" in caught.value.message


def test_the_rule_is_read_off_the_recipe_not_a_list_of_names():
    """A mode that starts rendering the key is accepted the moment it does,
    with nothing here to remember to update. Pinned by checking that the
    accepted set is exactly the set of recipes whose roles carry the key in
    their connector descriptor."""
    from gpustack.routes.models import _mode_renders_connector_key
    from gpustack.server.pd_mode_catalog import get_pd_mode

    for mode in PDModeEnum:
        recipe = get_pd_mode(mode.value)
        renders = recipe is not None and _mode_renders_connector_key(
            recipe, "kv_load_failure_policy"
        )
        try:
            _reject_a_policy_the_mode_cannot_apply(_spec(mode, "recompute"))
            accepted = True
        except BadRequestException:
            accepted = False
        assert accepted is renders, mode.value


def test_an_old_row_carrying_router_kind_still_loads():
    """`router_kind` was dropped, and `disaggregation` is a JSON column — rows
    written while the field existed still carry the key. Deserialising must
    ignore it rather than raise, which is what makes the removal need no
    migration."""
    spec = DisaggregationSpec.model_validate(
        {
            "mode": "vllm-nixl",
            "readiness": "all",
            "kv_load_failure_policy": "fail",
            "router_kind": "sgl-router",
        }
    )
    assert "router_kind" not in spec.model_dump()
    assert spec.readiness == "all"
